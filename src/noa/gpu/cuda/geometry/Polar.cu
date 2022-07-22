#include "noa/common/Assert.h"
#include "noa/common/geometry/Polar.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/geometry/Interpolate.h"
#include "noa/gpu/cuda/geometry/Polar.h"
#include "noa/gpu/cuda/geometry/Prefilter.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    template<InterpMode MODE, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    cartesian2polar_(cudaTextureObject_t cartesian, T* polar, uint3_t polar_stride, uint2_t polar_shape,
                     float2_t center, float start_angle, float start_magnitude,
                     float step_angle, float step_magnitude, bool log) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.x * blockDim.x + threadIdx.x};
        if (gid[1] >= polar_shape[0] || gid[2] >= polar_shape[1])
            return;

        const float2_t polar_coordinate{gid[1], gid[2]};
        const float angle_rad = polar_coordinate[0] * step_angle + start_angle;
        const float magnitude = log ?
                                math::exp(polar_coordinate[1] * step_magnitude) - 1 + start_magnitude :
                                (polar_coordinate[1] * step_magnitude) + start_magnitude;

        float2_t cartesian_coordinates{magnitude * math::sin(angle_rad),
                                       magnitude * math::cos(angle_rad)};
        cartesian_coordinates += center;
        cartesian_coordinates += 0.5f;

        polar[indexing::at(gid, polar_stride)] = cuda::geometry::tex2D<T, MODE>(cartesian, cartesian_coordinates);
    }

    template<InterpMode MODE, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    polar2cartesian_(cudaTextureObject_t polar, T* cartesian, uint3_t cartesian_stride, uint2_t cartesian_shape,
                     float2_t center, float start_angle, float start_magnitude,
                     float step_angle, float step_magnitude, bool log) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.x * blockDim.x + threadIdx.x};
        if (gid[1] >= cartesian_shape[0] || gid[2] >= cartesian_shape[1])
            return;

        float2_t cartesian_coordinate{gid[1], gid[2]};
        cartesian_coordinate -= center;

        const float angle_rad = geometry::cartesian2angle(cartesian_coordinate);
        const float magnitude = geometry::cartesian2magnitude(cartesian_coordinate);

        const float phi = (angle_rad - start_angle) / step_angle;
        const float rho = log ?
                          math::log(magnitude + 1 - start_magnitude) / step_magnitude :
                          (magnitude - start_magnitude) / step_magnitude;
        float2_t polar_coordinate{phi, rho};
        polar_coordinate += 0.5f;

        cartesian[indexing::at(gid, cartesian_stride)] = cuda::geometry::tex2D<T, MODE>(polar, polar_coordinate);
    }
}

namespace noa::cuda::geometry {
    template<bool PREFILTER, typename T, typename>
    void cartesian2polar(const shared_t<T[]>& cartesian, size4_t cartesian_stride, size4_t cartesian_shape,
                         const shared_t<T[]>& polar, size4_t polar_stride, size4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, Stream& stream) {
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);

        if (cartesian_stride[0] == 0)
            cartesian_shape[0] = 1;

        // Prepare the cartesian array:
        cuda::memory::PtrDevice<T> buffer;
        const T* buffer_ptr;
        size_t buffer_pitch;
        size_t buffer_offset;
        if (PREFILTER && (interp == INTERP_CUBIC_BSPLINE || interp == INTERP_CUBIC_BSPLINE_FAST)) {
            buffer = cuda::memory::PtrDevice<T>{cartesian_shape.elements(), stream};
            const size4_t contiguous_stride = cartesian_shape.strides();
            cuda::geometry::bspline::prefilter(cartesian, cartesian_stride,
                                               buffer.share(), contiguous_stride, cartesian_shape, stream);
            buffer_ptr = buffer.get();
            buffer_pitch = contiguous_stride[2];
            buffer_offset = contiguous_stride[0];
        } else {
            NOA_ASSERT(indexing::isContiguous(cartesian_stride, cartesian_shape)[3]);
            buffer_ptr = cartesian.get();
            buffer_pitch = cartesian_stride[2];
            buffer_offset = cartesian_stride[0];
        }

        // Broadcast cartesian if it is not batched:
        const size4_t o_shape{cartesian_shape[0] > 1 ? 1 : polar_shape[0],
                              polar_shape[1], polar_shape[2], polar_shape[3]};

        // Copy to texture and launch (per input batch):
        const size3_t shape_3d{1, cartesian_shape[2], cartesian_shape[3]};
        cuda::memory::PtrArray<T> array{shape_3d};
        cuda::memory::PtrTexture texture{array.get(), interp, BORDER_ZERO};
        for (size_t i = 0; i < cartesian_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_offset, buffer_pitch, array.get(), shape_3d, stream);
            cuda::geometry::cartesian2polar(
                    texture.get(), interp, polar.get() + i * polar_stride[0], polar_stride, o_shape,
                    cartesian_center, radius_range, angle_range, log, stream);
        }
        stream.attach(cartesian, polar, array.share(), texture.share());
        if (!buffer.empty())
            stream.attach(buffer.share());
    }

    template<bool PREFILTER, typename T, typename>
    void polar2cartesian(const shared_t<T[]>& polar, size4_t polar_stride, size4_t polar_shape,
                         const shared_t<T[]>& cartesian, size4_t cartesian_stride, size4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, Stream& stream) {
        NOA_ASSERT(polar_shape[0] == 1 || polar_shape[0] == cartesian_shape[0]);
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);

        if (polar_stride[0] == 0)
            polar_shape[0] = 1;

        // Prepare the polar array:
        cuda::memory::PtrDevice<T> buffer;
        const T* buffer_ptr;
        size_t buffer_pitch;
        size_t buffer_offset;
        if (PREFILTER && (interp == INTERP_CUBIC_BSPLINE || interp == INTERP_CUBIC_BSPLINE_FAST)) {
            buffer = cuda::memory::PtrDevice<T>{polar_shape.elements(), stream};
            const size4_t contiguous_stride = polar_shape.strides();
            cuda::geometry::bspline::prefilter(polar, polar_stride,
                                               buffer.share(), contiguous_stride, polar_shape, stream);
            buffer_ptr = buffer.get();
            buffer_pitch = contiguous_stride[2];
            buffer_offset = contiguous_stride[0];
        } else {
            NOA_ASSERT(indexing::isContiguous(polar_stride, polar_shape)[3]);
            buffer_ptr = polar.get();
            buffer_pitch = polar_stride[2];
            buffer_offset = polar_stride[0];
        }

        // Broadcast polar if it is not batched:
        const size4_t o_shape{polar_shape[0] > 1 ? 1 : cartesian_shape[0],
                              cartesian_shape[1], cartesian_shape[2], cartesian_shape[3]};

        // Copy to texture and launch (per input batch):
        const size3_t shape_3d{1, polar_shape[2], polar_shape[3]};
        cuda::memory::PtrArray<T> array{shape_3d};
        cuda::memory::PtrTexture texture{array.get(), interp, BORDER_ZERO};
        for (size_t i = 0; i < polar_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_offset, buffer_pitch, array.get(), shape_3d, stream);
            cuda::geometry::polar2cartesian(
                    texture.get(), interp, float2_t{polar_shape[2], polar_shape[3]},
                    cartesian.get() + i * cartesian_stride[0], cartesian_stride, o_shape,
                    cartesian_center, radius_range, angle_range, log, stream);
        }
        stream.attach(polar, cartesian, array.share(), texture.share());
        if (!buffer.empty())
            stream.attach(buffer.share());
    }

    template<typename T, typename>
    void cartesian2polar(cudaTextureObject_t cartesian, InterpMode cartesian_interp,
                         T* polar, size4_t polar_stride, size4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, Stream& stream) {
        NOA_ASSERT(polar_shape[1] == 1);
        const uint2_t o_shape{polar_shape.get() + 2};
        const uint3_t o_stride{polar_stride[0], polar_stride[2], polar_stride[3]};
        const dim3 blocks(math::divideUp(o_shape[1], THREADS.x),
                          math::divideUp(o_shape[0], THREADS.y),
                          polar_shape[0]);
        const LaunchConfig config{blocks, THREADS};

        NOA_ASSERT(radius_range[1] - radius_range[0] >= 0);
        const float2_t shape{o_shape - 1}; // endpoint = true, so N-1
        const float step_angle = (angle_range[1] - angle_range[0]) / shape[0];
        const float step_magnitude = log ?
                                     math::log(radius_range[1] - radius_range[0]) / shape[1] :
                                     (radius_range[1] - radius_range[0]) / shape[1];

        NOA_ASSERT(!memory::PtrTexture::hasNormalizedCoordinates(cartesian));
        switch (cartesian_interp) {
            case INTERP_NEAREST:
                return stream.enqueue("geometry::cartesian2polar",
                                      cartesian2polar_<INTERP_NEAREST, T>,
                                      config, cartesian, polar, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_LINEAR:
                return stream.enqueue("geometry::cartesian2polar",
                                      cartesian2polar_<INTERP_LINEAR, T>,
                                      config, cartesian, polar, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_COSINE:
                return stream.enqueue("geometry::cartesian2polar",
                                      cartesian2polar_<INTERP_COSINE, T>,
                                      config, cartesian, polar, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_CUBIC:
                return stream.enqueue("geometry::cartesian2polar",
                                      cartesian2polar_<INTERP_CUBIC, T>,
                                      config, cartesian, polar, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_CUBIC_BSPLINE:
                return stream.enqueue("geometry::cartesian2polar",
                                      cartesian2polar_<INTERP_CUBIC_BSPLINE, T>,
                                      config, cartesian, polar, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_LINEAR_FAST:
                return stream.enqueue("geometry::cartesian2polar",
                                      cartesian2polar_<INTERP_LINEAR_FAST, T>,
                                      config, cartesian, polar, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_COSINE_FAST:
                return stream.enqueue("geometry::cartesian2polar",
                                      cartesian2polar_<INTERP_COSINE_FAST, T>,
                                      config, cartesian, polar, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_CUBIC_BSPLINE_FAST:
                return stream.enqueue("geometry::cartesian2polar",
                                      cartesian2polar_<INTERP_CUBIC_BSPLINE_FAST, T>,
                                      config, cartesian, polar, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
        }
    }

    template<typename T, typename>
    void polar2cartesian(cudaTextureObject_t polar, InterpMode polar_interp, float2_t polar_shape,
                         T* cartesian, size4_t cartesian_stride, size4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, Stream& stream) {
        NOA_ASSERT(cartesian_shape[1] == 1);
        const uint2_t o_shape{cartesian_shape.get() + 2};
        const uint3_t o_stride{cartesian_stride[0], cartesian_stride[2], cartesian_stride[3]};
        const dim3 blocks(math::divideUp(o_shape[1], THREADS.x),
                          math::divideUp(o_shape[0], THREADS.y),
                          cartesian_shape[0]);
        const LaunchConfig config{blocks, THREADS};

        NOA_ASSERT(radius_range[1] - radius_range[0] >= 0);
        const float2_t shape{polar_shape - 1}; // endpoint = true, so N-1
        const float step_angle = (angle_range[1] - angle_range[0]) / shape[0];
        const float step_magnitude = log ?
                                     math::log(radius_range[1] - radius_range[0]) / shape[1] :
                                     (radius_range[1] - radius_range[0]) / shape[1];

        NOA_ASSERT(!memory::PtrTexture::hasNormalizedCoordinates(polar));
        switch (polar_interp) {
            case INTERP_NEAREST:
                return stream.enqueue("geometry::polar2cartesian",
                                      polar2cartesian_<INTERP_NEAREST, T>,
                                      config, polar, cartesian, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_LINEAR:
                return stream.enqueue("geometry::polar2cartesian",
                                      polar2cartesian_<INTERP_LINEAR, T>,
                                      config, polar, cartesian, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_COSINE:
                return stream.enqueue("geometry::polar2cartesian",
                                      polar2cartesian_<INTERP_COSINE, T>,
                                      config, polar, cartesian, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_CUBIC:
                return stream.enqueue("geometry::polar2cartesian",
                                      polar2cartesian_<INTERP_CUBIC, T>,
                                      config, polar, cartesian, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_CUBIC_BSPLINE:
                return stream.enqueue("geometry::polar2cartesian",
                                      polar2cartesian_<INTERP_CUBIC_BSPLINE, T>,
                                      config, polar, cartesian, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_LINEAR_FAST:
                return stream.enqueue("geometry::polar2cartesian",
                                      polar2cartesian_<INTERP_LINEAR_FAST, T>,
                                      config, polar, cartesian, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_COSINE_FAST:
                return stream.enqueue("geometry::polar2cartesian",
                                      polar2cartesian_<INTERP_COSINE_FAST, T>,
                                      config, polar, cartesian, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            case INTERP_CUBIC_BSPLINE_FAST:
                return stream.enqueue("geometry::polar2cartesian",
                                      polar2cartesian_<INTERP_CUBIC_BSPLINE_FAST, T>,
                                      config, polar, cartesian, o_stride, o_shape, cartesian_center,
                                      angle_range[0], radius_range[0], step_angle, step_magnitude, log);
        }
    }

    #define INSTANTIATE_POLAR(T) \
    template void cartesian2polar<true,T,void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, Stream&); \
    template void cartesian2polar<false,T,void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, Stream&);\
    template void polar2cartesian<true,T,void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, Stream&); \
    template void polar2cartesian<false,T,void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, float2_t, bool, InterpMode, Stream&)

    INSTANTIATE_POLAR(float);
    INSTANTIATE_POLAR(cfloat_t);
}
