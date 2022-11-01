#include "noa/common/Assert.h"
#include "noa/common/geometry/Polar.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/geometry/Polar.h"
#include "noa/gpu/cuda/geometry/Prefilter.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/util/Pointers.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    template<typename data_t, typename interpolator_t>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    cartesian2polar_(interpolator_t cartesian,
                     Accessor<data_t, 3, uint32_t> polar, uint2_t polar_shape,
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

        polar(gid) = cartesian(cartesian_coordinates);
    }

    template<typename data_t, typename interpolator_t>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    polar2cartesian_(interpolator_t polar,
                     Accessor<data_t, 3, uint32_t> cartesian, uint2_t cartesian_shape,
                     float2_t center, float start_angle, float start_magnitude,
                     float step_angle, float step_magnitude, bool log) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.x * blockDim.x + threadIdx.x};
        if (gid[1] >= cartesian_shape[0] || gid[2] >= cartesian_shape[1])
            return;

        float2_t cartesian_coordinate{gid[1], gid[2]};
        cartesian_coordinate -= center;

        const float phi = geometry::cartesian2phi(cartesian_coordinate);
        const float rho = geometry::cartesian2rho(cartesian_coordinate);

        const float py = (phi - start_angle) / step_angle;
        const float px = log ?
                          math::log(rho + 1 - start_magnitude) / step_magnitude :
                          (rho - start_magnitude) / step_magnitude;
        float2_t polar_coordinate{py, px};

        cartesian(gid) = polar(polar_coordinate);
    }

    template<typename data_t>
    void launchCartesianPolarTexture_(cudaTextureObject_t cartesian, InterpMode cartesian_interp,
                                      data_t* polar, dim4_t polar_strides, dim4_t polar_shape,
                                      float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                                      bool log, cuda::Stream& stream) {
        NOA_ASSERT(polar_shape[1] == 1);
        const auto o_shape = safe_cast<uint2_t>(dim2_t(polar_shape.get(2)));
        const auto o_strides = safe_cast<uint3_t>(dim3_t{polar_strides[0], polar_strides[2], polar_strides[3]});
        const dim3 blocks(math::divideUp(o_shape[1], THREADS.x),
                          math::divideUp(o_shape[0], THREADS.y),
                          polar_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};

        NOA_ASSERT(radius_range[1] - radius_range[0] >= 0);
        const float2_t shape(o_shape - 1); // endpoint = true, so N-1
        const float step_angle = (angle_range[1] - angle_range[0]) / shape[0];
        const float step_magnitude = log ?
                                     math::log(radius_range[1] - radius_range[0]) / shape[1] :
                                     (radius_range[1] - radius_range[0]) / shape[1];

        const Accessor<data_t, 3, uint32_t> polar_accessor(polar, o_strides);
        switch (cartesian_interp) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, data_t>;
                return stream.enqueue(
                        "geometry::cartesian2polar",
                        cartesian2polar_<data_t, interpolator_t>, config,
                        interpolator_t(cartesian), polar_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, data_t>;
                return stream.enqueue(
                        "geometry::cartesian2polar",
                        cartesian2polar_<data_t, interpolator_t>, config,
                        interpolator_t(cartesian), polar_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, data_t>;
                return stream.enqueue(
                        "geometry::cartesian2polar",
                        cartesian2polar_<data_t, interpolator_t>, config,
                        interpolator_t(cartesian), polar_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_CUBIC: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC, data_t>;
                return stream.enqueue(
                        "geometry::cartesian2polar",
                        cartesian2polar_<data_t, interpolator_t>, config,
                        interpolator_t(cartesian), polar_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE, data_t>;
                return stream.enqueue(
                        "geometry::cartesian2polar",
                        cartesian2polar_<data_t, interpolator_t>, config,
                        interpolator_t(cartesian), polar_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, data_t>;
                return stream.enqueue(
                        "geometry::cartesian2polar",
                        cartesian2polar_<data_t, interpolator_t>, config,
                        interpolator_t(cartesian), polar_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, data_t>;
                return stream.enqueue(
                        "geometry::cartesian2polar",
                        cartesian2polar_<data_t, interpolator_t>, config,
                        interpolator_t(cartesian), polar_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE_FAST, data_t>;
                return stream.enqueue(
                        "geometry::cartesian2polar",
                        cartesian2polar_<data_t, interpolator_t>, config,
                        interpolator_t(cartesian), polar_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
        }
    }

    template<typename data_t>
    void launchPolarCartesianTexture_(cudaTextureObject_t polar, InterpMode polar_interp, float2_t polar_shape,
                                      data_t* cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                                      float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                                      bool log, cuda::Stream& stream) {
        NOA_ASSERT(cartesian_shape[1] == 1);
        const auto o_shape = safe_cast<uint2_t>(dim2_t(cartesian_shape.get(2)));
        const auto o_strides = safe_cast<uint3_t>(dim3_t{cartesian_strides[0], cartesian_strides[2], cartesian_strides[3]});
        const dim3 blocks(math::divideUp(o_shape[1], THREADS.x),
                          math::divideUp(o_shape[0], THREADS.y),
                          cartesian_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};

        NOA_ASSERT(radius_range[1] - radius_range[0] >= 0);
        const float2_t shape(polar_shape - 1); // endpoint = true, so N-1
        const float step_angle = (angle_range[1] - angle_range[0]) / shape[0];
        const float step_magnitude = log ?
                                     math::log(radius_range[1] - radius_range[0]) / shape[1] :
                                     (radius_range[1] - radius_range[0]) / shape[1];

        const Accessor<data_t, 3, uint32_t> cartesian_accessor(cartesian, o_strides);
        switch (polar_interp) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, data_t>;
                return stream.enqueue(
                        "geometry::polar2cartesian",
                        polar2cartesian_<data_t, interpolator_t>, config,
                        interpolator_t(polar), cartesian_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, data_t>;
                return stream.enqueue(
                        "geometry::polar2cartesian",
                        polar2cartesian_<data_t, interpolator_t>, config,
                        interpolator_t(polar), cartesian_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, data_t>;
                return stream.enqueue(
                        "geometry::polar2cartesian",
                        polar2cartesian_<data_t, interpolator_t>, config,
                        interpolator_t(polar), cartesian_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_CUBIC: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC, data_t>;
                return stream.enqueue(
                        "geometry::polar2cartesian",
                        polar2cartesian_<data_t, interpolator_t>, config,
                        interpolator_t(polar), cartesian_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE, data_t>;
                return stream.enqueue(
                        "geometry::polar2cartesian",
                        polar2cartesian_<data_t, interpolator_t>, config,
                        interpolator_t(polar), cartesian_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, data_t>;
                return stream.enqueue(
                        "geometry::polar2cartesian",
                        polar2cartesian_<data_t, interpolator_t>, config,
                        interpolator_t(polar), cartesian_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, data_t>;
                return stream.enqueue(
                        "geometry::polar2cartesian",
                        polar2cartesian_<data_t, interpolator_t>, config,
                        interpolator_t(polar), cartesian_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE_FAST, data_t>;
                return stream.enqueue(
                        "geometry::polar2cartesian",
                        polar2cartesian_<data_t, interpolator_t>, config,
                        interpolator_t(polar), cartesian_accessor, o_shape, cartesian_center,
                        angle_range[0], radius_range[0], step_angle, step_magnitude, log);
            }
        }
    }
}

namespace noa::cuda::geometry {
    template<typename T, typename>
    void cartesian2polar(const shared_t<T[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         const shared_t<T[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, bool prefilter, Stream& stream) {
        NOA_ASSERT(cartesian && polar && all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(polar.get(), stream.device());
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);

        if (cartesian_strides[0] == 0)
            cartesian_shape[0] = 1;
        else if (cartesian_shape[0] == 1)
            cartesian_strides[0] = 0;

        // Prepare the cartesian array:
        shared_t<T[]> buffer;
        const T* buffer_ptr;
        dim_t buffer_pitch;
        dim_t buffer_offset;
        if (prefilter && (interp == INTERP_CUBIC_BSPLINE || interp == INTERP_CUBIC_BSPLINE_FAST)) {
            buffer = memory::PtrDevice<T>::alloc(cartesian_shape.elements(), stream);
            const dim4_t contiguous_strides = cartesian_shape.strides();
            bspline::prefilter(cartesian, cartesian_strides, buffer, contiguous_strides, cartesian_shape, stream);
            buffer_ptr = buffer.get();
            buffer_pitch = contiguous_strides[2];
            buffer_offset = contiguous_strides[0];
        } else {
            NOA_ASSERT(indexing::isContiguous(cartesian_strides, cartesian_shape)[3]);
            buffer_ptr = cartesian.get();
            buffer_pitch = cartesian_strides[2];
            buffer_offset = cartesian_strides[0];
        }

        // Broadcast cartesian if it is not batched:
        const dim4_t o_shape{cartesian_shape[0] > 1 ? 1 : polar_shape[0],
                              polar_shape[1], polar_shape[2], polar_shape[3]};

        // Copy to texture and launch (per input batch):
        const dim3_t shape_3d{1, cartesian_shape[2], cartesian_shape[3]};
        memory::PtrArray<T> array(shape_3d);
        memory::PtrTexture texture(array.get(), interp, BORDER_ZERO);
        for (dim_t i = 0; i < cartesian_shape[0]; ++i) {
            memory::copy(buffer_ptr + i * buffer_offset, buffer_pitch, array.get(), shape_3d, stream);
            launchCartesianPolarTexture_(
                    texture.get(), interp, polar.get() + i * polar_strides[0], polar_strides, o_shape,
                    cartesian_center, radius_range, angle_range, log, stream);
        }
        stream.attach(cartesian, polar, array.share(), texture.share());
    }

    template<typename T, typename>
    void polar2cartesian(const shared_t<T[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         const shared_t<T[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, bool prefilter, Stream& stream) {
        NOA_ASSERT(cartesian && polar && all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(cartesian.get(), stream.device());
        NOA_ASSERT(polar_shape[0] == 1 || polar_shape[0] == cartesian_shape[0]);
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);

        if (polar_strides[0] == 0)
            polar_shape[0] = 1;
        else if (polar_shape[0] == 1)
            polar_strides[0] = 0;

        // Prepare the polar array:
        shared_t<T[]> buffer;
        const T* buffer_ptr;
        dim_t buffer_pitch;
        dim_t buffer_offset;
        if (prefilter && (interp == INTERP_CUBIC_BSPLINE || interp == INTERP_CUBIC_BSPLINE_FAST)) {
            buffer = memory::PtrDevice<T>::alloc(polar_shape.elements(), stream);
            const dim4_t contiguous_strides = polar_shape.strides();
            bspline::prefilter(polar, polar_strides, buffer, contiguous_strides, polar_shape, stream);
            buffer_ptr = buffer.get();
            buffer_pitch = contiguous_strides[2];
            buffer_offset = contiguous_strides[0];
        } else {
            NOA_ASSERT(indexing::isContiguous(polar_strides, polar_shape)[3]);
            buffer_ptr = polar.get();
            buffer_pitch = polar_strides[2];
            buffer_offset = polar_strides[0];
        }

        // Broadcast polar if it is not batched:
        const dim4_t o_shape{polar_shape[0] > 1 ? 1 : cartesian_shape[0],
                             cartesian_shape[1], cartesian_shape[2], cartesian_shape[3]};

        // Copy to texture and launch (per input batch):
        const dim3_t shape_3d{1, polar_shape[2], polar_shape[3]};
        memory::PtrArray<T> array(shape_3d);
        memory::PtrTexture texture(array.get(), interp, BORDER_ZERO);
        for (dim_t i = 0; i < polar_shape[0]; ++i) {
            memory::copy(buffer_ptr + i * buffer_offset, buffer_pitch, array.get(), shape_3d, stream);
            launchPolarCartesianTexture_(
                    texture.get(), interp, float2_t{polar_shape[2], polar_shape[3]},
                    cartesian.get() + i * cartesian_strides[0], cartesian_strides, o_shape,
                    cartesian_center, radius_range, angle_range, log, stream);
        }
        stream.attach(polar, cartesian, array.share(), texture.share());
    }

    template<typename T, typename>
    void cartesian2polar(const shared_t<cudaArray>& array,
                         const shared_t<cudaTextureObject_t>& cartesian, InterpMode cartesian_interp,
                         const shared_t<T[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, Stream& stream) {
        NOA_ASSERT(all(polar_shape > 0) && array && cartesian);
        NOA_ASSERT_DEVICE_PTR(polar.get(), stream.device());
        launchCartesianPolarTexture_(*cartesian, cartesian_interp, polar.get(), polar_strides, polar_shape,
                                     cartesian_center, radius_range, angle_range, log, stream);
        stream.attach(array, cartesian, polar);
    }

    template<typename T, typename>
    void polar2cartesian(const shared_t<cudaArray>& array,
                         const shared_t<cudaTextureObject_t>& polar, InterpMode polar_interp, float2_t polar_shape,
                         const shared_t<T[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, Stream& stream) {
        NOA_ASSERT(all(polar_shape > 0) && array && polar);
        NOA_ASSERT_DEVICE_PTR(cartesian.get(), stream.device());
        launchPolarCartesianTexture_(*polar, polar_interp, polar_shape,
                                     cartesian.get(), cartesian_strides, cartesian_shape,
                                     cartesian_center, radius_range, angle_range, log, stream);
        stream.attach(array, cartesian, polar);
    }

    #define INSTANTIATE_POLAR(T) \
    template void cartesian2polar<T,void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, float2_t, bool, InterpMode, bool, Stream&);                       \
    template void polar2cartesian<T,void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, float2_t, bool, InterpMode, bool, Stream&);                       \
    template void cartesian2polar<T,void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, float2_t, bool, Stream&); \
    template void polar2cartesian<T,void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, float2_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, float2_t, bool, Stream&)

    INSTANTIATE_POLAR(float);
    INSTANTIATE_POLAR(cfloat_t);
}
