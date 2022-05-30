#include "noa/common/Assert.h"
#include "noa/common/geometry/Polar.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/geometry/fft/Polar.h"
#include "noa/gpu/cuda/geometry/Interpolate.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    template<InterpMode MODE, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    cartesian2polar_(cudaTextureObject_t cartesian, T* polar, uint3_t polar_stride, uint2_t polar_shape,
                     float center, float start_angle, float2_t start_radius,
                     float step_angle, float2_t step_radius, bool log) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.x * blockDim.x + threadIdx.x};
        if (gid[1] >= polar_shape[0] || gid[2] >= polar_shape[1])
            return;

        const float2_t polar_coordinate{gid[1], gid[2]};
        const float angle_rad = polar_coordinate[0] * step_angle + start_angle;
        float magnitude_y, magnitude_x;
        if (log) {
            magnitude_y = math::exp(polar_coordinate[1] * step_radius[0]) - 1 + start_radius[0];
            magnitude_x = math::exp(polar_coordinate[1] * step_radius[1]) - 1 + start_radius[1];
        } else {
            magnitude_y = polar_coordinate[1] * step_radius[0] + start_radius[0];
            magnitude_x = polar_coordinate[1] * step_radius[1] + start_radius[1];
        }

        float2_t cartesian_coordinates{magnitude_y * math::sin(angle_rad),
                                       magnitude_x * math::cos(angle_rad)};
        float conj = 1;
        if (cartesian_coordinates[1] < 0) {
            cartesian_coordinates = -cartesian_coordinates;
            if constexpr (traits::is_complex_v<T>)
                conj = -1;
        } else {
            (void) conj;
        }
        cartesian_coordinates[0] += center; // center_x = 0;

        T value = cuda::geometry::tex2D<T, MODE>(cartesian, cartesian_coordinates + 0.5f);
        if constexpr (traits::is_complex_v<T>)
            value.imag *= conj;

        polar[indexing::at(gid, polar_stride)] = value;
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename T, typename>
    void cartesian2polar(const shared_t<T[]>& cartesian, size4_t cartesian_stride, size4_t cartesian_shape,
                         const shared_t<T[]>& polar, size4_t polar_stride, size4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, InterpMode interp, Stream& stream) {
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);
        NOA_ASSERT(cartesian_stride[3] == 1);

        if (cartesian_stride[0] == 0)
            cartesian_shape[0] = 1;

        // Broadcast input if it is not batched:
        const size4_t o_shape{cartesian_shape[0] > 1 ? 1 : polar_shape[0],
                              polar_shape[1], polar_shape[2], polar_shape[3]};

        // Copy to texture and launch (per input batch):
        const size3_t shape_3d{1, cartesian_shape[2], cartesian_shape[3] / 2 + 1};
        cuda::memory::PtrArray<T> array{shape_3d};
        cuda::memory::PtrTexture texture{array.get(), interp, BORDER_ZERO};
        for (size_t i = 0; i < cartesian_shape[0]; ++i) {
            cuda::memory::copy(cartesian.get() + i * cartesian_stride[0], cartesian_stride[2],
                               array.get(), shape_3d, stream);
            cuda::geometry::fft::cartesian2polar(
                    texture.get(), interp, size2_t{cartesian_shape.get() + 2},
                    polar.get() + i * polar_stride[0], polar_stride, o_shape,
                    frequency_range, angle_range, log, stream);
        }
        stream.attach(cartesian, polar, array.share(), texture.share());
    }

    template<typename T, typename>
    void cartesian2polar(cudaTextureObject_t cartesian, InterpMode cartesian_interp, size2_t cartesian_shape,
                         T* polar, size4_t polar_stride, size4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, Stream& stream) {
        NOA_ASSERT(polar_shape[1] == 1);
        const uint2_t o_shape{polar_shape.get() + 2};
        const uint3_t o_stride{polar_stride[0], polar_stride[2], polar_stride[3]};
        const dim3 blocks(math::divideUp(o_shape[1], THREADS.x),
                          math::divideUp(o_shape[0], THREADS.y),
                          polar_shape[0]);
        const LaunchConfig config{blocks, THREADS};

        const float2_t shape{o_shape - 1};
        const float step_angle = (angle_range[1] - angle_range[0]) / shape[0];

        const float2_t half_shape{cartesian_shape / 2};
        const float2_t radius_y_range{frequency_range * 2 * half_shape[0]};
        const float2_t radius_x_range{frequency_range * 2 * half_shape[1]};

        float step_radius_y, step_radius_x;
        if (log) {
            step_radius_y = math::log(radius_y_range[1] - radius_y_range[0]) / shape[1];
            step_radius_x = math::log(radius_x_range[1] - radius_x_range[0]) / shape[1];
        } else {
            step_radius_y = (radius_y_range[1] - radius_y_range[0]) / shape[1];
            step_radius_x = (radius_x_range[1] - radius_x_range[0]) / shape[1];
        }

        NOA_ASSERT(!memory::PtrTexture::hasNormalizedCoordinates(cartesian));
        switch (cartesian_interp) {
            case INTERP_NEAREST:
                return stream.enqueue("geometry::fft::cartesian2polar",
                                      cartesian2polar_<INTERP_NEAREST, T>,
                                      config, cartesian, polar, o_stride, o_shape,
                                      half_shape[0], angle_range[0], float2_t{radius_y_range[0], radius_x_range[0]},
                                      step_angle, float2_t{step_radius_y, step_radius_x}, log);
            case INTERP_LINEAR:
                return stream.enqueue("geometry::fft::cartesian2polar",
                                      cartesian2polar_<INTERP_COSINE, T>,
                                      config, cartesian, polar, o_stride, o_shape,
                                      half_shape[0], angle_range[0], float2_t{radius_y_range[0], radius_x_range[0]},
                                      step_angle, float2_t{step_radius_y, step_radius_x}, log);
            case INTERP_COSINE:
                return stream.enqueue("geometry::fft::cartesian2polar",
                                      cartesian2polar_<INTERP_COSINE, T>,
                                      config, cartesian, polar, o_stride, o_shape,
                                      half_shape[0], angle_range[0], float2_t{radius_y_range[0], radius_x_range[0]},
                                      step_angle, float2_t{step_radius_y, step_radius_x}, log);
            case INTERP_CUBIC:
                return stream.enqueue("geometry::fft::cartesian2polar",
                                      cartesian2polar_<INTERP_CUBIC, T>,
                                      config, cartesian, polar, o_stride, o_shape,
                                      half_shape[0], angle_range[0], float2_t{radius_y_range[0], radius_x_range[0]},
                                      step_angle, float2_t{step_radius_y, step_radius_x}, log);
            case INTERP_LINEAR_FAST:
                return stream.enqueue("geometry::fft::cartesian2polar",
                                      cartesian2polar_<INTERP_LINEAR_FAST, T>,
                                      config, cartesian, polar, o_stride, o_shape,
                                      half_shape[0], angle_range[0], float2_t{radius_y_range[0], radius_x_range[0]},
                                      step_angle, float2_t{step_radius_y, step_radius_x}, log);
            case INTERP_COSINE_FAST:
                return stream.enqueue("geometry::fft::cartesian2polar",
                                      cartesian2polar_<INTERP_COSINE_FAST, T>,
                                      config, cartesian, polar, o_stride, o_shape,
                                      half_shape[0], angle_range[0], float2_t{radius_y_range[0], radius_x_range[0]},
                                      step_angle, float2_t{step_radius_y, step_radius_x}, log);
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST:
                NOA_THROW("{} is not supported", cartesian_interp);
        }
    }

    #define INSTANTIATE_POLAR(T) \
    template void cartesian2polar<Remap::HC2FC, T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float2_t, bool, InterpMode, Stream&)

    INSTANTIATE_POLAR(float);
    INSTANTIATE_POLAR(cfloat_t);
}
