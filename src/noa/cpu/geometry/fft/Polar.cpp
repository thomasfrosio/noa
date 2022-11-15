#include "noa/common/geometry/Interpolator.h"
#include "noa/common/geometry/details/PolarTransformFourier.h"

#include "noa/cpu/geometry/fft/Polar.h"
#include "noa/cpu/utils/Loops.h"

namespace noa::cpu::geometry::fft {
    template<Remap, typename Value, typename>
    void cartesian2polar(const shared_t<Value[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         const shared_t<Value[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(cartesian && polar && cartesian.get() != polar.get() &&
                   all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        // Broadcast the input to every output batch.
        if (cartesian_shape[0] == 1)
            cartesian_strides[0] = 0;
        else if (cartesian_strides[0] == 0)
            cartesian_shape[0] = 1;

        const dim_t threads = stream.threads();
        stream.enqueue([=]() {
            const auto iwise_shape = long3_t(polar_shape[0], polar_shape[2], polar_shape[3]);
            const auto cartesian_shape_2d = long2_t{cartesian_shape[2], cartesian_shape[3]};
            const auto cartesian_strides_2d = long3_t{cartesian_strides[0], cartesian_strides[2], cartesian_strides[3]};
            const auto polar_strides_2d = long3_t{polar_strides[0], polar_strides[2], polar_strides[3]};

            const auto cartesian_accessor = AccessorRestrict<const Value, 3, int64_t>(cartesian.get(), cartesian_strides_2d);
            const auto polar_accessor = AccessorRestrict<Value, 3, int64_t>(polar.get(), polar_strides_2d);

            switch (interp_mode) {
                case INTERP_NEAREST: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(
                            cartesian_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::fft::details::cartesian2polar<int64_t>(
                            interpolator, cartesian_shape, polar_accessor, polar_shape,
                            frequency_range, angle_range, log);
                    return cpu::utils::iwise3D(iwise_shape, kernel, threads);
                }
                case INTERP_LINEAR: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(
                            cartesian_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::fft::details::cartesian2polar<int64_t>(
                            interpolator, cartesian_shape, polar_accessor, polar_shape,
                            frequency_range, angle_range, log);
                    return cpu::utils::iwise3D(iwise_shape, kernel, threads);
                }
                case INTERP_COSINE: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(
                            cartesian_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::fft::details::cartesian2polar<int64_t>(
                            interpolator, cartesian_shape, polar_accessor, polar_shape,
                            frequency_range, angle_range, log);
                    return cpu::utils::iwise3D(iwise_shape, kernel, threads);
                }
                case INTERP_LINEAR_FAST: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR_FAST>(
                            cartesian_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::fft::details::cartesian2polar<int64_t>(
                            interpolator, cartesian_shape, polar_accessor, polar_shape,
                            frequency_range, angle_range, log);
                    return cpu::utils::iwise3D(iwise_shape, kernel, threads);
                }
                case INTERP_COSINE_FAST: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE_FAST>(
                            cartesian_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::fft::details::cartesian2polar<int64_t>(
                            interpolator, cartesian_shape, polar_accessor, polar_shape,
                            frequency_range, angle_range, log);
                    return cpu::utils::iwise3D(iwise_shape, kernel, threads);
                }
                case INTERP_CUBIC:
                case INTERP_CUBIC_BSPLINE:
                case INTERP_CUBIC_BSPLINE_FAST:
                    NOA_THROW("{} is not supported", interp_mode);
            }
        });
    }

    #define INSTANTIATE_POLAR(T)                            \
    template void cartesian2polar<Remap::HC2FC, T, void>(   \
        const shared_t<T[]>&, dim4_t, dim4_t,               \
        const shared_t<T[]>&, dim4_t, dim4_t,               \
        float2_t, float2_t, bool, InterpMode, Stream&)

    INSTANTIATE_POLAR(float);
    INSTANTIATE_POLAR(double);
    INSTANTIATE_POLAR(cfloat_t);
    INSTANTIATE_POLAR(cdouble_t);
}
