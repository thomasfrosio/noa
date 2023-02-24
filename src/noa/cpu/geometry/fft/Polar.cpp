#include "noa/core/geometry/Interpolator.hpp"
#include "noa/algorithms/geometry/PolarTransformRFFT.hpp"
#include "noa/cpu/geometry/fft/Polar.h"
#include "noa/cpu/utils/Iwise.hpp"

namespace noa::cpu::geometry::fft {
    template<noa::fft::Remap, typename Value, typename>
    void cartesian2polar(
            const Value* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
            Value* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, const Vec2<f32>& angle_range,
            bool log, InterpMode interp_mode, i64 threads) {
        NOA_ASSERT(cartesian && polar && cartesian != polar &&
                   noa::all(cartesian_shape > 0) && noa::all(polar_shape > 0));
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        // Broadcast the input to every output batch.
        if (cartesian_shape[0] == 1)
            cartesian_strides[0] = 0;
        else if (cartesian_strides[0] == 0)
            cartesian_shape[0] = 1;

        const auto iwise_shape = polar_shape.filter(0,2,3);
        const auto cartesian_shape_2d = cartesian_shape.filter(2,3);
        const auto cartesian_accessor = AccessorRestrict<const Value, 3, i64>(cartesian, cartesian_strides.filter(0, 2, 3));
        const auto polar_accessor = AccessorRestrict<Value, 3, i64>(polar, polar_strides.filter(0, 2, 3));

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i64>(
                        interpolator, cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::LINEAR: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i64>(
                        interpolator, cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::COSINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i64>(
                        interpolator, cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::LINEAR_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR_FAST>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i64>(
                        interpolator, cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::COSINE_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE_FAST>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i64>(
                        interpolator, cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::CUBIC:
            case InterpMode::CUBIC_BSPLINE:
            case InterpMode::CUBIC_BSPLINE_FAST:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    #define NOA_INSTANTIATE_POLAR(T)                                    \
    template void cartesian2polar<noa::fft::Remap::HC2FC, T, void>(     \
        const T*, Strides4<i64>, Shape4<i64>,                           \
        T*, const Strides4<i64>&, const Shape4<i64>&,                   \
        const Vec2<f32>&, const Vec2<f32>&,                             \
        bool, InterpMode, i64)

    NOA_INSTANTIATE_POLAR(f32);
    NOA_INSTANTIATE_POLAR(f64);
    NOA_INSTANTIATE_POLAR(c32);
    NOA_INSTANTIATE_POLAR(c64);
}
