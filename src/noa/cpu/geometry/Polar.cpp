#include "noa/core/geometry/Polar.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/algorithms/geometry/PolarTransform.hpp"

#include "noa/cpu/geometry/Polar.hpp"
#include "noa/cpu/utils/Iwise.hpp"

namespace noa::cpu::geometry {
    template<typename Value, typename>
    void cartesian2polar(const Value* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
                         Value* polar, Strides4<i64> polar_strides, Shape4<i64> polar_shape,
                         Vec2<f32> cartesian_center, const Vec2<f32>& radius_range, const Vec2<f32>& angle_range,
                         bool log, InterpMode interp, i64 threads) {
        NOA_ASSERT(cartesian && polar && cartesian != polar &&
                   noa::all(cartesian_shape > 0) && noa::all(polar_shape > 0));
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        // Broadcast the input to every output batch.
        if (cartesian_shape[0] == 1)
            cartesian_strides[0] = 0;
        else if (cartesian_strides[0] == 0)
            cartesian_shape[0] = 1;

        // Reorder to rightmost if necessary.
        const auto order_2d = noa::indexing::order(polar_strides.filter(2, 3), polar_shape.filter(2, 3));
        if (noa::any(order_2d != Vec2<i64>{0, 1})) {
            cartesian_strides = cartesian_strides.filter(0, 1, 3, 2); // flip HW
            cartesian_shape = cartesian_shape.filter(0, 1, 3, 2);
            polar_strides = polar_strides.filter(0, 1, 3, 2);
            polar_shape = polar_shape.filter(0, 1, 3, 2);
            cartesian_center = cartesian_center.filter(1, 0);
        }

        const auto cartesian_shape_2d = cartesian_shape.filter(2, 3);
        const auto polar_shape_2d = polar_shape.filter(0, 2, 3);
        const auto input_accessor = AccessorRestrict<const Value, 3, i64>(cartesian, cartesian_strides.filter(0, 2, 3));
        const auto output_accessor = AccessorRestrict<Value, 3, i64>(polar, polar_strides.filter(0, 2, 3));

        switch (interp) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i64>(
                        interpolator, output_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cpu::utils::iwise_3d(polar_shape_2d, kernel, threads);
            }
            case InterpMode::LINEAR_FAST:
            case InterpMode::LINEAR: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i64>(
                        interpolator, output_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cpu::utils::iwise_3d(polar_shape_2d, kernel, threads);
            }
            case InterpMode::COSINE_FAST:
            case InterpMode::COSINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i64>(
                        interpolator, output_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cpu::utils::iwise_3d(polar_shape_2d, kernel, threads);
            }
            case InterpMode::CUBIC: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::CUBIC>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i64>(
                        interpolator, output_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cpu::utils::iwise_3d(polar_shape_2d, kernel, threads);
            }
            case InterpMode::CUBIC_BSPLINE_FAST:
            case InterpMode::CUBIC_BSPLINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::CUBIC_BSPLINE>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i64>(
                        interpolator, output_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cpu::utils::iwise_3d(polar_shape_2d, kernel, threads);
            }
        }
    }

    template<typename Value, typename>
    void polar2cartesian(const Value* polar, Strides4<i64> polar_strides, Shape4<i64> polar_shape,
                         Value* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
                         Vec2<f32> cartesian_center, const Vec2<f32>& radius_range, const Vec2<f32>& angle_range,
                         bool log, InterpMode interp, i64 threads) {
        NOA_ASSERT(cartesian && polar && cartesian != polar &&
                   noa::all(cartesian_shape > 0) && noa::all(polar_shape > 0));
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        // Broadcast the input to every output batch.
        if (polar_shape[0] == 1)
            polar_strides[0] = 0;
        else if (polar_strides[0] == 0)
            polar_shape[0] = 1;

        // Reorder to rightmost if necessary.
        const auto order_2d = noa::indexing::order(cartesian_strides.filter(2, 3), cartesian_shape.filter(2, 3));
        if (noa::any(order_2d != Vec2<i64>{0, 1})) {
            cartesian_strides = cartesian_strides.filter(0, 1, 3, 2); // flip HW
            cartesian_shape = cartesian_shape.filter(0, 1, 3, 2);
            polar_strides = polar_strides.filter(0, 1, 3, 2);
            polar_shape = polar_shape.filter(0, 1, 3, 2);
            cartesian_center = cartesian_center.filter(1, 0);
        }

        const auto polar_shape_2d = polar_shape.filter(2, 3);
        const auto cartesian_shape_2d = cartesian_shape.filter(0, 2, 3);
        const auto input_accessor = AccessorRestrict<const Value, 3, i64>(polar, polar_strides.filter(0, 2, 3));
        const auto output_accessor = AccessorRestrict<Value, 3, i64>(cartesian, cartesian_strides.filter(0, 2, 3));

        switch (interp) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(
                        input_accessor, polar_shape_2d);
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i64>(
                        interpolator, polar_shape, output_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cpu::utils::iwise_3d(cartesian_shape_2d, kernel, threads);
            }
            case InterpMode::LINEAR_FAST:
            case InterpMode::LINEAR: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                        input_accessor, polar_shape_2d);
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i64>(
                        interpolator, polar_shape, output_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cpu::utils::iwise_3d(cartesian_shape_2d, kernel, threads);
            }
            case InterpMode::COSINE_FAST:
            case InterpMode::COSINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(
                        input_accessor, polar_shape_2d);
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i64>(
                        interpolator, polar_shape, output_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cpu::utils::iwise_3d(cartesian_shape_2d, kernel, threads);
            }
            case InterpMode::CUBIC: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::CUBIC>(
                        input_accessor, polar_shape_2d);
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i64>(
                        interpolator, polar_shape, output_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cpu::utils::iwise_3d(cartesian_shape_2d, kernel, threads);
            }
            case InterpMode::CUBIC_BSPLINE_FAST:
            case InterpMode::CUBIC_BSPLINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::CUBIC_BSPLINE>(
                        input_accessor, polar_shape_2d);
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i64>(
                        interpolator, polar_shape, output_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cpu::utils::iwise_3d(cartesian_shape_2d, kernel, threads);
            }
        }
    }

    #define INSTANTIATE_POLAR(T)                    \
    template void cartesian2polar<T, void>(         \
        const T*, Strides4<i64>, Shape4<i64>,       \
        T*, Strides4<i64>, Shape4<i64>,             \
        Vec2<f32>, const Vec2<f32>&,                \
        const Vec2<f32>&, bool, InterpMode, i64);   \
    template void polar2cartesian<T, void>(         \
        const T*, Strides4<i64>, Shape4<i64>,       \
        T*, Strides4<i64>, Shape4<i64>,             \
        Vec2<f32>, const Vec2<f32>&,                \
        const Vec2<f32>&, bool, InterpMode, i64)

    INSTANTIATE_POLAR(f32);
    INSTANTIATE_POLAR(f64);
    INSTANTIATE_POLAR(c32);
    INSTANTIATE_POLAR(c64);
}
