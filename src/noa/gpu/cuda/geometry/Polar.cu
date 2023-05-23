#include "noa/core/Assert.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/algorithms/geometry/PolarTransform.hpp"

#include "noa/gpu/cuda/geometry/Interpolator.hpp"
#include "noa/gpu/cuda/geometry/Polar.hpp"
#include "noa/gpu/cuda/memory/PtrArray.hpp"
#include "noa/gpu/cuda/memory/PtrTexture.hpp"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace {
    using namespace ::noa;

    template<bool LAYERED, typename Value>
    void launch_cartesian2polar_texture_(
            cudaTextureObject_t cartesian, InterpMode cartesian_interp,
            Value* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& cartesian_center,
            const Vec2<f32>& radius_range, bool radius_range_endpoint,
            const Vec2<f32>& angle_range, bool angle_range_endpoint,
            cuda::Stream& stream) {

        const auto i_polar_shape = polar_shape.as_safe<i32>();
        const auto iwise_shape = i_polar_shape.filter(0, 2, 3);
        const auto polar_accessor = AccessorRestrict<Value, 3, i32>(polar, polar_strides.filter(0, 2, 3).as_safe<i32>());

        switch (cartesian_interp) {
            case InterpMode::NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i32>(
                        interpolator_t(cartesian), polar_accessor, i_polar_shape, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i32>(
                        interpolator_t(cartesian), polar_accessor, i_polar_shape, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i32>(
                        interpolator_t(cartesian), polar_accessor, i_polar_shape, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i32>(
                        interpolator_t(cartesian), polar_accessor, i_polar_shape, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC_BSPLINE, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i32>(
                        interpolator_t(cartesian), polar_accessor, i_polar_shape, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i32>(
                        interpolator_t(cartesian), polar_accessor, i_polar_shape, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i32>(
                        interpolator_t(cartesian), polar_accessor, i_polar_shape, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC_BSPLINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i32>(
                        interpolator_t(cartesian), polar_accessor, i_polar_shape, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
        }
    }

    template<bool LAYERED, typename Value>
    void launch_polar2cartesian_texture_(
            cudaTextureObject_t polar, InterpMode polar_interp, const Shape4<i64>& polar_shape,
            Value* cartesian, const Strides4<i64>& cartesian_strides, const Shape4<i64>& cartesian_shape,
            const Vec2<f32>& cartesian_center,
            const Vec2<f32>& radius_range, bool radius_range_endpoint,
            const Vec2<f32>& angle_range, bool angle_range_endpoint,
            cuda::Stream& stream) {

        const auto i_polar_shape = polar_shape.as_safe<i32>();
        const auto iwise_shape = cartesian_shape.filter(0, 2, 3).as_safe<i32>();
        const auto cartesian_accessor = AccessorRestrict<Value, 3, i32>(
                cartesian, cartesian_strides.filter(0, 2, 3).as_safe<i32>());

        switch (polar_interp) {
            case InterpMode::NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i32>(
                        interpolator_t(polar), i_polar_shape, cartesian_accessor, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i32>(
                        interpolator_t(polar), i_polar_shape, cartesian_accessor, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i32>(
                        interpolator_t(polar), i_polar_shape, cartesian_accessor, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i32>(
                        interpolator_t(polar), i_polar_shape, cartesian_accessor, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC_BSPLINE, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i32>(
                        interpolator_t(polar), i_polar_shape, cartesian_accessor, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i32>(
                        interpolator_t(polar), i_polar_shape, cartesian_accessor, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i32>(
                        interpolator_t(polar), i_polar_shape, cartesian_accessor, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC_BSPLINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i32>(
                        interpolator_t(polar), i_polar_shape, cartesian_accessor, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
        }
    }
}

namespace noa::cuda::geometry {
    template<typename Value, typename>
    void cartesian2polar(const Value* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
                         Value* polar, Strides4<i64> polar_strides, Shape4<i64> polar_shape,
                         Vec2<f32> cartesian_center,
                         const Vec2<f32>& radius_range, bool radius_range_endpoint,
                         const Vec2<f32>& angle_range, bool angle_range_endpoint,
                         InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(cartesian != polar && noa::all(cartesian_shape > 0) && noa::all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(cartesian, stream.device());
        NOA_ASSERT_DEVICE_PTR(polar, stream.device());
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);

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

        const auto cartesian_shape_2d = cartesian_shape.filter(2, 3).as_safe<i32>();
        const auto i_polar_shape = polar_shape.as_safe<i32>();
        const auto i_polar_shape_2d = i_polar_shape.filter(0, 2, 3);
        const auto input_accessor = AccessorRestrict<const Value, 3, u32>(
                cartesian, cartesian_strides.filter(0, 2, 3).as_safe<u32>());
        const auto output_accessor = AccessorRestrict<Value, 3, u32>(
                polar, polar_strides.filter(0, 2, 3).as_safe<u32>());

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i32>(
                        interpolator, output_accessor, i_polar_shape, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::cartesian2polar", i_polar_shape_2d, kernel, stream);
            }
            case InterpMode::LINEAR_FAST:
            case InterpMode::LINEAR: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i32>(
                        interpolator, output_accessor, i_polar_shape, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::cartesian2polar", i_polar_shape_2d, kernel, stream);
            }
            case InterpMode::COSINE_FAST:
            case InterpMode::COSINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i32>(
                        interpolator, output_accessor, i_polar_shape, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::cartesian2polar", i_polar_shape_2d, kernel, stream);
            }
            case InterpMode::CUBIC: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::CUBIC>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i32>(
                        interpolator, output_accessor, i_polar_shape, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::cartesian2polar", i_polar_shape_2d, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE_FAST:
            case InterpMode::CUBIC_BSPLINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::CUBIC_BSPLINE>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar<i32>(
                        interpolator, output_accessor, i_polar_shape, cartesian_center,
                        radius_range, radius_range_endpoint,
                        angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("geometry::cartesian2polar", i_polar_shape_2d, kernel, stream);
            }
        }
    }

    template<typename Value, typename>
    void cartesian2polar(cudaArray* array,
                         cudaTextureObject_t cartesian,
                         InterpMode cartesian_interp, const Shape4<i64>& cartesian_shape,
                         Value* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
                         const Vec2<f32>& cartesian_center,
                         const Vec2<f32>& radius_range, bool radius_range_endpoint,
                         const Vec2<f32>& angle_range, bool angle_range_endpoint,
                         Stream& stream) {
        NOA_ASSERT(array && cartesian && noa::all(cartesian_shape > 0) && noa::all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(polar, stream.device());
        NOA_ASSERT(noa::cuda::memory::PtrTexture::array(cartesian) == array);
        NOA_ASSERT(polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[1] == 1);
        const bool is_layered = noa::cuda::memory::PtrArray<Value>::is_layered(array);

        if (is_layered) {
            NOA_ASSERT(cartesian_shape[0] == polar_shape[0]);
            launch_cartesian2polar_texture_<true>(
                    cartesian, cartesian_interp,
                    polar, polar_strides, polar_shape,
                    cartesian_center,
                    radius_range, radius_range_endpoint,
                    angle_range, angle_range_endpoint,
                    stream);
        } else {
            NOA_ASSERT(cartesian_shape[0] == 1);
            launch_cartesian2polar_texture_<false>(
                    cartesian, cartesian_interp,
                    polar, polar_strides, polar_shape,
                    cartesian_center,
                    radius_range, radius_range_endpoint,
                    angle_range, angle_range_endpoint,
                    stream);
        }
    }

    template<typename Value, typename>
    void polar2cartesian(const Value* polar, Strides4<i64> polar_strides, Shape4<i64> polar_shape,
                         Value* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
                         Vec2<f32> cartesian_center,
                         const Vec2<f32>& radius_range, bool radius_range_endpoint,
                         const Vec2<f32>& angle_range, bool angle_range_endpoint,
                         InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(cartesian && polar && noa::all(cartesian_shape > 0) && noa::all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(cartesian, stream.device());
        NOA_ASSERT(polar_shape[0] == 1 || polar_shape[0] == cartesian_shape[0]);
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);

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

        const auto i_polar_shape = polar_shape.as_safe<i32>();
        const auto i_polar_shape_2d = i_polar_shape.filter(2, 3);
        const auto cartesian_shape_2d = cartesian_shape.filter(0, 2, 3).as_safe<i32>();
        const auto input_accessor = AccessorRestrict<const Value, 3, u32>(
                polar, polar_strides.filter(0, 2, 3).as_safe<u32>());
        const auto output_accessor = AccessorRestrict<Value, 3, u32>(
                cartesian, cartesian_strides.filter(0, 2, 3).as_safe<u32>());

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(
                        input_accessor, i_polar_shape_2d);
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i32>(
                        interpolator, i_polar_shape, output_accessor, cartesian_center,
                        radius_range, radius_range_endpoint, angle_range, angle_range_endpoint);
                noa::cuda::utils::iwise_3d("geometry::polar2cartesian", cartesian_shape_2d, kernel, stream);
                break;
            }
            case InterpMode::LINEAR_FAST:
            case InterpMode::LINEAR: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                        input_accessor, i_polar_shape_2d);
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i32>(
                        interpolator, i_polar_shape, output_accessor, cartesian_center,
                        radius_range, radius_range_endpoint, angle_range, angle_range_endpoint);
                noa::cuda::utils::iwise_3d("geometry::polar2cartesian", cartesian_shape_2d, kernel, stream);
                break;
            }
            case InterpMode::COSINE_FAST:
            case InterpMode::COSINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(
                        input_accessor, i_polar_shape_2d);
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i32>(
                        interpolator, i_polar_shape, output_accessor, cartesian_center,
                        radius_range, radius_range_endpoint, angle_range, angle_range_endpoint);
                noa::cuda::utils::iwise_3d("geometry::polar2cartesian", cartesian_shape_2d, kernel, stream);
                break;
            }
            case InterpMode::CUBIC: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::CUBIC>(
                        input_accessor, i_polar_shape_2d);
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i32>(
                        interpolator, i_polar_shape, output_accessor, cartesian_center,
                        radius_range, radius_range_endpoint, angle_range, angle_range_endpoint);
                noa::cuda::utils::iwise_3d("geometry::polar2cartesian", cartesian_shape_2d, kernel, stream);
                break;
            }
            case InterpMode::CUBIC_BSPLINE_FAST:
            case InterpMode::CUBIC_BSPLINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::CUBIC_BSPLINE>(
                        input_accessor, i_polar_shape_2d);
                const auto kernel = noa::algorithm::geometry::polar2cartesian<i32>(
                        interpolator, i_polar_shape, output_accessor, cartesian_center,
                        radius_range, radius_range_endpoint, angle_range, angle_range_endpoint);
                noa::cuda::utils::iwise_3d("geometry::polar2cartesian", cartesian_shape_2d, kernel, stream);
                break;
            }
        }
    }

    template<typename Value, typename>
    void polar2cartesian(cudaArray* array,
                         cudaTextureObject_t polar,
                         InterpMode polar_interp, const Shape4<i64>& polar_shape,
                         Value* cartesian, const Strides4<i64>& cartesian_strides, const Shape4<i64>& cartesian_shape,
                         const Vec2<f32>& cartesian_center,
                         const Vec2<f32>& radius_range, bool radius_range_endpoint,
                         const Vec2<f32>& angle_range, bool angle_range_endpoint,
                         Stream& stream) {
        NOA_ASSERT(array && cartesian && noa::all(cartesian_shape > 0) && noa::all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(cartesian, stream.device());
        const bool is_layered = noa::cuda::memory::PtrArray<Value>::is_layered(array);
        NOA_ASSERT(noa::cuda::memory::PtrTexture::array(polar) == array);
        NOA_ASSERT(polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[1] == 1);

        if (is_layered) {
            NOA_ASSERT(polar_shape[0] == cartesian_shape[0]);
            launch_polar2cartesian_texture_<true>(
                    polar, polar_interp, polar_shape,
                    cartesian, cartesian_strides, cartesian_shape,
                    cartesian_center,
                    radius_range, radius_range_endpoint,
                    angle_range, angle_range_endpoint,
                    stream);
        } else {
            NOA_ASSERT(polar_shape[0] == 1);
            launch_polar2cartesian_texture_<false>(
                    polar, polar_interp, polar_shape,
                    cartesian, cartesian_strides, cartesian_shape,
                    cartesian_center,
                    radius_range, radius_range_endpoint,
                    angle_range, angle_range_endpoint,
                    stream);
        }
    }

    #define INSTANTIATE_POLAR(T)                        \
    template void cartesian2polar<T,void>(              \
        const T*, Strides4<i64>, Shape4<i64>,           \
        T*, Strides4<i64>, Shape4<i64>, Vec2<f32>,      \
        const Vec2<f32>&, bool,                         \
        const Vec2<f32>&, bool,                         \
        InterpMode, Stream&);                           \
    template void polar2cartesian<T,void>(              \
        const T*, Strides4<i64>, Shape4<i64>,           \
        T*, Strides4<i64>, Shape4<i64>, Vec2<f32>,      \
        const Vec2<f32>&, bool,                         \
        const Vec2<f32>&, bool,                         \
        InterpMode, Stream&)

    #define INSTANTIATE_POLAR_TEXTURE(T)                \
    template void cartesian2polar<T,void>(              \
        cudaArray*, cudaTextureObject_t,                \
        InterpMode, const Shape4<i64>&,                 \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const Vec2<f32>&,                               \
        const Vec2<f32>&, bool,                         \
        const Vec2<f32>&, bool,                         \
        Stream&);                                       \
    template void polar2cartesian<T,void>(              \
        cudaArray*, cudaTextureObject_t,                \
        InterpMode, const Shape4<i64>&,                 \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const Vec2<f32>&,                               \
        const Vec2<f32>&, bool,                         \
        const Vec2<f32>&, bool,                         \
        Stream&)

    INSTANTIATE_POLAR(f32);
    INSTANTIATE_POLAR(f64);
    INSTANTIATE_POLAR(c32);
    INSTANTIATE_POLAR(c64);
    INSTANTIATE_POLAR_TEXTURE(f32);
    INSTANTIATE_POLAR_TEXTURE(c32);
}
