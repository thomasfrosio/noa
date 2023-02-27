#include "noa/core/Assert.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/algorithms/geometry/PolarTransformRFFT.hpp"

#include "noa/gpu/cuda/geometry/fft/Polar.hpp"
#include "noa/gpu/cuda/geometry/Interpolator.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/memory/PtrArray.hpp"
#include "noa/gpu/cuda/memory/PtrTexture.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace {
    using namespace ::noa;

    template<bool LAYERED, typename Value>
    void launch_cartesian2polar_rfft_(
            cudaTextureObject_t cartesian, InterpMode cartesian_interp, const Shape4<i64>& cartesian_shape,
            Value* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, const Vec2<f32>& angle_range, bool log, cuda::Stream& stream) {
        NOA_ASSERT(polar_shape[1] == 1);
        NOA_ASSERT((LAYERED && polar_shape[0] == cartesian_shape[0]) ||
                   (!LAYERED && cartesian_shape[0] == 1));

        const auto i_cartesian_shape = cartesian_shape.as_safe<u32>();
        const auto i_polar_shape = polar_shape.as_safe<u32>();
        const auto iwise_shape = i_polar_shape.filter(0, 2, 3);
        const auto polar_accessor = AccessorRestrict<Value, 3, u32>(polar, polar_strides.filter(0, 2, 3).as_safe<u32>());

        switch (cartesian_interp) {
            case InterpMode::NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<u32>(
                        interpolator_t(cartesian), i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, angle_range, log);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<u32>(
                        interpolator_t(cartesian), i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, angle_range, log);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<u32>(
                        interpolator_t(cartesian), i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, angle_range, log);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<u32>(
                        interpolator_t(cartesian), i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, angle_range, log);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<u32>(
                        interpolator_t(cartesian), i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, angle_range, log);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC:
            case InterpMode::CUBIC_BSPLINE:
            case InterpMode::CUBIC_BSPLINE_FAST:
                NOA_THROW_FUNC("cartesian2polar_rfft", "{} is not supported", cartesian_interp);
        }
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename Value, typename>
    void cartesian2polar(
            const Value* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
            Value* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, const Vec2<f32>& angle_range,
            bool log, InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(cartesian && noa::all(cartesian_shape > 0) && noa::all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(polar, stream.device());
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        // Broadcast the input to every output batch.
        if (cartesian_shape[0] == 1)
            cartesian_strides[0] = 0;
        else if (cartesian_strides[0] == 0)
            cartesian_shape[0] = 1;

        const auto i_cartesian_shape = cartesian_shape.as_safe<i32>();
        const auto i_polar_shape = polar_shape.as_safe<i32>();
        const auto cartesian_shape_2d = i_cartesian_shape.filter(2, 3);
        const auto polar_shape_2d = i_polar_shape.filter(0, 2, 3);
        const auto cartesian_accessor = AccessorRestrict<const Value, 3, u32>(
                cartesian, cartesian_strides.filter(0, 2, 3).as_safe<u32>());
        const auto polar_accessor = AccessorRestrict<Value, 3, u32>(
                polar, polar_strides.filter(0, 2, 3).as_safe<u32>());

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i32>(
                        interpolator, i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, angle_range, log);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", polar_shape_2d, kernel, stream);
            }
            case InterpMode::LINEAR: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i32>(
                        interpolator, i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, angle_range, log);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", polar_shape_2d, kernel, stream);
            }
            case InterpMode::COSINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i32>(
                        interpolator, i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, angle_range, log);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", polar_shape_2d, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR_FAST>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i32>(
                        interpolator, i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, angle_range, log);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", polar_shape_2d, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE_FAST>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i32>(
                        interpolator, i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, angle_range, log);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", polar_shape_2d, kernel, stream);
            }
            case InterpMode::CUBIC:
            case InterpMode::CUBIC_BSPLINE:
            case InterpMode::CUBIC_BSPLINE_FAST:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    template<typename Value, typename>
    void cartesian2polar(
            cudaArray* array, cudaTextureObject_t cartesian,
            InterpMode cartesian_interp, const Shape4<i64>& cartesian_shape,
            Value* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, const Vec2<f32>& angle_range,
            bool log, Stream& stream) {
        NOA_ASSERT(array && cartesian && noa::all(cartesian_shape > 0) && noa::all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(polar, stream.device());

        const bool is_layered = noa::cuda::memory::PtrArray<Value>::is_layered(array);
        NOA_ASSERT(noa::cuda::memory::PtrTexture::array(cartesian) == array);

        if (is_layered) {
            launch_cartesian2polar_rfft_<true>(
                    cartesian, cartesian_interp, cartesian_shape,
                    polar, polar_strides, polar_shape,
                    frequency_range, angle_range, log, stream);
        } else {
            launch_cartesian2polar_rfft_<false>(
                    cartesian, cartesian_interp, cartesian_shape,
                    polar, polar_strides, polar_shape,
                    frequency_range, angle_range, log, stream);
        }
    }

    #define INSTANTIATE_POLAR(T) \
    template void cartesian2polar<Remap::HC2FC, T, void>(   \
        const T*, Strides4<i64>, Shape4<i64>,               \
        T*, const Strides4<i64>&, const Shape4<i64>&,       \
        const Vec2<f32>&, const Vec2<f32>&, bool, InterpMode, Stream&)

    #define INSTANTIATE_POLAR_TEXTURE(T)                \
    template void cartesian2polar<T, void>(             \
        cudaArray*, cudaTextureObject_t,                \
        InterpMode, const Shape4<i64>&,                 \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const Vec2<f32>&, const Vec2<f32>&, bool, Stream&)

    INSTANTIATE_POLAR(f32);
    INSTANTIATE_POLAR(c32);
    INSTANTIATE_POLAR(f64);
    INSTANTIATE_POLAR(c64);
    INSTANTIATE_POLAR_TEXTURE(f32);
    INSTANTIATE_POLAR_TEXTURE(c32);
}
