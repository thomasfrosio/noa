#include "noa/core/Assert.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/algorithms/geometry/PolarTransformRFFT.hpp"
#include "noa/algorithms/geometry/RotationalAverage.hpp"

#include "noa/gpu/cuda/geometry/fft/Polar.hpp"
#include "noa/gpu/cuda/geometry/Interpolator.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/memory/PtrArray.hpp"
#include "noa/gpu/cuda/memory/PtrDevice.hpp"
#include "noa/gpu/cuda/memory/PtrTexture.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/utils/EwiseBinary.cuh"

namespace {
    using namespace ::noa;

    template<bool LAYERED, typename Input, typename Output>
    void launch_cartesian2polar_rfft_(
            cudaTextureObject_t cartesian, InterpMode cartesian_interp, const Shape4<i64>& cartesian_shape,
            Output* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, bool frequency_range_endpoint,
            const Vec2<f32>& angle_range, bool angle_range_endpoint,
            cuda::Stream& stream) {
        NOA_ASSERT(polar_shape[1] == 1);
        NOA_ASSERT((LAYERED && polar_shape[0] == cartesian_shape[0]) ||
                   (!LAYERED && cartesian_shape[0] == 1));

        const auto i_cartesian_shape = cartesian_shape.as_safe<u32>();
        const auto i_polar_shape = polar_shape.as_safe<u32>();
        const auto iwise_shape = i_polar_shape.filter(0, 2, 3);
        const auto polar_accessor = AccessorRestrict<Output, 3, u32>(polar, polar_strides.filter(0, 2, 3).as_safe<u32>());

        switch (cartesian_interp) {
            case InterpMode::NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::NEAREST, Input, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft(
                        interpolator_t(cartesian), i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, frequency_range_endpoint, angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR, Input, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft(
                        interpolator_t(cartesian), i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, frequency_range_endpoint, angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE, Input, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft(
                        interpolator_t(cartesian), i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, frequency_range_endpoint, angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Input, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft(
                        interpolator_t(cartesian), i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, frequency_range_endpoint, angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE_FAST, Input, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft(
                        interpolator_t(cartesian), i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, frequency_range_endpoint, angle_range, angle_range_endpoint);
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
    template<Remap REMAP, typename Input, typename Output, typename>
    void cartesian2polar(
            const Input* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
            Output* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, bool frequency_range_endpoint,
            const Vec2<f32>& angle_range, bool angle_range_endpoint,
            InterpMode interp_mode, Stream& stream) {
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
        const auto cartesian_accessor = AccessorRestrict<const Input, 3, u32>(
                cartesian, cartesian_strides.filter(0, 2, 3).as_safe<u32>());
        const auto polar_accessor = AccessorRestrict<Output, 3, u32>(
                polar, polar_strides.filter(0, 2, 3).as_safe<u32>());

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft(
                        interpolator, i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, frequency_range_endpoint, angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", polar_shape_2d, kernel, stream);
            }
            case InterpMode::LINEAR: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft(
                        interpolator, i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, frequency_range_endpoint, angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", polar_shape_2d, kernel, stream);
            }
            case InterpMode::COSINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft(
                        interpolator, i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, frequency_range_endpoint, angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", polar_shape_2d, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR_FAST>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft(
                        interpolator, i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, frequency_range_endpoint, angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", polar_shape_2d, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE_FAST>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft(
                        interpolator, i_cartesian_shape, polar_accessor, i_polar_shape,
                        frequency_range, frequency_range_endpoint, angle_range, angle_range_endpoint);
                return noa::cuda::utils::iwise_3d("cartesian2polar_rfft", polar_shape_2d, kernel, stream);
            }
            case InterpMode::CUBIC:
            case InterpMode::CUBIC_BSPLINE:
            case InterpMode::CUBIC_BSPLINE_FAST:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    template<typename Input, typename Output, typename>
    void cartesian2polar(
            cudaArray* array, cudaTextureObject_t cartesian,
            InterpMode cartesian_interp, const Shape4<i64>& cartesian_shape,
            Output* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, bool frequency_range_endpoint,
            const Vec2<f32>& angle_range, bool angle_range_endpoint,
            Stream& stream) {
        NOA_ASSERT(array && cartesian && noa::all(cartesian_shape > 0) && noa::all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(polar, stream.device());

        const bool is_layered = noa::cuda::memory::PtrArray<Output>::is_layered(array);
        NOA_ASSERT(noa::cuda::memory::PtrTexture::array(cartesian) == array);

        if (is_layered) {
            launch_cartesian2polar_rfft_<true, Input>(
                    cartesian, cartesian_interp, cartesian_shape,
                    polar, polar_strides, polar_shape,
                    frequency_range, frequency_range_endpoint,
                    angle_range, angle_range_endpoint, stream);
        } else {
            launch_cartesian2polar_rfft_<false, Input>(
                    cartesian, cartesian_interp, cartesian_shape,
                    polar, polar_strides, polar_shape,
                    frequency_range, frequency_range_endpoint,
                    angle_range, angle_range_endpoint, stream);
        }
    }

    template<noa::fft::Remap REMAP, typename Input, typename Output, typename Weight, typename>
    void rotational_average(
            const Input* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
            Output* output, Weight* weight, bool average, Stream& stream) {

        const auto shell_count = noa::math::min(input_shape) / 2 + 1;

        // When computing the average, the weights must be valid.
        using unique_t = typename noa::cuda::memory::PtrDevice<Weight>::unique_type;
        unique_t weight_buffer;
        Weight* weight_ptr = weight;
        if (weight_ptr == nullptr && average) {
            weight_buffer = noa::cuda::memory::PtrDevice<Weight>::alloc(shell_count, stream);
            weight_ptr = weight_buffer.get();
        }

        constexpr bool IS_HALF = static_cast<u8>(REMAP) & noa::fft::Layout::SRC_HALF;
        if (input_shape.ndim() == 2) {
            // Reorder HW dimensions to rightmost.
            const auto order = noa::indexing::order(input_strides.filter(2, 3), input_shape.filter(2, 3));
            if (noa::any(order != Vec2<i64>{0, 1})) {
                std::swap(input_strides[2], input_strides[3]);
                std::swap(input_shape[2], input_shape[3]);
            }

            const auto kernel = noa::algorithm::geometry::rotational_average_2d<REMAP>(
                    input, input_strides, input_shape, output, weight_ptr, shell_count);

            auto iwise_shape = input_shape.filter(0, 2, 3);
            if constexpr (IS_HALF)
                iwise_shape = iwise_shape.rfft();
            noa::cuda::utils::iwise_3d("rotational_average", iwise_shape, kernel, stream);

        } else {
            // Reorder BHW dimensions to rightmost.
            const auto order = noa::indexing::order(input_strides.pop_front(), input_shape.pop_front());
            if (noa::any(order != Vec3<i64>{0, 1, 2})) {
                const auto order_3d = (order + 1).push_front(0);
                input_strides = noa::indexing::reorder(input_strides, order_3d);
                input_shape = noa::indexing::reorder(input_shape, order_3d);
            }

            const auto kernel = noa::algorithm::geometry::rotational_average_3d<REMAP>(
                    input, input_strides, input_shape, output, weight_ptr, shell_count);

            noa::cuda::utils::iwise_4d(
                    "rotational_average", IS_HALF ? input_shape.rfft() : input_shape, kernel, stream);
        }

        if (average) {
            // The weights are necessarily larger than zero, so do a simple division to take the mean.
            const auto shell_shape = Shape4<i64>{1, 1, 1, shell_count};
            const auto shell_strides = shell_shape.strides();
            noa::cuda::utils::ewise_binary(
                    "rotational_average",
                    output, shell_strides,
                    weight_ptr, shell_strides,
                    output, shell_strides, shell_shape,
                    stream, noa::divide_t{});
        }
    }

    #define INSTANTIATE_POLAR(Input, Output) \
    template void cartesian2polar<Remap::HC2FC, Input, Output, void>(   \
        const Input*, Strides4<i64>, Shape4<i64>,                       \
        Output*, const Strides4<i64>&, const Shape4<i64>&,              \
        const Vec2<f32>&, bool, const Vec2<f32>&, bool,                 \
        InterpMode, Stream&)

    #define INSTANTIATE_POLAR_TEXTURE(Input, Output)        \
    template void cartesian2polar<Input, Output, void>(     \
        cudaArray*, cudaTextureObject_t,                    \
        InterpMode, const Shape4<i64>&,                     \
        Output*, const Strides4<i64>&, const Shape4<i64>&,  \
        const Vec2<f32>&, bool, const Vec2<f32>&, bool,     \
        Stream&)

    INSTANTIATE_POLAR(f32, f32);
    INSTANTIATE_POLAR(f64, f64);
    INSTANTIATE_POLAR(c32, c32);
    INSTANTIATE_POLAR(c64, c64);
    INSTANTIATE_POLAR(c32, f32);
    INSTANTIATE_POLAR(c64, f64);

    INSTANTIATE_POLAR_TEXTURE(f32, f32);
    INSTANTIATE_POLAR_TEXTURE(c32, c32);
    INSTANTIATE_POLAR_TEXTURE(c32, f32);

    #define NOA_INSTANT_ROTATIONAL_AVERAGE(Remap, Input, Output, Weight)    \
    template void rotational_average<Remap, Input, Output, Weight, void>(   \
            const Input*, Strides4<i64>, Shape4<i64>,                       \
            Output*, Weight*, bool, Stream&)

    #define NOA_INSTANT_ROTATIONAL_AVERAGE_REMAP(Input, Output, Weight)             \
    NOA_INSTANT_ROTATIONAL_AVERAGE(noa::fft::Remap::H2H, Input, Output, Weight);    \
    NOA_INSTANT_ROTATIONAL_AVERAGE(noa::fft::Remap::HC2H, Input, Output, Weight);   \
    NOA_INSTANT_ROTATIONAL_AVERAGE(noa::fft::Remap::F2H, Input, Output, Weight);    \
    NOA_INSTANT_ROTATIONAL_AVERAGE(noa::fft::Remap::FC2H, Input, Output, Weight)

    NOA_INSTANT_ROTATIONAL_AVERAGE_REMAP(f32, f32, f32);
    NOA_INSTANT_ROTATIONAL_AVERAGE_REMAP(f64, f64, f64);
    NOA_INSTANT_ROTATIONAL_AVERAGE_REMAP(c32, c32, f32);
    NOA_INSTANT_ROTATIONAL_AVERAGE_REMAP(c64, c64, f64);
    NOA_INSTANT_ROTATIONAL_AVERAGE_REMAP(c32, f32, f32);
    NOA_INSTANT_ROTATIONAL_AVERAGE_REMAP(c64, f64, f64);
}
