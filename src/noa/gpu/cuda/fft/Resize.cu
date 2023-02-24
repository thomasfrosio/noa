#include "noa/algorithms/fft/Resize.hpp"
#include "noa/gpu/cuda/fft/Resize.hpp"
#include "noa/gpu/cuda/memory/Resize.hpp"
#include "noa/gpu/cuda/memory/Set.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/utils/Pointers.hpp"

namespace noa::cuda::fft {
    template<noa::fft::Remap REMAP, typename T, typename>
    void resize(const T* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
                T* output, Strides4<i64> output_strides, Shape4<i64> output_shape,
                Stream& stream) {
        // For centered layouts, use the memory::resize instead.
        if constexpr (REMAP == noa::fft::HC2HC) {
            auto [border_left, border_right] = noa::algorithm::memory::borders(input_shape.fft(), output_shape.fft());
            border_right[3] += std::exchange(border_left[3], 0); // for width, padding goes to the right side only
            return noa::cuda::memory::resize(
                    input, input_strides, input_shape.fft(),
                    border_left, border_right,
                    output, output_strides,
                    BorderMode::ZERO, T{0}, stream);
        } else if constexpr (REMAP == noa::fft::FC2FC) {
            return noa::cuda::memory::resize(
                    input, input_strides, input_shape,
                    output, output_strides, output_shape,
                    BorderMode::ZERO, T{0}, stream);
        }

        NOA_ASSERT(input != output && noa::all(input_shape > 0) && noa::all(input_shape > 0));
        NOA_ASSERT(input_shape.batch() == output_shape.batch());
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        // For the full layout, we can reorder the DHW dimensions if necessary.
        if (REMAP == noa::fft::F2F) {
            const auto order_3d = noa::indexing::order(output_strides.pop_front(), output_shape.pop_front());
            if (noa::any(order_3d != Vec3<i64>{0, 1, 2})) {
                const auto order = (order_3d + 1).push_front(0);
                input_strides = input_strides.reorder(order);
                output_strides = output_strides.reorder(order);
                input_shape = input_shape.reorder(order);
                output_shape = output_shape.reorder(order);
            }
        }

        noa::algorithm::fft::ResizeMode mode{};
        if (noa::all(input_shape >= output_shape)) {
            if constexpr (REMAP == noa::fft::H2H)
                mode = noa::algorithm::fft::ResizeMode::CROP_H2H;
            else if constexpr (REMAP == noa::fft::F2F)
                mode = noa::algorithm::fft::ResizeMode::CROP_F2F;

        } else if (noa::all(input_shape <= output_shape)) {
            // The way the padding is currently implemented requires the output
            // padded elements to be set to 0, so do it here, on the entire array.
            if constexpr (REMAP == noa::fft::H2H) {
                mode = noa::algorithm::fft::ResizeMode::PAD_H2H;
                noa::cuda::memory::set(output, output_strides, output_shape.fft(), T{0}, stream);
            } else if constexpr (REMAP == noa::fft::F2F) {
                mode = noa::algorithm::fft::ResizeMode::PAD_F2F;
                noa::cuda::memory::set(output, output_strides, output_shape, T{0}, stream);
            }

        } else {
            NOA_THROW("Cannot crop and pad at the same time with layout {}", REMAP);
        }

        switch (mode) {
            case noa::algorithm::fft::ResizeMode::PAD_H2H: {
                constexpr auto MODE = noa::algorithm::fft::ResizeMode::PAD_H2H;
                const auto [kernel, iwise_shape] = noa::algorithm::fft::resize<MODE>(
                        input, input_strides, input_shape, output, output_strides, output_shape);
                return noa::cuda::utils::iwise_4d("resize", iwise_shape, kernel, stream);
            }
            case noa::algorithm::fft::ResizeMode::PAD_F2F: {
                constexpr auto MODE = noa::algorithm::fft::ResizeMode::PAD_F2F;
                const auto [kernel, iwise_shape] = noa::algorithm::fft::resize<MODE>(
                        input, input_strides, input_shape, output, output_strides, output_shape);
                return noa::cuda::utils::iwise_4d("resize", iwise_shape, kernel, stream);
            }
            case noa::algorithm::fft::ResizeMode::CROP_H2H: {
                constexpr auto MODE = noa::algorithm::fft::ResizeMode::CROP_H2H;
                const auto [kernel, iwise_shape] = noa::algorithm::fft::resize<MODE>(
                        input, input_strides, input_shape, output, output_strides, output_shape);
                return noa::cuda::utils::iwise_4d("resize", iwise_shape, kernel, stream);
            }
            case noa::algorithm::fft::ResizeMode::CROP_F2F: {
                constexpr auto MODE = noa::algorithm::fft::ResizeMode::CROP_F2F;
                const auto [kernel, iwise_shape] = noa::algorithm::fft::resize<MODE>(
                        input, input_strides, input_shape, output, output_strides, output_shape);
                return noa::cuda::utils::iwise_4d("resize", iwise_shape, kernel, stream);
            }
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(R, T) \
    template void resize<R, T, void>(const T*, Strides4<i64>, Shape4<i64>, T*, Strides4<i64>, Shape4<i64>, Stream&)

    #define NOA_INSTANTIATE_RESIZE_ALL(T)           \
    NOA_INSTANTIATE_RESIZE_(noa::fft::HC2HC, T);    \
    NOA_INSTANTIATE_RESIZE_(noa::fft::FC2FC, T);    \
    NOA_INSTANTIATE_RESIZE_(noa::fft::H2H, T);      \
    NOA_INSTANTIATE_RESIZE_(noa::fft::F2F, T)

    NOA_INSTANTIATE_RESIZE_ALL(f16);
    NOA_INSTANTIATE_RESIZE_ALL(f32);
    NOA_INSTANTIATE_RESIZE_ALL(f64);
    NOA_INSTANTIATE_RESIZE_ALL(c16);
    NOA_INSTANTIATE_RESIZE_ALL(c32);
    NOA_INSTANTIATE_RESIZE_ALL(c64);
}
