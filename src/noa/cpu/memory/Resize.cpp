#include "noa/cpu/memory/Copy.hpp"
#include "noa/cpu/memory/Resize.hpp"
#include "noa/cpu/utils/Iwise.hpp"
#include "noa/algorithms/memory/Resize.hpp"

namespace noa::cpu::memory {
    template<typename T, typename>
    void resize(const T* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
                Vec4<i64> border_left, Vec4<i64> border_right,
                T* output, Strides4<i64> output_strides,
                BorderMode border_mode, T border_value, i64 threads) {

        if (noa::all(border_left == 0) && noa::all(border_right == 0))
            return copy(input, input_strides, output, output_strides, input_shape, threads);

        auto output_shape = Shape4<i64>(input_shape.vec() + border_left + border_right);
        NOA_ASSERT(noa::all(output_shape >= 1));

        // Rearrange:
        const auto order = indexing::order(output_strides, output_shape);
        if (noa::any(order != Vec4<i64>{0, 1, 2, 3})) {
            input_strides = indexing::reorder(input_strides, order);
            input_shape = indexing::reorder(input_shape, order);
            border_left = indexing::reorder(border_left, order);
            border_right = indexing::reorder(border_right, order);
            output_strides = indexing::reorder(output_strides, order);
            output_shape = indexing::reorder(output_shape, order);
        }

        switch (border_mode) {
            case BorderMode::NOTHING: {
                // Special case. We can copy input subregion into the output subregion,
                // and simply ignore the padded elements.
                const auto [cropped_input, cropped_output] =
                        noa::algorithm::memory::extract_common_subregion(
                                input_strides, input_shape,
                                output_strides, output_shape,
                                border_left, border_right);
                return copy(input + cropped_input.offset, cropped_input.strides,
                            output + cropped_output.offset, cropped_output.strides,
                            cropped_output.shape, threads);
            }
            case BorderMode::ZERO: {
                const auto kernel = noa::algorithm::memory::resize<BorderMode::ZERO, i64, i64>(
                        input, input_strides, input_shape,
                        output, output_strides, output_shape,
                        border_left, border_right);
                return noa::cpu::utils::iwise_4d(output_shape, kernel, threads);
            }
            case BorderMode::VALUE: {
                const auto kernel = noa::algorithm::memory::resize<BorderMode::VALUE, i64, i64>(
                        input, input_strides, input_shape,
                        output, output_strides, output_shape,
                        border_left, border_right, border_value);
                return noa::cpu::utils::iwise_4d(output_shape, kernel, threads);
            }
            case BorderMode::CLAMP: {
                const auto kernel = noa::algorithm::memory::resize<BorderMode::CLAMP, i64, i64>(
                        input, input_strides, input_shape,
                        output, output_strides, output_shape,
                        border_left, border_right);
                return noa::cpu::utils::iwise_4d(output_shape, kernel, threads);
            }
            case BorderMode::PERIODIC: {
                const auto kernel = noa::algorithm::memory::resize<BorderMode::PERIODIC, i64, i64>(
                        input, input_strides, input_shape,
                        output, output_strides, output_shape,
                        border_left, border_right);
                return noa::cpu::utils::iwise_4d(output_shape, kernel, threads);
            }
            case BorderMode::REFLECT: {
                const auto kernel = noa::algorithm::memory::resize<BorderMode::REFLECT, i64, i64>(
                        input, input_strides, input_shape,
                        output, output_strides, output_shape,
                        border_left, border_right);
                return noa::cpu::utils::iwise_4d(output_shape, kernel, threads);
            }
            case BorderMode::MIRROR: {
                const auto kernel = noa::algorithm::memory::resize<BorderMode::MIRROR, i64, i64>(
                        input, input_strides, input_shape,
                        output, output_strides, output_shape,
                        border_left, border_right);
                return noa::cpu::utils::iwise_4d(output_shape, kernel, threads);
            }
            default:
                NOA_THROW("BorderMode not supported. Got: {}", border_mode);
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T)          \
    template void resize<T, void>(              \
        const T*, Strides4<i64>, Shape4<i64>,   \
        Vec4<i64>, Vec4<i64>,                   \
        T*, Strides4<i64>,                      \
        BorderMode, T, i64)

    NOA_INSTANTIATE_RESIZE_(i8);
    NOA_INSTANTIATE_RESIZE_(i16);
    NOA_INSTANTIATE_RESIZE_(i32);
    NOA_INSTANTIATE_RESIZE_(i64);
    NOA_INSTANTIATE_RESIZE_(u8);
    NOA_INSTANTIATE_RESIZE_(u16);
    NOA_INSTANTIATE_RESIZE_(u32);
    NOA_INSTANTIATE_RESIZE_(u64);
    NOA_INSTANTIATE_RESIZE_(f16);
    NOA_INSTANTIATE_RESIZE_(f32);
    NOA_INSTANTIATE_RESIZE_(f64);
    NOA_INSTANTIATE_RESIZE_(c16);
    NOA_INSTANTIATE_RESIZE_(c32);
    NOA_INSTANTIATE_RESIZE_(c64);
}
