#pragma once

#include "noa/core/Iwise.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Cast.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa {
    /// Sets the number of element(s) to pad/crop, for each border of each dimension, to get from input_shape to
    /// output_shape, while keeping the centers (defined as shape / 2) of the input and output array aligned.
    /// \param input_shape  Current shape.
    /// \param output_shape Desired shape.
    /// \return             1: The number of elements to add/remove from the left side of the dimensions.
    ///                     2: The number of elements to add/remove from the right side of the dimension.
    ///                     Positive values correspond to padding, while negative values correspond to cropping.
    [[nodiscard]] inline auto shape2borders(const Shape4<i64>& input_shape, const Shape4<i64>& output_shape) {
        const auto diff = output_shape - input_shape;
        const auto border_left = output_shape / 2 - input_shape / 2;
        const auto border_right = diff - border_left;
        return Pair{border_left.vec, border_right.vec};
    }

    /// Resizes the input array(s) by padding and/or cropping the edges of the array.
    /// \param[in] input    Input array.
    /// \param[out] output  Output array. The output shape should be the sum of the input shape and the borders.
    /// \param border_left  Elements to add/remove from the left side of the axes.
    /// \param border_right Elements to add/remove from the right side of the axes.
    /// \param border_mode  Border mode to use. See Border for more details.
    /// \param border_value Border value. Only used for padding if \p mode is Border::VALUE.
    /// \note \p output should not overlap with \p input.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay Output>
    requires (nt::varray_decay_with_compatible_or_spectrum_types<Input, Output>)
    void resize(
        Input&& input,
        Output&& output,
        Vec4<i64> border_left,
        Vec4<i64> border_right,
        Border border_mode = Border::ZERO,
        nt::value_type_t<Output> border_value = {}
    ) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(input, output), "The input and output arrays should not overlap");

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={} and output:device={}",
              input.device(), device);

        auto input_shape = input.shape();
        auto input_strides = input.strides();
        auto output_shape = output.shape();
        auto output_strides = output.strides();
        check(vall(Equal{}, output_shape.vec, input_shape.vec + border_left + border_right),
              "The output shape {} does not match the expected shape (input:shape={}, left:shape={}, right:shape={})",
              output_shape, input.shape(), border_left, border_right);

        if (vall(IsZero{}, border_left) and vall(IsZero{}, border_right)) {
            // Nothing to pad or crop.
            if constexpr (nt::same_mutable_value_type<Input, Output>)
                return copy(std::forward<Input>(input), std::forward<Output>(output));
            else
                return cast(std::forward<Input>(input), std::forward<Output>(output));
        }
        if (border_mode == Border::NOTHING) {
            // Special case. We can simply copy the input subregion into the output subregion.
            const auto [cropped_input, cropped_output] = ng::extract_common_subregion(
                    input_shape, output_shape, border_left, border_right);
            if constexpr (nt::same_mutable_value_type<Input, Output>) {
                return copy(std::forward<Input>(input).subregion(cropped_input),
                            std::forward<Output>(output).subregion(cropped_output));
            } else {
                return cast(std::forward<Input>(input).subregion(cropped_input),
                            std::forward<Output>(output).subregion(cropped_output));
            }
        }

        // Rearrange output to rightmost:
        if (const auto order = ni::order(output_strides, output_shape);
            vany(NotEqual{}, order, Vec4<i64>{0, 1, 2, 3})) {
            input_strides = ni::reorder(input_strides, order);
            input_shape = ni::reorder(input_shape, order);
            border_left = ni::reorder(border_left, order);
            border_right = ni::reorder(border_right, order);
            output_strides = ni::reorder(output_strides, order);
            output_shape = ni::reorder(output_shape, order);
        }

        using input_accessor_t = AccessorRestrict<nt::const_value_type_t<Input>, 4, i64>;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 4, i64>;
        auto input_accessor = input_accessor_t(input.get(), input_strides);
        auto output_accessor = output_accessor_t(output.get(), output_strides);

        switch (border_mode) {
            #define NOA_GENERATE_RESIZE_(border)                                              \
            case border: {                                                                    \
                const auto op = ng::Resize<border, i64, input_accessor_t, output_accessor_t>( \
                    input_accessor, output_accessor, input_shape, output_shape,               \
                    border_left, border_right, border_value);                                 \
                return iwise(output_shape, device, op,                                        \
                             std::forward<Input>(input),                                      \
                             std::forward<Output>(output));                                   \
            }
            NOA_GENERATE_RESIZE_(Border::ZERO)
            NOA_GENERATE_RESIZE_(Border::VALUE)
            NOA_GENERATE_RESIZE_(Border::CLAMP)
            NOA_GENERATE_RESIZE_(Border::PERIODIC)
            NOA_GENERATE_RESIZE_(Border::REFLECT)
            NOA_GENERATE_RESIZE_(Border::MIRROR)

            case Border::NOTHING: {
                break; // unreachable
            }
            default:
                panic("Border not supported. Got: {}", border_mode);
        }
    }

    /// Resizes the input array(s) by padding and/or cropping the edges of the array.
    /// \param[in] input    Input array.
    /// \param border_left  Elements to add/remove from the left side of the axes.
    /// \param border_right Elements to add/remove from the right side of the axes.
    /// \param border_mode  Border mode to use. See Border for more details.
    /// \param border_value Border value. Only used for padding if \p mode is Border::VALUE.
    template<nt::readable_varray_decay_of_numeric Input>
    [[nodiscard]] auto resize(
        Input&& input,
        const Vec4<i64>& border_left,
        const Vec4<i64>& border_right,
        Border border_mode = Border::ZERO,
        nt::mutable_value_type_t<Input> border_value = {}
    ) {
        const auto output_shape = Shape4<i64>(input.shape().vec + border_left + border_right);
        check(not output_shape.is_empty(),
              "Cannot resize [left:{}, right:{}] an array of shape {} into an array of shape {}",
              border_left, border_right, input.shape(), output_shape);

        Array<decltype(border_value)> output(output_shape, input.options());
        resize(std::forward<Input>(input), output, border_left, border_right, border_mode, border_value);
        return output;
    }

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \param[in] input    Input array.
    /// \param[out] output  Output array.
    /// \param border_mode  Border mode to use. See Border for more details.
    /// \param border_value Border value. Only used if \p mode is Border::VALUE.
    /// \note \p output == \p input is not valid.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay Output>
    requires nt::varray_decay_with_compatible_or_spectrum_types<Input, Output>
    void resize(
        Input&& input, Output&& output,
        Border border_mode = Border::ZERO,
        nt::value_type_t<Output> border_value = {}
    ) {
        const auto [border_left, border_right] = shape2borders(input.shape(), output.shape());
        resize(std::forward<Input>(input), std::forward<Output>(output),
               border_left, border_right, border_mode, border_value);
    }

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \param[in] input            Input array.
    /// \param[out] output_shape    Output shape.
    /// \param border_mode          Border mode to use. See Border for more details.
    /// \param border_value         Border value. Only used if \p mode is Border::VALUE.
    template<nt::readable_varray_decay Input>
    [[nodiscard]] auto resize(
        Input&& input,
        const Shape4<i64>& output_shape,
        Border border_mode = Border::ZERO,
        nt::mutable_value_type_t<Input> border_value = {}
    ) {
        Array<decltype(border_value)> output(output_shape, input.options());
        resize(std::forward<Input>(input), output, border_mode, border_value);
        return output;
    }
}
