#pragma once

#include "noa/core/Resize.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa {
    /// Sets the number of element(s) to pad/crop, for each border of each dimension, to get from input_shape to
    /// output_shape, while keeping the centers (defined as shape / 2) of the input and output array aligned.
    /// \param input_shape  Current shape.
    /// \param output_shape Desired shape.
    /// \return             1: The elements to add/remove from the left side of the dimensions.
    ///                     2: The elements to add/remove from the right side of the dimension.
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
    template<typename Input, typename Output,
             typename Value = nt::value_type_t<Output>>
    requires (nt::are_varray_v<Input, Output> &&
              nt::are_almost_same_value_type_v<Input, Output> &&
              nt::is_almost_same_v<nt::value_type_t<Output>, Value>)
    void resize(
            const Input& input,
            const Output& output,
            Vec4<i64> border_left,
            Vec4<i64> border_right,
            Border border_mode = Border::ZERO,
            Value border_value = Value{0}
    ) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(input, output), "The input and output arrays should not overlap");

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:{} and output:{}",
              input.device(), device);

        auto input_shape = input.shape();
        auto input_strides = input.strides();
        auto output_shape = output.shape();
        auto output_strides = output.strides();
        check(all(output_shape.vec == input_shape.vec + border_left + border_right),
              "The output shape {} does not match the expected shape (input:{}, left:{}, right:{})",
              output_shape, input.shape(), border_left, border_right);

        if (all(border_left == 0) and all(border_right == 0)) {
            return copy(input, output);
        } else if (border_mode == Border::NOTHING) {
            // Special case. We can simply copy the input subregion into the output subregion.
            const auto [cropped_input, cropped_output] = extract_common_subregion(
                    input_strides, input_shape,
                    output_strides, output_shape,
                    border_left, border_right);
            return copy(input.subregion(cropped_input), output.subregion(cropped_output));
        }

        // Rearrange output to rightmost:
        const auto order = ni::order(output_strides, output_shape);
        if (any(order != Vec4<i64>{0, 1, 2, 3})) {
            input_strides = ni::reorder(input_strides, order);
            input_shape = ni::reorder(input_shape, order);
            border_left = ni::reorder(border_left, order);
            border_right = ni::reorder(border_right, order);
            output_strides = ni::reorder(output_strides, order);
            output_shape = ni::reorder(output_shape, order);
        }

        using input_accessor_t = AccessorRestrict<const nt::mutable_value_type_t<Input>, 4, i64>;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 4, i64>;
        auto input_accessor = input_accessor_t(input.get(), input_strides);
        auto output_accessor = output_accessor_t(output.get(), output_strides);

        switch (border_mode) {
            case Border::ZERO: {
                const auto op = Resize<Border::ZERO, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, input_shape, output_shape,
                        border_left, border_right, border_value);
                return iwise(output_shape, device, op, input, output);
            }
            case Border::VALUE: {
               const auto op = Resize<Border::VALUE, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, input_shape, output_shape,
                        border_left, border_right, border_value);
                return iwise(output_shape, device, op, input, output);
            }
            case Border::CLAMP: {
               const auto op = Resize<Border::CLAMP, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, input_shape, output_shape,
                        border_left, border_right, border_value);
                return iwise(output_shape, device, op, input, output);
            }
            case Border::PERIODIC: {
               const auto op = Resize<Border::PERIODIC, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, input_shape, output_shape,
                        border_left, border_right, border_value);
                return iwise(output_shape, device, op, input, output);
            }
            case Border::REFLECT: {
               const auto op = Resize<Border::REFLECT, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, input_shape, output_shape,
                        border_left, border_right, border_value);
                return iwise(output_shape, device, op, input, output);
            }
            case Border::MIRROR: {
               const auto op = Resize<Border::MIRROR, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, input_shape, output_shape,
                        border_left, border_right, border_value);
                return iwise(output_shape, device, op, input, output);
            }
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
    template<typename Input, typename Value = nt::mutable_value_type_t<Input>>
    requires (nt::is_varray_of_numeric_v<Input> and
              nt::is_almost_same_v<nt::value_type_t<Input>, Value>)
    [[nodiscard]] auto resize(
            const Input& input,
            const Vec4<i64>& border_left,
            const Vec4<i64>& border_right,
            Border border_mode = Border::ZERO,
            Value border_value = Value{0}
    ) {
        const auto output_shape = Shape4<i64>(input.shape().vec + border_left + border_right);
        check(all(output_shape > 0),
              "Cannot resize [left:{}, right:{}] an array of shape {} into an array of shape {}",
              border_left, border_right, input.shape(), output_shape);
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(output_shape, input.options());
        resize(input, output, border_left, border_right, border_mode, border_value);
        return output;
    }

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \param[in] input    Input array.
    /// \param[out] output  Output array.
    /// \param border_mode  Border mode to use. See Border for more details.
    /// \param border_value Border value. Only used if \p mode is Border::VALUE.
    /// \note \p output == \p input is not valid.
    template<typename Input, typename Output,
             typename Value = nt::value_type_t<Output>>
    requires (nt::are_varray_v<Input, Output> and
              nt::are_almost_same_value_type_v<Input, Output> and
              nt::is_almost_same_v<nt::value_type_t<Input>, Value>)
    void resize(
            const Input& input, const Output& output,
            Border border_mode = Border::ZERO,
            Value border_value = Value{0}
    ) {
        const auto [border_left, border_right] = shape2borders(input.shape(), output.shape());
        resize(input, output, border_left, border_right, border_mode, border_value);
    }

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \param[in] input            Input array.
    /// \param[out] output_shape    Output shape.
    /// \param border_mode          Border mode to use. See Border for more details.
    /// \param border_value         Border value. Only used if \p mode is Border::VALUE.
    template<typename Input, typename Value = nt::mutable_value_type_t<Input>>
    requires (nt::is_varray_v<Input> and nt::is_almost_same_v<nt::value_type_t<Input>, Value>)
    [[nodiscard]] auto resize(
            const Input& input,
            const Shape4<i64>& output_shape,
            Border border_mode = Border::ZERO,
            Value border_value = Value{0}
    ) {
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(output_shape, input.options());
        resize(input, output, border_mode, border_value);
        return output;
    }
}
