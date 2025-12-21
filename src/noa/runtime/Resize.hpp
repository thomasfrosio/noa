#pragma once

#include "noa/runtime/Array.hpp"
#include "noa/runtime/Iwise.hpp"

namespace noa::details {
    /// 4d index-wise operator to resize an array, out of place.
    /// \details The "border_left" and "border_right", specify the number of elements to crop (negative value)
    ///          or pad (positive value) on the left or right side of each dimension. Padded elements are handled
    ///          according to the Border. The input and output arrays should not overlap.
    template<Border MODE,
             nt::sinteger Index,
             nt::readable_nd<4> Input,
             nt::writable_nd<4> Output>
    class Resize {
    public:
        static_assert(MODE != Border::NOTHING);

        using input_accessor_type = Input;
        using output_accessor_type = Output;
        using output_value_type = nt::value_type_t<output_accessor_type>;
        using index_type = Index;
        using indices_type = Vec<index_type, 4>;
        using shape_type = Shape<index_type, 4>;
        using output_value_or_empty_type = std::conditional_t<MODE == Border::VALUE, output_value_type, Empty>;

        static constexpr bool IS_BOUNDLESS = MODE != Border::VALUE and MODE != Border::ZERO;
        using index4_or_empty_type = std::conditional_t<IS_BOUNDLESS, Empty, indices_type>;

    public:
        constexpr Resize(
            const input_accessor_type& input_accessor,
            const output_accessor_type& output_accessor,
            const shape_type& input_shape,
            const shape_type& output_shape,
            const indices_type& border_left,
            const indices_type& border_right,
            output_value_type cvalue
        ) :
            m_input(input_accessor),
            m_output(output_accessor),
            m_input_shape(input_shape),
            m_crop_left(min(border_left, index_type{}) * -1),
            m_pad_left(max(border_left, index_type{}))
        {
            if constexpr (MODE == Border::VALUE or MODE == Border::ZERO) {
                const auto pad_right = max(border_right, index_type{});
                m_right = output_shape.vec - pad_right;
            } else {
                (void) border_right;
                (void) output_shape;
            }

            if constexpr (MODE == Border::VALUE)
                m_cvalue = cvalue;
            else
                (void) cvalue;
        }

        constexpr void operator()(const indices_type& output_indices) const {
            const auto input_indices = output_indices - m_pad_left + m_crop_left;

            if constexpr (MODE == Border::VALUE or MODE == Border::ZERO) {
                if constexpr (MODE == Border::VALUE) {
                    m_output(output_indices) =
                        output_indices >= m_pad_left and output_indices < m_right ?
                        cast_or_abs_squared<output_value_type>(m_input(input_indices)) : m_cvalue;
                } else {
                    m_output(output_indices) =
                        output_indices >= m_pad_left and output_indices < m_right ?
                        cast_or_abs_squared<output_value_type>(m_input(input_indices)) : output_value_type{};
                }
            } else { // CLAMP or PERIODIC or MIRROR or REFLECT
                const indices_type indices_bounded = index_at<MODE>(input_indices, m_input_shape);
                m_output(output_indices) = cast_or_abs_squared<output_value_type>(m_input(indices_bounded));
            }
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        shape_type m_input_shape;
        indices_type m_crop_left;
        indices_type m_pad_left;
        NOA_NO_UNIQUE_ADDRESS index4_or_empty_type m_right;
        NOA_NO_UNIQUE_ADDRESS output_value_or_empty_type m_cvalue;
    };

    /// Computes the common subregions between the input and output.
    /// These can then be used to copy the input subregion into the output subregion.
    [[nodiscard]] constexpr auto extract_common_subregion(
        const Shape4& input_shape, const Shape4& output_shape,
        const Vec<isize, 4>& border_left, const Vec<isize, 4>& border_right
    ) noexcept -> Pair<Subregion<4, Slice, Slice, Slice, Slice>,
                       Subregion<4, Slice, Slice, Slice, Slice>> {
        // Exclude the regions in the input that don't end up in the output.
        const auto crop_left = min(border_left, 0) * -1;
        const auto crop_right = min(border_right, 0) * -1;
        const auto cropped_input = make_subregion<4>(
            Slice{crop_left[0], input_shape[0] - crop_right[0]},
            Slice{crop_left[1], input_shape[1] - crop_right[1]},
            Slice{crop_left[2], input_shape[2] - crop_right[2]},
            Slice{crop_left[3], input_shape[3] - crop_right[3]});

        // Exclude the regions in the output that are not from the input.
        const auto pad_left = max(border_left, 0);
        const auto pad_right = max(border_right, 0);
        const auto cropped_output = make_subregion<4>(
            Slice{pad_left[0], output_shape[0] - pad_right[0]},
            Slice{pad_left[1], output_shape[1] - pad_right[1]},
            Slice{pad_left[2], output_shape[2] - pad_right[2]},
            Slice{pad_left[3], output_shape[3] - pad_right[3]});

        // One can now copy cropped_input -> cropped_output.
        return {cropped_input, cropped_output};
    }
}

namespace noa {
    /// Sets the number of element(s) to pad/crop, for each border of each dimension, to get from input_shape to
    /// output_shape, while keeping the centers (defined as shape / 2) of the input and output array aligned.
    /// \param input_shape  Current shape.
    /// \param output_shape Desired shape.
    /// \return             1: The number of elements to add/remove from the left side of the dimensions.
    ///                     2: The number of elements to add/remove from the right side of the dimension.
    ///                     Positive values correspond to padding, while negative values correspond to cropping.
    template<usize N> requires (1 <= N and N <= 4)
    [[nodiscard]] auto shape2borders(const Shape<isize, N>& input_shape, const Shape<isize, N>& output_shape) {
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
        Vec<isize, 4> border_left,
        Vec<isize, 4> border_right,
        Border border_mode = Border::ZERO,
        nt::value_type_t<Output> border_value = {}
    ) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not are_overlapped(input, output), "The input and output arrays should not overlap");

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={} and output:device={}",
              input.device(), device);

        auto input_shape = input.shape();
        auto input_strides = input.strides();
        auto output_shape = output.shape();
        auto output_strides = output.strides();
        check(output_shape.vec == (input_shape.vec + border_left + border_right),
              "The output shape {} does not match the expected shape (input:shape={}, left:shape={}, right:shape={})",
              output_shape, input.shape(), border_left, border_right);

        if (border_left == 0 and border_right == 0) {
            // Nothing to pad or crop.
            if constexpr (nt::same_mutable_value_type<Input, Output>)
                return copy(std::forward<Input>(input), std::forward<Output>(output));
            else
                return cast(std::forward<Input>(input), std::forward<Output>(output));
        }
        if (border_mode == Border::NOTHING) {
            // Special case. We can simply copy the input subregion into the output subregion.
            const auto [cropped_input, cropped_output] = nd::extract_common_subregion(
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
        const auto order = rightmost_order(output_strides, output_shape);
        if (order != Vec<isize, 4>{0, 1, 2, 3}) {
            input_strides = reorder(input_strides, order);
            input_shape = reorder(input_shape, order);
            border_left = reorder(border_left, order);
            border_right = reorder(border_right, order);
            output_strides = reorder(output_strides, order);
            output_shape = reorder(output_shape, order);
        }

        using input_accessor_t = AccessorRestrict<nt::const_value_type_t<Input>, 4, isize>;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 4, isize>;
        auto input_accessor = input_accessor_t(input.get(), input_strides);
        auto output_accessor = output_accessor_t(output.get(), output_strides);

        switch (border_mode) {
            #define NOA_GENERATE_RESIZE_(border)                                                \
            case border: {                                                                      \
                const auto op = nd::Resize<border, isize, input_accessor_t, output_accessor_t>( \
                    input_accessor, output_accessor, input_shape, output_shape,                 \
                    border_left, border_right, border_value);                                   \
                return iwise(output_shape, device, op,                                          \
                             std::forward<Input>(input),                                        \
                             std::forward<Output>(output));                                     \
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
        const Vec<isize, 4>& border_left,
        const Vec<isize, 4>& border_right,
        Border border_mode = Border::ZERO,
        nt::mutable_value_type_t<Input> border_value = {}
    ) {
        const auto output_shape = Shape4(input.shape().vec + border_left + border_right);
        check(not output_shape.is_empty(),
              "Cannot resize [left:{}, right:{}] an array of shape {} into an array of shape {}",
              border_left, border_right, input.shape(), output_shape);

        auto output = Array<decltype(border_value)>(output_shape, input.options());
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
        const Shape4& output_shape,
        Border border_mode = Border::ZERO,
        nt::mutable_value_type_t<Input> border_value = {}
    ) {
        auto output = Array<decltype(border_value)>(output_shape, input.options());
        resize(std::forward<Input>(input), output, border_mode, border_value);
        return output;
    }
}
