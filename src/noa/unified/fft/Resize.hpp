#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/Enums.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/unified/Resize.hpp"
#include "noa/unified/Factory.hpp"

// TODO Rescale values like in IMOD?

namespace noa::fft::guts {
    enum class FourierResizeMode {
        PAD_H2H,
        PAD_F2F,
        CROP_H2H,
        CROP_F2F
    };

    template<FourierResizeMode MODE,
             nt::sinteger Index,
             nt::readable_nd<4> Input,
             nt::writable_nd<4> Output>
    class FourierResize {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::compatible_or_spectrum_types<input_value_type, output_value_type>);

        using dh_shape_type = std::conditional_t<
            MODE == FourierResizeMode::CROP_H2H or MODE == FourierResizeMode::PAD_H2H,
            Shape2<index_type>, Empty>;
        using dhw_vec_type = std::conditional_t<
            MODE == FourierResizeMode::CROP_F2F or MODE == FourierResizeMode::PAD_F2F,
            Shape3<index_type>, Empty>;

        constexpr FourierResize(
            const input_type& input,
            const output_type& output,
            const Shape3<index_type>& input_shape,
            const Shape3<index_type>& output_shape
        ) : m_input(input),
            m_output(output)
        {
            if constexpr (MODE == FourierResizeMode::CROP_H2H) {
                m_input_shape = input_shape.pop_back();
                m_output_shape = output_shape.pop_back();

            } else if constexpr (MODE == FourierResizeMode::PAD_H2H) {
                m_input_shape = input_shape.pop_back();
                m_output_shape = output_shape.pop_back();

            } else if constexpr (MODE == FourierResizeMode::CROP_F2F) {
                m_offset = input_shape - output_shape;
                m_limit = (output_shape + 1) / 2;

            } else if constexpr (MODE == FourierResizeMode::PAD_F2F) {
                m_offset = output_shape - input_shape;
                m_limit = (input_shape + 1) / 2;
            }
        }

        constexpr void operator()(index_type i, index_type j, index_type k, index_type l) const {
            if constexpr (MODE == FourierResizeMode::CROP_H2H) {
                const auto ij = j < (m_output_shape[0] + 1) / 2 ? j : j + m_input_shape[0] - m_output_shape[0];
                const auto ik = k < (m_output_shape[1] + 1) / 2 ? k : k + m_input_shape[1] - m_output_shape[1];
                m_output(i, j, k, l) = cast_or_abs_squared<output_value_type>(m_input(i, ij, ik, l));

            } else if constexpr (MODE == FourierResizeMode::PAD_H2H) {
                const auto oj = j < (m_input_shape[0] + 1) / 2 ? j : j + m_output_shape[0] - m_input_shape[0];
                const auto ok = k < (m_input_shape[1] + 1) / 2 ? k : k + m_output_shape[1] - m_input_shape[1];
                m_output(i, oj, ok, l) = cast_or_abs_squared<output_value_type>(m_input(i, j, k, l));

            } else if constexpr (MODE == FourierResizeMode::CROP_F2F) {
                const auto ij = j < m_limit[0] ? j : j + m_offset[0];
                const auto ik = k < m_limit[1] ? k : k + m_offset[1];
                const auto il = l < m_limit[2] ? l : l + m_offset[2];
                m_output(i, j, k, l) =  cast_or_abs_squared<output_value_type>(m_input(i, ij, ik, il));

            } else if constexpr (MODE == FourierResizeMode::PAD_F2F) {
                const auto oj = j < m_limit[0] ? j : j + m_offset[0];
                const auto ok = k < m_limit[1] ? k : k + m_offset[1];
                const auto ol = l < m_limit[2] ? l : l + m_offset[2];
                m_output(i, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(i, j, k, l));

            } else {
                static_assert(nt::always_false<index_type>);
            }
        }

    private:
        input_type m_input;
        output_type m_output;
        NOA_NO_UNIQUE_ADDRESS dh_shape_type m_input_shape{};
        NOA_NO_UNIQUE_ADDRESS dh_shape_type m_output_shape{};
        NOA_NO_UNIQUE_ADDRESS dhw_vec_type m_offset{};
        NOA_NO_UNIQUE_ADDRESS dhw_vec_type m_limit{};
    };
}

namespace noa::fft {
    /// Crops or zero-pads FFT(s).
    /// \tparam REMAP       FFT layouts. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  BDHW logical shape of \p input.
    /// \param[out] output  Resized FFT.
    /// \param output_shape BDHW logical shape of \p output.
    ///
    /// \note The batch dimension cannot be resized.
    /// \note If \p REMAP is H2H or F2C, this function can either crop or pad, but cannot do both.
    /// \note This function can also perform a cast or compute the power spectrum of the input, depending on the
    ///       input and output types.
    template<Layout REMAP, nt::readable_varray_decay Input, nt::writable_varray_decay Output>
    requires (nt::varray_decay_with_compatible_or_spectrum_types<Input, Output> and not REMAP.has_layout_change())
    void resize(
        Input&& input, Shape4<i64> input_shape,
        Output&& output, Shape4<i64> output_shape
    ) {
        using guts::FourierResizeMode;
        using input_value_t = nt::mutable_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;

        if (vall(Equal{}, input_shape, output_shape)) {
            if constexpr (nt::same_as<input_value_t, output_value_t>) {
                return copy(std::forward<Input>(input), std::forward<Output>(output));
            } else {
                return cast(std::forward<Input>(input), std::forward<Output>(output));
            }
        }

        constexpr bool is_full = REMAP.is_fx2fx();
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(input, output), "Input and output arrays should not overlap");
        check(input.shape()[0] == output.shape()[0], "The batch dimension cannot be resized");

        check(vall(Equal{}, input.shape(), (is_full ? input_shape : input_shape.rfft())),
              "Given the {} remap, the input fft is expected to have a physical shape of {}, but got {}",
              REMAP, is_full ? input_shape : input_shape.rfft(), input.shape());
        check(vall(Equal{}, output.shape(), (is_full ? output_shape : output_shape.rfft())),
              "Given the {} remap, the output fft is expected to have a physical shape of {}, but got {}",
              REMAP, is_full ? output_shape : output_shape.rfft(), output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        if constexpr (REMAP == Layout::HC2HC) {
            // For centered layouts, use the normal resize instead.
            auto [border_left, border_right] = shape2borders(input_shape.rfft(), output_shape.rfft());
            border_right[3] += std::exchange(border_left[3], 0); // for width, padding goes to the right side only
            return noa::resize(std::forward<Input>(input), std::forward<Output>(output), border_left, border_right);

        } else if constexpr (REMAP == Layout::FC2FC) {
            // For centered layouts, use the normal resize instead.
            return noa::resize(std::forward<Input>(input), std::forward<Output>(output));

        } else {
            FourierResizeMode mode{};
            if (vall(GreaterEqual{}, input_shape, output_shape)) {
                if constexpr (REMAP == Layout::H2H)
                    mode = FourierResizeMode::CROP_H2H;
                else if constexpr (REMAP == Layout::F2F)
                    mode = FourierResizeMode::CROP_F2F;
            } else if (vall(LessEqual{}, input_shape, output_shape)) {
                if constexpr (REMAP == Layout::H2H)
                    mode = FourierResizeMode::PAD_H2H;
                else if constexpr (REMAP == Layout::F2F)
                    mode = FourierResizeMode::PAD_F2F;

                // The way the padding is currently implemented requires the padded elements
                // to be zero initialized, so do it here, on the entire array.
                fill(output, output_value_t{});
            } else {
                panic("Cannot crop and pad at the same time with non-centered layouts ({})", REMAP);
            }

            auto input_strides = input.strides();
            auto output_strides = output.strides();
            if (REMAP == Layout::F2F) { // TODO h2h depth-height can be reordered
                const auto order_3d = ni::order(output_strides.pop_front(), output_shape.pop_front());
                if (vany(NotEqual{}, order_3d, Vec{0, 1, 2})) {
                    const auto order = (order_3d + 1).push_front(0);
                    input_strides = input_strides.reorder(order);
                    output_strides = output_strides.reorder(order);
                    input_shape = input_shape.reorder(order);
                    output_shape = output_shape.reorder(order);
                }
            }

            using input_accessor_t = AccessorRestrictI64<const input_value_t, 4>;
            using output_accessor_t = AccessorRestrictI64<output_value_t, 4>;
            const auto input_accessor = input_accessor_t(input.get(), input_strides);
            const auto output_accessor = output_accessor_t(output.get(), output_strides);
            const auto input_shape_3d = input_shape.pop_front();
            const auto output_shape_3d = output_shape.pop_front();

            // We always loop through the smallest shape. This implies that for padding, the padded elements
            // in the output are NOT set and the backend should make sure these are set to zeros at some point.
            switch (mode) {
                case FourierResizeMode::PAD_H2H: {
                    auto op = guts::FourierResize<FourierResizeMode::PAD_H2H, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, input_shape_3d, output_shape_3d);
                    return iwise(input_shape.rfft(), device, op,
                                 std::forward<Input>(input), std::forward<Output>(output));
                }
                case FourierResizeMode::PAD_F2F: {
                    auto op = guts::FourierResize<FourierResizeMode::PAD_F2F, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, input_shape_3d, output_shape_3d);
                    return iwise(input_shape, device, op,
                                 std::forward<Input>(input), std::forward<Output>(output));
                }
                case FourierResizeMode::CROP_H2H: {
                    auto op = guts::FourierResize<FourierResizeMode::CROP_H2H, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, input_shape_3d, output_shape_3d);
                    return iwise(output_shape.rfft(), device, op,
                                 std::forward<Input>(input), std::forward<Output>(output));
                }
                case FourierResizeMode::CROP_F2F: {
                    auto op = guts::FourierResize<FourierResizeMode::CROP_F2F, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, input_shape_3d, output_shape_3d);
                    return iwise(output_shape, device, op,
                                 std::forward<Input>(input), std::forward<Output>(output));
                }
            }
        }
    }

    /// Returns a cropped or zero-padded FFT.
    /// \tparam REMAP       FFT layouts. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  BDHW logical shape of \p input.
    /// \param output_shape BDHW logical shape of the output.
    /// \note The batch dimension cannot be resized.
    /// \note If \p REMAP is H2H or F2C, this function can either crop or pad, but cannot do both.
    template<Layout REMAP, nt::readable_varray_decay_of_numeric Input>
    requires (not REMAP.has_layout_change())
    [[nodiscard]] auto resize(
        Input&& input,
        const Shape4<i64>& input_shape,
        const Shape4<i64>& output_shape
    ) {
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(REMAP.is_fx2fx() ? output_shape : output_shape.rfft(), input.options());
        resize<REMAP>(std::forward<Input>(input), input_shape, output, output_shape);
        return output;
    }
}
