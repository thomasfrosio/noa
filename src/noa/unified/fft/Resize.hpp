#pragma once

#include "noa/core/fft/FourierResize.hpp"
#include "noa/core/fft/RemapInterface.hpp"
#include "noa/unified/Resize.hpp"
#include "noa/unified/Factory.hpp"

namespace noa::fft::guts {
    template<Remap REMAP>
    constexpr bool is_valid_resize =
            REMAP == Remap::H2H || REMAP == Remap::F2F ||
            REMAP == Remap::HC2HC || REMAP == Remap::FC2FC;
}

// TODO Rescale values like in IMOD?

namespace noa::fft {
    /// Crops or zero-pads FFT(s).
    /// \tparam REMAP       FFT Remap. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  BDHW logical shape of \p input.
    /// \param[out] output  Resized FFT.
    /// \param output_shape BDHW logical shape of \p output.
    /// \note The batch dimension cannot be resized.
    /// \note If \p REMAP is H2H or F2C, this function can either crop or pad, but cannot do both.
    template<RemapInterface REMAP, typename Input, typename Output>
    requires (nt::are_varray_of_complex_v<Input, Output> or
              nt::are_varray_of_real_v<Input, Output> or
              (nt::is_varray_of_complex_v<Input> and nt::is_varray_of_real_v<Output>) and
              guts::is_valid_resize<REMAP.remap>)
    void resize(
            const Input& input, Shape4<i64> input_shape,
            const Output& output, Shape4<i64> output_shape
    ) {
        if (all(input_shape == output_shape))
            return copy(input, output);

        constexpr bool is_full = to_underlying(REMAP.remap) & Layout::SRC_FULL;
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(input, output), "Input and output arrays should not overlap");
        check(input.shape()[0] == output.shape()[0], "The batch dimension cannot be resized");

        check(all(input.shape() == (is_full ? input_shape : input_shape.rfft())),
              "Given the {} remap, the input fft is expected to have a physical shape of {}, but got {}",
              REMAP.remap, is_full ? input_shape : input_shape.rfft(), input.shape());
        check(all(output.shape() == (is_full ? output_shape : output_shape.rfft())),
              "Given the {} remap, the output fft is expected to have a physical shape of {}, but got {}",
              REMAP.remap, is_full ? output_shape : output_shape.rfft(), output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:{}, output:{}",
              input.device(), device);

        if constexpr (REMAP.remap == Remap::HC2HC) {
            // For centered layouts, use the normal resize instead.
            auto [border_left, border_right] = shape2borders(input_shape.rfft(), output_shape.rfft());
            border_right[3] += std::exchange(border_left[3], 0); // for width, padding goes to the right side only
            noa::resize(input, output, border_left, border_right);
        } else if constexpr (REMAP.remap == Remap::FC2FC) {
            // For centered layouts, use the normal resize instead.
            noa::resize(input, output);
        } else {
            guts::ResizeMode mode{};
            if (all(input_shape >= output_shape)) {
                if constexpr (REMAP.remap == Remap::H2H)
                    mode = guts::ResizeMode::CROP_H2H;
                else if constexpr (REMAP.remap == Remap::F2F)
                    mode = guts::ResizeMode::CROP_F2F;
            } else if (all(input_shape <= output_shape)) {
                if constexpr (REMAP.remap == Remap::H2H)
                    mode = guts::ResizeMode::PAD_H2H;
                else if constexpr (REMAP.remap == Remap::F2F)
                    mode = guts::ResizeMode::PAD_F2F;

                // The way the padding is currently implemented requires the output
                // padded elements to be set to 0, so do it here, on the entire array.
                fill(output, 0);
            } else {
                panic("Cannot crop and pad at the same time with non-centered layouts ({})", REMAP.remap);
            }

            // For the full layout, we can reorder the DHW dimensions if necessary.
            auto input_strides = input.strides();
            auto output_strides = output.strides();
            if (REMAP.remap == Remap::F2F) {
                const auto order_3d = ni::order(output_strides.pop_front(), output_shape.pop_front());
                if (any(order_3d != Vec3<i64>{0, 1, 2})) {
                    const auto order = (order_3d + 1).push_front(0);
                    input_strides = input_strides.reorder(order);
                    output_strides = output_strides.reorder(order);
                    input_shape = input_shape.reorder(order);
                    output_shape = output_shape.reorder(order);
                }
            }

            using input_value_t = nt::mutable_value_type_t<Input>;
            using output_value_t = nt::value_type_t<Output>;
            using input_accessor_t = AccessorRestrictI64<const input_value_t, 4>;
            using output_accessor_t = AccessorRestrictI64<output_value_t, 4>;
            const auto input_accessor = input_accessor_t(input.get(), input_strides);
            const auto output_accessor = output_accessor_t(output.get(), output_strides);
            const auto input_shape_3d = input_shape.pop_front();
            const auto output_shape_3d = output_shape.pop_front();

            // We always loop through the smallest shape. This implies that for padding, the padded elements
            // in the output are NOT set and the backend should make sure these are set to zeros at some point.
            switch (mode) {
                case guts::ResizeMode::PAD_H2H: {
                    auto op = FourierResize<guts::ResizeMode::PAD_H2H, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, input_shape_3d, output_shape_3d);
                    return iwise(input_shape.rfft(), device, op, input, output);
                }
                case guts::ResizeMode::PAD_F2F: {
                    auto op = FourierResize<guts::ResizeMode::PAD_F2F, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, input_shape_3d, output_shape_3d);
                    return iwise(input_shape, device, op, input, output);
                }
                case guts::ResizeMode::CROP_H2H: {
                    auto op = FourierResize<guts::ResizeMode::CROP_H2H, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, input_shape_3d, output_shape_3d);
                    return iwise(output_shape.rfft(), device, op, input, output);
                }
                case guts::ResizeMode::CROP_F2F: {
                    auto op = FourierResize<guts::ResizeMode::CROP_F2F, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, input_shape_3d, output_shape_3d);
                    return iwise(output_shape, device, op, input, output);
                }
            }
        }
    }

    /// Returns a cropped or zero-padded FFT.
    /// \tparam REMAP       FFT Remap. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  BDHW logical shape of \p input.
    /// \param output_shape BDHW logical shape of the output.
    /// \note The batch dimension cannot be resized.
    /// \note If \p REMAP is H2H or F2C, this function can either crop or pad, but cannot do both.
    template<RemapInterface REMAP, typename Input>
    requires (nt::is_varray_of_real_or_complex_v<Input> and guts::is_valid_resize<REMAP.remap>)
    [[nodiscard]] auto resize(
            const Input& input,
            const Shape4<i64>& input_shape,
            const Shape4<i64>& output_shape
    ) {
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(
                to_underlying(REMAP.remap) & Layout::DST_FULL ?
                output_shape : output_shape.rfft(),
                input.options());
        resize<REMAP>(input, input_shape, output, output_shape);
        return output;
    }
}
