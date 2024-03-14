#pragma once

#include "noa/core/fft/FourierRemap.hpp"
#include "noa/core/fft/RemapInterface.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa::fft {
    /// Remaps FFT(s).
    /// \param remap        Remapping operation.
    /// \param[in] input    Input fft to remap.
    /// \param[out] output  Remapped fft.
    /// \param shape        BDHW logical shape.
    /// \note If \p remap is \c H2HC, \p input can be equal to \p output, iff the height and depth are even or 1.
    ///
    /// \bug See noa/docs/rfft_remap.md.
    ///      Remapping a 2d/3d rfft from/to a 2d/3d fft, i.e. \p remap of \c H(C)2F(C) or \c F(C)2H(C), should be
    ///      used with care. If \p input has an anisotropic field, with an anisotropic angle that is not a multiple
    ///      of \c pi/2, and has even-sized dimensions, the remapping will not be correct for the real-Nyquist
    ///      frequencies, except the ones on the pure DHW axes. This situation is rare but can happen if a geometric
    ///      scaling factor is applied to the \p input, or with CTFs with astigmatic defocus.
    ///      While this could be fixed for the remapping fft->rfft, i.e. \c F(C)2H(C), it cannot be fixed for
    ///      the opposite rfft->fft, i.e. \c H(C)2F(C). This shouldn't be a big issue since, 1) these remapping should
    ///      only be done for debugging and visualization anyway, 2) if the remap is done directly after a dft,
    ///      it will work since the field is isotropic in this case, and 3) the problematic frequencies are
    ///      past the Nyquist frequency, so lowpass filtering to Nyquist (fftfreq=0.5) fixes this issue.
    template<typename Input, typename Output>
    requires (nt::are_varray_of_complex_v<Input, Output> or
              nt::are_varray_of_real_v<Input, Output> or
              (nt::is_varray_of_complex_v<Input> and nt::is_varray_of_real_v<Output>))
    void remap(RemapInterface remap, const Input& input, const Output& output, Shape4<i64> shape) {
        using noa::fft::Layout;
        using noa::fft::Remap;
        const auto u8_remap = static_cast<u8>(remap.remap);
        const bool is_src_full = u8_remap & Layout::SRC_FULL;
        const bool is_dst_full = u8_remap & Layout::DST_FULL;

        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(all(input.shape() == (is_src_full ? shape : shape.rfft())),
              "Given the {} remap, the input fft is expected to have a physical shape of {}, but got {}",
              remap.remap, is_src_full ? shape : shape.rfft(), input.shape());
        check(all(output.shape() == (is_dst_full ? shape : shape.rfft())),
              "Given the {} remap, the output fft is expected to have a physical shape of {}, but got {}",
              remap.remap, is_dst_full ? shape : shape.rfft(), output.shape());

        bool is_inplace{};
        if (not ni::are_overlapped(input, output)) {
            check(remap.remap == Remap::H2HC or remap.remap == Remap::HC2H,
                  "In-place remapping is not supported with {}", remap.remap);
            check(input.get() == output.get() and
                  all(input.strides() == output.strides()),
                  "Arrays are overlapping (which triggers in-place remapping), but do not point to the same elements. "
                  "Got input:data={}, input:strides={}, output:data={} and output:strides={}",
                  static_cast<const void*>(input.get()), input.strides(),
                  static_cast<const void*>(output.get()), output.strides());
            check((shape[2] == 1 or is_even(shape[2])) and
                  (shape[1] == 1 or is_even(shape[1])),
                  "In-place remapping requires the depth and height dimensions to have an even number of elements, "
                  "but got shape={}", shape);
            is_inplace = true;
        }

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:{}, output:{}",
              input.device(), device);

        if (remap.remap == Remap::H2H or remap.remap == Remap::HC2HC or
            remap.remap == Remap::F2F or remap.remap == Remap::FC2FC or
            remap.remap == Remap::NONE) {
            if (input != output)
                copy(input, output);
            return;
        }

        // Reordering is only possible for some remaps and this entirely depends on the algorithm we use.
        // Regardless, the batch dimension cannot be reordered.
        auto input_strides = input.strides();
        auto output_strides = output.strides();
        if (remap.remap == Remap::FC2F or remap.remap == Remap::F2FC or
            remap.remap == Remap::FC2H or remap.remap == Remap::F2H) {
            const auto order_3d = ni::order(output_strides.pop_front(), shape.pop_front());
            if (any(order_3d != Vec3<i64>{0, 1, 2})) {
                const auto order = (order_3d + 1).push_front(0);
                input_strides = input_strides.reorder(order);
                output_strides = output_strides.reorder(order);
                shape = shape.reorder(order);
            }
        }

        using input_value_t = nt::mutable_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using input_accessor_t = AccessorRestrictI64<const input_value_t, 4>;
        using output_accessor_t = AccessorRestrictI64<output_value_t, 4>;
        const auto input_accessor = input_accessor_t(input.get(), input_strides);
        const auto output_accessor = output_accessor_t(output.get(), output_strides);
        const auto shape_3d = shape.pop_front();

        switch (remap.remap) {
            case Remap::H2H:
            case Remap::HC2HC:
            case Remap::F2F:
            case Remap::FC2FC:
                break;
            case Remap::H2HC: {
                if (is_inplace) {
                    auto op = FourierRemapInplace<Remap::H2HC, i64, output_accessor_t>(output_accessor, shape_3d);
                    return iwise(shape.rfft(), device, op, input, output);
                } else {
                    auto op = FourierRemap<Remap::H2HC, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, shape_3d);
                    return iwise(shape.rfft(), device, op, input, output);
                }
            }
            case Remap::HC2H: {
                if (is_inplace) {
                    auto op = FourierRemapInplace<Remap::HC2H, i64, output_accessor_t>(output_accessor, shape_3d);
                    return iwise(shape.rfft(), device, op, input, output);
                } else {
                    auto op = FourierRemap<Remap::HC2H, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, shape_3d);
                    return iwise(shape.rfft(), device, op, input, output);
                }
            }
            case Remap::H2F: {
                auto op = FourierRemap<Remap::H2F, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, shape_3d);
                    return iwise(shape, device, op, input, output);
            }
            case Remap::F2H: {
                auto op = FourierRemap<Remap::F2H, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, shape_3d);
                    return iwise(shape.rfft(), device, op, input, output);
            }
            case Remap::F2FC: {
                auto op = FourierRemap<Remap::F2FC, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, shape_3d);
                    return iwise(shape, device, op, input, output);
            }
            case Remap::FC2F: {
                auto op = FourierRemap<Remap::FC2F, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, shape_3d);
                    return iwise(shape, device, op, input, output);
            }
            case Remap::HC2F: {
                auto op = FourierRemap<Remap::HC2F, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, shape_3d);
                    return iwise(shape, device, op, input, output);
            }
            case Remap::F2HC: {
                auto op = FourierRemap<Remap::F2HC, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, shape_3d);
                    return iwise(shape.rfft(), device, op, input, output);
            }
            case Remap::FC2H: {
                auto op = FourierRemap<Remap::FC2H, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, shape_3d);
                    return iwise(shape.rfft(), device, op, input, output);
            }
            case Remap::FC2HC: {
                auto op = FourierRemap<Remap::FC2HC, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, shape_3d);
                    return iwise(shape.rfft(), device, op, input, output);
            }
            case Remap::HC2FC: {
                auto op = FourierRemap<Remap::HC2FC, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, shape_3d);
                    return iwise(shape, device, op, input, output);
            }
            case Remap::H2FC: {
                auto op = FourierRemap<Remap::H2FC, i64, input_accessor_t, output_accessor_t>(
                            input_accessor, output_accessor, shape_3d);
                    return iwise(shape, device, op, input, output);
            }
        }
    }

    /// Remaps fft(s).
    template<typename Input> requires nt::is_varray_of_real_or_complex_v<Input>
    [[nodiscard]] auto remap(RemapInterface remap, const Input& input, const Shape4<i64>& shape) {
        const auto output_shape = to_underlying(remap.remap) & Layout::DST_FULL ? shape : shape.rfft();
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(output_shape, input.options());
        noa::fft::remap(remap, input, output, shape);
        return output;
    }
}
