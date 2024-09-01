#pragma once

#include "noa/core/fft/FourierRemap.hpp"
#include "noa/core/Remap.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Cast.hpp"

namespace noa::fft {
    /// Remaps fft(s).
    /// \param remap        Remapping operation.
    /// \param[in] input    Input fft to remap.
    /// \param[out] output  Remapped fft.
    /// \param shape        BDHW logical shape.
    ///
    /// \note If \p remap is \c h2hc, \p input can be equal to \p output, iff the height and depth are even or 1.
    /// \note This function can also perform a cast or compute the power spectrum of the input, depending on the
    ///       input and output types.
    ///
    /// \bug See noa/docs/rfft_remap.md.
    ///      The rfft<->fft remapping should be used with care. If \p input has an anisotropic field, with an
    ///      anisotropic angle that is not a multiple of \c pi/2, and has even-sized dimensions, the remapping will
    ///      not be correct for the real-Nyquist frequencies, except the ones on the DHW axes. This situation is rare
    ///      but can happen if a geometric scaling factor is applied to the \p input, or with CTFs with astigmatic
    ///      defocus. While this could be fixed for the fft->rfft remapping, it cannot be fixed for the opposite
    ///      rfft->fft. This shouldn't be a big issue since, 1) these remapping should only be done for debugging and
    ///      visualization anyway, 2) if the remap is done directly after a dft, it will work since the field is
    ///      isotropic in this case, and 3) the problematic frequencies are past the Nyquist frequency, so lowpass
    ///      filtering to Nyquist (fftfreq=0.5) fixes this issue.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay Output>
    requires nt::varray_decay_with_compatible_or_spectrum_types<Input, Output>
    void remap(Remap remap, Input&& input, Output&& output, Shape4<i64> shape) {
        using input_t = nt::mutable_value_type_t<Input>;
        using output_t = nt::value_type_t<Output>;

        // Check shape.
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(vall(Equal{}, input.shape(), (remap.is_fx2xx() ? shape : shape.rfft())),
              "Given the {} remap, the input fft is expected to have a physical shape of {}, but got {}",
              remap, remap.is_fx2xx() ? shape : shape.rfft(), input.shape());
        check(vall(Equal{}, output.shape(), (remap.is_xx2fx() ? shape : shape.rfft())),
              "Given the {} remap, the output fft is expected to have a physical shape of {}, but got {}",
              remap, remap.is_xx2fx() ? shape : shape.rfft(), output.shape());

        // Shortcut if there's no remapping.
        const bool is_inplace = ni::are_overlapped(input, output);
        if (not remap.has_layout_change()) {
            if constexpr (nt::same_as<input_t, output_t>) {
                if (not is_inplace)
                    copy(std::forward<Input>(input), std::forward<Output>(output));
            } else {
                cast(std::forward<Input>(input), std::forward<Output>(output));
            }
            return;
        }

        // Special in-place rfft case:
        if (is_inplace) {
            check(remap.is_any(Remap::H2HC, Remap::HC2H),
                  "In-place remapping is not supported with {}", remap);
            check(static_cast<const void*>(input.get()) == static_cast<const void*>(output.get()) and
                  vall(Equal{}, input.strides(), output.strides()),
                  "Arrays are overlapping (which triggers in-place remapping), but do not point to the same elements. "
                  "Got input:data={}, input:strides={}, output:data={} and output:strides={}",
                  static_cast<const void*>(input.get()), input.strides(),
                  static_cast<const void*>(output.get()), output.strides());
            check((shape[2] == 1 or is_even(shape[2])) and
                  (shape[1] == 1 or is_even(shape[1])),
                  "In-place remapping requires the depth and height dimensions to have an even number of elements, "
                  "but got shape={}", shape);
        }

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        // Reordering is only possible for some remaps.
        // Regardless, the batch dimension cannot be reordered.
        auto input_strides = input.strides();
        auto output_strides = output.strides();
        if (remap.is_any(Remap::FC2F, Remap::F2FC)) {
            const auto order_3d = ni::order(output_strides.pop_front(), shape.pop_front());
            if (vany(NotEqual{}, order_3d, Vec{0, 1, 2})) {
                const auto order = (order_3d + 1).push_front(0);
                input_strides = input_strides.reorder(order);
                output_strides = output_strides.reorder(order);
                shape = shape.reorder(order);
            }
        }

        using input_accessor_t = AccessorRestrictI64<const input_t, 4>;
        using output_accessor_t = AccessorRestrictI64<output_t, 4>;
        const auto input_accessor = input_accessor_t(input.get(), input_strides);
        const auto output_accessor = output_accessor_t(output.get(), output_strides);
        const auto shape_3d = shape.pop_front();

        auto iwise_shape = remap.is_xx2fx() ? shape : shape.rfft();
        if (is_inplace)
            iwise_shape[2] = max(iwise_shape[2] / 2, i64{1}); // iterate only through half of height

        switch (remap) {
            case Remap::H2H:
            case Remap::HC2HC:
            case Remap::F2F:
            case Remap::FC2FC:
                break;
            case Remap::H2HC: {
                if (is_inplace) {
                    auto op = guts::FourierRemapInplace<Remap::H2HC, i64, output_accessor_t>(output_accessor, shape_3d);
                    return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
                }
                auto op = guts::FourierRemap<Remap::H2HC, i64, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Remap::HC2H: {
                if (is_inplace) {
                    auto op = guts::FourierRemapInplace<Remap::HC2H, i64, output_accessor_t>(output_accessor, shape_3d);
                    return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
                }
                auto op = guts::FourierRemap<Remap::HC2H, i64, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Remap::H2F: {
                auto op = guts::FourierRemap<Remap::H2F, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Remap::F2H: {
                auto op = guts::FourierRemap<Remap::F2H, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Remap::F2FC: {
                auto op = guts::FourierRemap<Remap::F2FC, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Remap::FC2F: {
                auto op = guts::FourierRemap<Remap::FC2F, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Remap::HC2F: {
                auto op = guts::FourierRemap<Remap::HC2F, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Remap::F2HC: {
                auto op = guts::FourierRemap<Remap::F2HC, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Remap::FC2H: {
                auto op = guts::FourierRemap<Remap::FC2H, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Remap::FC2HC: {
                auto op = guts::FourierRemap<Remap::FC2HC, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Remap::HC2FC: {
                auto op = guts::FourierRemap<Remap::HC2FC, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Remap::H2FC: {
                auto op = guts::FourierRemap<Remap::H2FC, i64, input_accessor_t, output_accessor_t>(
                        input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
        }
    }

    /// Remaps fft(s).
    template<nt::readable_varray_decay_of_numeric Input>
    [[nodiscard]] auto remap(Remap remap, Input&& input, const Shape4<i64>& shape) {
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(remap.is_xx2fx() ? shape : shape.rfft(), input.options());
        noa::fft::remap(remap, std::forward<Input>(input), output, shape);
        return output;
    }
}
