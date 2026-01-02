#pragma once

#include "noa/runtime/core/Traits.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/Iwise.hpp"

#include "noa/fft/core/Frequency.hpp"
#include "noa/fft/core/Layout.hpp"

namespace noa::fft::details {
    template<Layout REMAP,
             nt::sinteger Index,
             nt::writable_nd<4> Output>
    requires nt::numeric<nt::value_type_t<Output>>
    class FourierRemapInplace {
    public:
        using index_type = Index;
        using dhw_shape_type = Shape<index_type, 3>;
        using dh_shape_type = Shape<index_type, 2>;
        using output_type = Output;
        using value_type = nt::mutable_value_type_t<output_type>;

        constexpr FourierRemapInplace(
            const output_type& array,
            const dhw_shape_type& shape
        ) : m_array(array),
            m_shape(shape.pop_back()) {}

        constexpr void operator()(index_type oi, index_type oj, index_type ok, index_type ol) const {
            index_type ij;
            index_type ik;
            if constexpr (REMAP == Layout::H2HC) {
                ij = ifftshift(oj, m_shape[0]);
                ik = ifftshift(ok, m_shape[1]);
            } else if constexpr (REMAP == Layout::HC2H) {
                ij = fftshift(oj, m_shape[0]);
                ik = fftshift(ok, m_shape[1]);
            } else {
                static_assert(nt::always_false<output_type>);
            }

            auto& o = m_array(oi, oj, ok, ol);
            auto& i = m_array(oi, ij, ik, ol);

            // Swap
            auto tmp = o;
            o = i;
            i = tmp;
        }

    private:
        output_type m_array;
        dh_shape_type m_shape;
    };

    template<Layout REMAP,
             nt::sinteger Index,
             nt::readable_nd<4> Input,
             nt::writable_nd<4> Output>
    class FourierRemap {
    public:
        using index_type = Index;
        using dhw_shape_type = Shape<index_type, 3>;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::compatible_or_spectrum_types<input_value_type, output_value_type>);

        constexpr FourierRemap(
            const input_type& input,
            const output_type& output,
            const dhw_shape_type& shape
        ) : m_input(input),
            m_output(output),
            m_shape(shape) {}

        constexpr void operator()(index_type oi, index_type oj, index_type ok, index_type ol) const {
            if constexpr (REMAP == Layout::HC2H) {
                const auto ij = fftshift(oj, m_shape[0]);
                const auto ik = fftshift(ok, m_shape[1]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, ij, ik, ol));

            } else if constexpr (REMAP == Layout::H2HC) {
                const auto ij = ifftshift(oj, m_shape[0]);
                const auto ik = ifftshift(ok, m_shape[1]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, ij, ik, ol));

            } else if constexpr (REMAP == Layout::FC2F) {
                const auto ij = fftshift(oj, m_shape[0]);
                const auto ik = fftshift(ok, m_shape[1]);
                const auto il = fftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, ij, ik, il));

            } else if constexpr (REMAP == Layout::F2FC) {
                const auto ij = ifftshift(oj, m_shape[0]);
                const auto ik = ifftshift(ok, m_shape[1]);
                const auto il = ifftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, ij, ik, il));

            } else if constexpr (REMAP == Layout::F2H) {
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, oj, ok, ol)); // copy

            } else if constexpr (REMAP == Layout::F2HC) {
                const auto ij = ifftshift(oj, m_shape[0]);
                const auto ik = ifftshift(ok, m_shape[1]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, ij, ik, ol));

            } else if constexpr (REMAP == Layout::FC2H) {
                const auto ij = fftshift(oj, m_shape[0]);
                const auto ik = fftshift(ok, m_shape[1]);
                const auto il = fftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, ij, ik, il));

            } else if constexpr (REMAP == Layout::FC2HC) {
                const auto il = fftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, oj, ok, il));

            } else if constexpr (REMAP == Layout::HC2F or REMAP == Layout::HC2FC) {
                input_value_type value;
                if (ol < m_shape[2] / 2 + 1) {
                    // Copy first non-redundant half.
                    const auto ij = fftshift(oj, m_shape[0]);
                    const auto ik = fftshift(ok, m_shape[1]);
                    value = m_input(oi, ij, ik, ol);
                } else {
                    // Rebase to the symmetric row in the non-redundant input.
                    // Then copy in reverse order.
                    const auto ij = fftshift(oj != 0 ? m_shape[0] - oj : oj, m_shape[0]);
                    const auto ik = fftshift(ok != 0 ? m_shape[1] - ok : ok, m_shape[1]);
                    value = m_input(oi, ij, ik, m_shape[2] - ol);
                    if constexpr (nt::complex<output_value_type>)
                        value = conj(value);
                }

                if constexpr (REMAP == Layout::HC2F) {
                    m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(value);
                } else { // HC2FC: HC2F -> F2FC
                    const auto ooj = fftshift(oj, m_shape[0]);
                    const auto ook = fftshift(ok, m_shape[1]);
                    const auto ool = fftshift(ol, m_shape[2]);
                    m_output(oi, ooj, ook, ool) = cast_or_abs_squared<output_value_type>(value);
                }

            } else if constexpr (REMAP == Layout::H2F or REMAP == Layout::H2FC) {
                input_value_type value;
                if (ol < m_shape[2] / 2 + 1) {
                    // Copy first non-redundant half.
                    value = m_input(oi, oj, ok, ol);
                } else {
                    // Rebase to the symmetric row in the non-redundant input.
                    // Then copy in reverse order.
                    const auto ij = oj != 0 ? m_shape[0] - oj : oj;
                    const auto ik = ok != 0 ? m_shape[1] - ok : ok;
                    value = m_input(oi, ij, ik, m_shape[2] - ol);
                    if constexpr (nt::complex<output_value_type>)
                        value = conj(value);
                }

                if constexpr (REMAP == Layout::H2F) {
                    m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(value);
                } else { // H2FC: H2F -> F2FC
                    const auto ooj = fftshift(oj, m_shape[0]);
                    const auto ook = fftshift(ok, m_shape[1]);
                    const auto ool = fftshift(ol, m_shape[2]);
                    m_output(oi, ooj, ook, ool) = cast_or_abs_squared<output_value_type>(value);
                }
            } else {
                static_assert(nt::always_false<input_type>);
            }
        }

    private:
        input_type m_input;
        output_type m_output;
        dhw_shape_type m_shape;
    };
}

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
    /// \bug See noa/docs/040_fft_layouts.md.
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
    void remap(Layout remap, Input&& input, Output&& output, Shape4 shape) {
        using input_t = nt::mutable_value_type_t<Input>;
        using output_t = nt::value_type_t<Output>;

        // Check shape.
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(input.shape() == (remap.is_fx2xx() ? shape : shape.rfft()),
              "Given the {} remap, the input fft is expected to have a physical shape of {}, but got {}",
              remap, remap.is_fx2xx() ? shape : shape.rfft(), input.shape());
        check(output.shape() == (remap.is_xx2fx() ? shape : shape.rfft()),
              "Given the {} remap, the output fft is expected to have a physical shape of {}, but got {}",
              remap, remap.is_xx2fx() ? shape : shape.rfft(), output.shape());

        // Shortcut if there's no remapping.
        const bool is_inplace = are_overlapped(input, output);
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
            check(remap.is_any(Layout::H2HC, Layout::HC2H),
                  "In-place remapping is not supported with {}", remap);
            check(static_cast<const void*>(input.get()) == static_cast<const void*>(output.get()) and
                  input.strides() == output.strides(),
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
        if (remap.is_any(Layout::FC2F, Layout::F2FC)) {
            const auto order_3d = output_strides.pop_front().rightmost_order(shape.pop_front());
            if (order_3d != Vec<isize, 3>{0, 1, 2}) {
                const auto order = (order_3d + 1).push_front(0);
                nd::permute_all(order, input_strides, output_strides, shape);
            }
        }

        using input_accessor_t = AccessorRestrict<const input_t, 4, isize>;
        using output_accessor_t = AccessorRestrict<output_t, 4, isize>;
        const auto input_accessor = input_accessor_t(input.get(), input_strides);
        const auto output_accessor = output_accessor_t(output.get(), output_strides);
        const auto shape_3d = shape.pop_front();

        auto iwise_shape = remap.is_xx2fx() ? shape : shape.rfft();
        if (is_inplace)
            iwise_shape[2] = max(iwise_shape[2] / 2, isize{1}); // iterate only through half of height

        switch (remap) {
            case Layout::H2H:
            case Layout::HC2HC:
            case Layout::F2F:
            case Layout::FC2FC:
                break;
            case Layout::H2HC: {
                if (is_inplace) {
                    auto op = details::FourierRemapInplace<Layout::H2HC, isize, output_accessor_t>(output_accessor, shape_3d);
                    return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
                }
                auto op = details::FourierRemap<Layout::H2HC, isize, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::HC2H: {
                if (is_inplace) {
                    auto op = details::FourierRemapInplace<Layout::HC2H, isize, output_accessor_t>(output_accessor, shape_3d);
                    return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
                }
                auto op = details::FourierRemap<Layout::HC2H, isize, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::H2F: {
                auto op = details::FourierRemap<Layout::H2F, isize, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::F2H: {
                auto op = details::FourierRemap<Layout::F2H, isize, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::F2FC: {
                auto op = details::FourierRemap<Layout::F2FC, isize, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::FC2F: {
                auto op = details::FourierRemap<Layout::FC2F, isize, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::HC2F: {
                auto op = details::FourierRemap<Layout::HC2F, isize, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::F2HC: {
                auto op = details::FourierRemap<Layout::F2HC, isize, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::FC2H: {
                auto op = details::FourierRemap<Layout::FC2H, isize, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::FC2HC: {
                auto op = details::FourierRemap<Layout::FC2HC, isize, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::HC2FC: {
                auto op = details::FourierRemap<Layout::HC2FC, isize, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::H2FC: {
                auto op = details::FourierRemap<Layout::H2FC, isize, input_accessor_t, output_accessor_t>(
                    input_accessor, output_accessor, shape_3d);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
        }
    }

    /// Remaps fft(s).
    template<nt::readable_varray_decay_of_numeric Input>
    [[nodiscard]] auto remap(Layout remap, Input&& input, const Shape4& shape) {
        using value_t = nt::mutable_value_type_t<Input>;
        auto output = Array<value_t>(remap.is_xx2fx() ? shape : shape.rfft(), input.options());
        nf::remap(remap, std::forward<Input>(input), output, shape);
        return output;
    }
}
