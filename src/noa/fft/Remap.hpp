#pragma once

#include "noa/runtime/core/Traits.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/Iwise.hpp"

#include "noa/fft/core/Frequency.hpp"
#include "noa/fft/core/Transform.hpp"
#include "noa/fft/core/Layout.hpp"

// TODO Add compile-time rank (B+RANK dimensions), to not have to collapse
//      batch dimensions and keep problem to RANK-D instead of 3D.

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
            m_shape_dh(shape.pop_back()) {}

        constexpr void operator()(index_type oi, index_type oj, index_type ok, index_type ol) const {
            index_type ij;
            index_type ik;
            if constexpr (REMAP == Layout::H2HC) {
                ij = ifftshift(oj, m_shape_dh[0]);
                ik = ifftshift(ok, m_shape_dh[1]);
            } else if constexpr (REMAP == Layout::HC2H) {
                ij = fftshift(oj, m_shape_dh[0]);
                ik = fftshift(ok, m_shape_dh[1]);
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
        dh_shape_type m_shape_dh;
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
    struct RemapOptions {
        /// Rank of the transform.
        /// This determines which dimensions are considered batch dimensions.
        /// Batch dimensions are collapsed to a single dimensions and should therefore be collapsible.
        /// See Shape::rank_checked and ranked_shape for more details.
        i32 rank{-1};
    };

    /// Remaps FFT(s).
    /// \param remap:
    ///     Remapping operation.
    /// \param[in] input:
    ///     Input FFT to remap.
    ///     The rank of the transform, therefore which dimensions are batch dimensions, depends on options.rank.
    /// \param[out] output:
    ///     Remapped FFT.
    ///     The rank of the transform, therefore which dimensions are batch dimensions, depends on options.rank.
    /// \param shape:
    ///     Logical shape.
    /// \param options:
    ///     Remap options.
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
    template<nt::readable_array_decay Input, nt::writable_array_decay Output, usize N>
        requires (nt::array_decay_with_compatible_or_spectrum_types<Input, Output> and
                  nt::array_decay_nd<Input, N> and
                  nt::array_decay_nd<Output, N>)
    void remap(Layout remap, Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        using input_value_t = nt::mutable_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;

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
            if constexpr (nt::same_as<input_value_t, output_value_t>) {
                if (not is_inplace)
                    copy(std::forward<Input>(input), std::forward<Output>(output));
            } else {
                cast(std::forward<Input>(input), std::forward<Output>(output));
            }
            return;
        }

        // Transform to B(N=DHW).
        constexpr usize B = 1; // collapse batch dimensions to 1, or add empty batch for N <= 3
        constexpr usize BN = B + 3; // scale 1D and 2D to 3D.
        auto input_bn = input.span().template as_nd<BN>();
        auto output_bn = output.span().template as_nd<BN>();
        auto shape_bn = shape.template as_nd<BN>();
        i32 rank = shape.rank_checked(options.rank);
        details::prepare_ranked_spans(input_bn, output_bn, shape_bn, rank);

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
            check((shape_bn[N - 2] == 1 or is_even(shape_bn[N - 2])) and
                  (shape_bn[N - 1] == 1 or is_even(shape_bn[N - 1])),
                  "In-place remapping requires the depth and height dimensions to have an even number of elements, "
                  "but got shape_nd={}", shape_bn);
        }

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        using input_t = AccessorRestrict<const input_value_t, BN, isize>;
        using output_t = AccessorRestrict<output_value_t, BN, isize>;
        const auto iaccessor = input_t(input_bn.get(), input_bn.strides());
        const auto oaccessor = output_t(output_bn.get(), output_bn.strides());
        const auto shape_n = shape_bn.template pop_front<B>();

        auto iwise_shape = remap.is_xx2fx() ? shape_bn : shape_bn.rfft();
        if (is_inplace)
            iwise_shape[N - 2] = max(iwise_shape[N - 2] / 2, isize{1}); // iterate only through half of height

        switch (remap) {
            using namespace details;
            case Layout::H2H:
            case Layout::HC2HC:
            case Layout::F2F:
            case Layout::FC2FC:
                break;
            case Layout::H2HC: {
                if (is_inplace) {
                    auto op = FourierRemapInplace<Layout::H2HC, isize, output_t>(oaccessor, shape_n);
                    return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
                }
                auto op = FourierRemap<Layout::H2HC, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::HC2H: {
                if (is_inplace) {
                    auto op = FourierRemapInplace<Layout::HC2H, isize, output_t>(oaccessor, shape_n);
                    return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
                }
                auto op = FourierRemap<Layout::HC2H, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::H2F: {
                auto op = FourierRemap<Layout::H2F, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::F2H: {
                auto op = FourierRemap<Layout::F2H, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::F2FC: {
                auto op = FourierRemap<Layout::F2FC, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::FC2F: {
                auto op = FourierRemap<Layout::FC2F, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::HC2F: {
                auto op = FourierRemap<Layout::HC2F, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::F2HC: {
                auto op = FourierRemap<Layout::F2HC, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::FC2H: {
                auto op = FourierRemap<Layout::FC2H, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::FC2HC: {
                auto op = FourierRemap<Layout::FC2HC, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::HC2FC: {
                auto op = FourierRemap<Layout::HC2FC, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::H2FC: {
                auto op = FourierRemap<Layout::H2FC, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
        }
    }

    /// Remaps fft(s).
    template<nt::readable_array_decay_of_numeric Input, usize N>
        requires nt::array_decay_nd<Input, N>
    [[nodiscard]] auto remap(Layout remap, Input&& input, const Shape<isize, N>& shape, RemapOptions options = {}) {
        using value_t = nt::mutable_value_type_t<Input>;
        auto output = Array<value_t, N>(remap.is_xx2fx() ? shape : shape.rfft(), input.options());
        nf::remap(remap, std::forward<Input>(input), output, shape, options);
        return output;
    }
}
