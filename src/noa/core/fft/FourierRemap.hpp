#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa::fft::guts {
    template<Remap REMAP,
             nt::sinteger Index,
             nt::writable_nd<4> Output>
    requires nt::numeric<nt::value_type_t<Output>>
    class FourierRemapInplace {
    public:
        using index_type = Index;
        using dhw_shape_type = Shape3<index_type>;
        using dh_shape_type = Shape2<index_type>;
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
            if constexpr (REMAP == Remap::H2HC) {
                ij = ifftshift(oj, m_shape[0]);
                ik = ifftshift(ok, m_shape[1]);
            } else if constexpr (REMAP == Remap::HC2H) {
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

    template<Remap REMAP,
             nt::sinteger Index,
             nt::readable_nd<4> Input,
             nt::writable_nd<4> Output>
    class FourierRemap {
    public:
        using index_type = Index;
        using dhw_shape_type = Shape3<index_type>;
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
            if constexpr (REMAP == Remap::HC2H) {
                const auto ij = fftshift(oj, m_shape[0]);
                const auto ik = fftshift(ok, m_shape[1]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, ij, ik, ol));

            } else if constexpr (REMAP == Remap::H2HC) {
                const auto ij = ifftshift(oj, m_shape[0]);
                const auto ik = ifftshift(ok, m_shape[1]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, ij, ik, ol));

            } else if constexpr (REMAP == Remap::FC2F) {
                const auto ij = fftshift(oj, m_shape[0]);
                const auto ik = fftshift(ok, m_shape[1]);
                const auto il = fftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, ij, ik, il));

            } else if constexpr (REMAP == Remap::F2FC) {
                const auto ij = ifftshift(oj, m_shape[0]);
                const auto ik = ifftshift(ok, m_shape[1]);
                const auto il = ifftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, ij, ik, il));

            } else if constexpr (REMAP == Remap::F2H) {
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, oj, ok, ol)); // copy

            } else if constexpr (REMAP == Remap::F2HC) {
                const auto ij = ifftshift(oj, m_shape[0]);
                const auto ik = ifftshift(ok, m_shape[1]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, ij, ik, ol));

            } else if constexpr (REMAP == Remap::FC2H) {
                const auto ij = fftshift(oj, m_shape[0]);
                const auto ik = fftshift(ok, m_shape[1]);
                const auto il = fftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, ij, ik, il));

            } else if constexpr (REMAP == Remap::FC2HC) {
                const auto il = fftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(oi, oj, ok, il));

            } else if constexpr (REMAP == Remap::HC2F or REMAP == Remap::HC2FC) {
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

                if constexpr (REMAP == Remap::HC2F) {
                    m_output(oi, oj, ok, ol) = cast_or_abs_squared<output_value_type>(value);
                } else { // HC2FC: HC2F -> F2FC
                    const auto ooj = fftshift(oj, m_shape[0]);
                    const auto ook = fftshift(ok, m_shape[1]);
                    const auto ool = fftshift(ol, m_shape[2]);
                    m_output(oi, ooj, ook, ool) = cast_or_abs_squared<output_value_type>(value);
                }

            } else if constexpr (REMAP == Remap::H2F or REMAP == Remap::H2FC) {
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

                if constexpr (REMAP == Remap::H2F) {
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
