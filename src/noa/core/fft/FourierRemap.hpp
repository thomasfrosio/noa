#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/Enums.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::fft {
    template<Remap REMAP, typename Index, typename Accessor>
    requires (nt::is_accessor_pure_nd<Accessor, 4>::value and nt::is_sint<Index>::value)
    class FourierRemapInplace {
    public:
        using index_type = Index;
        using dhw_shape_type = Shape3<index_type>;
        using accessor_type = Accessor;
        using value_type = accessor_type::mutable_value_type;
        static_assert(nt::is_real_or_complex_v<value_type>);

        FourierRemapInplace(
                const accessor_type& array,
                const Shape3<index_type>& shape
        ) : m_array(array),
            m_shape(shape) {}

        NOA_HD constexpr void operator()(index_type oi, index_type oj, index_type ok, index_type ol) const noexcept {
            index_type ij;
            index_type ik;
            if constexpr (REMAP == Remap::H2HC) {
                ij = ifftshift(oj, m_shape[0]);
                ik = ifftshift(ok, m_shape[1]);
            } else if constexpr (REMAP == Remap::HC2H) {
                ij = fftshift(oj, m_shape[0]);
                ik = fftshift(ok, m_shape[1]);
            } else {
                static_assert(nt::always_false_v<index_type>);
            }

            // Swap
            auto tmp = m_array(oi, oj, ok, ol);
            m_array(oi, oj, ok, ol) = m_array(oi, ij, ik, ol);
            m_array(oi, ij, ik, ol) = tmp;
        }

    private:
        accessor_type m_array;
        dhw_shape_type m_shape;
    };

    template<Remap REMAP, typename Index, typename InputAccessor, typename OutputAccessor>
    requires (nt::are_accessor_pure_nd<4, InputAccessor, OutputAccessor>::value and nt::is_sint<Index>::value)
    class FourierRemap {
    public:
        using index_type = Index;
        using dhw_shape_type = Shape3<index_type>;
        using input_accessor_type = InputAccessor;
        using output_accessor_type = OutputAccessor;
        using input_value_type = input_accessor_type::mutable_value_type;
        using output_value_type = output_accessor_type::value_type;

        static_assert(nt::are_complex_v<input_value_type, output_value_type> or
                      nt::are_real_v<input_value_type, output_value_type> or
                      (nt::is_complex_v<input_value_type> and nt::is_real_v<output_value_type>));

        FourierRemap(
                const input_accessor_type& input,
                const output_accessor_type& output,
                const Shape3<index_type>& shape
        ) : m_input(input),
            m_output(output),
            m_shape(shape) {}

        NOA_HD constexpr void operator()(index_type oi, index_type oj, index_type ok, index_type ol) const noexcept {
            if constexpr (REMAP == Remap::HC2H) {
                const auto ij = fftshift(oj, m_shape[0]);
                const auto ik = fftshift(ok, m_shape[1]);
                m_output(oi, oj, ok, ol) = to_output_(m_input(oi, ij, ik, ol));

            } else if constexpr (REMAP == Remap::H2HC) {
                const auto ij = ifftshift(oj, m_shape[0]);
                const auto ik = ifftshift(ok, m_shape[1]);
                m_output(oi, oj, ok, ol) = to_output_(m_input(oi, ij, ik, ol));

            } else if constexpr (REMAP == Remap::FC2F) {
                const auto ij = fftshift(oj, m_shape[0]);
                const auto ik = fftshift(ok, m_shape[1]);
                const auto il = fftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = to_output_(m_input(oi, ij, ik, il));

            } else if constexpr (REMAP == Remap::F2FC) {
                const auto ij = ifftshift(oj, m_shape[0]);
                const auto ik = ifftshift(ok, m_shape[1]);
                const auto il = ifftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = to_output_(m_input(oi, ij, ik, il));

            } else if constexpr (REMAP == Remap::F2H) {
                m_output(oi, oj, ok, ol) = to_output_(m_input(oi, oj, ok, ol)); // copy

            } else if constexpr (REMAP == Remap::F2HC) {
                const auto ij = ifftshift(oj, m_shape[0]);
                const auto ik = ifftshift(ok, m_shape[1]);
                m_output(oi, oj, ok, ol) = to_output_(m_input(oi, ij, ik, ol));

            } else if constexpr (REMAP == Remap::FC2H) {
                const auto ij = fftshift(oj, m_shape[0]);
                const auto ik = fftshift(ok, m_shape[1]);
                const auto il = fftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = to_output_(m_input(oi, ij, ik, il));

            } else if constexpr (REMAP == Remap::FC2HC) {
                const auto il = fftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = to_output_(m_input(oi, oj, ok, il));

            } else if constexpr (REMAP == Remap::HC2F || REMAP == Remap::HC2FC) {
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
                    if constexpr (nt::is_complex_v<output_value_type>)
                        value = conj(value);
                }

                if constexpr (REMAP == Remap::HC2F) {
                    m_output(oi, oj, ok, ol) = to_output_(value);
                } else { // HC2FC: HC2F -> F2FC
                    const auto ooj = fftshift(oj, m_shape[0]);
                    const auto ook = fftshift(ok, m_shape[1]);
                    const auto ool = fftshift(ol, m_shape[2]);
                    m_output(oi, ooj, ook, ool) = to_output_(value);
                }

            } else if constexpr (REMAP == Remap::H2F || REMAP == Remap::H2FC) {
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
                    if constexpr (nt::is_complex_v<output_value_type>)
                        value = conj(value);
                }

                if constexpr (REMAP == Remap::H2F) {
                    m_output(oi, oj, ok, ol) = to_output_(value);
                } else { // H2FC: H2F -> F2FC
                    const auto ooj = fftshift(oj, m_shape[0]);
                    const auto ook = fftshift(ok, m_shape[1]);
                    const auto ool = fftshift(ol, m_shape[2]);
                    m_output(oi, ooj, ook, ool) = to_output_(value);
                }
            } else {
                static_assert(nt::always_false_v<input_value_type>);
            }
        }

    private:
        NOA_HD constexpr output_value_type to_output_(const input_value_type& value) {
            if constexpr (nt::is_complex_v<input_value_type> and nt::is_real_v<output_value_type>)
                return static_cast<output_value_type>(abs_squared(value));
            else
                return static_cast<output_value_type>(value);
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        dhw_shape_type m_shape;
    };
}
