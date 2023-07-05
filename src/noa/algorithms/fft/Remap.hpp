#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/fft/Enums.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/traits/Utilities.hpp"

namespace noa::algorithm::fft {
    // FFT remapping.
    template<noa::fft::Remap REMAP, typename Value, typename Index, typename Offset>
    class Remap {
    public:
        using value_type = Value;
        using index_type = Index;
        using offset_type = Offset;
        using dhw_shape_type = Shape3<index_type>;
        using input_accessor_type = AccessorRestrict<const value_type, 4, offset_type>;
        using output_accessor_type = AccessorRestrict<value_type, 4, offset_type>;

        Remap(const input_accessor_type& input,
              const output_accessor_type& output,
              const Shape3<index_type>& shape)
                : m_input(input),
                  m_output(output),
                  m_shape(shape) {
            NOA_ASSERT(input.get() != output.get());
        }

        NOA_HD constexpr void operator()(index_type oi, index_type oj, index_type ok, index_type ol) const noexcept {
            if constexpr (REMAP == noa::fft::HC2H) {
                const auto ij = noa::fft::fftshift(oj, m_shape[0]);
                const auto ik = noa::fft::fftshift(ok, m_shape[1]);
                m_output(oi, oj, ok, ol) = m_input(oi, ij, ik, ol);

            } else if constexpr (REMAP == noa::fft::H2HC) {
                const auto ij = noa::fft::ifftshift(oj, m_shape[0]);
                const auto ik = noa::fft::ifftshift(ok, m_shape[1]);
                m_output(oi, oj, ok, ol) = m_input(oi, ij, ik, ol);

            } else if constexpr (REMAP == noa::fft::FC2F) {
                const auto ij = noa::fft::fftshift(oj, m_shape[0]);
                const auto ik = noa::fft::fftshift(ok, m_shape[1]);
                const auto il = noa::fft::fftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = m_input(oi, ij, ik, il);

            } else if constexpr (REMAP == noa::fft::F2FC) {
                const auto ij = noa::fft::ifftshift(oj, m_shape[0]);
                const auto ik = noa::fft::ifftshift(ok, m_shape[1]);
                const auto il = noa::fft::ifftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = m_input(oi, ij, ik, il);

            } else if constexpr (REMAP == noa::fft::F2H) {
                m_output(oi, oj, ok, ol) = m_input(oi, oj, ok, ol); // copy

            } else if constexpr (REMAP == noa::fft::F2HC) {
                const auto ij = noa::fft::ifftshift(oj, m_shape[0]);
                const auto ik = noa::fft::ifftshift(ok, m_shape[1]);
                m_output(oi, oj, ok, ol) = m_input(oi, ij, ik, ol);

            } else if constexpr (REMAP == noa::fft::FC2H) {
                const auto ij = noa::fft::fftshift(oj, m_shape[0]);
                const auto ik = noa::fft::fftshift(ok, m_shape[1]);
                const auto il = noa::fft::fftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = m_input(oi, ij, ik, il);

            } else if constexpr (REMAP == noa::fft::FC2HC) {
                const auto il = noa::fft::fftshift(ol, m_shape[2]);
                m_output(oi, oj, ok, ol) = m_input(oi, oj, ok, il);

            } else if constexpr (REMAP == noa::fft::HC2F || REMAP == noa::fft::HC2FC) {
                value_type value;
                if (ol < m_shape[2] / 2 + 1) {
                    // Copy first non-redundant half.
                    const auto ij = noa::fft::fftshift(oj, m_shape[0]);
                    const auto ik = noa::fft::fftshift(ok, m_shape[1]);
                    value = m_input(oi, ij, ik, ol);
                } else {
                    // Rebase to the symmetric row in the non-redundant input.
                    // Then copy in reverse order.
                    const auto ij = noa::fft::fftshift(oj != 0 ? m_shape[0] - oj : oj, m_shape[0]);
                    const auto ik = noa::fft::fftshift(ok != 0 ? m_shape[1] - ok : ok, m_shape[1]);
                    value = m_input(oi, ij, ik, m_shape[2] - ol);
                    if constexpr (noa::traits::is_complex_v<value_type>)
                        value = noa::math::conj(value);
                }

                if constexpr (REMAP == noa::fft::HC2F) {
                    m_output(oi, oj, ok, ol) = value;
                } else { // HC2FC: HC2F -> F2FC
                    const auto ooj = noa::fft::fftshift(oj, m_shape[0]);
                    const auto ook = noa::fft::fftshift(ok, m_shape[1]);
                    const auto ool = noa::fft::fftshift(ol, m_shape[2]);
                    m_output(oi, ooj, ook, ool) = value;
                }

            } else if constexpr (REMAP == noa::fft::H2F || REMAP == noa::fft::H2FC) {
                value_type value;
                if (ol < m_shape[2] / 2 + 1) {
                    // Copy first non-redundant half.
                    value = m_input(oi, oj, ok, ol);
                } else {
                    // Rebase to the symmetric row in the non-redundant input.
                    // Then copy in reverse order.
                    const auto ij = oj != 0 ? m_shape[0] - oj : oj;
                    const auto ik = ok != 0 ? m_shape[1] - ok : ok;
                    value = m_input(oi, ij, ik, m_shape[2] - ol);
                    if constexpr (noa::traits::is_complex_v<value_type>)
                        value = noa::math::conj(value);
                }

                if constexpr (REMAP == noa::fft::H2F) {
                    m_output(oi, oj, ok, ol) = value;
                } else { // H2FC: H2F -> F2FC
                    const auto ooj = noa::fft::fftshift(oj, m_shape[0]);
                    const auto ook = noa::fft::fftshift(ok, m_shape[1]);
                    const auto ool = noa::fft::fftshift(ol, m_shape[2]);
                    m_output(oi, ooj, ook, ool) = value;
                }
            } else {
                static_assert(noa::traits::always_false_v<value_type>);
            }
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        dhw_shape_type m_shape;
    };

    template<noa::fft::Remap REMAP, typename Value, typename Index, typename Offset>
    auto remap(const Value* input, const Strides4<Offset>& input_strides,
               Value* output, const Strides4<Offset>& output_strides,
               const Shape4<Index>& shape) {
        const auto input_accessor = AccessorRestrict<const Value, 4, Offset>(input, input_strides);
        const auto output_accessor = AccessorRestrict<Value, 4, Offset>(output, output_strides);

        const auto kernel = Remap<REMAP, Value, Index, Offset>(input_accessor, output_accessor, shape.pop_front());
        const auto iwise_shape =
                noa::traits::to_underlying(REMAP) & noa::traits::to_underlying(noa::fft::Layout::DST_HALF) ?
                shape.rfft() : shape;

        return std::pair{kernel, iwise_shape};
    }
}
