#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/signal/fft/CTF.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::algorithm::signal::fft {
    using Remap = noa::fft::Remap;

    template<typename CTF>
    struct is_valid_ctf {
        using ctf_iso32_type = noa::signal::fft::CTFIsotropic<f32>;
        using ctf_iso64_type = noa::signal::fft::CTFIsotropic<f64>;
        using ctf_aniso32_type = noa::signal::fft::CTFAnisotropic<f32>;
        using ctf_aniso64_type = noa::signal::fft::CTFAnisotropic<f64>;

        static constexpr bool IS_ISOTROPIC = noa::traits::is_any_v<
                CTF, ctf_iso32_type, ctf_iso64_type, const ctf_iso32_type*, const ctf_iso64_type*>;
        static constexpr bool IS_ANISOTROPIC = noa::traits::is_any_v<
                CTF, ctf_aniso32_type, ctf_aniso64_type, const ctf_aniso32_type*, const ctf_aniso64_type*>;

        static constexpr bool value = IS_ISOTROPIC || IS_ANISOTROPIC;
    };

    template<typename CTF>
    static constexpr bool is_valid_ctf_v = is_valid_ctf<CTF>::value;

    // Computes/applies a CTF onto a 1d/2d/3d dft.
    //  * the input/output can be both the full or both half fft. The centering can be different between the two.
    //  * if the input is complex and the output is real, the abs() of the ctf-multiplied input is saved.
    //  * the absolute and/or square of the ctf can be also computed.
    //  * if the input is empty, the ctf is simply saved in the complex or real output.
    //  * for 2d arrays, the anisotropic ctf can be used.
    template<Remap REMAP, size_t N, typename Input, typename Output, typename CTF,
             typename Index, typename Offset, typename Coord>
    class CTFOperator {
    public:
        static constexpr bool SRC_IS_HALF = static_cast<u8>(REMAP) & noa::fft::Layout::SRC_HALF;
        static constexpr bool DST_IS_HALF = static_cast<u8>(REMAP) & noa::fft::Layout::DST_HALF;
        static constexpr bool DST_IS_CENTERED = static_cast<u8>(REMAP) & noa::fft::Layout::DST_CENTERED;
        static_assert(SRC_IS_HALF == DST_IS_HALF);

        using input_type = Input;
        using output_type = Output;
        using index_type = Index;
        using offset_type = Offset;
        using coord_type = Coord;
        using ctf_type = CTF;

        static_assert(noa::traits::is_sint_v<index_type> &&
                      noa::traits::is_int_v<offset_type> &&
                      noa::traits::is_any_v<coord_type, f32, f64>);
        static_assert(noa::traits::are_same_value_type_v<input_type, output_type> &&
                      (noa::traits::are_real_or_complex_v<input_type, output_type> ||
                      (noa::traits::is_complex_v<input_type> && noa::traits::is_real_v<output_type>)));

        static constexpr bool IS_ISOTROPIC = is_valid_ctf<ctf_type>::IS_ISOTROPIC;
        static_assert(is_valid_ctf_v<ctf_type> && (N == 2 || IS_ISOTROPIC));

        using real_type = noa::traits::value_type_t<output_type>;
        using shape_nd_type = Shape<index_type, N>;
        using coord_nd_type = Vec<coord_type, N>;
        using input_accessor_type = Accessor<const input_type, N + 1, offset_type>;
        using output_accessor_type = Accessor<output_type, N + 1, offset_type>;

    public:
        CTFOperator(const input_accessor_type& input,
                     const output_accessor_type& output,
                     const shape_nd_type& shape,
                     const ctf_type& ctf,
                     bool ctf_squared,
                     bool ctf_abs) noexcept
                : m_input(input),
                  m_output(output),
                  m_ctf(ctf),
                  m_shape(shape),
                  m_norm(coord_type{1} / coord_nd_type{shape.vec()}),
                  m_ctf_squared(ctf_squared),
                  m_ctf_abs(ctf_abs) {}

    public:
        template<typename Void = void, typename = std::enable_if_t<(N == 1) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type x) const noexcept {
            // Get the output normalized frequency.
            auto frequency = static_cast<coord_type>(
                    DST_IS_HALF ? x : noa::fft::index2frequency<DST_IS_CENTERED>(x, m_shape[0]));
            frequency *= m_norm[0];

            const auto ctf = get_ctf_value_(frequency, batch);

            if (!m_input.is_empty()) {
                // Get the input index corresponding to this output index.
                const auto x_input = DST_IS_HALF ? x : noa::fft::remap_index<REMAP, true>(x, m_shape[0]);

                // Multiply the ctf with the input value.
                m_output(batch, x) = get_input_value_and_apply_ctf_(ctf, batch, x_input);
            } else {
                m_output(batch, x) = ctf;
            }
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 2) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            // Get the output normalized frequency.
            auto frequency = coord_nd_type{
                    noa::fft::index2frequency<DST_IS_CENTERED>(y, m_shape[0]),
                    DST_IS_HALF ? x : noa::fft::index2frequency<DST_IS_CENTERED>(x, m_shape[1])};
            frequency *= m_norm;

            // Get the ctf value at this frequency.
            real_type ctf;
            if constexpr (IS_ISOTROPIC)
                ctf = get_ctf_value_(noa::math::norm(frequency), batch);
            else // anisotropic
                ctf = get_ctf_value_(frequency, batch);

            if (!m_input.is_empty()) {
                // Get the input index corresponding to this output index.
                const auto y_input = noa::fft::remap_index<REMAP, true>(y, m_shape[0]);
                const auto x_input = DST_IS_HALF ? x : noa::fft::remap_index<REMAP, true>(x, m_shape[1]);

                // Multiply the ctf with the input value.
                m_output(batch, y, x) = get_input_value_and_apply_ctf_(ctf, batch, y_input, x_input);
            } else {
                m_output(batch, y, x) = ctf;
            }
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 3) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type z, index_type y, index_type x) const noexcept {
            // Get the output normalized frequency.
            auto frequency = coord_nd_type{
                    noa::fft::index2frequency<DST_IS_CENTERED>(z, m_shape[0]),
                    noa::fft::index2frequency<DST_IS_CENTERED>(y, m_shape[1]),
                    DST_IS_HALF ? x : noa::fft::index2frequency<DST_IS_CENTERED>(x, m_shape[2])};
            frequency *= m_norm;

            // Get the ctf value at this frequency.
            const auto ctf = get_ctf_value_(noa::math::norm(frequency), batch);

            if (!m_input.is_empty()) {
                // Get the input index corresponding to this output index.
                const auto z_input = noa::fft::remap_index<REMAP, true>(z, m_shape[0]);
                const auto y_input = noa::fft::remap_index<REMAP, true>(y, m_shape[1]);
                const auto x_input = DST_IS_HALF ? x : noa::fft::remap_index<REMAP, true>(x, m_shape[2]);

                // Multiply the ctf with the input value.
                m_output(batch, z, y, x) = get_input_value_and_apply_ctf_(
                        ctf, batch, z_input, y_input, x_input);
            } else {
                m_output(batch, z, y, x) = ctf;
            }
        }

    private:
        template<typename T>
        NOA_HD constexpr auto get_ctf_value_(T frequency, index_type batch) const noexcept {
            real_type ctf;
            if constexpr (std::is_pointer_v<ctf_type>) {
                ctf = static_cast<real_type>(m_ctf[batch].value_at(frequency));
            } else {
                ctf = static_cast<real_type>(m_ctf.value_at(frequency));
            }
            if (m_ctf_squared)
                ctf *= ctf;
            if (m_ctf_abs)
                ctf = noa::math::abs(ctf);
            return ctf;
        }

        template<typename Value, typename... Indexes>
        NOA_HD constexpr output_type get_input_value_and_apply_ctf_(Value ctf, Indexes... indexes) const noexcept {
            const auto value = m_input(indexes...) * static_cast<real_type>(ctf);
            if constexpr (noa::traits::is_complex_v<input_type> &&
                          noa::traits::is_real_v<output_type>)
                return noa::math::abs(value);
            else
                return value;
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        ctf_type m_ctf;
        shape_nd_type m_shape;
        coord_nd_type m_norm;
        bool m_ctf_squared;
        bool m_ctf_abs;
    };

    template<Remap REMAP, size_t N, typename Coord = f32,
             typename Input, typename Output, typename CTF, typename Index, typename Offset>
    auto ctf(const Input* input, const Strides<Offset, N + 1>& input_strides,
             Output* output, const Strides<Offset, N + 1>& output_strides,
             Shape<Index, N> shape, const CTF& ctf, bool ctf_square, bool ctf_abs) {

        const auto input_accessor = Accessor<const Input, N + 1, Offset>(input, input_strides);
        const auto output_accessor = Accessor<Output, N + 1, Offset>(output, output_strides);
        return CTFOperator<REMAP, N, Input, Output, CTF, Index, Offset, Coord>(
                input_accessor, output_accessor, shape, ctf, ctf_square, ctf_abs);
    }
}
