#pragma once

#include "noa/algorithms/memory/Linspace.hpp"
#include "noa/core/Types.hpp"
#include "noa/core/signal/fft/CTF.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::algorithm::signal::fft {
    using Remap = noa::fft::Remap;
    namespace nt = noa::traits;

    template<typename CTF>
    struct is_valid_ctf {
        using ctf_iso32_type = noa::signal::fft::CTFIsotropic<f32>;
        using ctf_iso64_type = noa::signal::fft::CTFIsotropic<f64>;
        using ctf_aniso32_type = noa::signal::fft::CTFAnisotropic<f32>;
        using ctf_aniso64_type = noa::signal::fft::CTFAnisotropic<f64>;

        static constexpr bool IS_ISOTROPIC = nt::is_any_v<
                CTF, ctf_iso32_type, ctf_iso64_type, const ctf_iso32_type*, const ctf_iso64_type*>;
        static constexpr bool IS_ANISOTROPIC = nt::is_any_v<
                CTF, ctf_aniso32_type, ctf_aniso64_type, const ctf_aniso32_type*, const ctf_aniso64_type*>;

        static constexpr bool value = IS_ISOTROPIC || IS_ANISOTROPIC;
    };

    template<typename CTF>
    static constexpr bool is_valid_ctf_v = is_valid_ctf<CTF>::value;
    template<typename CTF>
    static constexpr bool is_valid_iso_ctf_v = is_valid_ctf<CTF>::IS_ISOTROPIC;
    template<typename CTF>
    static constexpr bool is_valid_aniso_ctf_v = is_valid_ctf<CTF>::IS_ANISOTROPIC;

    enum class CTFOperatorMode {
        // Input->Output. The frequency range is fixed to the full fft/rfft (like it is usually).
        INPUT_OUTPUT,

        // Output only. The frequency range of the output is defined at runtime.
        OUTPUT_RANGE
    };

    // Computes/applies a CTF onto a 1d/2d/3d dft.
    //  * the input/output can be both the full or both half fft. The centering can be different between the two.
    //  * if the input is valid and complex and the output is real, the input is preprocessed to abs(input)^2.
    //  * the absolute and/or square of the ctf can be also computed.
    //  * for 2d arrays, the anisotropic ctf can be used.
    template<CTFOperatorMode MODE, Remap REMAP, size_t N,
             typename Input, typename Output, typename CTF,
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

        static_assert(nt::is_sint_v<index_type> &&
                      nt::is_int_v<offset_type> &&
                      nt::is_any_v<coord_type, f32, f64>);
        static_assert(nt::are_same_value_type_v<input_type, output_type> &&
                      (nt::are_real_or_complex_v<input_type, output_type> ||
                      (nt::is_complex_v<input_type> && nt::is_real_v<output_type>)));

        static constexpr bool IS_INPUT_OUTPUT = MODE == CTFOperatorMode::INPUT_OUTPUT;
        static constexpr bool IS_ISOTROPIC = is_valid_ctf<ctf_type>::IS_ISOTROPIC;
        static_assert(is_valid_ctf_v<ctf_type> && (N == 2 || IS_ISOTROPIC));

        using real_type = nt::value_type_t<output_type>;
        using shape_nd_type = Shape<index_type, N>;
        using coord_nd_type = Vec<coord_type, N>;
        using frequency_range_type = Vec2<coord_type>;
        using input_accessor_type = Accessor<const input_type, N + 1, offset_type>;
        using output_accessor_type = Accessor<output_type, N + 1, offset_type>;
        using input_accessor_type_or_empty = std::conditional_t<IS_INPUT_OUTPUT, input_accessor_type, nt::Empty>;
        using coord_type_or_empty = std::conditional_t<IS_INPUT_OUTPUT, nt::Empty, coord_type>;

    public:
        template<typename Void = void, typename = std::enable_if_t<IS_INPUT_OUTPUT && std::is_void_v<Void>>>
        CTFOperator(const input_accessor_type& input,
                    const output_accessor_type& output,
                    const shape_nd_type& shape,
                    const ctf_type& ctf,
                    bool ctf_abs,
                    bool ctf_squared) noexcept
                : m_ctf(ctf),
                  m_input(input),
                  m_output(output),
                  m_shape(shape),
                  m_frequency_step(coord_type{1} / coord_nd_type{shape.vec()}),
                  m_ctf_abs(ctf_abs),
                  m_ctf_squared(ctf_squared) {}

        template<typename Void = void, typename = std::enable_if_t<!IS_INPUT_OUTPUT && std::is_void_v<Void>>>
        CTFOperator(const output_accessor_type& output,
                    const shape_nd_type& shape,
                    const ctf_type& ctf,
                    bool ctf_abs,
                    bool ctf_squared,
                    const frequency_range_type& frequency_range,
                    bool frequency_range_endpoint) noexcept
                : m_ctf(ctf),
                  m_output(output),
                  m_shape(shape),
                  m_frequency_start(frequency_range[0]),
                  m_ctf_abs(ctf_abs),
                  m_ctf_squared(ctf_squared) {
            // If frequency-end is negative, defaults to the highest frequency.
            // If the frequency-start is 0, this results in no rescaling and is identical
            // to the frequency range in the IS_INPUT_OUTPUT mode.
            for (size_t i = 0; i < N; ++i) {
                const auto max_sample_size = shape[i] / 2 + 1;
                const auto frequency_end =
                        frequency_range[1] < 0 ?
                        noa::fft::highest_normalized_frequency<coord_type>(shape[i]) :
                        frequency_range[1];
                m_frequency_step[i] = noa::algorithm::memory::linspace_step(
                        max_sample_size, frequency_range[0], frequency_end, frequency_range_endpoint);
            }
        }

    public:
        template<typename Void = void, typename = std::enable_if_t<(N == 1) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type ox) const noexcept {
            // Get the output normalized frequency.
            auto frequency = static_cast<coord_type>(
                    DST_IS_HALF ? ox : noa::fft::index2frequency<DST_IS_CENTERED>(ox, m_shape[0]));
            frequency *= m_frequency_step[0];
            if constexpr (!std::is_empty_v<coord_type_or_empty>)
                frequency += m_frequency_start;

            const auto ctf = get_ctf_value_(frequency, batch);

            if constexpr (IS_INPUT_OUTPUT) {
                // Get the input index corresponding to this output index.
                // In the mode, there's no user defined range, so a remap is enough to get the input index.
                const auto ix = DST_IS_HALF ? ox : noa::fft::remap_index<REMAP, true>(ox, m_shape[0]);

                // Multiply the ctf with the input value.
                m_output(batch, ox) = get_input_value_and_apply_ctf_(ctf, batch, ix);
            } else {
                m_output(batch, ox) = ctf;
            }
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 2) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type oy, index_type ox) const noexcept {
            // Get the output normalized frequency.
            auto frequency = coord_nd_type{
                    noa::fft::index2frequency<DST_IS_CENTERED>(oy, m_shape[0]),
                    DST_IS_HALF ? ox : noa::fft::index2frequency<DST_IS_CENTERED>(ox, m_shape[1])};
            frequency *= m_frequency_step;
            if constexpr (!std::is_empty_v<coord_type_or_empty>)
                frequency += m_frequency_start;

            // Get the ctf value at this frequency.
            real_type ctf;
            if constexpr (IS_ISOTROPIC)
                ctf = get_ctf_value_(noa::math::norm(frequency), batch);
            else // anisotropic
                ctf = get_ctf_value_(frequency, batch);

            if constexpr (IS_INPUT_OUTPUT) {
                const auto iy = noa::fft::remap_index<REMAP, true>(oy, m_shape[0]);
                const auto ix = DST_IS_HALF ? ox : noa::fft::remap_index<REMAP, true>(ox, m_shape[1]);
                m_output(batch, oy, ox) = get_input_value_and_apply_ctf_(ctf, batch, iy, ix);
            } else {
                m_output(batch, oy, ox) = ctf;
            }
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 3) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type oz, index_type oy, index_type ox) const noexcept {
            // Get the output normalized frequency.
            auto frequency = coord_nd_type{
                    noa::fft::index2frequency<DST_IS_CENTERED>(oz, m_shape[0]),
                    noa::fft::index2frequency<DST_IS_CENTERED>(oy, m_shape[1]),
                    DST_IS_HALF ? ox : noa::fft::index2frequency<DST_IS_CENTERED>(ox, m_shape[2])};
            frequency *= m_frequency_step;
            if constexpr (!std::is_empty_v<coord_type_or_empty>)
                frequency += m_frequency_start;

            // Get the ctf value at this frequency.
            const auto ctf = get_ctf_value_(noa::math::norm(frequency), batch);

            if constexpr (IS_INPUT_OUTPUT) {
                const auto iz = noa::fft::remap_index<REMAP, true>(oz, m_shape[0]);
                const auto iy = noa::fft::remap_index<REMAP, true>(oy, m_shape[1]);
                const auto ix = DST_IS_HALF ? ox : noa::fft::remap_index<REMAP, true>(ox, m_shape[2]);
                m_output(batch, oz, oy, ox) = get_input_value_and_apply_ctf_(ctf, batch, iz, iy, ix);
            } else {
                m_output(batch, oz, oy, ox) = ctf;
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
            if (m_ctf_abs)
                ctf = noa::math::abs(ctf);
            if (m_ctf_squared)
                ctf *= ctf;
            return ctf;
        }

        template<typename Value, typename... Indexes>
        NOA_HD constexpr output_type get_input_value_and_apply_ctf_(Value ctf, Indexes... indexes) const noexcept {
            const auto value = m_input(indexes...) * static_cast<real_type>(ctf);
            if constexpr (nt::is_complex_v<input_type> &&
                          nt::is_real_v<output_type>)
                return noa::abs_squared_t{}(value);
            else
                return value;
        }

    private:
        ctf_type m_ctf;
        NOA_NO_UNIQUE_ADDRESS input_accessor_type_or_empty m_input{};
        output_accessor_type m_output;
        shape_nd_type m_shape;
        coord_nd_type m_frequency_step;
        NOA_NO_UNIQUE_ADDRESS coord_type_or_empty m_frequency_start{};
        bool m_ctf_abs;
        bool m_ctf_squared;
    };

    template<Remap REMAP, size_t N, typename Coord = f32,
             typename Input, typename Output, typename CTF, typename Index, typename Offset>
    auto ctf(const Input* input, const Strides<Offset, N + 1>& input_strides,
             Output* output, const Strides<Offset, N + 1>& output_strides,
             Shape<Index, N> shape, const CTF& ctf, bool ctf_abs, bool ctf_square) {

        const auto input_accessor = Accessor<const Input, N + 1, Offset>(input, input_strides);
        const auto output_accessor = Accessor<Output, N + 1, Offset>(output, output_strides);
        return CTFOperator<CTFOperatorMode::INPUT_OUTPUT, REMAP, N, Input, Output, CTF, Index, Offset, Coord>(
                input_accessor, output_accessor, shape, ctf, ctf_abs, ctf_square);
    }

    template<Remap REMAP, size_t N, typename Coord = f32,
             typename Output, typename CTF, typename Index, typename Offset>
    auto ctf(Output* output, const Strides<Offset, N + 1>& output_strides,
             Shape<Index, N> shape, const CTF& ctf, bool ctf_abs, bool ctf_square,
             const Vec2<Coord>& fftfreq_range, bool fftfreq_range_endpoint) {

        const auto output_accessor = Accessor<Output, N + 1, Offset>(output, output_strides);
        return CTFOperator<CTFOperatorMode::OUTPUT_RANGE, REMAP, N, Output, Output, CTF, Index, Offset, Coord>(
                output_accessor, shape, ctf, ctf_abs, ctf_square, fftfreq_range, fftfreq_range_endpoint);
    }
}
