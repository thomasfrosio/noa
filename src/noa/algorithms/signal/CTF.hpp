#pragma once

#include "noa/algorithms/memory/Linspace.hpp"
#include "noa/core/Types.hpp"
#include "noa/core/signal/fft/CTF.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::algorithm::signal::fft {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, size_t NDIM, typename Coord, typename Index,
             typename InputAccessor, typename OutputAccessor, typename CTFAccessor>
    struct CTFOpConcept {
        static constexpr bool IS_VALID_CTF =
                nt::is_ctf_v<CTFAccessor> ||
                (nt::is_accessor_1d_restrict_contiguous_v<CTFAccessor> &&
                 std::is_const_v<nt::value_type_t<CTFAccessor>> &&
                 nt::is_ctf_v<nt::value_type_t<CTFAccessor>>);

        static constexpr bool IS_VALID_REMAP =
                REMAP == Remap::H2H || REMAP == Remap::HC2HC ||
                REMAP == Remap::HC2H || REMAP == Remap::H2HC ||
                REMAP == Remap::F2F || REMAP == Remap::FC2FC ||
                REMAP == Remap::FC2F || REMAP == Remap::F2FC;
        static constexpr bool IS_VALID_COORD_INDEX =
                nt::is_sint_v<Index> && nt::is_any_v<Coord, f32, f64>;

        using output_value_t = nt::value_type_t<OutputAccessor>;
        static constexpr bool IS_VALID_OUTPUT =
                nt::is_accessor_nd_v<OutputAccessor, NDIM + 1> &&
                !std::is_const_v<output_value_t> &&
                nt::is_real_or_complex_v<output_value_t>;

        using input_value_t = nt::value_type_t<InputAccessor>;
        static constexpr bool IS_VALID_INPUT =
                std::is_empty_v<InputAccessor> ||
                (nt::is_accessor_nd_v<InputAccessor, NDIM + 1> &&
                 std::is_const_v<input_value_t> &&
                 (nt::are_complex_v<output_value_t, input_value_t> ||
                  (nt::is_real_v<output_value_t> && nt::is_real_or_complex_v<input_value_t>)));

        static constexpr bool IS_VALID =
                IS_VALID_CTF && IS_VALID_REMAP && IS_VALID_COORD_INDEX && IS_VALID_OUTPUT && IS_VALID_INPUT;
    };

    /// Index-wise operator, to compute/apply CTFs to {1|2|3}d DFTs.
    /// \details If the input is valid, input*ctf->output is computed for the full fftfreq range.
    ///          Note that if the input is complex and the output is real, abs(input*ctf)^2->output
    ///          is computed instead. If the input is empty, ctf->output is computed, for a user-defined
    ///          frequency range.
    ///
    /// \tparam REMAP           H(C)2H(C) or F(C)2F(C).
    /// \tparam NDIM            1d, 2d, or 3d.
    /// \tparam Coord           Coordinate type. f32 or f64.
    /// \tparam Index           Index/Integer type. i32 or i64.
    /// \tparam InputAccessor   Accessor<const {f32|f64|c32|c64},NDIM+1> (+1 for the batch), or Empty.
    /// \tparam OutputAccessor  Accessor<{f32|f64|c32|c64},NDIM+1> (+1 for the batch).
    /// \tparam CTFAccessor     AccessorRestrictContiguous<const CTF,1> or CTF, where CTF is satisfies nt::is_ctf.
    template<Remap REMAP, size_t NDIM, typename Coord, typename Index,
             typename InputAccessor, typename OutputAccessor, typename CTFAccessor>
    class CTFOp {
    public:
        static_assert(CTFOpConcept<REMAP, NDIM, Coord, Index, InputAccessor, OutputAccessor, CTFAccessor>::IS_VALID);

        using input_type = InputAccessor;
        using input_or_empty_type = InputAccessor;
        using output_type = OutputAccessor;
        using ctf_type = CTFAccessor;
        using input_value_type = nt::mutable_value_type_t<InputAccessor>; // ignore const
        using output_value_type = nt::value_type_t<OutputAccessor>;
        using ctf_value_type = nt::value_type_twice_t<ctf_type>;
        using input_real_type = nt::value_type_t<input_value_type>;

        using index_type = Index;
        using coord_type = Coord;
        using coord_or_empty_type = std::conditional_t<std::is_empty_v<input_or_empty_type>, coord_type, Empty>;
        using shape_nd_type = Shape<index_type, NDIM>;
        using coord_nd_type = Vec<coord_type, NDIM>;
        using frequency_range_type = Vec2<coord_type>;

        static constexpr bool HAS_INPUT = !std::is_empty_v<input_or_empty_type>;
        static constexpr bool IS_RFFT = static_cast<u8>(REMAP) & noa::fft::Layout::DST_HALF; // or SRC_HALF
        static constexpr bool IS_DST_CENTERED = static_cast<u8>(REMAP) & noa::fft::Layout::DST_CENTERED;
        static constexpr bool IS_ISOTROPIC =
                nt::is_ctf_isotropic_v<ctf_type> ||
                (nt::is_accessor_v<ctf_type> && nt::is_ctf_isotropic_v<nt::value_type_t<ctf_type>>);

    public:
        template<typename Void = void, typename = std::enable_if_t<HAS_INPUT && std::is_void_v<Void>>>
        CTFOp(
                const input_type& input,
                const output_type& output,
                const shape_nd_type& shape,
                const ctf_type& ctf,
                bool ctf_abs,
                bool ctf_squared
        ) noexcept :
                m_ctf(ctf),
                m_input(input),
                m_output(output),
                m_shape(shape),
                m_frequency_step(coord_type{1} / coord_nd_type{shape.vec()}),
                m_ctf_abs(ctf_abs),
                m_ctf_squared(ctf_squared) {}

        template<typename Void = void, typename = std::enable_if_t<!HAS_INPUT && std::is_void_v<Void>>>
        CTFOp(const output_type& output,
              const shape_nd_type& shape,
              const ctf_type& ctf,
              bool ctf_abs,
              bool ctf_squared,
              const frequency_range_type& frequency_range,
              bool frequency_range_endpoint
        ) noexcept:
                m_ctf(ctf),
                m_output(output),
                m_shape(shape),
                m_frequency_start(frequency_range[0]),
                m_ctf_abs(ctf_abs),
                m_ctf_squared(ctf_squared) {
            // If frequency-end is negative, defaults to the highest frequency.
            // In this case, and if the frequency-start is 0, this results in the full frequency range.
            for (size_t i = 0; i < NDIM; ++i) {
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
        template<typename Void = void, typename = std::enable_if_t<(NDIM == 1) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type ox) const noexcept {
            auto fftfreq = static_cast<coord_type>(
                    IS_RFFT ? ox : noa::fft::index2frequency<IS_DST_CENTERED>(ox, m_shape[0]));
            fftfreq *= m_frequency_step[0];
            if constexpr (!HAS_INPUT)
                fftfreq += m_frequency_start;

            const auto ctf = get_ctf_value_(fftfreq, batch);

            if constexpr (HAS_INPUT) {
                // Get the input index corresponding to this output index.
                // In the mode, there's no user defined range, so a remap is enough to get the input index.
                const auto ix = IS_RFFT ? ox : noa::fft::remap_index<REMAP, true>(ox, m_shape[0]);
                m_output(batch, ox) = get_input_value_and_apply_ctf_(ctf, batch, ix);
            } else {
                m_output(batch, ox) = static_cast<output_value_type>(ctf);
            }
        }

        template<typename Void = void, typename = std::enable_if_t<(NDIM == 2) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type oy, index_type ox) const noexcept {
            auto fftfreq = coord_nd_type{
                    noa::fft::index2frequency<IS_DST_CENTERED>(oy, m_shape[0]),
                    IS_RFFT ? ox : noa::fft::index2frequency<IS_DST_CENTERED>(ox, m_shape[1])};
            fftfreq *= m_frequency_step;
            if constexpr (!HAS_INPUT)
                fftfreq += m_frequency_start;

            ctf_value_type ctf;
            if constexpr (IS_ISOTROPIC)
                ctf = get_ctf_value_(noa::math::norm(fftfreq), batch);
            else // anisotropic
                ctf = get_ctf_value_(fftfreq, batch);

            if constexpr (HAS_INPUT) {
                const auto iy = noa::fft::remap_index<REMAP, true>(oy, m_shape[0]);
                const auto ix = IS_RFFT ? ox : noa::fft::remap_index<REMAP, true>(ox, m_shape[1]);
                m_output(batch, oy, ox) = get_input_value_and_apply_ctf_(ctf, batch, iy, ix);
            } else {
                m_output(batch, oy, ox) = static_cast<output_value_type>(ctf);
            }
        }

        template<typename Void = void, typename = std::enable_if_t<(NDIM == 3) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type oz, index_type oy, index_type ox) const noexcept {
            auto fftfreq = coord_nd_type{
                    noa::fft::index2frequency<IS_DST_CENTERED>(oz, m_shape[0]),
                    noa::fft::index2frequency<IS_DST_CENTERED>(oy, m_shape[1]),
                    IS_RFFT ? ox : noa::fft::index2frequency<IS_DST_CENTERED>(ox, m_shape[2])};
            fftfreq *= m_frequency_step;
            if constexpr (!HAS_INPUT)
                fftfreq += m_frequency_start;

            const auto ctf = get_ctf_value_(noa::math::norm(fftfreq), batch);

            if constexpr (HAS_INPUT) {
                const auto iz = noa::fft::remap_index<REMAP, true>(oz, m_shape[0]);
                const auto iy = noa::fft::remap_index<REMAP, true>(oy, m_shape[1]);
                const auto ix = IS_RFFT ? ox : noa::fft::remap_index<REMAP, true>(ox, m_shape[2]);
                m_output(batch, oz, oy, ox) = get_input_value_and_apply_ctf_(ctf, batch, iz, iy, ix);
            } else {
                m_output(batch, oz, oy, ox) = static_cast<output_value_type>(ctf);
            }
        }

    private:
        template<typename T>
        NOA_HD constexpr auto get_ctf_value_(T frequency, index_type batch) const noexcept {
            ctf_value_type ctf;
            if constexpr (nt::is_accessor_1d_restrict_contiguous_v<ctf_type>) {
                ctf = m_ctf[batch].value_at(frequency);
            } else {
                ctf = m_ctf.value_at(frequency);
            }
            if (m_ctf_abs)
                ctf = noa::math::abs(ctf);
            if (m_ctf_squared)
                ctf *= ctf;
            return ctf;
        }

        template<typename... Indexes>
        NOA_HD constexpr auto get_input_value_and_apply_ctf_(ctf_value_type ctf, Indexes... indexes) const noexcept {
            const auto value = m_input(indexes...) * static_cast<input_real_type>(ctf);
            if constexpr (nt::is_complex_v<input_value_type> && nt::is_real_v<output_value_type>)
                return static_cast<output_value_type>(noa::abs_squared_t{}(value));
            else
                return static_cast<output_value_type>(value);
        }

    private:
        ctf_type m_ctf;
        NOA_NO_UNIQUE_ADDRESS input_or_empty_type m_input{};
        output_type m_output;
        shape_nd_type m_shape;
        coord_nd_type m_frequency_step;
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_frequency_start{};
        bool m_ctf_abs;
        bool m_ctf_squared;
    };

    template<Remap REMAP, size_t NDIM, typename Coord = f32,
             typename Input, typename Output, typename CTF, typename Index, typename Offset>
    auto ctf(const Input* input, const Strides<Offset, NDIM + 1>& input_strides,
             Output* output, const Strides<Offset, NDIM + 1>& output_strides,
             Shape<Index, NDIM> shape, const CTF& ctf, bool ctf_abs, bool ctf_square) {

        using input_accessor_t = Accessor<const Input, NDIM + 1, Offset>;
        using output_accessor_t = Accessor<Output, NDIM + 1, Offset>;

        if constexpr (std::is_pointer_v<CTF>) {
            using ctf_value_t = nt::remove_ref_cv_t<nt::remove_pointer_cv_t<CTF>>;
            using ctf_accessor_t = AccessorRestrictContiguous<const ctf_value_t, 1, Offset>;
            return CTFOp<REMAP, NDIM, Coord, Index, input_accessor_t, output_accessor_t, ctf_accessor_t> (
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    shape, ctf_accessor_t(ctf), ctf_abs, ctf_square);
        } else {
            return CTFOp<REMAP, NDIM, Coord, Index, input_accessor_t, output_accessor_t, CTF> (
                    input_accessor_t(input, input_strides),
                    output_accessor_t(output, output_strides),
                    shape, ctf, ctf_abs, ctf_square);
        }
    }

    template<Remap REMAP, size_t NDIM, typename Coord = f32,
             typename Output, typename CTF, typename Index, typename Offset>
    auto ctf(Output* output, const Strides<Offset, NDIM + 1>& output_strides,
             Shape<Index, NDIM> shape, const CTF& ctf, bool ctf_abs, bool ctf_square,
             const Vec2<Coord>& fftfreq_range, bool fftfreq_range_endpoint) {

        using output_accessor_t = Accessor<Output, NDIM + 1, Offset>;
        if constexpr (std::is_pointer_v<CTF>) {
            using ctf_value_t = nt::remove_ref_cv_t<nt::remove_pointer_cv_t<CTF>>;
            using ctf_accessor_t = AccessorRestrictContiguous<const ctf_value_t, 1, Offset>;
            return CTFOp<REMAP, NDIM, Coord, Index, Empty, output_accessor_t, ctf_accessor_t>(
                    output_accessor_t(output, output_strides),
                    shape, ctf_accessor_t(ctf), ctf_abs, ctf_square,
                    fftfreq_range, fftfreq_range_endpoint);
        } else {
            return CTFOp < REMAP, NDIM, Coord, Index, Empty, output_accessor_t, CTF>(
                    output_accessor_t(output, output_strides),
                    shape, ctf, ctf_abs, ctf_square,
                    fftfreq_range, fftfreq_range_endpoint);
        }
    }
}
