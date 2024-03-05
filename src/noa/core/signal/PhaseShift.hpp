#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::signal {
    /// Computes the phase shift at a given normalized-frequency.
    template<typename Complex, typename Coord, size_t N,
            nt::enable_if_bool_t<nt::is_complex_v<Complex> && (N == 2 || N == 3)> = true>
    [[nodiscard]] NOA_FHD Complex phase_shift(
            const Vec<Coord, N>& shift,
            const Vec<Coord, N>& fftfreq
    ) {
        using real_t = typename Complex::value_type;
        const auto factor = static_cast<real_t>(-dot(2 * Constant<Coord>::PI * shift, fftfreq));
        Complex phase_shift;
        sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    /// 4d iwise operator to phase shift each dimension by size / 2 (floating-point division).
    template<noa::fft::Remap REMAP, typename Index, typename InputAccessor, typename OutputAccessor>
    requires (nt::are_accessor_pure_nd_v<4, InputAccessor, OutputAccessor>)
    class PhaseShiftHalf {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_SRC_CENTERED = u8_REMAP & Layout::SRC_CENTERED;
        static_assert(u8_REMAP & Layout::SRC_HALF and u8_REMAP & Layout::DST_HALF);

        using index_type = Index;
        using index3_type = Vec3<index_type>;
        using shape2_type = Shape2<index_type>;
        using shape4_type = Shape4<index_type>;
        using input_accessor_type = InputAccessor;
        using output_accessor_type = OutputAccessor;
        using input_value_type = input_accessor_type::mutable_value_type;
        using output_value_type = output_accessor_type::value_type;
        using input_real_type = nt::value_type_t<input_value_type>;

    public:
        PhaseShiftHalf(
                const input_accessor_type& input,
                const output_accessor_type& output,
                const shape4_type& shape
        ) : m_input(input), m_output(output), m_bh_shape(shape.filter(1, 2)) {}

        NOA_HD constexpr void operator()(index_type ii, index_type ij, index_type ik, index_type il) const noexcept {
            using namespace noa::fft;
            const auto freq = index3_type{
                    index2frequency<IS_SRC_CENTERED>(ij, m_bh_shape[0]),
                    index2frequency<IS_SRC_CENTERED>(ik, m_bh_shape[1]),
                    il};
            const auto phase_shift = static_cast<input_real_type>(product(1 - 2 * abs(freq % 2)));

            const auto oj = remap_index<REMAP>(ij, m_bh_shape[0]);
            const auto ok = remap_index<REMAP>(ik, m_bh_shape[1]);
            const auto value = m_input ? m_input(ii, ij, ik, il) * phase_shift : phase_shift;
            m_output(ii, oj, ok, il) = static_cast<output_value_type>(value);
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        shape2_type m_bh_shape;
    };

    /// 3d or 4d iwise operator to phase shift 2d or 3d array(s).
    template<noa::fft::Remap REMAP, size_t NDIM,
             typename Index, typename Shift,
             typename InputAccessor, typename OutputAccessor>
    requires ((NDIM == 2 or NDIM == 3) and nt::are_accessor_pure_nd_v<NDIM + 1, InputAccessor, OutputAccessor>)
    class PhaseShift {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_SRC_CENTERED = u8_REMAP & Layout::SRC_CENTERED;
        static_assert(u8_REMAP & Layout::SRC_HALF && u8_REMAP & Layout::DST_HALF);

        using index_type = Index;
        using shape_nd_type = Shape<index_type, NDIM>;
        using shape_type = Shape<index_type, NDIM - 1>;

        using shift_ = Shift;
        using shift_single_type = std::remove_const_t<std::remove_pointer_t<shift_>>;
        using shift_pointer_type = const shift_single_type*;
        static constexpr bool is_pointer = std::is_pointer_v<Shift>;
        using shift_type = std::conditional_t<is_pointer, shift_pointer_type, shift_single_type>;
        using coord_type = nt::value_type_t<shift_single_type>;
        using vec_nd_type = Vec<coord_type, NDIM>;
        static_assert(std::is_same_v<shift_single_type, Vec<coord_type, NDIM>>);
        static_assert(nt::is_any_v<coord_type, f32, f64>);

        using input_accessor_type = InputAccessor;
        using output_accessor_type = OutputAccessor;
        using input_value_type = input_accessor_type::mutable_value_type;
        using output_value_type = output_accessor_type::value_type;

    public:
        PhaseShift(
                const input_accessor_type& input,
                const output_accessor_type& output,
                const shape_nd_type& shape,
                const shift_type& shift,
                coord_type cutoff
        ) : m_input(input), m_output(output),
            m_norm(coord_type{1} / vec_nd_type::from_vec(shape.vec)),
            m_shape(shape.pop_back()),
            m_shift(shift),
            m_cutoff_fftfreq_sqd(cutoff * cutoff) {}

        NOA_HD constexpr void operator()(index_type ii, index_type ik, index_type il) const requires (NDIM == 2) {
            const auto frequency = vec_nd_type::from_values(
                    noa::fft::index2frequency<IS_SRC_CENTERED>(ik, m_shape[0]), il);

            const vec_nd_type fftfreq = frequency * m_norm;
            const input_value_type phase_shift =
                    dot(fftfreq, fftfreq) <= m_cutoff_fftfreq_sqd ?
                    phase_shift_(ii, fftfreq) : input_value_type{1, 0};

            const auto ok = noa::fft::remap_index<REMAP>(ik, m_shape[0]);
            const auto value = m_input ? m_input(ii, ik, il) * phase_shift : phase_shift;
            m_output(ii, ok, il) = static_cast<output_value_type>(value);
        }

        NOA_HD constexpr void operator()(index_type ii, index_type ij, index_type ik, index_type il) const requires (NDIM == 3) {
            const auto frequency = vec_nd_type::from_values(
                    noa::fft::index2frequency<IS_SRC_CENTERED>(ij, m_shape[0]),
                    noa::fft::index2frequency<IS_SRC_CENTERED>(ik, m_shape[1]),
                    il);

            const vec_nd_type fftfreq = frequency * m_norm;
            const input_value_type phase_shift =
                    dot(fftfreq, fftfreq) <= m_cutoff_fftfreq_sqd ?
                    phase_shift_(ii, fftfreq) : input_value_type{1, 0};

            // FIXME If even, the real nyquist should stay real, so add the conjugate pair?

            const auto oj = noa::fft::remap_index<REMAP>(ij, m_shape[0]);
            const auto ok = noa::fft::remap_index<REMAP>(ik, m_shape[1]);
            const auto value = m_input ? m_input(ii, ij, ik, il) * phase_shift : phase_shift;
            m_output(ii, oj, ok, il) = static_cast<output_value_type>(value);
        }

    private:
        NOA_HD constexpr input_value_type phase_shift_(index_type batch, const vec_nd_type& fftfreq) const noexcept {
            if constexpr (std::is_pointer_v<shift_type>)
                return phase_shift<input_value_type>(m_shift[batch], fftfreq);
            else
                return phase_shift<input_value_type>(m_shift, fftfreq);
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        vec_nd_type m_norm;
        shape_type m_shape;
        shift_type m_shift;
        coord_type m_cutoff_fftfreq_sqd;
    };
}
