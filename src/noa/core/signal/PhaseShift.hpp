#pragma once

#include "noa/core/types/Shape.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::signal::guts {
    /// 4d iwise operator to phase shift each dimension by size / 2 (floating-point division).
    template<Remap REMAP, size_t N,
             nt::sinteger Index,
             nt::readable_nd_or_empty<N + 1> Input,
             nt::writable_nd<N + 1> Output>
    requires (N == 2 or N == 3)
    class PhaseShiftHalf {
    public:
        static constexpr bool IS_SRC_CENTERED = REMAP.is_xc2xx();
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();
        static_assert(REMAP.is_hx2hx() or REMAP.is_fx2fx());

        using index_type = Index;
        using shape_nd_type = Shape<index_type, N>;
        using shape_type = Shape<index_type, N - IS_RFFT>;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::complex<output_value_type, input_value_type>);

    public:
        constexpr PhaseShiftHalf(
            const input_type& input,
            const output_type& output,
            const shape_nd_type& shape
        ) :
            m_input(input),
            m_output(output),
            m_shape(shape.template pop_back<IS_RFFT>()) {}

        template<nt::same_as<index_type>... I> requires (sizeof...(I) == N)
        constexpr void operator()(index_type batch, I... indices) const {
            const auto frequency = noa::fft::index2frequency<IS_SRC_CENTERED, IS_RFFT>(Vec{indices...}, m_shape);
            const auto phase_shift = static_cast<input_real_type>(product(1 - 2 * abs(frequency % 2)));

            const auto output_indices = noa::fft::remap_indices<REMAP>(Vec{indices...}, m_shape);
            auto& output = m_output(output_indices.push_front(batch));

            if (m_input)
                output = static_cast<output_value_type>(m_input(batch, indices...) * phase_shift);
            else
                output = static_cast<output_value_type>(phase_shift);
        }

    private:
        input_type m_input;
        output_type m_output;
        shape_type m_shape;
    };

    /// 3d or 4d iwise operator to phase shift 2d or 3d array(s).
    template<Remap REMAP, size_t N,
             nt::sinteger Index,
             nt::batched_parameter Shift,
             nt::readable_nd_optional<N + 1> Input,
             nt::writable_nd<N + 1> Output>
    requires (N == 2 or N == 3)
    class PhaseShift {
    public:
        static constexpr bool IS_SRC_CENTERED = REMAP.is_xc2xx();
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();
        static_assert(REMAP.is_hx2hx() or REMAP.is_fx2fx());

        using index_type = Index;
        using shape_nd_type = Shape<index_type, N>;
        using shape_type = Shape<index_type, N - IS_RFFT>;

        using shift_parameter_type = Shift;
        using vec_nd_type = nt::value_type_t<shift_parameter_type>;
        using coord_type = nt::value_type_t<vec_nd_type>;
        static_assert(nt::vec_real_size<vec_nd_type, N>);

        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::complex<output_value_type, input_value_type>);

    public:
        constexpr PhaseShift(
            const input_type& input,
            const output_type& output,
            const shape_nd_type& shape,
            const shift_parameter_type& shift,
            coord_type cutoff
        ) :
            m_input(input), m_output(output),
            m_norm(coord_type{1} / vec_nd_type::from_vec(shape.vec)),
            m_shape(shape.template pop_back<IS_RFFT>()),
            m_shift(shift),
            m_cutoff_fftfreq_sqd(cutoff * cutoff) {}

        template<nt::same_as<index_type>... I> requires (sizeof...(I) == N)
        constexpr void operator()(index_type batch, I... indices) const {
            const auto frequency = noa::fft::index2frequency<IS_SRC_CENTERED, IS_RFFT>(Vec{indices...}, m_shape);
            const auto fftfreq = vec_nd_type::from_vec(frequency) * m_norm;

            input_value_type phase_shift{1, 0};
            if (dot(fftfreq, fftfreq) <= m_cutoff_fftfreq_sqd)
                phase_shift = noa::fft::phase_shift<input_value_type>(m_shift[batch], fftfreq);
            // TODO If even, the real nyquist should stay real, so add the conjugate pair?

            const auto output_indices = noa::fft::remap_indices<REMAP>(Vec{indices...}, m_shape);
            auto& output = m_output(output_indices.push_front(batch));
            if (m_input)
                output = static_cast<output_value_type>(m_input(batch, indices...) * phase_shift);
            else
                output = static_cast<output_value_type>(phase_shift);
        }

    private:
        input_type m_input;
        output_type m_output;
        vec_nd_type m_norm;
        shape_type m_shape;
        shift_parameter_type m_shift;
        coord_type m_cutoff_fftfreq_sqd;
    };
}
