#pragma once

#include "noa/core/Iwise.hpp"
#include "noa/core/Remap.hpp"
#include "noa/core/Types.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/signal/CTF.hpp"
#include "noa/core/utils/Atomic.hpp"

namespace noa::geometry {
    /// 3d or 4d iwise operator to compute a rotational average of 2d or 3d array(s).
    /// * The output layout is noted "H", since often the number of output shells is min(shape) // 2 + 1
    ///   Otherwise, the input can be any of the for layouts (H, HC, F or FC).
    /// * A lerp is used to add frequencies in its two neighbor shells, instead of rounding to the nearest shell.
    /// * The frequencies are normalized, so input dimensions don't have to be equal.
    /// * The user sets the number of output shells, as well as the output frequency range.
    /// * If input is complex and output real, the input is preprocessed to abs(input)^2.
    template<Remap REMAP,
             size_t N,
             nt::real Coord,
             nt::sinteger Index,
             nt::readable_nd<N + 1> InputAccessor,
             nt::atomic_addable_nd<2> OutputAccessor,
             nt::atomic_addable_nd_optional<2> WeightAccessor,
             nt::batched_parameter CTFAccessor>
    class RotationalAverage {
    public:
        static_assert(REMAP.is_xx2h());
        static constexpr bool IS_CENTERED = REMAP.is_xc2xx();
        static constexpr bool IS_RFFT = REMAP.is_hc2xx();

        using index_type = Index;
        using coord_type = Coord;
        using shape_type = Shape<index_type, N - IS_RFFT>;
        using coord_nd_type = Vec<coord_type, N>;
        using coord2_type = Vec2<coord_type>;
        using shape_nd_type = Shape<index_type, N>;

        using input_accessor_type = InputAccessor;
        using output_accessor_type = OutputAccessor;
        using weight_accessor_type = WeightAccessor;
        using input_type = nt::mutable_value_type_t<input_accessor_type>;
        using output_value_type = nt::value_type_t<output_accessor_type>;
        using output_real_type = nt::value_type_t<output_value_type>;
        using weight_type = nt::value_type_t<weight_accessor_type>;
        static_assert(nt::are_power_spectrum_value_types_v<input_type, output_value_type>);
        static_assert(nt::real<weight_type>);

        using ctf_type = nt::mutable_value_type_t<CTFAccessor>;
        static_assert(nt::empty<ctf_type> or (N == 2 and nt::ctf_anisotropic<ctf_type>));

    public:
        constexpr RotationalAverage(
                const input_accessor_type& input,
                const shape_nd_type& input_shape,
                const ctf_type& input_ctf,
                const output_accessor_type& output,
                const weight_accessor_type& weight,
                index_type n_shells,
                coord2_type frequency_range,
                bool frequency_range_endpoint
        ) :
                m_input(input), m_output(output), m_weight(weight), m_ctf(input_ctf),
                m_max_shell_index(n_shells - 1),
                m_fftfreq_step(coord_type{1} / coord_nd_type::from_vec(input_shape.vec))
        {

            // Transform to inclusive range so that we only have to deal with one case.
            if (not frequency_range_endpoint) {
                auto step = Linspace{frequency_range[0], frequency_range[1], false}.for_size(n_shells).step;
                frequency_range[1] -= step;
            }
            if constexpr (nt::empty<ctf_type>)
                m_frequency_range_sqd = frequency_range * frequency_range;
            else
                m_frequency_range_sqd = frequency_range;
            m_frequency_range_start = frequency_range[0];
            m_frequency_range_span = frequency_range[1] - frequency_range[0];

            if constexpr (IS_RFFT)
                m_shape = input_shape.pop_back();
            else
                m_shape = input_shape;
        }

        template<nt::same_as<index_type>... I>
        NOA_HD void operator()(index_type batch, I... indices) const noexcept requires (N == 2) {
            const auto frequency = noa::fft::index2frequency<IS_CENTERED, IS_RFFT>(Vec{indices...}, m_shape);
            const auto fftfreq_nd = coord_nd_type::from_vec(frequency) * m_fftfreq_step;

            coord_type fftfreq;
            if constexpr (nt::empty<ctf_type>) {
                fftfreq = dot(fftfreq_nd, fftfreq_nd);
            } else {
                // Correct for anisotropic pixel size and defocus.
                fftfreq = static_cast<coord_type>(m_ctf[batch].isotropic_fftfreq(fftfreq_nd));
            }

            if (fftfreq < m_frequency_range_sqd[0] or
                fftfreq > m_frequency_range_sqd[1])
                return;

            if constexpr (nt::empty<ctf_type>)
                fftfreq = sqrt(fftfreq);

            // Scale the normalized frequency back to the corresponding output shell.
            const coord_type scaled_fftfreq = (fftfreq - m_frequency_range_start) / m_frequency_range_span;
            const coord_type radius = scaled_fftfreq * static_cast<coord_type>(m_max_shell_index);
            const coord_type radius_floor = floor(radius);

            // Since by this point fftfreq has to be within the output frequency range,
            // "radius" is guaranteed to be within [0, m_max_shell_index].
            NOA_ASSERT(radius >= 0 and radius <= static_cast<coord_type>(m_max_shell_index));

            // Compute lerp weights.
            const index_type shell_low = static_cast<index_type>(radius_floor);
            const index_type shell_high = min(m_max_shell_index, shell_low + 1); // shell_low can be the last index
            const coord_type fraction_high = radius - radius_floor;
            const coord_type fraction_low = 1 - fraction_high;

            // TODO In CUDA, we could do the atomic reduction in shared memory to reduce global memory transfers?
            const auto value = cast_or_abs_squared<output_value_type>(m_input(batch, indices...));
            ng::atomic_add(m_output, value * static_cast<output_real_type>(fraction_low), batch, shell_low);
            ng::atomic_add(m_output, value * static_cast<output_real_type>(fraction_high), batch, shell_high);
            if (m_weight) {
                ng::atomic_add(m_weight, static_cast<weight_type>(fraction_low), batch, shell_low);
                ng::atomic_add(m_weight, static_cast<weight_type>(fraction_high), batch, shell_high);
            }
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        weight_accessor_type m_weight;
        NOA_NO_UNIQUE_ADDRESS ctf_type m_ctf;

        shape_type m_shape;
        index_type m_max_shell_index;
        coord_nd_type m_fftfreq_step;
        coord2_type m_frequency_range_sqd;
        coord_type m_frequency_range_start;
        coord_type m_frequency_range_span;
    };
}
