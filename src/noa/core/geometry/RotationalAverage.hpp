#pragma once

#include "noa/core/fft/Frequency.hpp"
#include "noa/core/fft/RemapInterface.hpp"
#include "noa/core/signal/CTF.hpp"
#include "noa/core/utils/Atomic.hpp"
#include "noa/core/Types.hpp"
#include "noa/core/Linspace.hpp"

namespace noa::geometry {
    /// 3d or 4d iwise operator to compute a rotational average of 2d or 3d array(s).
    /// * The output layout is noted "H", since often the number of output shells is min(shape) // 2 + 1
    ///   Otherwise, the input can be any of the for layouts (H, HC, F or FC).
    /// * A lerp is used to add frequencies in its two neighbor shells, instead of rounding to the nearest shell.
    /// * The frequencies are normalized, so input dimensions don't have to be equal.
    /// * The user sets the number of output shells, as well as the output frequency range.
    /// * If input is complex and output real, the input is preprocessed to abs(input)^2.
    template<noa::fft::RemapInterface Remap,
             size_t N, typename Coord, typename Index,
             typename InputAccessor,    // Accessor<f32|f64|c32|c64, N+1>
             typename OutputAccessor,   // Accessor<f32|f64|c32|c64, 2>
             typename WeightAccessor,   // Accessor<f32|f64, 2>
             typename CTFAccessor>      // Accessor<CTF, 1>, AccessorValue<CTF> or Empty
    requires (nt::is_sint_v<Index> and nt::is_real_v<Coord> and
              nt::is_accessor_nd_v<InputAccessor, N + 1> and
              nt::are_accessor_nd_v<2, OutputAccessor, WeightAccessor> and
              ((N == 2 and nt::is_accessor_nd_v<CTFAccessor, 1>) or std::is_empty_v<CTFAccessor>))
    class RotationalAverage {
    public:
        static_assert(Remap.is_xx2h());
        static constexpr bool is_centered = Remap.is_xc2xx();
        static constexpr bool is_rfft = Remap.is_hc2xx();

        using index_type = Index;
        using coord_type = Coord;
        using shape_type = Shape<index_type, N - is_rfft>;
        using coord_nd_type = Vec<coord_type, N>;
        using coord2_type = Vec2<coord_type>;
        using shape_nd_type = Shape<index_type, N>;

        using input_accessor_type = InputAccessor;
        using output_accessor_type = OutputAccessor;
        using weight_accessor_type = WeightAccessor;
        using input_type = nt::mutable_value_type_t<input_accessor_type>;
        using output_type = nt::value_type_t<output_accessor_type>;
        using weight_type = nt::value_type_t<weight_accessor_type>;
        static_assert(nt::are_complex_v<input_type, output_type> or
                      nt::are_real_v<input_type, output_type> or
                      (nt::is_complex_v<input_type> and nt::is_real_v<output_type>));
        static_assert(nt::is_real_v<weight_type>);

        using ctf_type = nt::mutable_value_type_t<CTFAccessor>;
        static_assert(nt::is_ctf_anisotropic_v<ctf_type>);

    public:
        RotationalAverage(
                const input_accessor_type& input,
                const shape_nd_type& input_shape,
                const ctf_type& input_ctf,
                const output_accessor_type& output,
                const weight_accessor_type& weight,
                index_type n_shells,
                coord2_type frequency_range,
                bool frequency_range_endpoint
        ) : m_input(input), m_output(output), m_weight(weight), m_ctf(input_ctf),
            m_max_shell_index(n_shells - 1),
            m_fftfreq_step(coord_type{1} / coord_nd_type::from_vec(input_shape.vec)) {

            // Transform to inclusive range so that we only have to deal with one case.
            if (not frequency_range_endpoint) {
                auto step = Linspace<coord_type, index_type>::from_range(
                        frequency_range[0], frequency_range[1], n_shells, false).step;
                frequency_range[1] -= step;
            }
            if constexpr (std::is_empty_v<ctf_type>)
                m_frequency_range_sqd = frequency_range * frequency_range;
            else
                m_frequency_range_sqd = frequency_range;
            m_frequency_range_start = frequency_range[0];
            m_frequency_range_span = frequency_range[1] - frequency_range[0];

            if constexpr (is_rfft)
                m_shape = input_shape.pop_back();
            else
                m_shape = input_shape;
        }

        // Batched 2d.
        NOA_HD void operator()(index_type batch, index_type iy, index_type ix) const noexcept requires (N == 2) {
            const auto frequency = coord_nd_type::from_values(
                    noa::fft::index2frequency<is_centered>(iy, m_shape[0]),
                    is_rfft ? ix : noa::fft::index2frequency<is_centered>(ix, m_shape[1]));
            const auto fftfreq_2d = frequency * m_fftfreq_step;

            coord_type fftfreq;
            if constexpr (std::is_empty_v<ctf_type>) {
                fftfreq = dot(fftfreq_2d, fftfreq_2d);
            } else {
                // Correct for anisotropic pixel size and defocus.
                fftfreq = static_cast<coord_type>(m_ctf[batch].isotropic_fftfreq(fftfreq_2d));
            }

            if (fftfreq < m_frequency_range_sqd[0] or
                fftfreq > m_frequency_range_sqd[1])
                return;

            if constexpr (std::is_empty_v<ctf_type>)
                fftfreq = sqrt(fftfreq);

            const auto output = input_to_output_(m_input(batch, iy, ix));
            add_to_output_(batch, fftfreq, output);
        }

        // Batched 3d.
        NOA_HD void operator()(index_type batch, index_type iz, index_type iy, index_type ix) const noexcept requires (N == 3) {
            auto frequency = coord_nd_type::from_values(
                    noa::fft::index2frequency<is_centered>(iz, m_shape[0]),
                    noa::fft::index2frequency<is_centered>(iy, m_shape[1]),
                    is_rfft ? ix : noa::fft::index2frequency<is_centered>(ix, m_shape[2]));
            frequency *= m_fftfreq_step;

            const auto fftfreq_sqd = dot(frequency, frequency);
            if (fftfreq_sqd < m_frequency_range_sqd[0] or
                fftfreq_sqd > m_frequency_range_sqd[1])
                return;

            const auto output = input_to_output_(m_input(batch, iz, iy, ix));
            add_to_output_(batch, sqrt(fftfreq_sqd), output);
        }

    private:
        NOA_HD static output_type input_to_output_(input_type input) noexcept {
            if constexpr (nt::is_complex_v<input_type> and nt::is_real_v<output_type>) {
                return static_cast<output_type>(abs_squared(input));
            } else {
                return static_cast<output_type>(input);
            }
        }

        NOA_HD void add_to_output_(index_type batch, coord_type fftfreq, output_type value) const noexcept {
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
            using output_real_t = nt::value_type_t<output_type>;
            ng::atomic_add(m_output, value * static_cast<output_real_t>(fraction_low), batch, shell_low);
            ng::atomic_add(m_output, value * static_cast<output_real_t>(fraction_high), batch, shell_high);
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

        shape_type m_shape; // width is removed
        index_type m_max_shell_index;
        coord_nd_type m_fftfreq_step;
        coord2_type m_frequency_range_sqd;
        coord_type m_frequency_range_start;
        coord_type m_frequency_range_span;
    };
}
