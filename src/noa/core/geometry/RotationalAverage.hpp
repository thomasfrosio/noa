#pragma once

#include "noa/core/fft/Frequency.hpp"
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
    template<noa::fft::Remap REMAP, size_t N,
             typename Coord, typename Index, typename Offset,
             typename Input, typename Output, typename CTF = Empty>
    class RotationalAverage {
    public:
        static_assert(REMAP == noa::fft::H2H || REMAP == noa::fft::HC2H ||
                      REMAP == noa::fft::F2H || REMAP == noa::fft::FC2H);
        static_assert(nt::is_sint_v<Index>);
        static_assert(nt::is_int_v<Offset>);
        static_assert(nt::is_real_v<Coord>);
        static_assert((nt::are_all_same_v<Input, Output> and
                       nt::are_real_or_complex_v<Input, Output>) ||
                      (nt::is_complex_v<Input> and
                       nt::is_real_v<Output>));
        static_assert((N == 2 and
                       (nt::is_ctf_anisotropic_v<CTF> ||
                        (std::is_pointer_v<CTF> and nt::is_ctf_anisotropic_v<nt::remove_pointer_cv_t<CTF>>))) ||
                      std::is_empty_v<CTF>);

        static constexpr bool IS_CENTERED = static_cast<u8>(REMAP) & noa::fft::Layout::SRC_CENTERED;
        static constexpr bool IS_HALF = static_cast<u8>(REMAP) & noa::fft::Layout::SRC_HALF;

        using index_type = Index;
        using offset_type = Offset;
        using coord_type = Coord;
        using input_type = Input;
        using output_type = Output;
        using real_type = nt::value_type_t<output_type>;
        using ctf_type = CTF;

        using shape_type = Shape<index_type, N - IS_HALF>;
        using coord_nd_type = Vec<coord_type, N>;
        using coord2_type = Vec2<coord_type>;
        using shape_nd_type = Shape<index_type, N>;
        using input_accessor_type = AccessorRestrict<const input_type, (N + 1), offset_type>;
        using output_accessor_type = AccessorRestrictContiguous<output_type, 2, offset_type>;
        using weight_accessor_type = AccessorRestrictContiguous<real_type, 2, offset_type>;

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

            if constexpr (IS_HALF)
                m_shape = input_shape.pop_back();
            else
                m_shape = input_shape;
        }

        // Batched 2d.
        NOA_HD void operator()(index_type batch, index_type iy, index_type ix) const noexcept requires (N == 2) {
            const auto frequency = coord_nd_type::from_values(
                    noa::fft::index2frequency<IS_CENTERED>(iy, m_shape[0]),
                    IS_HALF ? ix : noa::fft::index2frequency<IS_CENTERED>(ix, m_shape[1]));
            const auto fftfreq_2d = frequency * m_fftfreq_step;

            coord_type fftfreq;
            if constexpr (std::is_empty_v<ctf_type>) {
                fftfreq = dot(fftfreq_2d, fftfreq_2d);
            } else {
                // Correct for anisotropic pixel size and defocus.
                if constexpr (std::is_pointer_v<ctf_type>)
                    fftfreq = static_cast<coord_type>(m_ctf[batch].isotropic_fftfreq(fftfreq_2d));
                else
                    fftfreq = static_cast<coord_type>(m_ctf.isotropic_fftfreq(fftfreq_2d));
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
                    noa::fft::index2frequency<IS_CENTERED>(iz, m_shape[0]),
                    noa::fft::index2frequency<IS_CENTERED>(iy, m_shape[1]),
                    IS_HALF ? ix : noa::fft::index2frequency<IS_CENTERED>(ix, m_shape[2]));
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
            if constexpr (nt::is_complex_v<input_type> and
                          nt::is_real_v<output_type>) {
                return abs_squared(input);
            } else {
                return input;
            }
        }

        NOA_HD void add_to_output_(index_type batch, coord_type fftfreq, output_type value) const noexcept {
            // Scale the normalized frequency back to the corresponding output shell.
            const auto scaled_fftfreq = (fftfreq - m_frequency_range_start) / m_frequency_range_span;
            const auto radius = scaled_fftfreq * static_cast<coord_type>(m_max_shell_index);
            const auto radius_floor = floor(radius);

            // Since by this point fftfreq has to be within the output frequency range,
            // "radius" is guaranteed to be within [0, m_max_shell_index].
            NOA_ASSERT(radius >= 0 and radius <= static_cast<coord_type>(m_max_shell_index));

            // Compute lerp weights.
            const auto shell_low = static_cast<index_type>(radius_floor);
            const auto shell_high = min(m_max_shell_index, shell_low + 1); // shell_low can be the last index
            const auto fraction_high = static_cast<real_type>(radius - radius_floor);
            const auto fraction_low = 1 - fraction_high;

            // TODO In CUDA, we could do the atomic reduction in shared memory to reduce global memory transfers.
            ng::atomic_add(m_output, value * fraction_low, batch, shell_low);
            ng::atomic_add(m_output, value * fraction_high, batch, shell_high);
            if (m_weight) {
                ng::atomic_add(m_weight, fraction_low, batch, shell_low);
                ng::atomic_add(m_weight, fraction_high, batch, shell_high);
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
