#pragma once

#include "noa/core/geometry/Polar.hpp"

namespace noa::geometry {
    /// 3d iwise operator to compute the spectrum->polar transformation of 2d centered rFFT(s).
    template<typename Index, typename Coord, typename Interpolator, typename OutputAccessor>
    requires (nt::is_interpolator_nd<Interpolator, 2>::value and
              nt::is_accessor_pure_nd<OutputAccessor, 3>::value and
              nt::is_sint<Index>::value and
              nt::is_any<Coord, f32, f64>::value)
    class Spectrum2Polar {
    public:
        using index_type = Index;
        using interpolator_type = Interpolator;
        using output_accessor_type = OutputAccessor;
        using input_value_type = interpolator_type::mutable_value_type;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = output_accessor_type::value_type;
        using coord_type = Coord;
        using coord2_type = Vec2<coord_type>;
        static_assert(nt::are_real_v<input_value_type, output_value_type> or
                      nt::are_complex_v<input_value_type, output_value_type> or
                      (nt::is_complex_v<input_value_type> and nt::is_real_v<output_value_type>));

    public:
        Spectrum2Polar(
                const interpolator_type& spectrum,
                const Shape4<index_type>& spectrum_shape,
                const output_accessor_type& polar,
                const Shape4<index_type>& polar_shape,
                coord2_type frequency_range,
                bool frequency_range_endpoint,
                const coord2_type& angle_range,
                bool angle_range_endpoint
        ) : m_spectrum(spectrum),
            m_polar(polar),
            m_start_angle(angle_range[0]),
            m_start_frequency(frequency_range[0])
        {
            NOA_ASSERT(polar_shape[2] > 1 && polar_shape[3] > 1);
            NOA_ASSERT(polar_shape[1] == 1 && spectrum_shape[1] == 1);
            NOA_ASSERT(frequency_range[1] - frequency_range[0] >= 0);

            using linspace_t = Linspace<coord_type, index_type>;
            m_step_angle = linspace_t::from_range(angle_range[0], angle_range[1], polar_shape[2], angle_range_endpoint).step;
            m_step_frequency = linspace_t::from_range(frequency_range[0], frequency_range[1], polar_shape[3], frequency_range_endpoint).step;

            // Scale the frequency range to the polar dimension [0,width).
            m_scale = coord2_type::from_vec(spectrum_shape.filter(2, 3).vec);
            m_height_dc_center = static_cast<coord_type>(spectrum_shape[2] / 2);
        }

        NOA_HD constexpr void operator()(index_type batch, index_type y, index_type x) const {
            const auto polar_coordinate = coord2_type::from_values(y, x);
            const coord_type phi = polar_coordinate[0] * m_step_angle + m_start_angle;
            const coord_type rho = polar_coordinate[1] * m_step_frequency + m_start_frequency;

            coord2_type spectrum_coordinates = (rho * sincos(phi)) * m_scale;
            input_real_type conj = 1;
            if (spectrum_coordinates[1] < 0) {
                spectrum_coordinates = -spectrum_coordinates;
                if constexpr (nt::is_complex_v<input_value_type>)
                    conj = -1;
            } else {
                (void) conj;
            }
            spectrum_coordinates[0] += m_height_dc_center;

            auto value = m_spectrum(spectrum_coordinates, batch);
            if constexpr (nt::are_complex_v<input_value_type, output_value_type>)
                value.imag *= conj;

            if constexpr (nt::is_complex_v<input_value_type> and nt::is_real_v<output_value_type>)
                m_polar(batch, y, x) = static_cast<output_value_type>(abs_squared(value));
            else
                m_polar(batch, y, x) = static_cast<output_value_type>(value);
        }

    private:
        interpolator_type m_spectrum;
        output_accessor_type m_polar;
        coord2_type m_scale;
        coord_type m_height_dc_center;
        coord_type m_step_angle;
        coord_type m_start_angle;
        coord_type m_step_frequency;
        coord_type m_start_frequency;
    };
}
