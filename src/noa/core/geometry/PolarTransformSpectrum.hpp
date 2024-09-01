#pragma once

#include "noa/core/Iwise.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/geometry/Polar.hpp"

namespace noa::geometry::guts {
    /// 3d iwise operator to compute the spectrum->polar transformation of 2d centered rFFT(s).
    template<nt::sinteger Index,
             nt::any_of<f32, f64> Coord,
             nt::interpolator_spectrum_nd<2> Input,
             nt::writable_nd<3> Output>
    class Spectrum2Polar {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type<input_type>;
        using output_value_type = nt::value_type<output_type>;
        using coord_type = Coord;
        using coord2_type = Vec2<coord_type>;
        static_assert(nt::are_power_spectrum_value_types_v<input_value_type, output_value_type>);

    public:
        constexpr Spectrum2Polar(
                const input_type& spectrum,
                const Shape4<index_type>& spectrum_shape,
                const output_type& polar,
                const Shape4<index_type>& polar_shape,
                coord2_type frequency_range,
                bool frequency_range_endpoint,
                const coord2_type& angle_range,
                bool angle_range_endpoint
        ) :
                m_spectrum(spectrum),
                m_polar(polar),
                m_start_angle(angle_range[0]),
                m_start_frequency(frequency_range[0])
        {
            NOA_ASSERT(polar_shape[2] > 1 and polar_shape[3] > 1);
            NOA_ASSERT(polar_shape[1] == 1 and spectrum_shape[1] == 1);
            NOA_ASSERT(frequency_range[1] - frequency_range[0] >= 0);

            m_step_angle = Linspace{angle_range[0], angle_range[1], angle_range_endpoint}.for_size(polar_shape[2]).step;
            m_step_frequency = Linspace{frequency_range[0], frequency_range[1], frequency_range_endpoint}.for_size(polar_shape[3]).step;

            // Scale the frequency range to the polar dimension [0,width).
            m_scale = coord2_type::from_vec(spectrum_shape.filter(2, 3).vec);
        }

        NOA_HD constexpr void operator()(index_type batch, index_type y, index_type x) const {
            const auto polar_coordinate = coord2_type::from_values(y, x);
            const coord_type phi = polar_coordinate[0] * m_step_angle + m_start_angle;
            const coord_type rho = polar_coordinate[1] * m_step_frequency + m_start_frequency;
            const coord2_type frequency = (rho * sincos(phi)) * m_scale;
            auto value = m_spectrum.interpolate_spectrum_at(frequency, batch);
            m_polar(batch, y, x) = cast_or_abs_squared<output_value_type>(value);
        }

    private:
        input_type m_spectrum;
        output_type m_polar;
        coord2_type m_scale;
        coord_type m_step_angle;
        coord_type m_start_angle;
        coord_type m_step_frequency;
        coord_type m_start_frequency;
    };
}
