#pragma once

#include "noa/core/Iwise.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/geometry/Polar.hpp"

namespace noa::geometry::guts {
    /// 3d iwise operator to compute the spectrum->polar transformation of 2d (r)FFT(s).
    template<nt::sinteger Index,
             nt::any_of<f32, f64> Coord,
             nt::interpolator_spectrum_nd<2> Input,
             nt::writable_nd<3> Output>
    class Spectrum2Polar {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using coord_type = Coord;
        using coord2_type = Vec2<coord_type>;
        using shape2_type = Shape2<Index>;
        static_assert(nt::spectrum_types<input_value_type, output_value_type>);

    public:
        constexpr Spectrum2Polar(
            const input_type& spectrum,
            const shape2_type& spectrum_shape,
            const output_type& polar,
            const shape2_type& polar_shape,
            coord2_type fftfreq_range,
            bool fftfreq_range_endpoint,
            const coord2_type& angle_range,
            bool angle_range_endpoint
        ) :
            m_spectrum(spectrum),
            m_polar(polar),
            m_start_angle(angle_range[0]),
            m_start_fftfreq(fftfreq_range[0])
        {
            NOA_ASSERT(fftfreq_range[1] - fftfreq_range[0] >= 0);
            m_step_angle = Linspace{angle_range[0], angle_range[1], angle_range_endpoint}.for_size(polar_shape[0]).step;
            m_step_fftfreq = Linspace{fftfreq_range[0], fftfreq_range[1], fftfreq_range_endpoint}.for_size(polar_shape[1]).step;

            // Scale the frequency range to the polar dimension [0,width).
            m_scale = coord2_type::from_vec(spectrum_shape.vec);
        }

        NOA_HD constexpr void operator()(index_type batch, index_type y, index_type x) const {
            const auto polar_coordinate = coord2_type::from_values(y, x);
            const coord_type phi = polar_coordinate[0] * m_step_angle + m_start_angle;
            const coord_type rho = polar_coordinate[1] * m_step_fftfreq + m_start_fftfreq;
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
        coord_type m_step_fftfreq;
        coord_type m_start_fftfreq;
    };
}
