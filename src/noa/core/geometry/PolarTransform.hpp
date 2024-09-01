#pragma once

#include "noa/core/Iwise.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/geometry/Polar.hpp"

namespace noa::geometry::guts {
    /// 3d iwise operator to compute 2d cartesian->polar transformation(s).
    template<nt::sinteger Index,
             nt::any_of<f32, f64> Coord,
             nt::interpolator_nd<2> Input,
             nt::writable_nd<3> Output>
    class Polar2Cartesian {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using coord_type = Coord;
        using coord2_type = Vec2<coord_type>;
        static_assert(nt::are_power_spectrum_value_types_v<input_value_type, output_value_type>);

    public:
        Polar2Cartesian(
                const input_type& polar,
                const Shape4<index_type>& polar_shape,
                const output_type& cartesian,
                const coord2_type& cartesian_center,
                const coord2_type& radius_range,
                bool radius_range_endpoint,
                const coord2_type& angle_range,
                bool angle_range_endpoint
        ) : m_polar(polar), m_cartesian(cartesian), m_center(cartesian_center),
            m_start_angle(angle_range[0]), m_start_radius(radius_range[0])
        {
            NOA_ASSERT(polar_shape[1] == 1);
            NOA_ASSERT(radius_range[1] - radius_range[0] >= 0);

            m_step_angle = Linspace{angle_range[0], angle_range[1], angle_range_endpoint}.for_size(polar_shape[2]).step;
            m_step_radius = Linspace{radius_range[0], radius_range[1], radius_range_endpoint}.for_size(polar_shape[3]).step;
        }

        NOA_HD constexpr void operator()(index_type batch, index_type y, index_type x) const {
            auto cartesian_coordinate = coord2_type::from_values(y, x);
            cartesian_coordinate -= m_center;

            const coord_type phi = cartesian2phi(cartesian_coordinate);
            const coord_type rho = cartesian2rho(cartesian_coordinate);
            const coord2_type polar_coordinate{
                    (phi - m_start_angle) / m_step_angle,
                    (rho - m_start_radius) / m_step_radius
            };

            auto value = m_polar.interpolate_at(polar_coordinate, batch);
            m_cartesian(batch, y, x) = cast_or_abs_squared<output_value_type>(value);
        }

    private:
        input_type m_polar;
        output_type m_cartesian;
        coord2_type m_center;
        coord_type m_step_angle;
        coord_type m_step_radius;
        coord_type m_start_angle;
        coord_type m_start_radius;
    };

    /// 3d iwise operator to compute 2d polar->cartesian transformation(s).
    template<nt::sinteger Index,
             nt::any_of<f32, f64> Coord,
             nt::interpolator_nd<2> Input,
             nt::writable_nd<3> Output>
    class Cartesian2Polar {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using coord_type = Coord;
        using coord2_type = Vec2<coord_type>;
        static_assert(nt::are_power_spectrum_value_types_v<input_value_type, output_value_type>);

    public:
        Cartesian2Polar(
                const input_type& cartesian,
                const output_type& polar,
                const Shape4<index_type>& polar_shape,
                const coord2_type& cartesian_center,
                const coord2_type& radius_range,
                bool radius_range_endpoint,
                const coord2_type& angle_range,
                bool angle_range_endpoint
        ) : m_cartesian(cartesian), m_polar(polar), m_center(cartesian_center),
            m_start_angle(angle_range[0]), m_start_radius(radius_range[0])
        {
            NOA_ASSERT(polar_shape[1] == 1);
            NOA_ASSERT(radius_range[1] - radius_range[0] >= 0);

            // We could use polar2phi() and polar2rho() in the loop, but instead, precompute the step here.
            m_step_angle = Linspace{angle_range[0], angle_range[1], angle_range_endpoint}.for_size(polar_shape[2]).step;
            m_step_radius = Linspace{radius_range[0], radius_range[1], radius_range_endpoint}.for_size(polar_shape[3]).step;
        }

        NOA_HD constexpr void operator()(index_type batch, index_type y, index_type x) const {
            const auto polar_coordinate = coord2_type::from_values(y, x);
            const coord_type phi = polar_coordinate[0] * m_step_angle + m_start_angle;
            const coord_type rho = polar_coordinate[1] * m_step_radius + m_start_radius;

            coord2_type cartesian_coordinate = sincos(phi);
            cartesian_coordinate *= rho;
            cartesian_coordinate += m_center;

            auto value = m_cartesian.interpolate_at(cartesian_coordinate, batch);
            m_polar(batch, y, x) = cast_or_abs_squared<output_value_type>(value);
        }

    private:
        input_type m_cartesian;
        output_type m_polar;
        coord2_type m_center;
        coord_type m_step_angle;
        coord_type m_step_radius;
        coord_type m_start_angle;
        coord_type m_start_radius;
    };
}
