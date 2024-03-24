#pragma once

#include "noa/core/Linspace.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/geometry/Polar.hpp"

namespace noa::geometry {
    /// 3d iwise operator to compute 2d cartesian->polar transformation(s).
    template<typename Index, typename Coord, typename Interpolator, typename OutputAccessor>
    requires (nt::is_interpolator_nd<Interpolator, 2>::value and
              nt::is_accessor_pure_nd<OutputAccessor, 3>::value and
              nt::is_sint<Index>::value and
              nt::is_any<Coord, f32, f64>::value)
    class Polar2Cartesian {
    public:
        using index_type = Index;
        using interpolator_type = Interpolator;
        using output_accessor_type = OutputAccessor;
        using input_value_type = interpolator_type::mutable_value_type;
        using output_value_type = output_accessor_type::value_type;
        using coord_type = Coord;
        using coord2_type = Vec2<coord_type>;
        static_assert(nt::are_real_v<input_value_type, output_value_type> or
                      nt::are_complex_v<input_value_type, output_value_type> or
                      (nt::is_complex_v<input_value_type> and nt::is_real_v<output_value_type>));

    public:
        Polar2Cartesian(
                const interpolator_type& polar,
                const Shape4<index_type>& polar_shape,
                const output_accessor_type& cartesian,
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

            using linspace_t = Linspace<coord_type, index_type>;
            m_step_angle = linspace_t::from_range(angle_range[0], angle_range[1], polar_shape[2], angle_range_endpoint).step;
            m_step_radius = linspace_t::from_range(radius_range[0], radius_range[1], polar_shape[3], radius_range_endpoint).step;
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

            auto value = m_polar(polar_coordinate, batch);
            if (nt::is_complex_v<input_value_type> and nt::is_real_v<output_value_type>) {
                m_cartesian(batch, y, x) = static_cast<output_value_type>(abs_squared(value));
            } else {
                m_cartesian(batch, y, x) = static_cast<output_value_type>(value);
            }
        }

    private:
        interpolator_type m_polar;
        output_accessor_type m_cartesian;
        coord2_type m_center;
        coord_type m_step_angle;
        coord_type m_step_radius;
        coord_type m_start_angle;
        coord_type m_start_radius;
    };

    /// 3d iwise operator to compute 2d polar->cartesian transformation(s).
    template<typename Index, typename Coord, typename Interpolator, typename OutputAccessor>
    requires (nt::is_interpolator_nd<Interpolator, 2>::value and
              nt::is_accessor_pure_nd<OutputAccessor, 3>::value and
              nt::is_sint<Index>::value and
              nt::is_any<Coord, f32, f64>::value)
    class Cartesian2Polar {
    public:
        using index_type = Index;
        using interpolator_type = Interpolator;
        using output_accessor_type = OutputAccessor;
        using input_value_type = interpolator_type::mutable_value_type;
        using output_value_type = output_accessor_type::value_type;
        using coord_type = Coord;
        using coord2_type = Vec2<coord_type>;
        static_assert(nt::are_real_v<input_value_type, output_value_type> or
                      nt::are_complex_v<input_value_type, output_value_type> or
                      (nt::is_complex_v<input_value_type> and nt::is_real_v<output_value_type>));

    public:
        Cartesian2Polar(
                const interpolator_type& cartesian,
                const output_accessor_type& polar,
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
            using linspace_t = Linspace<coord_type, index_type>;
            m_step_angle = linspace_t::from_range(angle_range[0], angle_range[1], polar_shape[2], angle_range_endpoint).step;
            m_step_radius = linspace_t::from_range(radius_range[0], radius_range[1], polar_shape[3], radius_range_endpoint).step;
        }

        NOA_HD constexpr void operator()(index_type batch, index_type y, index_type x) const {
            const auto polar_coordinate = coord2_type::from_values(y, x);
            const coord_type phi = polar_coordinate[0] * m_step_angle + m_start_angle;
            const coord_type rho = polar_coordinate[1] * m_step_radius + m_start_radius;

            coord2_type cartesian_coordinate = sincos(phi);
            cartesian_coordinate *= rho;
            cartesian_coordinate += m_center;

            auto value = m_cartesian(cartesian_coordinate, batch);
            if (nt::is_complex_v<input_value_type> and nt::is_real_v<output_value_type>) {
                m_polar(batch, y, x) = static_cast<output_value_type>(abs_squared(value));
            } else {
                m_polar(batch, y, x) = static_cast<output_value_type>(value);
            }
        }

    private:
        interpolator_type m_cartesian;
        output_accessor_type m_polar;
        coord2_type m_center;
        coord_type m_step_angle;
        coord_type m_step_radius;
        coord_type m_start_angle;
        coord_type m_start_radius;
    };
}
