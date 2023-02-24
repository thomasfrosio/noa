#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/geometry/Polar.hpp"

namespace noa::algorithm::geometry {
    template<typename Index, typename Value, typename Interpolator, typename Offset>
    class Cartesian2PolarRFFT {
    public:
        static_assert(noa::traits::is_real_or_complex_v<Value>);
        static_assert(noa::traits::is_int_v<Index>);

        using value_type = Value;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;

        using coord_type = typename Interpolator::coord_type;
        using coord2_type = Vec2<coord_type>;
        using real_type = traits::value_type_t<value_type>;
        using accessor_type = AccessorRestrict<value_type, 3, offset_type>;

    public:
        Cartesian2PolarRFFT(
                const interpolator_type& cartesian,
                const Shape4<index_type>& cartesian_shape,
                const accessor_type& polar,
                const Shape4<index_type>& polar_shape,
                const coord2_type& frequency_range,
                const coord2_type& angle_range,
                bool log)
                : m_cartesian(cartesian), m_polar(polar),
                  m_start_angle(angle_range[0]), m_log(log) {

            NOA_ASSERT(polar_shape[2] > 1 && polar_shape[3] > 1);
            NOA_ASSERT(polar_shape[1] == 1 && cartesian_shape[1] == 1);
            NOA_ASSERT(frequency_range[1] - frequency_range[0] >= 0);
            const auto polar_shape_2d = polar_shape.filter(2, 3);
            const auto f_polar_shape_2d = coord2_type((polar_shape_2d - 1).vec());
            m_step_angle = (angle_range[1] - angle_range[0]) / f_polar_shape_2d[0];

            const auto half_shape = coord2_type((cartesian_shape.filter(2, 3) / 2).vec());
            m_center = half_shape[0]; // center in x is 0

            const auto radius_y_range = frequency_range * 2 * half_shape[0];
            const auto radius_x_range = frequency_range * 2 * half_shape[1];
            m_start_radius = coord2_type{radius_y_range[0], radius_x_range[0]};
            if (log) {
                m_step_radius[0] = noa::math::log(radius_y_range[1] - radius_y_range[0]) / f_polar_shape_2d[1];
                m_step_radius[1] = noa::math::log(radius_x_range[1] - radius_x_range[0]) / f_polar_shape_2d[1];
            } else {
                m_step_radius[0] = (radius_y_range[1] - radius_y_range[0]) / f_polar_shape_2d[1];
                m_step_radius[1] = (radius_x_range[1] - radius_x_range[0]) / f_polar_shape_2d[1];
            }
        }

        NOA_IHD void operator()(index_type batch, index_type phi, index_type rho) const noexcept {
            const coord2_type polar_coordinate{phi, rho};
            const coord_type angle_rad = polar_coordinate[0] * m_step_angle + m_start_angle;
            coord2_type magnitude;
            if (m_log) {
                magnitude[0] = noa::math::exp(polar_coordinate[1] * m_step_radius[0]) - 1 + m_start_radius[0];
                magnitude[1] = noa::math::exp(polar_coordinate[1] * m_step_radius[1]) - 1 + m_start_radius[1];
            } else {
                magnitude[0] = polar_coordinate[1] * m_step_radius[0] + m_start_radius[0];
                magnitude[1] = polar_coordinate[1] * m_step_radius[1] + m_start_radius[1];
            }

            coord2_type cartesian_coordinates = magnitude * noa::math::sincos(angle_rad);
            real_type conj = 1;
            if (cartesian_coordinates[1] < 0) {
                cartesian_coordinates = -cartesian_coordinates;
                if constexpr (noa::traits::is_complex_v<value_type>)
                    conj = -1;
            } else {
                (void) conj;
            }
            cartesian_coordinates[0] += m_center;

            auto value = static_cast<value_type>(m_cartesian(cartesian_coordinates, batch));
            if constexpr (noa::traits::is_complex_v<value_type>)
                value.imag *= conj;

            m_polar(batch, phi, rho) = value;
        }

    private:
        interpolator_type m_cartesian;
        accessor_type m_polar;
        coord_type m_center;
        coord_type m_step_angle;
        coord_type m_start_angle;
        coord2_type m_start_radius;
        coord2_type m_step_radius;
        bool m_log;
    };

    template<typename Index, typename Value, typename Interpolator, typename Coord, typename Offset>
    auto cartesian2polar_rfft(const Interpolator& cartesian,
                              const Shape4<Index>& cartesian_shape,
                              const AccessorRestrict<Value, 3, Offset>& polar,
                              const Shape4<Index>& polar_shape,
                              const Vec2<Coord>& frequency_range,
                              const Vec2<Coord>& angle_range,
                              bool log) noexcept {
        return Cartesian2PolarRFFT<Index, Value, Interpolator, Offset>(
                cartesian, cartesian_shape, polar, polar_shape, frequency_range, angle_range, log);
    }
}
