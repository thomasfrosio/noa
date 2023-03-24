#pragma once
#include "noa/core/Types.hpp"
#include "noa/core/geometry/Polar.hpp"

namespace noa::algorithm::geometry {
    template<typename Index, typename Data, typename Interpolator, typename Offset>
    class Cartesian2Polar {
    public:
        static_assert(noa::traits::is_real_or_complex_v<Data>);
        static_assert(noa::traits::is_int_v<Index>);

        using value_type = Data;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;

        using coord_type = typename Interpolator::coord_type;
        using coord2_type = Vec2<coord_type>;
        using accessor_type = AccessorRestrict<value_type, 3, offset_type>;

    public:
        Cartesian2Polar(
                const interpolator_type& cartesian,
                const accessor_type& polar,
                const Shape4<index_type>& polar_shape,
                const coord2_type& cartesian_center,
                const coord2_type& radius_range,
                const coord2_type& angle_range,
                bool log) noexcept
                : m_cartesian(cartesian), m_polar(polar), m_center(cartesian_center),
                  m_start_angle(angle_range[0]), m_start_radius(radius_range[0]), m_log(log) {

            NOA_ASSERT(polar_shape[1] == 1);
            NOA_ASSERT(radius_range[1] - radius_range[0] >= 0);
            const auto size = coord2_type((polar_shape.filter(2, 3) - 1).vec()); // endpoint=true, so N-1
            m_step_angle = (angle_range[1] - angle_range[0]) / size[0];
            m_step_radius = log ?
                            noa::math::log(radius_range[1] - radius_range[0]) / size[1] :
                            (radius_range[1] - radius_range[0]) / size[1];
        }

        NOA_IHD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            const coord2_type polar_coordinate{y, x};
            const coord_type phi = polar_coordinate[0] * m_step_angle + m_start_angle;
            const coord_type rho =
                    m_log ?
                    noa::math::exp(polar_coordinate[1] * m_step_radius) - 1 + m_start_radius :
                    (polar_coordinate[1] * m_step_radius) + m_start_radius;

            coord2_type cartesian_coordinate = noa::math::sincos(phi);
            cartesian_coordinate *= rho;
            cartesian_coordinate += m_center;

            m_polar(batch, y, x) = static_cast<value_type>(m_cartesian(cartesian_coordinate, batch));
        }

    private:
        interpolator_type m_cartesian;
        accessor_type m_polar;
        coord2_type m_center;
        coord_type m_step_angle;
        coord_type m_step_radius;
        coord_type m_start_angle;
        coord_type m_start_radius;
        bool m_log;
    };

    template<typename Index, typename Value, typename Interpolator, typename Offset>
    class Polar2Cartesian {
    public:
        static_assert(noa::traits::is_real_or_complex_v<Value>);
        static_assert(noa::traits::is_int_v<Index>);

        using value_type = Value;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;

        using coord_type = typename Interpolator::coord_type;
        using coord2_type = Vec2<coord_type>;
        using accessor_type = AccessorRestrict<value_type, 3, offset_type>;

    public:
        Polar2Cartesian(
                const interpolator_type& polar,
                const Shape4<index_type>& polar_shape,
                const accessor_type& cartesian,
                const coord2_type& cartesian_center,
                const coord2_type& radius_range,
                const coord2_type& angle_range,
                bool log) noexcept
                : m_polar(polar), m_cartesian(cartesian), m_center(cartesian_center),
                  m_start_angle(angle_range[0]), m_start_radius(radius_range[0]), m_log(log) {

            NOA_ASSERT(polar_shape[1] == 1);
            NOA_ASSERT(radius_range[1] - radius_range[0] >= 0);
            const auto size = coord2_type((polar_shape.filter(2, 3) - 1).vec()); // endpoint=true, so N-1
            m_step_angle = (angle_range[1] - angle_range[0]) / size[0];
            m_step_radius = log ?
                            noa::math::log(radius_range[1] - radius_range[0]) / size[1] :
                            (radius_range[1] - radius_range[0]) / size[1];
        }

        NOA_IHD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            coord2_type cartesian_coordinate{y, x};
            cartesian_coordinate -= m_center;

            const coord_type phi = noa::geometry::cartesian2phi(cartesian_coordinate);
            const coord_type rho = noa::geometry::cartesian2rho(cartesian_coordinate);

            const coord_type py = (phi - m_start_angle) / m_step_angle;
            const coord_type px =
                    m_log ?
                    noa::math::log(rho + 1 - m_start_radius) / m_step_radius :
                    (rho - m_start_radius) / m_step_radius;
            const coord2_type polar_coordinate{py, px};

            m_cartesian(batch, y, x) = m_polar(polar_coordinate, batch);
        }

    private:
        interpolator_type m_polar;
        accessor_type m_cartesian;
        coord2_type m_center;
        coord_type m_step_angle;
        coord_type m_step_radius;
        coord_type m_start_angle;
        coord_type m_start_radius;
        bool m_log;
    };
}

namespace noa::algorithm::geometry {
    template<typename Index, typename Data, typename Interpolator, typename Coord, typename Offset>
    auto cartesian2polar(const Interpolator& cartesian,
                         const AccessorRestrict<Data, 3, Offset>& polar,
                         const Shape4<Index>& polar_shape,
                         const Vec2<Coord>& cartesian_center,
                         const Vec2<Coord>& radius_range,
                         const Vec2<Coord>& angle_range,
                         bool log) noexcept {
        return Cartesian2Polar<Index, Data, Interpolator, Offset>(
                cartesian, polar, polar_shape, cartesian_center, radius_range, angle_range, log);
    }

    template<typename Index, typename Data, typename Interpolator, typename Coord, typename Offset>
    auto polar2cartesian(const Interpolator& polar,
                         const Shape4<Index>& polar_shape,
                         const AccessorRestrict<Data, 3, Offset>& cartesian,
                         const Vec2<Coord>& cartesian_center,
                         const Vec2<Coord>& radius_range,
                         const Vec2<Coord>& angle_range,
                         bool log) noexcept {
        return Polar2Cartesian<Index, Data, Interpolator, Offset>(
                polar, polar_shape, cartesian, cartesian_center, radius_range, angle_range, log);
    }
}
