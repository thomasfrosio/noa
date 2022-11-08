#pragma once
#include "noa/common/Types.h"
#include "noa/common/geometry/Polar.h"

namespace noa::geometry::details {
    template<typename Index, typename Data, typename Interpolator, typename Offset>
    class Cartesian2Polar {
    public:
        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<Index>);

        using data_type = Data;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using accessor_type = AccessorRestrict<data_type , 3, offset_type>;

    public:
        Cartesian2Polar(interpolator_type cartesian,
                        accessor_type polar, dim4_t polar_shape,
                        float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                        bool log) noexcept
                : m_cartesian(cartesian), m_polar(polar), m_center(cartesian_center), m_log(log) {

            NOA_ASSERT(polar_shape[1] == 1);
            NOA_ASSERT(radius_range[1] - radius_range[0] >= 0);
            const float2_t size{polar_shape[2] - 1, polar_shape[3] - 1}; // endpoint = true, so N-1
            m_start_angle = angle_range[0];
            m_start_radius = radius_range[0];
            m_step_angle = (angle_range[1] - angle_range[0]) / size[0];
            m_step_radius = log ?
                            math::log(radius_range[1] - radius_range[0]) / size[1] :
                            (radius_range[1] - radius_range[0]) / size[1];
        }

        NOA_IHD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            const float2_t polar_coordinate{y, x};
            const float angle_rad = polar_coordinate[0] * m_step_angle + m_start_angle;
            const float radius = m_log ?
                                 math::exp(polar_coordinate[1] * m_step_radius) - 1 + m_start_radius :
                                 (polar_coordinate[1] * m_step_radius) + m_start_radius;

            float2_t cartesian_coordinate{math::sin(angle_rad), math::cos(angle_rad)};
            cartesian_coordinate *= radius;
            cartesian_coordinate += m_center;

            m_polar(batch, y, x) = m_cartesian(cartesian_coordinate, batch);
        }

    private:
        interpolator_type m_cartesian;
        accessor_type m_polar;
        float2_t m_center;
        float m_step_angle;
        float m_step_radius;
        float m_start_angle;
        float m_start_radius;
        bool m_log;
    };

    template<typename Index, typename Data, typename Interpolator, typename Offset>
    auto cartesian2polar(Interpolator cartesian,
                         const AccessorRestrict<Data, 3, Offset>& polar, dim4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log) noexcept {
        return Cartesian2Polar<Index, Data, Interpolator, Offset>(
                cartesian, polar, polar_shape, cartesian_center, radius_range, angle_range, log);
    }
}

namespace noa::geometry::details {
    template<typename Index, typename Data, typename Interpolator, typename Offset>
    class Polar2Cartesian {
    public:
        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<Index>);

        using data_type = Data;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using accessor_type = AccessorRestrict<data_type , 3, offset_type>;

    public:
        Polar2Cartesian(interpolator_type polar, dim4_t polar_shape,
                        accessor_type cartesian,
                        float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                        bool log) noexcept
                : m_polar(polar), m_cartesian(cartesian), m_center(cartesian_center), m_log(log) {

            NOA_ASSERT(polar_shape[1] == 1);
            NOA_ASSERT(radius_range[1] - radius_range[0] >= 0);
            const float2_t size{polar_shape[2] - 1, polar_shape[3] - 1}; // endpoint = true, so N-1
            m_start_angle = angle_range[0];
            m_start_radius = radius_range[0];
            m_step_angle = (angle_range[1] - angle_range[0]) / size[0];
            m_step_radius = log ?
                            math::log(radius_range[1] - radius_range[0]) / size[1] :
                            (radius_range[1] - radius_range[0]) / size[1];
        }

        NOA_IHD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            float2_t cartesian_coordinate{y, x};
            cartesian_coordinate -= m_center;

            const float phi = noa::geometry::cartesian2phi(cartesian_coordinate);
            const float rho = noa::geometry::cartesian2rho(cartesian_coordinate);

            const float py = (phi - m_start_angle) / m_step_angle;
            const float px = m_log ?
                             math::log(rho + 1 - m_start_radius) / m_step_radius :
                             (rho - m_start_radius) / m_step_radius;
            float2_t polar_coordinate{py, px};

            m_cartesian(batch, y, x) = m_polar(polar_coordinate, batch);
        }

    private:
        interpolator_type m_polar;
        accessor_type m_cartesian;
        float2_t m_center;
        float m_step_angle;
        float m_step_radius;
        float m_start_angle;
        float m_start_radius;
        bool m_log;
    };

    template<typename Index, typename Data, typename Interpolator, typename Offset>
    auto polar2cartesian(Interpolator polar, dim4_t polar_shape,
                         const AccessorRestrict<Data, 3, Offset>& cartesian,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log) noexcept {
        return Polar2Cartesian<Index, Data, Interpolator, Offset>(
                polar, polar_shape, cartesian, cartesian_center, radius_range, angle_range, log);
    }
}
