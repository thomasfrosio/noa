#pragma once
#include "noa/common/Types.h"
#include "noa/common/geometry/Polar.h"

namespace noa::geometry::fft::details {
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
        Cartesian2Polar(interpolator_type cartesian, dim4_t cartesian_shape,
                        accessor_type polar, dim4_t polar_shape,
                        float2_t frequency_range, float2_t angle_range, bool log)
                : m_cartesian(cartesian), m_polar(polar), m_log(log) {

            NOA_ASSERT(polar_shape[1] == 1 && cartesian_shape[1] == 1);
            NOA_ASSERT(frequency_range[1] - frequency_range[0] >= 0);
            const auto polar_shape_2d = safe_cast<long2_t>(dim2_t(polar_shape.get(2)));
            const auto f_polar_shape_2d = float2_t(polar_shape_2d - 1);
            m_start_angle = angle_range[0];
            m_step_angle = (angle_range[1] - angle_range[0]) / f_polar_shape_2d[0];

            const auto half_shape = float2_t(dim2_t(cartesian_shape.get(2)) / 2);
            m_center = half_shape[0]; // center in x is 0

            const auto radius_y_range = frequency_range * 2 * half_shape[0];
            const auto radius_x_range = frequency_range * 2 * half_shape[1];
            m_start_radius = {radius_y_range[0], radius_x_range[0]};
            if (log) {
                m_step_radius[0] = math::log(radius_y_range[1] - radius_y_range[0]) / f_polar_shape_2d[1];
                m_step_radius[1] = math::log(radius_x_range[1] - radius_x_range[0]) / f_polar_shape_2d[1];
            } else {
                m_step_radius[0] = (radius_y_range[1] - radius_y_range[0]) / f_polar_shape_2d[1];
                m_step_radius[1] = (radius_x_range[1] - radius_x_range[0]) / f_polar_shape_2d[1];
            }
        }

        NOA_IHD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            const float2_t polar_coordinate{y, x};
            const float angle_rad = polar_coordinate[0] * m_step_angle + m_start_angle;
            float magnitude_y, magnitude_x;
            if (m_log) {
                magnitude_y = math::exp(polar_coordinate[1] * m_step_radius[0]) - 1 + m_start_radius[0];
                magnitude_x = math::exp(polar_coordinate[1] * m_step_radius[1]) - 1 + m_start_radius[1];
            } else {
                magnitude_y = polar_coordinate[1] * m_step_radius[0] + m_start_radius[0];
                magnitude_x = polar_coordinate[1] * m_step_radius[1] + m_start_radius[1];
            }

            float2_t cartesian_coordinates{magnitude_y * math::sin(angle_rad),
                                           magnitude_x * math::cos(angle_rad)};
            float conj = 1;
            if (cartesian_coordinates[1] < 0) {
                cartesian_coordinates = -cartesian_coordinates;
                if constexpr (traits::is_complex_v<data_type>)
                    conj = -1;
            } else {
                (void) conj;
            }
            cartesian_coordinates[0] += m_center;

            data_type value = m_cartesian(cartesian_coordinates, batch);
            if constexpr (traits::is_complex_v<data_type>)
                value.imag *= conj;

            m_polar(batch, y, x) = value;
        }

    private:
        interpolator_type m_cartesian;
        accessor_type m_polar;
        float m_center;
        float m_step_angle;
        float m_start_angle;
        float2_t m_start_radius;
        float2_t m_step_radius;
        bool m_log;
    };

    template<typename Index, typename Data, typename Interpolator, typename Offset>
    auto cartesian2polar(Interpolator cartesian, dim4_t cartesian_shape,
                         AccessorRestrict<Data, 3, Offset> polar, dim4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range, bool log) noexcept {
        return Cartesian2Polar<Index, Data, Interpolator, Offset>(
                cartesian, cartesian_shape, polar, polar_shape, frequency_range, angle_range, log);
    }
}
