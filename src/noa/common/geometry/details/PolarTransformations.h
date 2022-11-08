#pragma once
#include "noa/common/Types.h"
#include "noa/common/geometry/Polar.h"

namespace noa::geometry::details {
    template<bool LAYERED, typename data_t, typename interpolator_t, typename index_t, typename offset_t>
    class Cartesian2Polar {
    public:
        static_assert(traits::is_any_v<data_t, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<index_t>);
        using output_accessor = AccessorRestrict<data_t, 3, offset_t>;

    public:
        Cartesian2Polar(interpolator_t cartesian,
                        output_accessor polar, dim4_t polar_shape,
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

        NOA_IHD void operator()(index_t i, index_t y, index_t x) const noexcept {
            const float2_t polar_coordinate{y, x};
            const float angle_rad = polar_coordinate[0] * m_step_angle + m_start_angle;
            const float radius = m_log ?
                                 math::exp(polar_coordinate[1] * m_step_radius) - 1 + m_start_radius :
                                 (polar_coordinate[1] * m_step_radius) + m_start_radius;

            float2_t cartesian_coordinate{math::sin(angle_rad), math::cos(angle_rad)};
            cartesian_coordinate *= radius;
            cartesian_coordinate += m_center;

            m_polar(i, y, x) = interpolate_(cartesian_coordinate, i);
        }

    private:
        NOA_FHD data_t interpolate_(float2_t coordinate, index_t i) const noexcept {
            if constexpr (LAYERED)
                return m_cartesian(coordinate, i);
            else
                return m_cartesian(coordinate);
        }

    private:
        interpolator_t m_cartesian;
        output_accessor m_polar;
        float2_t m_center;
        float m_step_angle;
        float m_step_radius;
        float m_start_angle;
        float m_start_radius;
        bool m_log;
    };

    template<bool LAYERED, typename index_t, typename data_t, typename interpolator_t, typename offset_t>
    auto cartesian2polar(interpolator_t cartesian,
                         const AccessorRestrict<data_t, 3, offset_t>& polar, dim4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log) noexcept {
        return Cartesian2Polar<LAYERED, data_t, interpolator_t, index_t, offset_t>(
                cartesian, polar, polar_shape, cartesian_center, radius_range, angle_range, log);
    }
}

namespace noa::geometry::details {
    template<bool LAYERED, typename data_t, typename interpolator_t, typename index_t, typename offset_t>
    class Polar2Cartesian {
    public:
        static_assert(traits::is_any_v<data_t, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<index_t>);
        using output_accessor = AccessorRestrict<data_t, 3, offset_t>;

    public:
        Polar2Cartesian(interpolator_t polar, dim4_t polar_shape,
                        output_accessor cartesian,
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

        NOA_IHD void operator()(index_t i, index_t y, index_t x) const noexcept {
            float2_t cartesian_coordinate{y, x};
            cartesian_coordinate -= m_center;

            const float phi = noa::geometry::cartesian2phi(cartesian_coordinate);
            const float rho = noa::geometry::cartesian2rho(cartesian_coordinate);

            const float py = (phi - m_start_angle) / m_step_angle;
            const float px = m_log ?
                             math::log(rho + 1 - m_start_radius) / m_step_radius :
                             (rho - m_start_radius) / m_step_radius;
            float2_t polar_coordinate{py, px};

            m_cartesian(i, y, x) = interpolate_(polar_coordinate, i);
        }

    private:
        NOA_FHD data_t interpolate_(float2_t coordinate, index_t i) const noexcept {
            if constexpr (LAYERED)
                return m_polar(coordinate, i);
            else
                return m_polar(coordinate);
        }

    private:
        interpolator_t m_polar;
        output_accessor m_cartesian;
        float2_t m_center;
        float m_step_angle;
        float m_step_radius;
        float m_start_angle;
        float m_start_radius;
        bool m_log;
    };

    template<bool LAYERED, typename index_t, typename data_t, typename interpolator_t, typename offset_t>
    auto polar2cartesian(interpolator_t polar, dim4_t polar_shape,
                         const AccessorRestrict<data_t, 3, offset_t>& cartesian,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log) noexcept {
        return Polar2Cartesian<LAYERED, data_t, interpolator_t, index_t, offset_t>(
                polar, polar_shape, cartesian, cartesian_center, radius_range, angle_range, log);
    }
}

namespace noa::geometry::fft::details {
    template<bool LAYERED, typename data_t, typename interpolator_t, typename index_t, typename offset_t>
    class Cartesian2Polar {
    public:
        static_assert(traits::is_any_v<data_t, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<index_t>);
        using output_accessor = AccessorRestrict<data_t, 3, offset_t>;

    public:
        Cartesian2Polar(interpolator_t cartesian, dim4_t cartesian_shape,
                        output_accessor polar, dim4_t polar_shape,
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

        NOA_IHD void operator()(index_t i, index_t y, index_t x) const noexcept {
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
                if constexpr (traits::is_complex_v<data_t>)
                    conj = -1;
            } else {
                (void) conj;
            }
            cartesian_coordinates[0] += m_center;

            data_t value = interpolate_(cartesian_coordinates, i);
            if constexpr (traits::is_complex_v<data_t>)
                value.imag *= conj;

            m_polar(i, y, x) = value;
        }

    private:
        NOA_FHD data_t interpolate_(float2_t coordinate, index_t i) const noexcept {
            if constexpr (LAYERED)
                return m_cartesian(coordinate, i);
            else
                return m_cartesian(coordinate);
        }

    private:
        interpolator_t m_cartesian;
        output_accessor m_polar;
        float m_center;
        float m_step_angle;
        float m_start_angle;
        float2_t m_start_radius;
        float2_t m_step_radius;
        bool m_log;
    };

    template<bool LAYERED, typename index_t, typename data_t, typename interpolator_t, typename offset_t>
    auto cartesian2polar(interpolator_t cartesian, dim4_t cartesian_shape,
                         AccessorRestrict<data_t, 3, offset_t> polar, dim4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range, bool log) noexcept {
        return Cartesian2Polar<LAYERED, data_t, interpolator_t, index_t, offset_t>(
                cartesian, cartesian_shape, polar, polar_shape, frequency_range, angle_range, log);
    }
}
