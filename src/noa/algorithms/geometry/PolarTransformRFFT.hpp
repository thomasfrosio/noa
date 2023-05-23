#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/geometry/Polar.hpp"
#include "noa/algorithms/memory/Linspace.hpp"

namespace noa::algorithm::geometry {
    template<typename Index, typename Offset, typename Interpolator, typename OutputValue>
    class Cartesian2PolarRFFT {
    public:
        using input_type = typename Interpolator::value_type;
        using output_type = OutputValue;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;

        static_assert(noa::traits::is_int_v<Index>);
        static_assert((noa::traits::are_all_same_v<input_type, output_type> &&
                       noa::traits::are_real_or_complex_v<input_type, output_type>) ||
                      (noa::traits::is_complex_v<input_type> &&
                       noa::traits::is_real_v<output_type> &&
                       noa::traits::are_same_value_type_v<input_type, output_type>));

        using coord_type = typename Interpolator::coord_type;
        using coord2_type = Vec2<coord_type>;
        using real_type = traits::value_type_t<output_type>;
        using accessor_type = AccessorRestrict<output_type, 3, offset_type>;

    public:
        Cartesian2PolarRFFT(
                const interpolator_type& cartesian,
                const Shape4<index_type>& cartesian_shape,
                const accessor_type& polar,
                const Shape4<index_type>& polar_shape,
                const coord2_type& frequency_range,
                bool frequency_range_endpoint,
                const coord2_type& angle_range,
                bool angle_range_endpoint) noexcept
                : m_cartesian(cartesian), m_polar(polar),
                  m_start_angle(angle_range[0]) {

            NOA_ASSERT(polar_shape[2] > 1 && polar_shape[3] > 1);
            NOA_ASSERT(polar_shape[1] == 1 && cartesian_shape[1] == 1);
            NOA_ASSERT(frequency_range[1] - frequency_range[0] >= 0);

            using namespace noa::algorithm::memory;
            m_step_angle = linspace_step(polar_shape[2], angle_range[0], angle_range[1], angle_range_endpoint);

            const auto half_shape = coord2_type((cartesian_shape.filter(2, 3) / 2).vec());
            m_center = half_shape[0]; // center in x is 0

            // Scale the frequency range to the dimensions size.
            const auto radius_range_along_y = 2 * frequency_range * half_shape[0];
            const auto radius_range_along_x = 2 * frequency_range * half_shape[1];
            m_start_radius = {radius_range_along_y[0], radius_range_along_x[0]};
            m_step_radius[0] = linspace_step(
                    polar_shape[3], radius_range_along_y[0], radius_range_along_y[1], frequency_range_endpoint);
            m_step_radius[1] = linspace_step(
                    polar_shape[3], radius_range_along_x[0], radius_range_along_x[1], frequency_range_endpoint);
        }

        NOA_IHD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            const coord2_type polar_coordinate{y, x};
            const coord_type phi = polar_coordinate[0] * m_step_angle + m_start_angle;
            coord2_type rho{
                    polar_coordinate[1] * m_step_radius[0] + m_start_radius[0],
                    polar_coordinate[1] * m_step_radius[1] + m_start_radius[1]
            };

            coord2_type cartesian_coordinates = rho * noa::math::sincos(phi);
            real_type conj = 1;
            if (cartesian_coordinates[1] < 0) {
                cartesian_coordinates = -cartesian_coordinates;
                if constexpr (noa::traits::is_complex_v<input_type>)
                    conj = -1;
            } else {
                (void) conj;
            }
            cartesian_coordinates[0] += m_center;

            input_type value = m_cartesian(cartesian_coordinates, batch);
            if constexpr (noa::traits::are_complex_v<input_type, output_type>)
                value.imag *= conj;

            if constexpr (noa::traits::is_complex_v<input_type> && noa::traits::is_real_v<output_type>)
                m_polar(batch, y, x) = noa::math::abs(value);
            else
                m_polar(batch, y, x) = value;
        }

    private:
        interpolator_type m_cartesian;
        accessor_type m_polar;
        coord_type m_center;
        coord_type m_step_angle;
        coord_type m_start_angle;
        coord2_type m_start_radius;
        coord2_type m_step_radius;
    };

    template<typename Index, typename Coord, typename Offset, typename Interpolator, typename OutputValue>
    auto cartesian2polar_rfft(const Interpolator& cartesian,
                              const Shape4<Index>& cartesian_shape,
                              const AccessorRestrict<OutputValue, 3, Offset>& polar,
                              const Shape4<Index>& polar_shape,
                              const Vec2<Coord>& frequency_range,
                              bool frequency_range_endpoint,
                              const Vec2<Coord>& angle_range,
                              bool angle_range_endpoint) noexcept {
        return Cartesian2PolarRFFT<Index, Offset, Interpolator, OutputValue>(
                cartesian, cartesian_shape, polar, polar_shape,
                frequency_range, frequency_range_endpoint,
                angle_range, angle_range_endpoint);
    }
}
