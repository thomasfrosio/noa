#pragma once

#include "noa/core/Linspace.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Vec.hpp"

/// Convention:
///  - Cartesian: x, y(, z)
///  - Polar: ρ (radial distance from origin), φ (elevation angle from x to y, in [-pi,pi])
///  - Spherical: ρ (same as polar), φ (same as polar), θ (inclination angle from z to polar axis, in [0,pi])
///
/// The polar grid can map a reduced range of the cartesian grid, hence the function that converts
/// (ρ, φ) to/from the polar coordinates (e.g. polar2rho).

// Polar -> Cartesian
namespace noa::geometry {
    /// Returns the magnitude rho at a polar mapped coordinate.
    /// \tparam T               Any floating-point.
    /// \param polar_coordinate Radial polar coordinate. Should be withing [0,\p polar_size].
    /// \param polar_size       Size of the radial dimension (usually the height) of the polar grid.
    /// \param radius_range     Radius [start,end] range of the bounding circle, in pixels.
    /// \param log              Whether this is the log-polar coordinates system.
    template<typename T>
    [[nodiscard]] NOA_IHD T polar2rho(T polar_coordinate, i64 polar_size, Vec2<T> radius_range, bool log) {
        NOA_ASSERT(radius_range[1] - radius_range[0] > 1);
        const T effective_size = static_cast<T>(polar_size - 1);
        if (log) {
            const T step = log(radius_range[1] - radius_range[0]) / effective_size;
            return exp(polar_coordinate * step) - 1 + radius_range[0];
        } else {
            const T step = (radius_range[1] - radius_range[0]) / effective_size;
            return polar_coordinate * step + radius_range[0];
        }
    }

    /// Returns the angle phi, in radians, at a polar mapped coordinate.
    /// \tparam T           Any floating-point.
    /// \param coordinate   Angle polar coordinate. Should be withing [0,\p size].
    /// \param size         Size of the angle dimension (usually the width) of the polar grid.
    /// \param angle_range  Angle [start,end] range, in radians, of the bounding (truncated)-circle.
    ///                     Increases in the counterclockwise orientation (i.e. unit circle).
    template<typename T>
    [[nodiscard]] NOA_IHD T polar2phi(T coordinate, i64 size, Vec2<T> angle_range) {
        const T step = (angle_range[1] - angle_range[0]) / static_cast<T>(size - 1);
        return coordinate * step + angle_range[0];
    }

    /// Returns the original cartesian coordinate from a polar mapped coordinate: (phi,rho) -> (y,x).
    /// \param polar_coordinate Rightmost polar coordinates (phi, rho).
    /// \param polar_shape      Rightmost shape of the polar grid.
    /// \param cartesian_center Rightmost center of transformation in the cartesian space.
    /// \param radius_range     Radius [start,end] range of the bounding circle, in pixels.
    /// \param angle_range      Angle [start,end] range, in radians, of the bounding (truncated)-circle.
    ///                         Increases in the counterclockwise orientation (i.e. unit circle).
    /// \param log              Whether this is the log-polar coordinates system.
    ///
    /// \note A great explanation and representation of the cartesian to polar transformation is
    ///       available at https://docs.opencv.org/3.4 in warpPolar (imgproc module). Here, the radius
    ///       range is explicitly defined on both ends (they assume the starting radius is 0), the
    ///       angle range is also explicitly defined (they assume 0 -> 2pi). Also, we use a counterclockwise
    ///       rotation, as always.
    template<typename T>
    [[nodiscard]] NOA_IHD auto polar2cartesian(
            Vec2<T> polar_coordinate, Vec2<i64> polar_shape, Vec2<T> cartesian_center,
            Vec2<T> radius_range, Vec2<T> angle_range, bool log
    ) -> Vec2<T> {
        const T phi = polar2phi(polar_coordinate[0], polar_shape[0], angle_range);
        const T rho = polar2rho(polar_coordinate[1], polar_shape[1], radius_range, log);
        return {cartesian_center[0] + rho * sin(phi),
                cartesian_center[1] + rho * cos(phi)};
    }

    /// Returns the (y,x) coordinates corresponding to the polar coordinates (rho, phi[0,2pi]).
    template<typename T>
    [[nodiscard]] NOA_IHD Vec2<T> polar2cartesian(T rho, T phi) {
        return {rho * sin(phi), rho * cos(phi)};
    }

    /// Returns the (z,y,x) coordinates corresponding to the spherical coordinates (rho, phi[0,2pi], theta[0,pi]).
    template<typename T>
    [[nodiscard]] NOA_IHD Vec3<T> spherical2cartesian(T rho, T phi, T theta) {
        return {rho * cos(theta),
                rho * sin(phi) * sin(theta),
                rho * cos(phi) * sin(theta)};
    }
}

// Cartesian -> Polar
namespace noa::geometry {
    /// Returns the magnitude rho of a cartesian \p coordinate.
    template<typename T, size_t N>
    requires (nt::is_real_v<T> and (N == 2 or N == 3))
    [[nodiscard]] NOA_IHD auto cartesian2rho(const Vec<T, N>& cartesian_coordinate) {
        return norm(cartesian_coordinate);
    }

    /// Maps rho to the radial polar coordinate.
    /// \tparam T           Any floating-point.
    /// \param rho          Magnitude rho.
    /// \param polar_size   Size of the radial dimension (usually the height) of polar grid.
    /// \param radius_range Radius [start,end] range of the bounding circle, in pixels.
    /// \param log          Whether this is the log-polar coordinates system.
    template<typename T>
    [[nodiscard]] NOA_IHD T rho2polar(T rho, i64 polar_size, Vec2<T> radius_range, bool log) {
        const T effective_size = static_cast<T>(polar_size - 1);
        if (log) {
            const T step = log(radius_range[1] - radius_range[0]) / effective_size;
            return log(rho + 1 - radius_range[0]) / step;
        } else {
            const T step = (radius_range[1] - radius_range[0]) / effective_size;
            return (rho - radius_range[0]) / step;
        }
    }

    /// Returns the phi angle of the (y,x) cartesian \p coordinate.
    /// If \p OFFSET, the returned values is between [0,2pi], otherwise [-pi,pi].
    template<bool OFFSET = true, typename T>
    [[nodiscard]] NOA_IHD T cartesian2phi(Vec2<T> coordinate) {
        T angle = atan2(coordinate[0], coordinate[1]); // [-pi,pi]
        if (OFFSET and angle < 0)
            angle += Constant<T>::PI * 2; // [0,2pi]
        return angle;
    }

    /// Returns the phi angle of the (z,y,x) cartesian \p coordinate.
    /// If \p OFFSET, the returned values is between [0,2pi], otherwise [-pi,pi].
    template<bool OFFSET = true, typename T>
    [[nodiscard]] NOA_IHD T cartesian2phi(Vec3<T> coordinate) {
        T angle = atan2(coordinate[1], coordinate[2]); // [-pi,pi]
        if (OFFSET and angle < 0)
            angle += Constant<T>::PI * 2; // [0,2pi]
        return angle;
    }

    /// Returns the theta angle [0,pi] of the (z,y,x) cartesian \p coordinate.
    template<typename T>
    [[nodiscard]] NOA_IHD T cartesian2theta(Vec3<T> coordinate) {
        T angle = atan2(hypot(coordinate[1], coordinate[2]), coordinate[0]); // [0,pi]
        return angle;
    }

    /// Maps phi to the angle polar coordinate.
    /// \tparam T           Any floating-point.
    /// \param phi          Phi, from [0,2pi].
    /// \param polar_size   Size of the angle dimension (usually the width) of polar grid.
    /// \param angle_range  Angle [start,end] range, in radians, of the bounding (truncated)-circle.
    ///                     Increases in the counterclockwise orientation (i.e. unit circle).
    template<typename T>
    [[nodiscard]] NOA_IHD T phi2polar(T phi, i64 polar_size, Vec2<T> angle_range) {
        const T effective_size = static_cast<T>(polar_size - 1);
        const T step_angle = (angle_range[1] - angle_range[0]) / effective_size;
        return (phi - angle_range[0]) / step_angle;
    }

    /// Returns the mapped polar HW coordinates for a given cartesian HW coordinate.
    /// \param cartesian_coordinate HW coordinates (y,x).
    /// \param cartesian_center     HW center of transformation in the cartesian space.
    /// \param polar_shape          HW shape of the polar grid.
    /// \param radius_range         Radius [start,end] range of the bounding circle, in pixels.
    /// \param angle_range          Angle [start,end] range, in radians, of the bounding (truncated)-circle.
    ///                             Increases in the counterclockwise orientation (i.e. unit circle).
    /// \param log                  Whether this is the log-polar coordinates system.
    template<typename T>
    [[nodiscard]] NOA_IHD auto cartesian2polar(
            Vec2<T> cartesian_coordinate, Vec2<T> cartesian_center,
            Vec2<i64> polar_shape, Vec2<T> radius_range,
            Vec2<T> angle_range, bool log
    ) -> Vec2<T> {
        cartesian_coordinate -= cartesian_center;
        const T phi = cartesian2phi(cartesian_coordinate);
        const T rho = cartesian2rho(cartesian_coordinate);
        return {phi2polar(phi, polar_shape[0], angle_range),
                rho2polar(rho, polar_shape[1], radius_range, log)};
    }
}

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
