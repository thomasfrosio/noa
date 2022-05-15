#pragma once

#include "noa/common/Types.h"

// Polar -> Cartesian
namespace noa::geometry {
    /// Returns the magnitude at a polar mapped coordinate.
    /// \tparam T               Any floating-point.
    /// \param polar_coordinate Radial coordinate rho. Should be withing [0,\p size].
    /// \param polar_size       Size of the innermost dimension of the polar grid.
    /// \param radius_range     Radius [start,end) range of the bounding circle, in pixels.
    /// \param log              Whether this is the log-polar coordinates system.
    template<typename T>
    [[nodiscard]] NOA_IHD T rho2magnitude(T polar_coordinate, size_t polar_size, Float2<T> radius_range, bool log) {
        NOA_ASSERT(radius_range[1] - radius_range[0] > 1);
        const T size_ = static_cast<T>(polar_size);
        if (log) {
            NOA_ASSERT(polar_coordinate > 0);
            const T step = math::log(radius_range[1] - radius_range[0]) / size_;
            return math::exp(polar_coordinate * step) - 1 + radius_range[0];
        } else {
            const T step = (radius_range[1] - radius_range[0]) / size_;
            return polar_coordinate * step + radius_range[0];
        }
    }

    /// Returns the angle, in radians, at a polar mapped coordinate.
    /// \tparam T           Any floating-point.
    /// \param coordinate   Angle coordinate phi. Should be withing [0,\p size].
    /// \param size         Size of the second-most dimension of the polar grid.
    /// \param angle_range  Angle [start,end) range, in radians, increasing in the counterclockwise orientation
    ///                     (i.e. unit circle), onto which the polar grid was mapped.
    template<typename T>
    [[nodiscard]] NOA_IHD T phi2angle(T coordinate, size_t size, Float2<T> angle_range) {
        NOA_ASSERT(math::abs(angle_range[1] - angle_range[0]) > 1e-6);
        const T step = math::abs(angle_range[1] - angle_range[0]) / static_cast<T>(size);
        return coordinate * step + angle_range[0];
    }

    /// Returns the original cartesian coordinate from a polar mapped coordinate: (phi,rho) -> (y,x).
    /// \param polar_coordinate Rightmost polar coordinates (phi, rho).
    /// \param polar_shape      Rightmost shape of the polar grid.
    /// \param cartesian_center Rightmost center of transformation in the cartesian space.
    /// \param radius_range     Radius [start,end) range of the bounding circle, in pixels.
    /// \param angle_range      Angle [start,end) range increasing in the counterclockwise orientation, in radians.
    /// \param log              Whether this is the log-polar coordinates system.
    ///
    /// \note A great explanation and representation of the cartesian to polar transformation is
    ///       available at https://docs.opencv.org/3.4 in warpPolar (imgproc module). Here, the radius
    ///       range is explicitly defined on both ends (they assume the starting radius is 0), the
    ///       angle range is also explicitly defined (they assume 0 -> 2pi). Also, we use a counterclockwise
    ///       rotation, as always.
    template<typename T>
    [[nodiscard]] NOA_IHD Float2<T>
    polar2cartesian(Float2<T> polar_coordinate, size2_t polar_shape, Float2<T> cartesian_center,
                    Float2<T> radius_range, Float2<T> angle_range, bool log) {
        const T angle_rad = phi2angle(polar_coordinate[0], polar_shape[0], angle_range);
        const T magnitude = rho2magnitude(polar_coordinate[1], polar_shape[1], radius_range, log);
        return {cartesian_center[0] + magnitude * math::sin(angle_rad),
                cartesian_center[1] + magnitude * math::cos(angle_rad)};
    }
}

// Cartesian -> Polar
namespace noa::geometry {
    /// Returns the magnitude of the rightmost \p coordinate.
    template<typename T>
    [[nodiscard]] NOA_IHD T cartesian2magnitude(Float2<T> cartesian_coordinate) {
        return math::sqrt(cartesian_coordinate[0] * cartesian_coordinate[0] +
                          cartesian_coordinate[1] * cartesian_coordinate[1]);
    }

    /// Returns the mapped radial polar coordinate rho for a given magnitude.
    /// \tparam T           Any floating-point.
    /// \param magnitude    Magnitude of the cartesian coordinate.
    /// \param polar_size   Shape of the innermost dimension of polar grid.
    /// \param radius_range Radius [start,end) range of the bounding circle, in pixels.
    /// \param log          Whether this is the log-polar coordinates system.
    template<typename T>
    [[nodiscard]] NOA_IHD T magnitude2rho(T magnitude, size_t polar_size, Float2<T> radius_range, bool log) {
        const T size_ = static_cast<T>(polar_size);
        if (log) {
            const T step = math::log(radius_range[1] - radius_range[0]) / size_;
            return math::log(magnitude + 1 - radius_range[0]) / step;
        } else {
            const T step = (radius_range[1] - radius_range[0]) / size_;
            return (magnitude - radius_range[0]) / step;
        }
    }

    /// Returns the [0,2pi] angle of the rightmost cartesian \p coordinate.
    template<typename T>
    [[nodiscard]] NOA_IHD T cartesian2angle(Float2<T> coordinate) {
        T angle = math::atan2(coordinate[0], coordinate[1]);
        if (angle < 0)
            angle += math::Constants<T>::PI2;
        return angle;
    }

    /// Returns the mapped angle polar coordinate phi for a given angle.
    /// \tparam T           Any floating-point.
    /// \param angle_rad    Angle [0,2pi] of the cartesian coordinate.
    /// \param polar_size   Shape of the second-most dimension of polar grid.
    /// \param angle_range  Angle [start,end) range increasing in the counterclockwise orientation, in radians.
    template<typename T>
    [[nodiscard]] NOA_IHD T angle2phi(T angle_rad, size_t polar_size, Float2<T> angle_range) {
        const T size_ = static_cast<T>(polar_size);
        const T step_angle = (angle_range[1] - angle_range[0]) / size_;
        return (angle_rad - angle_range[0]) / step_angle;
    }

    /// Returns the mapped polar coordinates (phi,rho) for a given cartesian coordinate.
    /// \tparam T
    /// \param cartesian_coordinate Rightmost coordinates (y,x).
    /// \param cartesian_center     Rightmost center of transformation in the cartesian space.
    /// \param polar_shape          Rightmost shape of the polar grid.
    /// \param radius_range         Radius [start,end) range of the bounding circle, in pixels.
    /// \param angle_range          Angle [start,end) range increasing in the counterclockwise orientation, in radians.
    /// \param log                  Whether this is the log-polar coordinates system.
    template<typename T>
    [[nodiscard]] NOA_IHD Float2<T>
    cartesian2polar(Float2<T> cartesian_coordinate, Float2<T> cartesian_center,
                    size2_t polar_shape,
                    Float2<T> radius_range, Float2<T> angle_range, bool log) {
        cartesian_coordinate -= cartesian_center;
        const T angle_rad = cartesian2angle(cartesian_coordinate);
        const T magnitude = cartesian2magnitude(cartesian_coordinate);
        return {angle2phi(angle_rad, polar_shape[0], angle_range),
                magnitude2rho(magnitude, polar_shape[1], radius_range, log)};
    }
}
