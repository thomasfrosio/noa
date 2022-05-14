#pragma once

#include "noa/common/Types.h"

namespace noa::geometry {
    /// Returns the magnitude at a polar mapped coordinate.
    /// \tparam T           Any floating-point.
    /// \param coordinate   Radial coordinate rho. Should be withing [0,\p size].
    /// \param size         Size of the innermost dimension of the polar grid.
    /// \param radius_range Radius [start,end) range of the bounding circle, in pixels.
    /// \param log          Whether this is the log-polar coordinates system.
    template<typename T>
    [[nodiscard]] T rho2magnitude(T coordinate, size_t size, Float2<T> radius_range, bool log) {
        NOA_ASSERT(radius_range[1] - radius_range[0] > 1);
        const T size_ = static_cast<T>(size);
        if (log) {
            NOA_ASSERT(coordinate > 0);
            const T step = math::log(radius_range[1] - radius_range[0]) / size_;
            return math::exp(coordinate * step) - 1 + radius_range[0];
        } else {
            const T step = (radius_range[1] - radius_range[0]) / size_;
            return coordinate * step + radius_range[0];
        }
    }

    /// Returns the angle, in radians, at a polar mapped coordinate.
    /// \tparam T           Any floating-point.
    /// \param coordinate   Angle coordinate phi. Should be withing [0,\p size].
    /// \param size         Size of the second-most dimension of the polar grid.
    /// \param angle_range  Angle [start,end) range, in radians, increasing in the counterclockwise orientation
    ///                     (i.e. unit circle), onto which the polar grid was mapped.
    template<typename T>
    [[nodiscard]] T phi2angle(T coordinate, size_t size, Float2<T> angle_range) {
        NOA_ASSERT(math::abs(angle_range[1] - angle_range[0]) > 1e-6);
        const T step = math::abs(angle_range[1] - angle_range[0]) / static_cast<T>(size);
        return coordinate * step + angle_range[0];
    }

    /// Returns the original cartesian coordinate from a polar mapped coordinate: (phi,rho) -> (y,x).
    /// \param coordinate       Rightmost polar coordinates (phi, rho).
    /// \param size             Rightmost shape of the polar grid.
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
    [[nodiscard]] Float2<T> polar2cartesian(Float2<T> coordinate, size2_t shape, Float2<T> cartesian_center,
                                            Float2<T> radius_range, Float2<T> angle_range, bool log) {
        const T angle_rad = phi2angle(coordinate[0], shape[0], angle_range);
        const T magnitude = rho2magnitude(coordinate[1], shape[1], radius_range, log);
        return {cartesian_center[0] + magnitude * math::sin(angle_rad),
                cartesian_center[1] + magnitude * math::cos(angle_rad)};
    }
}
