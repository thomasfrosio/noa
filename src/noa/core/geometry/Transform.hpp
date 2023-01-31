#pragma once

#include "noa/core/Assert.hpp"
#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"

// See docs/Usage.md for more details on the convention used for transformations.

// -- 2D transformations -- //
namespace noa::geometry {
    /// Returns a 2x2 HW scaling matrix.
    /// \param s HW scaling factors for each axis.
    template<typename T>
    NOA_IHD constexpr Mat22<T> scale(Vec2<T> s) noexcept {
        return Mat22<T>{s};
    }

    /// Returns the HW 2x2 rotation matrix describing an
    /// in-plane rotation by \p angle radians.
    template<typename T>
    NOA_IHD constexpr Mat22<T> rotate(T angle) noexcept {
        T c = noa::math::cos(angle);
        T s = noa::math::sin(angle);
        return {c, s,
                -s, c};
    }

    /// Returns the DHW 3x3 affine translation matrix encoding the
    /// HW translation \p shift, in elements.
    template<typename T>
    NOA_IHD constexpr Mat33<T> translate(Vec2<T> shift) noexcept {
        return {1, 0, shift[0],
                0, 1, shift[1],
                0, 0, 1};
    }
}

// -- 3D transformations -- //
namespace noa::geometry {
    /// Returns a DHW 3x3 scaling matrix.
    /// \param s DHW scaling factors for each axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> scale(Vec3<T> s) noexcept {
        return Mat33<T>{s};
    }

    /// Returns the DHW 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the outermost axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotate_z(T angle) noexcept {
        T c = noa::math::cos(angle);
        T s = noa::math::sin(angle);
        return {1, 0, 0,
                0, c, s,
                0, -s, c};
    }

    /// Returns the DHW 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the second-most axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotate_y(T angle) noexcept {
        T c = noa::math::cos(angle);
        T s = noa::math::sin(angle);
        return {c, 0, -s,
                0, 1, 0,
                s, 0, c};
    }

    /// Returns the DHW 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the innermost axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotate_x(T angle) noexcept {
        T c = noa::math::cos(angle);
        T s = noa::math::sin(angle);
        return {c, s, 0,
                -s, c, 0,
                0, 0, 1};
    }

    /// Returns a DHW 3x3 matrix describing a rotation by an \p angle around a given \p axis.
    /// \param axis     Normalized axis, using the rightmost {Z,Y,X} coordinates.
    /// \param angle    Rotation angle, in radians.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotate(Vec3<T> axis, T angle) noexcept {
        // see https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
        NOA_ASSERT(noa::math::are_almost_equal(noa::math::norm(axis), static_cast<T>(1))); // axis should be normalized.

        T c = noa::math::cos(static_cast<T>(angle));
        T s = noa::math::sin(static_cast<T>(angle));
        T t = 1 - c;
        return {axis[0] * axis[0] * t + c,
                axis[1] * axis[0] * t + axis[2] * s,
                axis[2] * axis[0] * t - axis[1] * s,

                axis[1] * axis[0] * t - axis[2] * s,
                axis[1] * axis[1] * t + c,
                axis[2] * axis[1] * t + axis[0] * s,

                axis[2] * axis[0] * t + axis[1] * s,
                axis[2] * axis[1] * t - axis[0] * s,
                axis[2] * axis[2] * t + c};
    }

    /// Returns a DHW 4x4 affine translation matrix encoding the
    /// DHW translation \p shift, in elements.
    template<typename T>
    NOA_IHD constexpr Mat44<T> translate(Vec3<T> shift) noexcept {
        return {1, 0, 0, shift[0],
                0, 1, 0, shift[1],
                0, 0, 1, shift[2],
                0, 0, 0, 1};
    }
}
