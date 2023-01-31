/// \file noa/core/geometry/Transform.h
/// \brief Basic geometry operations.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2020

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
    NOA_IHD constexpr Mat22<T> scale(Float2<T> s) noexcept {
        return Mat22<T>{s};
    }

    /// Returns the HW 2x2 rotation matrix describing an
    /// in-plane rotation by \p angle radians.
    template<typename T>
    NOA_IHD constexpr Mat22<T> rotate(T angle) noexcept {
        T c = math::cos(angle);
        T s = math::sin(angle);
        return {c, s,
                -s, c};
    }

    /// Returns the DHW 3x3 affine translation matrix encoding the
    /// HW translation \p shift, in elements.
    template<typename T>
    NOA_IHD constexpr Mat33<T> translate(Float2<T> shift) noexcept {
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
    NOA_IHD constexpr Mat33<T> scale(Float3<T> s) noexcept {
        return Mat33<T>{s};
    }

    /// Returns the DHW 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the outermost axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotateZ(T angle) noexcept {
        T c = math::cos(angle);
        T s = math::sin(angle);
        return {1, 0, 0,
                0, c, s,
                0, -s, c};
    }

    /// Returns the DHW 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the second-most axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotateY(T angle) noexcept {
        T c = math::cos(angle);
        T s = math::sin(angle);
        return {c, 0, -s,
                0, 1, 0,
                s, 0, c};
    }

    /// Returns the DHW 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the innermost axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotateX(T angle) noexcept {
        T c = math::cos(angle);
        T s = math::sin(angle);
        return {c, s, 0,
                -s, c, 0,
                0, 0, 1};
    }

    /// Returns a DHW 3x3 matrix describing a rotation by an \p angle around a given \p axis.
    /// \tparam T       float or double
    /// \param axis     Normalized axis, using the rightmost {Z,Y,X} coordinates.
    /// \param angle    Rotation angle, in radians.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotate(Float3<T> axis, T angle) noexcept {
        // see https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
        NOA_ASSERT(math::isEqual(math::length(axis), static_cast<T>(1))); // axis should be normalized.

        T c = math::cos(static_cast<T>(angle));
        T s = math::sin(static_cast<T>(angle));
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
    NOA_IHD constexpr Mat44<T> translate(Float3<T> shift) noexcept {
        return {1, 0, 0, shift[0],
                0, 1, 0, shift[1],
                0, 0, 1, shift[2],
                0, 0, 0, 1};
    }
}
