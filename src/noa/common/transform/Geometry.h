/// \file noa/common/Geometry.h
/// \brief Basic geometry operations.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2020

#pragma once

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

// Links:
//  - https://rock-learning.github.io/pytransform3d/transformation_ambiguities.html
//
// Conventions:
//  - Transformations are active (alibi), i.e. body rotates about the origin of the coordinate system.
//  - Transformations assume a right-handed coordinate system.
//  - Angles are given in radians by default.
//  - Positive angles specify a counter-clockwise rotation, when looking at the origin from a positive point.
//  - Positive translations specify a translation to the right, when looking at the origin from a positive point.
//  - Rotation/affine matrices pre-multiply column vectors to produce transformed column vectors: M * v = v'
//
// Transforming coordinates to then interpolate:
//  If the coordinates are the query of interpolation (output -> input), the transformation should be inverted.
//  Instead of computing the inverse of the rotation matrix (affine or not), we can simply:
//      - take the transpose of the 2x2 or 3x3 rotation matrix, which is equivalent to invert a pure rotation.
//        Note that if the rotation matrix has a determinant != 1, i.e. it has a scaling != 1, transpose != inverse.
//      - negate the translation, which is equivalent to invert a pure 3x3 or 4x4 (affine) translation matrix.
//      - invert the scaling values (1/scalar), which is equivalent to invert a pure 2x2 or 3x3 scaling matrix.
//
// Chaining multiple transformations:
//  Since we pre-multiply column vectors, the order of the transformations goes from right to left,
//  e.g. A = T * R * S, scales, rotates then translates. However, as mentioned above, if we perform the inverse
//  transformation, that is, transforming the output coordinates to the input frame to then interpolate input data,
//  we actually want the inverse transformation, i.e. inverse(A). Since inverting a 3x3 or 4x4 affine matrix is
//  "expensive", we can instead invert the individual transformations and revert the order: inverse(T * R * S) is
//  equivalent to inverse(S) * inverse(R) * inverse(T). Note that inverting pure transformations is trivial,
//  as explained above.

namespace noa::transform {
    // -- 3D transformations --

    /// Returns a 3x3 scaling matrix.
    /// \tparam T   float or double.
    /// \param s    Scaling factors. One per axis.
    /// \return     The 3x3 matrix.
    template<typename T>
    NOA_IHD constexpr noa::Mat33<T> scale(const noa::Float3<T>& s) noexcept {
        return noa::Mat33<T>(s);
    }

    /// Returns a 3x3 matrix describing a rotation by an \a angle (in radians) around a given \a axis.
    /// \warning @a axis should be normalized, see math::normalize().
    /// \tparam T       float or double
    /// \param axis     Normalized vector with the {X,Y,Z} coordinates of the axis.
    /// \param angle    Rotation angle, in radians.
    /// \return
    template<typename T>
    NOA_IHD constexpr noa::Mat33<T> rotate(const noa::Float3<T>& axis, T angle) noexcept {
        // see https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
        NOA_ASSERT(math::isEqual(math::length(axis), static_cast<T>(1))); // axis should be normalized.

        T c = math::cos(static_cast<T>(angle));
        T s = math::sin(static_cast<T>(angle));
        noa::Float3<T> tmp((static_cast<T>(1) - c) * axis);

        return noa::Mat33<T>(axis.x * tmp[0] + c,
                             axis.x * tmp[1] - axis.z * s,
                             axis.x * tmp[2] + axis.y * s,
                             axis.y * tmp[0] + axis.z * s,
                             axis.y * tmp[1] + c,
                             axis.y * tmp[2] - axis.x * s,
                             axis.z * tmp[0] - axis.y * s,
                             axis.z * tmp[1] + axis.x * s,
                             axis.z * tmp[2] + c);
    }

    /// Returns a 3x3 matrix describing rotation by @a angle radians
    /// around an in-plane axis, i.e. the XY plane, at \a axis_angle radians.
    /// \tparam T           float or double
    /// \param axis_angle   In-plane angle of the axis.
    /// \param angle        Angle of rotation around the in-plane axis.
    /// \return             3x3 rotation matrix.
    template<typename T>
    NOA_IHD constexpr noa::Mat33<T> rotateInPlane(T axis_angle, T angle) noexcept {
        return rotate(noa::Float3<T>(math::cos(axis_angle), math::sin(axis_angle), 0), angle);
    }

    /// Returns the 3x3 rotation matrix describing the rotation by \a angle around the X axis.
    /// \tparam T       float or double.
    /// \param angle    Angle in radians.
    /// \return         3x3 rotation matrix.
    template<typename T>
    NOA_IHD constexpr noa::Mat33<T> rotateX(T angle) noexcept {
        T c = math::cos(angle);
        T s = math::sin(angle);
        return noa::Mat33<T>(1, 0, 0, 0, c, -s, 0, s, c);
    }

    /// Returns the 3x3 rotation matrix describing the rotation by \a angle around the Y axis.
    /// \tparam T       float or double.
    /// \param angle    Angle in radians.
    /// \return         3x3 rotation matrix.
    template<typename T>
    NOA_IHD constexpr noa::Mat33<T> rotateY(T angle) noexcept {
        T c = math::cos(angle);
        T s = math::sin(angle);
        return noa::Mat33<T>(c, 0, s, 0, 1, 0, -s, 0, c);
    }

    /// Returns the 3x3 rotation matrix describing the rotation by \a angle around the Z axis.
    /// \tparam T       float or double.
    /// \param angle    Angle in radians.
    /// \return         3x3 rotation matrix.
    template<typename T>
    NOA_IHD constexpr noa::Mat33<T> rotateZ(T angle) noexcept {
        T c = math::cos(angle);
        T s = math::sin(angle);
        return noa::Mat33<T>(c, -s, 0, s, c, 0, 0, 0, 1);
    }

    /// Returns a 4x4 translation matrix for a 3D affine transform.
    /// \tparam T   float or double.
    /// \param v    {X, Y, Z} translations.
    /// \return     4x4 affine matrix.
    template<typename T>
    NOA_IHD constexpr noa::Mat44<T> translate(const noa::Float3<T>& v) noexcept {
        return noa::Mat44<T>(1, 0, 0, v.x,
                             0, 1, 0, v.y,
                             0, 0, 1, v.z,
                             0, 0, 0, 1);
    }

    // -- 2D transformations --

    /// Returns a 2x2 scaling matrix.
    /// \tparam T   float or double.
    /// \param s    Scaling factors. One per axis.
    /// \return     The 2x2 matrix.
    template<typename T>
    NOA_IHD constexpr noa::Mat22<T> scale(const noa::Float2<T>& s) noexcept {
        return noa::Mat22<T>(s);
    }

    /// Returns the 2x2 rotation matrix describing the in-plane rotation by \a angle radians.
    /// \tparam T       float or double.
    /// \param angle    Angle in radians.
    /// \return         2x2 rotation matrix.
    template<typename T>
    NOA_IHD constexpr noa::Mat22<T> rotate(T angle) noexcept {
        T c = math::cos(angle);
        T s = math::sin(angle);
        return noa::Mat22<T>(c, -s, s, c);
    }

    /// Returns a 3x3 translation matrix for a 2D affine transform.
    /// \tparam T   float or double.
    /// \param v    {X, Y} translations.
    /// \return     3x3 affine matrix.
    template<typename T>
    NOA_IHD constexpr noa::Mat33<T> translate(const noa::Float2<T>& v) noexcept {
        return noa::Mat33<T>(1, 0, v.x,
                             0, 1, v.y,
                             0, 0, 1);
    }
}
