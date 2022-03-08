/// \file noa/common/geometry/Transform.h
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
//  - Rotation/affine matrices pre-multiply rightmost (i.e. {Z,Y,X}) column vectors, to produce transformed
//    column vectors: M * v = v'.
//
// Transforming coordinates to then interpolate:
//  If the coordinates are the query of interpolation, we are often talking about an inverse transformation,
//  i.e. we go from the coordinates in the output reference frame to the coordinates of in the input reference frame.
//  Instead of computing the inverse of the rotation matrix (affine or not), we can simply:
//      - take the transpose of the 2x2 or 3x3 rotation matrix, which is equivalent to invert a pure rotation.
//        Note that if the rotation matrix has a determinant != 1, i.e. it has a scaling != 1, transpose != inverse.
//      - negate the translation, which is equivalent to invert a pure 3x3 or 4x4 (affine) translation matrix.
//      - invert the scaling values (1/scalar), which is equivalent to invert a pure 2x2 or 3x3 scaling matrix.
//
// Chaining multiple transformations:
//  Since we pre-multiply column vectors, the order of the transformations goes from right to left,
//  e.g. A = T * R * S, scales, rotates then translates. However, as mentioned above, if we perform the inverse
//  transformation, the inverse matrix, i.e. inverse(A), is needed. Since inverting a 3x3 or 4x4 affine matrix
//  is "expensive", we can instead invert the individual transformations and revert the order: inverse(T * R * S) is
//  equivalent to inverse(S) * inverse(R) * inverse(T). Note that inverting pure transformations is trivial,
//  as explained above. As such, when matrices are passed directly, they are assumed to be already inverted, unless
//  specified otherwise.
//
// Left-most vs right-most order:
//  Right-most ordering denotes the C/C++ standard multidimensional array index mapping where the right-most index
//  is stride one and strides increase right-to-left as the product of indexes. This is often referred to as row
//  major (as opposed to column major). The library uses right-most indexes, and as such, our axes are always specified
//  in the right-most order: {Z,Y,X}, where Z is the outermost axis and X is the innermost one. This should be quite
//  familiar to people using NumPy. This convention is uniformly applied across the library for vectors (e.g. shapes,
//  stride, shift, scale), including all matrices, which should be pre-multiplied by rightmost {Z,Y,X} column vectors.

// -- 2D transformations -- //
namespace noa::geometry {
    /// Returns a 2x2 scaling matrix.
    /// \param s Scaling factors for each axis.
    template<typename T>
    NOA_IHD constexpr Mat22<T> scale(Float2<T> s) noexcept {
        return Mat22<T>{s};
    }

    /// Returns the rightmost 2x2 rotation matrix describing an
    /// in-plane rotation by \p angle radians.
    template<typename T>
    NOA_IHD constexpr Mat22<T> rotate(T angle) noexcept {
        T c = math::cos(angle);
        T s = math::sin(angle);
        return {c, s,
                -s, c};
    }

    /// Returns the rightmost 3x3 affine translation matrix encoding the
    /// rightmost translation \p shift, in elements.
    template<typename T>
    NOA_IHD constexpr Mat33<T> translate(Float2<T> shift) noexcept {
        return {1, 0, shift[0],
                0, 1, shift[1],
                0, 0, 1};
    }
}

// -- 3D transformations -- //
namespace noa::geometry {
    /// Returns a 3x3 scaling matrix.
    /// \param s Scaling factors for each axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> scale(Float3<T> s) noexcept {
        return Mat33<T>{s};
    }

    /// Returns the rightmost 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the outermost axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotateZ(T angle) noexcept {
        T c = math::cos(angle);
        T s = math::sin(angle);
        return {1, 0, 0,
                0, c, s,
                0, -s, c};
    }

    /// Returns the rightmost 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the second-most axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotateY(T angle) noexcept {
        T c = math::cos(angle);
        T s = math::sin(angle);
        return {c, 0, -s,
                0, 1, 0,
                s, 0, c};
    }

    /// Returns the rightmost 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the innermost axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotateX(T angle) noexcept {
        T c = math::cos(angle);
        T s = math::sin(angle);
        return {c, s, 0,
                -s, c, 0,
                0, 0, 1};
    }

    /// Returns a rightmost 3x3 matrix describing a rotation by an \p angle around a given \p axis.
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

    /// Returns a rightmost 4x4 affine translation matrix encoding the
    /// rightmost translation \p shift, in elements.
    template<typename T>
    NOA_IHD constexpr Mat44<T> translate(Float3<T> shift) noexcept {
        return {1, 0, 0, shift[0],
                0, 1, 0, shift[1],
                0, 0, 1, shift[2],
                0, 0, 0, 1};
    }
}
