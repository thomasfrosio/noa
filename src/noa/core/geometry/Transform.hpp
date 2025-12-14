#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Mat.hpp"
#include "noa/core/math/Generic.hpp"

namespace noa::geometry {
    /// Returns a 2x2 HW scaling matrix.
    /// \param s HW scaling factors for each axis.
    template<bool AFFINE = false, nt::real T, usize A>
    constexpr auto scale(const Vec<T, 2, A>& s) -> Mat<T, 2 + AFFINE, 2 + AFFINE> {
        if constexpr (AFFINE)
            return Mat33<T>::from_diagonal(s.push_back(1));
        else
            return Mat22<T>::from_diagonal(s);
    }

    /// Returns the HW 2x2 rotation matrix describing an
    /// in-plane rotation by \p angle radians.
    template<bool AFFINE = false, nt::real T>
    constexpr auto rotate(T angle) -> Mat<T, 2 + AFFINE, 2 + AFFINE> {
        T c = cos(angle);
        T s = sin(angle);
        if constexpr (AFFINE) {
            return {{{ c, s, 0},
                     {-s, c, 0},
                     { 0, 0, 1}}};
        } else {
            return {{{ c, s},
                     {-s, c}}};
        }
    }

    /// Returns the DHW 3x3 affine translation matrix encoding the
    /// HW translation \p shift, in elements.
    template<nt::real T, usize A = 0>
    constexpr auto translate(const Vec<T, 2, A>& shift) -> Mat33<T> {
        return {{{1, 0, shift[0]},
                 {0, 1, shift[1]},
                 {0, 0, 1}}};
    }

    template<typename T, usize A = 0>
    constexpr auto linear2affine(const Mat22<T>& linear, const Vec<T, 2, A>& translate = {}) -> Mat33<T> {
        return {{{linear[0][0], linear[0][1], translate[0]},
                 {linear[1][0], linear[1][1], translate[1]},
                 {0, 0, 1}}};
    }

    template<typename T, usize A = 0>
    constexpr auto linear2truncated(const Mat22<T>& linear, const Vec<T, 2, A>& translate = {}) -> Mat23<T> {
        return {{{linear[0][0], linear[0][1], translate[0]},
                 {linear[1][0], linear[1][1], translate[1]}}};
    }

    template<typename T>
    constexpr auto truncated2affine(const Mat23<T>& truncated) -> Mat33<T> {
        return {{{truncated[0][0], truncated[0][1], truncated[0][2]},
                 {truncated[1][0], truncated[1][1], truncated[1][2]},
                 {0, 0, 1}}};
    }

    template<typename T>
    constexpr auto affine2linear(const Mat33<T>& affine) -> Mat22<T> {
        return {{{affine[0][0], affine[0][1]},
                 {affine[1][0], affine[1][1]}}};
    }

    template<typename T>
    constexpr auto truncated2linear(const Mat23<T>& truncated) -> Mat22<T> {
        return {{{truncated[0][0], truncated[0][1]},
                 {truncated[1][0], truncated[1][1]}}};
    }

    template<typename T>
    constexpr auto affine2truncated(const Mat33<T>& affine) -> Mat23<T> {
        return {{{affine[0][0], affine[0][1], affine[0][2]},
                 {affine[1][0], affine[1][1], affine[1][2]}}};
    }
}

namespace noa::geometry {
    /// Returns a DHW 3x3 scaling matrix.
    /// \param s DHW scaling factors for each axis.
    template<bool AFFINE = false, nt::real T, usize A>
    constexpr auto scale(const Vec<T, 3, A>& s) -> Mat<T, 3 + AFFINE, 3 + AFFINE> {
        if constexpr (AFFINE)
            return Mat44<T>::from_diagonal(s.push_back(1));
        else
            return Mat33<T>::from_diagonal(s);
    }

    /// Returns the DHW 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the outermost axis.
    template<bool AFFINE = false, nt::real T>
    constexpr auto rotate_z(T angle) -> Mat<T, 3 + AFFINE, 3 + AFFINE> {
        T c = cos(angle);
        T s = sin(angle);
        if constexpr (AFFINE) {
            return {{{1, 0, 0, 0},
                     {0, c, s, 0},
                     {0,-s, c, 0},
                     {0, 0, 0, 1}}};
        } else {
            return {{{1, 0, 0},
                     {0, c, s},
                     {0,-s, c}}};
        }
    }

    /// Returns the DHW 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the second-most axis.
    template<bool AFFINE = false, nt::real T>
    constexpr auto rotate_y(T angle) -> Mat<T, 3 + AFFINE, 3 + AFFINE> {
        T c = cos(angle);
        T s = sin(angle);
        if constexpr (AFFINE) {
            return {{{c, 0,-s, 0},
                     {0, 1, 0, 0},
                     {s, 0, c, 0},
                     {0, 0, 0, 1}}};
        } else {
            return {{{c, 0,-s},
                     {0, 1, 0},
                     {s, 0, c}}};
        }
    }

    /// Returns the DHW 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the innermost axis.
    template<bool AFFINE = false, nt::real T>
    constexpr auto rotate_x(T angle) -> Mat<T, 3 + AFFINE, 3 + AFFINE> {
        T c = cos(angle);
        T s = sin(angle);
        if constexpr (AFFINE) {
            return {{{ c, s, 0, 0},
                     {-s, c, 0, 0},
                     { 0, 0, 1, 0},
                     { 0, 0, 0, 1}}};
        } else {
            return {{{ c, s, 0},
                     {-s, c, 0},
                     { 0, 0, 1}}};
        }
    }

    /// Returns a DHW 3x3 matrix describing a rotation by an \p angle around a given \p axis.
    /// \param axis     Normalized axis, using the rightmost {Z,Y,X} coordinates.
    /// \param angle    Rotation angle, in radians.
    template<nt::real T, usize A>
    constexpr auto rotate(const Vec<T, 3, A>& axis, T angle) -> Mat33<T> {
        // see https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
        NOA_ASSERT(allclose(norm(axis), static_cast<T>(1))); // axis should be normalized.

        T c = cos(static_cast<T>(angle));
        T s = sin(static_cast<T>(angle));
        T t = 1 - c;
        return {{{axis[0] * axis[0] * t + c,
                  axis[1] * axis[0] * t + axis[2] * s,
                  axis[2] * axis[0] * t - axis[1] * s},
                 {axis[1] * axis[0] * t - axis[2] * s,
                  axis[1] * axis[1] * t + c,
                  axis[2] * axis[1] * t + axis[0] * s},
                 {axis[2] * axis[0] * t + axis[1] * s,
                  axis[2] * axis[1] * t - axis[0] * s,
                  axis[2] * axis[2] * t + c}}};
    }

    /// Returns a DHW 4x4 affine translation matrix encoding the
    /// DHW translation \p shift, in elements.
    template<nt::real T, usize A>
    constexpr auto translate(const Vec<T, 3, A>& shift) -> Mat44<T> {
        return {{{1, 0, 0, shift[0]},
                 {0, 1, 0, shift[1]},
                 {0, 0, 1, shift[2]},
                 {0, 0, 0, 1}}};
    }

    template<nt::real T, usize A = 0>
    constexpr auto linear2affine(const Mat33<T>& linear, const Vec<T, 3, A>& translate = {}) -> Mat44<T> {
        return {{{linear[0][0], linear[0][1], linear[0][2], translate[0]},
                 {linear[1][0], linear[1][1], linear[1][2], translate[1]},
                 {linear[2][0], linear[2][1], linear[2][2], translate[2]},
                 {0, 0, 0, 1}}};
    }

    template<nt::real T, usize A = 0>
    constexpr auto linear2truncated(const Mat33<T>& linear, const Vec<T, 3, A>& translate = {}) -> Mat34<T> {
        return {{{linear[0][0], linear[0][1], linear[0][2], translate[0]},
                 {linear[1][0], linear[1][1], linear[1][2], translate[1]},
                 {linear[2][0], linear[2][1], linear[2][2], translate[2]}}};
    }

    template<typename T>
    constexpr auto truncated2affine(const Mat34<T>& truncated) -> Mat44<T> {
        return {{{truncated[0][0], truncated[0][1], truncated[0][2], truncated[0][3]},
                 {truncated[1][0], truncated[1][1], truncated[1][2], truncated[1][3]},
                 {truncated[2][0], truncated[2][1], truncated[2][2], truncated[2][3]},
                 {0, 0, 0, 1}}};
    }

    template<typename T>
    constexpr auto affine2linear(const Mat44<T>& affine) -> Mat33<T> {
        return {{{affine[0][0], affine[0][1], affine[0][2]},
                 {affine[1][0], affine[1][1], affine[1][2]},
                 {affine[2][0], affine[2][1], affine[2][2]}}};
    }

    template<typename T>
    constexpr auto truncated2linear(const Mat34<T>& truncated) -> Mat33<T> {
        return {{{truncated[0][0], truncated[0][1], truncated[0][2]},
                 {truncated[1][0], truncated[1][1], truncated[1][2]},
                 {truncated[2][0], truncated[2][1], truncated[2][2]}}};
    }

    template<typename T>
    constexpr auto affine2truncated(const Mat44<T>& affine) -> Mat34<T> {
        return {{{affine[0][0], affine[0][1], affine[0][2], affine[0][3]},
                 {affine[1][0], affine[1][1], affine[1][2], affine[1][3]},
                 {affine[2][0], affine[2][1], affine[2][2], affine[2][3]}}};
    }
}

namespace noa::geometry {
    /// Transform the vector.
    /// \param xform    Transform operator.
    /// \param vector   Coordinate. The non-homogeneous vector should be passed.
    template<nt::any_of<f32, f64> T,
             usize N, usize A,
             typename X>
    requires (N == 2 or N == 3)
    constexpr auto transform_vector(
        const X& xform,
        const Vec<T, N, A>& vector
    ) -> Vec<T, N, A> {
        if constexpr (nt::mat_of_shape<X, N, N>) {
            return xform * vector;
        } else if constexpr (nt::mat_of_shape<X, N, N + 1>) {
            return xform * vector.push_back(1); // truncated
        } else if constexpr (nt::mat_of_shape<X, N + 1, N + 1>) {
            return affine2truncated(xform) * vector.push_back(1); // affine
        } else if constexpr (N == 3 and nt::quaternion<X>) {
            return xform.rotate(vector);
        } else if constexpr (nt::empty<X>) {
            return vector;
        } else {
            static_assert(nt::always_false<X>);
        }
    }
}
