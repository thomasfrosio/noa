#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/types/Mat.hpp"
#include "noa/core/math/Generic.hpp"

// See docs/Usage.md for more details on the convention used for transformations.

// -- 2D transformations -- //
namespace noa::geometry {
    /// Returns a 2x2 HW scaling matrix.
    /// \param s HW scaling factors for each axis.
    template<typename T>
    NOA_IHD constexpr Mat22<T> scale(Vec2<T> s) noexcept {
        return Mat22<T>::from_diagonal(s);
    }

    /// Returns the HW 2x2 rotation matrix describing an
    /// in-plane rotation by \p angle radians.
    template<typename T>
    NOA_IHD constexpr Mat22<T> rotate(T angle) noexcept {
        T c = noa::cos(angle);
        T s = noa::sin(angle);
        return {{{c, s},
                 {-s, c}}};
    }

    /// Returns the DHW 3x3 affine translation matrix encoding the
    /// HW translation \p shift, in elements.
    template<typename T>
    NOA_IHD constexpr Mat33<T> translate(Vec2<T> shift) noexcept {
        return {{{1, 0, shift[0]},
                 {0, 1, shift[1]},
                 {0, 0, 1}}};
    }

    template<typename T>
    NOA_IHD constexpr Mat33<T> linear2affine(const Mat22<T>& linear, const Vec2<T>& translate = {}) noexcept {
        return {{{linear[0][0], linear[0][1], translate[0]},
                 {linear[1][0], linear[1][1], translate[1]},
                 {0, 0, 1}}};
    }

    template<typename T>
    NOA_IHD constexpr Mat23<T> linear2truncated(const Mat22<T>& linear, const Vec2<T>& translate = {}) noexcept {
        return {{{linear[0][0], linear[0][1], translate[0]},
                 {linear[1][0], linear[1][1], translate[1]}}};
    }

    template<typename T>
    NOA_IHD constexpr Mat33<T> truncated2affine(const Mat23<T>& truncated) noexcept {
        return {{{truncated[0][0], truncated[0][1], truncated[0][2]},
                 {truncated[1][0], truncated[1][1], truncated[1][2]},
                 {0, 0, 1}}};
    }

    template<typename T>
    NOA_IHD constexpr Mat22<T> affine2linear(const Mat33<T>& affine) noexcept {
        return {{{affine[0][0], affine[0][1]},
                 {affine[1][0], affine[1][1]}}};
    }

    template<typename T>
    NOA_IHD constexpr Mat22<T> truncated2linear(const Mat23<T>& truncated) noexcept {
        return {{{truncated[0][0], truncated[0][1]},
                 {truncated[1][0], truncated[1][1]}}};
    }

    template<typename T>
    NOA_IHD constexpr Mat23<T> affine2truncated(const Mat33<T>& affine) noexcept {
        return {{{affine[0][0], affine[0][1], affine[0][2]},
                 {affine[1][0], affine[1][1], affine[1][2]}}};
    }
}

// -- 3D transformations -- //
namespace noa::geometry {
    /// Returns a DHW 3x3 scaling matrix.
    /// \param s DHW scaling factors for each axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> scale(Vec3<T> s) noexcept {
        return Mat33<T>::from_diagonal(s);
    }

    /// Returns the DHW 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the outermost axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotate_z(T angle) noexcept {
        T c = noa::cos(angle);
        T s = noa::sin(angle);
        return {{{1, 0, 0},
                 {0, c, s},
                 {0, -s, c}}};
    }

    /// Returns the DHW 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the second-most axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotate_y(T angle) noexcept {
        T c = noa::cos(angle);
        T s = noa::sin(angle);
        return {{{c, 0, -s},
                 {0, 1, 0},
                 {s, 0, c}}};
    }

    /// Returns the DHW 3x3 rotation matrix describing the rotation
    /// by \p angle radians around the innermost axis.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotate_x(T angle) noexcept {
        T c = noa::cos(angle);
        T s = noa::sin(angle);
        return {{{c, s, 0},
                 {-s, c, 0},
                 {0, 0, 1}}};
    }

    /// Returns a DHW 3x3 matrix describing a rotation by an \p angle around a given \p axis.
    /// \param axis     Normalized axis, using the rightmost {Z,Y,X} coordinates.
    /// \param angle    Rotation angle, in radians.
    template<typename T>
    NOA_IHD constexpr Mat33<T> rotate(Vec3<T> axis, T angle) noexcept {
        // see https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
        NOA_ASSERT(noa::allclose(noa::norm(axis), static_cast<T>(1))); // axis should be normalized.

        T c = noa::cos(static_cast<T>(angle));
        T s = noa::sin(static_cast<T>(angle));
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
    template<typename T>
    NOA_IHD constexpr Mat44<T> translate(Vec3<T> shift) noexcept {
        return {{{1, 0, 0, shift[0]},
                 {0, 1, 0, shift[1]},
                 {0, 0, 1, shift[2]},
                 {0, 0, 0, 1}}};
    }

    template<typename T>
    NOA_IHD constexpr Mat44<T> linear2affine(const Mat33<T>& linear, const Vec3<T>& translate = {}) noexcept {
        return {{{linear[0][0], linear[0][1], linear[0][2], translate[0]},
                 {linear[1][0], linear[1][1], linear[1][2], translate[1]},
                 {linear[2][0], linear[2][1], linear[2][2], translate[2]},
                 {0, 0, 0, 1}}};
    }

    template<typename T>
    NOA_IHD constexpr Mat34<T> linear2truncated(const Mat33<T>& linear, const Vec3<T>& translate = {}) noexcept {
        return {{{linear[0][0], linear[0][1], linear[0][2], translate[0]},
                 {linear[1][0], linear[1][1], linear[1][2], translate[1]},
                 {linear[2][0], linear[2][1], linear[2][2], translate[2]}}};
    }

    template<typename T>
    NOA_IHD constexpr Mat44<T> truncated2affine(const Mat34<T>& truncated) noexcept {
        return {{{truncated[0][0], truncated[0][1], truncated[0][2], truncated[0][3]},
                 {truncated[1][0], truncated[1][1], truncated[1][2], truncated[1][3]},
                 {truncated[2][0], truncated[2][1], truncated[2][2], truncated[2][3]},
                 {0, 0, 0, 1}}};
    }

    template<typename T>
    NOA_IHD constexpr Mat33<T> affine2linear(const Mat44<T>& affine) noexcept {
        return {{{affine[0][0], affine[0][1], affine[0][2]},
                 {affine[1][0], affine[1][1], affine[1][2]},
                 {affine[2][0], affine[2][1], affine[2][2]}}};
    }

    template<typename T>
    NOA_IHD constexpr Mat33<T> truncated2linear(const Mat34<T>& truncated) noexcept {
        return {{{truncated[0][0], truncated[0][1], truncated[0][2]},
                 {truncated[1][0], truncated[1][1], truncated[1][2]},
                 {truncated[2][0], truncated[2][1], truncated[2][2]}}};
    }

    template<typename T>
    NOA_IHD constexpr Mat34<T> affine2truncated(const Mat44<T>& affine) noexcept {
        return {{{affine[0][0], affine[0][1], affine[0][2], affine[0][3]},
                 {affine[1][0], affine[1][1], affine[1][2], affine[1][3]},
                 {affine[2][0], affine[2][1], affine[2][2], affine[2][3]}}};
    }
}
