//#pragma once
//
//#include "noa/Definitions.h"
//#include "noa/Types.h"
//
/// @note Transformations are active (alibi) and assumes a right handed coordinate system.
///       All angles are given in radians, positive is counter-clockwise looking at the origin.
/// @note Even if the matrix contains some rotation-matrix specific functions (e.g. the rot*() functions),
///       it does not assume that the matrix is a "proper" rotation matrix (i.e. the determinant is not
///       necessarily equal to 1).

//namespace Math {
///*
// *
//        template<uint N, typename T> NOA_HD constexpr auto scale(T s) noexcept;
//        template<uint N, typename T> NOA_HD constexpr auto scale(const Float3<T>& s) noexcept;
//        template<uint N, typename T> NOA_HD constexpr auto rotX(T angle) noexcept;
//        template<uint N, typename T> NOA_HD constexpr auto rotY(T angle) noexcept;
//        template<uint N, typename T> NOA_HD constexpr auto rotZ(T angle) noexcept;
//        template<uint N, typename T> NOA_HD constexpr auto rot(const Float3<T>& axis, T angle) noexcept;
//        template<uint N, typename T> NOA_HD constexpr auto rotInPlane(T axis_angle, T angle) noexcept;
//
//        /// Scales @a m with the scalar @a s.
//        template<typename T, typename U> NOA_HD constexpr Mat3<T> scale(const Mat3<T>& m, U s) noexcept;
//
//        /// Scales @a m with the vector @a v. Equivalent to `m * Mat3<T>(s)`.
//        template<typename T, typename U> NOA_HD constexpr Mat3<T> scale(const Mat3<T>& m, const Float3<U>& s) noexcept;
//
//        template<typename T, typename U> NOA_HD constexpr Mat3<T> rotX(const Mat3<T>& m, U angle) noexcept;
//        template<typename T, typename U> NOA_HD constexpr Mat3<T> rotY(const Mat3<T>& m, U angle) noexcept;
//        template<typename T, typename U> NOA_HD constexpr Mat3<T> rotZ(const Mat3<T>& m, U angle) noexcept;
//
//        /// Rotates @a m by an @a angle (in radians) around a given @a axis.
//        /// @warning @a axis should be normalized, for instance with Math::normalize().
//        template<typename T, typename U>
//        NOA_HD constexpr Mat3<T> rot(const Mat3<T>& m, Float3<T> axis, U angle) noexcept;
//
//        /// Rotates @a m by @a angle radians around an in-place axis (on the XY plane) at @a axis_angle radians.
//        template<typename T, typename U, typename V>
//        NOA_HD constexpr Mat3<T> rotInPlane(const Mat3<T>& m, U axis_angle, V angle) noexcept;
// */
//    template<uint N, typename T>
//    NOA_HD constexpr auto scale(T s) noexcept {
//
//        if constexpr (N == 2) {
//
//        } else if constexpr (N == 3) {
//
//        } else if constexpr (N == 4) {
//
//        } else {
//            static_assert(Noa::Traits::always_false_v<T>);
//        }
//        return Mat3<T>(s);
//    }
//
//    template<typename T>
//    template<typename U>
//    NOA_HD constexpr Mat3<T>& Mat3<T>::scale(const Float3<U>& s) noexcept {
//        m_row[0] *= s;
//        m_row[1] *= s;
//        m_row[2] *= s;
//        return *this;
//    }
//
//    template<typename T>
//    template<typename U>
//    NOA_HD constexpr Mat3<T>& Mat3<T>::rotZ(U angle) noexcept {
//        T c = Math::cos(static_cast<T>(angle));
//        T s = Math::sin(static_cast<T>(angle));
//        *this *= Mat3<T>(c, -s, 0, s, c, 0, 0, 0, 1);
//        return *this;
//    }
//
//    template<typename T>
//    template<typename U>
//    NOA_HD constexpr Mat3<T>& Mat3<T>::rotY(U angle) noexcept {
//        T c = Math::cos(static_cast<T>(angle));
//        T s = Math::sin(static_cast<T>(angle));
//        *this *= Mat3<T>(c, 0, s, 0, 1, 0, -s, 0, c);
//        return *this;
//    }
//
//    template<typename T>
//    template<typename U>
//    NOA_HD constexpr Mat3<T>& Mat3<T>::rotX(U angle) noexcept {
//        T c = Math::cos(static_cast<T>(angle));
//        T s = Math::sin(static_cast<T>(angle));
//        *this *= Mat3<T>(1, 0, 0, 0, c, -s, 0, s, c);
//        return *this;
//    }
//
//    template<typename T>
//    template<typename U>
//    NOA_HD constexpr Mat3<T>& Mat3<T>::rot(const Float3<T>& axis, U angle) noexcept {
//        // see https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
//        NOA_ASSERT(Math::isEqual(Math::length(axis), static_cast<T>(1))); // axis should be normalized.
//
//        T c = Math::cos(static_cast<T>(angle));
//        T s = Math::sin(static_cast<T>(angle));
//        Float3<T> tmp((static_cast<T>(1) - c) * axis);
//
//        *this *= Mat3<T>(axis.x * tmp[0] + c,
//                         axis.x * tmp[1] - axis.z * s,
//                         axis.x * tmp[2] + axis.y * s,
//                         axis.y * tmp[0] + axis.z * s,
//                         axis.y * tmp[1] + c,
//                         axis.y * tmp[2] - axis.x * s,
//                         axis.z * tmp[0] - axis.y * s,
//                         axis.z * tmp[1] + axis.x * s,
//                         axis.z * tmp[2] + c);
//        return *this;
//    }
//
//    template<typename T>
//    template<typename U, typename V>
//    NOA_HD constexpr Mat3<T>& Mat3<T>::rotInPlane(U axis_angle, V angle) noexcept {
//        return this->rot(Float3<T>(Math::cos(axis_angle), Math::sin(axis_angle), 0), angle);
//    }
//
//
//    template<typename T, typename U>
//    NOA_FHD constexpr Mat3<T> scale(const Mat3<T>& m, U s) noexcept {
//        Mat3<T> scaled(m);
//        scaled.scale(s);
//        return scaled;
//    }
//
//    template<typename T, typename U>
//    NOA_FHD constexpr Mat3<T> scale(const Mat3<T>& m, const Float3<U>& s) noexcept {
//        Mat3<T> scaled(m);
//        scaled.scale(s);
//        return scaled;
//    }
//
//    template<typename T, typename U>
//    NOA_HD constexpr Mat3<T> rotX(const Mat3<T>& m, U angle) noexcept {
//        Mat3<T> out(m);
//        return out.rotX(angle);
//    }
//
//    template<typename T, typename U>
//    NOA_HD constexpr Mat3<T> rotY(const Mat3<T>& m, U angle) noexcept {
//        Mat3<T> out(m);
//        return out.rotY(angle);
//    }
//
//    template<typename T, typename U>
//    NOA_HD constexpr Mat3<T> rotZ(const Mat3<T>& m, U angle) noexcept {
//        Mat3<T> out(m);
//        return out.rotZ(angle);
//    }
//
//    template<typename T, typename U>
//    NOA_HD constexpr Mat3<T> rot(const Mat3<T>& m, Float3<T> axis, U angle) noexcept {
//        Mat3<T> out(m);
//        return out.rot(axis, angle);
//    }
//
//    template<typename T, typename U, typename V>
//    NOA_HD constexpr Mat3<T> rotInPlane(const Mat3<T>& m, U axis_angle, V angle) noexcept {
//        Mat3<T> out(m);
//        return out.rotInPlane(axis_angle, angle);
//    }
//}
