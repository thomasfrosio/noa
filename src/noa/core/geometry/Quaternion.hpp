#pragma once

#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Mat33.hpp"

namespace noa::geometry {
    /// Quaternion type to represent 3d rotations. The coefficients are saved in the z, y, x, w order.
    /// \note For our use cases, quaternions are mostly advantageous for (de)serializing 3d pure rotations.
    ///       For instance, a 3x3 matrix requires 9 load.f32|f64 instructions (Euler angles require 3), as opposed
    ///       to a quaternion which requires a single load.vec4.f32 instruction for single precision or two
    ///       load.vec2.f64 instruction for double precision. This can have a major performance benefit, especially
    ///       when multiple 3d rotations need to be loaded.
    template<typename Real>
    class Quaternion {
    public:
        using value_type = Real;
        using vec4_type = Vec4<value_type>;
        using vec3_type = Vec3<value_type>;
        using mat33_type = Mat33<value_type>;

    public:
        vec4_type coefficients; // z, y, x, w

    public: // Factory functions
        /// Converts a 3x3 special orthogonal matrix to a quaternion.
        /// \warning Only pure rotations are supported, ie no scaling, no reflection.
        [[nodiscard]] NOA_HD static constexpr Quaternion from_matrix(const mat33_type& matrix) noexcept {
            // This is also interesting, to handle the case with a scaling factor:
            // https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx#L978-L1001

            // From https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
            // Adapted to our zyx axis convention.
            Quaternion q;
            q.w() = noa::sqrt(noa::max(value_type{0}, 1 + matrix[2][2] + matrix[1][1] + matrix[0][0])) / 2;
            q.z() = noa::sqrt(noa::max(value_type{0}, 1 - matrix[2][2] - matrix[1][1] + matrix[0][0])) / 2;
            q.y() = noa::sqrt(noa::max(value_type{0}, 1 - matrix[2][2] + matrix[1][1] - matrix[0][0])) / 2;
            q.x() = noa::sqrt(noa::max(value_type{0}, 1 + matrix[2][2] - matrix[1][1] - matrix[0][0])) / 2;
            q.z() = noa::copysign(q.z(), matrix[1][2] - matrix[2][1]);
            q.y() = noa::copysign(q.y(), matrix[2][0] - matrix[0][2]);
            q.x() = noa::copysign(q.x(), matrix[0][1] - matrix[1][0]);
            return q;
        }

        [[nodiscard]] NOA_HD static constexpr Quaternion from_coefficients(const vec4_type& zyxw) noexcept {
            return {zyxw};
        }

        [[nodiscard]] NOA_HD static constexpr Quaternion from_coefficients(Real z, Real y, Real x, Real w) noexcept {
            return {z, y, x, w};
        }

    public: // Access
        [[nodiscard]] NOA_HD constexpr auto z() const noexcept -> value_type { return coefficients[0]; }
        [[nodiscard]] NOA_HD constexpr auto y() const noexcept -> value_type { return coefficients[1]; }
        [[nodiscard]] NOA_HD constexpr auto x() const noexcept -> value_type { return coefficients[2]; }
        [[nodiscard]] NOA_HD constexpr auto w() const noexcept -> value_type { return coefficients[3]; }
        [[nodiscard]] NOA_HD constexpr auto z() noexcept -> value_type& { return coefficients[0]; }
        [[nodiscard]] NOA_HD constexpr auto y() noexcept -> value_type& { return coefficients[1]; }
        [[nodiscard]] NOA_HD constexpr auto x() noexcept -> value_type& { return coefficients[2]; }
        [[nodiscard]] NOA_HD constexpr auto w() noexcept -> value_type& { return coefficients[3]; }

        [[nodiscard]] NOA_HD constexpr auto imag() const noexcept -> vec3_type {
            return {coefficients[0], coefficients[1], coefficients[2]};
        }

        template<typename T>
        [[nodiscard]] NOA_HD constexpr auto operator[](T i) noexcept -> value_type& { return coefficients[i]; }
        template<typename T>
        [[nodiscard]] NOA_HD constexpr auto operator[](T i) const noexcept -> const value_type& { return coefficients[i]; }

    public:
        NOA_HD constexpr auto operator*=(const Quaternion& rhs) noexcept -> Quaternion& {
            *this = *this * rhs;
            return *this;
        }

        NOA_HD constexpr auto operator*=(const value_type& rhs) noexcept -> Quaternion& {
            coefficients * rhs;
            return *this;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator*(
                const Quaternion& lhs,
                const Quaternion& rhs
        ) noexcept -> Quaternion {
            return {
                    /*z=*/ lhs.w() * rhs.z() + lhs.z() * rhs.w() + lhs.x() * rhs.y() - lhs.y() * rhs.x(),
                    /*y=*/ lhs.w() * rhs.y() + lhs.y() * rhs.w() + lhs.z() * rhs.x() - lhs.x() * rhs.z(),
                    /*x=*/ lhs.w() * rhs.x() + lhs.x() * rhs.w() + lhs.y() * rhs.z() - lhs.z() * rhs.y(),
                    /*w=*/ lhs.w() * rhs.w() - lhs.x() * rhs.x() - lhs.y() * rhs.y() - lhs.z() * rhs.z(),
            };
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator*(value_type lhs, Quaternion rhs) noexcept -> Quaternion {
            rhs *= lhs;
            return rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator*(Quaternion lhs, value_type rhs) noexcept -> Quaternion {
            lhs *= rhs;
            return lhs;
        }

        /// Converts the quaternion to a 3x3 rotation matrix.
        /// \warning The quaternion should be normalized, otherwise the result is undefined.
        NOA_HD constexpr auto to_matrix() const noexcept -> mat33_type {
            const auto tx  = 2 * x();
            const auto ty  = 2 * y();
            const auto tz  = 2 * z();
            const auto twx = tx * w();
            const auto twy = ty * w();
            const auto twz = tz * w();
            const auto txx = tx * x();
            const auto txy = ty * x();
            const auto txz = tz * x();
            const auto tyy = ty * y();
            const auto tyz = tz * y();
            const auto tzz = tz * z();

            mat33_type mat;
            mat[0][0] = 1 - (txx + tyy);
            mat[0][1] = tyz + twx;
            mat[0][2] = txz - twy;
            mat[1][0] = tyz - twx;
            mat[1][1] = 1 - (txx + tzz);
            mat[1][2] = txy + twz;
            mat[2][0] = txz + twy;
            mat[2][1] = txy - twz;
            mat[2][2] = 1 - (tyy + tzz);
            return mat;
        }

        /// Rotates a vector (this is equivalent but faster than \c v'=q*v*q.conj())
        /// \warning The quaternion should be normalized, otherwise the result is undefined.
        /// \remarks If the quaternion is used to rotate more than one point, then it is more efficient to first
        ///          convert it to a 3x3 matrix. Indeed, for n transformations, using a quaternion costs 30n
        ///          operations (18 multiplies and 12 adds|subtracts), whereas converting to a matrix first
        ///          costs 25 + 15n operations.
        [[nodiscard]] NOA_HD constexpr auto rotate(const vec3_type& zyx) const noexcept -> vec3_type {
            // Adapted from https://github.com/moble/quaternion/blob/main/src/quaternion.h
            const vec3_type sv_plus_rxv{
                    w() * zyx[2] + y() * zyx[0] - z() * zyx[1],
                    w() * zyx[1] + z() * zyx[2] - x() * zyx[0],
                    w() * zyx[0] + x() * zyx[1] - y() * zyx[2],
            };

            constexpr value_type two_over_m = 2; // here we assume quaternion is normalized, so m=1
            return {
                    zyx[0] + two_over_m * (x() * sv_plus_rxv[1] - y() * sv_plus_rxv[0]),
                    zyx[1] + two_over_m * (z() * sv_plus_rxv[0] - x() * sv_plus_rxv[2]),
                    zyx[2] + two_over_m * (y() * sv_plus_rxv[2] - z() * sv_plus_rxv[1]),
            };
        }

        [[nodiscard]] NOA_HD constexpr auto norm() const noexcept -> value_type {
            return noa::norm(coefficients);
        }

        [[nodiscard]] NOA_HD constexpr auto normalize() const noexcept -> Quaternion {
            return {noa::normalize(coefficients)};
        }

        [[nodiscard]] NOA_HD constexpr auto conj() const noexcept -> Quaternion {
            return {-z(), -y(), -x(), w()};
        }

        template<typename T, nt::enable_if_bool_t<nt::is_real_v<T>> = true>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept -> Quaternion<T> {
            return {coefficients.template as<T>()};
        }
    };
}

namespace noa::geometry {
    template<typename Real>
    NOA_HD auto matrix2quaternion(const Mat33<Real>& matrix) noexcept -> Quaternion<Real> {
        return Quaternion<Real>::from_matrix(matrix);
    }

    template<typename Real>
    NOA_HD auto quaternion2matrix(const Quaternion<Real>& quaternion) noexcept -> Mat33<Real> {
        return quaternion.to_matrix();
    }
}

namespace noa::traits {
    template<typename T> struct proclaim_is_quaternion : std::false_type {};
    template<typename T> struct proclaim_is_quaternion<noa::geometry::Quaternion<T>> : std::true_type {};

    template<typename T> using is_quaternion = std::bool_constant<proclaim_is_quaternion<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_quaternion_v = is_quaternion<T>::value;
    template<typename... Ts> constexpr bool are_quaternion_v = bool_and<is_quaternion_v<Ts>...>::value;

    template<typename T> constexpr bool is_quaternion_f32_v = is_quaternion_v<T> && std::is_same_v<value_type_t<T>, float>;
    template<typename T> constexpr bool is_quaternion_f64_v = is_quaternion_v<T> && std::is_same_v<value_type_t<T>, double>;
}
