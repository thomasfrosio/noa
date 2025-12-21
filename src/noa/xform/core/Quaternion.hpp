#pragma once

#include "noa/runtime/core/Vec.hpp"
#include "noa/xform/core/Mat.hpp"

namespace noa::xform {
    /// Quaternion type to represent 3d rotations. The coefficients are saved in the z, y, x, w order.
    /// \note For our use cases, quaternions are mostly helpful for (de)serializing 3d pure rotations.
    ///       For instance, in CUDA, a 3x3 matrix requires 9 load.f32|f64 instructions (Euler angles require 3),
    ///       as opposed to a quaternion which requires a single load.vec4.f32 instruction for single precision
    ///       or two load.vec2.f64 instructions for double precision. This can have a major performance benefit,
    ///       especially when multiple 3d rotations need to be loaded.
    template<nt::real T>
    class alignas(16) Quaternion {
    public:
        using value_type = T;
        using vec4_type = Vec<value_type, 4>;
        using vec3_type = Vec<value_type, 3>;
        using mat33_type = Mat33<value_type>;

    public:
        value_type z, y, x, w;

    public: // Factory functions
        /// Converts a 3x3 orthogonal matrix to a quaternion.
        /// \warning Only pure rotations are supported, so the matrix should have no scaling and no reflection.
        template<typename U>
        [[nodiscard]] NOA_HD static constexpr auto from_matrix(const Mat33<U>& matrix) noexcept -> Quaternion {
            // This is also interesting, to handle the case with a scaling factor:
            // https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx#L978-L1001

            // From https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
            // Adapted to our zyx axis convention.
            mat33_type mat = matrix.template as<value_type>();
            Quaternion q;
            q.w = sqrt(max(value_type{0}, 1 + mat[2][2] + mat[1][1] + mat[0][0])) / 2;
            q.z = sqrt(max(value_type{0}, 1 - mat[2][2] - mat[1][1] + mat[0][0])) / 2;
            q.y = sqrt(max(value_type{0}, 1 - mat[2][2] + mat[1][1] - mat[0][0])) / 2;
            q.x = sqrt(max(value_type{0}, 1 + mat[2][2] - mat[1][1] - mat[0][0])) / 2;
            q.z = copysign(q.z, mat[1][2] - mat[2][1]);
            q.y = copysign(q.y, mat[2][0] - mat[0][2]);
            q.x = copysign(q.x, mat[0][1] - mat[1][0]);
            return q;
        }

        template<typename U, usize A>
        [[nodiscard]] NOA_HD static constexpr auto from_coefficients(const Vec<U, 4, A>& zyxw) noexcept -> Quaternion {
            return {static_cast<value_type>(zyxw[0]),
                    static_cast<value_type>(zyxw[1]),
                    static_cast<value_type>(zyxw[2]),
                    static_cast<value_type>(zyxw[3])};
        }

        [[nodiscard]] NOA_HD static constexpr auto from_coefficients(auto z, auto y, auto x, auto w) noexcept -> Quaternion {
            return {static_cast<value_type>(z),
                    static_cast<value_type>(y),
                    static_cast<value_type>(x),
                    static_cast<value_type>(w)};
        }

    public: // Access
        [[nodiscard]] NOA_HD constexpr auto to_imag() const noexcept -> vec3_type { return {z, y, x}; }
        [[nodiscard]] NOA_HD constexpr auto to_vec() const noexcept -> vec4_type { return {z, y, x, w}; }
        [[nodiscard]] NOA_HD constexpr auto to_coefficients() const noexcept -> vec4_type { return to_vec(); }

        /// Converts the quaternion to a 3x3 rotation matrix.
        /// \warning The quaternion should be normalized, otherwise the result is unspecified.
        NOA_HD constexpr auto to_matrix() const noexcept -> mat33_type {
            const auto tx  = 2 * x;
            const auto ty  = 2 * y;
            const auto tz  = 2 * z;
            const auto twx = tx * w;
            const auto twy = ty * w;
            const auto twz = tz * w;
            const auto txx = tx * x;
            const auto txy = ty * x;
            const auto txz = tz * x;
            const auto tyy = ty * y;
            const auto tyz = tz * y;
            const auto tzz = tz * z;

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

        [[nodiscard]] NOA_HD constexpr auto operator[](nt::integer auto i) noexcept -> value_type& {
            ni::bounds_check(4, i);
            if (i == 0)
                return z;
            if (i == 1)
                return y;
            if (i == 2)
                return x;
            return w;
        }

        [[nodiscard]] NOA_HD constexpr auto operator[](nt::integer auto i) const noexcept -> const value_type& {
            ni::bounds_check(4, i);
            if (i == 0)
                return z;
            if (i == 1)
                return y;
            if (i == 2)
                return x;
            return w;
        }

    public:
        NOA_HD constexpr auto operator*=(const Quaternion& rhs) noexcept -> Quaternion& {
            *this = *this * rhs;
            return *this;
        }

        NOA_HD constexpr auto operator*=(const value_type& rhs) noexcept -> Quaternion& {
            z *= rhs;
            y *= rhs;
            x *= rhs;
            w *= rhs;
            return *this;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator*(
            const Quaternion& lhs,
            const Quaternion& rhs
        ) noexcept -> Quaternion {
            return {
                    .z=lhs.w * rhs.z + lhs.z * rhs.w + lhs.x * rhs.y - lhs.y * rhs.x,
                    .y=lhs.w * rhs.y + lhs.y * rhs.w + lhs.z * rhs.x - lhs.x * rhs.z,
                    .x=lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
                    .w=lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
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

        [[nodiscard]] friend NOA_HD constexpr auto operator==(const Quaternion& lhs, const Quaternion& rhs) noexcept -> bool {
            return lhs.z == rhs.z and lhs.y == rhs.y and lhs.x == rhs.x and lhs.w == rhs.w;
        }
        [[nodiscard]] friend NOA_HD constexpr auto operator!=(const Quaternion& lhs, const Quaternion& rhs) noexcept -> bool {
            return lhs.z != rhs.z or lhs.y != rhs.y or lhs.x != rhs.x or lhs.w != rhs.w;
        }

        /// Rotates a vector (this is equivalent but faster than \c v'=q*v*q.conj())
        /// \warning The quaternion should be normalized, otherwise the result is unspecified.
        /// \remarks If the quaternion is used to rotate more than one point, then it is more efficient to first
        ///          convert it to a 3x3 matrix. Indeed, for n transformations, using a quaternion costs 30n
        ///          operations (18 multiplies and 12 adds|subtracts), whereas converting to a matrix first
        ///          costs 25 + 15n operations.
        [[nodiscard]] NOA_HD constexpr auto rotate(const vec3_type& zyx) const noexcept -> vec3_type {
            // Adapted from https://github.com/moble/quaternion/blob/main/src/quaternion.h
            const vec3_type sv_plus_rxv{
                    w * zyx[2] + y * zyx[0] - z * zyx[1],
                    w * zyx[1] + z * zyx[2] - x * zyx[0],
                    w * zyx[0] + x * zyx[1] - y * zyx[2],
            };

            constexpr value_type two_over_m = 2; // here we assume quaternion is normalized, so m=1
            return {
                    zyx[0] + two_over_m * (x * sv_plus_rxv[1] - y * sv_plus_rxv[0]),
                    zyx[1] + two_over_m * (z * sv_plus_rxv[0] - x * sv_plus_rxv[2]),
                    zyx[2] + two_over_m * (y * sv_plus_rxv[2] - z * sv_plus_rxv[1]),
            };
        }

        [[nodiscard]] NOA_HD constexpr auto norm() const noexcept -> value_type {
            return noa::norm(to_vec());
        }

        [[nodiscard]] NOA_HD constexpr auto normalize() const noexcept -> Quaternion {
            return from_coefficients(noa::normalize(to_vec()));
        }

        [[nodiscard]] NOA_HD constexpr auto conj() const noexcept -> Quaternion {
            return {-z, -y, -x, w};
        }

        template<nt::real U>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept -> Quaternion<U> {
            return {static_cast<U>(z), static_cast<U>(y), static_cast<U>(x), static_cast<U>(w)};
        }
    };
}

namespace noa::xform {
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
    template<typename T> struct proclaim_is_quaternion<nx::Quaternion<T>> : std::true_type {};
    template<quaternion T> struct proclaim_is_trivial_zero<T> : std::true_type {};
}
