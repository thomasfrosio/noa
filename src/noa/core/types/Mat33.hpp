#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/math/Comparison.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Vec.hpp"

// A few necessary forward declarations:
namespace noa {
    inline namespace types {
        template<typename T>
        class Mat33;
    }

    template<typename T>
    NOA_IHD constexpr Mat33<T> transpose(const Mat33<T>& m) noexcept;

    template<typename T>
    NOA_IHD constexpr Mat33<T> inverse(const Mat33<T>& m) noexcept;
}

namespace noa::inline types {
    /// A 3x3 floating-point matrix.
    template<typename Real>
    class Mat33 {
    public: // Type definitions
        static_assert(!std::is_same_v<Real, Half>);

        using value_type = Real;
        using mutable_value_type = value_type;
        using row_type = Vec3<value_type>;

        static constexpr size_t ROWS = 3;
        static constexpr size_t COLS = 3;
        static constexpr size_t COUNT = ROWS * COLS;
        static constexpr size_t SIZE = COUNT;
        static constexpr int64_t SSIZE = ROWS * COLS;

    public:
        row_type row[ROWS];

    public: // Component accesses
        template<typename I> requires nt::is_int_v<I>
        NOA_HD constexpr row_type& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return row[i];
        }

        template<typename I> requires nt::is_int_v<I>
        NOA_HD constexpr const row_type& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return row[i];
        }

    public: // Static factory functions
        template<typename U> requires nt::is_scalar_v<U>
        [[nodiscard]] NOA_HD static constexpr Mat33 from_value(U s) noexcept {
            return {row_type::from_values(s, 0, 0),
                    row_type::from_values(0, s, 0),
                    row_type::from_values(0, 0, s)};
        }

        template<typename U> requires nt::is_scalar_v<U>
        [[nodiscard]] NOA_HD static constexpr Mat33 from_diagonal(U s) noexcept {
            return from_value(s);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat33 from_diagonal(const Vec3<U>& diagonal) noexcept {
            return {row_type::from_values(diagonal[0], 0, 0),
                    row_type::from_values(0, diagonal[1], 0),
                    row_type::from_values(0, 0, diagonal[2])};
        }

        template<typename U> requires nt::is_scalar_v<U>
        [[nodiscard]] NOA_HD static constexpr Mat33 eye(U s) noexcept {
            return from_diagonal(s);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat33 eye(const Vec3<U>& diagonal) noexcept {
            return from_diagonal(diagonal);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat33 from_matrix(const Mat33<U>& m) noexcept {
            return {m[0].template as<value_type>(),
                    m[1].template as<value_type>(),
                    m[2].template as<value_type>()};
        }

        template<typename X00, typename X01, typename X02,
                 typename Y10, typename Y11, typename Y12,
                 typename Z20, typename Z21, typename Z22>
        [[nodiscard]] NOA_HD static constexpr Mat33 from_values(
                X00 x00, X01 x01, X02 x02,
                Y10 y10, Y11 y11, Y12 y12,
                Z20 z20, Z21 z21, Z22 z22
        ) noexcept {
            return {row_type::from_values(x00, x01, x02),
                    row_type::from_values(y10, y11, y12),
                    row_type::from_values(z20, z21, z22)};
        }

        template<typename U> requires nt::is_scalar_v<U>
        [[nodiscard]] NOA_HD static constexpr Mat33 from_pointer(U* ptr) noexcept {
            return {row_type::from_values(ptr[0], ptr[1], ptr[2]),
                    row_type::from_values(ptr[3], ptr[4], ptr[5]),
                    row_type::from_values(ptr[6], ptr[7], ptr[8])};
        }

        template<typename V0, typename V1, typename V2>
        [[nodiscard]] NOA_HD static constexpr Mat33 from_rows(
                const Vec3<V0>& r0,
                const Vec3<V1>& r1,
                const Vec3<V2>& r2
        ) noexcept {
            return {r0.template as<value_type>(),
                    r1.template as<value_type>(),
                    r2.template as<value_type>()};
        }

        template<typename V0, typename V1, typename V2>
        [[nodiscard]] NOA_HD static constexpr Mat33 from_columns(
                const Vec3<V0>& c0,
                const Vec3<V1>& c1,
                const Vec3<V2>& c2
        ) noexcept {
            return {row_type::from_values(c0[0], c1[0], c2[0]),
                    row_type::from_values(c0[1], c1[1], c2[1]),
                    row_type::from_values(c0[2], c1[2], c2[2])};
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Mat33<U>>(Mat33<T>{}).
        template<typename U>
        [[nodiscard]] NOA_HD constexpr explicit operator Mat33<U>() const noexcept {
            return Mat33<U>::from_matrix(*this);
        }

    public: // Assignment operators
        NOA_HD constexpr Mat33& operator+=(const Mat33& m) noexcept {
            row[0] += m[0];
            row[1] += m[1];
            row[2] += m[2];
            return *this;
        }

        NOA_HD constexpr Mat33& operator-=(const Mat33& m) noexcept {
            row[0] -= m[0];
            row[1] -= m[1];
            row[2] -= m[2];
            return *this;
        }

        NOA_HD constexpr Mat33& operator*=(const Mat33& m) noexcept {
            const auto A00 = row[0][0];
            const auto A01 = row[0][1];
            const auto A02 = row[0][2];
            const auto A10 = row[1][0];
            const auto A11 = row[1][1];
            const auto A12 = row[1][2];
            const auto A20 = row[2][0];
            const auto A21 = row[2][1];
            const auto A22 = row[2][2];

            const auto B00 = m[0][0];
            const auto B01 = m[0][1];
            const auto B02 = m[0][2];
            const auto B10 = m[1][0];
            const auto B11 = m[1][1];
            const auto B12 = m[1][2];
            const auto B20 = m[2][0];
            const auto B21 = m[2][1];
            const auto B22 = m[2][2];

            row[0][0] = A00 * B00 + A01 * B10 + A02 * B20;
            row[0][1] = A00 * B01 + A01 * B11 + A02 * B21;
            row[0][2] = A00 * B02 + A01 * B12 + A02 * B22;
            row[1][0] = A10 * B00 + A11 * B10 + A12 * B20;
            row[1][1] = A10 * B01 + A11 * B11 + A12 * B21;
            row[1][2] = A10 * B02 + A11 * B12 + A12 * B22;
            row[2][0] = A20 * B00 + A21 * B10 + A22 * B20;
            row[2][1] = A20 * B01 + A21 * B11 + A22 * B21;
            row[2][2] = A20 * B02 + A21 * B12 + A22 * B22;
            return *this;
        }

        NOA_HD constexpr Mat33& operator/=(const Mat33& m) noexcept {
            *this *= noa::inverse(m);
            return *this;
        }

        NOA_HD constexpr Mat33& operator+=(value_type s) noexcept {
            row[0] += s;
            row[1] += s;
            row[2] += s;
            return *this;
        }

        NOA_HD constexpr Mat33& operator-=(value_type s) noexcept {
            row[0] -= s;
            row[1] -= s;
            row[2] -= s;
            return *this;
        }

        NOA_HD constexpr Mat33& operator*=(value_type s) noexcept {
            row[0] *= s;
            row[1] *= s;
            row[2] *= s;
            return *this;
        }

        NOA_HD constexpr Mat33& operator/=(value_type s) noexcept {
            row[0] /= s;
            row[1] /= s;
            row[2] /= s;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Mat33 operator+(const Mat33& m) noexcept {
            return m;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator-(const Mat33& m) noexcept {
            return {-m[0], -m[1], -m[2]};
        }

        // -- Binary arithmetic operators --
        [[nodiscard]] friend NOA_HD constexpr Mat33 operator+(const Mat33& m1, const Mat33& m2) noexcept {
            return {m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator+(value_type s, const Mat33& m) noexcept {
            return {s + m[0], s + m[1], s + m[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator+(const Mat33& m, value_type s) noexcept {
            return {m[0] + s, m[1] + s, m[2] + s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator-(const Mat33& m1, const Mat33& m2) noexcept {
            return {m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator-(value_type s, const Mat33& m) noexcept {
            return {s - m[0], s - m[1], s - m[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator-(const Mat33& m, value_type s) noexcept {
            return {m[0] - s, m[1] - s, m[2] - s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator*(Mat33 m1, const Mat33& m2) noexcept {
            m1 *= m2;
            return m1;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator*(value_type s, const Mat33& m) noexcept {
            return {m[0] * s, m[1] * s, m[2] * s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator*(const Mat33& m, value_type s) noexcept {
            return {m[0] * s, m[1] * s, m[2] * s};
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator*(const Mat33& m, const row_type& column) noexcept {
            return {dot(m[0], column),
                    dot(m[1], column),
                    dot(m[2], column)};
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator*(const row_type& r, const Mat33& m) noexcept {
            return {dot(row_type{m[0][0], m[1][0], m[2][0]}, r),
                    dot(row_type{m[0][1], m[1][1], m[2][1]}, r),
                    dot(row_type{m[0][2], m[1][2], m[2][2]}, r)};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator/(Mat33 m1, const Mat33& m2) noexcept {
            m1 /= m2;
            return m1;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator/(value_type s, const Mat33& m) noexcept {
            return {s / m[0], s / m[1], s / m[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator/(const Mat33& m, value_type s) noexcept {
            return {m[0] / s, m[1] / s, m[2] / s};
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator/(const Mat33& m, const row_type& c) noexcept {
            return noa::inverse(m) * c;
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator/(const row_type& r, const Mat33& m) noexcept {
            return r * noa::inverse(m);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(const Mat33& m1, const Mat33& m2) noexcept {
            return noa::all(m1[0] == m2[0]) && noa::all(m1[1] == m2[1]) && noa::all(m1[2] == m2[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(const Mat33& m1, const Mat33& m2) noexcept {
            return noa::any(m1[0] != m2[0]) || noa::any(m1[1] != m2[1]) || noa::any(m1[2] != m2[2]);
        }

    public:
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return row[0].data(); }
        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return row[0].data(); }

        template<typename T> requires nt::is_real_v<T>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return Mat33<T>::from_matrix(*this);
        }

        [[nodiscard]] NOA_HD constexpr Mat33 transpose() const noexcept {
            return noa::transpose(*this);
        }

        [[nodiscard]] NOA_HD constexpr Mat33 inverse() const noexcept {
            return noa::inverse(*this);
        }

#if defined(NOA_IS_CPU_CODE)
    public:
        [[nodiscard]] static std::string name() {
            if constexpr (std::is_same_v<value_type, float>)
                return "Mat33<f32>";
            else
                return "Mat33<f64>";
        }
#endif
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_mat33<Mat33<T>> : std::true_type {};
}

namespace noa {
    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat33<T> ewise_multiply(Mat33<T> m1, const Mat33<T>& m2) noexcept {
        for (size_t i = 0; i < Mat33<T>::ROWS; ++i)
            m1[i] *= m2[i];
        return m1;
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat33<T> outer_product(const Vec3<T>& column, const Vec3<T>& row) noexcept {
        return {{{column[0] * row[0], column[0] * row[1], column[0] * row[2]},
                 {column[1] * row[0], column[1] * row[1], column[1] * row[2]},
                 {column[2] * row[0], column[2] * row[1], column[2] * row[2]}}};
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat33<T> transpose(const Mat33<T>& m) noexcept {
        return {{{m[0][0], m[1][0], m[2][0]},
                 {m[0][1], m[1][1], m[2][1]},
                 {m[0][2], m[1][2], m[2][2]}}};
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr T determinant(const Mat33<T>& m) noexcept {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    template<typename T>
    [[nodiscard]] NOA_HD constexpr Mat33<T> inverse(const Mat33<T>& m) noexcept {
        const auto det = determinant(m);
        NOA_ASSERT(!allclose(det, T{0})); // non singular
        const auto one_over_determinant = 1 / det;
        return Mat33<T>::from_values(
                +(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * one_over_determinant,
                -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * one_over_determinant,
                +(m[0][1] * m[1][2] - m[0][2] * m[1][1]) * one_over_determinant,
                -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * one_over_determinant,
                +(m[0][0] * m[2][2] - m[0][2] * m[2][0]) * one_over_determinant,
                -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * one_over_determinant,
                +(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * one_over_determinant,
                -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * one_over_determinant,
                +(m[0][0] * m[1][1] - m[0][1] * m[1][0]) * one_over_determinant);
    }

    template<int32_t ULP = 2, typename T>
    [[nodiscard]] NOA_IHD constexpr bool allclose(
            const Mat33<T>& m1,
            const Mat33<T>& m2,
            T epsilon = static_cast<T>(1e-6)
    ) noexcept {
        return noa::all(allclose<ULP>(m1[0], m2[0], epsilon)) &&
               noa::all(allclose<ULP>(m1[1], m2[1], epsilon)) &&
               noa::all(allclose<ULP>(m1[2], m2[2], epsilon));
    }
}
