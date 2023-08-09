#pragma once

#include "noa/core/Assert.hpp"
#include "noa/core/Definitions.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/traits/Matrix.hpp"
#include "noa/core/traits/Numerics.hpp"
#include "noa/core/types/Vec.hpp"

// A few necessary forward declarations:
namespace noa {
    template<typename T>
    class Mat33;

    namespace math {
        template<typename T>
        NOA_IHD constexpr Mat33<T> transpose(const Mat33<T>& m) noexcept;

        template<typename T>
        NOA_IHD constexpr Mat33<T> inverse(const Mat33<T>& m) noexcept;
    }
}

namespace noa {
    // A 3x3 floating-point matrix.
    template<typename Real>
    class Mat33 {
    public: // Type definitions
        using value_type = Real;
        using mutable_value_type = value_type;
        using row_type = Vec3<value_type>;

    public: // Component accesses
        static constexpr size_t ROWS = 3;
        static constexpr size_t COLS = 3;
        static constexpr size_t COUNT = ROWS * COLS;
        static constexpr size_t SIZE = COUNT;
        static constexpr int64_t SSIZE = ROWS * COLS;

        template<typename I, typename = std::enable_if_t<nt::is_int_v<I>>>
        NOA_HD constexpr row_type& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

        template<typename I, typename = std::enable_if_t<nt::is_int_v<I>>>
        NOA_HD constexpr const row_type& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

    public: // Default constructors
        NOA_HD constexpr Mat33() noexcept
                : m_row{row_type(1, 0, 0),
                        row_type(0, 1, 0),
                        row_type(0, 0, 1)} {}

    public: // Conversion constructors
        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat33(U s) noexcept
                : m_row{row_type(s, 0, 0),
                        row_type(0, s, 0),
                        row_type(0, 0, s)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat33(const Vec3<U>& v) noexcept
                : m_row{row_type(v[0], 0, 0),
                        row_type(0, v[1], 0),
                        row_type(0, 0, v[2])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat33(const Mat33<U>& m) noexcept
                : m_row{row_type(m[0]),
                        row_type(m[1]),
                        row_type(m[2])} {}

        template<typename X00, typename X01, typename X02,
                 typename Y10, typename Y11, typename Y12,
                 typename Z20, typename Z21, typename Z22>
        NOA_HD constexpr Mat33(X00 x00, X01 x01, X02 x02,
                               Y10 y10, Y11 y11, Y12 y12,
                               Z20 z20, Z21 z21, Z22 z22) noexcept
                : m_row{row_type(x00, x01, x02),
                        row_type(y10, y11, y12),
                        row_type(z20, z21, z22)} {}

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat33(U* ptr) noexcept
                : m_row{row_type(ptr[0], ptr[1], ptr[2]),
                        row_type(ptr[3], ptr[4], ptr[5]),
                        row_type(ptr[6], ptr[7], ptr[8])} {}

        template<typename V0, typename V1, typename V2>
        NOA_HD constexpr Mat33(const Vec3<V0>& r0,
                               const Vec3<V1>& r1,
                               const Vec3<V2>& r2) noexcept
                : m_row{row_type(r0),
                        row_type(r1),
                        row_type(r2)} {}

    public: // Assignment operators
        NOA_HD constexpr Mat33& operator+=(const Mat33& m) noexcept {
            m_row[0] += m[0];
            m_row[1] += m[1];
            m_row[2] += m[2];
            return *this;
        }

        NOA_HD constexpr Mat33& operator-=(const Mat33& m) noexcept {
            m_row[0] -= m[0];
            m_row[1] -= m[1];
            m_row[2] -= m[2];
            return *this;
        }

        NOA_HD constexpr Mat33& operator*=(const Mat33& m) noexcept {
            const auto A00 = m_row[0][0];
            const auto A01 = m_row[0][1];
            const auto A02 = m_row[0][2];
            const auto A10 = m_row[1][0];
            const auto A11 = m_row[1][1];
            const auto A12 = m_row[1][2];
            const auto A20 = m_row[2][0];
            const auto A21 = m_row[2][1];
            const auto A22 = m_row[2][2];

            const auto B00 = m[0][0];
            const auto B01 = m[0][1];
            const auto B02 = m[0][2];
            const auto B10 = m[1][0];
            const auto B11 = m[1][1];
            const auto B12 = m[1][2];
            const auto B20 = m[2][0];
            const auto B21 = m[2][1];
            const auto B22 = m[2][2];

            m_row[0][0] = A00 * B00 + A01 * B10 + A02 * B20;
            m_row[0][1] = A00 * B01 + A01 * B11 + A02 * B21;
            m_row[0][2] = A00 * B02 + A01 * B12 + A02 * B22;
            m_row[1][0] = A10 * B00 + A11 * B10 + A12 * B20;
            m_row[1][1] = A10 * B01 + A11 * B11 + A12 * B21;
            m_row[1][2] = A10 * B02 + A11 * B12 + A12 * B22;
            m_row[2][0] = A20 * B00 + A21 * B10 + A22 * B20;
            m_row[2][1] = A20 * B01 + A21 * B11 + A22 * B21;
            m_row[2][2] = A20 * B02 + A21 * B12 + A22 * B22;
            return *this;
        }

        NOA_HD constexpr Mat33& operator/=(const Mat33& m) noexcept {
            *this *= noa::math::inverse(m);
            return *this;
        }

        NOA_HD constexpr Mat33& operator+=(value_type s) noexcept {
            m_row[0] += s;
            m_row[1] += s;
            m_row[2] += s;
            return *this;
        }

        NOA_HD constexpr Mat33& operator-=(value_type s) noexcept {
            m_row[0] -= s;
            m_row[1] -= s;
            m_row[2] -= s;
            return *this;
        }

        NOA_HD constexpr Mat33& operator*=(value_type s) noexcept {
            m_row[0] *= s;
            m_row[1] *= s;
            m_row[2] *= s;
            return *this;
        }

        NOA_HD constexpr Mat33& operator/=(value_type s) noexcept {
            m_row[0] /= s;
            m_row[1] /= s;
            m_row[2] /= s;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Mat33 operator+(const Mat33& m) noexcept {
            return m;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator-(const Mat33& m) noexcept {
            return Mat33(-m[0], -m[1], -m[2]);
        }

        // -- Binary arithmetic operators --
        [[nodiscard]] friend NOA_HD constexpr Mat33 operator+(const Mat33& m1, const Mat33& m2) noexcept {
            return Mat33(m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator+(value_type s, const Mat33& m) noexcept {
            return Mat33(s + m[0], s + m[1], s + m[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator+(const Mat33& m, value_type s) noexcept {
            return Mat33(m[0] + s, m[1] + s, m[2] + s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator-(const Mat33& m1, const Mat33& m2) noexcept {
            return Mat33(m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator-(value_type s, const Mat33& m) noexcept {
            return Mat33(s - m[0], s - m[1], s - m[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator-(const Mat33& m, value_type s) noexcept {
            return Mat33(m[0] - s, m[1] - s, m[2] - s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator*(Mat33 m1, const Mat33& m2) noexcept {
            m1 *= m2;
            return m1;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator*(value_type s, const Mat33& m) noexcept {
            return Mat33(m[0] * s, m[1] * s, m[2] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator*(const Mat33& m, value_type s) noexcept {
            return Mat33(m[0] * s, m[1] * s, m[2] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator*(const Mat33& m, const row_type& column) noexcept {
            return row_type(noa::math::dot(m[0], column),
                            noa::math::dot(m[1], column),
                            noa::math::dot(m[2], column));
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator*(const row_type& row, const Mat33& m) noexcept {
            return row_type(noa::math::dot(row_type(m[0][0], m[1][0], m[2][0]), row),
                            noa::math::dot(row_type(m[0][1], m[1][1], m[2][1]), row),
                            noa::math::dot(row_type(m[0][2], m[1][2], m[2][2]), row));
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator/(Mat33 m1, const Mat33& m2) noexcept {
            m1 /= m2;
            return m1;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator/(value_type s, const Mat33& m) noexcept {
            return Mat33(s / m[0], s / m[1], s / m[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat33 operator/(const Mat33& m, value_type s) noexcept {
            return Mat33(m[0] / s, m[1] / s, m[2] / s);
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator/(const Mat33& m, const row_type& column) noexcept {
            return noa::math::inverse(m) * column;
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator/(const row_type& row, const Mat33& m) noexcept {
            return row * noa::math::inverse(m);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(const Mat33& m1, const Mat33& m2) noexcept {
            return noa::all(m1[0] == m2[0]) && noa::all(m1[1] == m2[1]) && noa::all(m1[2] == m2[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(const Mat33& m1, const Mat33& m2) noexcept {
            return noa::any(m1[0] != m2[0]) || noa::any(m1[1] != m2[1]) || noa::any(m1[2] != m2[2]);
        }

    public:
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return m_row[0].data(); }
        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return m_row[0].data(); }

        template<typename T, std::enable_if_t<nt::is_real_v<T>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return Mat33<T>(*this);
        }

        [[nodiscard]] NOA_HD constexpr Mat33 transpose() const noexcept {
            return noa::math::transpose(*this);
        }

        [[nodiscard]] NOA_HD constexpr Mat33 inverse() const noexcept {
            return noa::math::inverse(*this);
        }

    public: // Support for noa::string::human<Vec>();
        [[nodiscard]] static std::string name() {
            if constexpr (std::is_same_v<value_type, float>)
                return "Float33";
            else
                return "Double33";
        }

    private:
        row_type m_row[ROWS];
    };

    template<typename T> struct nt::proclaim_is_mat33<Mat33<T>> : std::true_type {};

    using Float33 = Mat33<float>;
    using Double33 = Mat33<double>;
}

namespace noa::math {
    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat33<T> element_multiply(Mat33<T> m1, const Mat33<T>& m2) noexcept {
        for (size_t i = 0; i < Mat33<T>::ROWS; ++i)
            m1[i] *= m2[i];
        return m1;
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat33<T> outer_product(const Vec3<T>& column, const Vec3<T>& row) noexcept {
        return Mat33<T>(column[0] * row[0], column[0] * row[1], column[0] * row[2],
                        column[1] * row[0], column[1] * row[1], column[1] * row[2],
                        column[2] * row[0], column[2] * row[1], column[2] * row[2]);
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat33<T> transpose(const Mat33<T>& m) noexcept {
        return Mat33<T>(m[0][0], m[1][0], m[2][0],
                        m[0][1], m[1][1], m[2][1],
                        m[0][2], m[1][2], m[2][2]);
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
        NOA_ASSERT(!are_almost_equal(det, T{0})); // non singular
        const auto one_over_determinant = 1 / det;
        return Mat33<T>(+(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * one_over_determinant,
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
    [[nodiscard]] NOA_IHD constexpr bool are_almost_equal(
            const Mat33<T>& m1, const Mat33<T>& m2, T epsilon = 1e-6f) noexcept {
        return noa::all(are_almost_equal<ULP>(m1[0], m2[0], epsilon)) &&
               noa::all(are_almost_equal<ULP>(m1[1], m2[1], epsilon)) &&
               noa::all(are_almost_equal<ULP>(m1[2], m2[2], epsilon));
    }
}
