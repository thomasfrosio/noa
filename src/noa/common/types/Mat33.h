/// \file noa/common/types/Mat33.h
/// \author Thomas - ffyr2w
/// \date 2 Jun 2021
/// A 3x3 floating-point matrix.

#pragma once

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Float3.h"

// A few necessary forward declarations:
namespace noa {
    template<typename T>
    class Mat44;

    template<typename T>
    class Mat34;

    template<typename T>
    class Mat33;

    template<typename T>
    class Mat23;

    template<typename T>
    class Mat22;

    template<typename T>
    class Float2;

    template<typename T>
    class Int3;

    namespace math {
        template<typename T>
        NOA_IHD constexpr Mat33<T> inverse(Mat33<T> m) noexcept;
    }
}

namespace noa {
    /// A 3x3 floating-point matrix.
    /// \note The indexing is "row-first" (as opposed to "column-first", like in OpenGL Math),
    ///       i.e. M[r][c] with r = row index and c = column index. All indexes starts from 0.
    template<typename T>
    class Mat33 {
    public: // Type definitions
        typedef T value_type;

    public: // Component accesses
        static constexpr size_t ROWS = 3;
        static constexpr size_t COLS = 3;
        static constexpr size_t COUNT = ROWS * COLS;

        NOA_HD constexpr Float3<T>& operator[](size_t i) noexcept {
            NOA_ASSERT(i < this->ROWS);
            return m_row[i];
        }

        NOA_HD constexpr const Float3<T>& operator[](size_t i) const noexcept {
            NOA_ASSERT(i < this->ROWS);
            return m_row[i];
        }

    public: // Default constructors
        NOA_HD constexpr Mat33() noexcept
                : m_row{Float3<T>(1, 0, 0),
                        Float3<T>(0, 1, 0),
                        Float3<T>(0, 0, 1)} {}

        constexpr Mat33(const Mat33&) noexcept = default;
        constexpr Mat33(Mat33&&) noexcept = default;

    public: // Conversion constructors
        template<typename U>
        NOA_HD constexpr explicit Mat33(U s) noexcept
                : m_row{Float3<T>(s, 0, 0),
                        Float3<T>(0, s, 0),
                        Float3<T>(0, 0, s)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat33(Float3<U> v) noexcept
                : m_row{Float3<T>(v.x, 0, 0),
                        Float3<T>(0, v.y, 0),
                        Float3<T>(0, 0, v.z)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat33(Float2<U> v) noexcept
                : m_row{Float3<T>(v.x, 0, 0),
                        Float3<T>(0, v.y, 0),
                        Float3<T>(0, 0, 1)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat33(Mat44<U> m) noexcept
                : m_row{Float3<T>(m[0][0], m[0][1], m[0][2]),
                        Float3<T>(m[1][0], m[1][1], m[1][2]),
                        Float3<T>(m[2][0], m[2][1], m[2][2])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat33(Mat34<U> m) noexcept
                : m_row{Float3<T>(m[0][0], m[0][1], m[0][2]),
                        Float3<T>(m[1][0], m[1][1], m[1][2]),
                        Float3<T>(m[2][0], m[2][1], m[2][2])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat33(Mat33<U> m) noexcept
                : m_row{Float3<T>(m[0]),
                        Float3<T>(m[1]),
                        Float3<T>(m[2])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat33(Mat23<U> m) noexcept
                : m_row{Float3<T>(m[0]),
                        Float3<T>(m[1]),
                        Float3<T>(0, 0, 1)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat33(Mat22<U> m) noexcept
                : m_row{Float3<T>(m[0][0], m[0][1], 0),
                        Float3<T>(m[1][0], m[1][1], 0),
                        Float3<T>(0, 0, 1)} {}

        template<typename U, typename V>
        NOA_HD constexpr explicit Mat33(Mat22<U> m, Float2<V> v) noexcept
                : m_row{Float3<T>(m[0][0], m[0][1], v[0]),
                        Float3<T>(m[1][0], m[1][1], v[1]),
                        Float3<T>(0, 0, 1)} {}

        template<typename X00, typename X01, typename X02,
                 typename Y10, typename Y11, typename Y12,
                 typename Z20, typename Z21, typename Z22>
        NOA_HD constexpr Mat33(X00 x00, X01 x01, X02 x02,
                               Y10 y10, Y11 y11, Y12 y12,
                               Z20 z20, Z21 z21, Z22 z22) noexcept
                : m_row{Float3<T>(x00, x01, x02),
                        Float3<T>(y10, y11, y12),
                        Float3<T>(z20, z21, z22)} {}

        template<typename V0, typename V1, typename V2>
        NOA_HD constexpr Mat33(Float3<V0> r0,
                               Float3<V1> r1,
                               Float3<V2> r2) noexcept
                : m_row{Float3<T>(r0),
                        Float3<T>(r1),
                        Float3<T>(r2)} {}

        template<typename V0, typename V1, typename V2>
        NOA_HD constexpr Mat33(Int3<V0> r0,
                               Int3<V1> r1,
                               Int3<V2> r2) noexcept
                : m_row{Float3<T>(r0),
                        Float3<T>(r1),
                        Float3<T>(r2)} {}

    public: // Assignment operators
        constexpr Mat33& operator=(const Mat33& v) noexcept = default;
        constexpr Mat33& operator=(Mat33&& v) noexcept = default;

        NOA_HD constexpr Mat33& operator+=(Mat33 m) noexcept {
            m_row[0] += m[0];
            m_row[1] += m[1];
            m_row[2] += m[2];
            return *this;
        }

        NOA_HD constexpr Mat33& operator-=(Mat33 m) noexcept {
            m_row[0] -= m[0];
            m_row[1] -= m[1];
            m_row[2] -= m[2];
            return *this;
        }

        NOA_HD constexpr Mat33& operator*=(Mat33 m) noexcept {
            const T A00 = m_row[0][0];
            const T A01 = m_row[0][1];
            const T A02 = m_row[0][2];
            const T A10 = m_row[1][0];
            const T A11 = m_row[1][1];
            const T A12 = m_row[1][2];
            const T A20 = m_row[2][0];
            const T A21 = m_row[2][1];
            const T A22 = m_row[2][2];

            const T B00 = m[0][0];
            const T B01 = m[0][1];
            const T B02 = m[0][2];
            const T B10 = m[1][0];
            const T B11 = m[1][1];
            const T B12 = m[1][2];
            const T B20 = m[2][0];
            const T B21 = m[2][1];
            const T B22 = m[2][2];

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

        NOA_HD constexpr Mat33& operator/=(Mat33 m) noexcept {
            *this *= math::inverse(m);
            return *this;
        }

        NOA_HD constexpr Mat33& operator+=(T s) noexcept {
            m_row[0] += s;
            m_row[1] += s;
            m_row[2] += s;
            return *this;
        }

        NOA_HD constexpr Mat33& operator-=(T s) noexcept {
            m_row[0] -= s;
            m_row[1] -= s;
            m_row[2] -= s;
            return *this;
        }

        NOA_HD constexpr Mat33& operator*=(T s) noexcept {
            m_row[0] *= s;
            m_row[1] *= s;
            m_row[2] *= s;
            return *this;
        }

        NOA_HD constexpr Mat33& operator/=(T s) noexcept {
            m_row[0] /= s;
            m_row[1] /= s;
            m_row[2] /= s;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        friend NOA_HD constexpr Mat33 operator+(Mat33 m) noexcept {
            return m;
        }

        friend NOA_HD constexpr Mat33 operator-(Mat33 m) noexcept {
            return Mat33(-m[0], -m[1], -m[2]);
        }

        // -- Binary arithmetic operators --
        friend NOA_HD constexpr Mat33 operator+(Mat33 m1, Mat33 m2) noexcept {
            return Mat33(m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2]);
        }

        friend NOA_HD constexpr Mat33 operator+(T s, Mat33 m) noexcept {
            return Mat33(s + m[0], s + m[1], s + m[2]);
        }

        friend NOA_HD constexpr Mat33 operator+(Mat33 m, T s) noexcept {
            return Mat33(m[0] + s, m[1] + s, m[2] + s);
        }

        friend NOA_HD constexpr Mat33 operator-(Mat33 m1, Mat33 m2) noexcept {
            return Mat33(m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2]);
        }

        friend NOA_HD constexpr Mat33 operator-(T s, Mat33 m) noexcept {
            return Mat33(s - m[0], s - m[1], s - m[2]);
        }

        friend NOA_HD constexpr Mat33 operator-(Mat33 m, T s) noexcept {
            return Mat33(m[0] - s, m[1] - s, m[2] - s);
        }

        friend NOA_HD constexpr Mat33 operator*(Mat33 m1, Mat33 m2) noexcept {
            Mat33 out(m1);
            out *= m2;
            return out;
        }

        friend NOA_HD constexpr Mat33 operator*(T s, Mat33 m) noexcept {
            return Mat33(m[0] * s, m[1] * s, m[2] * s);
        }

        friend NOA_HD constexpr Mat33 operator*(Mat33 m, T s) noexcept {
            return Mat33(m[0] * s, m[1] * s, m[2] * s);
        }

        friend NOA_HD constexpr Float3<T> operator*(Mat33 m, const Float3<T>& column) noexcept {
            return Float3<T>(math::dot(m[0], column),
                             math::dot(m[1], column),
                             math::dot(m[2], column));
        }

        friend NOA_HD constexpr Float3<T> operator*(const Float3<T>& row, Mat33 m) noexcept {
            return Float3<T>(math::dot(Float3<T>(m[0][0], m[1][0], m[2][0]), row),
                             math::dot(Float3<T>(m[0][1], m[1][1], m[2][1]), row),
                             math::dot(Float3<T>(m[0][2], m[1][2], m[2][2]), row));
        }

        friend NOA_HD constexpr Mat33 operator/(Mat33 m1, Mat33 m2) noexcept {
            Mat33 out(m1);
            out /= m2;
            return out;
        }

        friend NOA_HD constexpr Mat33 operator/(T s, Mat33 m) noexcept {
            return Mat33(s / m[0], s / m[1], s / m[2]);
        }

        friend NOA_HD constexpr Mat33 operator/(Mat33 m, T s) noexcept {
            return Mat33(m[0] / s, m[1] / s, m[2] / s);
        }

        friend NOA_HD constexpr Float3<T> operator/(Mat33 m, const Float3<T>& column) noexcept {
            return math::inverse(m) * column;
        }

        friend NOA_HD constexpr Float3<T> operator/(const Float3<T>& row, Mat33 m) noexcept {
            return row * math::inverse(m);
        }

        friend NOA_HD constexpr bool operator==(Mat33 m1, Mat33 m2) noexcept {
            return all(m1[0] == m2[0]) && all(m1[1] == m2[1]) && all(m1[2] == m2[2]);
        }

        friend NOA_HD constexpr bool operator!=(Mat33 m1, Mat33 m2) noexcept {
            return all(m1[0] != m2[0]) && all(m1[1] != m2[1]) && all(m1[2] != m2[2]);
        }

    private:
        Float3<T> m_row[ROWS];
    };

    namespace math {
        /// Multiplies matrix \a lhs by matrix \a rhs element-wise, i.e. `out[i][j] = lhs[i][j] * rhs[i][j]`.
        template<typename T>
        NOA_IHD constexpr Mat33<T> elementMultiply(const Mat33<T>& m1, const Mat33<T>& m2) noexcept {
            Mat33<T> out;
            for (size_t i = 0; i < Mat33<T>::ROWS; ++i)
                out[i] = m1[i] * m2[i];
            return out;
        }

        /// Given the column vector \a column and row vector \a row,
        /// computes the linear algebraic matrix multiply `c * r`.
        template<typename T>
        NOA_IHD constexpr Mat33<T> outerProduct(Float3<T> column, Float3<T> row) noexcept {
            return Mat33<T>(column.x * row.x, column.x * row.y, column.x * row.z,
                            column.y * row.x, column.y * row.y, column.y * row.z,
                            column.z * row.x, column.z * row.y, column.z * row.z);
        }

        template<typename T>
        NOA_IHD constexpr Mat33<T> transpose(Mat33<T> m) noexcept {
            return Mat33<T>(m[0][0], m[1][0], m[2][0],
                            m[0][1], m[1][1], m[2][1],
                            m[0][2], m[1][2], m[2][2]);
        }

        template<typename T>
        NOA_IHD constexpr T determinant(Mat33<T> m) noexcept {
            return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                   m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                   m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        }

        template<typename T>
        NOA_HD constexpr Mat33<T> inverse(Mat33<T> m) noexcept {
            T det = determinant(m);
            NOA_ASSERT(!isEqual(det, static_cast<T>(0))); // non singular
            T one_over_determinant = 1 / det;
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

        template<uint ULP = 2, typename T>
        NOA_IHD constexpr bool isEqual(Mat33<T> m1, Mat33<T> m2, T e = 1e-6f) noexcept {
            return all(isEqual<ULP>(m1[0], m2[0], e)) &&
                   all(isEqual<ULP>(m1[1], m2[1], e)) &&
                   all(isEqual<ULP>(m1[2], m2[2], e));
        }
    }

    using float33_t = Mat33<float>;
    using double33_t = Mat33<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 9> toArray(Mat33<T> v) noexcept {
        return {v[0][0], v[0][1], v[0][2],
                v[1][0], v[1][1], v[1][2],
                v[2][0], v[2][1], v[2][2]};
    }

    template<>
    NOA_IH std::string string::typeName<float33_t>() { return "float33"; }
    template<>
    NOA_IH std::string string::typeName<double33_t>() { return "double33"; }
}
