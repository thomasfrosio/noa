/// \file noa/common/types/Mat2.h
/// \author Thomas - ffyr2w
/// \date 2 Jun 2021
/// A 2x2 floating-point matrix.

#pragma once

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Float2.h"

namespace noa {
    template<typename T> class Mat3;
    template<typename T> class Float2;
    template<typename T> class Int2;

    /// A 2x2 floating-point matrix.
    /// \note The indexing is "row-first" (as opposed to "column-first", like in OpenGL Math),
    ///       i.e. M[r][c] with r = row index and c = column index. All indexes starts from 0.
    template<typename T>
    class Mat2 {
    private:
        static constexpr uint ROWS = 2U;
        static constexpr uint COLS = 2U;
        Float2<T> m_row[ROWS];

    public: // Type definitions
        typedef T value_type;

    public: // Component accesses
        NOA_HD static constexpr size_t length() noexcept { return 2; }
        NOA_HD static constexpr size_t elements() noexcept { return 4; }
        NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr Float2<T>& operator[](size_t i);
        NOA_HD constexpr const Float2<T>& operator[](size_t i) const;

    public: // (Conversion) Constructors
        NOA_HD constexpr Mat2() noexcept; // identity matrix
        template<typename U> NOA_HD constexpr explicit Mat2(U s) noexcept; // equivalent to Mat2() * s
        template<typename U> NOA_HD constexpr explicit Mat2(const Float2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat2(const Mat2<U>& m) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat2(const Mat3<U>& m) noexcept;

        template<typename X00, typename X01,
                 typename Y10, typename Y11>
        NOA_HD constexpr Mat2(X00 x00, X01 x01,
                              Y10 y10, Y11 y11) noexcept;

        template<typename V0, typename V1>
        NOA_HD constexpr Mat2(const Float2<V0>& r0,
                              const Float2<V1>& r1) noexcept;

        template<typename V0, typename V1>
        NOA_HD constexpr Mat2(const Int2<V0>& r0,
                              const Int2<V1>& r1) noexcept;

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Mat2<T>& operator=(const Mat2<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat2<T>& operator+=(const Mat2<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat2<T>& operator-=(const Mat2<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat2<T>& operator*=(const Mat2<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat2<T>& operator/=(const Mat2<U>& m) noexcept;

        template<typename U> NOA_HD constexpr Mat2<T>& operator+=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat2<T>& operator-=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat2<T>& operator*=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat2<T>& operator/=(U s) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_IHD constexpr Mat2<T> operator+(const Mat2<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat2<T> operator-(const Mat2<T>& m) noexcept;

    // -- Binary operators --

    template<typename T> NOA_IHD constexpr Mat2<T> operator+(const Mat2<T>& m1, const Mat2<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat2<T> operator+(T s, const Mat2<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat2<T> operator+(const Mat2<T>& m, T s) noexcept;

    template<typename T> NOA_IHD constexpr Mat2<T> operator-(const Mat2<T>& m1, const Mat2<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat2<T> operator-(T s, const Mat2<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat2<T> operator-(const Mat2<T>& m, T s) noexcept;

    template<typename T> NOA_IHD constexpr Mat2<T> operator*(const Mat2<T>& m1, const Mat2<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat2<T> operator*(T s, const Mat2<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat2<T> operator*(const Mat2<T>& m, T s) noexcept;
    template<typename T> NOA_IHD constexpr Float2<T> operator*(const Mat2<T>& m, const Float2<T>& column) noexcept;
    template<typename T> NOA_IHD constexpr Float2<T> operator*(const Float2<T>& row, const Mat2<T>& m) noexcept;

    template<typename T> NOA_IHD constexpr Mat2<T> operator/(const Mat2<T>& m1, const Mat2<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat2<T> operator/(T s, const Mat2<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat2<T> operator/(const Mat2<T>& m, T s) noexcept;
    template<typename T> NOA_IHD constexpr Float2<T> operator/(const Mat2<T>& m, const Float2<T>& column) noexcept;
    template<typename T> NOA_IHD constexpr Float2<T> operator/(const Float2<T>& row, const Mat2<T>& m) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_IHD constexpr bool operator==(const Mat2<T>& m1, const Mat2<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr bool operator!=(const Mat2<T>& m1, const Mat2<T>& m2) noexcept;

    namespace math {
        /// Multiplies matrix \a lhs by matrix \a rhs element-wise, i.e. `out[i][j] = lhs[i][j] * rhs[i][j]`.
        template<typename T> NOA_IHD constexpr Mat2<T> elementMultiply(const Mat2<T>& m1, const Mat2<T>& m2) noexcept;

        /// Given the column vector \a column and row vector \a row,
        /// computes the linear algebraic matrix multiply `c * r`.
        template<typename T> NOA_IHD constexpr Mat2<T> outerProduct(const Float2<T>& column,
                                                                    const Float2<T>& row) noexcept;

        template<typename T> NOA_IHD constexpr Mat2<T> transpose(const Mat2<T>& m) noexcept;
        template<typename T> NOA_IHD constexpr T determinant(const Mat2<T>& m) noexcept;
        template<typename T> NOA_IHD constexpr Mat2<T> inverse(const Mat2<T>& m) noexcept;

        template<uint ULP = 2, typename T>
        NOA_IHD constexpr bool isEqual(const Mat2<T>& m1, const Mat2<T>& m2, T e = 1e-6f);
    }

    using float22_t = Mat2<float>;
    using double22_t = Mat2<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 4> toArray(const Mat2<T>& v) noexcept {
        return {v[0][0], v[0][1],
                v[1][0], v[1][1]};
    }

    template<> NOA_IH std::string string::typeName<float22_t>() { return "float22"; }
    template<> NOA_IH std::string string::typeName<double22_t>() { return "double22"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Mat2<T>& m) {
        os << string::format("({},{})", m[0], m[1]);
        return os;
    }
}

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr Float2<T>& Mat2<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->length());
        return m_row[i];
    }

    template<typename T>
    constexpr const Float2<T>& Mat2<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->length());
        return m_row[i];
    }

    // -- Conversion constructors --

    template<typename T>
    constexpr Mat2<T>::Mat2() noexcept
            : m_row{Float2<T>(1, 0),
                    Float2<T>(0, 1)} {}

    template<typename T>
    template<typename U>
    constexpr Mat2<T>::Mat2(U s) noexcept
            : m_row{Float2<T>(s, 0),
                    Float2<T>(0, s)} {}

    template<typename T>
    template<typename U>
    constexpr Mat2<T>::Mat2(const Float2<U>& v) noexcept
            : m_row{Float2<T>(v.x, 0),
                    Float2<T>(0, v.y)} {}

    template<typename T>
    template<typename U>
    constexpr Mat2<T>::Mat2(const Mat2<U>& m) noexcept
            : m_row{Float2<T>(m[0]),
                    Float2<T>(m[1])} {}

    template<typename T>
    template<typename U>
    constexpr Mat2<T>::Mat2(const Mat3<U>& m) noexcept
            : m_row{Float2<T>(m[0][0], m[0][1]),
                    Float2<T>(m[1][0], m[1][1])} {}

    template<typename T>
    template<typename X00, typename X01,
             typename Y10, typename Y11>
    constexpr Mat2<T>::Mat2(X00 x00, X01 x01,
                            Y10 y10, Y11 y11) noexcept
            : m_row{Float2<T>(x00, x01),
                    Float2<T>(y10, y11)} {}

    template<typename T>
    template<typename V0, typename V1>
    constexpr Mat2<T>::Mat2(const Float2<V0>& r0, const Float2<V1>& r1) noexcept
            : m_row{Float2<T>(r0),
                    Float2<T>(r1)} {}

    template<typename T>
    template<typename V0, typename V1>
    constexpr Mat2<T>::Mat2(const Int2<V0>& r0, const Int2<V1>& r1) noexcept
            : m_row{Float2<T>(r0),
                    Float2<T>(r1)} {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Mat2<T>& Mat2<T>::operator=(const Mat2<U>& m) noexcept {
        m_row[0] = m[0];
        m_row[1] = m[1];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat2<T>& Mat2<T>::operator+=(const Mat2<U>& m) noexcept {
        m_row[0] += m[0];
        m_row[1] += m[1];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat2<T>& Mat2<T>::operator-=(const Mat2<U>& m) noexcept {
        m_row[0] -= m[0];
        m_row[1] -= m[1];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat2<T>& Mat2<T>::operator*=(const Mat2<U>& m) noexcept {
        const T A00 = m_row[0][0];
        const T A01 = m_row[0][1];
        const T A10 = m_row[1][0];
        const T A11 = m_row[1][1];

        const T B00 = static_cast<T>(m[0][0]);
        const T B01 = static_cast<T>(m[0][1]);
        const T B10 = static_cast<T>(m[1][0]);
        const T B11 = static_cast<T>(m[1][1]);

        m_row[0][0] = A00 * B00 + A01 * B10;
        m_row[0][1] = A00 * B01 + A01 * B11;
        m_row[1][0] = A10 * B00 + A11 * B10;
        m_row[1][1] = A10 * B01 + A11 * B11;

        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat2<T>& Mat2<T>::operator/=(const Mat2<U>& m) noexcept {
        *this *= math::inverse(m);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat2<T>& Mat2<T>::operator+=(U s) noexcept {
        m_row[0] += s;
        m_row[1] += s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat2<T>& Mat2<T>::operator-=(U s) noexcept {
        m_row[0] -= s;
        m_row[1] -= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat2<T>& Mat2<T>::operator*=(U s) noexcept {
        m_row[0] *= s;
        m_row[1] *= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat2<T>& Mat2<T>::operator/=(U s) noexcept {
        m_row[0] /= s;
        m_row[1] /= s;
        return *this;
    }

    // -- Unary operators --

    template<typename T>
    constexpr Mat2<T> operator+(const Mat2<T>& m) noexcept {
        return m;
    }

    template<typename T>
    constexpr Mat2<T> operator-(const Mat2<T>& m) noexcept {
        return Mat2<T>(-m[0], -m[1]);
    }

    // -- Binary arithmetic operators --

    template<typename T>
    constexpr Mat2<T> operator+(const Mat2<T>& m1, const Mat2<T>& m2) noexcept {
        return Mat2<T>(m1[0] + m2[0], m1[1] + m2[1]);
    }
    template<typename T>
    constexpr Mat2<T> operator+(T s, const Mat2<T>& m) noexcept {
        return Mat2<T>(s + m[0], s + m[1]);
    }
    template<typename T>
    constexpr Mat2<T> operator+(const Mat2<T>& m, T s) noexcept {
        return Mat2<T>(m[0] + s, m[1] + s);
    }

    template<typename T>
    constexpr Mat2<T> operator-(const Mat2<T>& m1, const Mat2<T>& m2) noexcept {
        return Mat2<T>(m1[0] - m2[0], m1[1] - m2[1]);
    }
    template<typename T>
    constexpr Mat2<T> operator-(T s, const Mat2<T>& m) noexcept {
        return Mat2<T>(s - m[0], s - m[1]);
    }
    template<typename T>
    constexpr Mat2<T> operator-(const Mat2<T>& m, T s) noexcept {
        return Mat2<T>(m[0] - s, m[1] - s);
    }

    template<typename T>
    constexpr Mat2<T> operator*(const Mat2<T>& m1, const Mat2<T>& m2) noexcept {
        Mat2<T> out(m1);
        out *= m2;
        return out;
    }
    template<typename T>
    constexpr Mat2<T> operator*(T s, const Mat2<T>& m) noexcept {
        return Mat2<T>(m[0] * s,
                       m[1] * s);
    }
    template<typename T>
    constexpr Mat2<T> operator*(const Mat2<T>& m, T s) noexcept {
        return Mat2<T>(m[0] * s,
                       m[1] * s);
    }
    template<typename T>
    constexpr Float2<T> operator*(const Mat2<T>& m, const Float2<T>& column) noexcept {
        return Float2<T>(math::dot(m[0], column),
                         math::dot(m[1], column));
    }
    template<typename T>
    constexpr Float2<T> operator*(const Float2<T>& row, const Mat2<T>& m) noexcept {
        return Float2<T>(math::dot(Float2<T>(m[0][0], m[1][0]), row),
                         math::dot(Float2<T>(m[0][1], m[1][1]), row));
    }

    template<typename T>
    constexpr Mat2<T> operator/(const Mat2<T>& m1, const Mat2<T>& m2) noexcept {
        Mat2<T> out(m1);
        out /= m2;
        return out;
    }
    template<typename T>
    constexpr Mat2<T> operator/(T s, const Mat2<T>& m) noexcept {
        return Mat2<T>(s / m[0],
                       s / m[1]);
    }
    template<typename T>
    constexpr Mat2<T> operator/(const Mat2<T>& m, T s) noexcept {
        return Mat2<T>(m[0] / s,
                       m[1] / s);
    }
    template<typename T>
    constexpr Float2<T> operator/(const Mat2<T>& m, const Float2<T>& column) noexcept {
        return math::inverse(m) * column;
    }
    template<typename T>
    constexpr Float2<T> operator/(const Float2<T>& row, const Mat2<T>& m) noexcept {
        return row * math::inverse(m);
    }

    template<typename T>
    constexpr bool operator==(const Mat2<T>& m1, const Mat2<T>& m2) noexcept {
        return all(m1[0] == m2[0]) && all(m1[1] == m2[1]);
    }
    template<typename T>
    constexpr bool operator!=(const Mat2<T>& m1, const Mat2<T>& m2) noexcept {
        return all(m1[0] != m2[0]) && all(m1[1] != m2[1]);
    }

    namespace math {
        template<typename T>
        constexpr Mat2<T> elementMultiply(const Mat2<T>& m1, const Mat2<T>& m2) noexcept {
            Mat2<T> out;
            for (size_t i = 0; i < Mat2<T>::length(); ++i)
                out[i] = m1[i] * m2[i];
            return out;
        }

        template<typename T>
        constexpr Mat2<T> outerProduct(const Float2<T>& column, const Float2<T>& row) noexcept {
            return Mat2<T>(column.x * row.x, column.x * row.y,
                           column.y * row.x, column.y * row.y);
        }

        template<typename T>
        constexpr Mat2<T> transpose(const Mat2<T>& m) noexcept {
            return Mat2<T>(m[0][0], m[1][0],
                           m[0][1], m[1][1]);
        }

        template<typename T>
        constexpr T determinant(const Mat2<T>& m) noexcept {
            return m[0][0] * m[1][1] - m[0][1] * m[1][0];
        }

        template<typename T>
        constexpr Mat2<T> inverse(const Mat2<T>& m) noexcept {
            T det = determinant(m);
            NOA_ASSERT(!isEqual(det, static_cast<T>(0))); // non singular
            T one_over_determinant = static_cast<T>(1) / det;
            return Mat2<T>(+m[1][1] * one_over_determinant,
                           -m[0][1] * one_over_determinant,
                           -m[1][0] * one_over_determinant,
                           +m[0][0] * one_over_determinant);
        }

        template<uint ULP, typename T>
        constexpr bool isEqual(const Mat2<T>& m1, const Mat2<T>& m2, T e) {
            return all(isEqual<ULP>(m1[0], m2[0], e)) && all(isEqual<ULP>(m1[1], m2[1], e));
        }
    }
}
