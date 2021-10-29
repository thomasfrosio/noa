#ifndef NOA_INCLUDE_MAT22_
#error "This should not be directly included"
#endif

#include "noa/common/Assert.h"

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr Float2<T>& Mat22<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->ROWS);
        return m_row[i];
    }

    template<typename T>
    constexpr const Float2<T>& Mat22<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->ROWS);
        return m_row[i];
    }

    // -- Conversion constructors --

    template<typename T>
    constexpr Mat22<T>::Mat22() noexcept
            : m_row{Float2<T>(1, 0),
                    Float2<T>(0, 1)} {}

    template<typename T>
    template<typename U>
    constexpr Mat22<T>::Mat22(U s) noexcept
            : m_row{Float2<T>(s, 0),
                    Float2<T>(0, s)} {}

    template<typename T>
    template<typename U>
    constexpr Mat22<T>::Mat22(const Float2<U>& v) noexcept
            : m_row{Float2<T>(v.x, 0),
                    Float2<T>(0, v.y)} {}

    template<typename T>
    template<typename U>
    constexpr Mat22<T>::Mat22(const Mat22<U>& m) noexcept
            : m_row{Float2<T>(m[0]),
                    Float2<T>(m[1])} {}

    template<typename T>
    template<typename U>
    constexpr Mat22<T>::Mat22(const Mat33<U>& m) noexcept
            : m_row{Float2<T>(m[0][0], m[0][1]),
                    Float2<T>(m[1][0], m[1][1])} {}

    template<typename T>
    template<typename U>
    constexpr Mat22<T>::Mat22(const Mat23<U>& m) noexcept
            : m_row{Float2<T>(m[0][0], m[0][1]),
                    Float2<T>(m[1][0], m[1][1])} {}

    template<typename T>
    template<typename X00, typename X01,
             typename Y10, typename Y11>
    constexpr Mat22<T>::Mat22(X00 x00, X01 x01,
                              Y10 y10, Y11 y11) noexcept
            : m_row{Float2<T>(x00, x01),
                    Float2<T>(y10, y11)} {}

    template<typename T>
    template<typename V0, typename V1>
    constexpr Mat22<T>::Mat22(const Float2<V0>& r0, const Float2<V1>& r1) noexcept
            : m_row{Float2<T>(r0),
                    Float2<T>(r1)} {}

    template<typename T>
    template<typename V0, typename V1>
    constexpr Mat22<T>::Mat22(const Int2<V0>& r0, const Int2<V1>& r1) noexcept
            : m_row{Float2<T>(r0),
                    Float2<T>(r1)} {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Mat22<T>& Mat22<T>::operator=(const Mat22<U>& m) noexcept {
        m_row[0] = m[0];
        m_row[1] = m[1];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat22<T>& Mat22<T>::operator+=(const Mat22<U>& m) noexcept {
        m_row[0] += m[0];
        m_row[1] += m[1];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat22<T>& Mat22<T>::operator-=(const Mat22<U>& m) noexcept {
        m_row[0] -= m[0];
        m_row[1] -= m[1];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat22<T>& Mat22<T>::operator*=(const Mat22<U>& m) noexcept {
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
    constexpr Mat22<T>& Mat22<T>::operator/=(const Mat22<U>& m) noexcept {
        *this *= math::inverse(m);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat22<T>& Mat22<T>::operator+=(U s) noexcept {
        m_row[0] += s;
        m_row[1] += s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat22<T>& Mat22<T>::operator-=(U s) noexcept {
        m_row[0] -= s;
        m_row[1] -= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat22<T>& Mat22<T>::operator*=(U s) noexcept {
        m_row[0] *= s;
        m_row[1] *= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat22<T>& Mat22<T>::operator/=(U s) noexcept {
        m_row[0] /= s;
        m_row[1] /= s;
        return *this;
    }

    // -- Unary operators --

    template<typename T>
    constexpr Mat22<T> operator+(const Mat22<T>& m) noexcept {
        return m;
    }

    template<typename T>
    constexpr Mat22<T> operator-(const Mat22<T>& m) noexcept {
        return Mat22<T>(-m[0], -m[1]);
    }

    // -- Binary arithmetic operators --

    template<typename T>
    constexpr Mat22<T> operator+(const Mat22<T>& m1, const Mat22<T>& m2) noexcept {
        return Mat22<T>(m1[0] + m2[0], m1[1] + m2[1]);
    }
    template<typename T>
    constexpr Mat22<T> operator+(T s, const Mat22<T>& m) noexcept {
        return Mat22<T>(s + m[0], s + m[1]);
    }
    template<typename T>
    constexpr Mat22<T> operator+(const Mat22<T>& m, T s) noexcept {
        return Mat22<T>(m[0] + s, m[1] + s);
    }

    template<typename T>
    constexpr Mat22<T> operator-(const Mat22<T>& m1, const Mat22<T>& m2) noexcept {
        return Mat22<T>(m1[0] - m2[0], m1[1] - m2[1]);
    }
    template<typename T>
    constexpr Mat22<T> operator-(T s, const Mat22<T>& m) noexcept {
        return Mat22<T>(s - m[0], s - m[1]);
    }
    template<typename T>
    constexpr Mat22<T> operator-(const Mat22<T>& m, T s) noexcept {
        return Mat22<T>(m[0] - s, m[1] - s);
    }

    template<typename T>
    constexpr Mat22<T> operator*(const Mat22<T>& m1, const Mat22<T>& m2) noexcept {
        Mat22<T> out(m1);
        out *= m2;
        return out;
    }
    template<typename T>
    constexpr Mat22<T> operator*(T s, const Mat22<T>& m) noexcept {
        return Mat22<T>(m[0] * s,
                        m[1] * s);
    }
    template<typename T>
    constexpr Mat22<T> operator*(const Mat22<T>& m, T s) noexcept {
        return Mat22<T>(m[0] * s,
                        m[1] * s);
    }
    template<typename T>
    constexpr Float2<T> operator*(const Mat22<T>& m, const Float2<T>& column) noexcept {
        return Float2<T>(math::dot(m[0], column),
                         math::dot(m[1], column));
    }
    template<typename T>
    constexpr Float2<T> operator*(const Float2<T>& row, const Mat22<T>& m) noexcept {
        return Float2<T>(math::dot(Float2<T>(m[0][0], m[1][0]), row),
                         math::dot(Float2<T>(m[0][1], m[1][1]), row));
    }

    template<typename T>
    constexpr Mat22<T> operator/(const Mat22<T>& m1, const Mat22<T>& m2) noexcept {
        Mat22<T> out(m1);
        out /= m2;
        return out;
    }
    template<typename T>
    constexpr Mat22<T> operator/(T s, const Mat22<T>& m) noexcept {
        return Mat22<T>(s / m[0],
                        s / m[1]);
    }
    template<typename T>
    constexpr Mat22<T> operator/(const Mat22<T>& m, T s) noexcept {
        return Mat22<T>(m[0] / s,
                        m[1] / s);
    }
    template<typename T>
    constexpr Float2<T> operator/(const Mat22<T>& m, const Float2<T>& column) noexcept {
        return math::inverse(m) * column;
    }
    template<typename T>
    constexpr Float2<T> operator/(const Float2<T>& row, const Mat22<T>& m) noexcept {
        return row * math::inverse(m);
    }

    template<typename T>
    constexpr bool operator==(const Mat22<T>& m1, const Mat22<T>& m2) noexcept {
        return all(m1[0] == m2[0]) && all(m1[1] == m2[1]);
    }
    template<typename T>
    constexpr bool operator!=(const Mat22<T>& m1, const Mat22<T>& m2) noexcept {
        return all(m1[0] != m2[0]) && all(m1[1] != m2[1]);
    }

    namespace math {
        template<typename T>
        constexpr Mat22<T> elementMultiply(const Mat22<T>& m1, const Mat22<T>& m2) noexcept {
            Mat22<T> out;
            for (size_t i = 0; i < Mat22<T>::ROWS; ++i)
                out[i] = m1[i] * m2[i];
            return out;
        }

        template<typename T>
        constexpr Mat22<T> outerProduct(const Float2<T>& column, const Float2<T>& row) noexcept {
            return Mat22<T>(column.x * row.x, column.x * row.y,
                            column.y * row.x, column.y * row.y);
        }

        template<typename T>
        constexpr Mat22<T> transpose(const Mat22<T>& m) noexcept {
            return Mat22<T>(m[0][0], m[1][0],
                            m[0][1], m[1][1]);
        }

        template<typename T>
        constexpr T determinant(const Mat22<T>& m) noexcept {
            return m[0][0] * m[1][1] - m[0][1] * m[1][0];
        }

        template<typename T>
        constexpr Mat22<T> inverse(const Mat22<T>& m) noexcept {
            T det = determinant(m);
            NOA_ASSERT(!isEqual(det, static_cast<T>(0))); // non singular
            T one_over_determinant = static_cast<T>(1) / det;
            return Mat22<T>(+m[1][1] * one_over_determinant,
                            -m[0][1] * one_over_determinant,
                            -m[1][0] * one_over_determinant,
                            +m[0][0] * one_over_determinant);
        }

        template<uint ULP, typename T>
        constexpr bool isEqual(const Mat22<T>& m1, const Mat22<T>& m2, T e) {
            return all(isEqual<ULP>(m1[0], m2[0], e)) && all(isEqual<ULP>(m1[1], m2[1], e));
        }
    }
}
