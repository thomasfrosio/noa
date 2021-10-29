#ifndef NOA_INCLUDE_MAT33_
#error "This should not be directly included"
#endif

#include "noa/common/Assert.h"

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr Float3<T>& Mat33<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->ROWS);
        return m_row[i];
    }

    template<typename T>
    constexpr const Float3<T>& Mat33<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->ROWS);
        return m_row[i];
    }

    // -- Conversion constructors --

    template<typename T>
    constexpr Mat33<T>::Mat33() noexcept
            : m_row{Float3<T>(1, 0, 0),
                    Float3<T>(0, 1, 0),
                    Float3<T>(0, 0, 1)} {}

    template<typename T>
    template<typename U>
    constexpr Mat33<T>::Mat33(U s) noexcept
            : m_row{Float3<T>(s, 0, 0),
                    Float3<T>(0, s, 0),
                    Float3<T>(0, 0, s)} {}

    template<typename T>
    template<typename U>
    constexpr Mat33<T>::Mat33(const Float3<U>& v) noexcept
            : m_row{Float3<T>(v.x, 0, 0),
                    Float3<T>(0, v.y, 0),
                    Float3<T>(0, 0, v.z)} {}

    template<typename T>
    template<typename U>
    constexpr Mat33<T>::Mat33(const Float2<U>& v) noexcept
            : m_row{Float3<T>(v.x, 0, 0),
                    Float3<T>(0, v.y, 0),
                    Float3<T>(0, 0, 1)} {}

    template<typename T>
    template<typename U>
    constexpr Mat33<T>::Mat33(const Mat44<U>& m) noexcept
            : m_row{Float3<T>(m[0][0], m[0][1], m[0][2]),
                    Float3<T>(m[1][0], m[1][1], m[1][2]),
                    Float3<T>(m[2][0], m[2][1], m[2][2])} {}

    template<typename T>
    template<typename U>
    constexpr Mat33<T>::Mat33(const Mat34<U>& m) noexcept
            : m_row{Float3<T>(m[0][0], m[0][1], m[0][2]),
                    Float3<T>(m[1][0], m[1][1], m[1][2]),
                    Float3<T>(m[2][0], m[2][1], m[2][2])} {}

    template<typename T>
    template<typename U>
    constexpr Mat33<T>::Mat33(const Mat33<U>& m) noexcept
            : m_row{Float3<T>(m[0]),
                    Float3<T>(m[1]),
                    Float3<T>(m[2])} {}

    template<typename T>
    template<typename U>
    constexpr Mat33<T>::Mat33(const Mat23<U>& m) noexcept
            : m_row{Float3<T>(m[0]),
                    Float3<T>(m[1]),
                    Float3<T>(0, 0, 1)} {}

    template<typename T>
    template<typename U>
    constexpr Mat33<T>::Mat33(const Mat22<U>& m) noexcept
            : m_row{Float3<T>(m[0][0], m[0][1], 0),
                    Float3<T>(m[1][0], m[1][1], 0),
                    Float3<T>(0, 0, 1)} {}

    template<typename T>
    template<typename U, typename V>
    constexpr Mat33<T>::Mat33(const Mat22<U>& m, const Float2<V>& v) noexcept
            : m_row{Float3<T>(m[0][0], m[0][1], v[0]),
                    Float3<T>(m[1][0], m[1][1], v[1]),
                    Float3<T>(0, 0, 1)} {}

    template<typename T>
    template<typename X00, typename X01, typename X02,
             typename Y10, typename Y11, typename Y12,
             typename Z20, typename Z21, typename Z22>
    constexpr Mat33<T>::Mat33(X00 x00, X01 x01, X02 x02,
                              Y10 y10, Y11 y11, Y12 y12,
                              Z20 z20, Z21 z21, Z22 z22) noexcept
            : m_row{Float3<T>(x00, x01, x02),
                    Float3<T>(y10, y11, y12),
                    Float3<T>(z20, z21, z22)} {}

    template<typename T>
    template<typename V0, typename V1, typename V2>
    constexpr Mat33<T>::Mat33(const Float3<V0>& r0, const Float3<V1>& r1, const Float3<V2>& r2) noexcept
            : m_row{Float3<T>(r0),
                    Float3<T>(r1),
                    Float3<T>(r2)} {}

    template<typename T>
    template<typename V0, typename V1, typename V2>
    constexpr Mat33<T>::Mat33(const Int3<V0>& r0, const Int3<V1>& r1, const Int3<V2>& r2) noexcept
            : m_row{Float3<T>(r0),
                    Float3<T>(r1),
                    Float3<T>(r2)} {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Mat33<T>& Mat33<T>::operator=(const Mat33<U>& m) noexcept {
        m_row[0] = m[0];
        m_row[1] = m[1];
        m_row[2] = m[2];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat33<T>& Mat33<T>::operator+=(const Mat33<U>& m) noexcept {
        m_row[0] += m[0];
        m_row[1] += m[1];
        m_row[2] += m[2];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat33<T>& Mat33<T>::operator-=(const Mat33<U>& m) noexcept {
        m_row[0] -= m[0];
        m_row[1] -= m[1];
        m_row[2] -= m[2];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat33<T>& Mat33<T>::operator*=(const Mat33<U>& m) noexcept {
        const T A00 = m_row[0][0];
        const T A01 = m_row[0][1];
        const T A02 = m_row[0][2];
        const T A10 = m_row[1][0];
        const T A11 = m_row[1][1];
        const T A12 = m_row[1][2];
        const T A20 = m_row[2][0];
        const T A21 = m_row[2][1];
        const T A22 = m_row[2][2];

        const T B00 = static_cast<T>(m[0][0]);
        const T B01 = static_cast<T>(m[0][1]);
        const T B02 = static_cast<T>(m[0][2]);
        const T B10 = static_cast<T>(m[1][0]);
        const T B11 = static_cast<T>(m[1][1]);
        const T B12 = static_cast<T>(m[1][2]);
        const T B20 = static_cast<T>(m[2][0]);
        const T B21 = static_cast<T>(m[2][1]);
        const T B22 = static_cast<T>(m[2][2]);

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

    template<typename T>
    template<typename U>
    constexpr Mat33<T>& Mat33<T>::operator/=(const Mat33<U>& m) noexcept {
        *this *= math::inverse(m);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat33<T>& Mat33<T>::operator+=(U s) noexcept {
        m_row[0] += s;
        m_row[1] += s;
        m_row[2] += s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat33<T>& Mat33<T>::operator-=(U s) noexcept {
        m_row[0] -= s;
        m_row[1] -= s;
        m_row[2] -= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat33<T>& Mat33<T>::operator*=(U s) noexcept {
        m_row[0] *= s;
        m_row[1] *= s;
        m_row[2] *= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat33<T>& Mat33<T>::operator/=(U s) noexcept {
        m_row[0] /= s;
        m_row[1] /= s;
        m_row[2] /= s;
        return *this;
    }

    // -- Unary operators --

    template<typename T>
    constexpr Mat33<T> operator+(const Mat33<T>& m) noexcept {
        return m;
    }

    template<typename T>
    constexpr Mat33<T> operator-(const Mat33<T>& m) noexcept {
        return Mat33<T>(-m[0], -m[1], -m[2]);
    }

    // -- Binary arithmetic operators --

    template<typename T>
    constexpr Mat33<T> operator+(const Mat33<T>& m1, const Mat33<T>& m2) noexcept {
        return Mat33<T>(m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2]);
    }
    template<typename T>
    constexpr Mat33<T> operator+(T s, const Mat33<T>& m) noexcept {
        return Mat33<T>(s + m[0], s + m[1], s + m[2]);
    }
    template<typename T>
    constexpr Mat33<T> operator+(const Mat33<T>& m, T s) noexcept {
        return Mat33<T>(m[0] + s, m[1] + s, m[2] + s);
    }

    template<typename T>
    constexpr Mat33<T> operator-(const Mat33<T>& m1, const Mat33<T>& m2) noexcept {
        return Mat33<T>(m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2]);
    }
    template<typename T>
    constexpr Mat33<T> operator-(T s, const Mat33<T>& m) noexcept {
        return Mat33<T>(s - m[0], s - m[1], s - m[2]);
    }
    template<typename T>
    constexpr Mat33<T> operator-(const Mat33<T>& m, T s) noexcept {
        return Mat33<T>(m[0] - s, m[1] - s, m[2] - s);
    }

    template<typename T>
    constexpr Mat33<T> operator*(const Mat33<T>& m1, const Mat33<T>& m2) noexcept {
        Mat33<T> out(m1);
        out *= m2;
        return out;
    }
    template<typename T>
    constexpr Mat33<T> operator*(T s, const Mat33<T>& m) noexcept {
        return Mat33<T>(m[0] * s,
                        m[1] * s,
                        m[2] * s);
    }
    template<typename T>
    constexpr Mat33<T> operator*(const Mat33<T>& m, T s) noexcept {
        return Mat33<T>(m[0] * s,
                        m[1] * s,
                        m[2] * s);
    }
    template<typename T>
    constexpr Float3<T> operator*(const Mat33<T>& m, const Float3<T>& column) noexcept {
        return Float3<T>(math::dot(m[0], column),
                         math::dot(m[1], column),
                         math::dot(m[2], column));
    }
    template<typename T>
    constexpr Float3<T> operator*(const Float3<T>& row, const Mat33<T>& m) noexcept {
        return Float3<T>(math::dot(Float3<T>(m[0][0], m[1][0], m[2][0]), row),
                         math::dot(Float3<T>(m[0][1], m[1][1], m[2][1]), row),
                         math::dot(Float3<T>(m[0][2], m[1][2], m[2][2]), row));
    }

    template<typename T>
    constexpr Mat33<T> operator/(const Mat33<T>& m1, const Mat33<T>& m2) noexcept {
        Mat33<T> out(m1);
        out /= m2;
        return out;
    }
    template<typename T>
    constexpr Mat33<T> operator/(T s, const Mat33<T>& m) noexcept {
        return Mat33<T>(s / m[0],
                        s / m[1],
                        s / m[2]);
    }
    template<typename T>
    constexpr Mat33<T> operator/(const Mat33<T>& m, T s) noexcept {
        return Mat33<T>(m[0] / s,
                        m[1] / s,
                        m[2] / s);
    }
    template<typename T>
    constexpr Float3<T> operator/(const Mat33<T>& m, const Float3<T>& column) noexcept {
        return math::inverse(m) * column;
    }
    template<typename T>
    constexpr Float3<T> operator/(const Float3<T>& row, const Mat33<T>& m) noexcept {
        return row * math::inverse(m);
    }

    template<typename T>
    constexpr bool operator==(const Mat33<T>& m1, const Mat33<T>& m2) noexcept {
        return all(m1[0] == m2[0]) && all(m1[1] == m2[1]) && all(m1[2] == m2[2]);
    }
    template<typename T>
    constexpr bool operator!=(const Mat33<T>& m1, const Mat33<T>& m2) noexcept {
        return all(m1[0] != m2[0]) && all(m1[1] != m2[1]) && all(m1[2] != m2[2]);
    }

    namespace math {
        template<typename T>
        constexpr Mat33<T> elementMultiply(const Mat33<T>& m1, const Mat33<T>& m2) noexcept {
            Mat33<T> out;
            for (size_t i = 0; i < Mat33<T>::ROWS; ++i)
                out[i] = m1[i] * m2[i];
            return out;
        }

        template<typename T>
        constexpr Mat33<T> outerProduct(const Float3<T>& column, const Float3<T>& row) noexcept {
            return Mat33<T>(column.x * row.x, column.x * row.y, column.x * row.z,
                            column.y * row.x, column.y * row.y, column.y * row.z,
                            column.z * row.x, column.z * row.y, column.z * row.z);
        }

        template<typename T>
        constexpr Mat33<T> transpose(const Mat33<T>& m) noexcept {
            return Mat33<T>(m[0][0], m[1][0], m[2][0],
                            m[0][1], m[1][1], m[2][1],
                            m[0][2], m[1][2], m[2][2]);
        }

        template<typename T>
        constexpr T determinant(const Mat33<T>& m) noexcept {
            return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                   m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                   m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        }

        template<typename T>
        constexpr Mat33<T> inverse(const Mat33<T>& m) noexcept {
            T det = determinant(m);
            NOA_ASSERT(!isEqual(det, static_cast<T>(0))); // non singular
            T one_over_determinant = static_cast<T>(1) / det;
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

        template<uint ULP, typename T>
        constexpr bool isEqual(const Mat33<T>& m1, const Mat33<T>& m2, T e) {
            return all(isEqual<ULP>(m1[0], m2[0], e)) &&
                   all(isEqual<ULP>(m1[1], m2[1], e)) &&
                   all(isEqual<ULP>(m1[2], m2[2], e));
        }
    }
}
