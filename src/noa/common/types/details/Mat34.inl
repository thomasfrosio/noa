#ifndef NOA_INCLUDE_MAT34_
#error "This should not be directly included"
#endif

#include "noa/common/Assert.h"

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr Float4<T>& Mat34<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->ROWS);
        return m_row[i];
    }

    template<typename T>
    constexpr const Float4<T>& Mat34<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->ROWS);
        return m_row[i];
    }

    // -- Conversion constructors --

    template<typename T>
    constexpr Mat34<T>::Mat34() noexcept
            : m_row{Float4<T>(1, 0, 0, 0),
                    Float4<T>(0, 1, 0, 0),
                    Float4<T>(0, 0, 1, 0)} {}

    template<typename T>
    template<typename U>
    constexpr Mat34<T>::Mat34(U s) noexcept
            : m_row{Float4<T>(s, 0, 0, 0),
                    Float4<T>(0, s, 0, 0),
                    Float4<T>(0, 0, s, 0)} {}

    template<typename T>
    template<typename U>
    constexpr Mat34<T>::Mat34(const Float4<U>& v) noexcept
            : m_row{Float4<T>(v.x, 0, 0, 0),
                    Float4<T>(0, v.y, 0, 0),
                    Float4<T>(0, 0, v.z, 0)} {}

    template<typename T>
    template<typename U>
    constexpr Mat34<T>::Mat34(const Float3<U>& v) noexcept
            : m_row{Float4<T>(v.x, 0, 0, 0),
                    Float4<T>(0, v.y, 0, 0),
                    Float4<T>(0, 0, v.z, 0)} {}

    template<typename T>
    template<typename U>
    constexpr Mat34<T>::Mat34(const Mat44<U>& m) noexcept
            : m_row{Float4<T>(m[0]),
                    Float4<T>(m[1]),
                    Float4<T>(m[2])} {}

    template<typename T>
    template<typename U>
    constexpr Mat34<T>::Mat34(const Mat34<U>& m) noexcept
            : m_row{Float4<T>(m[0]),
                    Float4<T>(m[1]),
                    Float4<T>(m[2])} {}

    template<typename T>
    template<typename U>
    constexpr Mat34<T>::Mat34(const Mat33<U>& m) noexcept
            : m_row{Float4<T>(m[0][0], m[0][1], m[0][2], 0),
                    Float4<T>(m[1][0], m[1][1], m[1][2], 0),
                    Float4<T>(m[2][0], m[2][1], m[2][2], 0)} {}

    template<typename T>
    template<typename U, typename V>
    constexpr Mat34<T>::Mat34(const Mat33<U>& m, const Float3<V>& v) noexcept
            : m_row{Float4<T>(m[0][0], m[0][1], m[0][2], v[0]),
                    Float4<T>(m[1][0], m[1][1], m[1][2], v[1]),
                    Float4<T>(m[2][0], m[2][1], m[2][2], v[2])} {}

    template<typename T>
    template<typename X00, typename X01, typename X02, typename X03,
             typename Y10, typename Y11, typename Y12, typename Y13,
             typename Z20, typename Z21, typename Z22, typename Z23>
    constexpr Mat34<T>::Mat34(X00 x00, X01 x01, X02 x02, X03 x03,
                              Y10 y10, Y11 y11, Y12 y12, Y13 y13,
                              Z20 z20, Z21 z21, Z22 z22, Z23 z23) noexcept
            : m_row{Float4<T>(x00, x01, x02, x03),
                    Float4<T>(y10, y11, y12, y13),
                    Float4<T>(z20, z21, z22, z23)} {}

    template<typename T>
    template<typename V0, typename V1, typename V2>
    constexpr Mat34<T>::Mat34(const Float4<V0>& r0, const Float4<V1>& r1, const Float4<V2>& r2) noexcept
            : m_row{Float4<T>(r0),
                    Float4<T>(r1),
                    Float4<T>(r2)} {}

    template<typename T>
    template<typename V0, typename V1, typename V2>
    constexpr Mat34<T>::Mat34(const Int4<V0>& r0, const Int4<V1>& r1, const Int4<V2>& r2) noexcept
            : m_row{Float4<T>(r0),
                    Float4<T>(r1),
                    Float4<T>(r2)} {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Mat34<T>& Mat34<T>::operator=(const Mat34<U>& m) noexcept {
        m_row[0] = m[0];
        m_row[1] = m[1];
        m_row[2] = m[2];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat34<T>& Mat34<T>::operator+=(const Mat34<U>& m) noexcept {
        m_row[0] += m[0];
        m_row[1] += m[1];
        m_row[2] += m[2];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat34<T>& Mat34<T>::operator-=(const Mat34<U>& m) noexcept {
        m_row[0] -= m[0];
        m_row[1] -= m[1];
        m_row[2] -= m[2];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat34<T>& Mat34<T>::operator+=(U s) noexcept {
        m_row[0] += s;
        m_row[1] += s;
        m_row[2] += s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat34<T>& Mat34<T>::operator-=(U s) noexcept {
        m_row[0] -= s;
        m_row[1] -= s;
        m_row[2] -= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat34<T>& Mat34<T>::operator*=(U s) noexcept {
        m_row[0] *= s;
        m_row[1] *= s;
        m_row[2] *= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat34<T>& Mat34<T>::operator/=(U s) noexcept {
        m_row[0] /= s;
        m_row[1] /= s;
        m_row[2] /= s;
        return *this;
    }

    // -- Unary operators --

    template<typename T>
    constexpr Mat34<T> operator+(const Mat34<T>& m) noexcept {
        return m;
    }

    template<typename T>
    constexpr Mat34<T> operator-(const Mat34<T>& m) noexcept {
        return Mat34<T>(-m[0], -m[1], -m[2]);
    }

    // -- Binary arithmetic operators --

    template<typename T>
    constexpr Mat34<T> operator+(const Mat34<T>& m1, const Mat34<T>& m2) noexcept {
        return Mat34<T>(m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2]);
    }
    template<typename T>
    constexpr Mat34<T> operator+(T s, const Mat34<T>& m) noexcept {
        return Mat34<T>(s + m[0], s + m[1], s + m[2]);
    }
    template<typename T>
    constexpr Mat34<T> operator+(const Mat34<T>& m, T s) noexcept {
        return Mat34<T>(m[0] + s, m[1] + s, m[2] + s);
    }

    template<typename T>
    constexpr Mat34<T> operator-(const Mat34<T>& m1, const Mat34<T>& m2) noexcept {
        return Mat34<T>(m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2]);
    }
    template<typename T>
    constexpr Mat34<T> operator-(T s, const Mat34<T>& m) noexcept {
        return Mat34<T>(s - m[0], s - m[1], s - m[2]);
    }
    template<typename T>
    constexpr Mat34<T> operator-(const Mat34<T>& m, T s) noexcept {
        return Mat34<T>(m[0] - s, m[1] - s, m[2] - s);
    }

    template<typename T>
    constexpr Mat34<T> operator*(T s, const Mat34<T>& m) noexcept {
        return Mat34<T>(m[0] * s,
                        m[1] * s,
                        m[2] * s);
    }
    template<typename T>
    constexpr Mat34<T> operator*(const Mat34<T>& m, T s) noexcept {
        return Mat34<T>(m[0] * s,
                        m[1] * s,
                        m[2] * s);
    }
    template<typename T>
    constexpr Float3<T> operator*(const Mat34<T>& m, const Float4<T>& column) noexcept {
        return Float3<T>(math::dot(m[0], column),
                         math::dot(m[1], column),
                         math::dot(m[2], column));
    }
    template<typename T>
    constexpr Float4<T> operator*(const Float3<T>& row, const Mat34<T>& m) noexcept {
        return Float4<T>(math::dot(Float3<T>(m[0][0], m[1][0], m[2][0]), row),
                         math::dot(Float3<T>(m[0][1], m[1][1], m[2][1]), row),
                         math::dot(Float3<T>(m[0][2], m[1][2], m[2][2]), row),
                         math::dot(Float3<T>(m[0][3], m[1][3], m[2][3]), row));
    }

    template<typename T>
    constexpr Mat34<T> operator/(T s, const Mat34<T>& m) noexcept {
        return Mat34<T>(s / m[0],
                        s / m[1],
                        s / m[2]);
    }
    template<typename T>
    constexpr Mat34<T> operator/(const Mat34<T>& m, T s) noexcept {
        return Mat34<T>(m[0] / s,
                        m[1] / s,
                        m[2] / s);
    }

    template<typename T>
    constexpr bool operator==(const Mat34<T>& m1, const Mat34<T>& m2) noexcept {
        return all(m1[0] == m2[0]) && all(m1[1] == m2[1]) && all(m1[2] == m2[2]);
    }
    template<typename T>
    constexpr bool operator!=(const Mat34<T>& m1, const Mat34<T>& m2) noexcept {
        return all(m1[0] != m2[0]) && all(m1[1] != m2[1]) && all(m1[2] != m2[2]);
    }

    namespace math {
        template<typename T>
        constexpr Mat34<T> elementMultiply(const Mat34<T>& m1, const Mat34<T>& m2) noexcept {
            Mat34<T> out;
            for (size_t i = 0; i < Mat34<T>::ROWS; ++i)
                out[i] = m1[i] * m2[i];
            return out;
        }

        template<uint ULP, typename T>
        constexpr bool isEqual(const Mat34<T>& m1, const Mat34<T>& m2, T e) {
            return all(isEqual<ULP>(m1[0], m2[0], e)) &&
                   all(isEqual<ULP>(m1[1], m2[1], e));
        }
    }
}
