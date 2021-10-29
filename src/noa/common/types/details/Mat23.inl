#ifndef NOA_INCLUDE_MAT23_
#error "This should not be directly included"
#endif

#include "noa/common/Assert.h"

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr Float3<T>& Mat23<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->ROWS);
        return m_row[i];
    }

    template<typename T>
    constexpr const Float3<T>& Mat23<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->ROWS);
        return m_row[i];
    }

    // -- Conversion constructors --

    template<typename T>
    constexpr Mat23<T>::Mat23() noexcept
            : m_row{Float3<T>(1, 0, 0),
                    Float3<T>(0, 1, 0)} {}

    template<typename T>
    template<typename U>
    constexpr Mat23<T>::Mat23(U s) noexcept
            : m_row{Float3<T>(s, 0, 0),
                    Float3<T>(0, s, 0)} {}

    template<typename T>
    template<typename U>
    constexpr Mat23<T>::Mat23(const Float2<U>& v) noexcept
            : m_row{Float3<T>(v.x, 0, 0),
                    Float3<T>(0, v.y, 0)} {}

    template<typename T>
    template<typename U>
    constexpr Mat23<T>::Mat23(const Mat33<U>& m) noexcept
            : m_row{Float3<T>(m[0]),
                    Float3<T>(m[1])} {}

    template<typename T>
    template<typename U>
    constexpr Mat23<T>::Mat23(const Mat23<U>& m) noexcept
            : m_row{Float3<T>(m[0]),
                    Float3<T>(m[1])} {}

    template<typename T>
    template<typename U>
    constexpr Mat23<T>::Mat23(const Mat22<U>& m) noexcept
            : m_row{Float3<T>(m[0][0], m[0][1], 0),
                    Float3<T>(m[1][0], m[1][1], 0)} {}

    template<typename T>
    template<typename U, typename V>
    constexpr Mat23<T>::Mat23(const Mat22<U>& m, const Float2<V>& v) noexcept
            : m_row{Float3<T>(m[0][0], m[0][1], v[0]),
                    Float3<T>(m[1][0], m[1][1], v[1])} {}

    template<typename T>
    template<typename X00, typename X01, typename X02,
             typename Y10, typename Y11, typename Y12>
    constexpr Mat23<T>::Mat23(X00 x00, X01 x01, X02 x02,
                              Y10 y10, Y11 y11, Y12 y12) noexcept
            : m_row{Float3<T>(x00, x01, x02),
                    Float3<T>(y10, y11, y12)} {}

    template<typename T>
    template<typename V0, typename V1>
    constexpr Mat23<T>::Mat23(const Float3<V0>& r0, const Float3<V1>& r1) noexcept
            : m_row{Float3<T>(r0),
                    Float3<T>(r1)} {}

    template<typename T>
    template<typename V0, typename V1>
    constexpr Mat23<T>::Mat23(const Int3<V0>& r0, const Int3<V1>& r1) noexcept
            : m_row{Float3<T>(r0),
                    Float3<T>(r1)} {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Mat23<T>& Mat23<T>::operator=(const Mat23<U>& m) noexcept {
        m_row[0] = m[0];
        m_row[1] = m[1];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat23<T>& Mat23<T>::operator+=(const Mat23<U>& m) noexcept {
        m_row[0] += m[0];
        m_row[1] += m[1];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat23<T>& Mat23<T>::operator-=(const Mat23<U>& m) noexcept {
        m_row[0] -= m[0];
        m_row[1] -= m[1];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat23<T>& Mat23<T>::operator+=(U s) noexcept {
        m_row[0] += s;
        m_row[1] += s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat23<T>& Mat23<T>::operator-=(U s) noexcept {
        m_row[0] -= s;
        m_row[1] -= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat23<T>& Mat23<T>::operator*=(U s) noexcept {
        m_row[0] *= s;
        m_row[1] *= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat23<T>& Mat23<T>::operator/=(U s) noexcept {
        m_row[0] /= s;
        m_row[1] /= s;
        return *this;
    }

    // -- Unary operators --

    template<typename T>
    constexpr Mat23<T> operator+(const Mat23<T>& m) noexcept {
        return m;
    }

    template<typename T>
    constexpr Mat23<T> operator-(const Mat23<T>& m) noexcept {
        return Mat23<T>(-m[0], -m[1]);
    }

    // -- Binary arithmetic operators --

    template<typename T>
    constexpr Mat23<T> operator+(const Mat23<T>& m1, const Mat23<T>& m2) noexcept {
        return Mat23<T>(m1[0] + m2[0], m1[1] + m2[1]);
    }
    template<typename T>
    constexpr Mat23<T> operator+(T s, const Mat23<T>& m) noexcept {
        return Mat23<T>(s + m[0], s + m[1]);
    }
    template<typename T>
    constexpr Mat23<T> operator+(const Mat23<T>& m, T s) noexcept {
        return Mat23<T>(m[0] + s, m[1] + s);
    }

    template<typename T>
    constexpr Mat23<T> operator-(const Mat23<T>& m1, const Mat23<T>& m2) noexcept {
        return Mat23<T>(m1[0] - m2[0], m1[1] - m2[1]);
    }
    template<typename T>
    constexpr Mat23<T> operator-(T s, const Mat23<T>& m) noexcept {
        return Mat23<T>(s - m[0], s - m[1]);
    }
    template<typename T>
    constexpr Mat23<T> operator-(const Mat23<T>& m, T s) noexcept {
        return Mat23<T>(m[0] - s, m[1] - s);
    }

    template<typename T>
    constexpr Mat23<T> operator*(T s, const Mat23<T>& m) noexcept {
        return Mat23<T>(m[0] * s,
                        m[1] * s);
    }
    template<typename T>
    constexpr Mat23<T> operator*(const Mat23<T>& m, T s) noexcept {
        return Mat23<T>(m[0] * s,
                        m[1] * s);
    }
    template<typename T>
    constexpr Float2<T> operator*(const Mat23<T>& m, const Float3<T>& column) noexcept {
        return Float2<T>(math::dot(m[0], column),
                         math::dot(m[1], column));
    }
    template<typename T>
    constexpr Float3<T> operator*(const Float2<T>& row, const Mat23<T>& m) noexcept {
        return Float3<T>(math::dot(Float2<T>(m[0][0], m[1][0]), row),
                         math::dot(Float2<T>(m[0][1], m[1][1]), row),
                         math::dot(Float2<T>(m[0][2], m[1][2]), row));
    }

    template<typename T>
    constexpr Mat23<T> operator/(T s, const Mat23<T>& m) noexcept {
        return Mat23<T>(s / m[0],
                        s / m[1]);
    }
    template<typename T>
    constexpr Mat23<T> operator/(const Mat23<T>& m, T s) noexcept {
        return Mat23<T>(m[0] / s,
                        m[1] / s);
    }

    template<typename T>
    constexpr bool operator==(const Mat23<T>& m1, const Mat23<T>& m2) noexcept {
        return all(m1[0] == m2[0]) && all(m1[1] == m2[1]);
    }
    template<typename T>
    constexpr bool operator!=(const Mat23<T>& m1, const Mat23<T>& m2) noexcept {
        return all(m1[0] != m2[0]) && all(m1[1] != m2[1]);
    }

    namespace math {
        template<typename T>
        constexpr Mat23<T> elementMultiply(const Mat23<T>& m1, const Mat23<T>& m2) noexcept {
            Mat23<T> out;
            for (size_t i = 0; i < Mat23<T>::ROWS; ++i)
                out[i] = m1[i] * m2[i];
            return out;
        }

        template<uint ULP, typename T>
        constexpr bool isEqual(const Mat23<T>& m1, const Mat23<T>& m2, T e) {
            return all(isEqual<ULP>(m1[0], m2[0], e)) && all(isEqual<ULP>(m1[1], m2[1], e));
        }
    }
}
