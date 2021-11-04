#ifndef NOA_INCLUDE_MAT44_
#error "This should not be directly included"
#endif

#include "noa/common/Assert.h"

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr Float4<T>& Mat44<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->ROWS);
        return m_row[i];
    }

    template<typename T>
    constexpr const Float4<T>& Mat44<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->ROWS);
        return m_row[i];
    }

    // -- Conversion constructors --

    template<typename T>
    constexpr Mat44<T>::Mat44() noexcept
            : m_row{Float4<T>(1, 0, 0, 0),
                    Float4<T>(0, 1, 0, 0),
                    Float4<T>(0, 0, 1, 0),
                    Float4<T>(0, 0, 0, 1)} {}

    template<typename T>
    template<typename U>
    constexpr Mat44<T>::Mat44(U s) noexcept
            : m_row{Float4<T>(s, 0, 0, 0),
                    Float4<T>(0, s, 0, 0),
                    Float4<T>(0, 0, s, 0),
                    Float4<T>(0, 0, 0, s)} {}

    template<typename T>
    template<typename U>
    constexpr Mat44<T>::Mat44(const Float4<U>& v) noexcept
            : m_row{Float4<T>(v.x, 0, 0, 0),
                    Float4<T>(0, v.y, 0, 0),
                    Float4<T>(0, 0, v.z, 0),
                    Float4<T>(0, 0, 0, v.w)} {}

    template<typename T>
    template<typename U>
    constexpr Mat44<T>::Mat44(const Float3<U>& v) noexcept
            : m_row{Float4<T>(v.x, 0, 0, 0),
                    Float4<T>(0, v.y, 0, 0),
                    Float4<T>(0, 0, v.z, 0),
                    Float4<T>(0, 0, 0, 1)} {}

    template<typename T>
    template<typename U>
    constexpr Mat44<T>::Mat44(const Mat44<U>& m) noexcept
            : m_row{Float4<T>(m[0]),
                    Float4<T>(m[1]),
                    Float4<T>(m[2]),
                    Float4<T>(m[3])} {}

    template<typename T>
    template<typename U>
    constexpr Mat44<T>::Mat44(const Mat34<U>& m) noexcept
            : m_row{Float4<T>(m[0]),
                    Float4<T>(m[1]),
                    Float4<T>(m[2]),
                    Float4<T>(0, 0, 0, 1)} {}

    template<typename T>
    template<typename U>
    constexpr Mat44<T>::Mat44(const Mat33<U>& m) noexcept
            : m_row{Float4<T>(m[0][0], m[0][1], m[0][2], 0),
                    Float4<T>(m[1][0], m[1][1], m[1][2], 0),
                    Float4<T>(m[2][0], m[2][1], m[2][2], 0),
                    Float4<T>(0, 0, 0, 1)} {}

    template<typename T>
    template<typename U, typename V>
    constexpr Mat44<T>::Mat44(const Mat33<U>& m, const Float3<V>& v) noexcept
            : m_row{Float4<T>(m[0][0], m[0][1], m[0][2], v[0]),
                    Float4<T>(m[1][0], m[1][1], m[1][2], v[1]),
                    Float4<T>(m[2][0], m[2][1], m[2][2], v[2]),
                    Float4<T>(0, 0, 0, 1)} {}

    template<typename T>
    template<typename X00, typename X01, typename X02, typename X03,
             typename Y10, typename Y11, typename Y12, typename Y13,
             typename Z20, typename Z21, typename Z22, typename Z23,
             typename W30, typename W31, typename W32, typename W33>
    constexpr Mat44<T>::Mat44(X00 x00, X01 x01, X02 x02, X03 x03,
                              Y10 y10, Y11 y11, Y12 y12, Y13 y13,
                              Z20 z20, Z21 z21, Z22 z22, Z23 z23,
                              W30 w30, W31 w31, W32 w32, W33 w33) noexcept
            : m_row{Float4<T>(x00, x01, x02, x03),
                    Float4<T>(y10, y11, y12, y13),
                    Float4<T>(z20, z21, z22, z23),
                    Float4<T>(w30, w31, w32, w33)} {}

    template<typename T>
    template<typename V0, typename V1, typename V2, typename V3>
    constexpr Mat44<T>::Mat44(const Float4<V0>& r0, const Float4<V1>& r1,
                              const Float4<V2>& r2, const Float4<V3>& r3) noexcept
            : m_row{Float4<T>(r0),
                    Float4<T>(r1),
                    Float4<T>(r2),
                    Float4<T>(r3)} {}

    template<typename T>
    template<typename V0, typename V1, typename V2, typename V3>
    constexpr Mat44<T>::Mat44(const Int4<V0>& r0, const Int4<V1>& r1,
                              const Int4<V2>& r2, const Int4<V3>& r3) noexcept
            : m_row{Float4<T>(r0),
                    Float4<T>(r1),
                    Float4<T>(r2),
                    Float4<T>(r3)} {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Mat44<T>& Mat44<T>::operator=(const Mat44<U>& m) noexcept {
        m_row[0] = m[0];
        m_row[1] = m[1];
        m_row[2] = m[2];
        m_row[3] = m[3];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat44<T>& Mat44<T>::operator+=(const Mat44<U>& m) noexcept {
        m_row[0] += m[0];
        m_row[1] += m[1];
        m_row[2] += m[2];
        m_row[3] += m[3];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat44<T>& Mat44<T>::operator-=(const Mat44<U>& m) noexcept {
        m_row[0] -= m[0];
        m_row[1] -= m[1];
        m_row[2] -= m[2];
        m_row[3] -= m[3];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat44<T>& Mat44<T>::operator*=(const Mat44<U>& m) noexcept {
        const Float4<T> A0 = m_row[0];
        const Float4<T> A1 = m_row[1];
        const Float4<T> A2 = m_row[2];
        const Float4<T> A3 = m_row[3];

        const Float4<T> B0 = m[0];
        const Float4<T> B1 = m[1];
        const Float4<T> B2 = m[2];
        const Float4<T> B3 = m[3];

        m_row[0] = A0[0] * B0 + A0[1] * B1 + A0[2] * B2 + A0[3] * B3;
        m_row[1] = A1[0] * B0 + A1[1] * B1 + A1[2] * B2 + A1[3] * B3;
        m_row[2] = A2[0] * B0 + A2[1] * B1 + A2[2] * B2 + A2[3] * B3;
        m_row[3] = A3[0] * B0 + A3[1] * B1 + A3[2] * B2 + A3[3] * B3;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat44<T>& Mat44<T>::operator/=(const Mat44<U>& m) noexcept {
        *this *= math::inverse(m);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat44<T>& Mat44<T>::operator+=(U s) noexcept {
        m_row[0] += s;
        m_row[1] += s;
        m_row[2] += s;
        m_row[3] += s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat44<T>& Mat44<T>::operator-=(U s) noexcept {
        m_row[0] -= s;
        m_row[1] -= s;
        m_row[2] -= s;
        m_row[3] -= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat44<T>& Mat44<T>::operator*=(U s) noexcept {
        m_row[0] *= s;
        m_row[1] *= s;
        m_row[2] *= s;
        m_row[3] *= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat44<T>& Mat44<T>::operator/=(U s) noexcept {
        m_row[0] /= s;
        m_row[1] /= s;
        m_row[2] /= s;
        m_row[3] /= s;
        return *this;
    }

    // -- Unary operators --

    template<typename T>
    constexpr Mat44<T> operator+(const Mat44<T>& m) noexcept {
        return m;
    }

    template<typename T>
    constexpr Mat44<T> operator-(const Mat44<T>& m) noexcept {
        return Mat44<T>(-m[0], -m[1], -m[2], -m[3]);
    }

    // -- Binary arithmetic operators --

    template<typename T>
    constexpr Mat44<T> operator+(const Mat44<T>& m1, const Mat44<T>& m2) noexcept {
        return Mat44<T>(m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2], m1[3] + m2[3]);
    }
    template<typename T>
    constexpr Mat44<T> operator+(T s, const Mat44<T>& m) noexcept {
        return Mat44<T>(s + m[0], s + m[1], s + m[2], s + m[3]);
    }
    template<typename T>
    constexpr Mat44<T> operator+(const Mat44<T>& m, T s) noexcept {
        return Mat44<T>(m[0] + s, m[1] + s, m[2] + s, m[3] + s);
    }

    template<typename T>
    constexpr Mat44<T> operator-(const Mat44<T>& m1, const Mat44<T>& m2) noexcept {
        return Mat44<T>(m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2], m1[3] - m2[3]);
    }
    template<typename T>
    constexpr Mat44<T> operator-(T s, const Mat44<T>& m) noexcept {
        return Mat44<T>(s - m[0], s - m[1], s - m[2], s - m[3]);
    }
    template<typename T>
    constexpr Mat44<T> operator-(const Mat44<T>& m, T s) noexcept {
        return Mat44<T>(m[0] - s, m[1] - s, m[2] - s, m[3] - s);
    }

    template<typename T>
    constexpr Mat44<T> operator*(const Mat44<T>& m1, const Mat44<T>& m2) noexcept {
        Mat44<T> out(m1);
        out *= m2;
        return out;
    }
    template<typename T>
    constexpr Mat44<T> operator*(T s, const Mat44<T>& m) noexcept {
        return Mat44<T>(m[0] * s,
                        m[1] * s,
                        m[2] * s,
                        m[3] * s);
    }
    template<typename T>
    constexpr Mat44<T> operator*(const Mat44<T>& m, T s) noexcept {
        return Mat44<T>(m[0] * s,
                        m[1] * s,
                        m[2] * s,
                        m[3] * s);
    }
    template<typename T>
    constexpr Float4<T> operator*(const Mat44<T>& m, const Float4<T>& column) noexcept {
        return Float4<T>(math::dot(m[0], column),
                         math::dot(m[1], column),
                         math::dot(m[2], column),
                         math::dot(m[3], column));
    }
    template<typename T>
    constexpr Float4<T> operator*(const Float4<T>& row, const Mat44<T>& m) noexcept {
        return Float4<T>(math::dot(Float4<T>(m[0][0], m[1][0], m[2][0], m[3][0]), row),
                         math::dot(Float4<T>(m[0][1], m[1][1], m[2][1], m[3][1]), row),
                         math::dot(Float4<T>(m[0][2], m[1][2], m[2][2], m[3][2]), row),
                         math::dot(Float4<T>(m[0][3], m[1][3], m[2][3], m[3][3]), row));
    }

    template<typename T>
    constexpr Mat44<T> operator/(const Mat44<T>& m1, const Mat44<T>& m2) noexcept {
        Mat44<T> out(m1);
        out /= m2;
        return out;
    }
    template<typename T>
    constexpr Mat44<T> operator/(T s, const Mat44<T>& m) noexcept {
        return Mat44<T>(s / m[0],
                        s / m[1],
                        s / m[2],
                        s / m[3]);
    }
    template<typename T>
    constexpr Mat44<T> operator/(const Mat44<T>& m, T s) noexcept {
        return Mat44<T>(m[0] / s,
                        m[1] / s,
                        m[2] / s,
                        m[3] / s);
    }
    template<typename T>
    constexpr Float4<T> operator/(const Mat44<T>& m, const Float4<T>& column) noexcept {
        return math::inverse(m) * column;
    }
    template<typename T>
    constexpr Float4<T> operator/(const Float4<T>& row, const Mat44<T>& m) noexcept {
        return row * math::inverse(m);
    }

    template<typename T>
    constexpr bool operator==(const Mat44<T>& m1, const Mat44<T>& m2) noexcept {
        return all(m1[0] == m2[0]) && all(m1[1] == m2[1]) && all(m1[2] == m2[2]) && all(m1[3] == m2[3]);
    }
    template<typename T>
    constexpr bool operator!=(const Mat44<T>& m1, const Mat44<T>& m2) noexcept {
        return all(m1[0] != m2[0]) && all(m1[1] != m2[1]) && all(m1[2] != m2[2]) && all(m1[3] != m2[3]);
    }

    namespace math {
        template<typename T>
        constexpr Mat44<T> elementMultiply(const Mat44<T>& m1, const Mat44<T>& m2) noexcept {
            Mat44<T> out;
            for (size_t i = 0; i < Mat44<T>::ROWS; ++i)
                out[i] = m1[i] * m2[i];
            return out;
        }

        template<typename T>
        constexpr Mat44<T> outerProduct(const Float4<T>& column, const Float4<T>& row) noexcept {
            return Mat44<T>(column.x * row.x, column.x * row.y, column.x * row.z, column.x * row.w,
                            column.y * row.x, column.y * row.y, column.y * row.z, column.y * row.w,
                            column.z * row.x, column.z * row.y, column.z * row.z, column.z * row.w,
                            column.w * row.x, column.w * row.y, column.w * row.z, column.w * row.w);
        }

        template<typename T>
        constexpr Mat44<T> transpose(const Mat44<T>& m) noexcept {
            return Mat44<T>(m[0][0], m[1][0], m[2][0], m[3][0],
                            m[0][1], m[1][1], m[2][1], m[3][1],
                            m[0][2], m[1][2], m[2][2], m[3][2],
                            m[0][3], m[1][3], m[2][3], m[3][3]);
        }

        template<typename T>
        constexpr T determinant(const Mat44<T>& m) noexcept {
            T s00 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
            T s01 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
            T s02 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
            T s03 = m[0][2] * m[3][3] - m[0][3] * m[3][2];
            T s04 = m[0][2] * m[2][3] - m[0][3] * m[2][2];
            T s05 = m[0][2] * m[1][3] - m[0][3] * m[1][2];

            Float4<T> c(+(m[1][1] * s00 - m[2][1] * s01 + m[3][1] * s02),
                        -(m[0][1] * s00 - m[2][1] * s03 + m[3][1] * s04),
                        +(m[0][1] * s01 - m[1][1] * s03 + m[3][1] * s05),
                        -(m[0][1] * s02 - m[1][1] * s04 + m[2][1] * s05));

            return m[0][0] * c[0] + m[1][0] * c[1] +
                   m[2][0] * c[2] + m[3][0] * c[3];
        }

        template<typename T>
        constexpr Mat44<T> inverse(const Mat44<T>& m) noexcept {
            // From https://stackoverflow.com/a/44446912 and https://github.com/willnode/N-Matrix-Programmer
            T A2323 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
            T A1323 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
            T A1223 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
            T A0323 = m[2][0] * m[3][3] - m[2][3] * m[3][0];
            T A0223 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
            T A0123 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
            T A2313 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
            T A1313 = m[1][1] * m[3][3] - m[1][3] * m[3][1];
            T A1213 = m[1][1] * m[3][2] - m[1][2] * m[3][1];
            T A2312 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
            T A1312 = m[1][1] * m[2][3] - m[1][3] * m[2][1];
            T A1212 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
            T A0313 = m[1][0] * m[3][3] - m[1][3] * m[3][0];
            T A0213 = m[1][0] * m[3][2] - m[1][2] * m[3][0];
            T A0312 = m[1][0] * m[2][3] - m[1][3] * m[2][0];
            T A0212 = m[1][0] * m[2][2] - m[1][2] * m[2][0];
            T A0113 = m[1][0] * m[3][1] - m[1][1] * m[3][0];
            T A0112 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

            T det = m[0][0] * (m[1][1] * A2323 - m[1][2] * A1323 + m[1][3] * A1223) -
                    m[0][1] * (m[1][0] * A2323 - m[1][2] * A0323 + m[1][3] * A0223) +
                    m[0][2] * (m[1][0] * A1323 - m[1][1] * A0323 + m[1][3] * A0123) -
                    m[0][3] * (m[1][0] * A1223 - m[1][1] * A0223 + m[1][2] * A0123);
            NOA_ASSERT(!isEqual(det, static_cast<T>(0))); // non singular
            det = static_cast<T>(1) / det;

            return Mat44<T>(det * +(m[1][1] * A2323 - m[1][2] * A1323 + m[1][3] * A1223),
                            det * -(m[0][1] * A2323 - m[0][2] * A1323 + m[0][3] * A1223),
                            det * +(m[0][1] * A2313 - m[0][2] * A1313 + m[0][3] * A1213),
                            det * -(m[0][1] * A2312 - m[0][2] * A1312 + m[0][3] * A1212),
                            det * -(m[1][0] * A2323 - m[1][2] * A0323 + m[1][3] * A0223),
                            det * +(m[0][0] * A2323 - m[0][2] * A0323 + m[0][3] * A0223),
                            det * -(m[0][0] * A2313 - m[0][2] * A0313 + m[0][3] * A0213),
                            det * +(m[0][0] * A2312 - m[0][2] * A0312 + m[0][3] * A0212),
                            det * +(m[1][0] * A1323 - m[1][1] * A0323 + m[1][3] * A0123),
                            det * -(m[0][0] * A1323 - m[0][1] * A0323 + m[0][3] * A0123),
                            det * +(m[0][0] * A1313 - m[0][1] * A0313 + m[0][3] * A0113),
                            det * -(m[0][0] * A1312 - m[0][1] * A0312 + m[0][3] * A0112),
                            det * -(m[1][0] * A1223 - m[1][1] * A0223 + m[1][2] * A0123),
                            det * +(m[0][0] * A1223 - m[0][1] * A0223 + m[0][2] * A0123),
                            det * -(m[0][0] * A1213 - m[0][1] * A0213 + m[0][2] * A0113),
                            det * +(m[0][0] * A1212 - m[0][1] * A0212 + m[0][2] * A0112)
            );
        }

        template<uint ULP, typename T>
        constexpr bool isEqual(const Mat44<T>& m1, const Mat44<T>& m2, T e) {
            return all(isEqual<ULP>(m1[0], m2[0], e)) &&
                   all(isEqual<ULP>(m1[1], m2[1], e)) &&
                   all(isEqual<ULP>(m1[2], m2[2], e));
        }
    }
}