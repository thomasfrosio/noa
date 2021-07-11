/// \file noa/common/types/Mat34.h
/// \author Thomas - ffyr2w
/// \date 2 Jun 2021
/// A 3x4 floating-point matrix.

#pragma once

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Float4.h"

namespace noa {
    template<typename T> class Mat44;
    template<typename T> class Mat33;
    template<typename T> class Float4;
    template<typename T> class Float3;
    template<typename T> class Int4;

    /// A 3x4 floating-point matrix.
    /// \note The indexing is "row-first" (as opposed to "column-first", like in OpenGL Math),
    ///       i.e. M[r][c] with r = row index and c = column index. All indexes starts from 0.
    /// \note These matrices are quite limited compared to the squared ones and they're mostly there
    ///       to pre-multiple column vectors for 3D affine transforms.
    template<typename T>
    class Mat34 {
    private:
        static constexpr uint ROWS = 3U;
        static constexpr uint COLS = 4U;
        Float4<T> m_row[ROWS];

    public: // Type definitions
        typedef T value_type;

    public: // Component accesses
        NOA_HD static constexpr size_t length() noexcept { return 3; }
        NOA_HD static constexpr size_t elements() noexcept { return 12; }
        NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr Float4<T>& operator[](size_t i);
        NOA_HD constexpr const Float4<T>& operator[](size_t i) const;

    public: // (Conversion) Constructors
        NOA_HD constexpr Mat34() noexcept; // identity matrix
        template<typename U> NOA_HD constexpr explicit Mat34(U s) noexcept; // equivalent to Mat34() * s
        template<typename U> NOA_HD constexpr explicit Mat34(const Float4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat34(const Float3<U>& v) noexcept;

        template<typename U> NOA_HD constexpr explicit Mat34(const Mat44<U>& m) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat34(const Mat34<U>& m) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat34(const Mat33<U>& m) noexcept;
        template<typename U, typename V>
        NOA_HD constexpr explicit Mat34(const Mat33<U>& m, const Float3<V>& v) noexcept;

        template<typename X00, typename X01, typename X02, typename X03,
                 typename Y10, typename Y11, typename Y12, typename Y13,
                 typename Z20, typename Z21, typename Z22, typename Z23>
        NOA_HD constexpr Mat34(X00 x00, X01 x01, X02 x02, X03 x03,
                               Y10 y10, Y11 y11, Y12 y12, Y13 y13,
                               Z20 z20, Z21 z21, Z22 z22, Z23 z23) noexcept;

        template<typename V0, typename V1, typename V2>
        NOA_HD constexpr Mat34(const Float4<V0>& r0,
                               const Float4<V1>& r1,
                               const Float4<V2>& r2) noexcept;

        template<typename V0, typename V1, typename V2>
        NOA_HD constexpr Mat34(const Int4<V0>& r0,
                               const Int4<V1>& r1,
                               const Int4<V2>& r2) noexcept;

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Mat34<T>& operator=(const Mat34<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat34<T>& operator+=(const Mat34<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat34<T>& operator-=(const Mat34<U>& m) noexcept;

        template<typename U> NOA_HD constexpr Mat34<T>& operator+=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat34<T>& operator-=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat34<T>& operator*=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat34<T>& operator/=(U s) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_IHD constexpr Mat34<T> operator+(const Mat34<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat34<T> operator-(const Mat34<T>& m) noexcept;

    // -- Binary operators --

    template<typename T> NOA_IHD constexpr Mat34<T> operator+(const Mat34<T>& m1, const Mat34<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat34<T> operator+(T s, const Mat34<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat34<T> operator+(const Mat34<T>& m, T s) noexcept;

    template<typename T> NOA_IHD constexpr Mat34<T> operator-(const Mat34<T>& m1, const Mat34<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat34<T> operator-(T s, const Mat34<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat34<T> operator-(const Mat34<T>& m, T s) noexcept;

    template<typename T> NOA_IHD constexpr Mat34<T> operator*(T s, const Mat34<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat34<T> operator*(const Mat34<T>& m, T s) noexcept;
    template<typename T> NOA_IHD constexpr Float3<T> operator*(const Mat34<T>& m, const Float4<T>& column) noexcept;
    template<typename T> NOA_IHD constexpr Float4<T> operator*(const Float3<T>& row, const Mat34<T>& m) noexcept;

    template<typename T> NOA_IHD constexpr Mat34<T> operator/(T s, const Mat34<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat34<T> operator/(const Mat34<T>& m, T s) noexcept;


    // -- Boolean operators --

    template<typename T> NOA_IHD constexpr bool operator==(const Mat34<T>& m1, const Mat34<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr bool operator!=(const Mat34<T>& m1, const Mat34<T>& m2) noexcept;

    namespace math {
        /// Multiplies matrix \a lhs by matrix \a rhs element-wise, i.e. `out[i][j] = lhs[i][j] * rhs[i][j]`.
        template<typename T>
        NOA_IHD constexpr Mat34<T> elementMultiply(const Mat34<T>& m1, const Mat34<T>& m2) noexcept;

        template<uint ULP = 2, typename T>
        NOA_IHD constexpr bool isEqual(const Mat34<T>& m1, const Mat34<T>& m2, T e = 1e-6f);
    }

    using float34_t = Mat34<float>;
    using double34_t = Mat34<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 12> toArray(const Mat34<T>& v) noexcept {
        return {v[0][0], v[0][1], v[0][2], v[0][3],
                v[1][0], v[1][1], v[1][2], v[1][3],
                v[2][0], v[2][1], v[2][2], v[2][3]};
    }

    template<> NOA_IH std::string string::typeName<float34_t>() { return "float34"; }
    template<> NOA_IH std::string string::typeName<double34_t>() { return "double34"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Mat34<T>& m) {
        os << string::format("({},{},{})", m[0], m[1], m[2]);
        return os;
    }
}

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr Float4<T>& Mat34<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->length());
        return m_row[i];
    }

    template<typename T>
    constexpr const Float4<T>& Mat34<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->length());
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
            for (size_t i = 0; i < Mat34<T>::length(); ++i)
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
