#pragma once

#include "noa/Definitions.h"
#include "noa/util/Float3.h"
#include "noa/util/Int3.h"
#include "noa/util/traits/BaseTypes.h"

namespace Noa {
    template<typename>
    struct Int3;

    template<typename>
    struct Float3;

    /// A 3x3 floating-point matrix.
    /// @note The indexing is "row-first" (as opposed to "column-first", like in OpenGL Math), i.e. M[r][c] with
    ///       r = row index and c = column index. All indexes starts from 0.
    /// @note By convention, the M * v operation assumes v is a column vector (like in OpenGL Math).
    /// @note Transformations are active (alibi) and assumes a right handed coordinate system.
    ///       All angles are given in radians, positive is counter-clockwise looking at the origin.
    /// @note Even if the matrix contains some rotation-matrix specific functions (e.g. the rot*() functions),
    ///       it does not assume that the matrix is a "proper" rotation matrix (i.e. the determinant is not
    ///       necessarily equal to 1).
    template<typename T>
    class Mat3 {
    private:
        static constexpr uint ROWS = 3U;
        static constexpr uint COLS = 3U;
        Float3<T> m_row[ROWS];

    public: // Type definitions
        typedef T value_type;

    public: // Component accesses
        NOA_HD static constexpr size_t length() noexcept { return 3; }
        NOA_HD static constexpr size_t elements() noexcept { return 9; }
        NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr Float3<T>& operator[](size_t i) { return m_row[i]; }
        NOA_HD constexpr const Float3<T>& operator[](size_t i) const { return m_row[i]; }

    public: // (Conversion) Constructors
        NOA_HD constexpr Mat3() noexcept; // identity matrix
        template<typename U> NOA_HD constexpr explicit Mat3(U s) noexcept; // equivalent to Mat3() * s
        template<typename U> NOA_HD constexpr explicit Mat3(const Float3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat3(const Mat3<U>& m) noexcept;

        template<typename X00, typename X01, typename X02,
                 typename Y10, typename Y11, typename Y12,
                 typename Z20, typename Z21, typename Z22>
        NOA_HD constexpr Mat3(X00 x00, X01 x01, X02 x02,
                              Y10 y10, Y11 y11, Y12 y12,
                              Z20 z20, Z21 z21, Z22 z22) noexcept;

        template<typename V0, typename V1, typename V2>
        NOA_HD constexpr Mat3(const Float3<V0>& r0,
                              const Float3<V1>& r1,
                              const Float3<V2>& r2) noexcept;

        template<typename V0, typename V1, typename V2>
        NOA_HD constexpr Mat3(const Int3<V0>& r0,
                              const Int3<V1>& r1,
                              const Int3<V2>& r2) noexcept;

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Mat3<T>& operator=(const Mat3<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat3<T>& operator+=(const Mat3<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat3<T>& operator-=(const Mat3<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat3<T>& operator*=(const Mat3<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat3<T>& operator/=(const Mat3<U>& m) noexcept;

        template<typename U> NOA_HD constexpr Mat3<T>& operator+=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat3<T>& operator-=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat3<T>& operator*=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat3<T>& operator/=(U s) noexcept;

    public: // Public functions
        template<typename U> NOA_HD constexpr Mat3<T>& scale(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat3<T>& scale(const Float3<U>& s) noexcept;
        template<typename U> NOA_HD constexpr Mat3<T>& rotX(U angle) noexcept;
        template<typename U> NOA_HD constexpr Mat3<T>& rotY(U angle) noexcept;
        template<typename U> NOA_HD constexpr Mat3<T>& rotZ(U angle) noexcept;
        template<typename U> NOA_HD constexpr Mat3<T>& rot(const Float3<T>& axis, U angle) noexcept;
        template<typename U, typename V> NOA_HD constexpr Mat3<T>& rotInPlane(U axis_angle, V angle) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_HD constexpr Mat3<T> operator+(const Mat3<T>& m) noexcept;
    template<typename T> NOA_HD constexpr Mat3<T> operator-(const Mat3<T>& m) noexcept;

    // -- Binary operators --

    template<typename T> NOA_HD constexpr Mat3<T> operator+(const Mat3<T>& m1, const Mat3<T>& m2) noexcept;
    template<typename T> NOA_HD constexpr Mat3<T> operator+(T s, const Mat3<T>& m) noexcept;
    template<typename T> NOA_HD constexpr Mat3<T> operator+(const Mat3<T>& m, T s) noexcept;

    template<typename T> NOA_HD constexpr Mat3<T> operator-(const Mat3<T>& m1, const Mat3<T>& m2) noexcept;
    template<typename T> NOA_HD constexpr Mat3<T> operator-(T s, const Mat3<T>& m) noexcept;
    template<typename T> NOA_HD constexpr Mat3<T> operator-(const Mat3<T>& m, T s) noexcept;

    template<typename T> NOA_HD constexpr Mat3<T> operator*(const Mat3<T>& m1, const Mat3<T>& m2) noexcept;
    template<typename T> NOA_HD constexpr Mat3<T> operator*(T s, const Mat3<T>& m) noexcept;
    template<typename T> NOA_HD constexpr Mat3<T> operator*(const Mat3<T>& m, T s) noexcept;
    template<typename T> NOA_HD constexpr Float3<T> operator*(const Mat3<T>& m, const Float3<T>& column) noexcept;
    template<typename T> NOA_HD constexpr Mat3<T> operator*(const Float3<T>& row, const Mat3<T>& m) noexcept;

    template<typename T> NOA_HD constexpr Mat3<T> operator/(const Mat3<T>& m1, const Mat3<T>& m2) noexcept;
    template<typename T> NOA_HD constexpr Mat3<T> operator/(T s, const Mat3<T>& m) noexcept;
    template<typename T> NOA_HD constexpr Mat3<T> operator/(const Mat3<T>& m, T s) noexcept;
    template<typename T> NOA_HD constexpr Float3<T> operator/(const Mat3<T>& m, const Float3<T>& column) noexcept;
    template<typename T> NOA_HD constexpr Mat3<T> operator/(const Float3<T>& row, const Mat3<T>& m) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_HD constexpr bool operator==(const Mat3<T>& m1, const Mat3<T>& m2) noexcept;
    template<typename T> NOA_HD constexpr bool operator!=(const Mat3<T>& m1, const Mat3<T>& m2) noexcept;

    namespace Math {
        /// Multiplies matrix @a lhs by matrix @a rhs element-wise, i.e. `out[i][j] = lhs[i][j] * rhs[i][j]`.
        template<typename T> NOA_HD constexpr Mat3<T> elementMultiply(const Mat3<T>& m1, const Mat3<T>& m2) noexcept;

        /// Given the column vector @a column and row vector @a row,
        /// computes the linear algebraic matrix multiply `c * r`.
        template<typename T> NOA_HD constexpr Mat3<T> outerProduct(const Float3<T>& column,
                                                                   const Float3<T>& row) noexcept;

        /// Returns the transposed matrix of @a m.
        template<typename T> NOA_HD constexpr Mat3<T> transpose(const Mat3<T>& m) noexcept;

        /// Returns the determinant of @a m.
        template<typename T> NOA_HD constexpr T determinant(const Mat3<T>& m) noexcept;

        /// Returns the inverse matrix of @a m.
        template<typename T> NOA_HD constexpr Mat3<T> inverse(const Mat3<T>& m) noexcept;

        template<uint ULP = 2, typename T>
        NOA_HD constexpr bool isEqual(const Mat3<T>& m1, const Mat3<T>& m2, T e = 1e-6f);

        /// Scales @a m with the scalar @a s.
        template<typename T, typename U> NOA_HD constexpr Mat3<T> scale(const Mat3<T>& m, U s) noexcept;

        /// Scales @a m with the vector @a v. Equivalent to `m * Mat3<T>(s)`.
        template<typename T, typename U> NOA_HD constexpr Mat3<T> scale(const Mat3<T>& m, const Float3<U>& s) noexcept;

        template<typename T, typename U> NOA_HD constexpr Mat3<T> rotX(const Mat3<T>& m, U angle) noexcept;
        template<typename T, typename U> NOA_HD constexpr Mat3<T> rotY(const Mat3<T>& m, U angle) noexcept;
        template<typename T, typename U> NOA_HD constexpr Mat3<T> rotZ(const Mat3<T>& m, U angle) noexcept;

        /// Rotates @a m by an @a angle (in radians) around a given @a axis.
        template<typename T, typename U>
        NOA_HD constexpr Mat3<T> rot(const Mat3<T>& m, Float3<T> axis, U angle) noexcept;

        /// Rotates @a m by @a angle radians around an in-place axis (on the XY plane) at @a axis_angle radians.
        template<typename T, typename U, typename V>
        NOA_HD constexpr Mat3<T> rotInPlane(const Mat3<T>& m, U axis_angle, V angle) noexcept;
    }

    using float33_t = Mat3<float>;
    using double33_t = Mat3<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 9> toArray(const Mat3<T>& v) noexcept {
        return {v.x, v.y, v.z};
    }

    template<> NOA_IH std::string String::typeName<float33_t>() { return "float33"; }
    template<> NOA_IH std::string String::typeName<double33_t>() { return "double33"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Mat3<T>& m) {
        os << String::format("({}, {}, {})", m[0], m[1], m[2]);
        return os;
    }
}

namespace Noa {
    // -- Conversion constructors --

    template<typename T>
    NOA_HD constexpr Mat3<T>::Mat3() noexcept
            : m_row{Float3<T>(1, 0, 0),
                    Float3<T>(0, 1, 0),
                    Float3<T>(0, 0, 1)} {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Mat3<T>::Mat3(U s) noexcept
            : m_row{Float3<T>(s, 0, 0),
                    Float3<T>(0, s, 0),
                    Float3<T>(0, 0, s)} {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Mat3<T>::Mat3(const Float3<U>& v) noexcept
            : m_row{Float3<T>(v.x, 0, 0),
                    Float3<T>(0, v.y, 0),
                    Float3<T>(0, 0, v.z)} {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Mat3<T>::Mat3(const Mat3<U>& m) noexcept
            : m_row{Float3<T>(m[0]),
                    Float3<T>(m[1]),
                    Float3<T>(m[2])} {}

    template<typename T>
    template<typename X00, typename X01, typename X02,
             typename Y10, typename Y11, typename Y12,
             typename Z20, typename Z21, typename Z22>
    NOA_HD constexpr Mat3<T>::Mat3(X00 x00, X01 x01, X02 x02,
                                   Y10 y10, Y11 y11, Y12 y12,
                                   Z20 z20, Z21 z21, Z22 z22) noexcept
            : m_row{Float3<T>(x00, x01, x02),
                    Float3<T>(y10, y11, y12),
                    Float3<T>(z20, z21, z22)} {}

    template<typename T>
    template<typename V0, typename V1, typename V2>
    NOA_HD constexpr Mat3<T>::Mat3(const Float3<V0>& r0, const Float3<V1>& r1, const Float3<V2>& r2) noexcept
            : m_row{Float3<T>(r0),
                    Float3<T>(r1),
                    Float3<T>(r2)} {}

    template<typename T>
    template<typename V0, typename V1, typename V2>
    NOA_HD constexpr Mat3<T>::Mat3(const Int3<V0>& r0, const Int3<V1>& r1, const Int3<V2>& r2) noexcept
            : m_row{Float3<T>(r0),
                    Float3<T>(r1),
                    Float3<T>(r2)} {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Mat3<T>& Mat3<T>::operator=(const Mat3<U>& m) noexcept {
        m_row[0] = m[0];
        m_row[1] = m[1];
        m_row[2] = m[2];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat3<T>& Mat3<T>::operator+=(const Mat3<U>& m) noexcept {
        m_row[0] += m[0];
        m_row[1] += m[1];
        m_row[2] += m[2];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat3<T>& Mat3<T>::operator-=(const Mat3<U>& m) noexcept {
        m_row[0] -= m[0];
        m_row[1] -= m[1];
        m_row[2] -= m[2];
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat3<T>& Mat3<T>::operator*=(const Mat3<U>& m) noexcept {
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
    constexpr Mat3<T>& Mat3<T>::operator/=(const Mat3<U>& m) noexcept {
        *this *= Math::inverse(m);
    }

    template<typename T>
    template<typename U>
    constexpr Mat3<T>& Mat3<T>::operator+=(U s) noexcept {
        m_row[0] += s;
        m_row[1] += s;
        m_row[2] += s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat3<T>& Mat3<T>::operator-=(U s) noexcept {
        m_row[0] -= s;
        m_row[1] -= s;
        m_row[2] -= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat3<T>& Mat3<T>::operator*=(U s) noexcept {
        m_row[0] *= s;
        m_row[1] *= s;
        m_row[2] *= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Mat3<T>& Mat3<T>::operator/=(U s) noexcept {
        m_row[0] /= s;
        m_row[1] /= s;
        m_row[2] /= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Mat3<T>& Mat3<T>::scale(U s) noexcept {
        *this *= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Mat3<T>& Mat3<T>::scale(const Float3<U>& s) noexcept {
        m_row[0] *= s;
        m_row[1] *= s;
        m_row[2] *= s;
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Mat3<T>& Mat3<T>::rotZ(U angle) noexcept {
        T c = Math::cos(static_cast<T>(angle));
        T s = Math::sin(static_cast<T>(angle));
        *this *= Mat3<T>(c, -s, 0, s, c, 0, 0, 0, 1);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Mat3<T>& Mat3<T>::rotY(U angle) noexcept {
        T c = Math::cos(static_cast<T>(angle));
        T s = Math::sin(static_cast<T>(angle));
        *this *= Mat3<T>(c, 0, s, 0, 1, 0, -s, 0, c);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Mat3<T>& Mat3<T>::rotX(U angle) noexcept {
        T c = Math::cos(static_cast<T>(angle));
        T s = Math::sin(static_cast<T>(angle));
        *this *= Mat3<T>(1, 0, 0, 0, c, -s, 0, s, c);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Mat3<T>& Mat3<T>::rot(const Float3<T>& axis, U angle) noexcept {
        T c = Math::cos(static_cast<T>(angle));
        T s = Math::sin(static_cast<T>(angle));
        T c1 = static_cast<T>(1) - c;
        *this *= Mat3<T>(axis.x * axis.x * c1 + c,
                         axis.x * axis.y * c1 - axis.z * s,
                         axis.x * axis.z * c1 + axis.y * s,
                         axis.y * axis.x * c1 + axis.z * s,
                         axis.y * axis.y * c1 + c,
                         axis.y * axis.z * c1 - axis.x * s,
                         axis.z * axis.x * c1 - axis.y * s,
                         axis.z * axis.y * c1 + axis.x * s,
                         axis.z * axis.z * c1 + c);
        return *this;
    }

    template<typename T>
    template<typename U, typename V>
    NOA_HD constexpr Mat3<T>& Mat3<T>::rotInPlane(U axis_angle, V angle) noexcept {
        return this->rot(Float3<T>(Math::cos(axis_angle), Math::sin(axis_angle), 0), angle);
    }

    // -- Unary operators --

    template<typename T>
    NOA_IHD constexpr Mat3<T> operator+(const Mat3<T>& m) noexcept {
        return m;
    }

    template<typename T>
    NOA_IHD constexpr Mat3<T> operator-(const Mat3<T>& m) noexcept {
        return Mat3<T>(-m[0], -m[1], -m[2]);
    }

    // -- Binary arithmetic operators --

    template<typename T>
    NOA_IHD constexpr Mat3<T> operator+(const Mat3<T>& m1, const Mat3<T>& m2) noexcept {
        return Mat3<T>(m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2]);
    }
    template<typename T>
    NOA_IHD constexpr Mat3<T> operator+(T s, const Mat3<T>& m) noexcept {
        return Mat3<T>(s + m[0], s + m[1], s + m[2]);
    }
    template<typename T>
    NOA_IHD constexpr Mat3<T> operator+(const Mat3<T>& m, T s) noexcept {
        return Mat3<T>(m[0] + s, m[1] + s, m[2] + s);
    }

    template<typename T>
    NOA_IHD constexpr Mat3<T> operator-(const Mat3<T>& m1, const Mat3<T>& m2) noexcept {
        return Mat3<T>(m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2]);
    }
    template<typename T>
    NOA_IHD constexpr Mat3<T> operator-(T s, const Mat3<T>& m) noexcept {
        return Mat3<T>(s - m[0], s - m[1], s - m[2]);
    }
    template<typename T>
    NOA_IHD constexpr Mat3<T> operator-(const Mat3<T>& m, T s) noexcept {
        return Mat3<T>(m[0] - s, m[1] - s, m[2] - s);
    }

    template<typename T> NOA_IHD constexpr Mat3<T> operator*(const Mat3<T>& m1, const Mat3<T>& m2) noexcept {
        Mat3<T> out(m1);
        out *= m2;
        return out;
    }
    template<typename T> NOA_IHD constexpr Mat3<T> operator*(T s, const Mat3<T>& m) noexcept {
        return Mat3<T>(m[0] * s,
                       m[1] * s,
                       m[2] * s);
    }
    template<typename T> NOA_IHD constexpr Mat3<T> operator*(const Mat3<T>& m, T s) noexcept {
        return Mat3<T>(m[0] * s,
                       m[1] * s,
                       m[2] * s);
    }
    template<typename T> NOA_IHD constexpr Float3<T> operator*(const Mat3<T>& m, const Float3<T>& column) noexcept {
        return Float3<T>(Math::dot(m[0], column),
                         Math::dot(m[1], column),
                         Math::dot(m[2], column));
    }
    template<typename T> NOA_IHD constexpr Mat3<T> operator*(const Float3<T>& row, const Mat3<T>& m) noexcept {
        return Float3<T>(Math::dot(Float3<T>(m[0][0], m[1][0], m[2][0]), row),
                         Math::dot(Float3<T>(m[0][1], m[1][1], m[2][1]), row),
                         Math::dot(Float3<T>(m[0][2], m[1][2], m[2][2]), row));
    }

    template<typename T> NOA_HD constexpr Mat3<T> operator/(const Mat3<T>& m1, const Mat3<T>& m2) noexcept {
        Mat3<T> out(m1);
        out /= m2;
        return out;
    }
    template<typename T> NOA_HD constexpr Mat3<T> operator/(T s, const Mat3<T>& m) noexcept {
        return Mat3<T>(s / m[0],
                       s / m[1],
                       s / m[2]);
    }
    template<typename T> NOA_HD constexpr Mat3<T> operator/(const Mat3<T>& m, T s) noexcept {
        return Mat3<T>(m[0] / s,
                       m[1] / s,
                       m[2] / s);
    }
    template<typename T> NOA_HD constexpr Float3<T> operator/(const Mat3<T>& m, const Float3<T>& column) noexcept {
        return Math::inverse(m) * column;
    }
    template<typename T> NOA_HD constexpr Mat3<T> operator/(const Float3<T>& row, const Mat3<T>& m) noexcept {
        return row * Math::inverse(m);
    }

    template<typename T> NOA_HD constexpr bool operator==(const Mat3<T>& m1, const Mat3<T>& m2) noexcept {
        return all(m1[0] == m2[0]) && all(m1[1] == m2[1]) && all(m1[2] == m2[2]);
    }
    template<typename T> NOA_HD constexpr bool operator!=(const Mat3<T>& m1, const Mat3<T>& m2) noexcept {
        return all(m1[0] != m2[0]) && all(m1[1] != m2[1]) && all(m1[2] != m2[2]);
    }

    namespace Math {
        template<typename T>
        NOA_IHD constexpr Mat3<T> elementMultiply(const Mat3<T>& m1, const Mat3<T>& m2) noexcept {
            Mat3<T> out;
            for (size_t i = 0; i < Mat3<T>::length(); ++i)
                out[i] = m1[i] * m2[i];
            return out;
        }

        template<typename T>
        NOA_IHD constexpr Mat3<T> outerProduct(const Float3<T>& column, const Float3<T>& row) noexcept {
            return Mat3<T>(column.x * row.x, column.x * row.y, column.x * row.z,
                           column.y * row.x, column.y * row.y, column.y * row.z,
                           column.z * row.x, column.z * row.y, column.z * row.z);
        }

        template<typename T>
        NOA_IHD constexpr Mat3<T> transpose(const Mat3<T>& m) noexcept {
            return Mat3<T>(m[0][0], m[1][0], m[2][0],
                           m[0][1], m[1][1], m[2][1],
                           m[0][2], m[1][2], m[2][2]);
        }

        template<typename T>
        NOA_IHD constexpr T determinant(const Mat3<T>& m) noexcept {
            return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                   m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                   m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        }

        template<typename T>
        NOA_HD constexpr Mat3<T> inverse(const Mat3<T>& m) noexcept {
            T one_over_determinant = static_cast<T>(1) / determinant(m);
            Mat3<T> inverse;
            inverse[0][0] = +(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * one_over_determinant;
            inverse[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * one_over_determinant;
            inverse[0][2] = +(m[0][1] * m[1][2] - m[0][2] * m[1][1]) * one_over_determinant;
            inverse[1][0] = -(m[0][1] * m[2][2] - m[1][2] * m[2][0]) * one_over_determinant;
            inverse[1][1] = +(m[0][0] * m[2][2] - m[0][2] * m[2][0]) * one_over_determinant;
            inverse[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * one_over_determinant;
            inverse[2][0] = +(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * one_over_determinant;
            inverse[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * one_over_determinant;
            inverse[2][2] = +(m[0][0] * m[1][1] - m[0][1] * m[1][0]) * one_over_determinant;
            return inverse;
        }

        template<uint ULP, typename T>
        NOA_FHD constexpr bool isEqual(const Mat3<T>& m1, const Mat3<T>& m2, T e) {
            return all(isEqual<ULP>(m1[0], m2[0], e)) &&
                   all(isEqual<ULP>(m1[1], m2[1], e)) &&
                   all(isEqual<ULP>(m1[2], m2[2], e));
        }

        template<typename T, typename U>
        NOA_FHD constexpr Mat3<T> scale(const Mat3<T>& m, U s) noexcept {
            Mat3<T> scaled(m);
            scaled.scale(s);
            return scaled;
        }

        template<typename T, typename U>
        NOA_FHD constexpr Mat3<T> scale(const Mat3<T>& m, const Float3<U>& s) noexcept {
            Mat3<T> scaled(m);
            scaled.scale(s);
            return scaled;
        }

        template<typename T, typename U>
        NOA_HD constexpr Mat3<T> rotX(const Mat3<T>& m, U angle) noexcept {
            Mat3<T> out(m);
            return out.rotX(angle);
        }

        template<typename T, typename U>
        NOA_HD constexpr Mat3<T> rotY(const Mat3<T>& m, U angle) noexcept {
            Mat3<T> out(m);
            return out.rotY(angle);
        }

        template<typename T, typename U>
        NOA_HD constexpr Mat3<T> rotZ(const Mat3<T>& m, U angle) noexcept {
            Mat3<T> out(m);
            return out.rotZ(angle);
        }

        /// Rotates @a m by an @a angle (in radians) around a given @a axis.
        template<typename T, typename U>
        NOA_HD constexpr Mat3<T> rot(const Mat3<T>& m, Float3<T> axis, U angle) noexcept {
            Mat3<T> out(m);
            return out.rot(axis, angle);
        }

        /// Rotates @a m by @a angle radians around an in-place axis (on the XY plane) at @a axis_angle radians.
        template<typename T, typename U, typename V>
        NOA_HD constexpr Mat3<T> rotInPlane(const Mat3<T>& m, U axis_angle, V angle) noexcept {
            Mat3<T> out(m);
            return out.rotInPlane(axis_angle, angle);
        }
    }
}
