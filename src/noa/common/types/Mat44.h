/// \file noa/common/types/Mat44.h
/// \author Thomas - ffyr2w
/// \date 2 Jun 2021
/// A 4x4 floating-point matrix.

#pragma once

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Float4.h"

namespace noa {
    template<typename T> class Mat34;
    template<typename T> class Mat33;
    template<typename T> class Float4;
    template<typename T> class Float3;
    template<typename T> class Int4;

    /// A 4x4 floating-point matrix.
    /// \note The indexing is "row-first" (as opposed to "column-first", like in OpenGL Math),
    ///       i.e. M[r][c] with r = row index and c = column index. All indexes starts from 0.
    template<typename T>
    class Mat44 {
    public: // Type definitions
        typedef T value_type;

    public: // Component accesses
        static constexpr size_t ROWS = 4;
        static constexpr size_t COLS = 4;
        static constexpr size_t COUNT = ROWS * COLS;
        NOA_HD constexpr Float4<T>& operator[](size_t i);
        NOA_HD constexpr const Float4<T>& operator[](size_t i) const;

    public: // (Conversion) Constructors
        NOA_HD constexpr Mat44() noexcept; // identity matrix
        template<typename U> NOA_HD constexpr explicit Mat44(U s) noexcept; // equivalent to Mat44() * s
        template<typename U> NOA_HD constexpr explicit Mat44(const Float4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat44(const Float3<U>& v) noexcept;

        template<typename U> NOA_HD constexpr explicit Mat44(const Mat44<U>& m) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat44(const Mat34<U>& m) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat44(const Mat33<U>& m) noexcept;
        template<typename U, typename V>
        NOA_HD constexpr explicit Mat44(const Mat33<U>& m, const Float3<V>& v) noexcept;

        template<typename X00, typename X01, typename X02, typename X03,
                 typename Y10, typename Y11, typename Y12, typename Y13,
                 typename Z20, typename Z21, typename Z22, typename Z23,
                 typename W30, typename W31, typename W32, typename W33>
        NOA_HD constexpr Mat44(X00 x00, X01 x01, X02 x02, X03 x03,
                               Y10 y10, Y11 y11, Y12 y12, Y13 y13,
                               Z20 z20, Z21 z21, Z22 z22, Z23 z23,
                               W30 w30, W31 w31, W32 w32, W33 w33) noexcept;

        template<typename V0, typename V1, typename V2, typename V3>
        NOA_HD constexpr Mat44(const Float4<V0>& r0,
                               const Float4<V1>& r1,
                               const Float4<V2>& r2,
                               const Float4<V3>& r3) noexcept;

        template<typename V0, typename V1, typename V2, typename V3>
        NOA_HD constexpr Mat44(const Int4<V0>& r0,
                               const Int4<V1>& r1,
                               const Int4<V2>& r2,
                               const Int4<V3>& r3) noexcept;

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Mat44<T>& operator=(const Mat44<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat44<T>& operator+=(const Mat44<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat44<T>& operator-=(const Mat44<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat44<T>& operator*=(const Mat44<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat44<T>& operator/=(const Mat44<U>& m) noexcept;

        template<typename U> NOA_HD constexpr Mat44<T>& operator+=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat44<T>& operator-=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat44<T>& operator*=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat44<T>& operator/=(U s) noexcept;

    private:
        Float4<T> m_row[ROWS];
    };

    // -- Unary operators --

    template<typename T> NOA_IHD constexpr Mat44<T> operator+(const Mat44<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat44<T> operator-(const Mat44<T>& m) noexcept;

    // -- Binary operators --

    template<typename T> NOA_IHD constexpr Mat44<T> operator+(const Mat44<T>& m1, const Mat44<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat44<T> operator+(T s, const Mat44<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat44<T> operator+(const Mat44<T>& m, T s) noexcept;

    template<typename T> NOA_IHD constexpr Mat44<T> operator-(const Mat44<T>& m1, const Mat44<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat44<T> operator-(T s, const Mat44<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat44<T> operator-(const Mat44<T>& m, T s) noexcept;

    template<typename T> NOA_IHD constexpr Mat44<T> operator*(const Mat44<T>& m1, const Mat44<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat44<T> operator*(T s, const Mat44<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat44<T> operator*(const Mat44<T>& m, T s) noexcept;
    template<typename T> NOA_IHD constexpr Float4<T> operator*(const Mat44<T>& m, const Float4<T>& column) noexcept;
    template<typename T> NOA_IHD constexpr Float4<T> operator*(const Float4<T>& row, const Mat44<T>& m) noexcept;

    template<typename T> NOA_IHD constexpr Mat44<T> operator/(const Mat44<T>& m1, const Mat44<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat44<T> operator/(T s, const Mat44<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat44<T> operator/(const Mat44<T>& m, T s) noexcept;
    template<typename T> NOA_IHD constexpr Float4<T> operator/(const Mat44<T>& m, const Float4<T>& column) noexcept;
    template<typename T> NOA_IHD constexpr Float4<T> operator/(const Float4<T>& row, const Mat44<T>& m) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_IHD constexpr bool operator==(const Mat44<T>& m1, const Mat44<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr bool operator!=(const Mat44<T>& m1, const Mat44<T>& m2) noexcept;

    namespace math {
        /// Multiplies matrix \a lhs by matrix \a rhs element-wise, i.e. `out[i][j] = lhs[i][j] * rhs[i][j]`.
        template<typename T>
        NOA_IHD constexpr Mat44<T> elementMultiply(const Mat44<T>& m1, const Mat44<T>& m2) noexcept;

        /// Given the column vector \a column and row vector \a row,
        /// computes the linear algebraic matrix multiply `c * r`.
        template<typename T> NOA_IHD constexpr Mat44<T> outerProduct(const Float4<T>& column,
                                                                     const Float4<T>& row) noexcept;

        template<typename T> NOA_IHD constexpr Mat44<T> transpose(const Mat44<T>& m) noexcept;
        template<typename T> NOA_HD constexpr T determinant(const Mat44<T>& m) noexcept;
        template<typename T> NOA_HD constexpr Mat44<T> inverse(const Mat44<T>& m) noexcept;

        template<uint ULP = 2, typename T>
        NOA_IHD constexpr bool isEqual(const Mat44<T>& m1, const Mat44<T>& m2, T e = 1e-6f);
    }

    using float44_t = Mat44<float>;
    using double44_t = Mat44<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 16> toArray(const Mat44<T>& v) noexcept {
        return {v[0][0], v[0][1], v[0][2], v[0][3],
                v[1][0], v[1][1], v[1][2], v[1][3],
                v[2][0], v[2][1], v[2][2], v[2][3],
                v[3][0], v[3][1], v[3][2], v[3][3]};
    }

    template<> NOA_IH std::string string::typeName<float44_t>() { return "float44"; }
    template<> NOA_IH std::string string::typeName<double44_t>() { return "double44"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Mat44<T>& m) {
        os << string::format("({},{},{},{})", m[0], m[1], m[2], m[3]);
        return os;
    }
}

#define NOA_INCLUDE_MAT44_
#include "noa/common/types/details/Mat44.inl"
#undef NOA_INCLUDE_MAT44_
