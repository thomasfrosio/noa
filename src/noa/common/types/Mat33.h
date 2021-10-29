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

namespace noa {
    template<typename T> class Mat44;
    template<typename T> class Mat34;
    template<typename T> class Mat23;
    template<typename T> class Mat22;
    template<typename T> class Float3;
    template<typename T> class Float2;
    template<typename T> class Int3;

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
        NOA_HD constexpr Float3<T>& operator[](size_t i);
        NOA_HD constexpr const Float3<T>& operator[](size_t i) const;

    public: // (Conversion) Constructors
        NOA_HD constexpr Mat33() noexcept; // identity matrix
        template<typename U> NOA_HD constexpr explicit Mat33(U s) noexcept; // equivalent to Mat33() * s
        template<typename U> NOA_HD constexpr explicit Mat33(const Float3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat33(const Float2<U>& v) noexcept;

        template<typename U> NOA_HD constexpr explicit Mat33(const Mat44<U>& m) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat33(const Mat34<U>& m) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat33(const Mat33<U>& m) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat33(const Mat23<U>& m) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat33(const Mat22<U>& m) noexcept;
        template<typename U, typename V>
        NOA_HD constexpr explicit Mat33(const Mat22<U>& m, const Float2<V>& v) noexcept;

        template<typename X00, typename X01, typename X02,
                 typename Y10, typename Y11, typename Y12,
                 typename Z20, typename Z21, typename Z22>
        NOA_HD constexpr Mat33(X00 x00, X01 x01, X02 x02,
                               Y10 y10, Y11 y11, Y12 y12,
                               Z20 z20, Z21 z21, Z22 z22) noexcept;

        template<typename V0, typename V1, typename V2>
        NOA_HD constexpr Mat33(const Float3<V0>& r0,
                               const Float3<V1>& r1,
                               const Float3<V2>& r2) noexcept;

        template<typename V0, typename V1, typename V2>
        NOA_HD constexpr Mat33(const Int3<V0>& r0,
                               const Int3<V1>& r1,
                               const Int3<V2>& r2) noexcept;

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Mat33<T>& operator=(const Mat33<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat33<T>& operator+=(const Mat33<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat33<T>& operator-=(const Mat33<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat33<T>& operator*=(const Mat33<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat33<T>& operator/=(const Mat33<U>& m) noexcept;

        template<typename U> NOA_HD constexpr Mat33<T>& operator+=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat33<T>& operator-=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat33<T>& operator*=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat33<T>& operator/=(U s) noexcept;

    private:
        Float3<T> m_row[ROWS];
    };

    // -- Unary operators --

    template<typename T> NOA_IHD constexpr Mat33<T> operator+(const Mat33<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat33<T> operator-(const Mat33<T>& m) noexcept;

    // -- Binary operators --

    template<typename T> NOA_IHD constexpr Mat33<T> operator+(const Mat33<T>& m1, const Mat33<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat33<T> operator+(T s, const Mat33<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat33<T> operator+(const Mat33<T>& m, T s) noexcept;

    template<typename T> NOA_IHD constexpr Mat33<T> operator-(const Mat33<T>& m1, const Mat33<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat33<T> operator-(T s, const Mat33<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat33<T> operator-(const Mat33<T>& m, T s) noexcept;

    template<typename T> NOA_IHD constexpr Mat33<T> operator*(const Mat33<T>& m1, const Mat33<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat33<T> operator*(T s, const Mat33<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat33<T> operator*(const Mat33<T>& m, T s) noexcept;
    template<typename T> NOA_IHD constexpr Float3<T> operator*(const Mat33<T>& m, const Float3<T>& column) noexcept;
    template<typename T> NOA_IHD constexpr Float3<T> operator*(const Float3<T>& row, const Mat33<T>& m) noexcept;

    template<typename T> NOA_IHD constexpr Mat33<T> operator/(const Mat33<T>& m1, const Mat33<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat33<T> operator/(T s, const Mat33<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat33<T> operator/(const Mat33<T>& m, T s) noexcept;
    template<typename T> NOA_IHD constexpr Float3<T> operator/(const Mat33<T>& m, const Float3<T>& column) noexcept;
    template<typename T> NOA_IHD constexpr Float3<T> operator/(const Float3<T>& row, const Mat33<T>& m) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_IHD constexpr bool operator==(const Mat33<T>& m1, const Mat33<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr bool operator!=(const Mat33<T>& m1, const Mat33<T>& m2) noexcept;

    namespace math {
        /// Multiplies matrix \a lhs by matrix \a rhs element-wise, i.e. `out[i][j] = lhs[i][j] * rhs[i][j]`.
        template<typename T>
        NOA_IHD constexpr Mat33<T> elementMultiply(const Mat33<T>& m1, const Mat33<T>& m2) noexcept;

        /// Given the column vector \a column and row vector \a row,
        /// computes the linear algebraic matrix multiply `c * r`.
        template<typename T> NOA_IHD constexpr Mat33<T> outerProduct(const Float3<T>& column,
                                                                     const Float3<T>& row) noexcept;

        template<typename T> NOA_IHD constexpr Mat33<T> transpose(const Mat33<T>& m) noexcept;
        template<typename T> NOA_IHD constexpr T determinant(const Mat33<T>& m) noexcept;
        template<typename T> NOA_HD constexpr Mat33<T> inverse(const Mat33<T>& m) noexcept;

        template<uint ULP = 2, typename T>
        NOA_IHD constexpr bool isEqual(const Mat33<T>& m1, const Mat33<T>& m2, T e = 1e-6f);
    }

    using float33_t = Mat33<float>;
    using double33_t = Mat33<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 9> toArray(const Mat33<T>& v) noexcept {
        return {v[0][0], v[0][1], v[0][2],
                v[1][0], v[1][1], v[1][2],
                v[2][0], v[2][1], v[2][2]};
    }

    template<> NOA_IH std::string string::typeName<float33_t>() { return "float33"; }
    template<> NOA_IH std::string string::typeName<double33_t>() { return "double33"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Mat33<T>& m) {
        os << string::format("({},{},{})", m[0], m[1], m[2]);
        return os;
    }
}

#define NOA_INCLUDE_MAT33_
#include "noa/common/types/details/Mat33.inl"
#undef NOA_INCLUDE_MAT33_
