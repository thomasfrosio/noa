/// \file noa/common/types/Mat22.h
/// \author Thomas - ffyr2w
/// \date 2 Jun 2021
/// A 2x2 floating-point matrix.

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Float2.h"

namespace noa {
    template<typename T> class Mat33;
    template<typename T> class Mat23;
    template<typename T> class Int2;

    /// A 2x2 floating-point matrix.
    /// \note The indexing is "row-first" (as opposed to "column-first", like in OpenGL Math),
    ///       i.e. M[r][c] with r = row index and c = column index. All indexes starts from 0.
    template<typename T>
    class Mat22 {
    public: // Type definitions
        typedef T value_type;

    public: // Component accesses
        static constexpr size_t ROWS = 2;
        static constexpr size_t COLS = 2;
        static constexpr size_t COUNT = ROWS * COLS;
        NOA_HD constexpr Float2<T>& operator[](size_t i);
        NOA_HD constexpr const Float2<T>& operator[](size_t i) const;

    public: // (Conversion) Constructors
        NOA_HD constexpr Mat22() noexcept; // identity matrix
        template<typename U> NOA_HD constexpr explicit Mat22(U s) noexcept; // equivalent to Mat22() * s
        template<typename U> NOA_HD constexpr explicit Mat22(const Float2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat22(const Mat22<U>& m) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat22(const Mat33<U>& m) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat22(const Mat23<U>& m) noexcept;

        template<typename X00, typename X01,
                 typename Y10, typename Y11>
        NOA_HD constexpr Mat22(X00 x00, X01 x01,
                               Y10 y10, Y11 y11) noexcept;

        template<typename V0, typename V1>
        NOA_HD constexpr Mat22(const Float2<V0>& r0,
                               const Float2<V1>& r1) noexcept;

        template<typename V0, typename V1>
        NOA_HD constexpr Mat22(const Int2<V0>& r0,
                               const Int2<V1>& r1) noexcept;

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Mat22<T>& operator=(const Mat22<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat22<T>& operator+=(const Mat22<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat22<T>& operator-=(const Mat22<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat22<T>& operator*=(const Mat22<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat22<T>& operator/=(const Mat22<U>& m) noexcept;

        template<typename U> NOA_HD constexpr Mat22<T>& operator+=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat22<T>& operator-=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat22<T>& operator*=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat22<T>& operator/=(U s) noexcept;

    private:
        Float2<T> m_row[ROWS];
    };

    // -- Unary operators --

    template<typename T> NOA_IHD constexpr Mat22<T> operator+(const Mat22<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat22<T> operator-(const Mat22<T>& m) noexcept;

    // -- Binary operators --

    template<typename T> NOA_IHD constexpr Mat22<T> operator+(const Mat22<T>& m1, const Mat22<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat22<T> operator+(T s, const Mat22<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat22<T> operator+(const Mat22<T>& m, T s) noexcept;

    template<typename T> NOA_IHD constexpr Mat22<T> operator-(const Mat22<T>& m1, const Mat22<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat22<T> operator-(T s, const Mat22<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat22<T> operator-(const Mat22<T>& m, T s) noexcept;

    template<typename T> NOA_IHD constexpr Mat22<T> operator*(const Mat22<T>& m1, const Mat22<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat22<T> operator*(T s, const Mat22<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat22<T> operator*(const Mat22<T>& m, T s) noexcept;
    template<typename T> NOA_IHD constexpr Float2<T> operator*(const Mat22<T>& m, const Float2<T>& column) noexcept;
    template<typename T> NOA_IHD constexpr Float2<T> operator*(const Float2<T>& row, const Mat22<T>& m) noexcept;

    template<typename T> NOA_IHD constexpr Mat22<T> operator/(const Mat22<T>& m1, const Mat22<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat22<T> operator/(T s, const Mat22<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat22<T> operator/(const Mat22<T>& m, T s) noexcept;
    template<typename T> NOA_IHD constexpr Float2<T> operator/(const Mat22<T>& m, const Float2<T>& column) noexcept;
    template<typename T> NOA_IHD constexpr Float2<T> operator/(const Float2<T>& row, const Mat22<T>& m) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_IHD constexpr bool operator==(const Mat22<T>& m1, const Mat22<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr bool operator!=(const Mat22<T>& m1, const Mat22<T>& m2) noexcept;

    namespace math {
        /// Multiplies matrix \a lhs by matrix \a rhs element-wise, i.e. `out[i][j] = lhs[i][j] * rhs[i][j]`.
        template<typename T>
        NOA_IHD constexpr Mat22<T> elementMultiply(const Mat22<T>& m1, const Mat22<T>& m2) noexcept;

        /// Given the column vector \a column and row vector \a row,
        /// computes the linear algebraic matrix multiply `c * r`.
        template<typename T> NOA_IHD constexpr Mat22<T> outerProduct(const Float2<T>& column,
                                                                     const Float2<T>& row) noexcept;

        template<typename T> NOA_IHD constexpr Mat22<T> transpose(const Mat22<T>& m) noexcept;
        template<typename T> NOA_IHD constexpr T determinant(const Mat22<T>& m) noexcept;
        template<typename T> NOA_IHD constexpr Mat22<T> inverse(const Mat22<T>& m) noexcept;

        template<uint ULP = 2, typename T>
        NOA_IHD constexpr bool isEqual(const Mat22<T>& m1, const Mat22<T>& m2, T e = 1e-6f);
    }

    using float22_t = Mat22<float>;
    using double22_t = Mat22<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 4> toArray(const Mat22<T>& v) noexcept {
        return {v[0][0], v[0][1],
                v[1][0], v[1][1]};
    }

    template<> NOA_IH std::string string::typeName<float22_t>() { return "float22"; }
    template<> NOA_IH std::string string::typeName<double22_t>() { return "double22"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Mat22<T>& m) {
        os << string::format("({},{})", m[0], m[1]);
        return os;
    }
}

#define NOA_INCLUDE_MAT22_
#include "noa/common/types/details/Mat22.inl"
#undef NOA_INCLUDE_MAT22_
