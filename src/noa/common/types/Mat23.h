/// \file noa/common/types/Mat23.h
/// \author Thomas - ffyr2w
/// \date 2 Jun 2021
/// A 2x3 floating-point matrix.

#pragma once

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Float3.h"

namespace noa {
    template<typename T> class Mat33;
    template<typename T> class Mat22;
    template<typename T> class Float3;
    template<typename T> class Float2;
    template<typename T> class Int3;

    /// A 2x3 floating-point matrix.
    /// \note The indexing is "row-first" (as opposed to "column-first", like in OpenGL Math),
    ///       i.e. M[r][c] with r = row index and c = column index. All indexes starts from 0.
    /// \note These matrices are quite limited compared to the squared ones and they're mostly there
    ///       to pre-multiple column vectors for 2D affine transforms.
    template<typename T>
    class Mat23 {
    public: // Type definitions
        typedef T value_type;

    public: // Component accesses
        static constexpr size_t ROWS = 2;
        static constexpr size_t COLS = 3;
        static constexpr size_t COUNT = ROWS * COLS;
        NOA_HD constexpr Float3<T>& operator[](size_t i);
        NOA_HD constexpr const Float3<T>& operator[](size_t i) const;

    public: // (Conversion) Constructors
        NOA_HD constexpr Mat23() noexcept; // identity matrix
        template<typename U> NOA_HD constexpr explicit Mat23(U s) noexcept; // equivalent to Mat23() * s
        template<typename U> NOA_HD constexpr explicit Mat23(const Float2<U>& v) noexcept;

        template<typename U> NOA_HD constexpr explicit Mat23(const Mat33<U>& m) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat23(const Mat23<U>& m) noexcept;
        template<typename U> NOA_HD constexpr explicit Mat23(const Mat22<U>& m) noexcept;
        template<typename U, typename V>
        NOA_HD constexpr explicit Mat23(const Mat22<U>& m, const Float2<V>& v) noexcept;

        template<typename X00, typename X01, typename X02,
                 typename Y10, typename Y11, typename Y12>
        NOA_HD constexpr Mat23(X00 x00, X01 x01, X02 x02,
                               Y10 y10, Y11 y11, Y12 y12) noexcept;

        template<typename V0, typename V1>
        NOA_HD constexpr Mat23(const Float3<V0>& r0,
                               const Float3<V1>& r1) noexcept;

        template<typename V0, typename V1>
        NOA_HD constexpr Mat23(const Int3<V0>& r0,
                               const Int3<V1>& r1) noexcept;

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Mat23<T>& operator=(const Mat23<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat23<T>& operator+=(const Mat23<U>& m) noexcept;
        template<typename U> NOA_HD constexpr Mat23<T>& operator-=(const Mat23<U>& m) noexcept;

        template<typename U> NOA_HD constexpr Mat23<T>& operator+=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat23<T>& operator-=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat23<T>& operator*=(U s) noexcept;
        template<typename U> NOA_HD constexpr Mat23<T>& operator/=(U s) noexcept;

    private:
        Float3<T> m_row[ROWS];
    };

    // -- Unary operators --

    template<typename T> NOA_IHD constexpr Mat23<T> operator+(const Mat23<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat23<T> operator-(const Mat23<T>& m) noexcept;

    // -- Binary operators --

    template<typename T> NOA_IHD constexpr Mat23<T> operator+(const Mat23<T>& m1, const Mat23<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat23<T> operator+(T s, const Mat23<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat23<T> operator+(const Mat23<T>& m, T s) noexcept;

    template<typename T> NOA_IHD constexpr Mat23<T> operator-(const Mat23<T>& m1, const Mat23<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr Mat23<T> operator-(T s, const Mat23<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat23<T> operator-(const Mat23<T>& m, T s) noexcept;

    template<typename T> NOA_IHD constexpr Mat23<T> operator*(T s, const Mat23<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat23<T> operator*(const Mat23<T>& m, T s) noexcept;
    template<typename T> NOA_IHD constexpr Float2<T> operator*(const Mat23<T>& m, const Float3<T>& column) noexcept;
    template<typename T> NOA_IHD constexpr Float3<T> operator*(const Float2<T>& row, const Mat23<T>& m) noexcept;

    template<typename T> NOA_IHD constexpr Mat23<T> operator/(T s, const Mat23<T>& m) noexcept;
    template<typename T> NOA_IHD constexpr Mat23<T> operator/(const Mat23<T>& m, T s) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_IHD constexpr bool operator==(const Mat23<T>& m1, const Mat23<T>& m2) noexcept;
    template<typename T> NOA_IHD constexpr bool operator!=(const Mat23<T>& m1, const Mat23<T>& m2) noexcept;

    namespace math {
        /// Multiplies matrix \a lhs by matrix \a rhs element-wise, i.e. `out[i][j] = lhs[i][j] * rhs[i][j]`.
        template<typename T>
        NOA_IHD constexpr Mat23<T> elementMultiply(const Mat23<T>& m1, const Mat23<T>& m2) noexcept;

        template<uint ULP = 2, typename T>
        NOA_IHD constexpr bool isEqual(const Mat23<T>& m1, const Mat23<T>& m2, T e = 1e-6f);
    }

    using float23_t = Mat23<float>;
    using double23_t = Mat23<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 6> toArray(const Mat23<T>& v) noexcept {
        return {v[0][0], v[0][1], v[0][2],
                v[1][0], v[1][1], v[1][2]};
    }

    template<> NOA_IH std::string string::typeName<float23_t>() { return "float23"; }
    template<> NOA_IH std::string string::typeName<double23_t>() { return "double23"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Mat23<T>& m) {
        os << string::format("({},{})", m[0], m[1]);
        return os;
    }
}

#define NOA_INCLUDE_MAT23_
#include "noa/common/types/details/Mat23.inl"
#undef NOA_INCLUDE_MAT23_
