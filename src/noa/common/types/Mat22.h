/// \file noa/common/types/Mat22.h
/// \author Thomas - ffyr2w
/// \date 2 Jun 2021
/// A 2x2 floating-point matrix.

#pragma once

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/traits/ArrayTypes.h"
#include "noa/common/types/Float2.h"

// A few necessary forward declarations:
namespace noa {
    template<typename T>
    class Mat33;

    template<typename T>
    class Mat23;

    template<typename T>
    class Int2;

    template<typename T>
    class Mat22;

    namespace math {
        template<typename T>
        NOA_IHD constexpr Mat22<T> transpose(Mat22<T> m) noexcept;

        template<typename T>
        NOA_IHD constexpr Mat22<T> inverse(Mat22<T> m) noexcept;
    }
}

namespace noa {
    /// A 2x2 floating-point matrix.
    /// \note The indexing is "row-first" (as opposed to "column-first", like in OpenGL Math),
    ///       i.e. M[r][c] with r = row index and c = column index. All indexes starts from 0.
    template<typename T>
    class Mat22 {
    public: // Type definitions
        using value_type = T;
        using row_type = Float2<value_type>;

    public: // Component accesses
        static constexpr size_t ROWS = 2;
        static constexpr size_t COLS = 2;
        static constexpr size_t COUNT = ROWS * COLS;

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        NOA_HD constexpr Float2<T>& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        NOA_HD constexpr const Float2<T>& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

    public: // Default constructors
        NOA_HD constexpr Mat22() noexcept
                : m_row{Float2<T>(1, 0), Float2<T>(0, 1)} {}

        constexpr Mat22(const Mat22&) noexcept = default;
        constexpr Mat22(Mat22&&) noexcept = default;

    public: // Conversion constructors
        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat22(U s) noexcept
                : m_row{Float2<T>(s, 0),
                        Float2<T>(0, s)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat22(Float2<U> v) noexcept
                : m_row{Float2<T>(v[0], 0),
                        Float2<T>(0, v[1])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat22(Mat22<U> m) noexcept
                : m_row{Float2<T>(m[0]),
                        Float2<T>(m[1])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat22(Mat33<U> m) noexcept
                : m_row{Float2<T>(m[0][0], m[0][1]),
                        Float2<T>(m[1][0], m[1][1])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat22(Mat23<U> m) noexcept
                : m_row{Float2<T>(m[0][0], m[0][1]),
                        Float2<T>(m[1][0], m[1][1])} {}

        template<typename X00, typename X01,
                 typename Y10, typename Y11>
        NOA_HD constexpr Mat22(X00 x00, X01 x01,
                               Y10 y10, Y11 y11) noexcept
                : m_row{Float2<T>(x00, x01),
                        Float2<T>(y10, y11)} {}

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat22(U* ptr) noexcept
                : m_row{Float2<T>(ptr[0], ptr[1]),
                        Float2<T>(ptr[2], ptr[3])} {}

        template<typename V0, typename V1>
        NOA_HD constexpr Mat22(Float2 <V0> r0,
                               Float2 <V1> r1) noexcept
                : m_row{Float2<T>(r0),
                        Float2<T>(r1)} {}

        template<typename V0, typename V1>
        NOA_HD constexpr Mat22(Int2<V0> r0,
                               Int2<V1> r1) noexcept
                : m_row{Float2<T>(r0),
                        Float2<T>(r1)} {}

    public: // Assignment operators
        constexpr Mat22& operator=(const Mat22& v) noexcept = default;
        constexpr Mat22& operator=(Mat22&& v) noexcept = default;

        NOA_HD constexpr Mat22& operator+=(Mat22 m) noexcept {
            m_row[0] += m[0];
            m_row[1] += m[1];
            return *this;
        }

        NOA_HD constexpr Mat22& operator-=(Mat22 m) noexcept {
            m_row[0] -= m[0];
            m_row[1] -= m[1];
            return *this;
        }

        NOA_HD constexpr Mat22& operator*=(Mat22 m) noexcept {
            const T A00 = m_row[0][0];
            const T A01 = m_row[0][1];
            const T A10 = m_row[1][0];
            const T A11 = m_row[1][1];

            const T B00 = m[0][0];
            const T B01 = m[0][1];
            const T B10 = m[1][0];
            const T B11 = m[1][1];

            m_row[0][0] = A00 * B00 + A01 * B10;
            m_row[0][1] = A00 * B01 + A01 * B11;
            m_row[1][0] = A10 * B00 + A11 * B10;
            m_row[1][1] = A10 * B01 + A11 * B11;

            return *this;
        }

        NOA_HD constexpr Mat22& operator/=(Mat22 m) noexcept {
            *this *= math::inverse(m);
            return *this;
        }

        NOA_HD constexpr Mat22& operator+=(T s) noexcept {
            m_row[0] += s;
            m_row[1] += s;
            return *this;
        }

        NOA_HD constexpr Mat22& operator-=(T s) noexcept {
            m_row[0] -= s;
            m_row[1] -= s;
            return *this;
        }

        NOA_HD constexpr Mat22& operator*=(T s) noexcept {
            m_row[0] *= s;
            m_row[1] *= s;
            return *this;
        }

        NOA_HD constexpr Mat22& operator/=(T s) noexcept {
            m_row[0] /= s;
            m_row[1] /= s;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Mat22 operator+(Mat22 m) noexcept {
            return m;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator-(Mat22 m) noexcept {
            return Mat22(-m[0], -m[1]);
        }

        // -- Binary arithmetic operators --
        [[nodiscard]] friend NOA_HD constexpr Mat22 operator+(Mat22 m1, Mat22 m2) noexcept {
            return Mat22(m1[0] + m2[0], m1[1] + m2[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator+(T s, Mat22 m) noexcept {
            return Mat22(s + m[0], s + m[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator+(Mat22 m, T s) noexcept {
            return Mat22(m[0] + s, m[1] + s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator-(Mat22 m1, Mat22 m2) noexcept {
            return Mat22(m1[0] - m2[0], m1[1] - m2[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator-(T s, Mat22 m) noexcept {
            return Mat22(s - m[0], s - m[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator-(Mat22 m, T s) noexcept {
            return Mat22(m[0] - s, m[1] - s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator*(Mat22 m1, Mat22 m2) noexcept {
            Mat22 out(m1);
            out *= m2;
            return out;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator*(T s, Mat22 m) noexcept {
            return Mat22(m[0] * s,
                         m[1] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator*(Mat22 m, T s) noexcept {
            return Mat22(m[0] * s,
                         m[1] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr Float2<T> operator*(Mat22 m, Float2<T> column) noexcept {
            return Float2<T>(math::dot(m[0], column),
                             math::dot(m[1], column));
        }

        [[nodiscard]] friend NOA_HD constexpr Float2<T> operator*(Float2<T> row, Mat22 m) noexcept {
            return Float2<T>(math::dot(Float2<T>(m[0][0], m[1][0]), row),
                             math::dot(Float2<T>(m[0][1], m[1][1]), row));
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator/(Mat22 m1, Mat22 m2) noexcept {
            Mat22 out(m1);
            out /= m2;
            return out;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator/(T s, Mat22 m) noexcept {
            return Mat22(s / m[0],
                         s / m[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator/(Mat22 m, T s) noexcept {
            return Mat22(m[0] / s,
                         m[1] / s);
        }

        [[nodiscard]] friend NOA_HD constexpr Float2<T> operator/(Mat22 m, Float2<T> column) noexcept {
            return math::inverse(m) * column;
        }

        [[nodiscard]] friend NOA_HD constexpr Float2<T> operator/(Float2<T> row, Mat22 m) noexcept {
            return row * math::inverse(m);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(Mat22 m1, Mat22 m2) noexcept {
            return all(m1[0] == m2[0]) && all(m1[1] == m2[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(Mat22 m1, Mat22 m2) noexcept {
            return all(m1[0] != m2[0]) && all(m1[1] != m2[1]);
        }

    public:
        [[nodiscard]] NOA_HD constexpr const T* get() const noexcept { return m_row[0].get(); }
        [[nodiscard]] NOA_HD constexpr T* get() noexcept { return m_row[0].get(); }

        [[nodiscard]] NOA_IHD constexpr Mat22 transpose() const noexcept {
            return math::transpose(*this);
        }

    private:
        Float2<T> m_row[ROWS];
    };

    template<typename T> struct traits::proclaim_is_float22<Mat22<T>> : std::true_type {};
    template<typename T> struct traits::proclaim_is_mat22<Mat22<T>> : std::true_type {};

    using float22_t = Mat22<float>;
    using double22_t = Mat22<double>;

    template<typename T>
    [[nodiscard]] NOA_IH constexpr std::array<T, 4> toArray(const Mat22<T>& v) noexcept {
        return {v[0][0], v[0][1],
                v[1][0], v[1][1]};
    }

    template<> [[nodiscard]] NOA_IH std::string string::human<float22_t>() { return "float22"; }
    template<> [[nodiscard]] NOA_IH std::string string::human<double22_t>() { return "double22"; }
}

namespace noa::math {
    /// Multiplies matrix \a lhs by matrix \a rhs element-wise, i.e. `out[i][j] = lhs[i][j] * rhs[i][j]`.
    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat22<T> elementMultiply(const Mat22<T>& m1, const Mat22<T>& m2) noexcept {
        Mat22<T> out;
        for (size_t i = 0; i < Mat22<T>::ROWS; ++i)
            out[i] = m1[i] * m2[i];
        return out;
    }

    /// Given the column vector \a column and row vector \a row,
    /// computes the linear algebraic matrix multiply `c * r`.
    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat22<T> outerProduct(Float2<T> column, Float2<T> row) noexcept {
        return Mat22<T>(column[0] * row[0], column[0] * row[1],
                        column[1] * row[0], column[1] * row[1]);
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat22<T> transpose(Mat22<T> m) noexcept {
        return Mat22<T>(m[0][0], m[1][0],
                        m[0][1], m[1][1]);
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr T determinant(Mat22<T> m) noexcept {
        return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat22<T> inverse(Mat22<T> m) noexcept {
        T det = determinant(m);
        NOA_ASSERT(!isEqual(det, static_cast<T>(0))); // non singular
        T one_over_determinant = 1 / det;
        return Mat22<T>(+m[1][1] * one_over_determinant,
                        -m[0][1] * one_over_determinant,
                        -m[1][0] * one_over_determinant,
                        +m[0][0] * one_over_determinant);
    }

    template<uint ULP = 2, typename T>
    [[nodiscard]] NOA_IHD constexpr bool isEqual(Mat22<T> m1, Mat22<T> m2, T e = 1e-6f) {
        return all(isEqual<ULP>(m1[0], m2[0], e)) && all(isEqual<ULP>(m1[1], m2[1], e));
    }
}
