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
    template<typename T>
    class Mat33;

    template<typename T>
    class Mat22;

    template<typename T>
    class Float2;

    template<typename T>
    class Int3;

    /// A 2x3 floating-point matrix.
    /// \note The indexing is "row-first" (as opposed to "column-first", like in OpenGL Math),
    ///       i.e. M[r][c] with r = row index and c = column index. All indexes starts from 0.
    /// \note These matrices are quite limited compared to the squared ones and they're mostly here
    ///       to pre-multiple column vectors for 2D affine transforms.
    template<typename T>
    class Mat23 {
    public: // Type definitions
        typedef T value_type;

    public: // Component accesses
        static constexpr size_t ROWS = 2;
        static constexpr size_t COLS = 3;
        static constexpr size_t COUNT = ROWS * COLS;

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        NOA_HD constexpr Float3<T>& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        NOA_HD constexpr const Float3<T>& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

    public: // Default constructors
        NOA_HD constexpr Mat23() noexcept
                : m_row{Float3<T>(1, 0, 0),
                        Float3<T>(0, 1, 0)} {}

        constexpr Mat23(const Mat23&) noexcept = default;
        constexpr Mat23(Mat23&&) noexcept = default;

    public: // (Conversion) Constructors
        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat23(U s) noexcept
                : m_row{Float3<T>(s, 0, 0),
                        Float3<T>(0, s, 0)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat23(Float2<U> v) noexcept
                : m_row{Float3<T>(v[0], 0, 0),
                        Float3<T>(0, v[1], 0)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat23(Mat33<U> m) noexcept
                : m_row{Float3<T>(m[0]),
                        Float3<T>(m[1])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat23(Mat23<U> m) noexcept
                : m_row{Float3<T>(m[0]),
                        Float3<T>(m[1])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat23(Mat22<U> m) noexcept
                : m_row{Float3<T>(m[0][0], m[0][1], 0),
                        Float3<T>(m[1][0], m[1][1], 0)} {}

        template<typename U, typename V>
        NOA_HD constexpr explicit Mat23(Mat22<U> m, Float2<V> v) noexcept
                : m_row{Float3<T>(m[0][0], m[0][1], v[0]),
                        Float3<T>(m[1][0], m[1][1], v[1])} {}

        template<typename X00, typename X01, typename X02,
                 typename Y10, typename Y11, typename Y12>
        NOA_HD constexpr Mat23(X00 x00, X01 x01, X02 x02,
                               Y10 y10, Y11 y11, Y12 y12) noexcept
                : m_row{Float3<T>(x00, x01, x02),
                        Float3<T>(y10, y11, y12)} {}

        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat23(U* ptr) noexcept
                : m_row{Float3<T>(ptr[0], ptr[1], ptr[2]),
                        Float3<T>(ptr[3], ptr[4], ptr[5])} {}

        template<typename V0, typename V1>
        NOA_HD constexpr Mat23(Float3<V0> r0,
                               Float3<V1> r1) noexcept
                : m_row{Float3<T>(r0),
                        Float3<T>(r1)} {}

        template<typename V0, typename V1>
        NOA_HD constexpr Mat23(Int3<V0> r0,
                               Int3<V1> r1) noexcept
                : m_row{Float3<T>(r0),
                        Float3<T>(r1)} {}

    public: // Assignment operators
        constexpr Mat23& operator=(const Mat23& v) noexcept = default;
        constexpr Mat23& operator=(Mat23&& v) noexcept = default;

        NOA_HD constexpr Mat23& operator+=(Mat23 m) noexcept {
            m_row[0] += m[0];
            m_row[1] += m[1];
            return *this;
        }

        NOA_HD constexpr Mat23& operator-=(Mat23 m) noexcept {
            m_row[0] -= m[0];
            m_row[1] -= m[1];
            return *this;
        }

        NOA_HD constexpr Mat23& operator+=(T s) noexcept {
            m_row[0] += s;
            m_row[1] += s;
            return *this;
        }

        NOA_HD constexpr Mat23& operator-=(T s) noexcept {
            m_row[0] -= s;
            m_row[1] -= s;
            return *this;
        }

        NOA_HD constexpr Mat23& operator*=(T s) noexcept {
            m_row[0] *= s;
            m_row[1] *= s;
            return *this;
        }

        NOA_HD constexpr Mat23& operator/=(T s) noexcept {
            m_row[0] /= s;
            m_row[1] /= s;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        friend NOA_HD constexpr Mat23 operator+(Mat23 m) noexcept {
            return m;
        }

        friend NOA_HD constexpr Mat23 operator-(Mat23 m) noexcept {
            return Mat23(-m[0], -m[1]);
        }

        // -- Binary arithmetic operators --
        friend NOA_HD constexpr Mat23 operator+(Mat23 m1, Mat23 m2) noexcept {
            return Mat23(m1[0] + m2[0], m1[1] + m2[1]);
        }

        friend NOA_HD constexpr Mat23 operator+(T s, Mat23 m) noexcept {
            return Mat23(s + m[0], s + m[1]);
        }

        friend NOA_HD constexpr Mat23 operator+(Mat23 m, T s) noexcept {
            return Mat23(m[0] + s, m[1] + s);
        }

        friend NOA_HD constexpr Mat23 operator-(Mat23 m1, Mat23 m2) noexcept {
            return Mat23(m1[0] - m2[0], m1[1] - m2[1]);
        }

        friend NOA_HD constexpr Mat23 operator-(T s, Mat23 m) noexcept {
            return Mat23(s - m[0], s - m[1]);
        }

        friend NOA_HD constexpr Mat23 operator-(Mat23 m, T s) noexcept {
            return Mat23(m[0] - s, m[1] - s);
        }

        friend NOA_HD constexpr Mat23 operator*(T s, Mat23 m) noexcept {
            return Mat23(m[0] * s, m[1] * s);
        }

        friend NOA_HD constexpr Mat23 operator*(Mat23 m, T s) noexcept {
            return Mat23(m[0] * s, m[1] * s);
        }

        friend NOA_HD constexpr Float2<T> operator*(Mat23 m, const Float3<T>& column) noexcept {
            return Float2<T>(math::dot(m[0], column), math::dot(m[1], column));
        }

        friend NOA_HD constexpr Float3<T> operator*(const Float2<T>& row, Mat23 m) noexcept {
            return Float3<T>(math::dot(Float2<T>(m[0][0], m[1][0]), row),
                             math::dot(Float2<T>(m[0][1], m[1][1]), row),
                             math::dot(Float2<T>(m[0][2], m[1][2]), row));
        }

        friend NOA_HD constexpr Mat23 operator/(T s, Mat23 m) noexcept {
            return Mat23(s / m[0], s / m[1]);
        }

        friend NOA_HD constexpr Mat23 operator/(Mat23 m, T s) noexcept {
            return Mat23(m[0] / s, m[1] / s);
        }

        friend NOA_HD constexpr bool operator==(Mat23 m1, Mat23 m2) noexcept {
            return all(m1[0] == m2[0]) && all(m1[1] == m2[1]);
        }

        friend NOA_HD constexpr bool operator!=(Mat23 m1, Mat23 m2) noexcept {
            return all(m1[0] != m2[0]) && all(m1[1] != m2[1]);
        }

    public:
        [[nodiscard]] NOA_HD constexpr const T* get() const noexcept { return m_row[0].get(); }
        [[nodiscard]] NOA_HD constexpr T* get() noexcept { return m_row[0].get(); }

    private:
        Float3<T> m_row[ROWS];
    };

    namespace math {
        /// Multiplies matrix \a lhs by matrix \a rhs element-wise, i.e. `out[i][j] = lhs[i][j] * rhs[i][j]`.
        template<typename T>
        NOA_IHD constexpr Mat23<T> elementMultiply(Mat23<T> m1, Mat23<T> m2) noexcept {
            Mat23<T> out;
            for (size_t i = 0; i < Mat23<T>::ROWS; ++i)
                out[i] = m1[i] * m2[i];
            return out;
        }

        template<uint ULP = 2, typename T>
        NOA_IHD constexpr bool isEqual(Mat23<T> m1, Mat23<T> m2, T e = 1e-6f) noexcept {
            return all(isEqual<ULP>(m1[0], m2[0], e)) && all(isEqual<ULP>(m1[1], m2[1], e));
        }
    }

    namespace traits {
        template<typename>
        struct p_is_float23 : std::false_type {};
        template<typename T>
        struct p_is_float23<noa::Mat23<T>> : std::true_type {};
        template<typename T> using is_float23 = std::bool_constant<p_is_float23<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_float23_v = is_float23<T>::value;

        template<typename T>
        struct proclaim_is_floatXX<noa::Mat23<T>> : std::true_type {};
    }

    using float23_t = Mat23<float>;
    using double23_t = Mat23<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 6> toArray(Mat23<T> v) noexcept {
        return {v[0][0], v[0][1], v[0][2],
                v[1][0], v[1][1], v[1][2]};
    }

    template<>
    NOA_IH std::string string::human<float23_t>() { return "float23"; }
    template<>
    NOA_IH std::string string::human<double23_t>() { return "double23"; }
}
