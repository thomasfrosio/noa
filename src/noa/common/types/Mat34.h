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
    template<typename T>
    class Mat44;

    template<typename T>
    class Mat33;

    template<typename T>
    class Float3;

    template<typename T>
    class Int4;

    /// A 3x4 floating-point matrix.
    /// \note The indexing is "row-first" (as opposed to "column-first", like in OpenGL Math),
    ///       i.e. M[r][c] with r = row index and c = column index. All indexes starts from 0.
    /// \note These matrices are quite limited compared to the squared ones and they're mostly here
    ///       to pre-multiple column vectors for 3D affine transforms.
    template<typename T>
    class Mat34 {
    public: // Type definitions
        typedef T value_type;

    public: // Component accesses
        static constexpr size_t ROWS = 3;
        static constexpr size_t COLS = 4;
        static constexpr size_t COUNT = ROWS * COLS;

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        NOA_HD constexpr Float4<T>& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        NOA_HD constexpr const Float4<T>& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

    public: // Default constructors
        NOA_HD constexpr Mat34() noexcept
                : m_row{Float4<T>(1, 0, 0, 0),
                        Float4<T>(0, 1, 0, 0),
                        Float4<T>(0, 0, 1, 0)} {}

        constexpr Mat34(const Mat34&) noexcept = default;
        constexpr Mat34(Mat34&&) noexcept = default;

    public: // Conversion constructors
        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat34(U s) noexcept
                : m_row{Float4<T>(s, 0, 0, 0),
                        Float4<T>(0, s, 0, 0),
                        Float4<T>(0, 0, s, 0)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat34(Float4<U> v) noexcept
                : m_row{Float4<T>(v[0], 0, 0, 0),
                        Float4<T>(0, v[1], 0, 0),
                        Float4<T>(0, 0, v[2], 0)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat34(Float3<U> v) noexcept
                : m_row{Float4<T>(v[0], 0, 0, 0),
                        Float4<T>(0, v[1], 0, 0),
                        Float4<T>(0, 0, v[2], 0)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat34(Mat44<U> m) noexcept
                : m_row{Float4<T>(m[0]),
                        Float4<T>(m[1]),
                        Float4<T>(m[2])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat34(Mat34<U> m) noexcept
                : m_row{Float4<T>(m[0]),
                        Float4<T>(m[1]),
                        Float4<T>(m[2])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat34(Mat33<U> m) noexcept
                : m_row{Float4<T>(m[0][0], m[0][1], m[0][2], 0),
                        Float4<T>(m[1][0], m[1][1], m[1][2], 0),
                        Float4<T>(m[2][0], m[2][1], m[2][2], 0)} {}

        template<typename U, typename V>
        NOA_HD constexpr explicit Mat34(Mat33<U> m, Float3<V> v) noexcept
                : m_row{Float4<T>(m[0][0], m[0][1], m[0][2], v[0]),
                        Float4<T>(m[1][0], m[1][1], m[1][2], v[1]),
                        Float4<T>(m[2][0], m[2][1], m[2][2], v[2])} {}

        template<typename X00, typename X01, typename X02, typename X03,
                 typename Y10, typename Y11, typename Y12, typename Y13,
                 typename Z20, typename Z21, typename Z22, typename Z23>
        NOA_HD constexpr Mat34(X00 x00, X01 x01, X02 x02, X03 x03,
                               Y10 y10, Y11 y11, Y12 y12, Y13 y13,
                               Z20 z20, Z21 z21, Z22 z22, Z23 z23) noexcept
                : m_row{Float4<T>(x00, x01, x02, x03),
                        Float4<T>(y10, y11, y12, y13),
                        Float4<T>(z20, z21, z22, z23)} {}

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat34(U* ptr) noexcept
                : m_row{Float4<T>(ptr[0], ptr[1], ptr[2], ptr[3]),
                        Float4<T>(ptr[4], ptr[5], ptr[6], ptr[7]),
                        Float4<T>(ptr[8], ptr[9], ptr[10], ptr[11])} {}

        template<typename V0, typename V1, typename V2>
        NOA_HD constexpr Mat34(Float4<V0> r0,
                               Float4<V1> r1,
                               Float4<V2> r2) noexcept
                : m_row{Float4<T>(r0),
                        Float4<T>(r1),
                        Float4<T>(r2)} {}

        template<typename V0, typename V1, typename V2>
        NOA_HD constexpr Mat34(Int4<V0> r0,
                               Int4<V1> r1,
                               Int4<V2> r2) noexcept
                : m_row{Float4<T>(r0),
                        Float4<T>(r1),
                        Float4<T>(r2)} {}

    public: // Assignment operators
        constexpr Mat34& operator=(const Mat34& v) noexcept = default;
        constexpr Mat34& operator=(Mat34&& v) noexcept = default;

        NOA_HD constexpr Mat34& operator+=(Mat34 m) noexcept {
            m_row[0] += m[0];
            m_row[1] += m[1];
            m_row[2] += m[2];
            return *this;
        }

        NOA_HD constexpr Mat34& operator-=(Mat34 m) noexcept {
            m_row[0] -= m[0];
            m_row[1] -= m[1];
            m_row[2] -= m[2];
            return *this;
        }

        NOA_HD constexpr Mat34& operator+=(T s) noexcept {
            m_row[0] += s;
            m_row[1] += s;
            m_row[2] += s;
            return *this;
        }

        NOA_HD constexpr Mat34& operator-=(T s) noexcept {
            m_row[0] -= s;
            m_row[1] -= s;
            m_row[2] -= s;
            return *this;
        }

        NOA_HD constexpr Mat34& operator*=(T s) noexcept {
            m_row[0] *= s;
            m_row[1] *= s;
            m_row[2] *= s;
            return *this;
        }

        NOA_HD constexpr Mat34& operator/=(T s) noexcept {
            m_row[0] /= s;
            m_row[1] /= s;
            m_row[2] /= s;
            return *this;
        }

    public: // Assignment operators
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Mat34 operator+(Mat34 m) noexcept {
            return m;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator-(Mat34 m) noexcept {
            return Mat34(-m[0], -m[1], -m[2]);
        }

        // -- Binary arithmetic operators --
        [[nodiscard]] friend NOA_HD constexpr Mat34 operator+(Mat34 m1, Mat34 m2) noexcept {
            return Mat34(m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator+(T s, Mat34 m) noexcept {
            return Mat34(s + m[0], s + m[1], s + m[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator+(Mat34 m, T s) noexcept {
            return Mat34(m[0] + s, m[1] + s, m[2] + s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator-(Mat34 m1, Mat34 m2) noexcept {
            return Mat34(m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator-(T s, Mat34 m) noexcept {
            return Mat34(s - m[0], s - m[1], s - m[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator-(Mat34 m, T s) noexcept {
            return Mat34(m[0] - s, m[1] - s, m[2] - s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator*(T s, Mat34 m) noexcept {
            return Mat34(m[0] * s, m[1] * s, m[2] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator*(Mat34 m, T s) noexcept {
            return Mat34(m[0] * s, m[1] * s, m[2] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr Float3<T> operator*(Mat34 m, const Float4<T>& column) noexcept {
            return Float3<T>(math::dot(m[0], column),
                             math::dot(m[1], column),
                             math::dot(m[2], column));
        }

        [[nodiscard]] friend NOA_HD constexpr Float4<T> operator*(const Float3<T>& row, Mat34 m) noexcept {
            return Float4<T>(math::dot(Float3<T>(m[0][0], m[1][0], m[2][0]), row),
                             math::dot(Float3<T>(m[0][1], m[1][1], m[2][1]), row),
                             math::dot(Float3<T>(m[0][2], m[1][2], m[2][2]), row),
                             math::dot(Float3<T>(m[0][3], m[1][3], m[2][3]), row));
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator/(T s, Mat34 m) noexcept {
            return Mat34(s / m[0], s / m[1], s / m[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator/(Mat34 m, T s) noexcept {
            return Mat34(m[0] / s, m[1] / s, m[2] / s);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(Mat34 m1, Mat34 m2) noexcept {
            return all(m1[0] == m2[0]) && all(m1[1] == m2[1]) && all(m1[2] == m2[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(Mat34 m1, Mat34 m2) noexcept {
            return all(m1[0] != m2[0]) && all(m1[1] != m2[1]) && all(m1[2] != m2[2]);
        }

    public:
        [[nodiscard]] NOA_HD constexpr const T* get() const noexcept { return m_row[0].get(); }
        [[nodiscard]] NOA_HD constexpr T* get() noexcept { return m_row[0].get(); }

    private:
        Float4<T> m_row[ROWS];
    };

    namespace math {
        /// Multiplies matrix \a lhs by matrix \a rhs element-wise, i.e. `out[i][j] = lhs[i][j] * rhs[i][j]`.
        template<typename T>
        [[nodiscard]] NOA_IHD constexpr Mat34<T> elementMultiply(Mat34<T> m1, Mat34<T> m2) noexcept {
            Mat34<T> out;
            for (size_t i = 0; i < Mat34<T>::ROWS; ++i)
                out[i] = m1[i] * m2[i];
            return out;
        }

        template<uint ULP = 2, typename T>
        [[nodiscard]] NOA_IHD constexpr bool isEqual(Mat34<T> m1, Mat34<T> m2, T e = 1e-6f) noexcept {
            return all(isEqual<ULP>(m1[0], m2[0], e)) &&
                   all(isEqual<ULP>(m1[1], m2[1], e));
        }
    }

    namespace traits {
        template<typename>
        struct p_is_float34 : std::false_type {};
        template<typename T>
        struct p_is_float34<Mat34<T>> : std::true_type {};
        template<typename T> using is_float34 = std::bool_constant<p_is_float34<traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_float34_v = is_float34<T>::value;

        template<typename T>
        struct proclaim_is_floatXX<Mat34<T>> : std::true_type {};
    }

    using float34_t = Mat34<float>;
    using double34_t = Mat34<double>;

    template<typename T>
    [[nodiscard]] NOA_IH constexpr std::array<T, 12> toArray(Mat34<T> v) noexcept {
        return {v[0][0], v[0][1], v[0][2], v[0][3],
                v[1][0], v[1][1], v[1][2], v[1][3],
                v[2][0], v[2][1], v[2][2], v[2][3]};
    }

    template<> [[nodiscard]] NOA_IH std::string string::human<float34_t>() { return "float34"; }
    template<> [[nodiscard]] NOA_IH std::string string::human<double34_t>() { return "double34"; }
}
