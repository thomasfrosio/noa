#pragma once

#include "noa/core/Assert.hpp"
#include "noa/core/Definitions.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/traits/Matrix.hpp"
#include "noa/core/traits/Numerics.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa {
    template<typename T>
    class Mat44;

    template<typename T>
    class Mat33;

    // A 3x4 floating-point matrix.
    template<typename Real>
    class Mat34 {
    public: // Type definitions
        using value_type = Real;
        using row_type = Vec4<value_type>;
        using vec3_type = Vec3<value_type>;
        using vec4_type = Vec4<value_type>;

    public: // Component accesses
        static constexpr size_t ROWS = 3;
        static constexpr size_t COLS = 4;
        static constexpr size_t COUNT = ROWS * COLS;
        static constexpr size_t SIZE = COUNT;
        static constexpr int64_t SSIZE = ROWS * COLS;

        template<typename I, typename = std::enable_if_t<noa::traits::is_int_v<I>>>
        NOA_HD constexpr row_type& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

        template<typename I, typename = std::enable_if_t<noa::traits::is_int_v<I>>>
        NOA_HD constexpr const row_type& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

    public: // Default constructors
        NOA_HD constexpr Mat34() noexcept
                : m_row{row_type(1, 0, 0, 0),
                        row_type(0, 1, 0, 0),
                        row_type(0, 0, 1, 0)} {}

    public: // Conversion constructors
        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat34(U s) noexcept
                : m_row{row_type(s, 0, 0, 0),
                        row_type(0, s, 0, 0),
                        row_type(0, 0, s, 0)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat34(const Vec4<U>& v) noexcept
                : m_row{row_type(v[0], 0, 0, 0),
                        row_type(0, v[1], 0, 0),
                        row_type(0, 0, v[2], 0)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat34(const Vec3<U>& v) noexcept
                : m_row{row_type(v[0], 0, 0, 0),
                        row_type(0, v[1], 0, 0),
                        row_type(0, 0, v[2], 0)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat34(const Mat44<U>& m) noexcept
                : m_row{row_type(m[0]),
                        row_type(m[1]),
                        row_type(m[2])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat34(const Mat34<U>& m) noexcept
                : m_row{row_type(m[0]),
                        row_type(m[1]),
                        row_type(m[2])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat34(const Mat33<U>& m) noexcept
                : m_row{row_type(m[0][0], m[0][1], m[0][2], 0),
                        row_type(m[1][0], m[1][1], m[1][2], 0),
                        row_type(m[2][0], m[2][1], m[2][2], 0)} {}

        template<typename U, typename V>
        NOA_HD constexpr explicit Mat34(const Mat33<U>& m, const Vec3<V>& v) noexcept
                : m_row{row_type(m[0][0], m[0][1], m[0][2], v[0]),
                        row_type(m[1][0], m[1][1], m[1][2], v[1]),
                        row_type(m[2][0], m[2][1], m[2][2], v[2])} {}

        template<typename X00, typename X01, typename X02, typename X03,
                 typename Y10, typename Y11, typename Y12, typename Y13,
                 typename Z20, typename Z21, typename Z22, typename Z23>
        NOA_HD constexpr Mat34(X00 x00, X01 x01, X02 x02, X03 x03,
                               Y10 y10, Y11 y11, Y12 y12, Y13 y13,
                               Z20 z20, Z21 z21, Z22 z22, Z23 z23) noexcept
                : m_row{row_type(x00, x01, x02, x03),
                        row_type(y10, y11, y12, y13),
                        row_type(z20, z21, z22, z23)} {}

        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat34(U* ptr) noexcept
                : m_row{row_type(ptr[0], ptr[1], ptr[2], ptr[3]),
                        row_type(ptr[4], ptr[5], ptr[6], ptr[7]),
                        row_type(ptr[8], ptr[9], ptr[10], ptr[11])} {}

        template<typename V0, typename V1, typename V2>
        NOA_HD constexpr Mat34(const Vec4<V0>& r0,
                               const Vec4<V1>& r1,
                               const Vec4<V2>& r2) noexcept
                : m_row{row_type(r0),
                        row_type(r1),
                        row_type(r2)} {}

    public: // Assignment operators
        NOA_HD constexpr Mat34& operator+=(const Mat34& m) noexcept {
            m_row[0] += m[0];
            m_row[1] += m[1];
            m_row[2] += m[2];
            return *this;
        }

        NOA_HD constexpr Mat34& operator-=(const Mat34& m) noexcept {
            m_row[0] -= m[0];
            m_row[1] -= m[1];
            m_row[2] -= m[2];
            return *this;
        }

        NOA_HD constexpr Mat34& operator+=(value_type s) noexcept {
            m_row[0] += s;
            m_row[1] += s;
            m_row[2] += s;
            return *this;
        }

        NOA_HD constexpr Mat34& operator-=(value_type s) noexcept {
            m_row[0] -= s;
            m_row[1] -= s;
            m_row[2] -= s;
            return *this;
        }

        NOA_HD constexpr Mat34& operator*=(value_type s) noexcept {
            m_row[0] *= s;
            m_row[1] *= s;
            m_row[2] *= s;
            return *this;
        }

        NOA_HD constexpr Mat34& operator/=(value_type s) noexcept {
            m_row[0] /= s;
            m_row[1] /= s;
            m_row[2] /= s;
            return *this;
        }

    public: // Assignment operators
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Mat34 operator+(const Mat34& m) noexcept {
            return m;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator-(const Mat34& m) noexcept {
            return Mat34(-m[0], -m[1], -m[2]);
        }

        // -- Binary arithmetic operators --
        [[nodiscard]] friend NOA_HD constexpr Mat34 operator+(const Mat34& m1, const Mat34& m2) noexcept {
            return Mat34(m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator+(value_type s, const Mat34& m) noexcept {
            return Mat34(s + m[0], s + m[1], s + m[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator+(const Mat34& m, value_type s) noexcept {
            return Mat34(m[0] + s, m[1] + s, m[2] + s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator-(const Mat34& m1, const Mat34& m2) noexcept {
            return Mat34(m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator-(value_type s, const Mat34& m) noexcept {
            return Mat34(s - m[0], s - m[1], s - m[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator-(const Mat34& m, value_type s) noexcept {
            return Mat34(m[0] - s, m[1] - s, m[2] - s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator*(value_type s, const Mat34& m) noexcept {
            return Mat34(m[0] * s, m[1] * s, m[2] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator*(const Mat34& m, value_type s) noexcept {
            return Mat34(m[0] * s, m[1] * s, m[2] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr vec3_type operator*(const Mat34& m, const vec4_type& column) noexcept {
            return vec3_type(noa::math::dot(m[0], column),
                             noa::math::dot(m[1], column),
                             noa::math::dot(m[2], column));
        }

        [[nodiscard]] friend NOA_HD constexpr vec4_type operator*(const vec3_type& row, const Mat34& m) noexcept {
            return vec4_type(noa::math::dot(vec3_type(m[0][0], m[1][0], m[2][0]), row),
                             noa::math::dot(vec3_type(m[0][1], m[1][1], m[2][1]), row),
                             noa::math::dot(vec3_type(m[0][2], m[1][2], m[2][2]), row),
                             noa::math::dot(vec3_type(m[0][3], m[1][3], m[2][3]), row));
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator/(value_type s, const Mat34& m) noexcept {
            return Mat34(s / m[0], s / m[1], s / m[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator/(const Mat34& m, value_type s) noexcept {
            return Mat34(m[0] / s, m[1] / s, m[2] / s);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(const Mat34& m1, const Mat34& m2) noexcept {
            return noa::all(m1[0] == m2[0]) && noa::all(m1[1] == m2[1]) && noa::all(m1[2] == m2[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(const Mat34& m1, const Mat34& m2) noexcept {
            return noa::any(m1[0] != m2[0]) || noa::any(m1[1] != m2[1]) || noa::any(m1[2] != m2[2]);
        }

    public: // Support for noa::string::human<Vec>();
        [[nodiscard]] static std::string name() {
            if constexpr (std::is_same_v<value_type, float>)
                return "Float34";
            else
                return "Double34";
        }

    private:
        row_type m_row[ROWS];
    };

    template<typename T> struct traits::proclaim_is_mat34<Mat34<T>> : std::true_type {};

    using Float34 = Mat34<float>;
    using Double34 = Mat34<double>;
}

namespace noa::math {
    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat34<T> element_multiply(Mat34<T> m1, const Mat34<T>& m2) noexcept {
        for (size_t i = 0; i < Mat34<T>::ROWS; ++i)
            m1[i] *= m2[i];
        return m1;
    }

    template<int32_t ULP = 2, typename T>
    [[nodiscard]] NOA_IHD constexpr bool are_almost_equal(
            const Mat34<T>& m1, const Mat34<T>& m2, T epsilon = 1e-6f) noexcept {
        return noa::all(are_almost_equal<ULP>(m1[0], m2[0], epsilon)) &&
               noa::all(are_almost_equal<ULP>(m1[1], m2[1], epsilon)) &&
               noa::all(are_almost_equal<ULP>(m1[2], m2[2], epsilon));
    }
}
