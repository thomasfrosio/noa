#pragma once

#include "noa/core/Assert.hpp"
#include "noa/core/Definitions.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/traits/Matrix.hpp"
#include "noa/core/traits/Numerics.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa {
    template<typename T>
    class Mat33;

    template<typename T>
    class Mat22;

    // A 2x3 floating-point matrix.
    template<typename Real>
    class Mat23 {
    public: // Type definitions
        using value_type = Real;
        using row_type = Vec3<value_type>;
        using vec2_type = Vec2<value_type>;
        using vec3_type = Vec3<value_type>;

    public: // Component accesses
        static constexpr size_t ROWS = 2;
        static constexpr size_t COLS = 3;
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
        NOA_HD constexpr Mat23() noexcept
                : m_row{row_type(1, 0, 0),
                        row_type(0, 1, 0)} {}

    public: // (Conversion) Constructors
        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat23(U s) noexcept
                : m_row{row_type(s, 0, 0),
                        row_type(0, s, 0)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat23(const Vec2<U>& v) noexcept
                : m_row{row_type(v[0], 0, 0),
                        row_type(0, v[1], 0)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat23(const Mat33<U>& m) noexcept
                : m_row{row_type(m[0]),
                        row_type(m[1])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat23(const Mat23<U>& m) noexcept
                : m_row{row_type(m[0]),
                        row_type(m[1])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat23(const Mat22<U>& m) noexcept
                : m_row{row_type(m[0][0], m[0][1], 0),
                        row_type(m[1][0], m[1][1], 0)} {}

        template<typename U, typename V>
        NOA_HD constexpr explicit Mat23(const Mat22<U>& m, const Vec2<V>& v) noexcept
                : m_row{row_type(m[0][0], m[0][1], v[0]),
                        row_type(m[1][0], m[1][1], v[1])} {}

        template<typename X00, typename X01, typename X02,
                 typename Y10, typename Y11, typename Y12>
        NOA_HD constexpr Mat23(X00 x00, X01 x01, X02 x02,
                               Y10 y10, Y11 y11, Y12 y12) noexcept
                : m_row{row_type(x00, x01, x02),
                        row_type(y10, y11, y12)} {}

        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat23(U* ptr) noexcept
                : m_row{row_type(ptr[0], ptr[1], ptr[2]),
                        row_type(ptr[3], ptr[4], ptr[5])} {}

        template<typename V0, typename V1>
        NOA_HD constexpr Mat23(const Vec3<V0>& r0,
                               const Vec3<V1>& r1) noexcept
                : m_row{row_type(r0),
                        row_type(r1)} {}

    public: // Assignment operators
        NOA_HD constexpr Mat23& operator+=(const Mat23& m) noexcept {
            m_row[0] += m[0];
            m_row[1] += m[1];
            return *this;
        }

        NOA_HD constexpr Mat23& operator-=(const Mat23& m) noexcept {
            m_row[0] -= m[0];
            m_row[1] -= m[1];
            return *this;
        }

        NOA_HD constexpr Mat23& operator+=(value_type s) noexcept {
            m_row[0] += s;
            m_row[1] += s;
            return *this;
        }

        NOA_HD constexpr Mat23& operator-=(value_type s) noexcept {
            m_row[0] -= s;
            m_row[1] -= s;
            return *this;
        }

        NOA_HD constexpr Mat23& operator*=(value_type s) noexcept {
            m_row[0] *= s;
            m_row[1] *= s;
            return *this;
        }

        NOA_HD constexpr Mat23& operator/=(value_type s) noexcept {
            m_row[0] /= s;
            m_row[1] /= s;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Mat23 operator+(const Mat23& m) noexcept {
            return m;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator-(const Mat23& m) noexcept {
            return Mat23(-m[0], -m[1]);
        }

        // -- Binary arithmetic operators --
        [[nodiscard]] friend NOA_HD constexpr Mat23 operator+(const Mat23& m1, const Mat23& m2) noexcept {
            return Mat23(m1[0] + m2[0], m1[1] + m2[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator+(value_type s, const Mat23& m) noexcept {
            return Mat23(s + m[0], s + m[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator+(const Mat23& m, value_type s) noexcept {
            return Mat23(m[0] + s, m[1] + s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator-(const Mat23& m1, const Mat23& m2) noexcept {
            return Mat23(m1[0] - m2[0], m1[1] - m2[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator-(value_type s, const Mat23& m) noexcept {
            return Mat23(s - m[0], s - m[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator-(const Mat23& m, value_type s) noexcept {
            return Mat23(m[0] - s, m[1] - s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator*(value_type s, const Mat23& m) noexcept {
            return Mat23(m[0] * s, m[1] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator*(const Mat23& m, value_type s) noexcept {
            return Mat23(m[0] * s, m[1] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr vec2_type operator*(const Mat23& m, const vec3_type& column) noexcept {
            return vec2_type(math::dot(m[0], column), math::dot(m[1], column));
        }

        [[nodiscard]] friend NOA_HD constexpr vec3_type operator*(const vec2_type& row, const Mat23& m) noexcept {
            return vec3_type(math::dot(vec2_type(m[0][0], m[1][0]), row),
                             math::dot(vec2_type(m[0][1], m[1][1]), row),
                             math::dot(vec2_type(m[0][2], m[1][2]), row));
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator/(value_type s, const Mat23& m) noexcept {
            return Mat23(s / m[0], s / m[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator/(const Mat23& m, value_type s) noexcept {
            return Mat23(m[0] / s, m[1] / s);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(const Mat23& m1, const Mat23& m2) noexcept {
            return noa::all(m1[0] == m2[0]) && noa::all(m1[1] == m2[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(const Mat23& m1, const Mat23& m2) noexcept {
            return noa::any(m1[0] != m2[0]) || noa::any(m1[1] != m2[1]);
        }

    public: // Support for noa::string::human<Vec>();
        [[nodiscard]] static std::string name() {
            if constexpr (std::is_same_v<value_type, float>)
                return "Float23";
            else
                return "Double23";
        }

    private:
        row_type m_row[ROWS];
    };

    template<typename T> struct traits::proclaim_is_mat23<Mat23<T>> : std::true_type {};

    using Float23 = Mat23<float>;
    using Double23 = Mat23<double>;
}

namespace noa::math {
    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat23<T> element_multiply(Mat23<T> m1, const Mat23<T>& m2) noexcept {
        for (size_t i = 0; i < Mat23<T>::ROWS; ++i)
            m1[i] *= m2[i];
        return m1;
    }

    template<int32_t ULP = 2, typename T>
    [[nodiscard]] NOA_IHD constexpr bool are_almost_equal(
            const Mat23<T>& m1, const Mat23<T>& m2, T epsilon = 1e-6f) noexcept {
        return noa::all(are_almost_equal<ULP>(m1[0], m2[0], epsilon)) &&
               noa::all(are_almost_equal<ULP>(m1[1], m2[1], epsilon));
    }
}
