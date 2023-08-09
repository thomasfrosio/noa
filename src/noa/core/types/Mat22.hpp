#pragma once

#include "noa/core/Assert.hpp"
#include "noa/core/Definitions.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/traits/Matrix.hpp"
#include "noa/core/traits/Numerics.hpp"
#include "noa/core/types/Vec.hpp"

// A few necessary forward declarations:
namespace noa {
    template<typename T>
    class Mat22;

    namespace math {
        template<typename T>
        NOA_IHD constexpr Mat22<T> transpose(const Mat22<T>& m) noexcept;

        template<typename T>
        NOA_IHD constexpr Mat22<T> inverse(const Mat22<T>& m) noexcept;
    }
}

namespace noa {
    // A 2x2 floating-point matrix.
    template<typename Real>
    class Mat22 {
    public: // Type definitions
        static_assert(!std::is_same_v<Real, Half>);
        using value_type = Real;
        using mutable_value_type = value_type;
        using row_type = Vec2<value_type>;

    public: // Component accesses
        static constexpr size_t ROWS = 2;
        static constexpr size_t COLS = 2;
        static constexpr size_t COUNT = ROWS * COLS;
        static constexpr size_t SIZE = COUNT;
        static constexpr int64_t SSIZE = ROWS * COLS;

        template<typename I, typename = std::enable_if_t<nt::is_int_v<I>>>
        NOA_HD constexpr row_type& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

        template<typename I, typename = std::enable_if_t<nt::is_int_v<I>>>
        NOA_HD constexpr const row_type& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return m_row[i];
        }

    public: // Default constructors
        NOA_HD constexpr Mat22() noexcept
                : m_row{row_type(1, 0),
                        row_type(0, 1)} {}

    public: // Conversion constructors
        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat22(U s) noexcept
                : m_row{row_type(s, 0),
                        row_type(0, s)} {}

        template<typename U>
        NOA_HD constexpr explicit Mat22(const Vec2<U>& v) noexcept
                : m_row{row_type(v[0], 0),
                        row_type(0, v[1])} {}

        template<typename U>
        NOA_HD constexpr explicit Mat22(const Mat22<U>& m) noexcept
                : m_row{row_type(m[0]),
                        row_type(m[1])} {}

        template<typename X00, typename X01,
                 typename Y10, typename Y11>
        NOA_HD constexpr Mat22(X00 x00, X01 x01,
                               Y10 y10, Y11 y11) noexcept
                : m_row{row_type(x00, x01),
                        row_type(y10, y11)} {}

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        NOA_HD constexpr explicit Mat22(U* ptr) noexcept
                : m_row{row_type(ptr[0], ptr[1]),
                        row_type(ptr[2], ptr[3])} {}

        template<typename V0, typename V1>
        NOA_HD constexpr Mat22(const Vec2<V0>& r0,
                               const Vec2<V1>& r1) noexcept
                : m_row{row_type(r0),
                        row_type(r1)} {}

    public: // Assignment operators
        NOA_HD constexpr Mat22& operator+=(const Mat22& m) noexcept {
            m_row[0] += m[0];
            m_row[1] += m[1];
            return *this;
        }

        NOA_HD constexpr Mat22& operator-=(const Mat22& m) noexcept {
            m_row[0] -= m[0];
            m_row[1] -= m[1];
            return *this;
        }

        NOA_HD constexpr Mat22& operator*=(const Mat22& m) noexcept {
            const auto A00 = m_row[0][0];
            const auto A01 = m_row[0][1];
            const auto A10 = m_row[1][0];
            const auto A11 = m_row[1][1];

            const auto B00 = m[0][0];
            const auto B01 = m[0][1];
            const auto B10 = m[1][0];
            const auto B11 = m[1][1];

            m_row[0][0] = A00 * B00 + A01 * B10;
            m_row[0][1] = A00 * B01 + A01 * B11;
            m_row[1][0] = A10 * B00 + A11 * B10;
            m_row[1][1] = A10 * B01 + A11 * B11;

            return *this;
        }

        NOA_HD constexpr Mat22& operator/=(const Mat22& m) noexcept {
            *this *= math::inverse(m);
            return *this;
        }

        NOA_HD constexpr Mat22& operator+=(value_type s) noexcept {
            m_row[0] += s;
            m_row[1] += s;
            return *this;
        }

        NOA_HD constexpr Mat22& operator-=(value_type s) noexcept {
            m_row[0] -= s;
            m_row[1] -= s;
            return *this;
        }

        NOA_HD constexpr Mat22& operator*=(value_type s) noexcept {
            m_row[0] *= s;
            m_row[1] *= s;
            return *this;
        }

        NOA_HD constexpr Mat22& operator/=(value_type s) noexcept {
            m_row[0] /= s;
            m_row[1] /= s;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Mat22 operator+(const Mat22& m) noexcept {
            return m;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator-(const Mat22& m) noexcept {
            return Mat22(-m[0], -m[1]);
        }

        // -- Binary arithmetic operators --
        [[nodiscard]] friend NOA_HD constexpr Mat22 operator+(const Mat22& m1, const Mat22& m2) noexcept {
            return Mat22(m1[0] + m2[0], m1[1] + m2[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator+(value_type s, const Mat22& m) noexcept {
            return Mat22(s + m[0], s + m[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator+(const Mat22& m, value_type s) noexcept {
            return Mat22(m[0] + s, m[1] + s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator-(const Mat22& m1, const Mat22& m2) noexcept {
            return Mat22(m1[0] - m2[0], m1[1] - m2[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator-(value_type s, const Mat22& m) noexcept {
            return Mat22(s - m[0], s - m[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator-(const Mat22& m, value_type s) noexcept {
            return Mat22(m[0] - s, m[1] - s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator*(Mat22 m1, const Mat22& m2) noexcept {
            m1 *= m2;
            return m1;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator*(value_type s, const Mat22& m) noexcept {
            return Mat22(m[0] * s,
                         m[1] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator*(const Mat22& m, value_type s) noexcept {
            return Mat22(m[0] * s,
                         m[1] * s);
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator*(const Mat22& m, const row_type& column) noexcept {
            return row_type(noa::math::dot(m[0], column),
                            noa::math::dot(m[1], column));
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator*(const row_type& row, const Mat22& m) noexcept {
            return row_type(noa::math::dot(row_type(m[0][0], m[1][0]), row),
                            noa::math::dot(row_type(m[0][1], m[1][1]), row));
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator/(Mat22 m1, const Mat22& m2) noexcept {
            m1 /= m2;
            return m1;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator/(value_type s, const Mat22& m) noexcept {
            return Mat22(s / m[0],
                         s / m[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator/(const Mat22& m, value_type s) noexcept {
            return Mat22(m[0] / s,
                         m[1] / s);
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator/(const Mat22& m, const row_type& column) noexcept {
            return noa::math::inverse(m) * column;
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator/(const row_type& row, const Mat22& m) noexcept {
            return row * noa::math::inverse(m);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(const Mat22& m1, const Mat22& m2) noexcept {
            return noa::all(m1[0] == m2[0]) && noa::all(m1[1] == m2[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(const Mat22& m1, const Mat22& m2) noexcept {
            return noa::any(m1[0] != m2[0]) || noa::any(m1[1] != m2[1]);
        }

    public:
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return m_row[0].data(); }
        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return m_row[0].data(); }

        template<typename T, std::enable_if_t<nt::is_real_v<T>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return Mat22<T>(*this);
        }

        [[nodiscard]] NOA_IHD constexpr Mat22 transpose() const noexcept {
            return noa::math::transpose(*this);
        }

        [[nodiscard]] NOA_IHD constexpr Mat22 inverse() const noexcept {
            return noa::math::inverse(*this);
        }

    public: // Support for noa::string::human<Vec>();
        [[nodiscard]] static std::string name() {
            if constexpr (std::is_same_v<value_type, float>)
                return "Float22";
            else
                return "Double22";
        }

    private:
        row_type m_row[ROWS];
    };

    template<typename T> struct nt::proclaim_is_mat22<Mat22<T>> : std::true_type {};

    using Float22 = Mat22<float>;
    using Double22 = Mat22<double>;
}

namespace noa::math {
    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat22<T> element_multiply(Mat22<T> m1, const Mat22<T>& m2) noexcept {
        for (size_t i = 0; i < Mat22<T>::ROWS; ++i)
            m1[i] *= m2[i];
        return m1;
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat22<T> outer_product(const Vec<T, 2>& column, const Vec<T, 2>& row) noexcept {
        return Mat22<T>(column[0] * row[0], column[0] * row[1],
                        column[1] * row[0], column[1] * row[1]);
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat22<T> transpose(const Mat22<T>& m) noexcept {
        return Mat22<T>(m[0][0], m[1][0],
                        m[0][1], m[1][1]);
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr T determinant(const Mat22<T>& m) noexcept {
        return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat22<T> inverse(const Mat22<T>& m) noexcept {
        const auto det = determinant(m);
        NOA_ASSERT(!are_almost_equal(det, T{0})); // non singular
        const auto one_over_determinant = 1 / det;
        return Mat22<T>(+m[1][1] * one_over_determinant,
                        -m[0][1] * one_over_determinant,
                        -m[1][0] * one_over_determinant,
                        +m[0][0] * one_over_determinant);
    }

    template<int32_t ULP = 2, typename T>
    [[nodiscard]] NOA_IHD constexpr bool are_almost_equal(
            const Mat22<T>& m1, const Mat22<T>& m2, T epsilon = 1e-6f) noexcept {
        return noa::all(are_almost_equal<ULP>(m1[0], m2[0], epsilon)) &&
               noa::all(are_almost_equal<ULP>(m1[1], m2[1], epsilon));
    }
}
