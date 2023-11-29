#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa::guts {
    // float: 4*sizeof(float)=4*4=16
    // double: 4*sizeof(double)=4*8=32
    constexpr size_t MAT22_ALIGNMENT = 16;
}

// A few necessary forward declarations:
namespace noa {
    inline namespace types {
        template<typename T>
        class alignas(guts::MAT22_ALIGNMENT) Mat22;
    }

    template<typename T>
    NOA_IHD constexpr Mat22<T> transpose(const Mat22<T>& m) noexcept;

    template<typename T>
    NOA_IHD constexpr Mat22<T> inverse(const Mat22<T>& m) noexcept;
}

namespace noa::inline types {
    /// A 2x2 geometric matrix.
    template<typename Real>
    class alignas(guts::MAT22_ALIGNMENT) Mat22 {
    public: // Type definitions
        static_assert(!std::is_same_v<Real, Half>);

        using value_type = Real;
        using mutable_value_type = value_type;
        using row_type = Vec2<value_type>;

        static constexpr size_t ROWS = 2;
        static constexpr size_t COLS = 2;
        static constexpr size_t COUNT = ROWS * COLS;
        static constexpr size_t SIZE = COUNT;
        static constexpr int64_t SSIZE = ROWS * COLS;

    public:
        row_type row[ROWS]; // uninitialized

    public: // Component accesses
        template<typename I, typename = std::enable_if_t<nt::is_int_v<I>>>
        NOA_HD constexpr row_type& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return row[i];
        }

        template<typename I, typename = std::enable_if_t<nt::is_int_v<I>>>
        NOA_HD constexpr const row_type& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < ROWS);
            return row[i];
        }

    public: // Conversion constructors
        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat22 from_value(U s) noexcept {
            return {row_type::from_values(s, 0),
                    row_type::from_values(0, s)};
        }

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat22 from_diagonal(U s) noexcept {
            return from_value(s);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat22 from_diagonal(const Vec2<U>& diagonal) noexcept {
            return {row_type::from_values(diagonal[0], 0),
                    row_type::from_values(0, diagonal[1])};
        }

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat22 eye(U s) noexcept {
            return from_diagonal(s);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat22 eye(const Vec2<U>& diagonal) noexcept {
            return from_diagonal(diagonal);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat22 from_matrix(const Mat22<U>& m) noexcept {
            return {m[0].template as<value_type>(),
                    m[1].template as<value_type>()};
        }

        template<typename X00, typename X01,
                 typename Y10, typename Y11>
        [[nodiscard]] NOA_HD static constexpr Mat22 from_values(
                X00 x00, X01 x01,
                Y10 y10, Y11 y11
        ) noexcept {
            return {row_type::from_values(x00, x01),
                    row_type::from_values(y10, y11)};
        }

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat22 from_pointer(U* ptr) noexcept {
            return {row_type::from_values(ptr[0], ptr[1]),
                    row_type::from_values(ptr[2], ptr[3])};
        }

        template<typename V0, typename V1>
        [[nodiscard]] NOA_HD static constexpr Mat22 from_rows(
                const Vec2<V0>& r0,
                const Vec2<V1>& r1
        ) noexcept {
            return {r0.template as<value_type>(),
                    r1.template as<value_type>()};
        }

        template<typename V0, typename V1>
        [[nodiscard]] NOA_HD static constexpr Mat22 from_columns(
                const Vec2<V0>& c0,
                const Vec2<V1>& c1
        ) noexcept {
            return {row_type::from_values(c0[0], c1[0]),
                    row_type::from_values(c0[1], c1[1])};
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Mat22<U>>(Mat22<T>{}).
        template<typename U>
        [[nodiscard]] NOA_HD constexpr explicit operator Mat22<U>() const noexcept {
            return Mat22<U>::from_matrix(*this);
        }

    public: // Assignment operators
        NOA_HD constexpr Mat22& operator+=(const Mat22& m) noexcept {
            row[0] += m[0];
            row[1] += m[1];
            return *this;
        }

        NOA_HD constexpr Mat22& operator-=(const Mat22& m) noexcept {
            row[0] -= m[0];
            row[1] -= m[1];
            return *this;
        }

        NOA_HD constexpr Mat22& operator*=(const Mat22& m) noexcept {
            const auto A00 = row[0][0];
            const auto A01 = row[0][1];
            const auto A10 = row[1][0];
            const auto A11 = row[1][1];

            const auto B00 = m[0][0];
            const auto B01 = m[0][1];
            const auto B10 = m[1][0];
            const auto B11 = m[1][1];

            row[0][0] = A00 * B00 + A01 * B10;
            row[0][1] = A00 * B01 + A01 * B11;
            row[1][0] = A10 * B00 + A11 * B10;
            row[1][1] = A10 * B01 + A11 * B11;

            return *this;
        }

        NOA_HD constexpr Mat22& operator/=(const Mat22& m) noexcept {
            *this *= noa::inverse(m);
            return *this;
        }

        NOA_HD constexpr Mat22& operator+=(value_type s) noexcept {
            row[0] += s;
            row[1] += s;
            return *this;
        }

        NOA_HD constexpr Mat22& operator-=(value_type s) noexcept {
            row[0] -= s;
            row[1] -= s;
            return *this;
        }

        NOA_HD constexpr Mat22& operator*=(value_type s) noexcept {
            row[0] *= s;
            row[1] *= s;
            return *this;
        }

        NOA_HD constexpr Mat22& operator/=(value_type s) noexcept {
            row[0] /= s;
            row[1] /= s;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Mat22 operator+(const Mat22& m) noexcept {
            return m;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator-(const Mat22& m) noexcept {
            return {-m[0], -m[1]};
        }

        // -- Binary arithmetic operators --
        [[nodiscard]] friend NOA_HD constexpr Mat22 operator+(const Mat22& m1, const Mat22& m2) noexcept {
            return {m1[0] + m2[0], m1[1] + m2[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator+(value_type s, const Mat22& m) noexcept {
            return {s + m[0], s + m[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator+(const Mat22& m, value_type s) noexcept {
            return {m[0] + s, m[1] + s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator-(const Mat22& m1, const Mat22& m2) noexcept {
            return {m1[0] - m2[0], m1[1] - m2[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator-(value_type s, const Mat22& m) noexcept {
            return {s - m[0], s - m[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator-(const Mat22& m, value_type s) noexcept {
            return {m[0] - s, m[1] - s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator*(Mat22 m1, const Mat22& m2) noexcept {
            m1 *= m2;
            return m1;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator*(value_type s, const Mat22& m) noexcept {
            return {m[0] * s, m[1] * s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator*(const Mat22& m, value_type s) noexcept {
            return {m[0] * s, m[1] * s};
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator*(const Mat22& m, const row_type& column) noexcept {
            return {dot(m[0], column), dot(m[1], column)};
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator*(const row_type& r, const Mat22& m) noexcept {
            return {dot(row_type{m[0][0], m[1][0]}, r), dot(row_type{m[0][1], m[1][1]}, r)};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator/(Mat22 m1, const Mat22& m2) noexcept {
            m1 /= m2;
            return m1;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator/(value_type s, const Mat22& m) noexcept {
            return {s / m[0], s / m[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat22 operator/(const Mat22& m, value_type s) noexcept {
            return {m[0] / s, m[1] / s};
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator/(const Mat22& m, const row_type& column) noexcept {
            return noa::inverse(m) * column;
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator/(const row_type& r, const Mat22& m) noexcept {
            return r * noa::inverse(m);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(const Mat22& m1, const Mat22& m2) noexcept {
            return noa::all(m1[0] == m2[0]) && noa::all(m1[1] == m2[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(const Mat22& m1, const Mat22& m2) noexcept {
            return noa::any(m1[0] != m2[0]) || noa::any(m1[1] != m2[1]);
        }

    public:
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return row[0].data(); }
        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return row[0].data(); }

        template<typename T, nt::enable_if_bool_t<nt::is_real_v<T>> = true>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return Mat22<T>::from_matrix(*this);
        }

        [[nodiscard]] NOA_IHD constexpr Mat22 transpose() const noexcept {
            return noa::transpose(*this);
        }

        [[nodiscard]] NOA_IHD constexpr Mat22 inverse() const noexcept {
            return noa::inverse(*this);
        }

#if defined(NOA_IS_OFFLINE)
    public:
        [[nodiscard]] static std::string name() {
            if constexpr (std::is_same_v<value_type, float>)
                return "Mat22<f32>";
            else
                return "Mat22<f64>";
        }
#endif
    };
}

namespace noa::traits {
    template<typename> struct proclaim_is_mat22 : std::false_type {};
    template<typename T> struct proclaim_is_mat22<Mat22<T>> : std::true_type {};
    template<typename T> using is_mat22 = std::bool_constant<proclaim_is_mat22<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_mat22_v = is_mat22<T>::value;
}

namespace noa {
    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat22<T> ewise_multiply(Mat22<T> m1, const Mat22<T>& m2) noexcept {
        for (size_t i = 0; i < Mat22<T>::ROWS; ++i)
            m1[i] *= m2[i];
        return m1;
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat22<T> outer_product(const Vec<T, 2>& column, const Vec<T, 2>& row) noexcept {
        return {{{column[0] * row[0], column[0] * row[1]},
                 {column[1] * row[0], column[1] * row[1]}}};
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat22<T> transpose(const Mat22<T>& m) noexcept {
        return {{{m[0][0], m[1][0]},
                 {m[0][1], m[1][1]}}};
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr T determinant(const Mat22<T>& m) noexcept {
        return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat22<T> inverse(const Mat22<T>& m) noexcept {
        const auto det = determinant(m);
        NOA_ASSERT(!allclose(det, T{0})); // non singular
        const auto one_over_determinant = 1 / det;
        return {{{+m[1][1] * one_over_determinant, -m[0][1] * one_over_determinant},
                 {-m[1][0] * one_over_determinant, +m[0][0] * one_over_determinant}}};
    }

    template<int32_t ULP = 2, typename T>
    [[nodiscard]] NOA_IHD constexpr bool allclose(
            const Mat22<T>& m1,
            const Mat22<T>& m2,
            T epsilon = 1e-6f
    ) noexcept {
        return all(allclose<ULP>(m1[0], m2[0], epsilon)) &&
               all(allclose<ULP>(m1[1], m2[1], epsilon));
    }
}
