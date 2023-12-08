#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa::inline types {
    /// A 2x3 floating-point matrix.
    template<typename Real>
    class alignas(sizeof(Real) * 2) Mat23 {
    public: // Type definitions
        static_assert(!std::is_same_v<Real, Half>);

        using value_type = Real;
        using mutable_value_type = value_type;
        using row_type = Vec3<value_type>;
        using vec2_type = Vec2<value_type>;
        using vec3_type = Vec3<value_type>;

        static constexpr size_t ROWS = 2;
        static constexpr size_t COLS = 3;
        static constexpr size_t COUNT = ROWS * COLS;
        static constexpr size_t SIZE = COUNT;
        static constexpr int64_t SSIZE = ROWS * COLS;

    public:
        row_type row[ROWS];

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

    public: // Static factory functions
        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat23 from_value(U s) noexcept {
            return {row_type::from_values(s, 0, 0),
                    row_type::from_values(0, s, 0)};
        }

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat23 from_diagonal(U s) noexcept {
            return from_value(s);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat23 from_diagonal(const Vec2<U>& diagonal) noexcept {
            return {row_type::from_values(diagonal[0], 0, 0),
                    row_type::from_values(0, diagonal[1], 0)};
        }

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat23 eye(U s) noexcept {
            return from_diagonal(s);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat23 eye(const Vec2<U>& diagonal) noexcept {
            return from_diagonal(diagonal);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat23 from_matrix(const Mat23<U>& m) noexcept {
            return {m[0].template as<value_type>(),
                    m[1].template as<value_type>()};
        }

        template<typename X00, typename X01, typename X02,
                 typename Y10, typename Y11, typename Y12>
        [[nodiscard]] NOA_HD static constexpr Mat23 from_values(
                X00 x00, X01 x01, X02 x02,
                Y10 y10, Y11 y11, Y12 y12
        ) noexcept {
            return {row_type::from_values(x00, x01, x02),
                    row_type::from_values(y10, y11, y12)};
        }

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat23 from_pointer(U* ptr) noexcept {
            return {row_type::from_values(ptr[0], ptr[1], ptr[2]),
                    row_type::from_values(ptr[3], ptr[4], ptr[5])};
        }

        template<typename V0, typename V1>
        [[nodiscard]] NOA_HD static constexpr Mat23 from_rows(
                const Vec3<V0>& r0,
                const Vec3<V1>& r1
        ) noexcept {
            return {r0.template as<value_type>(),
                    r1.template as<value_type>()};
        }

        template<typename V0, typename V1, typename V2>
        [[nodiscard]] NOA_HD static constexpr Mat23 from_columns(
                const Vec2<V0>& c0,
                const Vec2<V1>& c1,
                const Vec2<V2>& c2
        ) noexcept {
            return {row_type::from_values(c0[0], c1[0], c2[0]),
                    row_type::from_values(c0[1], c1[1], c2[1])};
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Mat23<U>>(Mat23<T>{}).
        template<typename U>
        [[nodiscard]] NOA_HD constexpr explicit operator Mat23<U>() const noexcept {
            return Mat23<U>::from_matrix(*this);
        }

    public: // Assignment operators
        NOA_HD constexpr Mat23& operator+=(const Mat23& m) noexcept {
            row[0] += m[0];
            row[1] += m[1];
            return *this;
        }

        NOA_HD constexpr Mat23& operator-=(const Mat23& m) noexcept {
            row[0] -= m[0];
            row[1] -= m[1];
            return *this;
        }

        NOA_HD constexpr Mat23& operator+=(value_type s) noexcept {
            row[0] += s;
            row[1] += s;
            return *this;
        }

        NOA_HD constexpr Mat23& operator-=(value_type s) noexcept {
            row[0] -= s;
            row[1] -= s;
            return *this;
        }

        NOA_HD constexpr Mat23& operator*=(value_type s) noexcept {
            row[0] *= s;
            row[1] *= s;
            return *this;
        }

        NOA_HD constexpr Mat23& operator/=(value_type s) noexcept {
            row[0] /= s;
            row[1] /= s;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Mat23 operator+(const Mat23& m) noexcept {
            return m;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator-(const Mat23& m) noexcept {
            return {-m[0], -m[1]};
        }

        // -- Binary arithmetic operators --
        [[nodiscard]] friend NOA_HD constexpr Mat23 operator+(const Mat23& m1, const Mat23& m2) noexcept {
            return {m1[0] + m2[0], m1[1] + m2[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator+(value_type s, const Mat23& m) noexcept {
            return {s + m[0], s + m[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator+(const Mat23& m, value_type s) noexcept {
            return {m[0] + s, m[1] + s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator-(const Mat23& m1, const Mat23& m2) noexcept {
            return {m1[0] - m2[0], m1[1] - m2[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator-(value_type s, const Mat23& m) noexcept {
            return {s - m[0], s - m[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator-(const Mat23& m, value_type s) noexcept {
            return {m[0] - s, m[1] - s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator*(value_type s, const Mat23& m) noexcept {
            return {m[0] * s, m[1] * s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator*(const Mat23& m, value_type s) noexcept {
            return {m[0] * s, m[1] * s};
        }

        [[nodiscard]] friend NOA_HD constexpr vec2_type operator*(const Mat23& m, const vec3_type& column) noexcept {
            return {dot(m[0], column), dot(m[1], column)};
        }

        [[nodiscard]] friend NOA_HD constexpr vec3_type operator*(const vec2_type& r, const Mat23& m) noexcept {
            return {dot(vec2_type{m[0][0], m[1][0]}, r),
                    dot(vec2_type{m[0][1], m[1][1]}, r),
                    dot(vec2_type{m[0][2], m[1][2]}, r)};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator/(value_type s, const Mat23& m) noexcept {
            return {s / m[0], s / m[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat23 operator/(const Mat23& m, value_type s) noexcept {
            return {m[0] / s, m[1] / s};
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(const Mat23& m1, const Mat23& m2) noexcept {
            return noa::all(m1[0] == m2[0]) && noa::all(m1[1] == m2[1]);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(const Mat23& m1, const Mat23& m2) noexcept {
            return noa::any(m1[0] != m2[0]) || noa::any(m1[1] != m2[1]);
        }

    public:
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return row[0].data(); }
        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return row[0].data(); }

        template<typename T, std::enable_if_t<nt::is_real_v<T>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return Mat23<T>::from_matrix(*this);
        }

#if defined(NOA_IS_CPU_CODE)
    public:
        [[nodiscard]] static std::string name() {
            if constexpr (std::is_same_v<value_type, float>)
                return "Mat23<f32>";
            else
                return "Mat23<f64>";
        }
#endif
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_mat23<Mat23<T>> : std::true_type {};
}

namespace noa {
    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat23<T> ewise_multiply(Mat23<T> m1, const Mat23<T>& m2) noexcept {
        for (size_t i = 0; i < Mat23<T>::ROWS; ++i)
            m1[i] *= m2[i];
        return m1;
    }

    template<int32_t ULP = 2, typename T>
    [[nodiscard]] NOA_IHD constexpr bool allclose(
            const Mat23<T>& m1,
            const Mat23<T>& m2,
            T epsilon = 1e-6f
    ) noexcept {
        return noa::all(allclose<ULP>(m1[0], m2[0], epsilon)) &&
               noa::all(allclose<ULP>(m1[1], m2[1], epsilon));
    }
}
