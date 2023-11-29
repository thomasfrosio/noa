#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa::inline types {
    /// A 3x4 floating-point matrix.
    template<typename Real>
    class Mat34 {
    public: // Type definitions
        static_assert(!std::is_same_v<Real, Half>);

        using value_type = Real;
        using mutable_value_type = value_type;
        using row_type = Vec4<value_type>;
        using vec3_type = Vec3<value_type>;
        using vec4_type = Vec4<value_type>;

        static constexpr size_t ROWS = 3;
        static constexpr size_t COLS = 4;
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

    public: // Static factory functions
        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat34 from_value(U s) noexcept {
            return {row_type::from_values(s, 0, 0, 0),
                    row_type::from_values(0, s, 0, 0),
                    row_type::from_values(0, 0, s, 0)};
        }

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat34 from_diagonal(U s) noexcept {
            return from_value(s);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat34 from_diagonal(const Vec3<U>& diagonal) noexcept {
            return {row_type::from_values(diagonal[0], 0, 0, 0),
                    row_type::from_values(0, diagonal[1], 0, 0),
                    row_type::from_values(0, 0, diagonal[2], 0)};
        }

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat34 eye(U s) noexcept {
            return from_diagonal(s);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat34 eye(const Vec3<U>& diagonal) noexcept {
            return from_diagonal(diagonal);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat34 from_matrix(const Mat34<U>& m) noexcept {
            return {m[0].template as<value_type>(),
                    m[1].template as<value_type>(),
                    m[2].template as<value_type>()};
        }

        template<typename X00, typename X01, typename X02, typename X03,
                 typename Y10, typename Y11, typename Y12, typename Y13,
                 typename Z20, typename Z21, typename Z22, typename Z23>
        [[nodiscard]] NOA_HD static constexpr Mat34 from_values(
                X00 x00, X01 x01, X02 x02, X03 x03,
                Y10 y10, Y11 y11, Y12 y12, Y13 y13,
                Z20 z20, Z21 z21, Z22 z22, Z23 z23
        ) noexcept {
            return {row_type::from_values(x00, x01, x02, x03),
                    row_type::from_values(y10, y11, y12, y13),
                    row_type::from_values(z20, z21, z22, z23)};
        }

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat34 from_pointer(U* ptr) noexcept {
            return {row_type::from_values(ptr[0], ptr[1], ptr[2], ptr[3]),
                    row_type::from_values(ptr[4], ptr[5], ptr[6], ptr[7]),
                    row_type::from_values(ptr[8], ptr[9], ptr[10], ptr[11])};
        }

        template<typename V0, typename V1, typename V2>
        [[nodiscard]] NOA_HD static constexpr Mat34 from_rows(
                const Vec4<V0>& r0,
                const Vec4<V1>& r1,
                const Vec4<V2>& r2
        ) noexcept {
            return {r0.template as<value_type>(),
                    r1.template as<value_type>(),
                    r2.template as<value_type>()};
        }

        template<typename V0, typename V1, typename V2, typename V3>
        [[nodiscard]] NOA_HD static constexpr Mat34 from_columns(
                const Vec3<V0>& c0,
                const Vec3<V1>& c1,
                const Vec3<V2>& c2,
                const Vec3<V3>& c3
        ) noexcept {
            return {row_type::from_values(c0[0], c1[0], c2[0], c3[0]),
                    row_type::from_values(c0[1], c1[1], c2[1], c3[1]),
                    row_type::from_values(c0[2], c1[2], c2[2], c3[2])};
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Mat34<U>>(Mat34<T>{}).
        template<typename U>
        [[nodiscard]] NOA_HD constexpr explicit operator Mat34<U>() const noexcept {
            return Mat34<U>::from_matrix(*this);
        }

    public: // Assignment operators
        NOA_HD constexpr Mat34& operator+=(const Mat34& m) noexcept {
            row[0] += m[0];
            row[1] += m[1];
            row[2] += m[2];
            return *this;
        }

        NOA_HD constexpr Mat34& operator-=(const Mat34& m) noexcept {
            row[0] -= m[0];
            row[1] -= m[1];
            row[2] -= m[2];
            return *this;
        }

        NOA_HD constexpr Mat34& operator+=(value_type s) noexcept {
            row[0] += s;
            row[1] += s;
            row[2] += s;
            return *this;
        }

        NOA_HD constexpr Mat34& operator-=(value_type s) noexcept {
            row[0] -= s;
            row[1] -= s;
            row[2] -= s;
            return *this;
        }

        NOA_HD constexpr Mat34& operator*=(value_type s) noexcept {
            row[0] *= s;
            row[1] *= s;
            row[2] *= s;
            return *this;
        }

        NOA_HD constexpr Mat34& operator/=(value_type s) noexcept {
            row[0] /= s;
            row[1] /= s;
            row[2] /= s;
            return *this;
        }

    public: // Assignment operators
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Mat34 operator+(const Mat34& m) noexcept {
            return m;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator-(const Mat34& m) noexcept {
            return {-m[0], -m[1], -m[2]};
        }

        // -- Binary arithmetic operators --
        [[nodiscard]] friend NOA_HD constexpr Mat34 operator+(const Mat34& m1, const Mat34& m2) noexcept {
            return {m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator+(value_type s, const Mat34& m) noexcept {
            return {s + m[0], s + m[1], s + m[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator+(const Mat34& m, value_type s) noexcept {
            return {m[0] + s, m[1] + s, m[2] + s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator-(const Mat34& m1, const Mat34& m2) noexcept {
            return {m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator-(value_type s, const Mat34& m) noexcept {
            return {s - m[0], s - m[1], s - m[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator-(const Mat34& m, value_type s) noexcept {
            return {m[0] - s, m[1] - s, m[2] - s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator*(value_type s, const Mat34& m) noexcept {
            return {m[0] * s, m[1] * s, m[2] * s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator*(const Mat34& m, value_type s) noexcept {
            return {m[0] * s, m[1] * s, m[2] * s};
        }

        [[nodiscard]] friend NOA_HD constexpr vec3_type operator*(const Mat34& m, const vec4_type& c) noexcept {
            return {dot(m[0], c),
                    dot(m[1], c),
                    dot(m[2], c)};
        }

        [[nodiscard]] friend NOA_HD constexpr vec4_type operator*(const vec3_type& r, const Mat34& m) noexcept {
            return {dot(vec3_type{m[0][0], m[1][0], m[2][0]}, r),
                    dot(vec3_type{m[0][1], m[1][1], m[2][1]}, r),
                    dot(vec3_type{m[0][2], m[1][2], m[2][2]}, r),
                    dot(vec3_type{m[0][3], m[1][3], m[2][3]}, r)};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator/(value_type s, const Mat34& m) noexcept {
            return {s / m[0], s / m[1], s / m[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat34 operator/(const Mat34& m, value_type s) noexcept {
            return {m[0] / s, m[1] / s, m[2] / s};
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(const Mat34& m1, const Mat34& m2) noexcept {
            return noa::all(m1[0] == m2[0]) && noa::all(m1[1] == m2[1]) && noa::all(m1[2] == m2[2]);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(const Mat34& m1, const Mat34& m2) noexcept {
            return noa::any(m1[0] != m2[0]) || noa::any(m1[1] != m2[1]) || noa::any(m1[2] != m2[2]);
        }

    public:
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return row[0].data(); }
        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return row[0].data(); }

        template<typename T, std::enable_if_t<nt::is_real_v<T>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return Mat34<T>::from_matrix(*this);
        }

#if defined(NOA_IS_CPU_CODE)
    public:
        [[nodiscard]] static std::string name() {
            if constexpr (std::is_same_v<value_type, float>)
                return "Mat34<f32>";
            else
                return "Mat34<f64>";
        }
#endif
    };
}

namespace noa::traits {
    template<typename> struct proclaim_is_mat34 : std::false_type {};
    template<typename T> struct proclaim_is_mat34<Mat34<T>> : std::true_type {};
    template<typename T> using is_mat34 = std::bool_constant<proclaim_is_mat34<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_mat34_v = is_mat34<T>::value;
}

namespace noa {
    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat34<T> ewise_multiply(Mat34<T> m1, const Mat34<T>& m2) noexcept {
        for (size_t i = 0; i < Mat34<T>::ROWS; ++i)
            m1[i] *= m2[i];
        return m1;
    }

    template<int32_t ULP = 2, typename T>
    [[nodiscard]] NOA_IHD constexpr bool allclose(
            const Mat34<T>& m1,
            const Mat34<T>& m2,
            T epsilon = 1e-6f
    ) noexcept {
        return noa::all(allclose<ULP>(m1[0], m2[0], epsilon)) &&
               noa::all(allclose<ULP>(m1[1], m2[1], epsilon)) &&
               noa::all(allclose<ULP>(m1[2], m2[2], epsilon));
    }
}
