#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Vec.hpp"

// A few necessary forward declarations:
namespace noa {
    inline namespace types {
        template<typename T>
        class Mat44;
    }

    template<typename T>
    NOA_IHD constexpr Mat44<T> transpose(const Mat44<T>& m) noexcept;

    template<typename T>
    NOA_IHD constexpr Mat44<T> inverse(const Mat44<T>& m) noexcept;
}

namespace noa::inline types {
    /// A 4x4 floating-point matrix.
    template<typename Real>
    class Mat44 {
    public: // Type definitions
        static_assert(!std::is_same_v<Real, Half>);

        using value_type = Real;
        using mutable_value_type = value_type;
        using row_type = Vec4<value_type>;

        static constexpr size_t ROWS = 4;
        static constexpr size_t COLS = 4;
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
        [[nodiscard]] NOA_HD static constexpr Mat44 from_value(U s) noexcept {
            return {row_type::from_values(s, 0, 0, 0),
                    row_type::from_values(0, s, 0, 0),
                    row_type::from_values(0, 0, s, 0),
                    row_type::from_values(0, 0, 0, s)};
        }

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat44 from_diagonal(U s) noexcept {
            return from_value(s);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat44 from_diagonal(const Vec4<U>& diagonal) noexcept {
            return {row_type::from_values(diagonal[0], 0, 0, 0),
                    row_type::from_values(0, diagonal[1], 0, 0),
                    row_type::from_values(0, 0, diagonal[2], 0),
                    row_type::from_values(0, 0, 0, diagonal[3])};
        }

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat44 eye(U s) noexcept {
            return from_diagonal(s);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat44 eye(const Vec4<U>& diagonal) noexcept {
            return from_diagonal(diagonal);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr Mat44 from_matrix(const Mat44<U>& m) noexcept {
            return {m[0].template as<value_type>(),
                    m[1].template as<value_type>(),
                    m[2].template as<value_type>(),
                    m[3].template as<value_type>()};
        }

        template<typename X00, typename X01, typename X02, typename X03,
                 typename Y10, typename Y11, typename Y12, typename Y13,
                 typename Z20, typename Z21, typename Z22, typename Z23,
                 typename W30, typename W31, typename W32, typename W33>
        [[nodiscard]] NOA_HD static constexpr Mat44 from_values(
                X00 x00, X01 x01, X02 x02, X03 x03,
                Y10 y10, Y11 y11, Y12 y12, Y13 y13,
                Z20 z20, Z21 z21, Z22 z22, Z23 z23,
                W30 w30, W31 w31, W32 w32, W33 w33
        ) noexcept {
            return {row_type::from_values(x00, x01, x02, x03),
                    row_type::from_values(y10, y11, y12, y13),
                    row_type::from_values(z20, z21, z22, z23),
                    row_type::from_values(w30, w31, w32, w33)};
        }

        template<typename U, typename = std::enable_if_t<nt::is_scalar_v<U>>>
        [[nodiscard]] NOA_HD static constexpr Mat44 from_pointer(U* ptr) noexcept {
            return {row_type::from_values(ptr[0], ptr[1], ptr[2], ptr[3]),
                    row_type::from_values(ptr[4], ptr[5], ptr[6], ptr[7]),
                    row_type::from_values(ptr[8], ptr[9], ptr[10], ptr[11]),
                    row_type::from_values(ptr[12], ptr[13], ptr[14], ptr[15])};
        }

        template<typename V0, typename V1, typename V2, typename V3>
        [[nodiscard]] NOA_HD static constexpr Mat44 from_rows(
                const Vec4<V0>& r0,
                const Vec4<V1>& r1,
                const Vec4<V2>& r2,
                const Vec4<V3>& r3
        ) noexcept {
            return {r0.template as<value_type>(),
                    r1.template as<value_type>(),
                    r2.template as<value_type>(),
                    r3.template as<value_type>()};
        }

        template<typename V0, typename V1, typename V2, typename V3>
        [[nodiscard]] NOA_HD static constexpr Mat44 from_columns(
                const Vec4<V0>& c0,
                const Vec4<V1>& c1,
                const Vec4<V2>& c2,
                const Vec4<V3>& c3
        ) noexcept {
            return {row_type::from_values(c0[0], c1[0], c2[0], c3[0]),
                    row_type::from_values(c0[1], c1[1], c2[1], c3[1]),
                    row_type::from_values(c0[2], c1[2], c2[2], c3[2]),
                    row_type::from_values(c0[3], c1[3], c2[3], c3[3])};
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Mat44<U>>(Mat44<T>{}).
        template<typename U>
        [[nodiscard]] NOA_HD constexpr explicit operator Mat44<U>() const noexcept {
            return Mat44<U>::from_matrix(*this);
        }

    public: // Assignment operators
        NOA_HD constexpr Mat44& operator+=(const Mat44& m) noexcept {
            row[0] += m[0];
            row[1] += m[1];
            row[2] += m[2];
            row[3] += m[3];
            return *this;
        }

        NOA_HD constexpr Mat44& operator-=(const Mat44& m) noexcept {
            row[0] -= m[0];
            row[1] -= m[1];
            row[2] -= m[2];
            row[3] -= m[3];
            return *this;
        }

        NOA_HD constexpr Mat44& operator*=(const Mat44& m) noexcept {
            const row_type A0 = row[0];
            const row_type A1 = row[1];
            const row_type A2 = row[2];
            const row_type A3 = row[3];

            const row_type B0 = m[0];
            const row_type B1 = m[1];
            const row_type B2 = m[2];
            const row_type B3 = m[3];

            row[0] = A0[0] * B0 + A0[1] * B1 + A0[2] * B2 + A0[3] * B3;
            row[1] = A1[0] * B0 + A1[1] * B1 + A1[2] * B2 + A1[3] * B3;
            row[2] = A2[0] * B0 + A2[1] * B1 + A2[2] * B2 + A2[3] * B3;
            row[3] = A3[0] * B0 + A3[1] * B1 + A3[2] * B2 + A3[3] * B3;
            return *this;
        }

        NOA_HD constexpr Mat44& operator/=(const Mat44& m) noexcept {
            *this *= noa::inverse(m);
            return *this;
        }

        NOA_HD constexpr Mat44& operator+=(value_type s) noexcept {
            row[0] += s;
            row[1] += s;
            row[2] += s;
            row[3] += s;
            return *this;
        }

        NOA_HD constexpr Mat44& operator-=(value_type s) noexcept {
            row[0] -= s;
            row[1] -= s;
            row[2] -= s;
            row[3] -= s;
            return *this;
        }

        NOA_HD constexpr Mat44& operator*=(value_type s) noexcept {
            row[0] *= s;
            row[1] *= s;
            row[2] *= s;
            row[3] *= s;
            return *this;
        }

        NOA_HD constexpr Mat44& operator/=(value_type s) noexcept {
            row[0] /= s;
            row[1] /= s;
            row[2] /= s;
            row[3] /= s;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Mat44 operator+(const Mat44& m) noexcept {
            return m;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator-(const Mat44& m) noexcept {
            return {-m[0], -m[1], -m[2], -m[3]};
        }

        // -- Binary arithmetic operators --
        [[nodiscard]] friend NOA_HD constexpr Mat44 operator+(const Mat44& m1, const Mat44& m2) noexcept {
            return {m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2], m1[3] + m2[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator+(value_type s, const Mat44& m) noexcept {
            return {s + m[0], s + m[1], s + m[2], s + m[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator+(const Mat44& m, value_type s) noexcept {
            return {m[0] + s, m[1] + s, m[2] + s, m[3] + s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator-(const Mat44& m1, const Mat44& m2) noexcept {
            return {m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2], m1[3] - m2[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator-(value_type s, const Mat44& m) noexcept {
            return {s - m[0], s - m[1], s - m[2], s - m[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator-(const Mat44& m, value_type s) noexcept {
            return {m[0] - s, m[1] - s, m[2] - s, m[3] - s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator*(Mat44 m1, const Mat44& m2) noexcept {
            m1 *= m2;
            return m1;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator*(value_type s, const Mat44& m) noexcept {
            return {m[0] * s, m[1] * s, m[2] * s, m[3] * s};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator*(const Mat44& m, value_type s) noexcept {
            return {m[0] * s, m[1] * s, m[2] * s, m[3] * s};
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator*(const Mat44& m, const row_type& c) noexcept {
            return {dot(m[0], c),
                    dot(m[1], c),
                    dot(m[2], c),
                    dot(m[3], c)};
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator*(const row_type& r, const Mat44& m) noexcept {
            return {dot(row_type{m[0][0], m[1][0], m[2][0], m[3][0]}, r),
                    dot(row_type{m[0][1], m[1][1], m[2][1], m[3][1]}, r),
                    dot(row_type{m[0][2], m[1][2], m[2][2], m[3][2]}, r),
                    dot(row_type{m[0][3], m[1][3], m[2][3], m[3][3]}, r)};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator/(Mat44 m1, const Mat44& m2) noexcept {
            m1 /= m2;
            return m1;
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator/(value_type s, const Mat44& m) noexcept {
            return {s / m[0], s / m[1], s / m[2], s / m[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Mat44 operator/(const Mat44& m, value_type s) noexcept {
            return {m[0] / s, m[1] / s, m[2] / s, m[3] / s};
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator/(const Mat44& m, const row_type& c) noexcept {
            return noa::inverse(m) * c;
        }

        [[nodiscard]] friend NOA_HD constexpr row_type operator/(const row_type& r, const Mat44& m) noexcept {
            return r * noa::inverse(m);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(const Mat44& m1, const Mat44& m2) noexcept {
            return noa::all(m1[0] == m2[0]) && noa::all(m1[1] == m2[1]) &&
                   noa::all(m1[2] == m2[2]) && noa::all(m1[3] == m2[3]);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(const Mat44& m1, const Mat44& m2) noexcept {
            return noa::any(m1[0] != m2[0]) || noa::any(m1[1] != m2[1]) ||
                   noa::any(m1[2] != m2[2]) || noa::any(m1[3] != m2[3]);
        }

    public:
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return row[0].data(); }
        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return row[0].data(); }

        template<typename T, std::enable_if_t<nt::is_real_v<T>, bool> = true>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return Mat44<T>::from_matrix(*this);
        }

        [[nodiscard]] NOA_IHD constexpr Mat44 transpose() const noexcept {
            return noa::transpose(*this);
        }

        [[nodiscard]] NOA_IHD constexpr Mat44 inverse() const noexcept {
            return noa::inverse(*this);
        }

#if defined(NOA_IS_CPU_CODE)
    public:
        [[nodiscard]] static std::string name() {
            if constexpr (std::is_same_v<value_type, float>)
                return "Mat44<f32>";
            else
                return "Mat44<f64>";
        }
#endif
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_mat44<Mat44<T>> : std::true_type {};
}

namespace noa {
    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat44<T> ewise_multiply(Mat44<T> m1, const Mat44<T>& m2) noexcept {
        for (size_t i = 0; i < Mat44<T>::ROWS; ++i)
            m1[i] *= m2[i];
        return m1;
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat44<T> outer_product(const Vec4<T>& column, const Vec4<T>& row) noexcept {
        return {{{column[0] * row[0], column[0] * row[1], column[0] * row[2], column[0] * row[3]},
                 {column[1] * row[0], column[1] * row[1], column[1] * row[2], column[1] * row[3]},
                 {column[2] * row[0], column[2] * row[1], column[2] * row[2], column[2] * row[3]},
                 {column[3] * row[0], column[3] * row[1], column[3] * row[2], column[3] * row[3]}}};
    }

    template<typename T>
    [[nodiscard]] NOA_IHD constexpr Mat44<T> transpose(const Mat44<T>& m) noexcept {
        return {{{m[0][0], m[1][0], m[2][0], m[3][0]},
                 {m[0][1], m[1][1], m[2][1], m[3][1]},
                 {m[0][2], m[1][2], m[2][2], m[3][2]},
                 {m[0][3], m[1][3], m[2][3], m[3][3]}}};
    }

    template<typename T>
    [[nodiscard]] NOA_HD constexpr T determinant(const Mat44<T>& m) noexcept {
        const auto s00 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
        const auto s01 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
        const auto s02 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
        const auto s03 = m[0][2] * m[3][3] - m[0][3] * m[3][2];
        const auto s04 = m[0][2] * m[2][3] - m[0][3] * m[2][2];
        const auto s05 = m[0][2] * m[1][3] - m[0][3] * m[1][2];

        Vec4<T> c{+(m[1][1] * s00 - m[2][1] * s01 + m[3][1] * s02),
                  -(m[0][1] * s00 - m[2][1] * s03 + m[3][1] * s04),
                  +(m[0][1] * s01 - m[1][1] * s03 + m[3][1] * s05),
                  -(m[0][1] * s02 - m[1][1] * s04 + m[2][1] * s05)};

        return m[0][0] * c[0] + m[1][0] * c[1] +
               m[2][0] * c[2] + m[3][0] * c[3];
    }

    template<typename T>
    [[nodiscard]] NOA_HD constexpr Mat44<T> inverse(const Mat44<T>& m) noexcept {
        // From https://stackoverflow.com/a/44446912 and https://github.com/willnode/N-Matrix-Programmer
        const auto A2323 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
        const auto A1323 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
        const auto A1223 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
        const auto A0323 = m[2][0] * m[3][3] - m[2][3] * m[3][0];
        const auto A0223 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
        const auto A0123 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
        const auto A2313 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
        const auto A1313 = m[1][1] * m[3][3] - m[1][3] * m[3][1];
        const auto A1213 = m[1][1] * m[3][2] - m[1][2] * m[3][1];
        const auto A2312 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
        const auto A1312 = m[1][1] * m[2][3] - m[1][3] * m[2][1];
        const auto A1212 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        const auto A0313 = m[1][0] * m[3][3] - m[1][3] * m[3][0];
        const auto A0213 = m[1][0] * m[3][2] - m[1][2] * m[3][0];
        const auto A0312 = m[1][0] * m[2][3] - m[1][3] * m[2][0];
        const auto A0212 = m[1][0] * m[2][2] - m[1][2] * m[2][0];
        const auto A0113 = m[1][0] * m[3][1] - m[1][1] * m[3][0];
        const auto A0112 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

        auto det = m[0][0] * (m[1][1] * A2323 - m[1][2] * A1323 + m[1][3] * A1223) -
                   m[0][1] * (m[1][0] * A2323 - m[1][2] * A0323 + m[1][3] * A0223) +
                   m[0][2] * (m[1][0] * A1323 - m[1][1] * A0323 + m[1][3] * A0123) -
                   m[0][3] * (m[1][0] * A1223 - m[1][1] * A0223 + m[1][2] * A0123);
        NOA_ASSERT(!allclose(det, T{0})); // non singular
        det = 1 / det;

        return Mat44<T>::from_values(
                det * +(m[1][1] * A2323 - m[1][2] * A1323 + m[1][3] * A1223),
                det * -(m[0][1] * A2323 - m[0][2] * A1323 + m[0][3] * A1223),
                det * +(m[0][1] * A2313 - m[0][2] * A1313 + m[0][3] * A1213),
                det * -(m[0][1] * A2312 - m[0][2] * A1312 + m[0][3] * A1212),
                det * -(m[1][0] * A2323 - m[1][2] * A0323 + m[1][3] * A0223),
                det * +(m[0][0] * A2323 - m[0][2] * A0323 + m[0][3] * A0223),
                det * -(m[0][0] * A2313 - m[0][2] * A0313 + m[0][3] * A0213),
                det * +(m[0][0] * A2312 - m[0][2] * A0312 + m[0][3] * A0212),
                det * +(m[1][0] * A1323 - m[1][1] * A0323 + m[1][3] * A0123),
                det * -(m[0][0] * A1323 - m[0][1] * A0323 + m[0][3] * A0123),
                det * +(m[0][0] * A1313 - m[0][1] * A0313 + m[0][3] * A0113),
                det * -(m[0][0] * A1312 - m[0][1] * A0312 + m[0][3] * A0112),
                det * -(m[1][0] * A1223 - m[1][1] * A0223 + m[1][2] * A0123),
                det * +(m[0][0] * A1223 - m[0][1] * A0223 + m[0][2] * A0123),
                det * -(m[0][0] * A1213 - m[0][1] * A0213 + m[0][2] * A0113),
                det * +(m[0][0] * A1212 - m[0][1] * A0212 + m[0][2] * A0112));
    }

    template<int32_t ULP = 2, typename T>
    [[nodiscard]] NOA_IHD constexpr bool allclose(
            const Mat44<T>& m1,
            const Mat44<T>& m2,
            T epsilon = 1e-6f
    ) noexcept {
        return noa::all(allclose<ULP>(m1[0], m2[0], epsilon)) &&
               noa::all(allclose<ULP>(m1[1], m2[1], epsilon)) &&
               noa::all(allclose<ULP>(m1[2], m2[2], epsilon)) &&
               noa::all(allclose<ULP>(m1[3], m2[3], epsilon));
    }
}
