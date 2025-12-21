#pragma once

#include "noa/runtime/core/Config.hpp"
#include "noa/runtime/core/Math.hpp"
#include "noa/xform/Traits.hpp"
#include "noa/runtime/core/Tuple.hpp"
#include "noa/runtime/core/Vec.hpp"

namespace noa::xform {
    template<typename T, usize R, usize C>
    class alignas(16) Mat;

    template<typename T, usize R, usize C>
    NOA_HD constexpr auto transpose(const Mat<T, R, C>& m) noexcept -> Mat<T, C, R> {
        return [&]<usize... I>(std::index_sequence<I...>) {
            return Mat<T, C, R>::from_columns(m[I]...);
        }(std::make_index_sequence<R>{});
    }

    template<typename T, usize R0, usize C0, usize R1, usize C1> requires (C0 == R1)
    NOA_HD constexpr auto matmul(const Mat<T, R0, C0>& m0, const Mat<T, R1, C1>& m1) noexcept -> Mat<T, R0, C1> {
        using output_t = Mat<T, R0, C1>;
        output_t out{};
        for (usize r{}; r < R0; ++r)
            for (usize c{}; c < C0; ++c)
                out[r] += m0[r][c] * m1[c];
        return out;
    }

    template<typename T, usize R, usize C> requires (R == C)
    [[nodiscard]] NOA_HD constexpr auto determinant(const Mat<T, R, C>& m) noexcept -> T {
        if constexpr (R == 2) {
            return m[0][0] * m[1][1] - m[0][1] * m[1][0];

        } else if constexpr (R == 3) {
            return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                   m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                   m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        } else if constexpr (R == 4) {
            const auto s00 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
            const auto s01 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
            const auto s02 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
            const auto s03 = m[0][2] * m[3][3] - m[0][3] * m[3][2];
            const auto s04 = m[0][2] * m[2][3] - m[0][3] * m[2][2];
            const auto s05 = m[0][2] * m[1][3] - m[0][3] * m[1][2];

            Vec c{+(m[1][1] * s00 - m[2][1] * s01 + m[3][1] * s02),
                  -(m[0][1] * s00 - m[2][1] * s03 + m[3][1] * s04),
                  +(m[0][1] * s01 - m[1][1] * s03 + m[3][1] * s05),
                  -(m[0][1] * s02 - m[1][1] * s04 + m[2][1] * s05)};

            return m[0][0] * c[0] + m[1][0] * c[1] +
                   m[2][0] * c[2] + m[3][0] * c[3];

        } else {
            static_assert(nt::always_false<T>);
        }
    }

    template<typename T, usize R, usize C> requires (R == C)
    NOA_HD constexpr auto inverse(const Mat<T, R, C>& m) noexcept -> Mat<T, R, C> {
        if constexpr (R == 2) {
            const auto det = determinant(m);
            NOA_ASSERT(not allclose(det, T{})); // non singular
            const auto one_over_determinant = 1 / det;
            return Mat<T, 2, 2>::from_values(
                +m[1][1] * one_over_determinant, -m[0][1] * one_over_determinant,
                -m[1][0] * one_over_determinant, +m[0][0] * one_over_determinant
            );
        } else if constexpr (R == 3) {
            const auto det = determinant(m);
            NOA_ASSERT(not allclose(det, T{})); // non singular
            const auto one_over_determinant = 1 / det;
            return Mat<T, 3, 3>::from_values(
                +(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * one_over_determinant,
                -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * one_over_determinant,
                +(m[0][1] * m[1][2] - m[0][2] * m[1][1]) * one_over_determinant,
                -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * one_over_determinant,
                +(m[0][0] * m[2][2] - m[0][2] * m[2][0]) * one_over_determinant,
                -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * one_over_determinant,
                +(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * one_over_determinant,
                -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * one_over_determinant,
                +(m[0][0] * m[1][1] - m[0][1] * m[1][0]) * one_over_determinant
            );
        } else if constexpr (R == 4) {
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

            auto det =
                    m[0][0] * (m[1][1] * A2323 - m[1][2] * A1323 + m[1][3] * A1223) -
                    m[0][1] * (m[1][0] * A2323 - m[1][2] * A0323 + m[1][3] * A0223) +
                    m[0][2] * (m[1][0] * A1323 - m[1][1] * A0323 + m[1][3] * A0123) -
                    m[0][3] * (m[1][0] * A1223 - m[1][1] * A0223 + m[1][2] * A0123);
            NOA_ASSERT(not allclose(det, T{})); // non singular
            det = 1 / det;

            return Mat<T, 4, 4>::from_values(
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
                det * +(m[0][0] * A1212 - m[0][1] * A0212 + m[0][2] * A0112)
            );
        } else {
            static_assert(nt::always_false<T>);
        }
    }

    template<typename T, usize R, usize C>
    [[nodiscard]] NOA_HD constexpr auto ewise_multiply(
        Mat<T, R, C> m1,
        const Mat<T, R, C>& m2
    ) noexcept -> Mat<T, R, C> {
        for (usize i{}; i < R; ++i)
            m1[i] *= m2[i];
        return m1;
    }

    template<typename T, usize C, usize R, usize A0, usize A1>
    [[nodiscard]] NOA_HD constexpr auto outer_product(
        const Vec<T, R, A0>& column,
        const Vec<T, C, A1>& row
    ) noexcept -> Mat<T, R, C> {
        Mat<T, R, C> out;
        for (usize i{}; i < R; ++i)
            out[i] = column[i] * row;
        return out;
    }

    template<i32 ULP = 2, typename T, usize C, usize R>
    [[nodiscard]] NOA_HD constexpr auto allclose(
        const Mat<T, R, C>& m1,
        const Mat<T, R, C>& m2,
        std::type_identity_t<T> epsilon = static_cast<T>(1e-6)
    ) noexcept -> bool {
        for (usize r{}; r < R; ++r)
            for (usize c{}; c < C; ++c)
                if (not allclose<ULP>(m1[r][c], m2[r][c], epsilon))
                    return false;
        return true;
    }
}

namespace noa::xform {
    /// Aggregate type representing an RxC geometric (row-major) matrix.
    template<typename T, usize R, usize C>
    class alignas(16) Mat {
    public:
        static_assert(nt::real<T> and R > 0 and C > 0);
        using value_type = T;
        using mutable_value_type = value_type;
        using row_type = Vec<value_type, C>;
        using column_type = Vec<value_type, R>;

        static constexpr usize ROWS = R;
        static constexpr usize COLS = C;
        static constexpr usize DIAG = std::min(ROWS, COLS);
        static constexpr usize SIZE = ROWS;
        static constexpr isize SSIZE = ROWS;

    public:
        row_type rows[ROWS];

    public: // Static factory functions
        [[nodiscard]] NOA_HD static constexpr auto from_value(nt::scalar auto s) noexcept -> Mat {
            Mat mat{};
            for (usize r{}; r < ROWS; ++r)
                for (usize c{}; c < COLS; ++c)
                    if (r == c)
                        mat[r][c] = static_cast<value_type>(s);
            return mat;
        }

        [[nodiscard]] NOA_HD static constexpr auto from_diagonal(nt::scalar auto s) noexcept -> Mat {
            return from_value(s);
        }

        template<typename U, usize A>
        [[nodiscard]] NOA_HD static constexpr auto from_diagonal(const Vec<U, DIAG, A>& diagonal) noexcept -> Mat {
            Mat mat{};
            for (usize r{}; r < ROWS; ++r)
                for (usize c{}; c < COLS; ++c)
                    if (r == c)
                        mat[r][c] = static_cast<value_type>(diagonal[r]);
            return mat;
        }

        [[nodiscard]] NOA_HD static constexpr auto eye(nt::scalar auto s) noexcept -> Mat {
            return from_diagonal(s);
        }

        template<typename U, usize A>
        [[nodiscard]] NOA_HD static constexpr auto eye(const Vec<U, DIAG, A>& diagonal) noexcept -> Mat {
            return from_diagonal(diagonal);
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr auto from_matrix(const Mat<U, ROWS, COLS>& m) noexcept -> Mat {
            Mat mat;
            for (usize r{}; r < ROWS; ++r)
                mat[r] = m[r].template as<value_type>();
            return mat;
        }

        template<nt::scalar... U> requires (sizeof...(U) == ROWS * COLS)
        [[nodiscard]] NOA_HD static constexpr auto from_values(U... values) noexcept -> Mat {
            Mat mat;
            auto op = [&mat, v = noa::forward_as_tuple(values...)]<usize I, usize... J>(){
                ((mat[I][J] = static_cast<value_type>(v[Tag<I * COLS + J>{}])), ...);
            };
            [&op]<usize... I, usize... J>(std::index_sequence<I...>, std::index_sequence<J...>) {
                (op.template operator()<I, J...>(), ...);
            }(std::make_index_sequence<ROWS>{}, std::make_index_sequence<COLS>{});
            return mat;
        }

        [[nodiscard]] NOA_HD static constexpr auto from_pointer(nt::scalar auto* ptr) noexcept -> Mat {
            return [&ptr]<usize...I>(std::index_sequence<I...>) {
                return Mat::from_values(ptr[I]...);
            }(std::make_index_sequence<ROWS * COLS>{});
        }

        template<nt::vec_scalar_size<COLS>... V> requires (sizeof...(V) == ROWS)
        [[nodiscard]] NOA_HD static constexpr auto from_rows(const V&... r) noexcept -> Mat {
            return {r.template as<value_type>()...};
        }

        template<nt::vec_scalar_size<ROWS>... V> requires (sizeof...(V) == COLS)
        [[nodiscard]] NOA_HD static constexpr auto from_columns(const V&... c) noexcept -> Mat {
            Mat mat;
            auto op = [&mat, v = noa::forward_as_tuple(c...)]<usize I, usize... J>(){
                ((mat[I][J] = static_cast<value_type>(v[Tag<J>{}][I])), ...);
            };
            [&op]<usize...I, usize...J>(std::index_sequence<I...>, std::index_sequence<J...>) {
                (op.template operator()<I, J...>(), ...);
            }(std::make_index_sequence<ROWS>{}, std::make_index_sequence<COLS>{});
            return mat;
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Mat<U>>(Mat<T>{}).
        template<typename U>
        [[nodiscard]] NOA_HD constexpr explicit operator Mat<U, ROWS, COLS>() const noexcept {
            return Mat<U, ROWS, COLS>::from_matrix(*this);
        }

    public: // Component accesses
        NOA_HD constexpr auto operator[](nt::integer auto i) noexcept -> row_type& {
            NOA_ASSERT(static_cast<usize>(i) < ROWS);
            return rows[i];
        }

        NOA_HD constexpr auto operator[](nt::integer auto i) const noexcept -> const row_type& {
            NOA_ASSERT(static_cast<usize>(i) < ROWS);
            return rows[i];
        }

        // Structure binding support.
        template<int I> [[nodiscard]] NOA_HD constexpr auto get() const noexcept -> const row_type& { return (*this)[I]; }
        template<int I> [[nodiscard]] NOA_HD constexpr row_type& get() noexcept { return (*this)[I]; }

        [[nodiscard]] NOA_HD constexpr auto data() const noexcept -> const row_type* { return rows; }
        [[nodiscard]] NOA_HD constexpr auto data() noexcept -> row_type* { return rows; }
        [[nodiscard]] NOA_HD static constexpr auto size() noexcept -> usize { return SIZE; }
        [[nodiscard]] NOA_HD static constexpr auto ssize() noexcept -> isize { return SSIZE; }

    public: // Iterators -- support for range loops
        [[nodiscard]] NOA_HD constexpr auto begin() noexcept -> row_type* { return data(); }
        [[nodiscard]] NOA_HD constexpr auto begin() const noexcept -> const row_type* { return data(); }
        [[nodiscard]] NOA_HD constexpr auto cbegin() const noexcept -> const row_type* { return data(); }
        [[nodiscard]] NOA_HD constexpr auto end() noexcept -> row_type* { return data() + SIZE; }
        [[nodiscard]] NOA_HD constexpr auto end() const noexcept -> const row_type* { return data() + SIZE; }
        [[nodiscard]] NOA_HD constexpr auto cend() const noexcept -> const row_type* { return data() + SIZE; }

    public: // Assignment operators
        NOA_HD constexpr auto operator+=(const Mat& m) noexcept -> Mat& {
            for (usize i{}; i < ROWS; ++i)
                rows[i] += m[i];
            return *this;
        }

        NOA_HD constexpr auto operator-=(const Mat& m) noexcept -> Mat& {
            for (usize i{}; i < ROWS; ++i)
                rows[i] -= m[i];
            return *this;
        }

        NOA_HD constexpr auto operator*=(const Mat& m) noexcept -> Mat& requires (ROWS == COLS)  {
            *this = nx::matmul(*this, m);
            return *this;
        }

        NOA_HD constexpr auto operator/=(const Mat& m) noexcept -> Mat& requires (ROWS == COLS) {
            *this *= nx::inverse(m);
            return *this;
        }

        NOA_HD constexpr auto operator+=(value_type s) noexcept -> Mat& {
            for (auto& r: rows)
                r += s;
            return *this;
        }

        NOA_HD constexpr auto operator-=(value_type s) noexcept -> Mat& {
            for (auto& r: rows)
                r -= s;
            return *this;
        }

        NOA_HD constexpr auto operator*=(value_type s) noexcept -> Mat& {
            for (auto& r: rows)
                r *= s;
            return *this;
        }

        NOA_HD constexpr auto operator/=(value_type s) noexcept -> Mat& {
            for (auto& r: rows)
                r /= s;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr auto operator+(const Mat& m) noexcept -> Mat {
            return m;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator-(const Mat& m) noexcept -> Mat {
            return [&m]<usize... I>(std::index_sequence<I...>) {
                return Mat{-m[I]...};
            }(std::make_index_sequence<ROWS>{});
        }

        // -- Binary arithmetic operators --
        [[nodiscard]] friend NOA_HD constexpr auto operator+(const Mat& m1, const Mat& m2) noexcept -> Mat {
            return [&]<usize... I>(std::index_sequence<I...>) {
                return Mat{m1[I] + m2[I]...};
            }(std::make_index_sequence<ROWS>{});
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator+(value_type s, const Mat& m) noexcept -> Mat {
            return [&]<usize... I>(std::index_sequence<I...>) {
                return Mat{s + m[I]...};
            }(std::make_index_sequence<ROWS>{});
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator+(const Mat& m, value_type s) noexcept -> Mat {
            return [&]<usize... I>(std::index_sequence<I...>) {
                return Mat{m[I] + s...};
            }(std::make_index_sequence<ROWS>{});
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator-(const Mat& m1, const Mat& m2) noexcept -> Mat {
            return [&]<usize... I>(std::index_sequence<I...>) {
                return Mat{m1[I] - m2[I]...};
            }(std::make_index_sequence<ROWS>{});
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator-(value_type s, const Mat& m) noexcept -> Mat {
            return [&]<usize... I>(std::index_sequence<I...>) {
                return Mat{s - m[I]...};
            }(std::make_index_sequence<ROWS>{});
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator-(const Mat& m, value_type s) noexcept -> Mat {
            return [&]<usize... I>(std::index_sequence<I...>) {
                return Mat{m[I] - s...};
            }(std::make_index_sequence<ROWS>{});
        }

        template<usize C1>
        [[nodiscard]] friend NOA_HD constexpr auto operator*(const Mat& m1, const Mat<value_type, COLS, C1>& m2) noexcept {
            return matmul(m1, m2);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator*(value_type s, const Mat& m) noexcept -> Mat {
            return [&]<usize... I>(std::index_sequence<I...>) {
                return Mat{s * m[I]...};
            }(std::make_index_sequence<ROWS>{});
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator*(const Mat& m, value_type s) noexcept -> Mat {
            return [&]<usize... I>(std::index_sequence<I...>) {
                return Mat{m[I] * s...};
            }(std::make_index_sequence<ROWS>{});
        }

        template<usize A>
        [[nodiscard]] friend NOA_HD constexpr auto operator*(const Mat& m, const Vec<value_type, C, A>& c) noexcept {
            return [&]<usize... I>(std::index_sequence<I...>) {
                return column_type{dot(m[I], c)...};
            }(std::make_index_sequence<ROWS>{});
        }

        template<usize A>
        [[nodiscard]] friend NOA_HD constexpr auto operator*(const Vec<value_type, R, A>& r, const Mat& m) noexcept {
            row_type out{};
            for (usize i{}; i < ROWS; ++i)
                out += r[i] * m[i];
            return out;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator/(Mat m1, const Mat& m2) noexcept -> Mat requires (ROWS == COLS) {
            m1 /= m2;
            return m1;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator/(value_type s, const Mat& m) noexcept -> Mat {
            return [&]<usize... I>(std::index_sequence<I...>) {
                return Mat{s / m[I]...};
            }(std::make_index_sequence<ROWS>{});
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator/(const Mat& m, value_type s) noexcept -> Mat {
            return [&]<usize... I>(std::index_sequence<I...>) {
                return Mat{m[I] / s...};
            }(std::make_index_sequence<ROWS>{});
        }

        template<usize A> requires (ROWS == COLS)
        [[nodiscard]] friend NOA_HD constexpr column_type operator/(const Mat& m, const Vec<value_type, C, A>& c) noexcept {
            return nx::inverse(m) * c;
        }

        template<usize A> requires (ROWS == COLS)
        [[nodiscard]] friend NOA_HD constexpr row_type operator/(const Vec<value_type, R, A>& r, const Mat& m) noexcept {
            return r * nx::inverse(m);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(const Mat& m1, const Mat& m2) noexcept {
            for (usize r{}; r < ROWS; ++r)
                if (m1[r] != m2[r])
                    return false;
            return true;
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(const Mat& m1, const Mat& m2) noexcept {
            return not(m1 == m2);
        }

    public:
        template<nt::real U>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return Mat<U, ROWS, COLS>::from_matrix(*this);
        }

        template<usize S = 1> requires (ROWS > S)
        [[nodiscard]] NOA_HD constexpr auto pop_front() const noexcept {
            return []<usize... I>(std::index_sequence<I...>, auto& m) { // nvcc bug - no capture
                return Mat<T, ROWS - S, COLS>::from_rows(m[I + S]...);
            }(std::make_index_sequence<ROWS - S>{}, *this);
        }

        template<usize S = 1> requires (ROWS > S)
        [[nodiscard]] NOA_HD constexpr auto pop_back() const noexcept {
            return []<usize... I>(std::index_sequence<I...>, auto& m) { // nvcc bug - no capture
                return Mat<T, ROWS - S, COLS>::from_rows(m[I]...);
            }(std::make_index_sequence<ROWS - S>{}, *this);
        }

        [[nodiscard]] NOA_HD constexpr auto push_front(const row_type& r) const noexcept {
            return []<usize... I>(std::index_sequence<I...>, auto& m, auto& r_) { // nvcc bug - no capture
                return Mat<value_type, ROWS + 1, COLS>{r_, m[I]...};
            }(std::make_index_sequence<ROWS>{}, *this, r);
        }

        [[nodiscard]] NOA_HD constexpr auto push_back(const row_type& r) const noexcept {
            return []<usize... I>(std::index_sequence<I...>, auto& m, auto& r_) { // nvcc bug - no capture
                return Mat<value_type, ROWS + 1, COLS>{m[I]..., r_};
            }(std::make_index_sequence<ROWS>{}, *this, r);
        }

        template<nt::integer... I>
        [[nodiscard]] NOA_HD constexpr auto filter_rows(I... i) const noexcept {
            return Mat<value_type, sizeof...(I), COLS>::from_rows((*this)[i]...);
        }
        template<nt::integer... I>
        [[nodiscard]] NOA_HD constexpr auto filter_columns(I... i) const noexcept {
            return Mat<value_type, ROWS, sizeof...(I)>::from_columns((*this).col(i)...);
        }

        [[nodiscard]] NOA_IHD constexpr auto row(nt::integer auto i) const noexcept -> row_type {
            return (*this)[i];
        }
        [[nodiscard]] NOA_IHD constexpr auto col(nt::integer auto i) const noexcept -> column_type {
            return []<usize... I>(std::index_sequence<I...>, auto& m, auto i_) { // nvcc bug - no capture
                return column_type{m[I][i_]...};
            }(std::make_index_sequence<ROWS>{}, *this, i);
        }

        [[nodiscard]] NOA_IHD constexpr auto transpose() const noexcept -> Mat {
            return nx::transpose(*this);
        }

        [[nodiscard]] NOA_IHD constexpr auto inverse() const noexcept -> Mat {
            return nx::inverse(*this);
        }
    };

    template<typename T> using Mat22 = Mat<T, 2, 2>;
    template<typename T> using Mat23 = Mat<T, 2, 3>;
    template<typename T> using Mat33 = Mat<T, 3, 3>;
    template<typename T> using Mat34 = Mat<T, 3, 4>;
    template<typename T> using Mat44 = Mat<T, 4, 4>;
}

namespace std {
    template<typename T, noa::usize R, noa::usize C>
    struct tuple_size<noa::xform::Mat<T, R, C>> : std::integral_constant<noa::usize, R> {};

    template<typename T, noa::usize R, noa::usize C>
    struct tuple_size<const noa::xform::Mat<T, R, C>> : std::integral_constant<noa::usize, R> {};

    template<noa::usize I, noa::usize R, noa::usize C, typename T>
    struct tuple_element<I, noa::xform::Mat<T, R, C>> { using type = T; };

    template<noa::usize I, noa::usize R, noa::usize C, typename T>
    struct tuple_element<I, const noa::xform::Mat<T, R, C>> { using type = const T; };
}

namespace noa::traits {
    template<typename T, usize R, usize C>
    struct proclaim_is_mat<nx::Mat<T, R, C>> : std::true_type {};

    template<typename T, usize R0, usize C0, usize R, usize C>
    struct proclaim_is_mat_of_shape<nx::Mat<T, R0, C0>, R, C> : std::bool_constant<R0 == R and C0 == C> {};

    template<mat T> struct proclaim_is_trivial_zero<T> : std::true_type {};
}

namespace noa::access {
    /// Returns the reordered matrix according to the indexes in \p order.
    /// The columns are reordered, and then the rows. This can be useful to swap the axes of a matrix.
    /// \param[in] matrix   Square and (truncated) affine matrix to reorder.
    /// \param[in] order    Order of indexes. Should have the same number of elements as the matrices are rows.
    template<nt::mat T, nt::integer I, usize N> requires (T::ROWS == N)
    [[nodiscard]] NOA_HD constexpr T reorder(const T& matrix, const Vec<I, N>& order) noexcept {
        T reordered_matrix;
        for (usize row{}; row < N; ++row) {
            typename T::row_type reordered_row{}; // no need to initialize, but g++ warn it may be uninitialized before use...
            for (usize column{}; column < N; ++column)
                reordered_row[column] = matrix[row][order[column]];
            reordered_matrix[order[row]] = reordered_row;
        }
        return reordered_matrix;
    }
}

namespace noa::details {
    template<typename T, usize R, usize C>
    struct Stringify<nx::Mat<T, R, C>> {
        static auto get() -> std::string {
            return fmt::format("Mat<{},{},{}>", stringify<T>(), R, C);
        }
    };
}
