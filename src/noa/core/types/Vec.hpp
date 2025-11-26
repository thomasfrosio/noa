#pragma once

#include <algorithm> // sort
#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/indexing/Bounds.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/utils/ClampCast.hpp"
#include "noa/core/utils/SafeCast.hpp"
#include "noa/core/utils/Sort.hpp"
#include "noa/core/utils/Strings.hpp"
#include "noa/core/Ewise.hpp"

namespace noa::details {
    // Add support for alignment requirement.
    template<typename T, size_t N, size_t A>
    struct VecAlignment {
    private:
        static_assert(is_power_of_2(A));
        static constexpr size_t required = std::max(A, alignof(T)); // requires alignment cannot be less that the type
        static constexpr size_t default_max_alignment = 16; // default alignment clamped at 16 bytes
        static constexpr size_t size_of = std::max(size_t{1}, sizeof(T) * N); // if N=0, use alignment of 1
        static constexpr size_t align_of = alignof(T);
    public:
        // E.g. alignof(Vec0<f32>) == 1
        // E.g. alignof(Vec2<f32>) == 8
        // E.g. alignof(Vec3<f32>) == 4
        // E.g. alignof(Vec2<f64>) == 16
        // E.g. alignof(Vec4<f32>) == 16
        static constexpr size_t value =
                A != 0 ? required :
                is_power_of_2(size_of) ? std::min(size_of, default_max_alignment) :
                align_of;
    };

    // Add support for empty vectors.
    template<typename T, size_t N>
    struct VecStorage {
        using type = T[N];
        template<typename I> NOA_HD static constexpr auto ref(type& t, I n) noexcept -> T& { return t[n]; }
        template<typename I> NOA_HD static constexpr auto ref(const type& t, I n) noexcept -> const T& { return t[n]; }
        NOA_HD static constexpr T* ptr(const type& t) noexcept { return const_cast<T*>(t); }
    };
    template<typename T>
    struct VecStorage<T, 0> {
        struct type {};
        template<typename I> NOA_HD static constexpr auto ref(type&, I) noexcept -> T& { return *static_cast<T*>(nullptr); }
        template<typename I> NOA_HD static constexpr auto ref(const type&, I) noexcept -> const T& { return *static_cast<const T*>(nullptr); }
        NOA_HD static constexpr T* ptr(const type&) noexcept { return nullptr; }
    };
}

namespace noa::inline types {
    /// Aggregates of N values with the same type.
    /// \tparam T Numeric or Vec (nested vectors are allowed).
    /// \tparam N Size of the vector. Empty vectors (N=0) are allowed.
    /// \tparam A Alignment requirement of the vector.
    template<typename T, size_t N, size_t A = 0>
    class alignas(details::VecAlignment<T, N, A>::value) Vec {
    public:
        static_assert(nt::numeric<T> or nt::vec<T>);
        static_assert(not std::is_const_v<T>);
        static_assert(not std::is_reference_v<T>);

        using storage_type = details::VecStorage<T, N>;
        using array_type = storage_type::type;
        using value_type = T;
        using mutable_value_type = value_type;
        static constexpr i64 SSIZE = N;
        static constexpr size_t SIZE = N;

    public:
        NOA_NO_UNIQUE_ADDRESS array_type array;

    public: // Static factory functions
        template<nt::static_castable_to<value_type> U>
        [[nodiscard]] NOA_HD static constexpr Vec from_value(const U& value) noexcept {
            if constexpr (SIZE == 0) {
                return {};
            } else {
                NOA_ASSERT(is_safe_cast<value_type>(value));
                Vec vec;
                auto value_cast = static_cast<value_type>(value);
                for (size_t i{}; i < SIZE; ++i)
                    vec[i] = value_cast;
                return vec;
            }
        }

        template<nt::static_castable_to<value_type> U>
        [[nodiscard]] NOA_HD static constexpr Vec filled_with(const U& value) noexcept {
            return from_value(value); // filled_with is a better name, but keep from_value for consistency
        }

        template<nt::static_castable_to<value_type>... U> requires (sizeof...(U) == SIZE)
        [[nodiscard]] NOA_HD static constexpr Vec from_values(const U&... values) noexcept {
            NOA_ASSERT((is_safe_cast<value_type>(values) and ...));
            return {static_cast<value_type>(values)...};
        }

        template<nt::static_castable_to<value_type> U, size_t AR>
        [[nodiscard]] NOA_HD static constexpr Vec from_vec(const Vec<U, SIZE, AR>& vector) noexcept {
            if constexpr (SIZE == 0) {
                return {};
            } else {
                Vec vec;
                for (size_t i{}; i < SIZE; ++i) {
                    NOA_ASSERT(is_safe_cast<value_type>(vector[i]));
                    vec[i] = static_cast<value_type>(vector[i]);
                }
                return vec;
            }
        }

        template<nt::static_castable_to<value_type> U>
        [[nodiscard]] NOA_HD static constexpr Vec from_pointer(const U* values) noexcept {
            if constexpr (SIZE == 0) {
                return {};
            } else {
                Vec vec;
                for (size_t i{}; i < SIZE; ++i)
                    vec[i] = static_cast<value_type>(values[i]);
                return vec;
            }
        }

        [[nodiscard]] NOA_HD static constexpr Vec arange(
            value_type start = 0,
            value_type step = 1
        ) noexcept requires nt::numeric<value_type> {
            if constexpr (SIZE == 0) {
                return {};
            } else {
                Vec vec;
                for (size_t i{}; i < SIZE; ++i, start += step)
                    vec[i] = start;
                return vec;
            }
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Vec<U>>(Vec<T>{}).
        template<typename U, size_t AR>
        [[nodiscard]] NOA_HD constexpr explicit operator Vec<U, SIZE, AR>() const noexcept {
            return Vec<U, SIZE, AR>::from_vec(*this);
        }

        // Allow implicit conversion from a vec with a different alignment.
        template<size_t AR> requires (A != AR)
        [[nodiscard]] NOA_HD constexpr /*implicit*/ operator Vec<value_type, SIZE, AR>() const noexcept {
            return Vec<value_type, SIZE, AR>::from_vec(*this);
        }

    public: // Accessor operators and functions
        template<nt::integer I> requires (SIZE > 0)
        [[nodiscard]] NOA_HD constexpr value_type& operator[](I i) noexcept {
            ni::bounds_check(SSIZE, i);
            return storage_type::ref(array, i);
        }

        template<nt::integer I> requires (SIZE > 0)
        [[nodiscard]] NOA_HD constexpr const value_type& operator[](I i) const noexcept {
            ni::bounds_check(SSIZE, i);
            return storage_type::ref(array, i);
        }

        // Structure binding support.
        template<int I> [[nodiscard]] NOA_HD constexpr const value_type& get() const noexcept { return (*this)[I]; }
        template<int I> [[nodiscard]] NOA_HD constexpr value_type& get() noexcept { return (*this)[I]; }

        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return storage_type::ptr(array); }
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return storage_type::ptr(array); }
        [[nodiscard]] NOA_HD static constexpr size_t size() noexcept { return SIZE; };
        [[nodiscard]] NOA_HD static constexpr i64 ssize() noexcept { return SSIZE; };

    public: // Iterators -- support for range loops
        [[nodiscard]] NOA_HD constexpr value_type* begin() noexcept { return data(); }
        [[nodiscard]] NOA_HD constexpr const value_type* begin() const noexcept { return data(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cbegin() const noexcept { return data(); }
        [[nodiscard]] NOA_HD constexpr value_type* end() noexcept { return data() + SSIZE; }
        [[nodiscard]] NOA_HD constexpr const value_type* end() const noexcept { return data() + SSIZE; }
        [[nodiscard]] NOA_HD constexpr const value_type* cend() const noexcept { return data() + SSIZE; }

    public: // Assignment operators
        NOA_HD constexpr Vec& operator=(const value_type& value) noexcept {
            *this = Vec::filled_with(value);
            return *this;
        }

        #define NOA_VEC_ARITH_(op)                                                  \
        NOA_HD constexpr Vec& operator op##=(const Vec& vector) noexcept {          \
            *this = *this op vector;                                                \
            return *this;                                                           \
        }                                                                           \
        NOA_HD constexpr Vec& operator op##=(const value_type& value) noexcept {    \
            *this = *this op value;                                                 \
            return *this;                                                           \
        }
        NOA_VEC_ARITH_(+)
        NOA_VEC_ARITH_(-)
        NOA_VEC_ARITH_(*)
        NOA_VEC_ARITH_(/)
        #undef NOA_VEC_ARITH_

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Vec operator+(const Vec& v) noexcept {
            return v;
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator-(Vec v) noexcept {
            if constexpr (SIZE > 0) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
                if constexpr (std::same_as<value_type, f16> and is_even(SSIZE)) {
                    auto* alias = reinterpret_cast<__half2*>(v.data());
                    for (size_t i{}; i < SIZE / 2; ++i)
                        alias[i] = -alias[i];
                    return v;
                }
                #endif
                for (size_t i{}; i < SIZE; ++i)
                    v[i] = -v[i];
                return v;
            } else {
                return v;
            }
        }

        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        #define NOA_VEC_ARITH_(op)                                                                               \
        [[nodiscard]] friend NOA_HD constexpr Vec operator op(Vec lhs, const Vec& rhs) noexcept {                \
            if constexpr (SIZE > 0) {                                                                            \
                if constexpr (std::same_as<value_type, f16> and is_even(SSIZE)) {                                \
                    auto* alias0 = reinterpret_cast<__half2*>(lhs.data());                                       \
                    const auto* alias1 = reinterpret_cast<const __half2*>(rhs.data());                           \
                    for (size_t i{}; i < SIZE / 2; ++i)                                                          \
                        alias0[i] op##= alias1[i];                                                               \
                    return lhs;                                                                                  \
                } else {                                                                                         \
                    for (size_t i{}; i < SIZE; ++i)                                                              \
                        lhs[i] op##= rhs[i];                                                                     \
                    return lhs;                                                                                  \
                }                                                                                                \
            } else {                                                                                             \
                return lhs;                                                                                      \
            }                                                                                                    \
        }                                                                                                        \
        [[nodiscard]] friend NOA_HD constexpr Vec operator op(const Vec& lhs, const value_type& rhs) noexcept {  \
            return lhs op Vec::filled_with(rhs);                                                                 \
        }                                                                                                        \
        [[nodiscard]] friend NOA_HD constexpr Vec operator op(const value_type& lhs, const Vec& rhs) noexcept {  \
            return Vec::filled_with(lhs) op rhs;                                                                 \
        }
        #else
        #define NOA_VEC_ARITH_(op)                                                                               \
        [[nodiscard]] friend NOA_HD constexpr Vec operator op(Vec lhs, const Vec& rhs) noexcept {                \
            if constexpr (SIZE > 0) {                                                                            \
                for (size_t i{}; i < SIZE; ++i)                                                                  \
                    lhs[i] op##= rhs[i];                                                                         \
                return lhs;                                                                                      \
            } else {                                                                                             \
                return lhs;                                                                                      \
            }                                                                                                    \
        }                                                                                                        \
        [[nodiscard]] friend NOA_HD constexpr Vec operator op(const Vec& lhs, const value_type& rhs) noexcept {  \
            return lhs op Vec::filled_with(rhs);                                                                 \
        }                                                                                                        \
        [[nodiscard]] friend NOA_HD constexpr Vec operator op(const value_type& lhs, const Vec& rhs) noexcept {  \
            return Vec::filled_with(lhs) op rhs;                                                                 \
        }
        #endif
        NOA_VEC_ARITH_(+)
        NOA_VEC_ARITH_(-)
        NOA_VEC_ARITH_(*)
        NOA_VEC_ARITH_(/)
        #undef NOA_VEC_ARITH_

        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        #define NOA_VEC_COMP_(op, cuda_name)                                                                     \
        [[nodiscard]] friend NOA_HD constexpr auto operator op(Vec lhs, const Vec& rhs) noexcept {               \
            using bool_t = decltype(std::declval<value_type>() op std::declval<value_type>());                   \
            if constexpr (SIZE > 0) {                                                                            \
                if constexpr (std::same_as<value_type, f16> and is_even(SSIZE)) {                                \
                    auto* alias0 = reinterpret_cast<__half2*>(lhs.data());                                       \
                    const auto* alias1 = reinterpret_cast<const __half2*>(rhs.data());                           \
                    Vec<bool, N> output;                                                                         \
                    for (size_t i{}; i < SIZE / 2; ++i) {                                                        \
                        alias0[i] = cuda_name(alias0[i], alias1[i]);                                             \
                        output[i * 2 + 0] = static_cast<bool>(alias0[i].x);                                      \
                        output[i * 2 + 1] = static_cast<bool>(alias0[i].y);                                      \
                    }                                                                                            \
                    return output;                                                                               \
                } else {                                                                                         \
                    Vec<bool_t, N> output;                                                                       \
                    for (size_t i{}; i < SIZE; ++i)                                                              \
                        output[i] = lhs[i] op rhs[i];                                                            \
                    return output;                                                                               \
                }                                                                                                \
            } else {                                                                                             \
                return Vec<bool_t, 0>{};                                                                         \
            }                                                                                                    \
        }                                                                                                        \
        [[nodiscard]] friend NOA_HD constexpr auto operator op(const Vec& lhs, const value_type& rhs) noexcept { \
            return lhs op Vec::filled_with(rhs);                                                                 \
        }                                                                                                        \
        [[nodiscard]] friend NOA_HD constexpr auto operator op(const value_type& lhs, const Vec& rhs) noexcept { \
            return Vec::filled_with(lhs) op rhs;                                                                 \
        }
        #else
        #define NOA_VEC_COMP_(op, cuda_name)                                                                     \
        [[nodiscard]] friend NOA_HD constexpr auto operator op(Vec lhs, const Vec& rhs) noexcept {               \
            using bool_t = decltype(std::declval<value_type>() op std::declval<value_type>());                   \
            if constexpr (SIZE > 0) {                                                                            \
                Vec<bool_t, N> output;                                                                           \
                for (size_t i{}; i < SIZE; ++i)                                                                  \
                    output[i] = lhs[i] op rhs[i];                                                                \
                return output;                                                                                   \
            } else {                                                                                             \
                return Vec<bool_t, 0>{};                                                                         \
            }                                                                                                    \
        }                                                                                                        \
        [[nodiscard]] friend NOA_HD constexpr auto operator op(const Vec& lhs, const value_type& rhs) noexcept { \
            return lhs op Vec::filled_with(rhs);                                                                 \
        }                                                                                                        \
        [[nodiscard]] friend NOA_HD constexpr auto operator op(const value_type& lhs, const Vec& rhs) noexcept { \
            return Vec::filled_with(lhs) op rhs;                                                                 \
        }
        #endif
        NOA_VEC_COMP_(>, __hgt2)
        NOA_VEC_COMP_(<, __hlt2)
        NOA_VEC_COMP_(>=, __hge2)
        NOA_VEC_COMP_(<=, __hle2)
        NOA_VEC_COMP_(==, __heq2)
        NOA_VEC_COMP_(!=, __hne2)
        #undef NOA_VEC_COMP_

    public: // Type casts
        template<nt::static_castable_to<value_type> U, size_t AR = 0>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return static_cast<Vec<U, SIZE, AR>>(*this);
        }

        template<nt::static_castable_to<value_type> U, size_t AR = 0>
        [[nodiscard]] NOA_HD constexpr auto as_clamp() const noexcept {
            return clamp_cast<Vec<U, SIZE, AR>>(*this);
        }

        template<nt::static_castable_to<value_type> U, size_t AR = 0>
        [[nodiscard]] constexpr auto as_safe() const {
            return safe_cast<Vec<U, SIZE, AR>>(*this);
        }

    public:
        template<size_t S = 1, size_t AR = 0> requires (N >= S)
        [[nodiscard]] NOA_HD constexpr auto pop_front() const noexcept {
            return Vec<value_type, N - S, AR>::from_pointer(data() + S);
        }

        template<size_t S = 1, size_t AR = 0> requires (N >= S)
        [[nodiscard]] NOA_HD constexpr auto pop_back() const noexcept {
            return Vec<value_type, N - S, AR>::from_pointer(data());
        }

        template<size_t S = 1, size_t AR = 0>
        [[nodiscard]] NOA_HD constexpr auto push_front(const value_type& value) const noexcept {
            Vec<value_type, N + S, AR> output;
            for (size_t i{}; i < S; ++i)
                output[i] = value;
            if constexpr (N > 0) {
                for (size_t i{}; i < N; ++i)
                    output[i + S] = (*this)[i];
            }
            return output;
        }

        template<size_t S = 1, size_t AR = 0>
        [[nodiscard]] NOA_HD constexpr auto push_back(const value_type& value) const noexcept {
            Vec<value_type, N + S, AR> output;
            if constexpr (N > 0) {
                for (size_t i{}; i < N; ++i)
                    output[i] = (*this)[i];
            }
            for (size_t i{}; i < S; ++i)
                output[N + i] = value;
            return output;
        }

        template<size_t AR = 0, size_t S, size_t AR0>
        [[nodiscard]] NOA_HD constexpr auto push_front(const Vec<value_type, S, AR0>& vector) const noexcept {
            Vec<value_type, N + S, AR> output;
            if constexpr (S > 0) {
                for (size_t i{}; i < S; ++i)
                    output[i] = vector[i];
            }
            if constexpr (N > 0) {
                for (size_t i{}; i < N; ++i)
                    output[i + S] = (*this)[i];
            }
            return output;
        }

        template<size_t AR = 0, size_t S, size_t AR0>
        [[nodiscard]] NOA_HD constexpr auto push_back(const Vec<value_type, S, AR0>& vector) const noexcept {
            Vec<value_type, N + S, AR> output;
            if constexpr (N > 0) {
                for (size_t i{}; i < N; ++i)
                    output[i] = (*this)[i];
            }
            if constexpr (S > 0) {
                for (size_t i{}; i < S; ++i)
                    output[i + N] = vector[i];
            }
            return output;
        }

        template<nt::integer... I>
        [[nodiscard]] NOA_HD constexpr auto filter(I... indices) const noexcept {
            return Vec<value_type, sizeof...(I)>{(*this)[indices]...};
        }

        [[nodiscard]] NOA_HD constexpr Vec flip() const noexcept {
            if constexpr (SIZE == 0) {
                return {};
            } else {
                Vec output;
                for (size_t i{}; i < SIZE; ++i)
                    output[i] = (*this)[(N - 1) - i];
                return output;
            }
        }

        template<nt::integer I = std::conditional_t<nt::integer<value_type>, value_type, i64>, size_t AR = 0>
        [[nodiscard]] NOA_HD constexpr Vec reorder(const Vec<I, SIZE, AR>& order) const noexcept {
            if constexpr (SIZE == 0) {
                return {};
            } else {
                Vec output;
                for (size_t i{}; i < SIZE; ++i)
                    output[i] = (*this)[order[i]];
                return output;
            }
        }

        // Circular shifts the vector by a given amount.
        // If "count" is positive, shift to the right, otherwise, shift to the left.
        [[nodiscard]] NOA_HD constexpr Vec circular_shift(i64 count) {
            if constexpr (SIZE <= 1) {
                return *this;
            } else {
                Vec out;
                const bool right = count >= 0;
                if (not right)
                    count *= -1;
                for (i64 i{}; i < SSIZE; ++i) {
                    const i64 idx = (i + count) % SSIZE;
                    out[idx * right + (1 - right) * i] = array[i * right + (1 - right) * idx];
                }
                return out;
            }
        }

        [[nodiscard]] NOA_HD constexpr Vec copy() const noexcept {
            return *this;
        }

        template<size_t INDEX> requires (INDEX < SIZE)
        [[nodiscard]] NOA_HD constexpr Vec set(const value_type& value) const noexcept {
            auto output = *this;
            output[INDEX] = value;
            return output;
        }
    };

    /// Deduction guide.
    template<typename T, typename... U>
    Vec(T, U...) -> Vec<std::enable_if_t<(std::same_as<T, U> and ...), T>, 1 + sizeof...(U)>;

    /// Type aliases.
    template<typename T> using Vec1 = Vec<T, 1>;
    template<typename T> using Vec2 = Vec<T, 2>;
    template<typename T> using Vec3 = Vec<T, 3>;
    template<typename T> using Vec4 = Vec<T, 4>;

    /// Support for output stream:
    template<typename T, size_t N, size_t A>
    inline std::ostream& operator<<(std::ostream& os, const Vec<T, N, A>& v) {
        if constexpr (nt::real_or_complex<T>)
            os << fmt::format("{::.3f}", v); // {fmt} ranges
        else
            os << fmt::format("{}", v); // FIXME
        return os;
    }
}

// Support for structure bindings:
namespace std {
    template<typename T, size_t N, size_t A>
    struct tuple_size<noa::Vec<T, N, A>> : std::integral_constant<size_t, N> {};

    template<typename T, size_t N, size_t A>
    struct tuple_size<const noa::Vec<T, N, A>> : std::integral_constant<size_t, N> {};

    template<size_t I, size_t N, size_t A, typename T>
    struct tuple_element<I, noa::Vec<T, N, A>> { using type = T; };

    template<size_t I, size_t N, size_t A, typename T>
    struct tuple_element<I, const noa::Vec<T, N, A>> { using type = const T; };
}

namespace noa::traits {
    template<typename T, size_t N, size_t A> struct proclaim_is_vec<noa::Vec<T, N, A>> : std::true_type {};
    template<typename V1, size_t N, size_t A, typename V2> struct proclaim_is_vec_of_type<noa::Vec<V1, N, A>, V2> : std::bool_constant<std::is_same_v<V1, V2>> {};
    template<typename V, size_t N1, size_t A, size_t N2> struct proclaim_is_vec_of_size<noa::Vec<V, N1, A>, N2> : std::bool_constant<N1 == N2> {};
}

namespace noa::inline types {
    template<typename T, size_t N, size_t A> requires (nt::boolean<T> or nt::vec<T>)
    [[nodiscard]] NOA_HD constexpr auto operator!(Vec<T, N, A> vector) noexcept {
        if constexpr (N > 0) {
            for (size_t i{}; i < N; ++i)
                vector[i] = !vector[i];
        }
        return vector;
    }

    template<typename T, size_t N, size_t A> requires (nt::boolean<T> or nt::vec<T>)
    [[nodiscard]] NOA_HD constexpr auto operator&&(Vec<T, N, A> lhs, const Vec<T, N, A>& rhs) noexcept {
        if constexpr (N > 0) {
            for (size_t i{}; i < N; ++i)
                lhs[i] = lhs[i] && rhs[i];
        }
        return lhs;
    }

    template<typename T, size_t N, size_t A> requires (nt::boolean<T> or nt::vec<T>)
    [[nodiscard]] NOA_HD constexpr auto operator||(Vec<T, N, A> lhs, const Vec<T, N, A>& rhs) noexcept {
        if constexpr (N > 0) {
            for (size_t i{}; i < N; ++i)
                lhs[i] = lhs[i] || rhs[i];
        }
        return lhs;
    }

    // -- Modulo Operator --
    template<nt::vec_integer V>
    [[nodiscard]] NOA_HD constexpr V operator%(V lhs, const V& rhs) noexcept {
        if constexpr (V::SSIZE > 0) {
            for (size_t i{}; i < V::SIZE; ++i)
                lhs[i] %= rhs[i];
        }
        return lhs;
    }

    template<nt::vec_integer V, nt::integer I>
    [[nodiscard]] NOA_HD constexpr V operator%(const V& lhs, I rhs) noexcept {
        return lhs % V::filled_with(rhs);
    }

    template<nt::vec_integer V, nt::integer I>
    [[nodiscard]] NOA_HD constexpr V operator%(I lhs, const V& rhs) noexcept {
        return V::filled_with(lhs) % rhs;
    }
}

namespace noa {
    // -- Cast--
    template<typename To, typename From, size_t N, size_t A> requires nt::vec_of_size<To, N>
    [[nodiscard]] NOA_HD constexpr bool is_safe_cast(const Vec<From, N, A>& src) noexcept {
        if constexpr (N == 0) {
            return true;
        } else {
            return [&src]<size_t...I>(std::index_sequence<I...>) {
                return (is_safe_cast<typename To::value_type>(src[I]) and ...);
            }(std::make_index_sequence<N>{});
        }
    }

    template<typename To, typename From, size_t N, size_t A> requires nt::vec_of_size<To, N>
    [[nodiscard]] NOA_HD constexpr To clamp_cast(const Vec<From, N, A>& src) noexcept {
        if constexpr (N == 0) {
            return {};
        } else {
            To output;
            for (size_t i{}; i < N; ++i)
                output[i] = clamp_cast<typename To::value_type>(src[i]);
            return output;
        }
    }

    template<nt::real T>
    [[nodiscard]] NOA_HD auto sincos(T x) noexcept -> Vec<T, 2> {
        Vec<T, 2> sin_cos;
        sincos(x, sin_cos.data(), sin_cos.data() + 1);
        return sin_cos;
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    #define NOA_VEC_MATH_GEN_(name, cuda_name, cpt)                                \
    template<typename T, size_t N, size_t A> requires (nt::cpt<T> or nt::vec<T>)   \
    [[nodiscard]] NOA_HD constexpr auto name(Vec<T, N, A> vector) noexcept {       \
        if constexpr (std::same_as<T, f16> and is_even(N)) {                       \
            auto* alias = reinterpret_cast<__half2*>(vector.data());               \
            for (size_t i{}; i < N / 2; ++i)                                       \
                alias[i] = cuda_name(alias[i]);                                    \
            return vector;                                                         \
        }                                                                          \
        for (size_t i{}; i < N; ++i)                                               \
            vector[i] = name(vector[i]);                                           \
        return vector;                                                             \
    }
#else
    #define NOA_VEC_MATH_GEN_(name, cuda_name, cpt)                                \
    template<typename T, size_t N, size_t A> requires (nt::cpt<T> or nt::vec<T>)   \
    [[nodiscard]] NOA_HD constexpr auto name(Vec<T, N, A> vector) noexcept {       \
        for (size_t i{}; i < N; ++i)                                               \
            vector[i] = name(vector[i]);                                           \
        return vector;                                                             \
    }
#endif

    NOA_VEC_MATH_GEN_(cos, h2cos, real)
    NOA_VEC_MATH_GEN_(sin, h2sin, real)
    NOA_VEC_MATH_GEN_(exp, h2exp, real)
    NOA_VEC_MATH_GEN_(log, h2log, real)
    NOA_VEC_MATH_GEN_(log10, h2log10, real)
    NOA_VEC_MATH_GEN_(sqrt, h2sqrt, real)
    NOA_VEC_MATH_GEN_(rsqrt, h2rsqrt, real)
    NOA_VEC_MATH_GEN_(round, h2rint, real) // h2rint is rounding to nearest
    NOA_VEC_MATH_GEN_(rint, h2rint, real)
    NOA_VEC_MATH_GEN_(ceil, h2ceil, real)
    NOA_VEC_MATH_GEN_(floor, h2floor, real)
    NOA_VEC_MATH_GEN_(abs, __habs2, scalar)
    #undef NOA_VEC_MATH_GEN_

    #define NOA_VEC_MATH_GEN_(func)                                                             \
    template<typename T, size_t N, size_t A> requires (nt::real<T> or nt::vec<T>)               \
    [[nodiscard]] NOA_HD constexpr auto func(Vec<T, N, A> vector) noexcept {                    \
        if constexpr (std::same_as<T, f16>) {                                                   \
            return func(vector.template as<typename T::arithmetic_type>()).template as<T, A>(); \
        }                                                                                       \
        for (size_t i{}; i < N; ++i)                                                            \
            vector[i] = func(vector[i]);                                                        \
        return vector;                                                                          \
    }
    NOA_VEC_MATH_GEN_(sinc)
    NOA_VEC_MATH_GEN_(tan)
    NOA_VEC_MATH_GEN_(acos)
    NOA_VEC_MATH_GEN_(asinc)
    NOA_VEC_MATH_GEN_(atan)
    NOA_VEC_MATH_GEN_(rad2deg)
    NOA_VEC_MATH_GEN_(deg2rad)
    NOA_VEC_MATH_GEN_(cosh)
    NOA_VEC_MATH_GEN_(sinh)
    NOA_VEC_MATH_GEN_(tanh)
    NOA_VEC_MATH_GEN_(acosh)
    NOA_VEC_MATH_GEN_(asinh)
    NOA_VEC_MATH_GEN_(atanh)
    NOA_VEC_MATH_GEN_(log1p)
    #undef NOA_VEC_MATH_GEN_

    template<typename T, size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto sum(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::same_as<T, f16>)
            return sum(vector.template as<typename T::arithmetic_type>()).template as<T, A>();

        return [&vector]<size_t...I>(std::index_sequence<I...>) {
            return (... + vector[I]);
        }(std::make_index_sequence<N>{});
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_HD constexpr auto mean(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::same_as<T, f16>)
            return mean(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        return sum(vector) / 2;
    }

    template<typename T, size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto product(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::same_as<T, f16>)
            return product(vector.template as<typename T::arithmetic_type>()).template as<T, A>();

        return [&vector]<size_t...I>(std::index_sequence<I...>) {
            return (... * vector[I]);
        }(std::make_index_sequence<N>{});
    }

    template<typename T, size_t N, size_t A0, size_t A1> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto dot(const Vec<T, N, A0>& lhs, const Vec<T, N, A1>& rhs) noexcept {
        if constexpr (std::same_as<T, f16>) {
            return static_cast<T>(dot(lhs.template as<typename T::arithmetic_type>(),
                                      rhs.template as<typename T::arithmetic_type>()));
        }

        return [&lhs, &rhs]<size_t...I>(std::index_sequence<I...>) {
            return (... + (lhs[I] * rhs[I]));
        }(std::make_index_sequence<N>{});
    }

    template<typename T, size_t N, size_t A> requires ((nt::real<T> or nt::vec<T>) and (N > 0))
    [[nodiscard]] NOA_HD constexpr auto norm(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::same_as<T, f16>) {
            const auto tmp = vector.template as<typename T::arithmetic_type>();
            return norm(tmp).template as<T, A>();
        }

        return sqrt(dot(vector, vector)); // euclidean norm
    }

    template<typename T, size_t N, size_t A> requires (nt::real<T> or nt::vec<T>)
    [[nodiscard]] NOA_HD constexpr auto normalize(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::same_as<T, f16>) {
            const auto tmp = vector.template as<typename T::arithmetic_type>();
            return normalize(tmp).template as<T, A>();
        }

        return vector / norm(vector); // may divide by 0
    }

    template<nt::scalar T, size_t A>
    [[nodiscard]] NOA_HD constexpr auto cross_product(const Vec<T, 3, A>& lhs, const Vec<T, 3, A>& rhs) noexcept {
        if constexpr (std::same_as<T, f16>) {
            using arithmetic_type = typename T::arithmetic_type;
            return cross_product(lhs.template as<arithmetic_type>(),
                                 rhs.template as<arithmetic_type>()).template as<T, A>();
        }

        return Vec<T, 3, A>{
                lhs[1] * rhs[2] - lhs[2] * rhs[1],
                lhs[2] * rhs[0] - lhs[0] * rhs[2],
                lhs[0] * rhs[1] - lhs[1] * rhs[0]};
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    #define NOA_VEC_MATH_GEN_(name)                                             \
    template<typename T, size_t N, size_t A> requires (N > 0)                   \
    [[nodiscard]] NOA_HD constexpr T name(                                      \
        const Vec<T, N, A>& vector                                              \
    ) noexcept {                                                                \
        if constexpr (N == 1) {                                                 \
            return vector[0];                                                   \
        } else {                                                                \
            if constexpr (std::same_as<T, f16> && N == 4) {                     \
                auto* alias = reinterpret_cast<const __half2*>(vector.data());  \
                const __half2 tmp = __h##name##2(alias[0], alias[1]);           \
                return __h##name(tmp.x, tmp.y);                                 \
            } else if constexpr (std::same_as<T, f16> && N == 8) {              \
                auto* alias = reinterpret_cast<const __half2*>(vector.data());  \
                const __half2 tmp0 = __h##name##2(alias[0], alias[1]);          \
                const __half2 tmp1 = __h##name##2(alias[2], alias[3]);          \
                const __half2 tmp2 = __h##name##2(tmp0, tmp1);                  \
                return __h##name(tmp2.x, tmp2.y);                               \
            } else {                                                            \
                auto element = name(vector[0], vector[1]);                      \
                for (size_t i = 2; i < N; ++i)                                  \
                    element = name(element, vector[i]);                         \
                return element;                                                 \
            }                                                                   \
        }                                                                       \
    }                                                                           \
    template<typename T, size_t N, size_t A>                                    \
    [[nodiscard]] NOA_HD constexpr auto name(                                   \
        Vec<T, N, A> lhs,                                                       \
        const Vec<T, N, A>& rhs                                                 \
    ) noexcept {                                                                \
        if constexpr (std::same_as<T, f16> and is_even(N)) {                    \
            auto* alias0 = reinterpret_cast<__half2*>(lhs.data());              \
            auto* alias1 = reinterpret_cast<__half2*>(rhs.data());              \
            for (size_t i{}; i < N / 2; ++i)                                    \
                alias0[i] = __h##name##2(alias0[i], alias1[i]);                 \
            return lhs;                                                         \
        } else {                                                                \
            for (size_t i{}; i < N; ++i)                                        \
                lhs[i] = name(lhs[i], rhs[i]);                                  \
            return lhs;                                                         \
        }                                                                       \
    }
#else
    #define NOA_VEC_MATH_GEN_(name)                                             \
    template<typename T, size_t N, size_t A> requires (N > 0)                   \
    [[nodiscard]] NOA_HD constexpr T name(                                      \
        const Vec<T, N, A>& vector                                              \
    ) noexcept {                                                                \
        if constexpr (N == 1) {                                                 \
            return vector[0];                                                   \
        } else {                                                                \
            auto element = name(vector[0], vector[1]);                          \
            for (size_t i = 2; i < N; ++i)                                      \
                element = name(element, vector[i]);                             \
            return element;                                                     \
        }                                                                       \
    }                                                                           \
    template<typename T, size_t N, size_t A>                                    \
    [[nodiscard]] NOA_HD constexpr auto name(                                   \
        Vec<T, N, A> lhs,                                                       \
        const Vec<T, N, A>& rhs                                                 \
    ) noexcept {                                                                \
        for (size_t i{}; i < N; ++i)                                            \
            lhs[i] = name(lhs[i], rhs[i]);                                      \
        return lhs;                                                             \
    }
#endif
    NOA_VEC_MATH_GEN_(min)
    NOA_VEC_MATH_GEN_(max)
    #undef NOA_VEC_MATH_GEN_

    /// Computes the argmax. Returns first occurrence if equal values are present.
    template<typename I = size_t, nt::numeric T, size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto argmax(const Vec<T, N, A>& vector) noexcept -> I {
        if constexpr (N == 1) {
            return 0;
        } else {
            auto compare = [&](I i, I j) { return vector[i] < vector[j] ? j : i; };
            auto index = compare(0, 1);
            for (I i = 2; i < N; ++i)
                index = compare(index, i);
            return index;
        }
    }

    /// Computes the argmax. Returns first occurrence if equal values are present.
    template<typename I = size_t, nt::numeric T, size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto argmin(const Vec<T, N, A>& vector) noexcept -> I {
        if constexpr (N == 1) {
            return 0;
        } else {
            auto compare = [&](I i, I j) { return vector[j] < vector[i] ? j : i; };
            auto index = compare(0, 1);
            for (I i = 2; i < N; ++i)
                index = compare(index, i);
            return index;
        }
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto min(const Vec<T, N, A>& lhs, std::type_identity_t<T> rhs) noexcept {
        return min(lhs, Vec<T, N, A>::filled_with(rhs));
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto min(std::type_identity_t<T> lhs, const Vec<T, N, A>& rhs) noexcept {
        return min(Vec<T, N, A>::filled_with(lhs), rhs);
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto max(const Vec<T, N, A>& lhs, std::type_identity_t<T> rhs) noexcept {
        return max(lhs, Vec<T, N, A>::filled_with(rhs));
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto max(std::type_identity_t<T> lhs, const Vec<T, N, A>& rhs) noexcept {
        return max(Vec<T, N, A>::filled_with(lhs), rhs);
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto clamp(
            const Vec<T, N, A>& lhs,
            const Vec<T, N, A>& low,
            const Vec<T, N, A>& high
    ) noexcept {
        return min(max(lhs, low), high);
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto clamp(
            const Vec<T, N, A>& lhs,
            std::type_identity_t<T> low,
            std::type_identity_t<T> high
    ) noexcept {
        return min(max(lhs, low), high);
    }

    template<i32 ULP = 2, nt::numeric T, size_t N, size_t A, typename U = nt::value_type_t<T>>
    [[nodiscard]] NOA_HD constexpr auto allclose(
            const Vec<T, N, A>& lhs,
            const Vec<T, N, A>& rhs,
            std::type_identity_t<U> epsilon = static_cast<U>(1e-6)
    ) {
        Vec<bool, N> output;
        for (size_t i{}; i < N; ++i)
            output[i] = allclose<ULP>(lhs[i], rhs[i], epsilon);
        return output;
    }

    template<i32 ULP = 2, nt::numeric T, size_t N, size_t A, typename U = nt::value_type_t<T>>
    [[nodiscard]] NOA_FHD constexpr auto allclose(
        const Vec<T, N, A>& lhs,
        std::type_identity_t<T> rhs,
        std::type_identity_t<U> epsilon = static_cast<U>(1e-6)
    ) noexcept {
        return allclose<ULP>(lhs, Vec<T, N, A>::filled_with(rhs), epsilon);
    }

    template<i32 ULP = 2, nt::numeric T, size_t N, size_t A, typename U = nt::value_type_t<T>>
    [[nodiscard]] NOA_FHD constexpr auto allclose(
        std::type_identity_t<T> lhs,
        const Vec<T, N, A>& rhs,
        std::type_identity_t<U> epsilon = static_cast<U>(1e-6)
    ) noexcept {
        return allclose<ULP>(Vec<T, N, A>::filled_with(lhs), rhs, epsilon);
    }

    template<typename T, size_t N, size_t A, typename Op = Less> requires (N <= 4)
    [[nodiscard]] constexpr auto stable_sort(Vec<T, N, A> vector, Op&& comp = {}) noexcept {
        small_stable_sort<N>(vector.data(), std::forward<Op>(comp));
        return vector;
    }

    template<typename T, size_t N, size_t A, typename Op = Less> requires (N <= 4)
    [[nodiscard]] constexpr auto sort(Vec<T, N, A> vector, Op&& comp = {}) noexcept {
        small_stable_sort<N>(vector.data(), std::forward<Op>(comp));
        return vector;
    }

    template<typename T, size_t N, size_t A, typename Op = Less> requires (N > 4)
    [[nodiscard]] auto stable_sort(Vec<T, N, A> vector, Op&& comp = {}) noexcept {
        std::stable_sort(vector.begin(), vector.end(), std::forward<Op>(comp));
        return vector;
    }

    template<typename T, size_t N, size_t A, typename Op = Less> requires (N > 4)
    [[nodiscard]] auto sort(Vec<T, N, A> vector, Op&& comp = {}) noexcept {
        std::sort(vector.begin(), vector.end(), std::forward<Op>(comp));
        return vector;
    }
}

namespace noa {
    /// Reduces the elements with the "or/||" operator.
    template<typename T, size_t N, size_t A> requires (N > 0 and (nt::boolean<T> or nt::vec<T>))
    [[nodiscard]] constexpr auto any(const Vec<T, N, A>& vector) noexcept {
        return [&vector]<size_t...I>(std::index_sequence<I...>) {
            return (vector[I] or ...);
        }(std::make_index_sequence<N>{});
    }
    [[nodiscard]] constexpr bool any(bool v) noexcept { return v; }

    /// Reduces the elements with the "and/&&" operator.
    template<typename T, size_t N, size_t A> requires (N > 0 and (nt::boolean<T> or nt::vec<T>))
    [[nodiscard]] constexpr auto all(const Vec<T, N, A>& vector) noexcept {
        return [&vector]<size_t...I>(std::index_sequence<I...>) {
            return (vector[I] and ...);
        }(std::make_index_sequence<N>{});
    }
    [[nodiscard]] constexpr bool all(bool v) noexcept { return v; }

    namespace details { // nvcc workaround
        template<size_t I, typename Op, typename... Args>
        constexpr bool vec_indexer(Op&& op, Args&&... args) {
            return static_cast<bool>(op(args[I]...));
        }
        template<size_t I, typename Op, typename... Args>
        constexpr bool vec_enumerate_indexer(Op&& op, Args&&... args) {
            return static_cast<bool>(op.template operator()<I>(args[I]...));
        }
    }

    /// Successively applies a function to each element of the vector(s), until one application returns false.
    /// Returns true if every application returned true, otherwise returns false.
    /// The arguments can be Vec|Shape|Strides of the same size or numeric(s).
    /// \example \code
    /// Vec<i64, 3> indices{...};
    /// Shape<i64, 3> shape{...};
    /// bool a = indices[0] >= 0 and indices[0] < shape[0] and
    ///          indices[1] >= 0 and indices[1] < shape[1] and
    ///          indices[2] >= 0 and indices[2] < shape[2];
    /// bool b = vall([](i64 i, i64 s) { return i >= 0 and i < s; }, indices, shape); // equivalent to a
    /// bool c = all(indices >= 0 and indices < shape);
    /// // `c` produces different code because all conditions have to be evaluated (Vec is not lazy evaluated).
    /// \endcode
    template<typename Op, typename... Args>
    requires (nt::vec_shape_or_strides<std::decay_t<Args>...> and nt::have_same_size_v<std::decay_t<Args>...>)
    constexpr bool vall(Op&& op, Args&&... args) {
        if constexpr (sizeof...(Args) > 0) {
            using first_t = nt::first_t<std::remove_reference_t<Args>...>;
            return []<size_t... I>(std::index_sequence<I...>, auto& op_, auto&... args_) {
                return (details::vec_indexer<I>(op_, args_...) and ...);
            }(std::make_index_sequence<first_t::SIZE>{}, op, args...); // nvcc bug - no capture
        } else {
            return true;
        }
    }
    template<typename Op, typename... Args>
    requires nt::numeric<std::decay_t<Args>...>
    constexpr bool vall(Op&& op, Args&&... args) {
        return static_cast<bool>(op(args...));
    }

    /// Similar to vall, except that the index of the current iteration is passed
    /// to the operator as the first template parameter.
    template<typename Op, typename... Args>
    requires (nt::vec_shape_or_strides<std::decay_t<Args>...> and nt::have_same_size_v<std::decay_t<Args>...>)
    constexpr bool vall_enumerate(Op&& op, Args&&... args) {
        if constexpr (sizeof...(Args) > 0) {
            using first_t = nt::first_t<std::remove_reference_t<Args>...>;
            return []<size_t... I>(std::index_sequence<I...>, auto& op_, auto&... args_) {
                return (details::vec_enumerate_indexer<I>(op_, args_...) and ...);
            }(std::make_index_sequence<first_t::SIZE>{}, op, args...); // nvcc bug - no capture
        } else {
            return true;
        }
    }
    template<typename Op, typename... Args>
    requires nt::numeric<std::decay_t<Args>...>
    constexpr bool vall_enumerate(Op&& op, Args&&... args) {
        return static_cast<bool>(op.template operator()<0>(args...));
    }

    /// Similar to vall, except that it returns true at the first application that returns true, otherwise returns false.
    template<typename Op, typename... Args>
    requires (nt::vec_shape_or_strides<std::decay_t<Args>...> and nt::have_same_size_v<std::decay_t<Args>...>)
    constexpr bool vany(Op&& op, Args&&... args) {
        if constexpr (sizeof...(Args) > 0) {
            using first_t = nt::first_t<std::remove_reference_t<Args>...>;
            return []<size_t... I>(std::index_sequence<I...>, auto& op_, auto&... args_) {
                return (details::vec_indexer<I>(op_, args_...) or ...);
            }(std::make_index_sequence<first_t::SIZE>{}, op, args...); // nvcc bug - no capture
        } else {
            return true;
        }
    }
    template<typename Op, typename... Args>
    requires nt::numeric<std::decay_t<Args>...>
    constexpr bool vany(Op&& op, Args&&... args) {
        return static_cast<bool>(op(args...));
    }

    /// Similar to vany, except that the index of the current iteration is passed
    /// to the operator as the first template parameter.
    template<typename Op, typename... Args>
    requires (nt::vec_shape_or_strides<std::decay_t<Args>...> and nt::have_same_size_v<std::decay_t<Args>...>)
    constexpr bool vany_enumerate(Op&& op, Args&&... args) {
        if constexpr (sizeof...(Args) > 0) {
            using first_t = nt::first_t<std::remove_reference_t<Args>...>;
            return []<size_t... I>(std::index_sequence<I...>, auto& op_, auto&... args_) {
                return (details::vec_enumerate_indexer<I>(op_, args_...) or ...);
            }(std::make_index_sequence<first_t::SIZE>{}, op, args...); // nvcc bug - no capture
        } else {
            return true;
        }
    }
    template<typename Op, typename... Args>
    requires nt::numeric<std::decay_t<Args>...>
    constexpr bool vany_enumerate(Op&& op, Args&&... args) {
        return static_cast<bool>(op.template operator()<0>(args...));
    }
}

namespace noa::string {
    template<typename T, size_t N>
    struct Stringify<Vec<T, N>> {
        static auto get() -> std::string {
            return fmt::format("Vec<{},{}>", noa::string::stringify<T>(), N);
        }
    };
}
