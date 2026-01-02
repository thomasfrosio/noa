#pragma once

#include <algorithm> // sort

#include "noa/base/Bounds.hpp"
#include "noa/base/ClampCast.hpp"
#include "noa/base/Config.hpp"
#include "noa/base/Math.hpp"
#include "noa/base/Operators.hpp"
#include "noa/base/SafeCast.hpp"
#include "noa/base/Strings.hpp"
#include "noa/base/Traits.hpp"

namespace noa::details {
    template<typename T, usize N, usize A>
    consteval auto vec_alignment() -> usize {
        static_assert(is_power_of_2(A), "The alignment should be a power of 2");

        // Use specified alignment.
        if (A != 0)
            return std::max(A, alignof(T));

        // Try over-alignment, up to 16 bytes.
        constexpr auto SIZEOF = std::max(usize{1}, sizeof(T) * N);
        if (is_power_of_2(SIZEOF)) {
            constexpr auto MAX_ALIGNMENT = usize{16};
            return std::min(SIZEOF, MAX_ALIGNMENT);
        }

        // Default alignment.
        return alignof(T);
    }

    template<typename T, usize N, usize A>
    constexpr usize vec_alignas = vec_alignment<T, N, A>(); // nvcc bug

    // Add support for empty vectors.
    template<typename T, usize N>
    struct VecStorage {
        using type = T[N];
        template<typename I> NOA_FHD static constexpr auto ref(type& t, I n) noexcept -> T& { return t[n]; }
        template<typename I> NOA_FHD static constexpr auto ref(const type& t, I n) noexcept -> const T& { return t[n]; }
        NOA_FHD static constexpr T* ptr(const type& t) noexcept { return const_cast<T*>(t); }
    };
    template<typename T>
    struct VecStorage<T, 0> {
        struct type {};
        template<typename I> NOA_FHD static constexpr auto ref(type&, I) noexcept -> T& { return *static_cast<T*>(nullptr); }
        template<typename I> NOA_FHD static constexpr auto ref(const type&, I) noexcept -> const T& { return *static_cast<const T*>(nullptr); }
        NOA_FHD static constexpr T* ptr(const type&) noexcept { return nullptr; }
    };

    template<typename Op, typename Lhs, typename Rhs>
    NOA_FHD constexpr auto vec_op(Lhs lhs, const Rhs& rhs) {
        if constexpr (Rhs::SIZE > 0) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::same_as<nt::value_type_t<Lhs>, f16> and is_even(Rhs::SIZE)) {
                auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
                const auto* alias1 = reinterpret_cast<const __half2*>(rhs.data());
                for (usize i{}; i < Rhs::SIZE / 2; ++i) {
                    if constexpr (nt::same_as<Op, Plus>)
                        alias0[i] += alias1[i];
                    else if constexpr (nt::same_as<Op, Minus>)
                        alias0[i] -= alias1[i];
                    else if constexpr (nt::same_as<Op, Multiply>)
                        alias0[i] *= alias1[i];
                    else if constexpr (nt::same_as<Op, Divide>)
                        alias0[i] /= alias1[i];
                    else
                        static_assert(nt::always_false<Op>);
                }
                return lhs;
            }
            #endif
            for (usize i{}; i < Rhs::SIZE; ++i) {
                if constexpr (nt::same_as<Op, Plus>)
                    lhs[i] += rhs[i];
                else if constexpr (nt::same_as<Op, Minus>)
                    lhs[i] -= rhs[i];
                else if constexpr (nt::same_as<Op, Multiply>)
                    lhs[i] *= rhs[i];
                else if constexpr (nt::same_as<Op, Divide>)
                    lhs[i] /= rhs[i];
                else if constexpr (nt::same_as<Op, Modulo>)
                    lhs[i] %= rhs[i];
                else
                    static_assert(nt::always_false<Op>);
            }
            return lhs;
        } else {
            return lhs;
        }
    }

    template<typename Op, typename Lhs, typename Rhs>
    NOA_FHD constexpr bool vec_op_bool(const Lhs& lhs, const Rhs& rhs) {
        return [&]<usize...I>(std::index_sequence<I...>) {
            if constexpr (nt::same_as<Op, Equal>) {
                return ((lhs[I] == rhs[I]) and ...);
            } else if constexpr (nt::same_as<Op, NotEqual>) {
                return ((lhs[I] != rhs[I]) or ...);
            } else if constexpr (nt::same_as<Op, Less>) {
                return ((lhs[I] < rhs[I]) and ...);
            } else if constexpr (nt::same_as<Op, Greater>) {
                return ((lhs[I] > rhs[I]) and ...);
            } else if constexpr (nt::same_as<Op, LessEqual>) {
                return ((lhs[I] <= rhs[I]) and ...);
            } else if constexpr (nt::same_as<Op, GreaterEqual>) {
                return ((lhs[I] >= rhs[I]) and ...);
            } else {
                static_assert(nt::always_false<Op>);
            }
        }(std::make_index_sequence<Rhs::SIZE>{});
    }

    template<typename Op, typename Lhs, typename Rhs>
    [[nodiscard]] NOA_FHD constexpr auto vec_cmp(const Lhs& lhs, const Rhs& rhs) noexcept {
        constexpr usize N = Rhs::SIZE;
        using bool_t = decltype(std::declval<nt::value_type_t<Lhs>>() == std::declval<nt::value_type_t<Lhs>>());

        if constexpr (N > 0) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::same_as<nt::value_type_t<Rhs>, f16> and is_even(N)) {
                const auto* alias0 = reinterpret_cast<const __half2*>(lhs.data());
                const auto* alias1 = reinterpret_cast<const __half2*>(rhs.data());
                Vec<bool, N, 0> output;
                for (usize i{}; i < N / 2; ++i) {
                    __half2 tmp;
                    if constexpr (nt::same_as<Op, Equal>) {
                        tmp = __hgt2(alias0[i], alias1[i]);
                    } else if constexpr (nt::same_as<Op, NotEqual>) {
                        tmp = __hlt2(alias0[i], alias1[i]);
                    } else if constexpr (nt::same_as<Op, Less>) {
                        tmp = __hge2(alias0[i], alias1[i]);
                    } else if constexpr (nt::same_as<Op, Greater>) {
                        tmp = __hle2(alias0[i], alias1[i]);
                    } else if constexpr (nt::same_as<Op, LessEqual>) {
                        tmp = __heq2(alias0[i], alias1[i]);
                    } else if constexpr (nt::same_as<Op, GreaterEqual>) {
                        tmp = __hne2(alias0[i], alias1[i]);
                    } else {
                        static_assert(nt::always_false<Op>);
                    }
                    output[i * 2 + 0] = static_cast<bool>(tmp.x);
                    output[i * 2 + 1] = static_cast<bool>(tmp.y);
                }
                return output;
            }
            #endif
            Vec<bool_t, N, 0> output;
            for (usize i{}; i < N; ++i) {
                if constexpr (nt::same_as<Op, Equal>) {
                    output[i] = lhs[i] == rhs[i];
                } else if constexpr (nt::same_as<Op, NotEqual>) {
                    output[i] = lhs[i] != rhs[i];
                } else if constexpr (nt::same_as<Op, Less>) {
                    output[i] = lhs[i] < rhs[i];
                } else if constexpr (nt::same_as<Op, Greater>) {
                    output[i] = lhs[i] > rhs[i];
                } else if constexpr (nt::same_as<Op, LessEqual>) {
                    output[i] = lhs[i] <= rhs[i];
                } else if constexpr (nt::same_as<Op, GreaterEqual>) {
                    output[i] = lhs[i] >= rhs[i];
                } else {
                    static_assert(nt::always_false<Op>);
                }
            }
            return output;
        } else {
            return Vec<bool_t, 0, 0>{};
        }
    }

    template<typename Op, typename Lhs, typename Rhs>
    NOA_FHD constexpr bool vec_any(const Lhs& lhs, const Rhs& rhs) {
        return [&]<usize...I>(std::index_sequence<I...>) {
            if constexpr (nt::same_as<Op, Equal>) {
                return ((lhs[I] == rhs[I]) or ...);
            } else if constexpr (nt::same_as<Op, NotEqual>) {
                return ((lhs[I] != rhs[I]) or ...);
            } else if constexpr (nt::same_as<Op, Less>) {
                return ((lhs[I] < rhs[I]) or ...);
            } else if constexpr (nt::same_as<Op, Greater>) {
                return ((lhs[I] > rhs[I]) or ...);
            } else if constexpr (nt::same_as<Op, LessEqual>) {
                return ((lhs[I] <= rhs[I]) or ...);
            } else if constexpr (nt::same_as<Op, GreaterEqual>) {
                return ((lhs[I] >= rhs[I]) or ...);
            } else {
                static_assert(nt::always_false<Op>);
            }
        }(std::make_index_sequence<Lhs::SIZE>{});
    }

    template<typename Op, typename Lhs> requires nt::vec<nt::value_type_t<Lhs>>
    [[nodiscard]] NOA_HD constexpr auto vec_func(const Lhs& vector) noexcept {
        for (usize i{}; i < Lhs::SIZE; ++i)
            vector[i] = vec_func<Op>(vector[i]);
        return vector;
    }

    template<typename Op, typename Lhs> requires (not nt::vec<nt::value_type_t<Lhs>>)
    [[nodiscard]] NOA_HD constexpr auto vec_func(Lhs vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::same_as<nt::value_type_t<Lhs>, f16> and is_even(Lhs::SIZE)) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            for (usize i{}; i < Lhs::SIZE / 2; ++i) {
                if constexpr (std::same_as<Op, Cos>)
                    alias[i] = h2cos(alias[i]);
                else if constexpr (std::same_as<Op, Sin>)
                    alias[i] = h2sin(alias[i]);
                else if constexpr (std::same_as<Op, Exp>)
                    alias[i] = h2exp(alias[i]);
                else if constexpr (std::same_as<Op, Log>)
                    alias[i] = h2log(alias[i]);
                else if constexpr (std::same_as<Op, Log10>)
                    alias[i] = h2log10(alias[i]);
                else if constexpr (std::same_as<Op, Sqrt>)
                    alias[i] = h2sqrt(alias[i]);
                else if constexpr (std::same_as<Op, Rsqrt>)
                    alias[i] = h2rsqrt(alias[i]);
                else if constexpr (nt::any_of<Op, Round, Rint>)
                    alias[i] = h2rint(alias[i]);
                else if constexpr (std::same_as<Op, Ceil>)
                    alias[i] = h2ceil(alias[i]);
                else if constexpr (std::same_as<Op, Floor>)
                    alias[i] = h2floor(alias[i]);
                else if constexpr (std::same_as<Op, Abs>)
                    alias[i] = __habs2(alias[i]);
                else
                    static_assert(nt::always_false<Op>);
            }
            return vector;
        }
        #endif
        for (usize i{}; i < Lhs::SIZE; ++i)
            vector[i] = Op{}(vector[i]);
        return vector;
    }

    // In-place stable sort for small arrays.
    template<usize N, typename T, typename U> requires (N <= 4)
    NOA_HD constexpr void small_stable_sort(T* begin, U&& comp) noexcept {
        auto sswap = [](auto& a, auto& b) {
            T tmp = a;
            a = b;
            b = tmp;
        };

        if constexpr (N <= 1) {
            return;
        } else if constexpr (N == 2) {
            if (comp(begin[1], begin[0]))
                sswap(begin[0], begin[1]);
        } else if constexpr (N == 3) {
            // Insertion sort:
            if (comp(begin[1], begin[0]))
                sswap(begin[0], begin[1]);
            if (comp(begin[2], begin[1])) {
                sswap(begin[1], begin[2]);
                if (comp(begin[1], begin[0]))
                    sswap(begin[1], begin[0]);
            }
        } else if constexpr (N == 4) {
            // Sorting network, using insertion sort:
            if (comp(begin[3], begin[2]))
                sswap(begin[3], begin[2]);
            if (comp(begin[2], begin[1]))
                sswap(begin[2], begin[1]);
            if (comp(begin[3], begin[2]))
                sswap(begin[3], begin[2]);
            if (comp(begin[1], begin[0]))
                sswap(begin[1], begin[0]);
            if (comp(begin[2], begin[1]))
                sswap(begin[2], begin[1]);
            if (comp(begin[3], begin[2]))
                sswap(begin[3], begin[2]);
        } else {
            static_assert(nt::always_false<T>);
        }
    }

    // In-place stable sort for small arrays. Sort in ascending order.
    template<usize N, typename T> requires (N <= 4)
    NOA_HD constexpr void small_stable_sort(T* begin) noexcept {
        small_stable_sort<N>(begin, [](const T& a, const T& b) { return a < b; });
    }
}

namespace noa::traits {
    template<typename T> concept vec_real_or_vec = nt::vec_real<T> or nt::vec<nt::value_type_t<T>>;
    template<typename T> concept vec_scalar_or_vec = nt::vec_scalar<T> or nt::vec<nt::value_type_t<T>>;
}

namespace noa::inline types {
    /// Aggregates of N values with the same type.
    /// \tparam T Numeric or Vec (nested vectors are allowed).
    /// \tparam N Size of the vector. Empty vectors (N=0) are allowed.
    /// \tparam A Alignment requirement of the vector.
    template<typename T, usize N, usize A = 0>
    class alignas(nd::vec_alignas<T, N, A>) Vec {
    public:
        static_assert(nt::numeric<T> or nt::vec<T>);
        static_assert(not std::is_const_v<T>);
        static_assert(not std::is_reference_v<T>);

        using storage_type = details::VecStorage<T, N>;
        using array_type = storage_type::type;
        using value_type = T;
        using mutable_value_type = value_type;
        static constexpr isize SSIZE = N;
        static constexpr usize SIZE = N;

    public:
        NOA_NO_UNIQUE_ADDRESS array_type array;

    public: // Static factory functions
        template<nt::static_castable_to<value_type> U>
        [[nodiscard]] NOA_HD static constexpr auto from_value(const U& value) noexcept -> Vec {
            if constexpr (SIZE == 0) {
                return {};
            } else {
                NOA_ASSERT(is_safe_cast<value_type>(value));
                Vec vec;
                auto value_cast = static_cast<value_type>(value);
                for (usize i{}; i < SIZE; ++i)
                    vec[i] = value_cast;
                return vec;
            }
        }

        template<nt::static_castable_to<value_type> U>
        [[nodiscard]] NOA_HD static constexpr auto filled_with(const U& value) noexcept -> Vec {
            return from_value(value);
        }

        template<nt::static_castable_to<value_type>... U> requires (sizeof...(U) == SIZE)
        [[nodiscard]] NOA_HD static constexpr auto from_values(const U&... values) noexcept -> Vec {
            NOA_ASSERT((is_safe_cast<value_type>(values) and ...));
            return {static_cast<value_type>(values)...};
        }

        template<nt::static_castable_to<value_type> U, usize AR>
        [[nodiscard]] NOA_HD static constexpr auto from_vec(const Vec<U, SIZE, AR>& vector) noexcept -> Vec {
            if constexpr (SIZE == 0) {
                return {};
            } else {
                Vec vec;
                for (usize i{}; i < SIZE; ++i) {
                    NOA_ASSERT(is_safe_cast<value_type>(vector[i]));
                    vec[i] = static_cast<value_type>(vector[i]);
                }
                return vec;
            }
        }

        template<nt::static_castable_to<value_type> U>
        [[nodiscard]] NOA_HD static constexpr auto from_pointer(const U* values) noexcept -> Vec {
            if constexpr (SIZE == 0) {
                return {};
            } else {
                Vec vec;
                for (usize i{}; i < SIZE; ++i)
                    vec[i] = static_cast<value_type>(values[i]);
                return vec;
            }
        }

        [[nodiscard]] NOA_HD static constexpr auto arange(
            value_type start = 0,
            value_type step = 1
        ) noexcept -> Vec requires nt::numeric<value_type> {
            if constexpr (SIZE == 0) {
                return {};
            } else {
                Vec vec;
                for (usize i{}; i < SIZE; ++i, start += step)
                    vec[i] = start;
                return vec;
            }
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Vec<U>>(Vec<T>{}).
        template<typename U, usize AR>
        [[nodiscard]] NOA_HD constexpr explicit operator Vec<U, SIZE, AR>() const noexcept {
            return Vec<U, SIZE, AR>::from_vec(*this);
        }

        // Allow implicit conversion from a vec with a different alignment.
        template<usize AR> requires (A != AR)
        [[nodiscard]] NOA_HD constexpr /*implicit*/ operator Vec<value_type, SIZE, AR>() const noexcept {
            return Vec<value_type, SIZE, AR>::from_vec(*this);
        }

    public: // Accessor operators and functions
        template<nt::integer I> requires (SIZE > 0)
        [[nodiscard]] NOA_FHD constexpr auto operator[](I i) noexcept -> value_type& {
            noa::bounds_check(SSIZE, i);
            return storage_type::ref(array, i);
        }

        template<nt::integer I> requires (SIZE > 0)
        [[nodiscard]] NOA_FHD constexpr auto operator[](I i) const noexcept -> const value_type& {
            noa::bounds_check(SSIZE, i);
            return storage_type::ref(array, i);
        }

        // Structure binding support.
        template<int I> [[nodiscard]] NOA_FHD constexpr auto get() const noexcept -> const value_type& { return (*this)[I]; }
        template<int I> [[nodiscard]] NOA_FHD constexpr auto get() noexcept -> value_type& { return (*this)[I]; }

        [[nodiscard]] NOA_FHD constexpr auto data() const noexcept -> const value_type* { return storage_type::ptr(array); }
        [[nodiscard]] NOA_FHD constexpr auto data() noexcept -> value_type* { return storage_type::ptr(array); }
        [[nodiscard]] NOA_FHD static constexpr auto size() noexcept -> usize { return SIZE; };
        [[nodiscard]] NOA_FHD static constexpr auto ssize() noexcept -> isize { return SSIZE; };

    public: // Iterators -- support for range loops
        [[nodiscard]] NOA_FHD constexpr auto begin() noexcept -> value_type* { return data(); }
        [[nodiscard]] NOA_FHD constexpr auto begin() const noexcept -> const value_type* { return data(); }
        [[nodiscard]] NOA_FHD constexpr auto cbegin() const noexcept -> const value_type* { return data(); }
        [[nodiscard]] NOA_FHD constexpr auto end() noexcept -> value_type* { return data() + SSIZE; }
        [[nodiscard]] NOA_FHD constexpr auto end() const noexcept -> const value_type* { return data() + SSIZE; }
        [[nodiscard]] NOA_FHD constexpr auto cend() const noexcept -> const value_type* { return data() + SSIZE; }

    public: // Assignment operators
        NOA_FHD constexpr auto operator=(const value_type& value) noexcept -> Vec& {
            *this = Vec::filled_with(value);
            return *this;
        }

        NOA_FHD constexpr auto operator+=(const Vec& vector) noexcept -> Vec& { *this = *this + vector; return *this; }
        NOA_FHD constexpr auto operator-=(const Vec& vector) noexcept -> Vec& { *this = *this - vector; return *this; }
        NOA_FHD constexpr auto operator*=(const Vec& vector) noexcept -> Vec& { *this = *this * vector; return *this; }
        NOA_FHD constexpr auto operator/=(const Vec& vector) noexcept -> Vec& { *this = *this / vector; return *this; }

        NOA_FHD constexpr auto operator+=(const value_type& value) noexcept -> Vec& { *this = *this + value; return *this; }
        NOA_FHD constexpr auto operator-=(const value_type& value) noexcept -> Vec& { *this = *this - value; return *this; }
        NOA_FHD constexpr auto operator*=(const value_type& value) noexcept -> Vec& { *this = *this * value; return *this; }
        NOA_FHD constexpr auto operator/=(const value_type& value) noexcept -> Vec& { *this = *this / value; return *this; }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_FHD constexpr auto operator+(const Vec& v) noexcept -> Vec {
            return v;
        }

        [[nodiscard]] friend NOA_FHD constexpr auto operator-(Vec v) noexcept -> Vec {
            if constexpr (SIZE > 0) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
                if constexpr (std::same_as<value_type, f16> and is_even(SSIZE)) {
                    auto* alias = reinterpret_cast<__half2*>(v.data());
                    for (usize i{}; i < SIZE / 2; ++i)
                        alias[i] = -alias[i];
                    return v;
                }
                #endif
                for (usize i{}; i < SIZE; ++i)
                    v[i] = -v[i];
                return v;
            } else {
                return v;
            }
        }

    public: // arithmetic operators
        [[nodiscard]] NOA_FHD friend constexpr Vec operator+(const Vec& lhs, const Vec& rhs) noexcept { return nd::vec_op<Plus>(lhs, rhs); }
        [[nodiscard]] NOA_FHD friend constexpr Vec operator-(const Vec& lhs, const Vec& rhs) noexcept { return nd::vec_op<Minus>(lhs, rhs); }
        [[nodiscard]] NOA_FHD friend constexpr Vec operator*(const Vec& lhs, const Vec& rhs) noexcept { return nd::vec_op<Multiply>(lhs, rhs); }
        [[nodiscard]] NOA_FHD friend constexpr Vec operator/(const Vec& lhs, const Vec& rhs) noexcept { return nd::vec_op<Divide>(lhs, rhs); }
        [[nodiscard]] NOA_FHD friend constexpr Vec operator%(const Vec& lhs, const Vec& rhs) noexcept requires nt::integer<value_type> { return nd::vec_op<Modulo>(lhs, rhs); }

        [[nodiscard]] NOA_FHD friend constexpr auto operator+(const Vec& lhs, const value_type& rhs) noexcept -> Vec { return lhs + Vec::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr auto operator-(const Vec& lhs, const value_type& rhs) noexcept -> Vec { return lhs - Vec::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr auto operator*(const Vec& lhs, const value_type& rhs) noexcept -> Vec { return lhs * Vec::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr auto operator/(const Vec& lhs, const value_type& rhs) noexcept -> Vec { return lhs / Vec::filled_with(rhs); }

        [[nodiscard]] NOA_FHD friend constexpr auto operator+(const value_type& lhs, const Vec& rhs) noexcept -> Vec { return Vec::filled_with(lhs) + rhs; }
        [[nodiscard]] NOA_FHD friend constexpr auto operator-(const value_type& lhs, const Vec& rhs) noexcept -> Vec { return Vec::filled_with(lhs) - rhs; }
        [[nodiscard]] NOA_FHD friend constexpr auto operator*(const value_type& lhs, const Vec& rhs) noexcept -> Vec { return Vec::filled_with(lhs) * rhs; }
        [[nodiscard]] NOA_FHD friend constexpr auto operator/(const value_type& lhs, const Vec& rhs) noexcept -> Vec { return Vec::filled_with(lhs) / rhs; }

    public: // comparison operators
        [[nodiscard]] NOA_FHD friend constexpr bool operator==(const Vec& lhs, const Vec& rhs) noexcept { return nd::vec_op_bool<Equal>(lhs, rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator!=(const Vec& lhs, const Vec& rhs) noexcept { return nd::vec_op_bool<NotEqual>(lhs, rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<=(const Vec& lhs, const Vec& rhs) noexcept { return nd::vec_op_bool<LessEqual>(lhs, rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>=(const Vec& lhs, const Vec& rhs) noexcept { return nd::vec_op_bool<GreaterEqual>(lhs, rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<(const Vec& lhs, const Vec& rhs) noexcept { return nd::vec_op_bool<Less>(lhs, rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>(const Vec& lhs, const Vec& rhs) noexcept { return nd::vec_op_bool<Greater>(lhs, rhs); }

        [[nodiscard]] NOA_FHD friend constexpr bool operator==(const Vec& lhs, const value_type& rhs) noexcept { return lhs == Vec::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator!=(const Vec& lhs, const value_type& rhs) noexcept { return lhs != Vec::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<=(const Vec& lhs, const value_type& rhs) noexcept { return lhs <= Vec::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>=(const Vec& lhs, const value_type& rhs) noexcept { return lhs >= Vec::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<(const Vec& lhs, const value_type& rhs) noexcept { return lhs < Vec::filled_with(rhs); }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>(const Vec& lhs, const value_type& rhs) noexcept { return lhs > Vec::filled_with(rhs); }

        [[nodiscard]] NOA_FHD friend constexpr bool operator==(const value_type& lhs, const Vec& rhs) noexcept { return Vec::filled_with(lhs) == rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator!=(const value_type& lhs, const Vec& rhs) noexcept { return Vec::filled_with(lhs) != rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<=(const value_type& lhs, const Vec& rhs) noexcept { return Vec::filled_with(lhs) <= rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>=(const value_type& lhs, const Vec& rhs) noexcept { return Vec::filled_with(lhs) >= rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator<(const value_type& lhs, const Vec& rhs) noexcept { return Vec::filled_with(lhs) < rhs; }
        [[nodiscard]] NOA_FHD friend constexpr bool operator>(const value_type& lhs, const Vec& rhs) noexcept { return Vec::filled_with(lhs) > rhs; }

    public: // element-wise comparison
        [[nodiscard]] NOA_FHD constexpr auto cmp_eq(const Vec& rhs) const noexcept { return nd::vec_cmp<Equal>(*this, rhs); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_ne(const Vec& rhs) const noexcept { return nd::vec_cmp<NotEqual>(*this, rhs); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_le(const Vec& rhs) const noexcept { return nd::vec_cmp<LessEqual>(*this, rhs); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_ge(const Vec& rhs) const noexcept { return nd::vec_cmp<GreaterEqual>(*this, rhs); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_lt(const Vec& rhs) const noexcept { return nd::vec_cmp<Less>(*this, rhs); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_gt(const Vec& rhs) const noexcept { return nd::vec_cmp<Greater>(*this, rhs); }

        [[nodiscard]] NOA_FHD constexpr auto cmp_eq(const value_type& rhs) const noexcept { return nd::vec_cmp<Equal>(*this, Vec::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_ne(const value_type& rhs) const noexcept { return nd::vec_cmp<NotEqual>(*this, Vec::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_le(const value_type& rhs) const noexcept { return nd::vec_cmp<LessEqual>(*this, Vec::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_ge(const value_type& rhs) const noexcept { return nd::vec_cmp<GreaterEqual>(*this, Vec::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_lt(const value_type& rhs) const noexcept { return nd::vec_cmp<Less>(*this, Vec::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr auto cmp_gt(const value_type& rhs) const noexcept { return nd::vec_cmp<Greater>(*this, Vec::filled_with(rhs)); }
        // [[nodiscard]] NOA_FHD constexpr auto cmp_allclose(const value_type& rhs) const noexcept { return nd::vec_cmp<Greater>(*this, Vec::filled_with(rhs)); }

        [[nodiscard]] NOA_FHD constexpr bool any_eq(const Vec& rhs) const noexcept { return nd::vec_any<Equal>(*this, rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_ne(const Vec& rhs) const noexcept { return nd::vec_any<NotEqual>(*this, rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_le(const Vec& rhs) const noexcept { return nd::vec_any<LessEqual>(*this, rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_ge(const Vec& rhs) const noexcept { return nd::vec_any<GreaterEqual>(*this, rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_lt(const Vec& rhs) const noexcept { return nd::vec_any<Less>(*this, rhs); }
        [[nodiscard]] NOA_FHD constexpr bool any_gt(const Vec& rhs) const noexcept { return nd::vec_any<Greater>(*this, rhs); }

        [[nodiscard]] NOA_FHD constexpr bool any_eq(const value_type& rhs) const noexcept { return nd::vec_any<Equal>(*this, Vec::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr bool any_ne(const value_type& rhs) const noexcept { return nd::vec_any<NotEqual>(*this, Vec::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr bool any_le(const value_type& rhs) const noexcept { return nd::vec_any<LessEqual>(*this, Vec::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr bool any_ge(const value_type& rhs) const noexcept { return nd::vec_any<GreaterEqual>(*this, Vec::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr bool any_lt(const value_type& rhs) const noexcept { return nd::vec_any<Less>(*this, Vec::filled_with(rhs)); }
        [[nodiscard]] NOA_FHD constexpr bool any_gt(const value_type& rhs) const noexcept { return nd::vec_any<Greater>(*this, Vec::filled_with(rhs)); }

    public: // Type casts
        template<nt::static_castable_to<value_type> U, usize AR = 0>
        [[nodiscard]] NOA_FHD constexpr auto as() const noexcept {
            return static_cast<Vec<U, SIZE, AR>>(*this);
        }

        template<nt::static_castable_to<value_type> U, usize AR = 0>
        [[nodiscard]] NOA_FHD constexpr auto as_clamp() const noexcept {
            return clamp_cast<Vec<U, SIZE, AR>>(*this);
        }

        template<nt::static_castable_to<value_type> U, usize AR = 0>
        [[nodiscard]] constexpr auto as_safe() const {
            return safe_cast<Vec<U, SIZE, AR>>(*this);
        }

    public:
        template<usize S = 1, usize AR = 0> requires (N >= S)
        [[nodiscard]] NOA_FHD constexpr auto pop_front() const noexcept {
            return Vec<value_type, N - S, AR>::from_pointer(data() + S);
        }

        template<usize S = 1, usize AR = 0> requires (N >= S)
        [[nodiscard]] NOA_FHD constexpr auto pop_back() const noexcept {
            return Vec<value_type, N - S, AR>::from_pointer(data());
        }

        template<usize S = 1, usize AR = 0>
        [[nodiscard]] NOA_FHD constexpr auto push_front(const value_type& value) const noexcept {
            Vec<value_type, N + S, AR> output;
            for (usize i{}; i < S; ++i)
                output[i] = value;
            if constexpr (N > 0) {
                for (usize i{}; i < N; ++i)
                    output[i + S] = (*this)[i];
            }
            return output;
        }

        template<usize S = 1, usize AR = 0>
        [[nodiscard]] NOA_FHD constexpr auto push_back(const value_type& value) const noexcept {
            Vec<value_type, N + S, AR> output;
            if constexpr (N > 0) {
                for (usize i{}; i < N; ++i)
                    output[i] = (*this)[i];
            }
            for (usize i{}; i < S; ++i)
                output[N + i] = value;
            return output;
        }

        template<usize AR = 0, usize S, usize AR0>
        [[nodiscard]] NOA_FHD constexpr auto push_front(const Vec<value_type, S, AR0>& vector) const noexcept {
            Vec<value_type, N + S, AR> output;
            if constexpr (S > 0) {
                for (usize i{}; i < S; ++i)
                    output[i] = vector[i];
            }
            if constexpr (N > 0) {
                for (usize i{}; i < N; ++i)
                    output[i + S] = (*this)[i];
            }
            return output;
        }

        template<usize AR = 0, usize S, usize AR0>
        [[nodiscard]] NOA_FHD constexpr auto push_back(const Vec<value_type, S, AR0>& vector) const noexcept {
            Vec<value_type, N + S, AR> output;
            if constexpr (N > 0) {
                for (usize i{}; i < N; ++i)
                    output[i] = (*this)[i];
            }
            if constexpr (S > 0) {
                for (usize i{}; i < S; ++i)
                    output[i + N] = vector[i];
            }
            return output;
        }

        template<nt::integer... I>
        [[nodiscard]] NOA_FHD constexpr auto filter(I... indices) const noexcept {
            return Vec<value_type, sizeof...(I)>{(*this)[indices]...};
        }

        [[nodiscard]] NOA_FHD constexpr auto flip() const noexcept -> Vec {
            if constexpr (SIZE == 0) {
                return {};
            } else {
                Vec output;
                for (usize i{}; i < SIZE; ++i)
                    output[i] = (*this)[(N - 1) - i];
                return output;
            }
        }

        template<nt::integer I = std::conditional_t<nt::integer<value_type>, value_type, isize>, usize AR = 0>
        [[nodiscard]] NOA_FHD constexpr auto permute(const Vec<I, SIZE, AR>& permutation) const noexcept -> Vec {
            if constexpr (SIZE == 0) {
                return {};
            } else {
                Vec output;
                for (usize i{}; i < SIZE; ++i)
                    output[i] = (*this)[permutation[i]];
                return output;
            }
        }

        // Circular shifts the vector by a given amount.
        // If "count" is positive, shift to the right, otherwise, shift to the left.
        [[nodiscard]] NOA_FHD constexpr auto circular_shift(isize count) -> Vec {
            if constexpr (SIZE <= 1) {
                return *this;
            } else {
                Vec out;
                const bool right = count >= 0;
                if (not right)
                    count *= -1;
                for (isize i{}; i < SSIZE; ++i) {
                    const isize idx = (i + count) % SSIZE;
                    out[idx * right + (1 - right) * i] = array[i * right + (1 - right) * idx];
                }
                return out;
            }
        }

        [[nodiscard]] NOA_FHD constexpr auto copy() const noexcept -> Vec {
            return *this;
        }

        template<usize INDEX> requires (INDEX < SIZE)
        [[nodiscard]] NOA_FHD constexpr auto set(const value_type& value) const noexcept -> Vec {
            auto output = *this;
            output[INDEX] = value;
            return output;
        }
    };

    /// Deduction guide.
    template<typename T, typename... U>
    Vec(T, U...) -> Vec<std::enable_if_t<(std::same_as<T, U> and ...), T>, 1 + sizeof...(U)>;

    /// Support for output stream:
    template<typename T, usize N, usize A>
    auto operator<<(std::ostream& os, const Vec<T, N, A>& v) -> std::ostream& {
        if constexpr (nt::real_or_complex<T>)
            os << fmt::format("{::.3f}", v); // {fmt} ranges
        else
            os << fmt::format("{}", v); // FIXME
        return os;
    }
}

// Support for structure bindings:
namespace std {
    template<typename T, noa::usize N, noa::usize A>
    struct tuple_size<noa::Vec<T, N, A>> : std::integral_constant<noa::usize, N> {};

    template<typename T, noa::usize N, noa::usize A>
    struct tuple_size<const noa::Vec<T, N, A>> : std::integral_constant<noa::usize, N> {};

    template<noa::usize I, noa::usize N, noa::usize A, typename T>
    struct tuple_element<I, noa::Vec<T, N, A>> { using type = T; };

    template<noa::usize I, noa::usize N, noa::usize A, typename T>
    struct tuple_element<I, const noa::Vec<T, N, A>> { using type = const T; };
}

namespace noa::traits {
    template<typename T, usize N, usize A> struct proclaim_is_vec<noa::Vec<T, N, A>> : std::true_type {};
    template<typename V1, usize N, usize A, typename V2> struct proclaim_is_vec_of_type<noa::Vec<V1, N, A>, V2> : std::bool_constant<std::is_same_v<V1, V2>> {};
    template<typename V, usize N1, usize A, usize N2> struct proclaim_is_vec_of_size<noa::Vec<V, N1, A>, N2> : std::bool_constant<N1 == N2> {};
}

namespace noa {
    template<typename T, usize N, usize A> requires (nt::boolean<T> or nt::vec<T>)
    [[nodiscard]] NOA_HD constexpr auto operator!(Vec<T, N, A> vector) noexcept {
        if constexpr (N > 0) {
            for (usize i{}; i < N; ++i)
                vector[i] = !vector[i];
        }
        return vector;
    }

    template<typename T, usize N, usize A> requires (nt::boolean<T> or nt::vec<T>)
    [[nodiscard]] NOA_HD constexpr auto operator&&(Vec<T, N, A> lhs, const Vec<T, N, A>& rhs) noexcept {
        if constexpr (N > 0) {
            for (usize i{}; i < N; ++i)
                lhs[i] = lhs[i] && rhs[i];
        }
        return lhs;
    }

    template<typename T, usize N, usize A> requires (nt::boolean<T> or nt::vec<T>)
    [[nodiscard]] NOA_HD constexpr auto operator||(Vec<T, N, A> lhs, const Vec<T, N, A>& rhs) noexcept {
        if constexpr (N > 0) {
            for (usize i{}; i < N; ++i)
                lhs[i] = lhs[i] || rhs[i];
        }
        return lhs;
    }

    template<nt::vec_integer V>
    [[nodiscard]] NOA_HD constexpr V operator%(V lhs, const V& rhs) noexcept {
        if constexpr (V::SSIZE > 0) {
            for (usize i{}; i < V::SIZE; ++i)
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
    [[nodiscard]] NOA_FHD constexpr auto cos(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Cos>(v); }
    [[nodiscard]] NOA_FHD constexpr auto sin(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Sin>(v); }
    [[nodiscard]] NOA_FHD constexpr auto exp(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Exp>(v); }
    [[nodiscard]] NOA_FHD constexpr auto log(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Log>(v); }
    [[nodiscard]] NOA_FHD constexpr auto log10(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Log10>(v); }
    [[nodiscard]] NOA_FHD constexpr auto log1p(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Log1p>(v); }

    [[nodiscard]] NOA_FHD constexpr auto sqrt(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Sqrt>(v); }
    [[nodiscard]] NOA_FHD constexpr auto rsqrt(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Rsqrt>(v); }
    [[nodiscard]] NOA_FHD constexpr auto round(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Round>(v); }
    [[nodiscard]] NOA_FHD constexpr auto rint(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Rint>(v); }
    [[nodiscard]] NOA_FHD constexpr auto ceil(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Ceil>(v); }
    [[nodiscard]] NOA_FHD constexpr auto floor(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Floor>(v); }
    [[nodiscard]] NOA_FHD constexpr auto abs(nt::vec_scalar_or_vec auto const& v) noexcept { return nd::vec_func<Abs>(v); }

    [[nodiscard]] NOA_FHD constexpr auto sinc(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Sinc>(v); }
    [[nodiscard]] NOA_FHD constexpr auto tan(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Tan>(v); }
    [[nodiscard]] NOA_FHD constexpr auto acos(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Acos>(v); }
    [[nodiscard]] NOA_FHD constexpr auto asin(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Asin>(v); }
    [[nodiscard]] NOA_FHD constexpr auto atan(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Atan>(v); }
    [[nodiscard]] NOA_FHD constexpr auto acosh(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Acosh>(v); }
    [[nodiscard]] NOA_FHD constexpr auto asinh(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Asinh>(v); }
    [[nodiscard]] NOA_FHD constexpr auto atanh(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Atan>(v); }
    [[nodiscard]] NOA_FHD constexpr auto atan2(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Atan2>(v); }

    [[nodiscard]] NOA_FHD constexpr auto rad2deg(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Rad2Deg>(v); }
    [[nodiscard]] NOA_FHD constexpr auto deg2rad(nt::vec_real_or_vec auto const& v) noexcept { return nd::vec_func<Deg2Rad>(v); }
}

namespace noa {
    template<typename To, typename From, usize N, usize A> requires nt::vec_of_size<To, N>
    [[nodiscard]] NOA_HD constexpr bool is_safe_cast(const Vec<From, N, A>& src) noexcept {
        if constexpr (N == 0) {
            return true;
        } else {
            return [&src]<usize...I>(std::index_sequence<I...>) {
                return (is_safe_cast<typename To::value_type>(src[I]) and ...);
            }(std::make_index_sequence<N>{});
        }
    }

    template<typename To, typename From, usize N, usize A> requires nt::vec_of_size<To, N>
    [[nodiscard]] NOA_HD constexpr auto clamp_cast(const Vec<From, N, A>& src) noexcept -> To {
        if constexpr (N == 0) {
            return {};
        } else {
            To output;
            for (usize i{}; i < N; ++i)
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

    template<typename T, usize N, usize A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto sum(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::same_as<T, f16>)
            return sum(vector.template as<typename T::arithmetic_type>()).template as<T, A>();

        return [&vector]<usize...I>(std::index_sequence<I...>) {
            return (... + vector[I]);
        }(std::make_index_sequence<N>{});
    }

    template<typename T, usize N, usize A>
    [[nodiscard]] NOA_HD constexpr auto mean(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::same_as<T, f16>)
            return mean(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        return sum(vector) / 2;
    }

    template<typename T, usize N, usize A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto product(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::same_as<T, f16>)
            return product(vector.template as<typename T::arithmetic_type>()).template as<T, A>();

        return [&vector]<usize...I>(std::index_sequence<I...>) {
            return (... * vector[I]);
        }(std::make_index_sequence<N>{});
    }

    template<typename T, usize N, usize A0, usize A1> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr auto dot(const Vec<T, N, A0>& lhs, const Vec<T, N, A1>& rhs) noexcept {
        if constexpr (std::same_as<T, f16>) {
            return static_cast<T>(dot(lhs.template as<typename T::arithmetic_type>(),
                                      rhs.template as<typename T::arithmetic_type>()));
        }

        return [&lhs, &rhs]<usize...I>(std::index_sequence<I...>) {
            return (... + (lhs[I] * rhs[I]));
        }(std::make_index_sequence<N>{});
    }

    template<typename T, usize N, usize A> requires ((nt::real<T> or nt::vec<T>) and (N > 0))
    [[nodiscard]] NOA_HD constexpr auto norm(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::same_as<T, f16>) {
            const auto tmp = vector.template as<typename T::arithmetic_type>();
            return norm(tmp).template as<T, A>();
        }

        return sqrt(dot(vector, vector)); // euclidean norm
    }

    template<typename T, usize N, usize A> requires (nt::real<T> or nt::vec<T>)
    [[nodiscard]] NOA_HD constexpr auto normalize(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::same_as<T, f16>) {
            const auto tmp = vector.template as<typename T::arithmetic_type>();
            return normalize(tmp).template as<T, A>();
        }

        return vector / norm(vector); // may divide by 0
    }

    template<nt::scalar T, usize A>
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

    template<typename T, usize N, usize A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr T min(
        const Vec<T, N, A>& vector
    ) noexcept {
        if constexpr (N == 1) {
            return vector[0];
        } else {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (std::same_as<T, f16> && N == 4) {
                auto* alias = reinterpret_cast<const __half2*>(vector.data());
                const __half2 tmp = __hmin2(alias[0], alias[1]);
                return __hmin(tmp.x, tmp.y);
            } else if constexpr (std::same_as<T, f16> && N == 8) {
                auto* alias = reinterpret_cast<const __half2*>(vector.data());
                const __half2 tmp0 = __hmin2(alias[0], alias[1]);
                const __half2 tmp1 = __hmin2(alias[2], alias[3]);
                const __half2 tmp2 = __hmin2(tmp0, tmp1);
                return __hmin(tmp2.x, tmp2.y);
            }
            #endif
            auto element = min(vector[0], vector[1]);
            for (usize i = 2; i < N; ++i)
                element = min(element, vector[i]);
            return element;
        }
    }

    template<typename T, usize N, usize A>
    [[nodiscard]] NOA_HD constexpr auto min(
        Vec<T, N, A> lhs,
        const Vec<T, N, A>& rhs
    ) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::same_as<T, f16> and is_even(N)) {
            auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
            auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
            for (usize i{}; i < N / 2; ++i)
                alias0[i] = __hmin2(alias0[i], alias1[i]);
            return lhs;
        }
        #endif
        for (usize i{}; i < N; ++i)
            lhs[i] = min(lhs[i], rhs[i]);
        return lhs;
    }

    template<typename T, usize N, usize A> requires (N > 0)
    [[nodiscard]] NOA_HD constexpr T max(
        const Vec<T, N, A>& vector
    ) noexcept {
        if constexpr (N == 1) {
            return vector[0];
        } else {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (std::same_as<T, f16> && N == 4) {
                auto* alias = reinterpret_cast<const __half2*>(vector.data());
                const __half2 tmp = __hmax2(alias[0], alias[1]);
                return __hmax(tmp.x, tmp.y);
            } else if constexpr (std::same_as<T, f16> && N == 8) {
                auto* alias = reinterpret_cast<const __half2*>(vector.data());
                const __half2 tmp0 = __hmax2(alias[0], alias[1]);
                const __half2 tmp1 = __hmax2(alias[2], alias[3]);
                const __half2 tmp2 = __hmax2(tmp0, tmp1);
                return __hmax(tmp2.x, tmp2.y);
            }
            #endif
            auto element = max(vector[0], vector[1]);
            for (usize i = 2; i < N; ++i)
                element = max(element, vector[i]);
            return element;
        }
    }

    template<typename T, usize N, usize A>
    [[nodiscard]] NOA_HD constexpr auto max(
        Vec<T, N, A> lhs,
        const Vec<T, N, A>& rhs
    ) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::same_as<T, f16> and is_even(N)) {
            auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
            auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
            for (usize i{}; i < N / 2; ++i)
                alias0[i] = __hmax2(alias0[i], alias1[i]);
            return lhs;
        }
        #endif
        for (usize i{}; i < N; ++i)
            lhs[i] = max(lhs[i], rhs[i]);
        return lhs;
    }

    /// Computes the argmax. Returns first occurrence if equal values are present.
    template<typename I = usize, nt::numeric T, usize N, usize A> requires (N > 0)
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
    template<typename I = usize, nt::numeric T, usize N, usize A> requires (N > 0)
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

    template<typename T, usize N, usize A>
    [[nodiscard]] NOA_FHD constexpr auto min(const Vec<T, N, A>& lhs, std::type_identity_t<T> rhs) noexcept {
        return min(lhs, Vec<T, N, A>::filled_with(rhs));
    }

    template<typename T, usize N, usize A>
    [[nodiscard]] NOA_FHD constexpr auto min(std::type_identity_t<T> lhs, const Vec<T, N, A>& rhs) noexcept {
        return min(Vec<T, N, A>::filled_with(lhs), rhs);
    }

    template<typename T, usize N, usize A>
    [[nodiscard]] NOA_FHD constexpr auto max(const Vec<T, N, A>& lhs, std::type_identity_t<T> rhs) noexcept {
        return max(lhs, Vec<T, N, A>::filled_with(rhs));
    }

    template<typename T, usize N, usize A>
    [[nodiscard]] NOA_FHD constexpr auto max(std::type_identity_t<T> lhs, const Vec<T, N, A>& rhs) noexcept {
        return max(Vec<T, N, A>::filled_with(lhs), rhs);
    }

    template<typename T, usize N, usize A>
    [[nodiscard]] NOA_FHD constexpr auto clamp(
            const Vec<T, N, A>& lhs,
            const Vec<T, N, A>& low,
            const Vec<T, N, A>& high
    ) noexcept {
        return min(max(lhs, low), high);
    }

    template<typename T, usize N, usize A>
    [[nodiscard]] NOA_FHD constexpr auto clamp(
            const Vec<T, N, A>& lhs,
            std::type_identity_t<T> low,
            std::type_identity_t<T> high
    ) noexcept {
        return min(max(lhs, low), high);
    }

    template<i32 ULP = 2, nt::numeric T, usize N, usize A, typename U = nt::value_type_t<T>>
    [[nodiscard]] NOA_HD constexpr auto allclose(
        const Vec<T, N, A>& lhs,
        const Vec<T, N, A>& rhs,
        std::type_identity_t<U> epsilon = static_cast<U>(1e-6)
    ) -> bool {
        for (usize i{}; i < N; ++i)
            if (not allclose<ULP>(lhs[i], rhs[i], epsilon))
                return false;
        return true;
    }

    template<i32 ULP = 2, nt::numeric T, usize N, usize A, typename U = nt::value_type_t<T>>
    [[nodiscard]] NOA_FHD constexpr auto allclose(
        const Vec<T, N, A>& lhs,
        std::type_identity_t<T> rhs,
        std::type_identity_t<U> epsilon = static_cast<U>(1e-6)
    ) noexcept -> bool {
        return allclose<ULP>(lhs, Vec<T, N, A>::filled_with(rhs), epsilon);
    }

    template<i32 ULP = 2, nt::numeric T, usize N, usize A, typename U = nt::value_type_t<T>>
    [[nodiscard]] NOA_FHD constexpr auto allclose(
        std::type_identity_t<T> lhs,
        const Vec<T, N, A>& rhs,
        std::type_identity_t<U> epsilon = static_cast<U>(1e-6)
    ) noexcept -> bool {
        return allclose<ULP>(Vec<T, N, A>::filled_with(lhs), rhs, epsilon);
    }

    template<typename T, usize N, usize A, typename Op = Less> requires (N <= 4)
    [[nodiscard]] constexpr auto stable_sort(Vec<T, N, A> vector, Op&& comp = {}) noexcept {
        details::small_stable_sort<N>(vector.data(), std::forward<Op>(comp));
        return vector;
    }

    template<typename T, usize N, usize A, typename Op = Less> requires (N <= 4)
    [[nodiscard]] constexpr auto sort(Vec<T, N, A> vector, Op&& comp = {}) noexcept {
        details::small_stable_sort<N>(vector.data(), std::forward<Op>(comp));
        return vector;
    }

    template<typename T, usize N, usize A, typename Op = Less> requires (N > 4)
    [[nodiscard]] auto stable_sort(Vec<T, N, A> vector, Op&& comp = {}) noexcept {
        std::stable_sort(vector.begin(), vector.end(), std::forward<Op>(comp));
        return vector;
    }

    template<typename T, usize N, usize A, typename Op = Less> requires (N > 4)
    [[nodiscard]] auto sort(Vec<T, N, A> vector, Op&& comp = {}) noexcept {
        std::sort(vector.begin(), vector.end(), std::forward<Op>(comp));
        return vector;
    }
}

namespace noa::details {
    template<typename T, usize N>
    struct Stringify<Vec<T, N>> {
        static auto get() -> std::string {
            return fmt::format("Vec<{},{}>", nd::stringify<T>(), N);
        }
    };
}
