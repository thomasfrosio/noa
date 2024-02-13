#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/utils/ClampCast.hpp"
#include "noa/core/utils/SafeCast.hpp"
#include "noa/core/utils/Sort.hpp"

#if defined(NOA_IS_OFFLINE)
#include <algorithm>
#endif

namespace noa::guts {
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
    /// Similar to std::array<Value, N>, but restricted to "numerics", with a lot more functionalities.
    /// \note By default (A=0), the alignment is less or equal than sizeof(T) * N. The alignment can also be set explicitly.
    template<typename T, size_t N, size_t A = 0>
    class alignas(guts::VecAlignment<T, N, A>::value) Vec {
    public:
        static_assert(nt::is_numeric_v<T>, "Only numeric types are supported");
        static_assert(!std::is_const_v<T>, "The value type must be mutable. Const-ness is enforced by Vec");
        static_assert(!std::is_reference_v<T>, "The value type must be a value");

        using storage_type = guts::VecStorage<T, N>;
        using array_type = storage_type::type;
        using value_type = T;
        using mutable_value_type = value_type;
        static constexpr int64_t SSIZE = N;
        static constexpr size_t SIZE = N;

    public:
        NOA_NO_UNIQUE_ADDRESS array_type array;

    public: // Static factory functions
        template<typename U> requires nt::is_numeric_v<U>
        [[nodiscard]] NOA_HD static constexpr Vec from_value(U value) noexcept {
            NOA_ASSERT(is_safe_cast<value_type>(value));
            Vec vec;
            if constexpr (SIZE > 0) {
                const auto value_cast = static_cast<value_type>(value);
                for (int64_t i = 0; i < SSIZE; ++i)
                    vec[i] = value_cast;
            }
            return vec;
        }

        template<typename U> requires nt::is_numeric_v<U>
        [[nodiscard]] NOA_HD static constexpr Vec filled_with(U value) noexcept {
            return from_value(value); // filled_with is a better name, but keep from_value for consistency
        }

        template<typename... Args> requires (sizeof...(Args) == SIZE && nt::are_numeric_v<Args...>)
        [[nodiscard]] NOA_HD static constexpr Vec from_values(Args... values) noexcept {
            NOA_ASSERT((is_safe_cast<value_type>(values) && ...));
            return {static_cast<value_type>(values)...};
        }

        template<typename U, size_t AR>
        [[nodiscard]] NOA_HD static constexpr Vec from_vector(const Vec<U, SIZE, AR>& vector) noexcept {
            Vec vec;
            if constexpr (SIZE > 0) {
                for (int64_t i = 0; i < SSIZE; ++i) {
                    NOA_ASSERT(is_safe_cast<value_type>(vector[i]));
                    vec[i] = static_cast<value_type>(vector[i]);
                }
            }
            return vec;
        }

        template<typename U> requires nt::is_numeric_v<U>
        [[nodiscard]] NOA_HD static constexpr Vec from_pointer(const U* values) noexcept {
            Vec vec;
            if constexpr (SIZE > 0) {
                for (int64_t i = 0; i < SSIZE; ++i)
                    vec[i] = static_cast<value_type>(values[i]);
            }
            return vec;
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Vec<U>>(Vec<T>{}).
        template<typename U, size_t AR>
        [[nodiscard]] NOA_HD constexpr explicit operator Vec<U, SIZE, AR>() const noexcept {
            return Vec<U, SIZE, AR>::from_vector(*this);
        }

    public: // Accessor operators and functions
        template<typename Int> requires (std::is_integral_v<Int> and SIZE > 0)
        [[nodiscard]] NOA_HD constexpr value_type& operator[](Int i) noexcept {
            NOA_ASSERT(static_cast<int64_t>(i) < SSIZE);
            if constexpr (std::is_signed_v<Int>) {
                NOA_ASSERT(i >= 0);
            }
            return storage_type::ref(array, i);
        }

        template<typename Int> requires (std::is_integral_v<Int> and SIZE > 0)
        [[nodiscard]] NOA_HD constexpr const value_type& operator[](Int i) const noexcept {
            NOA_ASSERT(static_cast<int64_t>(i) < SSIZE);
            if constexpr (std::is_signed_v<Int>) {
                NOA_ASSERT(i >= 0);
            }
            return storage_type::ref(array, i);
        }

        // Structure binding support.
        // Note that this relies on the tuple-like binding, which appears to not be supported by nvrtc.
        template<int I> [[nodiscard]] NOA_HD constexpr const value_type& get() const noexcept { return (*this)[I]; }
        template<int I> [[nodiscard]] NOA_HD constexpr value_type& get() noexcept { return (*this)[I]; }

        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return storage_type::ptr(array); }
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return storage_type::ptr(array); }
        [[nodiscard]] NOA_HD constexpr size_t size() const noexcept { return SIZE; };
        [[nodiscard]] NOA_HD constexpr int64_t ssize() const noexcept { return SSIZE; };

    public: // Iterators -- support for range loops
        [[nodiscard]] NOA_HD constexpr value_type* begin() noexcept { return data(); }
        [[nodiscard]] NOA_HD constexpr const value_type* begin() const noexcept { return data(); }
        [[nodiscard]] NOA_HD constexpr const value_type* cbegin() const noexcept { return data(); }
        [[nodiscard]] NOA_HD constexpr value_type* end() noexcept { return data() + SSIZE; }
        [[nodiscard]] NOA_HD constexpr const value_type* end() const noexcept { return data() + SSIZE; }
        [[nodiscard]] NOA_HD constexpr const value_type* cend() const noexcept { return data() + SSIZE; }

    public: // Assignment operators
        NOA_HD constexpr Vec& operator=(value_type value) noexcept {
            *this = Vec::filled_with(value);
            return *this;
        }

        NOA_HD constexpr Vec& operator+=(const Vec& vector) noexcept {
            *this = *this + vector;
            return *this;
        }

        NOA_HD constexpr Vec& operator-=(const Vec& vector) noexcept {
            *this = *this - vector;
            return *this;
        }

        NOA_HD constexpr Vec& operator*=(const Vec& vector) noexcept {
            *this = *this * vector;
            return *this;
        }

        NOA_HD constexpr Vec& operator/=(const Vec& vector) noexcept {
            *this = *this / vector;
            return *this;
        }

        NOA_HD constexpr Vec& operator+=(value_type value) noexcept {
            *this = *this + value;
            return *this;
        }

        NOA_HD constexpr Vec& operator-=(value_type value) noexcept {
            *this = *this - value;
            return *this;
        }

        NOA_HD constexpr Vec& operator*=(value_type value) noexcept {
            *this = *this * value;
            return *this;
        }

        NOA_HD constexpr Vec& operator/=(value_type value) noexcept {
            *this = *this / value;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Vec operator+(const Vec& v) noexcept {
            return v;
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator-(Vec v) noexcept {
            if constexpr (SIZE > 0) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
                if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                    auto* alias = reinterpret_cast<__half2*>(v.data());
                    for (int64_t i = 0; i < SSIZE / 2; ++i)
                        alias[i] = -alias[i];
                    return v;
                }
                #endif
                for (int64_t i = 0; i < SSIZE; ++i)
                    v[i] = -v[i];
                return v;
            } else {
                return v;
            }
        }

        // -- Binary Arithmetic Operators --
        [[nodiscard]] friend NOA_HD constexpr Vec operator+(Vec lhs, Vec rhs) noexcept {
            if constexpr (SIZE > 0) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
                if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                    auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
                    auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
                    for (int64_t i = 0; i < SSIZE / 2; ++i)
                        alias0[i] += alias1[i];
                    return lhs;
                }
                #endif
                for (int64_t i = 0; i < SSIZE; ++i)
                    lhs[i] += rhs[i];
                return lhs;
            } else {
                return lhs;
            }
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator+(const Vec& lhs, value_type rhs) noexcept {
            return lhs + Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator+(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) + rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator-(Vec lhs, Vec rhs) noexcept {
            if constexpr (SIZE > 0) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
                if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                    auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
                    auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
                    for (int64_t i = 0; i < SSIZE / 2; ++i)
                        alias0[i] -= alias1[i];
                    return lhs;
                }
                #endif
                for (int64_t i = 0; i < SSIZE; ++i)
                    lhs[i] -= rhs[i];
                return lhs;
            } else {
                return lhs;
            }
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator-(const Vec& lhs, value_type rhs) noexcept {
            return lhs - Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator-(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) - rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator*(Vec lhs, Vec rhs) noexcept {
            if constexpr (SIZE > 0) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
                if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                    auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
                    auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
                    for (int64_t i = 0; i < SSIZE / 2; ++i)
                        alias0[i] *= alias1[i];
                    return lhs;
                }
                #endif
                for (int64_t i = 0; i < SSIZE; ++i)
                    lhs[i] *= rhs[i];
                return lhs;
            } else {
                return lhs;
            }
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator*(const Vec& lhs, value_type rhs) noexcept {
            return lhs * Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator*(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) * rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator/(Vec lhs, Vec rhs) noexcept {
            if constexpr (SIZE > 0) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
                if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                    auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
                    auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
                    for (int64_t i = 0; i < SSIZE / 2; ++i)
                        alias0[i] /= alias1[i];
                    return lhs;
                }
                #endif
                for (int64_t i = 0; i < SSIZE; ++i)
                    lhs[i] /= rhs[i];
                return lhs;
            } else {
                return lhs;
            }
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator/(const Vec& lhs, value_type rhs) noexcept {
            return lhs / Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator/(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) / rhs;
        }

        // -- Comparison Operators --
        [[nodiscard]] friend NOA_HD constexpr auto operator>(Vec lhs, Vec rhs) noexcept {
            if constexpr (SIZE > 0) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
                if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                    auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
                    auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
                    Vec<bool, N> output;
                    #pragma unroll
                    for (int64_t i = 0; i < SSIZE / 2; ++i) {
                        alias0[i] = __hgt2(alias0[i], alias1[i]);
                        output[i * 2 + 0] = static_cast<bool>(alias0[i].x);
                        output[i * 2 + 1] = static_cast<bool>(alias0[i].y);
                    }
                    return output;
                }
                #endif
                Vec<bool, N> output;
                for (int64_t i = 0; i < SSIZE; ++i)
                    output[i] = lhs[i] > rhs[i];
                return output;
            } else {
                return Vec<bool, 0>{};
            }
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>(const Vec& lhs, value_type rhs) noexcept {
            return lhs > Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) > rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(Vec lhs, Vec rhs) noexcept {
            if constexpr (SIZE > 0) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
                if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                    auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
                    auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
                    Vec<bool, N> output;
                    #pragma unroll
                    for (int64_t i = 0; i < SSIZE / 2; ++i) {
                        alias0[i] = __hlt2(alias0[i], alias1[i]);
                        output[i * 2 + 0] = static_cast<bool>(alias0[i].x);
                        output[i * 2 + 1] = static_cast<bool>(alias0[i].y);
                    }
                    return output;
                }
                #endif
                Vec<bool, N> output;
                for (int64_t i = 0; i < SSIZE; ++i)
                    output[i] = lhs[i] < rhs[i];
                return output;
            } else {
                return Vec<bool, 0>{};
            }
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(const Vec& lhs, value_type rhs) noexcept {
            return lhs < Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) < rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(Vec lhs, Vec rhs) noexcept {
            if constexpr (SIZE > 0) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
                if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                    auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
                    auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
                    Vec<bool, N> output;
                    #pragma unroll
                    for (int64_t i = 0; i < SSIZE / 2; ++i) {
                        alias0[i] = __hge2(alias0[i], alias1[i]);
                        output[i * 2 + 0] = static_cast<bool>(alias0[i].x);
                        output[i * 2 + 1] = static_cast<bool>(alias0[i].y);
                    }
                    return output;
                }
                #endif
                Vec<bool, N> output;
                for (int64_t i = 0; i < SSIZE; ++i)
                    output[i] = lhs[i] >= rhs[i];
                return output;
            } else {
                return Vec<bool, 0>{};
            }
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(const Vec& lhs, value_type rhs) noexcept {
            return lhs >= Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) >= rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(Vec lhs, Vec rhs) noexcept {
            if constexpr (SIZE > 0) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
                if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                    auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
                    auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
                    Vec<bool, N> output;
                    #pragma unroll
                    for (int64_t i = 0; i < SSIZE / 2; ++i) {
                        alias0[i] = __hle2(alias0[i], alias1[i]);
                        output[i * 2 + 0] = static_cast<bool>(alias0[i].x);
                        output[i * 2 + 1] = static_cast<bool>(alias0[i].y);
                    }
                    return output;
                }
                #endif
                Vec<bool, N> output;
                for (int64_t i = 0; i < SSIZE; ++i)
                    output[i] = lhs[i] <= rhs[i];
                return output;
            } else {
                return Vec<bool, N>{};
            }
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(const Vec& lhs, value_type rhs) noexcept {
            return lhs <= Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) <= rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(Vec lhs, Vec rhs) noexcept {
            if constexpr (SIZE > 0) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
                if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                    auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
                    auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
                    Vec<bool, N> output;
                    #pragma unroll
                    for (int64_t i = 0; i < SSIZE / 2; ++i) {
                        alias0[i] = __heq2(alias0[i], alias1[i]);
                        output[i * 2 + 0] = static_cast<bool>(alias0[i].x);
                        output[i * 2 + 1] = static_cast<bool>(alias0[i].y);
                    }
                    return output;
                }
                #endif
                Vec<bool, N> output;
                for (int64_t i = 0; i < SSIZE; ++i)
                    output[i] = lhs[i] == rhs[i];
                return output;
            } else {
                return Vec<bool, N>{};
            }
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(const Vec& lhs, value_type rhs) noexcept {
            return lhs == Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) == rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(Vec lhs, Vec rhs) noexcept {
            if constexpr (SIZE > 0) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
                if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                    auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
                    auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
                    Vec<bool, N> output;
                    #pragma unroll
                    for (int64_t i = 0; i < SSIZE / 2; ++i) {
                        alias0[i] = __hne2(alias0[i], alias1[i]);
                        output[i * 2 + 0] = static_cast<bool>(alias0[i].x);
                        output[i * 2 + 1] = static_cast<bool>(alias0[i].y);
                    }
                    return output;
                }
                #endif
                Vec<bool, N> output;
                for (int64_t i = 0; i < SSIZE; ++i)
                    output[i] = lhs[i] != rhs[i];
                return output;
            } else {
                return Vec<bool, N>{};
            }
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(const Vec& lhs, value_type rhs) noexcept {
            return lhs != Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) != rhs;
        }

    public: // Type casts
        template<typename U, size_t AR = 0> requires nt::is_numeric_v<U>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return static_cast<Vec<U, SIZE, AR>>(*this);
        }

        template<typename U, size_t AR = 0> requires nt::is_numeric_v<U>
        [[nodiscard]] NOA_HD constexpr auto as_clamp() const noexcept {
            return clamp_cast<Vec<U, SIZE, AR>>(*this);
        }

#if defined(NOA_IS_OFFLINE)
        template<typename U, size_t AR = 0> requires nt::is_numeric_v<U>
        [[nodiscard]] constexpr auto as_safe() const {
            return safe_cast<Vec<U, SIZE, AR>>(*this);
        }
#endif

    public:
        template<size_t S = 1, size_t AR = 0> requires (N >= S)
        [[nodiscard]] NOA_HD constexpr auto pop_front() const noexcept {
            return Vec<value_type, N - S, AR>::from_pointer(data() + S);
        }

        template<size_t S = 1, size_t AR = 0> requires (N >= S)
        [[nodiscard]] NOA_HD constexpr auto pop_back() const noexcept {
            return Vec<value_type, N - S, AR>::from_pointer(data());
        }

        [[nodiscard]] NOA_HD constexpr auto push_front(value_type value) const noexcept {
            Vec<value_type, N + 1> output;
            output[0] = value;
            if constexpr (N > 0) {
                for (size_t i = 0; i < N; ++i)
                    output[i + 1] = array[i];
            }
            return output;
        }

        [[nodiscard]] NOA_HD constexpr auto push_back(value_type value) const noexcept {
            Vec<value_type, N + 1> output;
            if constexpr (N > 0) {
                for (size_t i = 0; i < N; ++i)
                    output[i] = array[i];
            }
            output[N] = value;
            return output;
        }

        template<size_t S, size_t AR>
        [[nodiscard]] NOA_HD constexpr auto push_front(const Vec<value_type, S, AR>& vector) const noexcept {
            Vec<value_type, N + S> output;
            if constexpr (S > 0) {
                for (size_t i = 0; i < S; ++i)
                    output[i] = vector[i];
            }
            if constexpr (N > 0) {
                for (size_t i = 0; i < N; ++i)
                    output[i + S] = array[i];
            }
            return output;
        }

        template<size_t S, size_t AR>
        [[nodiscard]] NOA_HD constexpr auto push_back(const Vec<value_type, S, AR>& vector) const noexcept {
            Vec<value_type, N + S> output;
            if constexpr (N > 0) {
                for (size_t i = 0; i < N; ++i)
                    output[i] = array[i];
            }
            if constexpr (S > 0) {
                for (size_t i = 0; i < S; ++i)
                    output[i + N] = vector[i];
            }
            return output;
        }

        template<typename... Indexes> requires nt::are_int_v<Indexes...>
        [[nodiscard]] NOA_HD constexpr auto filter(Indexes... indexes) const noexcept {
            // TODO This can do a lot more than "filter". Rename?
            return Vec<value_type, sizeof...(Indexes)>{(*this)[indexes]...};
        }

        [[nodiscard]] NOA_HD constexpr Vec flip() const noexcept {
            Vec output;
            if constexpr (SIZE > 0) {
                for (size_t i = 0; i < SIZE; ++i)
                    output[i] = array[(N - 1) - i];
            }
            return output;
        }

        template<typename Int = std::conditional_t<nt::is_int_v<value_type>, value_type, int64_t>, size_t AR = 0>
        requires nt::is_int_v<Int>
        [[nodiscard]] NOA_HD constexpr Vec reorder(const Vec<Int, SIZE, AR>& order) const noexcept {
            Vec output;
            if constexpr (SIZE > 0) {
                for (size_t i = 0; i < SIZE; ++i)
                    output[i] = array[order[i]];
            }
            return output;
        }

        // Circular shifts the vector by a given amount.
        // If "count" is positive, shift to the right, otherwise, shift to the left.
        [[nodiscard]] NOA_HD constexpr Vec circular_shift(int64_t count) {
            if constexpr (SIZE <= 1) {
                return *this;
            } else {
                Vec out;
                const bool right = count >= 0;
                if (!right)
                    count *= -1;
                for (int64_t i = 0; i < SSIZE; ++i) {
                    const int64_t idx = (i + count) % SSIZE;
                    out[idx * right + (1 - right) * i] = array[i * right + (1 - right) * idx];
                }
                return out;
            }
        }

        [[nodiscard]] NOA_HD constexpr Vec copy() const noexcept {
            return *this;
        }

        template<size_t INDEX> requires (INDEX < SIZE)
        [[nodiscard]] NOA_HD constexpr Vec set(value_type value) const noexcept {
            auto output = *this;
            output[INDEX] = value;
            return output;
        }

#if defined(NOA_IS_OFFLINE)
    public:
        [[nodiscard]] static std::string name() {
            return fmt::format("Vec<{},{}>", ns::to_human_readable<value_type>(), SIZE);
        }
#endif
    };

    /// Deduction guide.
    template<typename T, typename... U>
    Vec(T, U...) -> Vec<std::enable_if_t<(std::is_same_v<T, U> and ...), T>, 1 + sizeof...(U)>;

    /// Type aliases.
    template<typename T> using Vec1 = Vec<T, 1>;
    template<typename T> using Vec2 = Vec<T, 2>;
    template<typename T> using Vec3 = Vec<T, 3>;
    template<typename T> using Vec4 = Vec<T, 4>;

#if defined(NOA_IS_OFFLINE)
    /// Support for output stream:
    template<typename T, size_t N, size_t A>
    inline std::ostream& operator<<(std::ostream& os, const Vec<T, N, A>& v) {
        if constexpr (nt::is_real_or_complex_v<T>)
            os << fmt::format("{::.3f}", v); // {fmt} ranges
        else
            os << fmt::format("{}", v);
        return os;
    }
#endif
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
    template<typename T, size_t N, size_t A> struct proclaim_is_vec<Vec<T, N, A>> : std::true_type {};
    template<typename V1, size_t N, size_t A, typename V2> struct proclaim_is_vec_of_type<Vec<V1, N, A>, V2> : std::bool_constant<std::is_same_v<V1, V2>> {};
    template<typename V, size_t N1, size_t A, size_t N2> struct proclaim_is_vec_of_size<Vec<V, N1, A>, N2> : std::bool_constant<N1 == N2> {};
}

namespace noa::inline types {
    template<size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto operator!(Vec<bool, N, A> vector) noexcept {
        if constexpr (N > 0) {
            for (size_t i = 0; i < N; ++i)
                vector[i] = !vector[i];
        }
        return vector;
    }

    template<size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto operator&&(Vec<bool, N, A> lhs, const Vec<bool, N, A>& rhs) noexcept {
        if constexpr (N > 0) {
            for (size_t i = 0; i < N; ++i)
                lhs[i] = lhs[i] && rhs[i];
        }
        return lhs;
    }

    template<size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto operator||(Vec<bool, N, A> lhs, const Vec<bool, N, A>& rhs) noexcept {
        if constexpr (N > 0) {
            for (size_t i = 0; i < N; ++i)
                lhs[i] = lhs[i] || rhs[i];
        }
        return lhs;
    }

    // -- Modulo Operator --
    template<typename Vec> requires nt::is_int_v<Vec>
    [[nodiscard]] NOA_HD constexpr Vec operator%(Vec lhs, const Vec& rhs) noexcept {
        if constexpr (Vec::SSIZE > 0) {
            for (int64_t i = 0; i < Vec::SSIZE; ++i)
                lhs[i] %= rhs[i];
        }
        return lhs;
    }

    template<typename Vec, typename Int> requires (nt::is_int_v<Vec> && nt::is_int_v<Int>)
    [[nodiscard]] NOA_HD constexpr Vec operator%(const Vec& lhs, Int rhs) noexcept {
        return lhs % Vec::filled_with(rhs);
    }

    template<typename Vec, typename Int> requires (nt::is_int_v<Vec> && nt::is_int_v<Int>)
    [[nodiscard]] NOA_HD constexpr Vec operator%(Int lhs, const Vec& rhs) noexcept {
        return Vec::filled_with(lhs) % rhs;
    }
}

namespace noa {
    template<size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_FHD constexpr bool any(const Vec<bool, N, A>& vector) noexcept {
        bool output = vector[0];
        for (size_t i = 1; i < N; ++i)
            output = output || vector[i];
        return output;
    }

    template<size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_FHD constexpr bool all(const Vec<bool, N, A>& vector) noexcept {
        for (size_t i = 0; i < N; ++i)
            if (vector[i] == false)
                return false;
        return true;
    }

    [[nodiscard]] NOA_FHD constexpr bool any(bool v) noexcept { return v; }
    [[nodiscard]] NOA_FHD constexpr bool all(bool v) noexcept { return v; }

    // -- Cast--
    template<typename TTo, typename TFrom, size_t N, size_t A> requires nt::is_vec_of_size_v<TTo, N>
    [[nodiscard]] NOA_HD constexpr bool is_safe_cast(const Vec<TFrom, N, A>& src) noexcept {
        if constexpr (N == 0) {
            return true;
        } else {
            bool output = is_safe_cast<typename TTo::value_type>(src[0]);
            for (size_t i = 1; i < N; ++i)
                output = output and is_safe_cast<typename TTo::value_type>(src[i]);
            return output;
        }
    }

    template<typename TTo, typename TFrom, size_t N, size_t A> requires nt::is_vec_of_size_v<TTo, N>
    [[nodiscard]] NOA_HD constexpr TTo clamp_cast(const Vec<TFrom, N, A>& src) noexcept {
        TTo output;
        if constexpr (N > 0) {
            for (size_t i = 0; i < N; ++i)
                output[i] = clamp_cast<typename TTo::value_type>(src[i]);
        }
        return output;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto cos(Vec<T, N, A> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2cos(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = cos(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto sin(Vec<T, N, A> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2sin(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = sin(vector[i]);
        return vector;
    }

    template<typename T> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD Vec<T, 2> sincos(T x) {
        Vec<T, 2> sin_cos;
        sincos(x, sin_cos.data(), sin_cos.data() + 1);
        return sin_cos; // auto [sin, cos] = sincos(x);
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto sinc(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return sinc(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = sinc(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto tan(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return tan(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = tan(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto acos(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return acos(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = acos(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto asin(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return asin(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = asin(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto atan(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return atan(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = atan(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto rad2deg(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return rad2deg(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = rad2deg(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto deg2rad(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return deg2rad(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = deg2rad(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto cosh(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return cosh(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = cosh(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto sinh(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return sinh(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = sinh(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto tanh(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return tanh(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = tanh(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto acosh(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return acosh(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = acosh(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto asinh(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return asinh(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = asinh(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto atanh(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return atanh(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = atanh(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto exp(Vec<T, N, A> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2exp(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = exp(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto log(Vec<T, N, A> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2log(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = log(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto log10(Vec<T, N, A> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2log10(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = log10(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto log1p(Vec<T, N, A> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>)
            return log1p(vector.template as<typename T::arithmetic_type>()).template as<T, A>();

        for (size_t i = 0; i < N; ++i)
            vector[i] = log1p(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto sqrt(Vec<T, N, A> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2sqrt(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = sqrt(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto rsqrt(Vec<T, N, A> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2rsqrt(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = rsqrt(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto round(Vec<T, N, A> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2rint(alias[i]); // h2rint is rounding to nearest
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = round(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto rint(Vec<T, N, A> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2))
            return round(vector);
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = rint(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto ceil(Vec<T, N, A> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2ceil(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = ceil(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto floor(Vec<T, N, A> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2floor(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = floor(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto abs(Vec<T, N, A> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = __habs2(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = abs(vector[i]);
        return vector;
    }

    template<typename T, size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_FHD constexpr auto sum(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::is_same_v<T, Half>)
            return sum(vector.template as<typename T::arithmetic_type>()).template as<T, A>();

        T output = vector[0];
        for (size_t i = 1; i < N; ++i)
            output += vector[i];
        return output;
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto mean(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::is_same_v<T, Half>)
            return mean(vector.template as<typename T::arithmetic_type>()).template as<T, A>();
        return sum(vector) / 2;
    }

    template<typename T, size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_FHD constexpr auto product(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::is_same_v<T, Half>)
            return product(vector.template as<typename T::arithmetic_type>()).template as<T, A>();

        T output = vector[0];
        for (size_t i = 1; i < N; ++i)
            output *= vector[i];
        return output;
    }

    template<typename T, size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_FHD constexpr auto dot(const Vec<T, N, A>& lhs, const Vec<T, N, A>& rhs) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return dot(lhs.template as<typename T::arithmetic_type>(),
                       rhs.template as<typename T::arithmetic_type>()).template as<T, A>();
        }

        T output{0};
        for (size_t i = 0; i < N; ++i)
            output += lhs[i] * rhs[i];
        return output;
    }

    template<typename T, size_t N, size_t A> requires (nt::is_real_v<T> and (N > 0))
    [[nodiscard]] NOA_FHD constexpr auto norm(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            const auto tmp = vector.template as<typename T::arithmetic_type>();
            return norm(tmp).template as<T, A>();
        }

        return sqrt(dot(vector, vector)); // euclidean norm
    }

    template<typename T, size_t N, size_t A> requires nt::is_real_v<T>
    [[nodiscard]] NOA_FHD constexpr auto normalize(const Vec<T, N, A>& vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            const auto tmp = vector.template as<typename T::arithmetic_type>();
            return normalize(tmp).template as<T, A>();
        }

        return vector / norm(vector); // may divide by 0
    }

    template<typename T, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto cross_product(const Vec<T, 3, A>& lhs, const Vec<T, 3, A>& rhs) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            using arithmetic_type = typename T::arithmetic_type;
            return cross_product(lhs.template as<arithmetic_type>(),
                                 rhs.template as<arithmetic_type>()).template as<T, A>();
        }

        return Vec<T, 3, A>{
                lhs[1] * rhs[2] - lhs[2] * rhs[1],
                lhs[2] * rhs[0] - lhs[0] * rhs[2],
                lhs[0] * rhs[1] - lhs[1] * rhs[0]};
    }

    template<typename T, size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_FHD constexpr T min(Vec<T, N, A> vector) noexcept {
        if constexpr (N == 1)
            return vector[0];

        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, Half> && N == 4) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            const __half2 tmp = __hmin2(alias[0], alias[1]);
            return min(tmp.x, tmp.y);
        } else if constexpr (std::is_same_v<T, Half> && N == 8) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            const __half2 tmp0 = __hmin2(alias[0], alias[1]);
            const __half2 tmp1 = __hmin2(alias[2], alias[3]);
            const __half2 tmp2 = __hmin2(tmp0, tmp1);
            return min(tmp2.x, tmp2.y);
        } // TODO Refactor for generic reduction for multiple of 4
        #endif

        auto min_element = min(vector[0], vector[1]);
        for (size_t i = 2; i < N; ++i)
            min_element = min(min_element, vector[i]);
        return min_element;
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto min(Vec<T, N, A> lhs, const Vec<T, N, A>& rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
            auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
            for (size_t i = 0; i < N / 2; ++i)
                alias0[i] = __hmin2(alias0[i], alias1[i]);
            return lhs;
        }
        #endif

        for (size_t i = 0; i < N; ++i)
            lhs[i] = min(lhs[i], rhs[i]);
        return lhs;
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto min(const Vec<T, N, A>& lhs, T rhs) noexcept {
        return min(lhs, Vec<T, N, A>::filled_with(rhs));
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto min(T lhs, const Vec<T, N, A>& rhs) noexcept {
        return min(Vec<T, N, A>::filled_with(lhs), rhs);
    }

    template<typename T, size_t N, size_t A> requires (N > 0)
    [[nodiscard]] NOA_FHD constexpr T max(Vec<T, N, A> vector) noexcept {
        if constexpr (N == 1)
            return vector[0];

        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, Half> && N == 4) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            const __half2 tmp = __hmax2(alias[0], alias[1]);
            return max(tmp.x, tmp.y);
        } else if constexpr (std::is_same_v<T, Half> && N == 8) {
            auto* alias = reinterpret_cast<__half2*>(vector.data());
            const __half2 tmp0 = __hmax2(alias[0], alias[1]);
            const __half2 tmp1 = __hmax2(alias[2], alias[3]);
            const __half2 tmp2 = __hmax2(tmp0, tmp1);
            return max(tmp2.x, tmp2.y);
        } // TODO Refactor for generic reduction for multiple of 4
        #endif

        auto max_element = max(vector[0], vector[1]);
        for (size_t i = 2; i < N; ++i)
            max_element = max(max_element, vector[i]);
        return max_element;
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto max(Vec<T, N, A> lhs, const Vec<T, N, A>& rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias0 = reinterpret_cast<__half2*>(lhs.data());
            auto* alias1 = reinterpret_cast<__half2*>(rhs.data());
            for (size_t i = 0; i < N / 2; ++i)
                alias0[i] = __hmax2(alias0[i], alias1[i]);
            return lhs;
        }
        #endif

        for (size_t i = 0; i < N; ++i)
            lhs[i] = max(lhs[i], rhs[i]);
        return lhs;
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto max(const Vec<T, N, A>& lhs, T rhs) noexcept {
        return max(lhs, Vec<T, N, A>::filled_with(rhs));
    }

    template<typename T, size_t N, size_t A>
    [[nodiscard]] NOA_FHD constexpr auto max(T lhs, const Vec<T, N, A>& rhs) noexcept {
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
    [[nodiscard]] NOA_FHD constexpr auto clamp(const Vec<T, N, A>& lhs, T low, T high) noexcept {
        return min(max(lhs, low), high);
    }

    template<int32_t ULP = 2, typename Real, size_t N, size_t A> requires nt::is_real_v<Real>
    [[nodiscard]] NOA_IHD constexpr auto allclose(
            const Vec<Real, N, A>& lhs,
            const Vec<Real, N, A>& rhs,
            Real epsilon = static_cast<Real>(1e-6)
    ) {
        Vec<bool, N> output;
        for (size_t i = 0; i < N; ++i)
            output[i] = allclose<ULP>(lhs[i], rhs[i], epsilon);
        return output;
    }

    template<int32_t ULP = 2, typename Real, size_t N, size_t A> requires nt::is_real_v<Real>
    [[nodiscard]] NOA_IHD constexpr auto allclose(
            const Vec<Real, N, A>& lhs,
            Real rhs,
            Real epsilon = static_cast<Real>(1e-6)
    ) {
        return allclose<ULP>(lhs, Vec<Real, N, A>::filled_with(rhs), epsilon);
    }

    template<int32_t ULP = 2, typename Real, size_t N, size_t A> requires nt::is_real_v<Real>
    [[nodiscard]] NOA_IHD constexpr auto allclose(
            Real lhs,
            const Vec<Real, N, A>& rhs,
            Real epsilon = static_cast<Real>(1e-6)
    ) {
        return allclose<ULP>(Vec<Real, N, A>::filled_with(lhs), rhs, epsilon);
    }

    template<typename T, size_t N, size_t A, typename Comparison> requires (N <= 4)
    [[nodiscard]] NOA_IHD constexpr auto stable_sort(Vec<T, N, A> vector, Comparison&& comp) noexcept {
        small_stable_sort<N>(vector.data(), std::forward<Comparison>(comp));
        return vector;
    }

    template<typename T, size_t N, size_t A, typename Comparison> requires (N <= 4)
    [[nodiscard]] NOA_IHD constexpr auto sort(Vec<T, N, A> vector, Comparison&& comp) noexcept {
        small_stable_sort<N>(vector.data(), std::forward<Comparison>(comp));
        return vector;
    }

    template<typename T, size_t N, size_t A> requires (N <= 4)
    [[nodiscard]] NOA_IHD constexpr auto stable_sort(Vec<T, N, A> vector) noexcept {
        small_stable_sort<N>(vector.data(), [](const auto& a, const auto& b) { return a < b; });
        return vector;
    }

    template<typename T, size_t N, size_t A> requires (N <= 4)
    [[nodiscard]] NOA_IHD constexpr auto sort(Vec<T, N, A> vector) noexcept {
        small_stable_sort<N>(vector.data(), [](const auto& a, const auto& b) { return a < b; });
        return vector;
    }

#if defined(NOA_IS_OFFLINE)
    template<typename T, size_t N, size_t A, typename Comparison> requires (N > 4)
    [[nodiscard]] auto stable_sort(Vec<T, N, A> vector, Comparison&& comp) noexcept {
        std::stable_sort(vector.begin(), vector.end(), std::forward<Comparison>(comp));
        return vector;
    }

    template<typename T, size_t N, size_t A, typename Comparison> requires (N > 4)
    [[nodiscard]] auto sort(Vec<T, N, A> vector, Comparison&& comp) noexcept {
        std::sort(vector.begin(), vector.end(), std::forward<Comparison>(comp));
        return vector;
    }

    template<typename T, size_t N, size_t A> requires (N > 4)
    [[nodiscard]] auto stable_sort(Vec<T, N, A> vector) noexcept {
        std::stable_sort(vector.begin(), vector.end());
        return vector;
    }

    template<typename T, size_t N, size_t A> requires (N > 4)
    [[nodiscard]] auto sort(Vec<T, N, A> vector) noexcept {
        std::sort(vector.begin(), vector.end());
        return vector;
    }
#endif
}
