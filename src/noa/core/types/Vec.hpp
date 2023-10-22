#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/math/Ops.hpp"
#include "noa/core/utils/ClampCast.hpp"
#include "noa/core/utils/SafeCast.hpp"
#include "noa/core/utils/Sort.hpp"

#if defined(NOA_IS_OFFLINE)
#include <algorithm>
#endif

namespace noa::guts {
    template<typename Value, size_t N>
    struct VecAlignment {
    private:
        static constexpr size_t MAX_ALIGNMENT = 16;
        static constexpr size_t SIZE_OF = sizeof(Value) * N;
        static constexpr size_t ALIGN_OF = alignof(Value);

    public:
        static constexpr size_t VALUE = is_power_of_2(SIZE_OF) ? min(SIZE_OF, MAX_ALIGNMENT) : ALIGN_OF;
    };
}

namespace noa {
    /// Aggregates of one or multiple values with the same type.
    /// Similar to std::array<Value, N>, but with a lot more functionalities.
    template<typename Value, size_t N>
    class alignas(guts::VecAlignment<Value, N>::VALUE) Vec {
    public:
        static_assert(nt::is_numeric_v<Value>, "Only numeric types are supported");
        static_assert(!std::is_const_v<Value>, "The value type must be mutable");
        static_assert(!std::is_reference_v<Value>, "The value type must be a value");
        static_assert(N > 0, "Empty vectors are not supported");

        using value_type = Value;
        using mutable_value_type = value_type;
        static constexpr int64_t SSIZE = N;
        static constexpr size_t SIZE = N;

    public:
        Value array[N];

    public: // Static factory functions
        template<typename T, typename = std::enable_if_t<nt::is_numeric_v<T>>>
        [[nodiscard]] NOA_HD static constexpr Vec from_value(T value) noexcept {
            NOA_ASSERT(is_safe_cast<value_type>(value));
            const auto value_cast = static_cast<value_type>(value);
            Vec vec;
            for (auto& e: vec)
                e = value_cast;
            return vec;
        }

        template<typename T, typename = std::enable_if_t<nt::is_numeric_v<T>>>
        [[nodiscard]] NOA_HD static constexpr Vec filled_with(T value) noexcept {
            return from_value(value); // filled_with has a better name but keep from_value for consistency
        }

        template<typename... Args, typename = std::enable_if_t<sizeof...(Args) == SIZE && nt::are_numeric_v<Args...>>>
        [[nodiscard]] NOA_HD static constexpr Vec from_values(Args... values) noexcept {
            NOA_ASSERT(is_safe_cast<value_type>(value) && ...);
            return {static_cast<value_type>(values)...};
        }

        template<typename T>
        [[nodiscard]] NOA_HD static constexpr Vec from_vector(const Vec<T, SIZE>& vector) noexcept {
            Vec vec;
            for (int64_t i = 0; i < SSIZE; ++i) {
                NOA_ASSERT(is_safe_cast<value_type>(vector[i]));
                vec[i] = static_cast<value_type>(vector[i]);
            }
            return vec;
        }

        template<typename T, typename = std::enable_if_t<nt::are_numeric_v<T>>>
        [[nodiscard]] NOA_HD static constexpr Vec from_pointer(const T* values) noexcept {
            Vec vec;
            for (int64_t i = 0; i < SSIZE; ++i)
                vec[i] = static_cast<value_type>(values[i]);
            return vec;
        }

    public:
        // Allow explicit conversion constructor (while still being an aggregate)
        // and add support for static_cast<Vec<U>>(Vec<T>{}).
        template<typename U>
        [[nodiscard]] NOA_HD constexpr explicit operator Vec<U, SIZE>() const noexcept {
            return Vec<U, SIZE>::from_vector(*this);
        }

        template<typename Void = void,
                 typename = std::enable_if_t<std::is_void_v<Void> && std::is_same_v<bool, value_type>>>
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept {
            for (size_t i = 0; i < N; ++i)
                if (array[i] == false)
                    return false;
            return true;
        }

    public: // Accessor operators and functions
        template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
        [[nodiscard]] NOA_HD constexpr value_type& operator[](Int i) noexcept {
            NOA_ASSERT(static_cast<int64_t>(i) < SSIZE);
            if constexpr (std::is_signed_v<Int>) {
                NOA_ASSERT(i >= 0);
            }
            return array[i];
        }

        template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
        [[nodiscard]] NOA_HD constexpr const value_type& operator[](Int i) const noexcept {
            NOA_ASSERT(static_cast<int64_t>(i) < SSIZE);
            if constexpr (std::is_signed_v<Int>) {
                NOA_ASSERT(i >= 0);
            }
            return array[i];
        }

        // Structure binding support.
        // Note that this relies on the tuple-like binding, which appears to not be supported by nvrtc.
        template<int I> [[nodiscard]] NOA_HD constexpr const value_type& get() const noexcept { return array[I]; }
        template<int I> [[nodiscard]] NOA_HD constexpr value_type& get() noexcept { return array[I]; }

        [[nodiscard]] NOA_HD constexpr const value_type* data() const noexcept { return array; }
        [[nodiscard]] NOA_HD constexpr value_type* data() noexcept { return array; }
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
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                auto* alias = reinterpret_cast<__half2*>(&v);
                for (int64_t i = 0; i < SSIZE / 2; ++i)
                    alias[i] = -alias[i];
                return v;
            }
            #endif
            for (int64_t i = 0; i < SSIZE; ++i)
                v[i] = -v[i];
            return v;
        }

        // -- Binary Arithmetic Operators --
        [[nodiscard]] friend NOA_HD constexpr Vec operator+(Vec lhs, Vec rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                auto* alias0 = reinterpret_cast<__half2*>(&lhs);
                auto* alias1 = reinterpret_cast<__half2*>(&rhs);
                for (int64_t i = 0; i < SSIZE / 2; ++i)
                    alias0[i] += alias1[i];
                return lhs;
            }
            #endif
            for (int64_t i = 0; i < SSIZE; ++i)
                lhs[i] += rhs[i];
            return lhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator+(const Vec& lhs, value_type rhs) noexcept {
            return lhs + Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator+(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) + rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator-(Vec lhs, Vec rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                auto* alias0 = reinterpret_cast<__half2*>(&lhs);
                auto* alias1 = reinterpret_cast<__half2*>(&rhs);
                for (int64_t i = 0; i < SSIZE / 2; ++i)
                    alias0[i] -= alias1[i];
                return lhs;
            }
            #endif
            for (int64_t i = 0; i < SSIZE; ++i)
                lhs[i] -= rhs[i];
            return lhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator-(const Vec& lhs, value_type rhs) noexcept {
            return lhs - Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator-(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) - rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator*(Vec lhs, Vec rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                auto* alias0 = reinterpret_cast<__half2*>(&lhs);
                auto* alias1 = reinterpret_cast<__half2*>(&rhs);
                for (int64_t i = 0; i < SSIZE / 2; ++i)
                    alias0[i] *= alias1[i];
                return lhs;
            }
            #endif
            for (int64_t i = 0; i < SSIZE; ++i)
                lhs[i] *= rhs[i];
            return lhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator*(const Vec& lhs, value_type rhs) noexcept {
            return lhs * Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator*(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) * rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator/(Vec lhs, Vec rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                auto* alias0 = reinterpret_cast<__half2*>(&lhs);
                auto* alias1 = reinterpret_cast<__half2*>(&rhs);
                for (int64_t i = 0; i < SSIZE / 2; ++i)
                    alias0[i] /= alias1[i];
                return lhs;
            }
            #endif
            for (int64_t i = 0; i < SSIZE; ++i)
                lhs[i] /= rhs[i];
            return lhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator/(const Vec& lhs, value_type rhs) noexcept {
            return lhs / Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr Vec operator/(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) / rhs;
        }

        // -- Comparison Operators --
        [[nodiscard]] friend NOA_HD constexpr auto operator>(Vec lhs, Vec rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                auto* alias0 = reinterpret_cast<__half2*>(&lhs);
                auto* alias1 = reinterpret_cast<__half2*>(&rhs);
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
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>(const Vec& lhs, value_type rhs) noexcept {
            return lhs > Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) > rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(Vec lhs, Vec rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                auto* alias0 = reinterpret_cast<__half2*>(&lhs);
                auto* alias1 = reinterpret_cast<__half2*>(&rhs);
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
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(const Vec& lhs, value_type rhs) noexcept {
            return lhs < Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) < rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(Vec lhs, Vec rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                auto* alias0 = reinterpret_cast<__half2*>(&lhs);
                auto* alias1 = reinterpret_cast<__half2*>(&rhs);
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
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(const Vec& lhs, value_type rhs) noexcept {
            return lhs >= Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator>=(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) >= rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(Vec lhs, Vec rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                auto* alias0 = reinterpret_cast<__half2*>(&lhs);
                auto* alias1 = reinterpret_cast<__half2*>(&rhs);
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
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(const Vec& lhs, value_type rhs) noexcept {
            return lhs <= Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator<=(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) <= rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(Vec lhs, Vec rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                auto* alias0 = reinterpret_cast<__half2*>(&lhs);
                auto* alias1 = reinterpret_cast<__half2*>(&rhs);
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
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(const Vec& lhs, value_type rhs) noexcept {
            return lhs == Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) == rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(Vec lhs, Vec rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half> && !(SSIZE % 2)) {
                auto* alias0 = reinterpret_cast<__half2*>(&lhs);
                auto* alias1 = reinterpret_cast<__half2*>(&rhs);
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
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(const Vec& lhs, value_type rhs) noexcept {
            return lhs != Vec::filled_with(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(value_type lhs, const Vec& rhs) noexcept {
            return Vec::filled_with(lhs) != rhs;
        }

    public: // Type casts
        template<typename T, nt::enable_if_bool_t<nt::is_numeric_v<T>> = true>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            return static_cast<Vec<T, SIZE>>(*this);
        }

        template<typename T, nt::enable_if_bool_t<nt::is_numeric_v<T>> = true>
        [[nodiscard]] NOA_HD constexpr auto as_clamp() const noexcept {
            return clamp_cast<Vec<T, SIZE>>(*this);
        }

#if defined(NOA_IS_OFFLINE)
        template<typename T, nt::enable_if_bool_t<nt::is_numeric_v<T>> = true>
        [[nodiscard]] constexpr auto as_safe() const {
            return safe_cast<Vec<T, SIZE>>(*this);
        }
#endif

    public:
        template<size_t S = 1, typename = std::enable_if_t<(N > S)>>
        [[nodiscard]] NOA_HD constexpr auto pop_front() const noexcept {
            return Vec<value_type, N - S>::from_pointer(data() + S);
        }

        template<size_t S = 1, typename = std::enable_if_t<(N > S)>>
        [[nodiscard]] NOA_HD constexpr auto pop_back() const noexcept {
            return Vec<value_type, N - S>::from_pointer(data());
        }

        [[nodiscard]] NOA_HD constexpr auto push_front(value_type value) const noexcept {
            Vec<value_type, N + 1> output;
            output[0] = value;
            for (size_t i = 0; i < N; ++i)
                output[i + 1] = array[i];
            return output;
        }

        [[nodiscard]] NOA_HD constexpr auto push_back(value_type value) const noexcept {
            Vec<value_type, N + 1> output;
            for (size_t i = 0; i < N; ++i)
                output[i] = array[i];
            output[N] = value;
            return output;
        }

        template<size_t S>
        [[nodiscard]] NOA_HD constexpr auto push_front(const Vec<value_type, S>& vector) const noexcept {
            Vec<value_type, N + S> output;
            for (size_t i = 0; i < S; ++i)
                output[i] = vector[i];
            for (size_t i = 0; i < N; ++i)
                output[i + S] = array[i];
            return output;
        }

        template<size_t S>
        [[nodiscard]] NOA_HD constexpr auto push_back(const Vec<value_type, S>& vector) const noexcept {
            Vec<value_type, N + S> output;
            for (size_t i = 0; i < N; ++i)
                output[i] = array[i];
            for (size_t i = 0; i < S; ++i)
                output[i + N] = vector[i];
            return output;
        }

        template<typename... Indexes, typename = std::enable_if_t<nt::are_restricted_int_v<Indexes...>>>
        [[nodiscard]] NOA_HD constexpr auto filter(Indexes... indexes) const noexcept {
            // TODO This can do a lot more than "filter". Rename?
            return Vec<value_type, sizeof...(Indexes)>{(*this)[indexes]...};
        }

        [[nodiscard]] NOA_HD constexpr Vec flip() const noexcept {
            Vec output;
            for (size_t i = 0; i < SIZE; ++i)
                output[i] = array[(N - 1) - i];
            return output;
        }

        template<typename Int = std::conditional_t<nt::is_int_v<value_type>, value_type, int64_t>,
                 typename = std::enable_if_t<nt::is_restricted_int_v<Int>>>
        [[nodiscard]] NOA_HD constexpr Vec reorder(const Vec<Int, SIZE>& order) const noexcept {
            Vec output;
            for (size_t i = 0; i < SIZE; ++i)
                output[i] = array[order[i]];
            return output;
        }

        // Circular shifts the vector by a given amount.
        // If "count" is positive, shift to the right, otherwise, shift to the left.
        [[nodiscard]] NOA_HD constexpr Vec circular_shift(int64_t count) {
            if constexpr (SIZE == 1)
                return *this;

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

        [[nodiscard]] NOA_HD constexpr Vec copy() const noexcept {
            return *this;
        }

        template<size_t INDEX>
        [[nodiscard]] NOA_HD constexpr Vec set(value_type value) const noexcept {
            static_assert(INDEX < SIZE);
            auto output = *this;
            output[INDEX] = value;
            return output;
        }

#if defined(NOA_IS_OFFLINE)
    public:
        [[nodiscard]] static std::string name() {
            return fmt::format("Vec<{},{}>", to_human_readable<value_type>(), SIZE);
        }
#endif
    };
}

#if defined(NOA_IS_OFFLINE)
// Support for output stream:
namespace noa {
    template<typename T, size_t N>
    inline std::ostream& operator<<(std::ostream& os, const Vec<T, N>& v) {
        if constexpr (nt::is_real_or_complex_v<T>)
            os << fmt::format("{::.3f}", v); // {fmt} ranges
        else
            os << fmt::format("{}", v);
        return os;
    }
}
#endif

// Support for structure bindings:
namespace std {
    template<typename T, size_t N>
    struct tuple_size<noa::Vec<T, N>> : std::integral_constant<size_t, N> {};

    template<typename T, size_t N>
    struct tuple_size<const noa::Vec<T, N>> : std::integral_constant<size_t, N> {};

    template<size_t I, size_t N, typename T>
    struct tuple_element<I, noa::Vec<T, N>> { using type = T; };

    template<size_t I, size_t N, typename T>
    struct tuple_element<I, const noa::Vec<T, N>> { using type = const T; };
}

namespace noa::traits {
    #if defined(NOA_IS_OFFLINE)
    static_assert(nt::is_detected_convertible_v<std::string, nt::has_name, Vec<bool, 1>>);
    #endif

    template<typename T> struct proclaim_is_vec : std::false_type {};
    template<typename T> using is_vec = std::bool_constant<proclaim_is_vec<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_vec_v = is_vec<T>::value;
    template<typename T> constexpr bool is_vecX_v = is_vec<T>::value;

    template<typename, typename> struct proclaim_is_vec_of_type : std::false_type {};
    template<typename T, typename V> using is_vec_of_type = std::bool_constant<proclaim_is_vec_of_type<remove_ref_cv_t<T>, V>::value>;
    template<typename T, typename V> constexpr bool is_vec_of_type_v = is_vec_of_type<T, V>::value;
    template<typename T, typename V> constexpr bool is_vecT_v = is_vec_of_type<T, V>::value;

    template<typename, size_t> struct proclaim_is_vec_of_size : std::false_type {};
    template<typename T, size_t N> using is_vec_of_size = std::bool_constant<proclaim_is_vec_of_size<remove_ref_cv_t<T>, N>::value>;
    template<typename T, size_t N> constexpr bool is_vec_of_size_v = is_vec_of_size<T, N>::value;
    template<typename T, size_t N> constexpr bool is_vecN_v = is_vec_of_size<T, N>::value;

    template<typename T> constexpr bool is_vec1_v = is_vecN_v<T, 1>;
    template<typename T> constexpr bool is_vec2_v = is_vecN_v<T, 2>;
    template<typename T> constexpr bool is_vec3_v = is_vecN_v<T, 3>;
    template<typename T> constexpr bool is_vec4_v = is_vecN_v<T, 4>;

    // Vector of integers:
    template<typename T> constexpr bool is_intX_v = is_vecX_v<T> && is_int_v<value_type_t<T>>;
    template<typename T, size_t N> constexpr bool is_intN_v = is_vecN_v<T, N> && is_int_v<value_type_t<T>>;
    template<typename T> constexpr bool is_int1_v = is_intN_v<T, 1>;
    template<typename T> constexpr bool is_int2_v = is_intN_v<T, 2>;
    template<typename T> constexpr bool is_int3_v = is_intN_v<T, 3>;
    template<typename T> constexpr bool is_int4_v = is_intN_v<T, 4>;

    // Vector of bool:
    template<typename T> constexpr bool is_boolX_v = is_vecX_v<T> && is_bool_v<value_type_t<T>>;
    template<typename T, size_t N> constexpr bool is_boolN_v = is_vecN_v<T, N> && is_bool_v<value_type_t<T>>;
    template<typename T> constexpr bool is_bool1_v = is_boolN_v<T, 1>;
    template<typename T> constexpr bool is_bool2_v = is_boolN_v<T, 2>;
    template<typename T> constexpr bool is_bool3_v = is_boolN_v<T, 3>;
    template<typename T> constexpr bool is_bool4_v = is_boolN_v<T, 4>;

    // Vector of signed integers:
    template<typename T> constexpr bool is_sintX_v = is_vecX_v<T> && is_sint_v<value_type_t<T>>;
    template<typename T, size_t N> constexpr bool is_sintN_v = is_vecN_v<T, N> && is_sint_v<value_type_t<T>>;
    template<typename T> constexpr bool is_sint1_v = is_sintN_v<T, 1>;
    template<typename T> constexpr bool is_sint2_v = is_sintN_v<T, 2>;
    template<typename T> constexpr bool is_sint3_v = is_sintN_v<T, 3>;
    template<typename T> constexpr bool is_sint4_v = is_sintN_v<T, 4>;

    // Vector of unsigned integers:
    template<typename T> constexpr bool is_uintX_v = is_vecX_v<T> && is_uint_v<value_type_t<T>>;
    template<typename T, size_t N> constexpr bool is_uintN_v = is_vecN_v<T, N> && is_uint_v<value_type_t<T>>;
    template<typename T> constexpr bool is_uint1_v = is_uintN_v<T, 1>;
    template<typename T> constexpr bool is_uint2_v = is_uintN_v<T, 2>;
    template<typename T> constexpr bool is_uint3_v = is_uintN_v<T, 3>;
    template<typename T> constexpr bool is_uint4_v = is_uintN_v<T, 4>;

    // Vector of real (= floating-points):
    template<typename T> constexpr bool is_realX_v = is_vecX_v<T> && is_real_v<value_type_t<T>>;
    template<typename T, size_t N> constexpr bool is_realN_v = is_vecN_v<T, N> && is_real_v<value_type_t<T>>;
    template<typename T> constexpr bool is_real1_v = is_realN_v<T, 1>;
    template<typename T> constexpr bool is_real2_v = is_realN_v<T, 2>;
    template<typename T> constexpr bool is_real3_v = is_realN_v<T, 3>;
    template<typename T> constexpr bool is_real4_v = is_realN_v<T, 4>;

    template<typename T, size_t N> struct proclaim_is_vec<Vec<T, N>> : std::true_type {};
    template<typename V1, size_t N, typename V2> struct proclaim_is_vec_of_type<Vec<V1, N>, V2> : std::bool_constant<std::is_same_v<V1, V2>> {};
    template<typename V, size_t N1, size_t N2> struct proclaim_is_vec_of_size<Vec<V, N1>, N2> : std::bool_constant<N1 == N2> {};
}

// Special case for Vec<bool,N>:
namespace noa {
    template<size_t N>
    [[nodiscard]] NOA_FHD constexpr auto operator!(Vec<bool, N> vector) noexcept {
        for (size_t i = 0; i < N; ++i)
            vector[i] = !vector[i];
        return vector;
    }

    template<size_t N>
    [[nodiscard]] NOA_FHD constexpr auto operator&&(Vec<bool, N> lhs, const Vec<bool, N>& rhs) noexcept {
        for (size_t i = 0; i < N; ++i)
            lhs[i] = lhs[i] && rhs[i];
        return lhs;
    }

    template<size_t N>
    [[nodiscard]] NOA_FHD constexpr auto operator||(Vec<bool, N> lhs, const Vec<bool, N>& rhs) noexcept {
        for (size_t i = 0; i < N; ++i)
            lhs[i] = lhs[i] || rhs[i];
        return lhs;
    }

    template<size_t N>
    [[nodiscard]] NOA_FHD constexpr bool any(const Vec<bool, N>& vector) noexcept {
        bool output = vector[0];
        for (size_t i = 1; i < N; ++i)
            output = output || vector[i];
        return output;
    }

    template<size_t N>
    [[nodiscard]] NOA_FHD constexpr bool all(const Vec<bool, N>& vector) noexcept {
        return static_cast<bool>(vector);
    }

    [[nodiscard]] NOA_FHD constexpr bool any(bool v) noexcept { return v; }
    [[nodiscard]] NOA_FHD constexpr bool all(bool v) noexcept { return v; }
}

namespace noa {
    // -- Modulo Operator --
    template<typename Vec, nt::enable_if_bool_t<nt::is_intX_v<Vec>> = true>
    [[nodiscard]] NOA_HD constexpr Vec operator%(Vec lhs, const Vec& rhs) noexcept {
        for (int64_t i = 0; i < Vec::SSIZE; ++i)
            lhs[i] %= rhs[i];
        return lhs;
    }

    template<typename Vec, typename Int, nt::enable_if_bool_t<nt::is_intX_v<Vec> && nt::is_int_v<Int>> = true>
    [[nodiscard]] NOA_HD constexpr Vec operator%(const Vec& lhs, Int rhs) noexcept {
        return lhs % Vec::filled_with(rhs);
    }

    template<typename Vec, typename Int, nt::enable_if_bool_t<nt::is_intX_v<Vec> && nt::is_int_v<Int>> = true>
    [[nodiscard]] NOA_HD constexpr Vec operator%(Int lhs, const Vec& rhs) noexcept {
        return Vec::filled_with(lhs) % rhs;
    }

    // -- Cast--
    template<typename TTo, typename TFrom, size_t N, nt::enable_if_bool_t<nt::is_vecN_v<TTo, N>> = true>
    [[nodiscard]] NOA_HD constexpr bool is_safe_cast(const Vec<TFrom, N>& src) noexcept {
        bool output = is_safe_cast<typename TTo::value_type>(src[0]);
        for (size_t i = 1; i < N; ++i)
            output = output && is_safe_cast<typename TTo::value_type>(src[i]);
        return output;
    }

    template<typename TTo, typename TFrom, size_t N, nt::enable_if_bool_t<nt::is_vecN_v<TTo, N>> = true>
    [[nodiscard]] NOA_HD constexpr TTo clamp_cast(const Vec<TFrom, N>& src) noexcept {
        TTo output;
        for (size_t i = 0; i < N; ++i)
            output[i] = clamp_cast<typename TTo::value_type>(src[i]);
        return output;
    }
}

// Type aliases:
namespace noa {
    template<typename T> using Vec1 = Vec<T, 1>;
    template<typename T> using Vec2 = Vec<T, 2>;
    template<typename T> using Vec3 = Vec<T, 3>;
    template<typename T> using Vec4 = Vec<T, 4>;
}

namespace noa {
    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto cos(Vec<T, N> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2cos(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = cos(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto sin(Vec<T, N> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2sin(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = sin(vector[i]);
        return vector;
    }

    template<typename T, nt::enable_if_bool_t<nt::is_real_v<T>> = true>
    [[nodiscard]] NOA_FHD Vec<T, 2> sincos(T x) {
        Vec<T, 2> sin_cos;
        sincos(x, sin_cos.data(), sin_cos.data() + 1);
        return sin_cos; // auto [sin, cos] = sincos(x);
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto sinc(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return sinc(vector.template as<typename T::arithmetic_type>()).template as<T>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = sinc(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto tan(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return tan(vector.template as<typename T::arithmetic_type>()).template as<T>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = tan(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto acos(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return acos(vector.template as<typename T::arithmetic_type>()).template as<T>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = acos(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto asin(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return asin(vector.template as<typename T::arithmetic_type>()).template as<T>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = asin(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto atan(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return atan(vector.template as<typename T::arithmetic_type>()).template as<T>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = atan(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto rad2deg(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return rad2deg(vector.template as<typename T::arithmetic_type>()).template as<T>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = rad2deg(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto deg2rad(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return deg2rad(vector.template as<typename T::arithmetic_type>()).template as<T>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = deg2rad(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto cosh(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return cosh(vector.template as<typename T::arithmetic_type>()).template as<T>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = cosh(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto sinh(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return sinh(vector.template as<typename T::arithmetic_type>()).template as<T>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = sinh(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto tanh(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return tanh(vector.template as<typename T::arithmetic_type>()).template as<T>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = tanh(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto acosh(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return acosh(vector.template as<typename T::arithmetic_type>()).template as<T>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = acosh(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto asinh(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return asinh(vector.template as<typename T::arithmetic_type>()).template as<T>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = asinh(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto atanh(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return atanh(vector.template as<typename T::arithmetic_type>()).template as<T>();
        }
        for (size_t i = 0; i < N; ++i)
            vector[i] = atanh(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto exp(Vec<T, N> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2exp(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = exp(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto log(Vec<T, N> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2log(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = log(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto log10(Vec<T, N> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2log10(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = log10(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto log1p(Vec<T, N> vector) noexcept {
        if constexpr (std::is_same_v<T, Half>)
            return log1p(vector.template as<typename T::arithmetic_type>()).template as<T>();

        for (size_t i = 0; i < N; ++i)
            vector[i] = log1p(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto sqrt(Vec<T, N> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2sqrt(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = sqrt(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto rsqrt(Vec<T, N> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2rsqrt(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = rsqrt(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto round(Vec<T, N> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2rint(alias[i]); // h2rint is rounding to nearest
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = round(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto rint(Vec<T, N> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2))
            return round(vector);
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = rint(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto ceil(Vec<T, N> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2ceil(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = ceil(vector[i]);
        return vector;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto floor(Vec<T, N> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = h2floor(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = floor(vector[i]);
        return vector;
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto abs(Vec<T, N> vector) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
            for (size_t i = 0; i < N / 2; ++i)
                alias[i] = __habs2(alias[i]);
            return vector;
        }
        #endif
        for (size_t i = 0; i < N; ++i)
            vector[i] = abs(vector[i]);
        return vector;
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto sum(const Vec<T, N>& vector) noexcept {
        if constexpr (std::is_same_v<T, Half>)
            return sum(vector.template as<typename T::arithmetic_type>()).template as<T>();

        T output = vector[0];
        for (size_t i = 1; i < N; ++i)
            output += vector[i];
        return output;
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto mean(const Vec<T, N>& vector) noexcept {
        if constexpr (std::is_same_v<T, Half>)
            return mean(vector.template as<typename T::arithmetic_type>()).template as<T>();
        return sum(vector) / 2;
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto product(const Vec<T, N>& vector) noexcept {
        if constexpr (std::is_same_v<T, Half>)
            return product(vector.template as<typename T::arithmetic_type>()).template as<T>();

        T output = vector[0];
        for (size_t i = 1; i < N; ++i)
            output *= vector[i];
        return output;
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto dot(const Vec<T, N>& lhs, const Vec<T, N>& rhs) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            return dot(lhs.template as<typename T::arithmetic_type>(),
                       rhs.template as<typename T::arithmetic_type>()).template as<T>();
        }

        T output{0};
        for (size_t i = 0; i < N; ++i)
            output += lhs[i] * rhs[i];
        return output;
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto norm(const Vec<T, N>& vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            const auto tmp = vector.template as<typename T::arithmetic_type>();
            return norm(tmp).template as<T>();
        }

        return sqrt(dot(vector, vector)); // euclidean norm
    }

    template<typename T, size_t N, typename = std::enable_if_t<nt::is_real_v<T>>>
    [[nodiscard]] NOA_FHD constexpr auto normalize(const Vec<T, N>& vector) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            const auto tmp = vector.template as<typename T::arithmetic_type>();
            return normalize(tmp).template as<T>();
        }

        return vector / norm(vector); // may divide by 0
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr auto cross_product(const Vec<T, 3>& lhs, const Vec<T, 3>& rhs) noexcept {
        if constexpr (std::is_same_v<T, Half>) {
            using arithmetic_type = typename T::arithmetic_type;
            return cross_product(lhs.template as<arithmetic_type>(),
                                 rhs.template as<arithmetic_type>()).template as<T>();
        }

        return Vec<T, 3>{lhs[1] * rhs[2] - lhs[2] * rhs[1],
                         lhs[2] * rhs[0] - lhs[0] * rhs[2],
                         lhs[0] * rhs[1] - lhs[1] * rhs[0]};
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr T min(Vec<T, N> vector) noexcept {
        if constexpr (N == 1)
            return vector[0];

        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, Half> && N == 4) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
            const __half2 tmp = __hmin2(alias[0], alias[1]);
            return min(tmp.x, tmp.y);
        } else if constexpr (std::is_same_v<T, Half> && N == 8) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
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

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto min(Vec<T, N> lhs, const Vec<T, N>& rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias0 = reinterpret_cast<__half2*>(&lhs);
            auto* alias1 = reinterpret_cast<__half2*>(&rhs);
            for (size_t i = 0; i < N / 2; ++i)
                alias0[i] = __hmin2(alias0[i], alias1[i]);
            return lhs;
        }
        #endif

        for (size_t i = 0; i < N; ++i)
            lhs[i] = min(lhs[i], rhs[i]);
        return lhs;
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto min(const Vec<T, N>& lhs, T rhs) noexcept {
        return min(lhs, Vec<T, N>(rhs));
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto min(T lhs, const Vec<T, N>& rhs) noexcept {
        return min(Vec<T, N>(lhs), rhs);
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr T max(Vec<T, N> vector) noexcept {
        if constexpr (N == 1)
            return vector[0];

        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, Half> && N == 4) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
            const __half2 tmp = __hmax2(alias[0], alias[1]);
            return max(tmp.x, tmp.y);
        } else if constexpr (std::is_same_v<T, Half> && N == 8) {
            auto* alias = reinterpret_cast<__half2*>(&vector);
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

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto max(Vec<T, N> lhs, const Vec<T, N>& rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, Half> && !(N % 2)) {
            auto* alias0 = reinterpret_cast<__half2*>(&lhs);
            auto* alias1 = reinterpret_cast<__half2*>(&rhs);
            for (size_t i = 0; i < N / 2; ++i)
                alias0[i] = __hmax2(alias0[i], alias1[i]);
            return lhs;
        }
        #endif

        for (size_t i = 0; i < N; ++i)
            lhs[i] = max(lhs[i], rhs[i]);
        return lhs;
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto max(const Vec<T, N>& lhs, T rhs) noexcept {
        return max(lhs, Vec<T, N>::filled_with(rhs));
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto max(T lhs, const Vec<T, N>& rhs) noexcept {
        return max(Vec<T, N>::filled_with(lhs), rhs);
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto clamp(
            const Vec<T, N>& lhs,
            const Vec<T, N>& low,
            const Vec<T, N>& high
    ) noexcept {
        return min(max(lhs, low), high);
    }

    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto clamp(const Vec<T, N>& lhs, T low, T high) noexcept {
        return min(max(lhs, low), high);
    }

    template<int32_t ULP = 2, typename Real, size_t N, typename = std::enable_if_t<nt::is_real_v<Real>>>
    [[nodiscard]] NOA_IHD constexpr auto are_almost_equal(
            const Vec<Real, N>& lhs,
            const Vec<Real, N>& rhs,
            Real epsilon = static_cast<Real>(1e-6)
    ) {
        Vec<bool, N> output;
        for (size_t i = 0; i < N; ++i)
            output[i] = are_almost_equal<ULP>(lhs[i], rhs[i], epsilon);
        return output;
    }

    template<int32_t ULP = 2, typename Real, size_t N, typename = std::enable_if_t<nt::is_real_v<Real>>>
    [[nodiscard]] NOA_IHD constexpr auto are_almost_equal(
            const Vec<Real, N>& lhs,
            Real rhs,
            Real epsilon = static_cast<Real>(1e-6)
    ) {
        return are_almost_equal<ULP>(lhs, Vec<Real, N>::filled_with(rhs), epsilon);
    }

    template<int32_t ULP = 2, typename Real, size_t N, typename = std::enable_if_t<nt::is_real_v<Real>>>
    [[nodiscard]] NOA_IHD constexpr auto are_almost_equal(
            Real lhs,
            const Vec<Real, N>& rhs,
            Real epsilon = static_cast<Real>(1e-6)
    ) {
        return are_almost_equal<ULP>(Vec<Real, N>::filled_with(lhs), rhs, epsilon);
    }
}

// Sort:
namespace noa {
    template<typename T, size_t N, typename Comparison, nt::enable_if_bool_t<(N <= 4)> = true>
    [[nodiscard]] NOA_IHD constexpr auto stable_sort(Vec<T, N> vector, Comparison&& comp) noexcept {
        small_stable_sort<N>(vector.data(), std::forward<Comparison>(comp));
        return vector;
    }

    template<typename T, size_t N, typename Comparison, nt::enable_if_bool_t<(N <= 4)> = true>
    [[nodiscard]] NOA_IHD constexpr auto sort(Vec<T, N> vector, Comparison&& comp) noexcept {
        small_stable_sort<N>(vector.data(), std::forward<Comparison>(comp));
        return vector;
    }

    template<typename T, size_t N, nt::enable_if_bool_t<(N <= 4)> = true>
    [[nodiscard]] NOA_IHD constexpr auto stable_sort(Vec<T, N> vector) noexcept {
        small_stable_sort<N>(vector.data(), less_t{});
        return vector;
    }

    template<typename T, size_t N, nt::enable_if_bool_t<(N <= 4)> = true>
    [[nodiscard]] NOA_IHD constexpr auto sort(Vec<T, N> vector) noexcept {
        small_stable_sort<N>(vector.data(), less_t{});
        return vector;
    }

#if defined(NOA_IS_OFFLINE)
    template<typename T, size_t N, typename Comparison, nt::enable_if_bool_t<(N > 4)> = true>
    [[nodiscard]] auto stable_sort(Vec<T, N> vector, Comparison&& comp) noexcept {
        std::stable_sort(vector.begin(), vector.end(), std::forward<Comparison>(comp));
        return vector;
    }

    template<typename T, size_t N, typename Comparison, nt::enable_if_bool_t<(N > 4)> = true>
    [[nodiscard]] auto sort(Vec<T, N> vector, Comparison&& comp) noexcept {
        std::sort(vector.begin(), vector.end(), std::forward<Comparison>(comp));
        return vector;
    }

    template<typename T, size_t N, nt::enable_if_bool_t<(N > 4)> = true>
    [[nodiscard]] auto stable_sort(Vec<T, N> vector) noexcept {
        std::stable_sort(vector.begin(), vector.end());
        return vector;
    }

    template<typename T, size_t N, nt::enable_if_bool_t<(N > 4)> = true>
    [[nodiscard]] auto sort(Vec<T, N> vector) noexcept {
        std::sort(vector.begin(), vector.end());
        return vector;
    }
#endif
}
