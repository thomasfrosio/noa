/// \file noa/common/types/Float4.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 4 floating-point numbers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/ArrayTypes.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Bool4.h"
#include "noa/common/types/ClampCast.h"
#include "noa/common/types/Half.h"
#include "noa/common/types/SafeCast.h"
#include "noa/common/utils/Sort.h"

namespace noa {
    template<typename>
    class Int4;

    template<typename T>
    class alignas(sizeof(T) * 4 >= 16 ? 16 : sizeof(T) * 4) Float4 {
    public:
        using value_type = T;

    public: // Default Constructors
        constexpr Float4() noexcept = default;
        constexpr Float4(const Float4&) noexcept = default;
        constexpr Float4(Float4&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y, typename Z, typename W,
                 typename = std::enable_if_t<traits::is_scalar_v<X> && traits::is_scalar_v<Y> &&
                                             traits::is_scalar_v<Z> && traits::is_scalar_v<W>>>
        NOA_HD constexpr Float4(X a0, Y a1, Z a2, W a3) noexcept
                : m_data{static_cast<T>(a0), static_cast<T>(a1), static_cast<T>(a2), static_cast<T>(a3)} {
            NOA_ASSERT(isSafeCast<T>(a0) && isSafeCast<T>(a1) && isSafeCast<T>(a2) && isSafeCast<T>(a3));
        }

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Float4(U v) noexcept
                : m_data{static_cast<T>(v), static_cast<T>(v), static_cast<T>(v), static_cast<T>(v)} {
            NOA_ASSERT(isSafeCast<T>(v));
        }

        template<typename U>
        NOA_HD constexpr explicit Float4(Float4<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2]), static_cast<T>(v[3])} {
            NOA_ASSERT(isSafeCast<T>(v[0]) && isSafeCast<T>(v[1]) && isSafeCast<T>(v[2]) && isSafeCast<T>(v[3]));
        }

        NOA_HD constexpr explicit Float4(Bool4 v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2]), static_cast<T>(v[3])} {}

        template<typename U>
        NOA_HD constexpr explicit Float4(Int4<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2]), static_cast<T>(v[3])} {
            NOA_ASSERT(isSafeCast<T>(v[0]) && isSafeCast<T>(v[1]) && isSafeCast<T>(v[2]) && isSafeCast<T>(v[3]));
        }

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Float4(const U* ptr) noexcept {
            NOA_ASSERT(ptr != nullptr);
            for (size_t i = 0; i < COUNT; ++i) {
                NOA_ASSERT(isSafeCast<T>(ptr[i]));
                m_data[i] = static_cast<T>(ptr[i]);
            }
        }

    public: // Assignment operators
        constexpr Float4& operator=(const Float4& v) noexcept = default;
        constexpr Float4& operator=(Float4&& v) noexcept = default;

        NOA_HD constexpr Float4& operator=(T v) noexcept {
            m_data[0] = v;
            m_data[1] = v;
            m_data[2] = v;
            m_data[3] = v;
            return *this;
        }

        NOA_HD constexpr Float4& operator+=(Float4 rhs) noexcept {
            *this = *this + rhs;
            return *this;
        }

        NOA_HD constexpr Float4& operator-=(Float4 rhs) noexcept {
            *this = *this - rhs;
            return *this;
        }

        NOA_HD constexpr Float4& operator*=(Float4 rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        NOA_HD constexpr Float4& operator/=(Float4 rhs) noexcept {
            *this = *this / rhs;
            return *this;
        }

        NOA_HD constexpr Float4& operator+=(T rhs) noexcept {
            *this = *this + rhs;
            return *this;
        }

        NOA_HD constexpr Float4& operator-=(T rhs) noexcept {
            *this = *this - rhs;
            return *this;
        }

        NOA_HD constexpr Float4& operator*=(T rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        NOA_HD constexpr Float4& operator/=(T rhs) noexcept {
            *this = *this / rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Float4 operator+(Float4 v) noexcept {
            return v;
        }

        [[nodiscard]] friend NOA_HD constexpr Float4 operator-(Float4 v) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&v);
                tmp[0] = -tmp[0];
                tmp[1] = -tmp[1];
                return v;
            }
            #endif
            return {-v[0], -v[1], -v[2], -v[3]};
        }

        // -- Binary Arithmetic Operators --
        [[nodiscard]] friend NOA_HD constexpr Float4 operator+(Float4 lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                tmp0[0] += tmp1[0];
                tmp0[1] += tmp1[1];
                return lhs;
            }
            #endif
            return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float4 operator+(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) + rhs;
            #endif
            return {lhs + rhs[0], lhs + rhs[1], lhs + rhs[2], lhs + rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float4 operator+(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs + Float4(rhs);
            #endif
            return {lhs[0] + rhs, lhs[1] + rhs, lhs[2] + rhs, lhs[3] + rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Float4 operator-(Float4 lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                tmp0[0] -= tmp1[0];
                tmp0[1] -= tmp1[1];
                return lhs;
            }
            #endif
            return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float4 operator-(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) - rhs;
            #endif
            return {lhs - rhs[0], lhs - rhs[1], lhs - rhs[2], lhs - rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float4 operator-(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs - Float4(rhs);
            #endif
            return {lhs[0] - rhs, lhs[1] - rhs, lhs[2] - rhs, lhs[3] - rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Float4 operator*(Float4 lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                tmp0[0] *= tmp1[0];
                tmp0[1] *= tmp1[1];
                return lhs;
            }
            #endif
            return {lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2], lhs[3] * rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float4 operator*(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) * rhs;
            #endif
            return {lhs * rhs[0], lhs * rhs[1], lhs * rhs[2], lhs * rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float4 operator*(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs * Float4(rhs);
            #endif
            return {lhs[0] * rhs, lhs[1] * rhs, lhs[2] * rhs, lhs[3] * rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Float4 operator/(Float4 lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                tmp0[0] /= tmp1[0];
                tmp0[1] /= tmp1[1];
                return lhs;
            }
            #endif
            return {lhs[0] / rhs[0], lhs[1] / rhs[1], lhs[2] / rhs[2], lhs[3] / rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float4 operator/(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) / rhs;
            #endif
            return {lhs / rhs[0], lhs / rhs[1], lhs / rhs[2], lhs / rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float4 operator/(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs / Float4(rhs);
            #endif
            return {lhs[0] / rhs, lhs[1] / rhs, lhs[2] / rhs, lhs[3] / rhs};
        }

        // -- Comparison Operators --
        [[nodiscard]] friend NOA_HD constexpr Bool4 operator>(Float4 lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                tmp0[0] = __hgt2(tmp0[0], tmp1[0]);
                tmp0[1] = __hgt2(tmp0[1], tmp1[1]);
                return Bool4(lhs[0], lhs[1], lhs[2], lhs[3]);
            }
            #endif
            return {lhs[0] > rhs[0], lhs[1] > rhs[1], lhs[2] > rhs[2], lhs[3] > rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator>(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs > Float4(rhs);
            #endif
            return {lhs[0] > rhs, lhs[1] > rhs, lhs[2] > rhs, lhs[3] > rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator>(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) > rhs;
            #endif
            return {lhs > rhs[0], lhs > rhs[1], lhs > rhs[2], lhs > rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator<(Float4 lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                tmp0[0] = __hlt2(tmp0[0], tmp1[0]);
                tmp0[1] = __hlt2(tmp0[1], tmp1[1]);
                return Bool4(lhs[0], lhs[1], lhs[2], lhs[3]);
            }
            #endif
            return {lhs[0] < rhs[0], lhs[1] < rhs[1], lhs[2] < rhs[2], lhs[3] < rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator<(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs < Float4(rhs);
            #endif
            return {lhs[0] < rhs, lhs[1] < rhs, lhs[2] < rhs, lhs[3] < rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator<(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) < rhs;
            #endif
            return {lhs < rhs[0], lhs < rhs[1], lhs < rhs[2], lhs < rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator>=(Float4 lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                tmp0[0] = __hge2(tmp0[0], tmp1[0]);
                tmp0[1] = __hge2(tmp0[1], tmp1[1]);
                return Bool4(lhs[0], lhs[1], lhs[2], lhs[3]);
            }
            #endif
            return {lhs[0] >= rhs[0], lhs[1] >= rhs[1], lhs[2] >= rhs[2], lhs[3] >= rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator>=(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs >= Float4(rhs);
            #endif
            return {lhs[0] >= rhs, lhs[1] >= rhs, lhs[2] >= rhs, lhs[3] >= rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator>=(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) >= rhs;
            #endif
            return {lhs >= rhs[0], lhs >= rhs[1], lhs >= rhs[2], lhs >= rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator<=(Float4 lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                tmp0[0] = __hle2(tmp0[0], tmp1[0]);
                tmp0[1] = __hle2(tmp0[1], tmp1[1]);
                return Bool4(lhs[0], lhs[1], lhs[2], lhs[3]);
            }
            #endif
            return {lhs[0] <= rhs[0], lhs[1] <= rhs[1], lhs[2] <= rhs[2], lhs[3] <= rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator<=(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs <= Float4(rhs);
            #endif
            return {lhs[0] <= rhs, lhs[1] <= rhs, lhs[2] <= rhs, lhs[3] <= rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator<=(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) <= rhs;
            #endif
            return {lhs <= rhs[0], lhs <= rhs[1], lhs <= rhs[2], lhs <= rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator==(Float4 lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                tmp0[0] = __heq2(tmp0[0], tmp1[0]);
                tmp0[1] = __heq2(tmp0[1], tmp1[1]);
                return Bool4(lhs[0], lhs[1], lhs[2], lhs[3]);
            }
            #endif
            return {lhs[0] == rhs[0], lhs[1] == rhs[1], lhs[2] == rhs[2], lhs[3] == rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator==(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs == Float4(rhs);
            #endif
            return {lhs[0] == rhs, lhs[1] == rhs, lhs[2] == rhs, lhs[3] == rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator==(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) == rhs;
            #endif
            return {lhs == rhs[0], lhs == rhs[1], lhs == rhs[2], lhs == rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator!=(Float4 lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                tmp0[0] = __hne2(tmp0[0], tmp1[0]);
                tmp0[1] = __hne2(tmp0[1], tmp1[1]);
                return Bool4(lhs[0], lhs[1], lhs[2], lhs[3]);
            }
            #endif
            return {lhs[0] != rhs[0], lhs[1] != rhs[1], lhs[2] != rhs[2], lhs[3] != rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator!=(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs != Float4(rhs);
            #endif
            return {lhs[0] != rhs, lhs[1] != rhs, lhs[2] != rhs, lhs[3] != rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator!=(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) != rhs;
            #endif
            return {lhs != rhs[0], lhs != rhs[1], lhs != rhs[2], lhs != rhs[3]};
        }

    public: // Component accesses
        static constexpr size_t COUNT = 4;

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr T& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr const T& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        [[nodiscard]] NOA_HD constexpr const T* get() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr T* get() noexcept { return m_data; }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr const T* get(I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) <= COUNT);
            return m_data + i;
        }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr T* get(I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) <= COUNT);
            return m_data + i;
        }

        [[nodiscard]] NOA_HD constexpr Float4 flip() const noexcept {
            return {m_data[3], m_data[2], m_data[1], m_data[0]};
        }

    private:
        static_assert(traits::is_float_v<T>);
        T m_data[4]{};
    };

    template<typename T>
    struct traits::proclaim_is_float4<Float4<T>> : std::true_type {};

    using half4_t = Float4<half_t>;
    using float4_t = Float4<float>;
    using double4_t = Float4<double>;

    template<typename T>
    [[nodiscard]] NOA_IH constexpr std::array<T, 4> toArray(Float4<T> v) noexcept {
        return {v[0], v[1], v[2], v[3]};
    }

    template<>
    [[nodiscard]] NOA_IH std::string string::human<half4_t>() { return "half4"; }
    template<>
    [[nodiscard]] NOA_IH std::string string::human<float4_t>() { return "float4"; }
    template<>
    [[nodiscard]] NOA_IH std::string string::human<double4_t>() { return "double4"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, Float4<T> v) {
        os << string::format("({:.3f},{:.3f},{:.3f},{:.3f})", v[0], v[1], v[2], v[3]);
        return os;
    }
}

namespace fmt {
    template<typename T>
    struct formatter<noa::Float4<T>> : formatter<T> {
        template<typename FormatContext>
        auto format(const noa::Float4<T>& vec, FormatContext& ctx) {
            auto out = ctx.out();
            *out = '(';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[0], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[1], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[2], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[3], ctx);
            *out = ')';
            return out;
        }
    };
}

namespace noa {
    template<typename TTo, typename TFrom, typename = std::enable_if_t<traits::is_float4_v<TTo>>>
    [[nodiscard]] NOA_FHD constexpr TTo clamp_cast(const Float4<TFrom>& src) noexcept {
        using value_t = traits::value_type_t<TTo>;
        return {clamp_cast<value_t>(src[0]), clamp_cast<value_t>(src[1]),
                clamp_cast<value_t>(src[2]), clamp_cast<value_t>(src[3])};
    }

    template<typename TTo, typename TFrom, typename = std::enable_if_t<traits::is_float4_v<TTo>>>
    [[nodiscard]] NOA_FHD constexpr bool isSafeCast(const Float4<TFrom>& src) noexcept {
        using value_t = traits::value_type_t<TTo>;
        return isSafeCast<value_t>(src[0]) && isSafeCast<value_t>(src[1]) &&
                isSafeCast<value_t>(src[2] && isSafeCast<value_t>(src[2]));
    }
}

namespace noa::math {
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> cos(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            tmp[0] = h2cos(tmp[0]);
            tmp[1] = h2cos(tmp[1]);
            return v;
        }
        #endif
        return Float4<T>(cos(v[0]), cos(v[1]), cos(v[2]), cos(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> sin(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            tmp[0] = h2sin(tmp[0]);
            tmp[1] = h2sin(tmp[1]);
            return v;
        }
        #endif
        return Float4<T>(sin(v[0]), sin(v[1]), sin(v[2]), sin(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> sinc(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(sinc(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(sinc(v[0]), sinc(v[1]), sinc(v[2]), sinc(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> tan(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(tan(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(tan(v[0]), tan(v[1]), tan(v[2]), tan(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> acos(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(acos(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(acos(v[0]), acos(v[1]), acos(v[2]), acos(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> asin(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(asin(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(asin(v[0]), asin(v[1]), asin(v[2]), asin(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> atan(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(atan(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(atan(v[0]), atan(v[1]), atan(v[2]), atan(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> rad2deg(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(rad2deg(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(rad2deg(v[0]), rad2deg(v[1]), rad2deg(v[2]), rad2deg(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> deg2rad(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(deg2rad(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(deg2rad(v[0]), deg2rad(v[1]), deg2rad(v[2]), deg2rad(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> cosh(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(cosh(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(cosh(v[0]), cosh(v[1]), cosh(v[2]), cosh(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> sinh(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(sinh(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(sinh(v[0]), sinh(v[1]), sinh(v[2]), sinh(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> tanh(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(tanh(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(tanh(v[0]), tanh(v[1]), tanh(v[2]), tanh(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> acosh(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(acosh(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(acosh(v[0]), acosh(v[1]), acosh(v[2]), acosh(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> asinh(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(asinh(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(asinh(v[0]), asinh(v[1]), asinh(v[2]), asinh(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> atanh(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(atanh(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(atanh(v[0]), atanh(v[1]), atanh(v[2]), atanh(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> exp(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            tmp[0] = h2exp(tmp[0]);
            tmp[1] = h2exp(tmp[1]);
            return v;
        }
        #endif
        return Float4<T>(exp(v[0]), exp(v[1]), exp(v[2]), exp(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> log(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            tmp[0] = h2log(tmp[0]);
            tmp[1] = h2log(tmp[1]);
            return v;
        }
        #endif
        return Float4<T>(log(v[0]), log(v[1]), log(v[2]), log(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> log10(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            tmp[0] = h2log10(tmp[0]);
            tmp[1] = h2log10(tmp[1]);
            return v;
        }
        #endif
        return Float4<T>(log10(v[0]), log10(v[1]), log10(v[2]), log10(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> log1p(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(log1p(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return Float4<T>(log1p(v[0]), log1p(v[1]), log1p(v[2]), log1p(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> sqrt(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            tmp[0] = h2sqrt(tmp[0]);
            tmp[1] = h2sqrt(tmp[1]);
            return v;
        }
        #endif
        return Float4<T>(sqrt(v[0]), sqrt(v[1]), sqrt(v[2]), sqrt(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> rsqrt(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            tmp[0] = h2rsqrt(tmp[0]);
            tmp[1] = h2rsqrt(tmp[1]);
            return v;
        }
        #endif
        return Float4<T>(rsqrt(v[0]), rsqrt(v[1]), rsqrt(v[2]), rsqrt(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> round(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            tmp[0] = h2rint(tmp[0]); // h2rint is rounding to nearest
            tmp[1] = h2rint(tmp[1]);
            return v;
        }
        #endif
        return Float4<T>(round(v[0]), round(v[1]), round(v[2]), round(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> rint(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            tmp[0] = h2rint(tmp[0]);
            tmp[1] = h2rint(tmp[1]);
            return v;
        }
        #endif
        return Float4<T>(rint(v[0]), rint(v[1]), rint(v[2]), rint(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> ceil(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            tmp[0] = h2ceil(tmp[0]);
            tmp[1] = h2ceil(tmp[1]);
            return v;
        }
        #endif
        return Float4<T>(ceil(v[0]), ceil(v[1]), ceil(v[2]), ceil(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> floor(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            tmp[0] = h2floor(tmp[0]);
            tmp[1] = h2floor(tmp[1]);
            return v;
        }
        #endif
        return Float4<T>(floor(v[0]), floor(v[1]), floor(v[2]), floor(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> abs(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            tmp[0] = __habs2(tmp[0]);
            tmp[1] = __habs2(tmp[1]);
            return v;
        }
        #endif
        return Float4<T>(abs(v[0]), abs(v[1]), abs(v[2]), abs(v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T sum(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(sum(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return v[0] + v[1] + v[2] + v[3];
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T prod(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(prod(Float4<HALF_ARITHMETIC_TYPE>(v)));
        return v[0] * v[1] * v[2] * v[3];
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T dot(Float4<T> a, Float4<T> b) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(dot(Float4<HALF_ARITHMETIC_TYPE>(a), Float4<HALF_ARITHMETIC_TYPE>(b)));
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T norm(Float4<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>) {
            auto tmp = Float4<HALF_ARITHMETIC_TYPE>(v);
            return static_cast<T>(sqrt(dot(tmp, tmp)));
        }
        return sqrt(dot(v, v));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T length(Float4<T> v) noexcept {
        return norm(v);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> normalize(Float4<T> v) noexcept {
        return v / norm(v);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T min(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&v);
                tmp[0] = __hmin2(tmp[0], tmp[1]);
                return min(v[0], v[1]);
            }
        #endif
        return min(min(v[0], v[1]), min(v[2], v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> min(Float4<T> lhs, Float4<T> rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                tmp0[0] = __hmin2(tmp0[0], tmp1[0]);
                tmp0[1] = __hmin2(tmp0[1], tmp1[1]);
                return lhs;
            }
        #endif
        return {min(lhs[0], rhs[0]), min(lhs[1], rhs[1]), min(lhs[2], rhs[2]), min(lhs[3], rhs[3])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> min(Float4<T> lhs, T rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>)
                return min(lhs, Float4<T>(rhs));
        #endif
        return {min(lhs[0], rhs), min(lhs[1], rhs), min(lhs[2], rhs), min(lhs[3], rhs)};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> min(T lhs, Float4<T> rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>)
                return min(Float4<T>(lhs), rhs);
        #endif
        return {min(lhs, rhs[0]), min(lhs, rhs[1]), min(lhs, rhs[2]), min(lhs, rhs[3])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T max(Float4<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&v);
                tmp[0] = __hmax2(tmp[0], tmp[1]);
                return max(v[0], v[1]);
            }
        #endif
        return max(max(v[0], v[1]), max(v[2], v[3]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> max(Float4<T> lhs, Float4<T> rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                tmp0[0] = __hmax2(tmp0[0], tmp1[0]);
                tmp0[1] = __hmax2(tmp0[1], tmp1[1]);
                return lhs;
            }
        #endif
        return {max(lhs[0], rhs[0]), max(lhs[1], rhs[1]), max(lhs[2], rhs[2]), max(lhs[3], rhs[3])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> max(Float4<T> lhs, T rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>)
                return min(lhs, Float4<T>(rhs));
        #endif
        return {max(lhs[0], rhs), max(lhs[1], rhs), max(lhs[2], rhs), max(lhs[3], rhs)};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> max(T lhs, Float4<T> rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>)
                return min(Float4<T>(lhs), rhs);
        #endif
        return {max(lhs, rhs[0]), max(lhs, rhs[1]), max(lhs, rhs[2]), max(lhs, rhs[3])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> clamp(Float4<T> lhs, Float4<T> low, Float4<T> high) noexcept {
        return min(max(lhs, low), high);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> clamp(Float4<T> lhs, T low, T high) noexcept {
        return min(max(lhs, low), high);
    }

    #define NOA_ULP_ 2
    #define NOA_EPSILON_ 1e-6f

    template<uint ULP = NOA_ULP_, typename T>
    [[nodiscard]] NOA_FHD constexpr Bool4 isEqual(Float4<T> a, Float4<T> b, T e = NOA_EPSILON_) noexcept {
        return {isEqual<ULP>(a[0], b[0], e), isEqual<ULP>(a[1], b[1], e),
                isEqual<ULP>(a[2], b[2], e), isEqual<ULP>(a[3], b[3], e)};
    }

    template<uint ULP = NOA_ULP_, typename T>
    [[nodiscard]] NOA_FHD constexpr Bool4 isEqual(Float4<T> a, T b, T e = NOA_EPSILON_) noexcept {
        return {isEqual<ULP>(b, a[0], e), isEqual<ULP>(b, a[1], e),
                isEqual<ULP>(b, a[2], e), isEqual<ULP>(b, a[3], e)};
    }

    template<uint ULP = NOA_ULP_, typename T>
    [[nodiscard]] NOA_FHD constexpr Bool4 isEqual(T a, Float4<T> b, T e = NOA_EPSILON_) noexcept {
        return {isEqual<ULP>(a, b[0], e), isEqual<ULP>(a, b[1], e),
                isEqual<ULP>(a, b[2], e), isEqual<ULP>(a, b[3], e)};
    }

    #undef NOA_ULP_
    #undef NOA_EPSILON_

    template<typename T, typename U>
    [[nodiscard]] NOA_FHD constexpr Float4<T> sort(Float4<T> v, U&& comp) noexcept {
        smallStableSort<4>(v.get(), std::forward<U>(comp));
        return v;
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float4<T> sort(Float4<T> v) noexcept {
        return sort(v, [](const T& a, const T& b) { return a < b; });
    }
}
