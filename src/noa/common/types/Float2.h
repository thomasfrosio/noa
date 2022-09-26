/// \file noa/common/types/Float2.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 2 floating-point numbers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/ArrayTypes.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Bool2.h"
#include "noa/common/types/ClampCast.h"
#include "noa/common/types/Half.h"
#include "noa/common/types/SafeCast.h"
#include "noa/common/utils/Sort.h"

namespace noa {
    template<typename>
    class Int2;

    template<typename T>
    class alignas(sizeof(T) * 2) Float2 {
    public:
        using value_type = T;

    public: // Default constructors
        constexpr Float2() noexcept = default;
        constexpr Float2(const Float2&) noexcept = default;
        constexpr Float2(Float2&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y, typename = std::enable_if_t<traits::is_scalar_v<X> && traits::is_scalar_v<Y>>>
        NOA_HD constexpr Float2(X a0, Y a1) noexcept
                : m_data{static_cast<T>(a0), static_cast<T>(a1)} {
            NOA_ASSERT(isSafeCast<T>(a0) && isSafeCast<T>(a1));
        }

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Float2(U x) noexcept
                : m_data{static_cast<T>(x), static_cast<T>(x)} {
            NOA_ASSERT(isSafeCast<T>(x));
        }

        template<typename U>
        NOA_HD constexpr explicit Float2(Float2<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1])} {
            NOA_ASSERT(isSafeCast<T>(v[0]) && isSafeCast<T>(v[1]));
        }

        NOA_HD constexpr explicit Float2(Bool2 v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1])} {}

        template<typename U>
        NOA_HD constexpr explicit Float2(Int2<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1])} {
            NOA_ASSERT(isSafeCast<T>(v[0]) && isSafeCast<T>(v[1]));
        }

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Float2(U* ptr) noexcept {
            NOA_ASSERT(ptr != nullptr);
            for (size_t i = 0; i < COUNT; ++i) {
                NOA_ASSERT(isSafeCast<T>(ptr[i]));
                m_data[i] = static_cast<T>(ptr[i]);
            }
        }

    public: // Assignment operators
        constexpr Float2& operator=(const Float2& v) noexcept = default;
        constexpr Float2& operator=(Float2&& v) noexcept = default;

        NOA_HD constexpr Float2& operator=(T v) noexcept {
            m_data[0] = v;
            m_data[1] = v;
            return *this;
        }

        NOA_HD constexpr Float2& operator+=(Float2 rhs) noexcept {
            *this = *this + rhs;
            return *this;
        }

        NOA_HD constexpr Float2& operator-=(Float2 rhs) noexcept {
            *this = *this - rhs;
            return *this;
        }

        NOA_HD constexpr Float2& operator*=(Float2 rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        NOA_HD constexpr Float2& operator/=(Float2 rhs) noexcept {
            *this = *this / rhs;
            return *this;
        }

        NOA_HD constexpr Float2& operator+=(T rhs) noexcept {
            *this = *this + rhs;
            return *this;
        }

        NOA_HD constexpr Float2& operator-=(T rhs) noexcept {
            *this = *this - rhs;
            return *this;
        }

        NOA_HD constexpr Float2& operator*=(T rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        NOA_HD constexpr Float2& operator/=(T rhs) noexcept {
            *this = *this / rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Float2 operator+(Float2 v) noexcept {
            return v;
        }

        [[nodiscard]] friend NOA_HD constexpr Float2 operator-(Float2 v) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&v);
                *tmp = -(*tmp);
                return v;
            }
            #endif
            return {-v[0], -v[1]};
        }

        // -- Binary Arithmetic Operators --
        [[nodiscard]] friend NOA_HD constexpr Float2 operator+(Float2 lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp += *reinterpret_cast<__half2*>(&rhs);
                return lhs;
            }
            #endif
            return {lhs[0] + rhs[0], lhs[1] + rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float2 operator+(T lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&rhs);
                *tmp += __half2half2(lhs.native());
                return rhs;
            }
            #endif
            return {lhs + rhs[0], lhs + rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float2 operator+(Float2 lhs, T rhs) noexcept {
            return rhs + lhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Float2 operator-(Float2 lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp -= *reinterpret_cast<__half2*>(&rhs);
                return lhs;
            }
            #endif
            return {lhs[0] - rhs[0], lhs[1] - rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float2 operator-(T lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                __half2 tmp = __half2half2(lhs.native());
                tmp -= *reinterpret_cast<__half2*>(&rhs);
                return *reinterpret_cast<Float2*>(&tmp);
            }
            #endif
            return {lhs - rhs[0], lhs - rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float2 operator-(Float2 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp -= __half2half2(rhs.native());
                return lhs;
            }
            #endif
            return {lhs[0] - rhs, lhs[1] - rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Float2 operator*(Float2 lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp *= *reinterpret_cast<__half2*>(&rhs);
                return lhs;
            }
            #endif
            return {lhs[0] * rhs[0], lhs[1] * rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float2 operator*(T lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&rhs);
                *tmp *= __half2half2(lhs.native());
                return rhs;
            }
            #endif
            return {lhs * rhs[0], lhs * rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float2 operator*(Float2 lhs, T rhs) noexcept {
            return rhs * lhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Float2 operator/(Float2 lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp /= *reinterpret_cast<__half2*>(&rhs);
                return lhs;
            }
            #endif
            return {lhs[0] / rhs[0], lhs[1] / rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float2 operator/(T lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                __half2 tmp = __half2half2(lhs.native());
                tmp /= *reinterpret_cast<__half2*>(&rhs);
                return *reinterpret_cast<Float2*>(&tmp);
            }
            #endif
            return {lhs / rhs[0], lhs / rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float2 operator/(Float2 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp -= __half2half2(rhs.native());
                return lhs;
            }
            #endif
            return {lhs[0] / rhs, lhs[1] / rhs};
        }

        // -- Comparison Operators --
        [[nodiscard]] friend NOA_HD constexpr Bool2 operator>(Float2 lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                *tmp0 = __hgt2(*tmp0, *tmp1);
                return Bool2(lhs[0], lhs[1]);
            }
            #endif
            return {lhs[0] > rhs[0], lhs[1] > rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator>(Float2 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs > Float2(rhs);
            #endif
            return {lhs[0] > rhs, lhs[1] > rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator>(T lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float2(lhs) > rhs;
            #endif
            return {lhs > rhs[0], lhs > rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator<(Float2 lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                *tmp0 = __hlt2(*tmp0, *tmp1);
                return Bool2(lhs[0], lhs[1]);
            }
            #endif
            return {lhs[0] < rhs[0], lhs[1] < rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator<(Float2 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs > Float2(rhs);
            #endif
            return {lhs[0] < rhs, lhs[1] < rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator<(T lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float2(lhs) > rhs;
            #endif
            return {lhs < rhs[0], lhs < rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator>=(Float2 lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                *tmp0 = __hge2(*tmp0, *tmp1);
                return Bool2(lhs[0], lhs[1]);
            }
            #endif
            return {lhs[0] >= rhs[0], lhs[1] >= rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator>=(Float2 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs > Float2(rhs);
            #endif
            return {lhs[0] >= rhs, lhs[1] >= rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator>=(T lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float2(lhs) > rhs;
            #endif
            return {lhs >= rhs[0], lhs >= rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator<=(Float2 lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                *tmp0 = __hle2(*tmp0, *tmp1);
                return Bool2(lhs[0], lhs[1]);
            }
            #endif
            return {lhs[0] <= rhs[0], lhs[1] <= rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator<=(Float2 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs > Float2(rhs);
            #endif
            return {lhs[0] <= rhs, lhs[1] <= rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator<=(T lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float2(lhs) > rhs;
            #endif
            return {lhs <= rhs[0], lhs <= rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator==(Float2 lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                *tmp0 = __heq2(*tmp0, *tmp1);
                return Bool2(lhs[0], lhs[1]);
            }
            #endif
            return {lhs[0] == rhs[0], lhs[1] == rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator==(Float2 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs > Float2(rhs);
            #endif
            return {lhs[0] == rhs, lhs[1] == rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator==(T lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float2(lhs) > rhs;
            #endif
            return {lhs == rhs[0], lhs == rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator!=(Float2 lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp0 = reinterpret_cast<__half2*>(&lhs);
                auto* tmp1 = reinterpret_cast<__half2*>(&rhs);
                *tmp0 = __hne2(*tmp0, *tmp1);
                return Bool2(lhs[0], lhs[1]);
            }
            #endif
            return {lhs[0] != rhs[0], lhs[1] != rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator!=(Float2 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs > Float2(rhs);
            #endif
            return {lhs[0] != rhs, lhs[1] != rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator!=(T lhs, Float2 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float2(lhs) > rhs;
            #endif
            return {lhs != rhs[0], lhs != rhs[1]};
        }

    public: // Component accesses
        static constexpr size_t COUNT = 2;

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

        [[nodiscard]] NOA_HD constexpr Float2 flip() const noexcept { return {m_data[1], m_data[0]}; }

    private:
        static_assert(traits::is_float_v<T>);
        T m_data[2]{};
    };

    template<typename T>
    struct traits::proclaim_is_float2<Float2<T>> : std::true_type {};

    using half2_t = Float2<half_t>;
    using float2_t = Float2<float>;
    using double2_t = Float2<double>;

    template<typename T>
    [[nodiscard]] NOA_IH constexpr std::array<T, 2> toArray(Float2<T> v) noexcept {
        return {v[0], v[1]};
    }

    template<>
    [[nodiscard]] NOA_IH std::string string::human<half2_t>() { return "half2"; }
    template<>
    [[nodiscard]] NOA_IH std::string string::human<float2_t>() { return "float2"; }
    template<>
    [[nodiscard]] NOA_IH std::string string::human<double2_t>() { return "double2"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, Float2<T> v) {
        os << string::format("({:.3f},{:.3f})", v[0], v[1]);
        return os;
    }
}

namespace fmt {
    template<typename T>
    struct formatter<noa::Float2<T>> : formatter<T> {
        template<typename FormatContext>
        auto format(const noa::Float2<T>& vec, FormatContext& ctx) {
            auto out = ctx.out();
            *out = '(';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[0], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[1], ctx);
            *out = ')';
            return out;
        }
    };
}

namespace noa {
    template<typename TTo, typename TFrom, typename = std::enable_if_t<traits::is_float2_v<TTo>>>
    [[nodiscard]] NOA_FHD constexpr TTo clamp_cast(const Float2<TFrom>& src) noexcept {
        using value_t = traits::value_type_t<TTo>;
        return {clamp_cast<value_t>(src[0]), clamp_cast<value_t>(src[1])};
    }

    template<typename TTo, typename TFrom, typename = std::enable_if_t<traits::is_float2_v<TTo>>>
    [[nodiscard]] NOA_FHD constexpr bool isSafeCast(const Float2<TFrom>& src) noexcept {
        using value_t = traits::value_type_t<TTo>;
        return isSafeCast<value_t>(src[0]) && isSafeCast<value_t>(src[1]);
    }
}

namespace noa::math {
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> cos(Float2<T> v) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            *tmp = h2cos(*tmp);
            return v;
        }
        #endif
        return Float2<T>(cos(v[0]), cos(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> sin(Float2<T> v) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            *tmp = h2sin(*tmp);
            return v;
        }
        #endif
        return Float2<T>(sin(v[0]), sin(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> sinc(Float2<T> v) {
        return Float2<T>(sinc(v[0]), sinc(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> tan(Float2<T> v) {
        return Float2<T>(tan(v[0]), tan(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> acos(Float2<T> v) {
        return Float2<T>(acos(v[0]), acos(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> asin(Float2<T> v) {
        return Float2<T>(asin(v[0]), asin(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T atan2(Float2<T> v) {
        return atan2(v[0], v[1]);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> rad2deg(Float2<T> v) noexcept {
        return Float2<T>(rad2deg(v[0]), rad2deg(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> deg2rad(Float2<T> v) noexcept {
        return Float2<T>(deg2rad(v[0]), deg2rad(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> cosh(Float2<T> v) {
        return Float2<T>(cosh(v[0]), cosh(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> sinh(Float2<T> v) {
        return Float2<T>(sinh(v[0]), sinh(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> tanh(Float2<T> v) {
        return Float2<T>(tanh(v[0]), tanh(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> acosh(Float2<T> v) {
        return Float2<T>(acosh(v[0]), acosh(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> asinh(Float2<T> v) {
        return Float2<T>(asinh(v[0]), asinh(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> atanh(Float2<T> v) {
        return Float2<T>(atanh(v[0]), atanh(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> exp(Float2<T> v) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            *tmp = h2exp(*tmp);
            return v;
        }
        #endif
        return Float2<T>(exp(v[0]), exp(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> log(Float2<T> v) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            *tmp = h2log(*tmp);
            return v;
        }
        #endif
        return Float2<T>(log(v[0]), log(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> log10(Float2<T> v) {
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            *tmp = h2log10(*tmp);
            return v;
        }
        #endif
        return Float2<T>(log10(v[0]), log10(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> log1p(Float2<T> v) {
        return Float2<T>(log1p(v[0]), log1p(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> sqrt(Float2<T> v) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            *tmp = h2sqrt(*tmp);
            return v;
        }
        #endif
        return Float2<T>(sqrt(v[0]), sqrt(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> rsqrt(Float2<T> v) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            *tmp = h2rsqrt(*tmp);
            return v;
        }
        #endif
        return Float2<T>(rsqrt(v[0]), rsqrt(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> rint(Float2<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            *tmp = h2rint(*tmp);
            return v;
        }
        #endif
        return Float2<T>(rint(v[0]), rint(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> ceil(Float2<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            *tmp = h2ceil(*tmp);
            return v;
        }
        #endif
        return Float2<T>(ceil(v[0]), ceil(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> floor(Float2<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            *tmp = h2floor(*tmp);
            return v;
        }
        #endif
        return Float2<T>(floor(v[0]), floor(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> abs(Float2<T> v) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&v);
            *tmp = __habs2(*tmp);
            return v;
        }
        #endif
        return Float2<T>(abs(v[0]), abs(v[1]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T dot(Float2<T> a, Float2<T> b) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return fma(a[0], b[0], a[1] * b[1]);
        return a[0] * b[0] + a[1] * b[1];
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T norm(Float2<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>) {
            Float2<float> tmp(v);
            return {sqrt(dot(tmp, tmp))};
        }
        return sqrt(dot(v, v));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T length(Float2<T> v) noexcept {
        return norm(v);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> normalize(Float2<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>) {
            Float2<float> tmp(v);
            return {tmp / sqrt(dot(tmp, tmp))};
        }
        return v / norm(v);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T sum(Float2<T> v) noexcept {
        return v[0] + v[1];
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T prod(Float2<T> v) noexcept {
        return v[0] * v[1];
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T min(Float2<T> v) noexcept {
        return min(v[0], v[1]);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> min(Float2<T> lhs, Float2<T> rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&lhs);
            *tmp = __hmin2(*tmp, *reinterpret_cast<__half2*>(&rhs));
            return lhs;
        }
        #endif
        return {min(lhs[0], rhs[0]), min(lhs[1], rhs[1])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> min(Float2<T> lhs, T rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&lhs);
            *tmp = __hmin2(*tmp, __half2half2(rhs.native()));
            return lhs;
        }
        #endif
        return {min(lhs[0], rhs), min(lhs[1], rhs)};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> min(T lhs, Float2<T> rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&rhs);
            *tmp = __hmin2(*tmp, __half2half2(lhs.native()));
            return rhs;
        }
        #endif
        return {min(lhs, rhs[0]), min(lhs, rhs[1])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T max(Float2<T> v) noexcept {
        return max(v[0], v[1]);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> max(Float2<T> lhs, Float2<T> rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&lhs);
            *tmp = __hmax2(*tmp, *reinterpret_cast<__half2*>(&rhs));
            return lhs;
        }
        #endif
        return {max(lhs[0], rhs[0]), max(lhs[1], rhs[1])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> max(Float2<T> lhs, T rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&lhs);
            *tmp = __hmax2(*tmp, __half2half2(rhs.native()));
            return lhs;
        }
        #endif
        return {max(lhs[0], rhs), max(lhs[1], rhs)};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> max(T lhs, Float2<T> rhs) noexcept {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (std::is_same_v<T, half_t>) {
            auto* tmp = reinterpret_cast<__half2*>(&rhs);
            *tmp = __hmax2(*tmp, __half2half2(lhs.native()));
            return rhs;
        }
        #endif
        return {max(lhs, rhs[0]), max(lhs, rhs[1])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> clamp(Float2<T> lhs, Float2<T> low, Float2<T> high) noexcept {
        return min(max(lhs, low), high);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> clamp(Float2<T> lhs, T low, T high) noexcept {
        return min(max(lhs, low), high);
    }

    #define NOA_ULP_ 2
    #define NOA_EPSILON_ 1e-6f

    template<uint ULP = NOA_ULP_, typename T>
    [[nodiscard]] NOA_FHD constexpr Bool2 isEqual(Float2<T> a, Float2<T> b, T e = NOA_EPSILON_) noexcept {
        return {isEqual<ULP>(a[0], b[0], e), isEqual<ULP>(a[1], b[1], e)};
    }

    template<uint ULP = NOA_ULP_, typename T>
    [[nodiscard]] NOA_FHD constexpr Bool2 isEqual(Float2<T> a, T b, T e = NOA_EPSILON_) noexcept {
        return {isEqual<ULP>(a[0], b, e), isEqual<ULP>(a[1], b, e)};
    }

    template<uint ULP = NOA_ULP_, typename T>
    [[nodiscard]] NOA_FHD constexpr Bool2 isEqual(T a, Float2<T> b, T e = NOA_EPSILON_) noexcept {
        return {isEqual<ULP>(a, b[0], e), isEqual<ULP>(a, b[1], e)};
    }

    #undef NOA_ULP_
    #undef NOA_EPSILON_

    template<typename T, typename U>
    [[nodiscard]] NOA_FHD constexpr Float2<T> sort(Float2<T> v, U&& comp) noexcept {
        smallStableSort<2>(v.get(), std::forward<U>(comp));
        return v;
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> sort(Float2<T> v) noexcept {
        return sort(v, [](const T& a, const T& b) { return a < b; });
    }
}
