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
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/string/Format.h"
#include "noa/common/types/Bool4.h"
#include "noa/common/types/Half.h"

namespace noa {
    template<typename>
    class Int4;

    template<typename T>
    class alignas(sizeof(T) * 4 >= 16 ? 16 : sizeof(T) * 4) Float4 {
    public:
        typedef T value_type;

    public: // Default Constructors
        constexpr Float4() noexcept = default;
        constexpr Float4(const Float4&) noexcept = default;
        constexpr Float4(Float4&&) noexcept = default;

    public: // Conversion constructors
        template<class X, class Y, class Z, class W>
        NOA_HD constexpr Float4(X x, Y y, Z z, W w) noexcept
                : m_data{static_cast<T>(x), static_cast<T>(y), static_cast<T>(z), static_cast<T>(w)} {}

        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Float4(U v) noexcept
                : m_data{static_cast<T>(v), static_cast<T>(v), static_cast<T>(v), static_cast<T>(v)} {}

        template<typename U>
        NOA_HD constexpr explicit Float4(Float4<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2]), static_cast<T>(v[3])} {}

        template<typename U>
        NOA_HD constexpr explicit Float4(Int4<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2]), static_cast<T>(v[3])} {}

        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Float4(const U* ptr) noexcept
                : m_data{static_cast<T>(ptr[0]), static_cast<T>(ptr[1]),
                         static_cast<T>(ptr[2]), static_cast<T>(ptr[3])} {}

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
        friend NOA_HD constexpr Float4 operator+(Float4 v) noexcept {
            return v;
        }

        friend NOA_HD constexpr Float4 operator-(Float4 v) noexcept {
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
        friend NOA_HD constexpr Float4 operator+(Float4 lhs, Float4 rhs) noexcept {
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

        friend NOA_HD constexpr Float4 operator+(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) + rhs;
            #endif
            return {lhs + rhs[0], lhs + rhs[1], lhs + rhs[2], lhs + rhs[3]};
        }

        friend NOA_HD constexpr Float4 operator+(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs + Float4(rhs);
            #endif
            return {lhs[0] + rhs, lhs[1] + rhs, lhs[2] + rhs, lhs[3] + rhs};
        }

        friend NOA_HD constexpr Float4 operator-(Float4 lhs, Float4 rhs) noexcept {
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

        friend NOA_HD constexpr Float4 operator-(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) - rhs;
            #endif
            return {lhs - rhs[0], lhs - rhs[1], lhs - rhs[2], lhs - rhs[3]};
        }

        friend NOA_HD constexpr Float4 operator-(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs - Float4(rhs);
            #endif
            return {lhs[0] - rhs, lhs[1] - rhs, lhs[2] - rhs, lhs[3] - rhs};
        }

        friend NOA_HD constexpr Float4 operator*(Float4 lhs, Float4 rhs) noexcept {
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

        friend NOA_HD constexpr Float4 operator*(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) * rhs;
            #endif
            return {lhs * rhs[0], lhs * rhs[1], lhs * rhs[2], lhs * rhs[3]};
        }

        friend NOA_HD constexpr Float4 operator*(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs * Float4(rhs);
            #endif
            return {lhs[0] * rhs, lhs[1] * rhs, lhs[2] * rhs, lhs[3] * rhs};
        }

        friend NOA_HD constexpr Float4 operator/(Float4 lhs, Float4 rhs) noexcept {
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

        friend NOA_HD constexpr Float4 operator/(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) / rhs;
            #endif
            return {lhs / rhs[0], lhs / rhs[1], lhs / rhs[2], lhs / rhs[3]};
        }

        friend NOA_HD constexpr Float4 operator/(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs / Float4(rhs);
            #endif
            return {lhs[0] / rhs, lhs[1] / rhs, lhs[2] / rhs, lhs[3] / rhs};
        }

        // -- Comparison Operators --
        friend NOA_HD constexpr Bool4 operator>(Float4 lhs, Float4 rhs) noexcept {
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

        friend NOA_HD constexpr Bool4 operator>(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs > Float4(rhs);
            #endif
            return {lhs[0] > rhs, lhs[1] > rhs, lhs[2] > rhs, lhs[3] > rhs};
        }

        friend NOA_HD constexpr Bool4 operator>(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) > rhs;
            #endif
            return {lhs > rhs[0], lhs > rhs[1], lhs > rhs[2], lhs > rhs[3]};
        }

        friend NOA_HD constexpr Bool4 operator<(Float4 lhs, Float4 rhs) noexcept {
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

        friend NOA_HD constexpr Bool4 operator<(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs < Float4(rhs);
            #endif
            return {lhs[0] < rhs, lhs[1] < rhs, lhs[2] < rhs, lhs[3] < rhs};
        }

        friend NOA_HD constexpr Bool4 operator<(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) < rhs;
            #endif
            return {lhs < rhs[0], lhs < rhs[1], lhs < rhs[2], lhs < rhs[3]};
        }

        friend NOA_HD constexpr Bool4 operator>=(Float4 lhs, Float4 rhs) noexcept {
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

        friend NOA_HD constexpr Bool4 operator>=(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs >= Float4(rhs);
            #endif
            return {lhs[0] >= rhs, lhs[1] >= rhs, lhs[2] >= rhs, lhs[3] >= rhs};
        }

        friend NOA_HD constexpr Bool4 operator>=(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) >= rhs;
            #endif
            return {lhs >= rhs[0], lhs >= rhs[1], lhs >= rhs[2], lhs >= rhs[3]};
        }

        friend NOA_HD constexpr Bool4 operator<=(Float4 lhs, Float4 rhs) noexcept {
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

        friend NOA_HD constexpr Bool4 operator<=(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs <= Float4(rhs);
            #endif
            return {lhs[0] <= rhs, lhs[1] <= rhs, lhs[2] <= rhs, lhs[3] <= rhs};
        }

        friend NOA_HD constexpr Bool4 operator<=(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) <= rhs;
            #endif
            return {lhs <= rhs[0], lhs <= rhs[1], lhs <= rhs[2], lhs <= rhs[3]};
        }

        friend NOA_HD constexpr Bool4 operator==(Float4 lhs, Float4 rhs) noexcept {
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

        friend NOA_HD constexpr Bool4 operator==(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs == Float4(rhs);
            #endif
            return {lhs[0] == rhs, lhs[1] == rhs, lhs[2] == rhs, lhs[3] == rhs};
        }

        friend NOA_HD constexpr Bool4 operator==(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) == rhs;
            #endif
            return {lhs == rhs[0], lhs == rhs[1], lhs == rhs[2], lhs == rhs[3]};
        }

        friend NOA_HD constexpr Bool4 operator!=(Float4 lhs, Float4 rhs) noexcept {
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

        friend NOA_HD constexpr Bool4 operator!=(Float4 lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return lhs != Float4(rhs);
            #endif
            return {lhs[0] != rhs, lhs[1] != rhs, lhs[2] != rhs, lhs[3] != rhs};
        }

        friend NOA_HD constexpr Bool4 operator!=(T lhs, Float4 rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return Float4(lhs) != rhs;
            #endif
            return {lhs != rhs[0], lhs != rhs[1], lhs != rhs[2], lhs != rhs[3]};
        }

    public: // Component accesses
        static constexpr size_t COUNT = 4;

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        NOA_HD constexpr T& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        NOA_HD constexpr const T& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        [[nodiscard]] NOA_HD constexpr const T* get() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr T* get() noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr Float4 flip() const noexcept { return {m_data[3], m_data[2], m_data[1], m_data[0]}; }

    private:
        static_assert(noa::traits::is_float_v<T>);
        T m_data[4]{};
    };

    namespace math {
        template<typename T>
        NOA_FHD constexpr Float4<T> floor(Float4<T> v) noexcept {
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
        NOA_FHD constexpr Float4<T> ceil(Float4<T> v) noexcept {
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
        NOA_FHD constexpr Float4<T> abs(Float4<T> v) noexcept {
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
        NOA_FHD constexpr T sum(Float4<T> v) noexcept {
            if constexpr (std::is_same_v<T, half_t>)
                return static_cast<T>(sum(Float4<HALF_ARITHMETIC_TYPE>(v)));
            return v[0] + v[1] + v[2] + v[3];
        }

        template<typename T>
        NOA_FHD constexpr T prod(Float4<T> v) noexcept {
            if constexpr (std::is_same_v<T, half_t>)
                return static_cast<T>(prod(Float4<HALF_ARITHMETIC_TYPE>(v)));
            return v[0] * v[1] * v[2] * v[3];
        }

        template<typename T>
        NOA_FHD constexpr T dot(Float4<T> a, Float4<T> b) noexcept {
            if constexpr (std::is_same_v<T, half_t>)
                return static_cast<T>(dot(Float4<HALF_ARITHMETIC_TYPE>(a), Float4<HALF_ARITHMETIC_TYPE>(b)));
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        }

        template<typename T>
        NOA_FHD constexpr T innerProduct(Float4<T> a, Float4<T> b) noexcept {
            return dot(a, b);
        }

        template<typename T>
        NOA_FHD constexpr T norm(Float4<T> v) noexcept {
            if constexpr (std::is_same_v<T, half_t>) {
                auto tmp = Float4<HALF_ARITHMETIC_TYPE>(v);
                return static_cast<T>(sqrt(dot(tmp, tmp)));
            }
            return sqrt(dot(v, v));
        }

        template<typename T>
        NOA_FHD constexpr T length(Float4<T> v) noexcept {
            return norm(v);
        }

        template<typename T>
        NOA_FHD constexpr Float4<T> normalize(Float4<T> v) noexcept {
            return v / norm(v);
        }

        template<typename T>
        NOA_FHD constexpr T min(Float4<T> v) noexcept {
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
        NOA_FHD constexpr Float4<T> min(Float4<T> lhs, Float4<T> rhs) noexcept {
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
        NOA_FHD constexpr Float4<T> min(Float4<T> lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (std::is_same_v<T, half_t>)
                return min(lhs, Float4<T>(rhs));
            #endif
            return {min(lhs[0], rhs), min(lhs[1], rhs), min(lhs[2], rhs), min(lhs[3], rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float4<T> min(T lhs, Float4<T> rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (std::is_same_v<T, half_t>)
                return min(Float4<T>(lhs), rhs);
            #endif
            return {min(lhs, rhs[0]), min(lhs, rhs[1]), min(lhs, rhs[2]), min(lhs, rhs[3])};
        }

        template<typename T>
        NOA_FHD constexpr T max(Float4<T> v) noexcept {
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
        NOA_FHD constexpr Float4<T> max(Float4<T> lhs, Float4<T> rhs) noexcept {
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
        NOA_FHD constexpr Float4<T> max(Float4<T> lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (std::is_same_v<T, half_t>)
                return min(lhs, Float4<T>(rhs));
            #endif
            return {max(lhs[0], rhs), max(lhs[1], rhs), max(lhs[2], rhs), max(lhs[3], rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float4<T> max(T lhs, Float4<T> rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if constexpr (std::is_same_v<T, half_t>)
                return min(Float4<T>(lhs), rhs);
            #endif
            return {max(lhs, rhs[0]), max(lhs, rhs[1]), max(lhs, rhs[2]), max(lhs, rhs[3])};
        }

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool4 isEqual(Float4<T> a, Float4<T> b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(a[0], b[0], e), isEqual<ULP>(a[1], b[1], e),
                    isEqual<ULP>(a[2], b[2], e), isEqual<ULP>(a[3], b[3], e)};
        }

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool4 isEqual(Float4<T> a, T b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(b, a[0], e), isEqual<ULP>(b, a[1], e),
                    isEqual<ULP>(b, a[2], e), isEqual<ULP>(b, a[3], e)};
        }

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool4 isEqual(T a, Float4<T> b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(a, b[0], e), isEqual<ULP>(a, b[1], e),
                    isEqual<ULP>(a, b[2], e), isEqual<ULP>(a, b[3], e)};
        }

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace traits {
        template<typename T>
        struct p_is_float4 : std::false_type {};
        template<typename T>
        struct p_is_float4<noa::Float4<T>> : std::true_type {};
        template<typename T> using is_float4 = std::bool_constant<p_is_float4<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_float4_v = is_float4<T>::value;

        template<typename T>
        struct proclaim_is_floatX<noa::Float4<T>> : std::true_type {};
    }

    using half4_t = Float4<half_t>;
    using float4_t = Float4<float>;
    using double4_t = Float4<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 4> toArray(Float4<T> v) noexcept {
        return {v[0], v[1], v[2], v[3]};
    }

    template<>
    NOA_IH std::string string::typeName<half4_t>() { return "half4"; }
    template<>
    NOA_IH std::string string::typeName<float4_t>() { return "float4"; }
    template<>
    NOA_IH std::string string::typeName<double4_t>() { return "double4"; }

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
