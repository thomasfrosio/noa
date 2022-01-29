/// \file noa/common/types/Int3.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 3 integers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Bool3.h"

namespace noa {
    template<typename>
    class Float3;

    template<typename T>
    class Int3 {
    public:
        typedef T value_type;

    public: // Default Constructors
        constexpr Int3() noexcept = default;
        constexpr Int3(const Int3&) noexcept = default;
        constexpr Int3(Int3&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y, typename Z>
        NOA_HD constexpr Int3(X x, Y y, Z z) noexcept
                : m_data{static_cast<T>(x), static_cast<T>(y), static_cast<T>(z)} {}

        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Int3(U x) noexcept
                : m_data{static_cast<T>(x), static_cast<T>(x), static_cast<T>(x)} {}

        NOA_HD constexpr explicit Int3(Bool3 v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2])} {}

        template<typename U>
        NOA_HD constexpr explicit Int3(Int3<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2])} {}

        template<typename U>
        NOA_HD constexpr explicit Int3(Float3<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2])} {}

        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Int3(const U* ptr) noexcept
                : m_data{static_cast<T>(ptr[0]), static_cast<T>(ptr[1]), static_cast<T>(ptr[2])} {}

    public: // Assignment operators
        constexpr Int3& operator=(const Int3& v) noexcept = default;
        constexpr Int3& operator=(Int3&& v) noexcept = default;

        NOA_HD constexpr Int3& operator=(T v) noexcept {
            m_data[0] = v;
            m_data[1] = v;
            m_data[2] = v;
            return *this;
        }

        NOA_HD constexpr Int3& operator+=(Int3 rhs) noexcept {
            m_data[0] += rhs[0];
            m_data[1] += rhs[1];
            m_data[2] += rhs[2];
            return *this;
        }

        NOA_HD constexpr Int3& operator-=(Int3 rhs) noexcept {
            m_data[0] -= rhs[0];
            m_data[1] -= rhs[1];
            m_data[2] -= rhs[2];
            return *this;
        }

        NOA_HD constexpr Int3& operator*=(Int3 rhs) noexcept {
            m_data[0] *= rhs[0];
            m_data[1] *= rhs[1];
            m_data[2] *= rhs[2];
            return *this;
        }

        NOA_HD constexpr Int3& operator/=(Int3 rhs) noexcept {
            m_data[0] /= rhs[0];
            m_data[1] /= rhs[1];
            m_data[2] /= rhs[2];
            return *this;
        }

        NOA_HD constexpr Int3& operator+=(T rhs) noexcept {
            m_data[0] += rhs;
            m_data[1] += rhs;
            m_data[2] += rhs;
            return *this;
        }

        NOA_HD constexpr Int3& operator-=(T rhs) noexcept {
            m_data[0] -= rhs;
            m_data[1] -= rhs;
            m_data[2] -= rhs;
            return *this;
        }

        NOA_HD constexpr Int3& operator*=(T rhs) noexcept {
            m_data[0] *= rhs;
            m_data[1] *= rhs;
            m_data[2] *= rhs;
            return *this;
        }

        NOA_HD constexpr Int3& operator/=(T rhs) noexcept {
            m_data[0] /= rhs;
            m_data[1] /= rhs;
            m_data[2] /= rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        friend NOA_HD constexpr Int3 operator+(Int3 v) noexcept {
            return v;
        }

        friend NOA_HD constexpr Int3 operator-(Int3 v) noexcept {
            return {-v[0], -v[1], -v[2]};
        }

        // -- Binary Arithmetic Operators --
        friend NOA_HD constexpr Int3 operator+(Int3 lhs, Int3 rhs) noexcept {
            return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
        }

        friend NOA_HD constexpr Int3 operator+(T lhs, Int3 rhs) noexcept {
            return {lhs + rhs[0], lhs + rhs[1], lhs + rhs[2]};
        }

        friend NOA_HD constexpr Int3 operator+(Int3 lhs, T rhs) noexcept {
            return {lhs[0] + rhs, lhs[1] + rhs, lhs[2] + rhs};
        }

        friend NOA_HD constexpr Int3 operator-(Int3 lhs, Int3 rhs) noexcept {
            return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
        }

        friend NOA_HD constexpr Int3 operator-(T lhs, Int3 rhs) noexcept {
            return {lhs - rhs[0], lhs - rhs[1], lhs - rhs[2]};
        }

        friend NOA_HD constexpr Int3 operator-(Int3 lhs, T rhs) noexcept {
            return {lhs[0] - rhs, lhs[1] - rhs, lhs[2] - rhs};
        }

        friend NOA_HD constexpr Int3 operator*(Int3 lhs, Int3 rhs) noexcept {
            return {lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2]};
        }

        friend NOA_HD constexpr Int3 operator*(T lhs, Int3 rhs) noexcept {
            return {lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]};
        }

        friend NOA_HD constexpr Int3 operator*(Int3 lhs, T rhs) noexcept {
            return {lhs[0] * rhs, lhs[1] * rhs, lhs[2] * rhs};
        }

        friend NOA_HD constexpr Int3 operator/(Int3 lhs, Int3 rhs) noexcept {
            return {lhs[0] / rhs[0], lhs[1] / rhs[1], lhs[2] / rhs[2]};
        }

        friend NOA_HD constexpr Int3 operator/(T lhs, Int3 rhs) noexcept {
            return {lhs / rhs[0], lhs / rhs[1], lhs / rhs[2]};
        }

        friend NOA_HD constexpr Int3 operator/(Int3 lhs, T rhs) noexcept {
            return {lhs[0] / rhs, lhs[1] / rhs, lhs[2] / rhs};
        }

        // -- Comparison Operators --
        friend NOA_HD constexpr Bool3 operator>(Int3 lhs, Int3 rhs) noexcept {
            return {lhs[0] > rhs[0], lhs[1] > rhs[1], lhs[2] > rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator>(Int3 lhs, T rhs) noexcept {
            return {lhs[0] > rhs, lhs[1] > rhs, lhs[2] > rhs};
        }

        friend NOA_HD constexpr Bool3 operator>(T lhs, Int3 rhs) noexcept {
            return {lhs > rhs[0], lhs > rhs[1], lhs > rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator<(Int3 lhs, Int3 rhs) noexcept {
            return {lhs[0] < rhs[0], lhs[1] < rhs[1], lhs[2] < rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator<(Int3 lhs, T rhs) noexcept {
            return {lhs[0] < rhs, lhs[1] < rhs, lhs[2] < rhs};
        }

        friend NOA_HD constexpr Bool3 operator<(T lhs, Int3 rhs) noexcept {
            return {lhs < rhs[0], lhs < rhs[1], lhs < rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator>=(Int3 lhs, Int3 rhs) noexcept {
            return {lhs[0] >= rhs[0], lhs[1] >= rhs[1], lhs[2] >= rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator>=(Int3 lhs, T rhs) noexcept {
            return {lhs[0] >= rhs, lhs[1] >= rhs, lhs[2] >= rhs};
        }

        friend NOA_HD constexpr Bool3 operator>=(T lhs, Int3 rhs) noexcept {
            return {lhs >= rhs[0], lhs >= rhs[1], lhs >= rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator<=(Int3 lhs, Int3 rhs) noexcept {
            return {lhs[0] <= rhs[0], lhs[1] <= rhs[1], lhs[2] <= rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator<=(Int3 lhs, T rhs) noexcept {
            return {lhs[0] <= rhs, lhs[1] <= rhs, lhs[2] <= rhs};
        }

        friend NOA_HD constexpr Bool3 operator<=(T lhs, Int3 rhs) noexcept {
            return {lhs <= rhs[0], lhs <= rhs[1], lhs <= rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator==(Int3 lhs, Int3 rhs) noexcept {
            return {lhs[0] == rhs[0], lhs[1] == rhs[1], lhs[2] == rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator==(Int3 lhs, T rhs) noexcept {
            return {lhs[0] == rhs, lhs[1] == rhs, lhs[2] == rhs};
        }

        friend NOA_HD constexpr Bool3 operator==(T lhs, Int3 rhs) noexcept {
            return {lhs == rhs[0], lhs == rhs[1], lhs == rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator!=(Int3 lhs, Int3 rhs) noexcept {
            return {lhs[0] != rhs[0], lhs[1] != rhs[1], lhs[2] != rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator!=(Int3 lhs, T rhs) noexcept {
            return {lhs[0] != rhs, lhs[1] != rhs, lhs[2] != rhs};
        }

        friend NOA_HD constexpr Bool3 operator!=(T lhs, Int3 rhs) noexcept {
            return {lhs != rhs[0], lhs != rhs[1], lhs != rhs[2]};
        }

        // -- Other Operators --
        friend NOA_HD constexpr Int3 operator%(Int3 lhs, Int3 rhs) noexcept {
            return {lhs[0] % rhs[0], lhs[1] % rhs[1], lhs[2] % rhs[2]};
        }

        friend NOA_HD constexpr Int3 operator%(Int3 lhs, T rhs) noexcept {
            return {lhs[0] % rhs, lhs[1] % rhs, lhs[2] % rhs};
        }

        friend NOA_HD constexpr Int3 operator%(T lhs, Int3 rhs) noexcept {
            return {lhs % rhs[0], lhs % rhs[1], lhs % rhs[2]};
        }

    public: // Component accesses
        static constexpr size_t COUNT = 3;

        NOA_HD constexpr T& operator[](size_t i) noexcept {
            NOA_ASSERT(i < COUNT);
            return m_data[i];
        }

        NOA_HD constexpr const T& operator[](size_t i) const noexcept {
            NOA_ASSERT(i < COUNT);
            return m_data[i];
        }

        NOA_HD [[nodiscard]] constexpr const T* get() const noexcept { return m_data; }
        NOA_HD [[nodiscard]] constexpr T* get() noexcept { return m_data; }
        NOA_HD [[nodiscard]] constexpr Int3 flip() const noexcept { return {m_data[2], m_data[1], m_data[0]}; }

        NOA_HD [[nodiscard]] constexpr T ndim() const noexcept {
            NOA_ASSERT(all(*this >= T{1}));
            return m_data[0] > 1 ? 3 :
                   m_data[1] > 1 ? 2 : 1;
        }

        NOA_HD [[nodiscard]] constexpr Int3 strides() const noexcept {
            return {m_data[2] * m_data[1],
                    m_data[2],
                    1};
        }

        NOA_HD [[nodiscard]] constexpr Int2<T> pitches() const noexcept {
            NOA_ASSERT(all(*this != 0) && "Cannot recover pitch from stride 0");
            return {m_data[0] / m_data[1], m_data[1]}; // assuming strides
        }

        NOA_HD [[nodiscard]] constexpr T elements() const noexcept {
            return m_data[0] * m_data[1] * m_data[2];
        }

        NOA_HD [[nodiscard]] constexpr Int3 fft() const noexcept {
            return {m_data[0], m_data[1], m_data[2] / 2 + 1};
        }

    public:
        static_assert(noa::traits::is_int_v<T> && !noa::traits::is_bool_v<T>);
        T m_data[3]{};
    };

    namespace math {
        template<typename T>
        NOA_FHD constexpr T sum(Int3<T> v) noexcept {
            return v[0] + v[1] + v[2];
        }

        template<typename T>
        NOA_FHD constexpr T prod(Int3<T> v) noexcept {
            return v[0] * v[1] * v[2];
        }

        template<typename T>
        NOA_FHD constexpr T min(Int3<T> v) noexcept {
            return (v[0] < v[1]) ? min(v[0], v[2]) : min(v[1], v[2]);
        }

        template<typename T>
        NOA_FHD constexpr Int3<T> min(Int3<T> lhs, Int3<T> rhs) noexcept {
            return {min(lhs[0], rhs[0]), min(lhs[1], rhs[1]), min(lhs[2], rhs[2])};
        }

        template<typename T>
        NOA_FHD constexpr Int3<T> min(Int3<T> lhs, T rhs) noexcept {
            return {min(lhs[0], rhs), min(lhs[1], rhs), min(lhs[2], rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Int3<T> min(T lhs, Int3<T> rhs) noexcept {
            return {min(lhs, rhs[0]), min(lhs, rhs[1]), min(lhs, rhs[2])};
        }

        template<typename T>
        NOA_FHD constexpr T max(Int3<T> v) noexcept {
            return (v[0] > v[1]) ? max(v[0], v[2]) : max(v[1], v[2]);
        }

        template<typename T>
        NOA_FHD constexpr Int3<T> max(Int3<T> lhs, Int3<T> rhs) noexcept {
            return {max(lhs[0], rhs[0]), max(lhs[1], rhs[1]), max(lhs[2], rhs[2])};
        }

        template<typename T>
        NOA_FHD constexpr Int3<T> max(Int3<T> lhs, T rhs) noexcept {
            return {max(lhs[0], rhs), max(lhs[1], rhs), max(lhs[2], rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Int3<T> max(T lhs, Int3<T> rhs) noexcept {
            return {max(lhs, rhs[0]), max(lhs, rhs[1]), max(lhs, rhs[2])};
        }
    }

    namespace traits {
        template<typename>
        struct p_is_int3 : std::false_type {};
        template<typename T>
        struct p_is_int3<noa::Int3<T>> : std::true_type {};
        template<typename T> using is_int3 = std::bool_constant<p_is_int3<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_int3_v = is_int3<T>::value;

        template<typename>
        struct p_is_uint3 : std::false_type {};
        template<typename T>
        struct p_is_uint3<noa::Int3<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
        template<typename T> using is_uint3 = std::bool_constant<p_is_uint3<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_uint3_v = is_uint3<T>::value;

        template<typename T>
        struct proclaim_is_intX<noa::Int3<T>> : std::true_type {};
        template<typename T>
        struct proclaim_is_uintX<noa::Int3<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
    }

    using int3_t = Int3<int>;
    using uint3_t = Int3<uint>;
    using long3_t = Int3<int64_t>;
    using ulong3_t = Int3<uint64_t>;
    using size3_t = Int3<size_t>;

    template<typename T>
    NOA_IH constexpr std::array<T, 3> toArray(Int3<T> v) noexcept {
        return {v[0], v[1], v[2]};
    }

    template<>
    NOA_IH std::string string::typeName<int3_t>() { return "int3"; }
    template<>
    NOA_IH std::string string::typeName<uint3_t>() { return "uint3"; }
    template<>
    NOA_IH std::string string::typeName<long3_t>() { return "long3"; }
    template<>
    NOA_IH std::string string::typeName<ulong3_t>() { return "ulong3"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, Int3<T> v) {
        os << string::format("({},{},{})", v[0], v[1], v[2]);
        return os;
    }
}

namespace fmt {
    template<typename T>
    struct formatter<noa::Int3<T>> : formatter<T> {
        template<typename FormatContext>
        auto format(const noa::Int3<T>& vec, FormatContext& ctx) {
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
            *out = ')';
            return out;
        }
    };
}
