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

    template<typename>
    class Int2;

    template<typename>
    class Int4;

    template<typename T>
    class Int3 {
    public: // Default Constructors
        constexpr Int3() noexcept = default;
        constexpr Int3(const Int3&) noexcept = default;
        constexpr Int3(Int3&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y, typename Z>
        NOA_HD constexpr Int3(X xi, Y yi, Z zi) noexcept
                : x(static_cast<T>(xi)),
                  y(static_cast<T>(yi)),
                  z(static_cast<T>(zi)) {}

        template<typename U>
        NOA_HD constexpr explicit Int3(U v) noexcept
                : x(static_cast<T>(v)),
                  y(static_cast<T>(v)),
                  z(static_cast<T>(v)) {}

        NOA_HD constexpr explicit Int3(Bool3 v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)),
                  z(static_cast<T>(v.z)) {}

        template<typename U>
        NOA_HD constexpr explicit Int3(Int3<U> v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)),
                  z(static_cast<T>(v.z)) {}

        template<typename U>
        NOA_HD constexpr explicit Int3(Float3<U> v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)),
                  z(static_cast<T>(v.z)) {}

        template<typename U>
        NOA_HD constexpr explicit Int3(U* ptr) noexcept
                : x(static_cast<T>(ptr[0])),
                  y(static_cast<T>(ptr[1])),
                  z(static_cast<T>(ptr[2])) {}

        template<typename U, typename V>
        NOA_HD constexpr explicit Int3(Int4<U> v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)),
                  z(static_cast<T>(v.z)) {}

        template<typename U, typename V>
        NOA_HD constexpr explicit Int3(Int2<U> v, V oz = V(0)) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)),
                  z(static_cast<T>(oz)) {}

    public: // Assignment operators
        constexpr Int3& operator=(const Int3& v) noexcept = default;
        constexpr Int3& operator=(Int3&& v) noexcept = default;

        NOA_HD constexpr Int3& operator=(T v) noexcept {
            this->x = v;
            this->y = v;
            this->z = v;
            return *this;
        }

        NOA_HD constexpr Int3& operator=(T* ptr) noexcept {
            this->x = ptr[0];
            this->y = ptr[1];
            this->z = ptr[2];
            return *this;
        }

        NOA_HD constexpr Int3& operator+=(Int3 rhs) noexcept {
            this->x += rhs.x;
            this->y += rhs.y;
            this->z += rhs.z;
            return *this;
        }

        NOA_HD constexpr Int3& operator-=(Int3 rhs) noexcept {
            this->x -= rhs.x;
            this->y -= rhs.y;
            this->z -= rhs.z;
            return *this;
        }

        NOA_HD constexpr Int3& operator*=(Int3 rhs) noexcept {
            this->x *= rhs.x;
            this->y *= rhs.y;
            this->z *= rhs.z;
            return *this;
        }

        NOA_HD constexpr Int3& operator/=(Int3 rhs) noexcept {
            this->x /= rhs.x;
            this->y /= rhs.y;
            this->z /= rhs.z;
            return *this;
        }

        NOA_HD constexpr Int3& operator+=(T rhs) noexcept {
            this->x += rhs;
            this->y += rhs;
            this->z += rhs;
            return *this;
        }

        NOA_HD constexpr Int3& operator-=(T rhs) noexcept {
            this->x -= rhs;
            this->y -= rhs;
            this->z -= rhs;
            return *this;
        }

        NOA_HD constexpr Int3& operator*=(T rhs) noexcept {
            this->x *= rhs;
            this->y *= rhs;
            this->z *= rhs;
            return *this;
        }

        NOA_HD constexpr Int3& operator/=(T rhs) noexcept {
            this->x /= rhs;
            this->y /= rhs;
            this->z /= rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        friend NOA_HD constexpr Int3 operator+(Int3 v) noexcept {
            return v;
        }

        friend NOA_HD constexpr Int3 operator-(Int3 v) noexcept {
            return {-v.x, -v.y, -v.z};
        }

        // -- Binary Arithmetic Operators --
        friend NOA_HD constexpr Int3 operator+(Int3 lhs, Int3 rhs) noexcept {
            return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
        }

        friend NOA_HD constexpr Int3 operator+(T lhs, Int3 rhs) noexcept {
            return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z};
        }

        friend NOA_HD constexpr Int3 operator+(Int3 lhs, T rhs) noexcept {
            return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs};
        }

        friend NOA_HD constexpr Int3 operator-(Int3 lhs, Int3 rhs) noexcept {
            return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
        }

        friend NOA_HD constexpr Int3 operator-(T lhs, Int3 rhs) noexcept {
            return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z};
        }

        friend NOA_HD constexpr Int3 operator-(Int3 lhs, T rhs) noexcept {
            return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs};
        }

        friend NOA_HD constexpr Int3 operator*(Int3 lhs, Int3 rhs) noexcept {
            return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
        }

        friend NOA_HD constexpr Int3 operator*(T lhs, Int3 rhs) noexcept {
            return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
        }

        friend NOA_HD constexpr Int3 operator*(Int3 lhs, T rhs) noexcept {
            return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
        }

        friend NOA_HD constexpr Int3 operator/(Int3 lhs, Int3 rhs) noexcept {
            return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z};
        }

        friend NOA_HD constexpr Int3 operator/(T lhs, Int3 rhs) noexcept {
            return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z};
        }

        friend NOA_HD constexpr Int3 operator/(Int3 lhs, T rhs) noexcept {
            return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
        }

        // -- Comparison Operators --
        friend NOA_HD constexpr Bool3 operator>(Int3 lhs, Int3 rhs) noexcept {
            return {lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator>(Int3 lhs, T rhs) noexcept {
            return {lhs.x > rhs, lhs.y > rhs, lhs.z > rhs};
        }

        friend NOA_HD constexpr Bool3 operator>(T lhs, Int3 rhs) noexcept {
            return {lhs > rhs.x, lhs > rhs.y, lhs > rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator<(Int3 lhs, Int3 rhs) noexcept {
            return {lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator<(Int3 lhs, T rhs) noexcept {
            return {lhs.x < rhs, lhs.y < rhs, lhs.z < rhs};
        }

        friend NOA_HD constexpr Bool3 operator<(T lhs, Int3 rhs) noexcept {
            return {lhs < rhs.x, lhs < rhs.y, lhs < rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator>=(Int3 lhs, Int3 rhs) noexcept {
            return {lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator>=(Int3 lhs, T rhs) noexcept {
            return {lhs.x >= rhs, lhs.y >= rhs, lhs.z >= rhs};
        }

        friend NOA_HD constexpr Bool3 operator>=(T lhs, Int3 rhs) noexcept {
            return {lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator<=(Int3 lhs, Int3 rhs) noexcept {
            return {lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator<=(Int3 lhs, T rhs) noexcept {
            return {lhs.x <= rhs, lhs.y <= rhs, lhs.z <= rhs};
        }

        friend NOA_HD constexpr Bool3 operator<=(T lhs, Int3 rhs) noexcept {
            return {lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator==(Int3 lhs, Int3 rhs) noexcept {
            return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator==(Int3 lhs, T rhs) noexcept {
            return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs};
        }

        friend NOA_HD constexpr Bool3 operator==(T lhs, Int3 rhs) noexcept {
            return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator!=(Int3 lhs, Int3 rhs) noexcept {
            return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator!=(Int3 lhs, T rhs) noexcept {
            return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs};
        }

        friend NOA_HD constexpr Bool3 operator!=(T lhs, Int3 rhs) noexcept {
            return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z};
        }

        // -- Other Operators --

        friend NOA_HD constexpr Int3 operator%(Int3 lhs, Int3 rhs) noexcept {
            return {lhs.x % rhs.x, lhs.y % rhs.y, lhs.z % rhs.z};
        }

        friend NOA_HD constexpr Int3 operator%(Int3 lhs, T rhs) noexcept {
            return {lhs.x % rhs, lhs.y % rhs, lhs.z % rhs};
        }

        friend NOA_HD constexpr Int3 operator%(T lhs, Int3 rhs) noexcept {
            return {lhs % rhs.x, lhs % rhs.y, lhs % rhs.z};
        }

    public: // Component accesses
        static constexpr size_t COUNT = 3;

        NOA_HD constexpr T& operator[](size_t i) noexcept {
            NOA_ASSERT(i < this->COUNT);
            switch (i) {
                default:
                case 0:
                    return this->x;
                case 1:
                    return this->y;
                case 2:
                    return this->z;
            }
        }

        NOA_HD constexpr const T& operator[](size_t i) const noexcept {
            NOA_ASSERT(i < this->COUNT);
            switch (i) {
                default:
                case 0:
                    return this->x;
                case 1:
                    return this->y;
                case 2:
                    return this->z;
            }
        }

        NOA_FHD constexpr T ndim() const noexcept {
            NOA_ASSERT(all(*this >= T{1}));
            return z > 1 ? 3 :
                   y > 1 ? 2 : 1;
        }

        template<typename I = T>
        NOA_FHD constexpr Int3<I> strides() const noexcept {
            Int3<I> out{1};
            for (size_t i = 1; i < COUNT; ++i)
                out[i] = out[i - 1] * static_cast<I>(this->operator[](i - 1));
            return out;
        }

        template<typename I = T>
        NOA_FHD constexpr I elements() const noexcept {
            return static_cast<I>(x) * static_cast<I>(y) * static_cast<I>(z);
        }

        template<typename I = T>
        NOA_FHD constexpr Int3<I> stridesFFT() const noexcept {
            return shapeFFT().strides();
        }

        template<typename I = T>
        NOA_FHD constexpr I elementsFFT() const noexcept {
            return static_cast<I>(x / 2 + 1) * static_cast<I>(y) * static_cast<I>(z);
        }

        template<typename I= T>
        NOA_FHD constexpr Int3<I> shapeFFT() const noexcept {
            return {static_cast<I>(x / 2 + 1), static_cast<I>(y), static_cast<I>(z)};
        }

    public:
        static_assert(noa::traits::is_int_v<T> && !noa::traits::is_bool_v<T>);
        typedef T value_t;
        T x{}, y{}, z{};
    };

    template<typename T>
    NOA_FHD constexpr T elements(Int3<T> v) noexcept {
        return v.x * v.y * v.z;
    }

    template<typename T>
    NOA_FHD constexpr T elementsSlice(Int3<T> v) noexcept {
        return v.x * v.y;
    }

    template<typename T>
    NOA_FHD constexpr T elementsFFT(Int3<T> v) noexcept {
        return (v.x / 2 + 1) * v.y * v.z;
    }

    template<typename T>
    NOA_FHD constexpr Int3<T> shapeFFT(Int3<T> v) noexcept {
        return {v.x / 2 + 1, v.y, v.z};
    }

    template<typename T>
    NOA_FHD constexpr Int3<T> slice(Int3<T> v) noexcept {
        return {v.x, v.y, 1};
    }

    namespace math {
        template<typename T>
        NOA_FHD constexpr T sum(Int3<T> v) noexcept {
            return v.x + v.y + v.z;
        }

        template<typename T>
        NOA_FHD constexpr T prod(Int3<T> v) noexcept {
            return v.x * v.y * v.z;
        }

        template<typename T>
        NOA_FHD constexpr T min(Int3<T> v) noexcept {
            return (v.x < v.y) ? min(v.x, v.z) : min(v.y, v.z);
        }

        template<typename T>
        NOA_FHD constexpr Int3<T> min(Int3<T> lhs, Int3<T> rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)};
        }

        template<typename T>
        NOA_FHD constexpr Int3<T> min(Int3<T> lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Int3<T> min(T lhs, Int3<T> rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z)};
        }

        template<typename T>
        NOA_FHD constexpr T max(Int3<T> v) noexcept {
            return (v.x > v.y) ? max(v.x, v.z) : max(v.y, v.z);
        }

        template<typename T>
        NOA_FHD constexpr Int3<T> max(Int3<T> lhs, Int3<T> rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)};
        }

        template<typename T>
        NOA_FHD constexpr Int3<T> max(Int3<T> lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Int3<T> max(T lhs, Int3<T> rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z)};
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
        return {v.x, v.y, v.z};
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
        os << string::format("({},{},{})", v.x, v.y, v.z);
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
            out = formatter<T>::format(vec.x, ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec.y, ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec.z, ctx);
            *out = ')';
            return out;
        }
    };
}
