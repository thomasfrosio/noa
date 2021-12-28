/// \file noa/common/types/Float3.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 3 floating-point numbers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/string/Format.h"
#include "noa/common/types/Bool3.h"
#include "noa/common/types/Half.h"

namespace noa {
    template<typename>
    class Int3;

    template<typename T>
    class Float3 {
    public:
        static_assert(noa::traits::is_float_v<T>);
        typedef T value_type;
        T x{}, y{}, z{};

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

    public: // Default Constructors
        constexpr Float3() noexcept = default;
        constexpr Float3(const Float3&) noexcept = default;
        constexpr Float3(Float3&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y, typename Z>
        NOA_HD constexpr Float3(X xi, Y yi, Z zi) noexcept
                : x(static_cast<T>(xi)),
                  y(static_cast<T>(yi)),
                  z(static_cast<T>(zi)) {}

        template<typename U>
        NOA_HD constexpr explicit Float3(U v) noexcept
                : x(static_cast<T>(v)),
                  y(static_cast<T>(v)),
                  z(static_cast<T>(v)) {}

        template<typename U>
        NOA_HD constexpr explicit Float3(Float3<U> v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)),
                  z(static_cast<T>(v.z)) {}

        template<typename U>
        NOA_HD constexpr explicit Float3(Int3<U> v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)),
                  z(static_cast<T>(v.z)) {}

        template<typename U>
        NOA_HD constexpr explicit Float3(U* ptr) noexcept
                : x(static_cast<T>(ptr[0])),
                  y(static_cast<T>(ptr[1])),
                  z(static_cast<T>(ptr[2])) {}

    public: // Assignment operators
        constexpr Float3& operator=(const Float3& v) noexcept = default;
        constexpr Float3& operator=(Float3&& v) noexcept = default;

        NOA_HD constexpr Float3& operator=(T v) noexcept {
            this->x = v;
            this->y = v;
            this->z = v;
            return *this;
        }

        NOA_HD constexpr Float3& operator=(T* ptr) noexcept {
            this->x = ptr[0];
            this->y = ptr[1];
            this->z = ptr[2];
            return *this;
        }

        NOA_HD constexpr Float3& operator+=(Float3 rhs) noexcept {
            *this = *this + rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator-=(Float3 rhs) noexcept {
            *this = *this - rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator*=(Float3 rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator/=(Float3 rhs) noexcept {
            *this = *this / rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator+=(T rhs) noexcept {
            *this = *this + rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator-=(T rhs) noexcept {
            *this = *this - rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator*=(T rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator/=(T rhs) noexcept {
            *this = *this / rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        friend NOA_HD constexpr Float3 operator+(Float3 v) noexcept {
            return v;
        }

        friend NOA_HD constexpr Float3 operator-(Float3 v) noexcept {
            return {-v.x, -v.y, -v.z};
        }

        // -- Binary Arithmetic Operators --
        friend NOA_HD constexpr Float3 operator+(Float3 lhs, Float3 rhs) noexcept {
            return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
        }

        friend NOA_HD constexpr Float3 operator+(T lhs, Float3 rhs) noexcept {
            return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z};
        }

        friend NOA_HD constexpr Float3 operator+(Float3 lhs, T rhs) noexcept {
            return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs};
        }

        friend NOA_HD constexpr Float3 operator-(Float3 lhs, Float3 rhs) noexcept {
            return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
        }

        friend NOA_HD constexpr Float3 operator-(T lhs, Float3 rhs) noexcept {
            return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z};
        }

        friend NOA_HD constexpr Float3 operator-(Float3 lhs, T rhs) noexcept {
            return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs};
        }

        friend NOA_HD constexpr Float3 operator*(Float3 lhs, Float3 rhs) noexcept {
            return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
        }

        friend NOA_HD constexpr Float3 operator*(T lhs, Float3 rhs) noexcept {
            return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
        }

        friend NOA_HD constexpr Float3 operator*(Float3 lhs, T rhs) noexcept {
            return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
        }

        friend NOA_HD constexpr Float3 operator/(Float3 lhs, Float3 rhs) noexcept {
            return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z};
        }

        friend NOA_HD constexpr Float3 operator/(T lhs, Float3 rhs) noexcept {
            return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z};
        }

        friend NOA_HD constexpr Float3 operator/(Float3 lhs, T rhs) noexcept {
            return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
        }

        // -- Comparison Operators --
        friend NOA_HD constexpr Bool3 operator>(Float3 lhs, Float3 rhs) noexcept {
            return {lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator>(Float3 lhs, T rhs) noexcept {
            return {lhs.x > rhs, lhs.y > rhs, lhs.z > rhs};
        }

        friend NOA_HD constexpr Bool3 operator>(T lhs, Float3 rhs) noexcept {
            return {lhs > rhs.x, lhs > rhs.y, lhs > rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator<(Float3 lhs, Float3 rhs) noexcept {
            return {lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator<(Float3 lhs, T rhs) noexcept {
            return {lhs.x < rhs, lhs.y < rhs, lhs.z < rhs};
        }

        friend NOA_HD constexpr Bool3 operator<(T lhs, Float3 rhs) noexcept {
            return {lhs < rhs.x, lhs < rhs.y, lhs < rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator>=(Float3 lhs, Float3 rhs) noexcept {
            return {lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator>=(Float3 lhs, T rhs) noexcept {
            return {lhs.x >= rhs, lhs.y >= rhs, lhs.z >= rhs};
        }

        friend NOA_HD constexpr Bool3 operator>=(T lhs, Float3 rhs) noexcept {
            return {lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator<=(Float3 lhs, Float3 rhs) noexcept {
            return {lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator<=(Float3 lhs, T rhs) noexcept {
            return {lhs.x <= rhs, lhs.y <= rhs, lhs.z <= rhs};
        }

        friend NOA_HD constexpr Bool3 operator<=(T lhs, Float3 rhs) noexcept {
            return {lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator==(Float3 lhs, Float3 rhs) noexcept {
            return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator==(Float3 lhs, T rhs) noexcept {
            return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs};
        }

        friend NOA_HD constexpr Bool3 operator==(T lhs, Float3 rhs) noexcept {
            return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator!=(Float3 lhs, Float3 rhs) noexcept {
            return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z};
        }

        friend NOA_HD constexpr Bool3 operator!=(Float3 lhs, T rhs) noexcept {
            return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs};
        }

        friend NOA_HD constexpr Bool3 operator!=(T lhs, Float3 rhs) noexcept {
            return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z};
        }
    };

    namespace math {
        template<typename T>
        NOA_FHD constexpr Float3<T> toRad(Float3<T> v) noexcept {
            return Float3<T>(toRad(v.x), toRad(v.y), toRad(v.z));
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> toDeg(Float3<T> v) noexcept {
            return Float3<T>(toDeg(v.x), toDeg(v.y), toDeg(v.z));
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> floor(Float3<T> v) noexcept {
            return Float3<T>(floor(v.x), floor(v.y), floor(v.z));
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> ceil(Float3<T> v) noexcept {
            return Float3<T>(ceil(v.x), ceil(v.y), ceil(v.z));
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> abs(Float3<T> v) noexcept {
            return Float3<T>(abs(v.x), abs(v.y), abs(v.z));
        }

        template<typename T>
        NOA_FHD constexpr T sum(Float3<T> v) noexcept {
            if constexpr (std::is_same_v<T, half_t>)
                return static_cast<T>(sum(Float3<HALF_ARITHMETIC_TYPE>(v)));
            return v.x + v.y + v.z;
        }

        template<typename T>
        NOA_FHD constexpr T prod(Float3<T> v) noexcept {
            if constexpr (std::is_same_v<T, half_t>)
                return static_cast<T>(prod(Float3<HALF_ARITHMETIC_TYPE>(v)));
            return v.x * v.y * v.z;
        }

        template<typename T>
        NOA_FHD constexpr T dot(Float3<T> a, Float3<T> b) noexcept {
            if constexpr (std::is_same_v<T, half_t>)
                return static_cast<T>(dot(Float3<HALF_ARITHMETIC_TYPE>(a), Float3<HALF_ARITHMETIC_TYPE>(b)));
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        template<typename T>
        NOA_FHD constexpr T innerProduct(Float3<T> a, Float3<T> b) noexcept {
            return dot(a, b);
        }

        template<typename T>
        NOA_FHD constexpr T norm(Float3<T> v) noexcept {
            if constexpr (std::is_same_v<T, half_t>) {
                auto tmp = Float3<HALF_ARITHMETIC_TYPE>(v);
                return static_cast<T>(sqrt(dot(tmp, tmp)));
            }
            return sqrt(dot(v, v));
        }

        template<typename T>
        NOA_FHD constexpr T length(Float3<T> v) noexcept {
            return norm(v);
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> normalize(Float3<T> v) noexcept {
            return v / norm(v);
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> cross(Float3<T> a, Float3<T> b) noexcept {
            if constexpr (std::is_same_v<T, half_t>)
                return Float3<T>(cross(Float3<HALF_ARITHMETIC_TYPE>(a), Float3<HALF_ARITHMETIC_TYPE>(b)));
            return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
        }

        template<typename T>
        NOA_FHD constexpr T min(Float3<T> v) noexcept {
            return (v.x < v.y) ? min(v.x, v.z) : min(v.y, v.z);
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> min(Float3<T> lhs, Float3<T> rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> min(Float3<T> lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> min(T lhs, Float3<T> rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z)};
        }

        template<typename T>
        NOA_FHD constexpr T max(Float3<T> v) noexcept {
            return (v.x > v.y) ? max(v.x, v.z) : max(v.y, v.z);
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> max(Float3<T> lhs, Float3<T> rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> max(Float3<T> lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> max(T lhs, Float3<T> rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z)};
        }

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool3 isEqual(Float3<T> a, Float3<T> b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(a.x, b.x, e), isEqual<ULP>(a.y, b.y, e), isEqual<ULP>(a.z, b.z, e)};
        }

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool3 isEqual(Float3<T> a, T b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(a.x, b, e), isEqual<ULP>(a.y, b, e), isEqual<ULP>(a.z, b, e)};
        }

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool3 isEqual(T a, Float3<T> b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(a, b.x, e), isEqual<ULP>(a, b.y, e), isEqual<ULP>(a, b.z, e)};
        }

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace traits {
        template<typename T>
        struct p_is_float3 : std::false_type {};
        template<typename T>
        struct p_is_float3<noa::Float3<T>> : std::true_type {};
        template<typename T> using is_float3 = std::bool_constant<p_is_float3<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_float3_v = is_float3<T>::value;

        template<typename T>
        struct proclaim_is_floatX<noa::Float3<T>> : std::true_type {};
    }

    using half3_t = Float3<half_t>;
    using float3_t = Float3<float>;
    using double3_t = Float3<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 3> toArray(Float3<T> v) noexcept {
        return {v.x, v.y, v.z};
    }

    template<>
    NOA_IH std::string string::typeName<half3_t>() { return "half3"; }
    template<>
    NOA_IH std::string string::typeName<float3_t>() { return "float3"; }
    template<>
    NOA_IH std::string string::typeName<double3_t>() { return "double3"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, Float3<T> v) {
        os << string::format("({:.3f},{:.3f},{:.3f})", v.x, v.y, v.z);
        return os;
    }
}

namespace fmt {
    template<typename T>
    struct formatter<noa::Float3<T>> : formatter<T> {
        template<typename FormatContext>
        auto format(const noa::Float3<T>& vec, FormatContext& ctx) {
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
