/**
 * @file FloatX.h
 * @brief Static arrays of floating-points.
 * @author Thomas - ffyr2w
 * @date 10/12/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/traits/Base.h"
#include "noa/util/string/Format.h"
#include "noa/util/Math.h"


namespace Noa {
    template<typename, typename>
    struct Int2;

    /** Static array of 2 floating-points. */
    template<typename Float = float, typename = std::enable_if_t<Noa::Traits::is_float_v<Float>>>
    struct Float2 {
        Float x{0}, y{0};

        // Constructors.
        constexpr Float2() = default;
        constexpr Float2(Float xi, Float yi) : x(xi), y(yi) {}

        constexpr explicit Float2(Float v) : x(v), y(v) {}
        constexpr explicit Float2(Float* ptr) : x(ptr[0]), y(ptr[1]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_float_v<U>>>
        constexpr explicit Float2(U* ptr) : x(Float(ptr[0])), y(Float(ptr[1])) {}

        template<typename U>
        constexpr explicit Float2(Float2<U> v) : x(Float(v.x)), y(Float(v.y)) {}

        template<typename U, typename V>
        constexpr explicit Float2(Int2<U, V> v) : x(Float(v.x)), y(Float(v.y)) {}

        // Assignment operators.
        inline constexpr auto& operator=(Float v) noexcept {
            x = v;
            y = v;
            return *this;
        }

        inline constexpr auto& operator=(Float* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        inline constexpr auto& operator=(U* ptr) noexcept {
            x = Float(ptr[0]);
            y = Float(ptr[1]);
            return *this;
        }

        template<typename U>
        inline constexpr auto& operator=(Float2<U> v) noexcept {
            x = Float(v.x);
            y = Float(v.y);
            return *this;
        }

        template<typename U, typename V>
        inline constexpr auto& operator=(Int2<U, V> v) noexcept {
            x = Float(v.x);
            y = Float(v.y);
            return *this;
        }

        //@CLION-formatter:off
        inline constexpr Float2<Float> operator*(Float2<Float> v) const noexcept { return {x * v.x, y * v.y}; }
        inline constexpr Float2<Float> operator/(Float2<Float> v) const noexcept { return {x / v.x, y / v.y}; }
        inline constexpr Float2<Float> operator+(Float2<Float> v) const noexcept { return {x + v.x, y + v.y}; }
        inline constexpr Float2<Float> operator-(Float2<Float> v) const noexcept { return {x - v.x, y - v.y}; }

        inline constexpr void operator*=(Float2<Float> v) noexcept { x *= v.x; y *= v.y; }
        inline constexpr void operator/=(Float2<Float> v) noexcept { x /= v.x; y /= v.y; }
        inline constexpr void operator+=(Float2<Float> v) noexcept { x += v.x; y += v.y; }
        inline constexpr void operator-=(Float2<Float> v) noexcept { x -= v.x; y -= v.y; }

        inline constexpr bool operator>(Float2<Float> v) const noexcept { return x > v.x && y > v.y; }
        inline constexpr bool operator<(Float2<Float> v) const noexcept { return x < v.x && y < v.y; }
        inline constexpr bool operator>=(Float2<Float> v) const noexcept { return x >= v.x && y >= v.y; }
        inline constexpr bool operator<=(Float2<Float> v) const noexcept { return x <= v.x && y <= v.y; }
        inline constexpr bool operator==(Float2<Float> v) const noexcept { return x == v.x && y == v.y; }
        inline constexpr bool operator!=(Float2<Float> v) const noexcept { return x != v.x || y != v.y; }

        inline constexpr Float2<Float> operator*(Float v) const noexcept { return {x * v, y * v}; }
        inline constexpr Float2<Float> operator/(Float v) const noexcept { return {x / v, y / v}; }
        inline constexpr Float2<Float> operator+(Float v) const noexcept { return {x + v, y + v}; }
        inline constexpr Float2<Float> operator-(Float v) const noexcept { return {x - v, y - v}; }

        inline constexpr void operator*=(Float v) noexcept { x *= v; y *= v; }
        inline constexpr void operator/=(Float v) noexcept { x /= v; y /= v; }
        inline constexpr void operator+=(Float v) noexcept { x += v; y += v; }
        inline constexpr void operator-=(Float v) noexcept { x -= v; y -= v; }

        inline constexpr bool operator>(Float v) const noexcept { return x > v && y > v; }
        inline constexpr bool operator<(Float v) const noexcept { return x < v && y < v; }
        inline constexpr bool operator>=(Float v) const noexcept { return x >= v && y >= v; }
        inline constexpr bool operator<=(Float v) const noexcept { return x <= v && y <= v; }
        inline constexpr bool operator==(Float v) const noexcept { return x == v && y == v; }
        inline constexpr bool operator!=(Float v) const noexcept { return x != v || y != v; }

        [[nodiscard]] inline constexpr Float2<Float> floor() const { return Float2<Float>(std::floor(x), std::floor(y)); }
        [[nodiscard]] inline constexpr Float2<Float> ceil() const { return Float2<Float>(std::ceil(x), std::ceil(y)); }

        [[nodiscard]] inline constexpr Float lengthSq() const noexcept { return x * x + y * y; }
        [[nodiscard]] inline constexpr Float length() const { return std::sqrt(lengthSq()); }
        [[nodiscard]] inline constexpr Float2<Float> normalize() const { return *this / length(); }

        [[nodiscard]] inline constexpr Float sum() const noexcept { return x + y; }
        [[nodiscard]] inline constexpr Float prod() const noexcept { return x * y; }

        [[nodiscard]] static inline constexpr size_t size() noexcept { return 2U; }

        [[nodiscard]] inline constexpr Float dot(Float2<Float> v) const noexcept { return x * v.x + y * v.y; }
        //@CLION-formatter:on

        [[nodiscard]] inline constexpr std::array<Float, 2U> toArray() const noexcept {
            return {x, y};
        }

        [[nodiscard]] inline std::string toString() const {
            return String::format("({}, {})", x, y);
        }
    };

    namespace Math {
        template<typename T>
        inline Float2<T> min(Float2<T> v1, Float2<T> v2) noexcept {
            return Float2<T>(std::min(v1.x, v2.x), std::min(v1.y, v2.y));
        }

        template<typename T>
        inline Float2<T> min(Float2<T> v1, T v2) noexcept {
            return Float2<T>(std::min(v1.x, v2), std::min(v1.y, v2));
        }

        template<typename T>
        inline Float2<T> max(Float2<T> v1, Float2<T> v2) noexcept {
            return Float2<T>(std::max(v1.x, v2.x), std::max(v1.y, v2.y));
        }

        template<typename T>
        inline Float2<T> max(Float2<T> v1, T v2) noexcept {
            return Float2<T>(std::max(v1.x, v2), std::max(v1.y, v2));
        }
    }
}


namespace Noa {
    template<typename, typename>
    struct Int3;

    /** Static array of 3 floating-points. */
    template<typename Float = float, typename = std::enable_if_t<Noa::Traits::is_float_v<Float>>>
    struct Float3 {
        Float x{0}, y{0}, z{0};

        // Constructors.
        constexpr Float3() = default;
        constexpr Float3(Float xi, Float yi, Float zi) : x(xi), y(yi), z(zi) {}

        constexpr explicit Float3(Float v) : x(v), y(v), z(v) {}
        constexpr explicit Float3(Float* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr explicit Float3(U* ptr) : x(Float(ptr[0])), y(Float(ptr[1])), z(Float(ptr[2])) {}

        template<typename U>
        constexpr explicit Float3(Float3<U> v) : x(Float(v.x)), y(Float(v.y)), z(Float(v.z)) {}

        template<typename U, typename V>
        constexpr explicit Float3(Int3<U, V> v) : x(Float(v.x)), y(Float(v.y)), z(Float(v.z)) {}

        // Assignment operators.
        inline constexpr auto& operator=(Float v) noexcept {
            x = v;
            y = v;
            z = v;
            return *this;
        }
        inline constexpr auto& operator=(Float* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            z = ptr[2];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        inline constexpr auto& operator=(U* ptr) noexcept {
            x = Float(ptr[0]);
            y = Float(ptr[1]);
            z = Float(ptr[2]);
            return *this;
        }

        template<typename U>
        inline constexpr auto& operator=(Float3<U> v) noexcept {
            x = Float(v.x);
            y = Float(v.y);
            z = Float(v.z);
            return *this;
        }

        template<typename U, typename V>
        inline constexpr auto& operator=(Int3<U, V> v) noexcept {
            x = Float(v.x);
            y = Float(v.y);
            z = Float(v.z);
            return *this;
        }

        //@CLION-formatter:off
        inline constexpr Float3<Float> operator*(Float3<Float> v) const noexcept { return {x * v.x, y * v.y, z * v.z}; }
        inline constexpr Float3<Float> operator/(Float3<Float> v) const noexcept { return {x / v.x, y / v.y, z / v.z}; }
        inline constexpr Float3<Float> operator+(Float3<Float> v) const noexcept { return {x + v.x, y + v.y, z + v.z}; }
        inline constexpr Float3<Float> operator-(Float3<Float> v) const noexcept { return {x - v.x, y - v.y, z - v.z}; }

        inline constexpr void operator*=(Float3<Float> v) noexcept { x *= v.x; y *= v.y; z *= v.z; }
        inline constexpr void operator/=(Float3<Float> v) noexcept { x /= v.x; y /= v.y; z /= v.z; }
        inline constexpr void operator+=(Float3<Float> v) noexcept { x += v.x; y += v.y; z += v.z; }
        inline constexpr void operator-=(Float3<Float> v) noexcept { x -= v.x; y -= v.y; z -= v.z; }

        inline constexpr bool operator>(Float3<Float> v) const noexcept { return x > v.x && y > v.y && z > v.z; }
        inline constexpr bool operator<(Float3<Float> v) const noexcept { return x < v.x && y < v.y && z < v.z; }
        inline constexpr bool operator>=(Float3<Float> v) const noexcept { return x >= v.x && y >= v.y && z >= v.z; }
        inline constexpr bool operator<=(Float3<Float> v) const noexcept { return x <= v.x && y <= v.y && z <= v.z; }
        inline constexpr bool operator==(Float3<Float> v) const noexcept { return x == v.x && y == v.y && z == v.z; }
        inline constexpr bool operator!=(Float3<Float> v) const noexcept { return x != v.x || y != v.y || z != v.z; }

        inline constexpr Float3<Float> operator*(Float v) const noexcept { return {x * v, y * v, z * v}; }
        inline constexpr Float3<Float> operator/(Float v) const noexcept { return {x / v, y / v, z / v}; }
        inline constexpr Float3<Float> operator+(Float v) const noexcept { return {x + v, y + v, z + v}; }
        inline constexpr Float3<Float> operator-(Float v) const noexcept { return {x - v, y - v, z - v}; }

        inline constexpr void operator*=(Float v) noexcept { x *= v; y *= v; z *= v; }
        inline constexpr void operator/=(Float v) noexcept { x /= v; y /= v; z /= v; }
        inline constexpr void operator+=(Float v) noexcept { x += v; y += v; z += v; }
        inline constexpr void operator-=(Float v) noexcept { x -= v; y -= v; z -= v; }

        inline constexpr bool operator>(Float v) const noexcept { return x > v && y > v && z > v; }
        inline constexpr bool operator<(Float v) const noexcept { return x < v && y < v && z < v; }
        inline constexpr bool operator>=(Float v) const noexcept { return x >= v && y >= v && z >= v; }
        inline constexpr bool operator<=(Float v) const noexcept { return x <= v && y <= v && z <= v; }
        inline constexpr bool operator==(Float v) const noexcept { return x == v && y == v && z == v; }
        inline constexpr bool operator!=(Float v) const noexcept { return x != v || y != v || z != v; }

        [[nodiscard]] inline constexpr Float3<Float> floor() const { return Float3<Float>(std::floor(x), std::floor(y), std::floor(z)); }
        [[nodiscard]] inline constexpr Float3<Float> ceil() const { return Float3<Float>(std::ceil(x), std::ceil(y), std::ceil(z)); }

        [[nodiscard]] inline constexpr Float lengthSq() const noexcept { return x * x + y * y + z * z; }
        [[nodiscard]] inline constexpr Float length() const { return std::sqrt(lengthSq()); }
        [[nodiscard]] inline constexpr Float3<Float> normalize() const { return *this / length(); }

        [[nodiscard]] inline constexpr Float sum() const noexcept { return x + y + z; }
        [[nodiscard]] inline constexpr Float prod() const noexcept { return x * y * z; }

        [[nodiscard]] static inline constexpr size_t size() noexcept { return 3U; }

        [[nodiscard]] inline constexpr Float dot(Float3<Float> v) const { return x * v.x + y * v.y + z * v.z; }
        [[nodiscard]] inline constexpr Float3<Float> cross(Float3<Float> v) const noexcept {
            return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
        }
        //@CLION-formatter:on

        [[nodiscard]] inline constexpr std::array<Float, 3U> toArray() const noexcept {
            return {x, y, z};
        }

        [[nodiscard]] inline std::string toString() const {
            return String::format("({}, {}, {})", x, y, z);
        }
    };

    namespace Math {
        template<typename T>
        inline Float3<T> min(Float3<T> v1, Float3<T> v2) noexcept {
            return {std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z)};
        }

        template<typename T>
        inline Float3<T> min(Float3<T> v1, T v2) noexcept {
            return {std::min(v1.x, v2), std::min(v1.y, v2), std::min(v1.z, v2)};
        }

        template<typename T>
        inline Float3<T> max(Float3<T> v1, Float3<T> v2) noexcept {
            return {std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z)};
        }

        template<typename T>
        inline Float3<T> max(Float3<T> v1, T v2) noexcept {
            return {std::max(v1.x, v2), std::max(v1.y, v2), std::max(v1.z, v2)};
        }

    }
}


namespace Noa {
    template<typename, typename>
    struct Int4;

    /** Static array of 4 floating-points. */
    template<typename Float = int, typename = std::enable_if_t<Noa::Traits::is_float_v<Float>>>
    struct Float4 {
        Float x{0}, y{0}, z{0}, w{0};

        // Constructors.
        constexpr Float4() = default;
        constexpr Float4(Float xi, Float yi, Float zi, Float wi) : x(xi), y(yi), z(zi), w(wi) {}

        constexpr explicit Float4(Float v) : x(v), y(v), z(v), w(v) {}
        constexpr explicit Float4(Float* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]), w(ptr[3]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_float_v<U>>>
        constexpr explicit Float4(U* ptr)
                : x(Float(ptr[0])), y(Float(ptr[1])), z(Float(ptr[2])), w(Float(ptr[3])) {}

        template<typename U>
        constexpr explicit Float4(Float4<U> v)
                : x(Float(v.x)), y(Float(v.y)), z(Float(v.z)), w(Float(v.w)) {}

        template<typename U, typename V>
        constexpr explicit Float4(Int4<U, V> v)
                : x(Float(v.x)), y(Float(v.y)), z(Float(v.z)), w(Float(v.w)) {}

        // Assignment operators.
        inline constexpr auto& operator=(Float v) noexcept {
            x = v;
            y = v;
            z = v;
            w = v;
            return *this;
        }

        inline constexpr auto& operator=(Float* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            z = ptr[2];
            w = ptr[3];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        inline constexpr auto& operator=(U* ptr) noexcept {
            x = Float(ptr[0]);
            y = Float(ptr[1]);
            z = Float(ptr[2]);
            w = Float(ptr[3]);
            return *this;
        }

        template<typename U>
        inline constexpr auto& operator=(Float4<U> v) noexcept {
            x = Float(v.x);
            y = Float(v.y);
            z = Float(v.z);
            w = Float(v.w);
            return *this;
        }

        template<typename U, typename V>
        inline constexpr auto& operator=(Int4<U, V> v) noexcept {
            x = Float(v.x);
            y = Float(v.y);
            z = Float(v.z);
            w = Float(v.w);
            return *this;
        }

        //@CLION-formatter:off
        inline constexpr Float4<Float> operator*(Float4<Float> v) const noexcept { return {x * v.x, y * v.y, z * v.z, w * v.w}; }
        inline constexpr Float4<Float> operator/(Float4<Float> v) const noexcept { return {x / v.x, y / v.y, z / v.z, w / v.w}; }
        inline constexpr Float4<Float> operator+(Float4<Float> v) const noexcept { return {x + v.x, y + v.y, z + v.z, w + v.w}; }
        inline constexpr Float4<Float> operator-(Float4<Float> v) const noexcept { return {x - v.x, y - v.y, z - v.z, w - v.w}; }

        inline constexpr void operator*=(Float4<Float> v) noexcept { x *= v.x; y *= v.y; z *= v.z; w *= v.w; }
        inline constexpr void operator/=(Float4<Float> v) noexcept { x /= v.x; y /= v.y; z /= v.z; w /= v.w; }
        inline constexpr void operator+=(Float4<Float> v) noexcept { x += v.x; y += v.y; z += v.z; w += v.w; }
        inline constexpr void operator-=(Float4<Float> v) noexcept { x -= v.x; y -= v.y; z -= v.z; w -= v.w; }

        inline constexpr bool operator>(Float4<Float> v) const noexcept { return x > v.x && y > v.y && z > v.z && w > v.w; }
        inline constexpr bool operator<(Float4<Float> v) const noexcept { return x < v.x && y < v.y && z < v.z && w < v.w; }
        inline constexpr bool operator>=(Float4<Float> v) const noexcept { return x >= v.x && y >= v.y && z >= v.z && w >= v.w; }
        inline constexpr bool operator<=(Float4<Float> v) const noexcept { return x <= v.x && y <= v.y && z <= v.z && w <= v.w; }
        inline constexpr bool operator==(Float4<Float> v) const noexcept { return x == v.x && y == v.y && z == v.z && w == v.w; }
        inline constexpr bool operator!=(Float4<Float> v) const noexcept { return x != v.x || y != v.y || z != v.z || w != v.w; }

        inline constexpr Float4<Float> operator*(Float v) const noexcept { return {x * v, y * v, z * v, w * v}; }
        inline constexpr Float4<Float> operator/(Float v) const noexcept { return {x / v, y / v, z / v, w / v}; }
        inline constexpr Float4<Float> operator+(Float v) const noexcept { return {x + v, y + v, z + v, w + v}; }
        inline constexpr Float4<Float> operator-(Float v) const noexcept { return {x - v, y - v, z - v, w - v}; }

        inline constexpr void operator*=(Float v) noexcept { x *= v; y *= v; z *= v; w *= v; }
        inline constexpr void operator/=(Float v) noexcept { x /= v; y /= v; z /= v; w /= v; }
        inline constexpr void operator+=(Float v) noexcept { x += v; y += v; z += v; w += v; }
        inline constexpr void operator-=(Float v) noexcept { x -= v; y -= v; z -= v; w -= v; }

        inline constexpr bool operator>(Float v) const noexcept { return x > v && y > v && z > v && w > v; }
        inline constexpr bool operator<(Float v) const noexcept { return x < v && y < v && z < v && w < v; }
        inline constexpr bool operator>=(Float v) const noexcept { return x >= v && y >= v && z >= v && w >= v; }
        inline constexpr bool operator<=(Float v) const noexcept { return x <= v && y <= v && z <= v && w <= v; }
        inline constexpr bool operator==(Float v) const noexcept { return x == v && y == v && z == v && w == v; }
        inline constexpr bool operator!=(Float v) const noexcept { return x != v || y != v || z != v || w != v; }

        [[nodiscard]] inline constexpr Float4<Float> floor() const { return Float4<Float>(std::floor(x), std::floor(y), std::floor(z), std::floor(w)); }
        [[nodiscard]] inline constexpr Float4<Float> ceil() const { return Float4<Float>(std::ceil(x), std::ceil(y), std::ceil(z), std::ceil(w)); }

        [[nodiscard]] inline constexpr Float lengthSq() const noexcept { return x * x + y * y + z * z + w * w; }
        [[nodiscard]] inline constexpr Float length() const { return std::sqrt(lengthSq()); }
        [[nodiscard]] inline constexpr Float4<Float> normalize() const { return *this / length(); }

        [[nodiscard]] static inline constexpr size_t size() noexcept { return 4; }

        [[nodiscard]] inline constexpr Float sum() const noexcept { return x + y + z + w; }
        [[nodiscard]] inline constexpr Float prod() const noexcept { return x * y * z * w; }
        //@CLION-formatter:on

        [[nodiscard]] inline constexpr std::array<Float, 4U> toArray() const noexcept {
            return {x, y, z, w};
        }

        [[nodiscard]] inline std::string toString() const {
            return String::format("({}, {}, {}, {})", x, y, z, w);
        }
    };

    namespace Math {
        template<typename T>
        inline Float4<T> min(Float4<T> i1, Float4<T> i2) noexcept {
            return {std::min(i1.x, i2.x), std::min(i1.y, i2.y),
                    std::min(i1.z, i2.z), std::min(i1.w, i2.w)};
        }

        template<typename T>
        inline Float4<T> min(Float4<T> i1, T i2) noexcept {
            return {std::min(i1.x, i2), std::min(i1.y, i2),
                    std::min(i1.z, i2), std::min(i1.w, i2)};
        }

        template<typename T>
        inline Float4<T> max(Float4<T> i1, Float4<T> i2) noexcept {
            return {std::max(i1.x, i2.x), std::max(i1.y, i2.y),
                    std::max(i1.z, i2.z), std::max(i1.w, i2.w)};
        }

        template<typename T>
        inline Float4<T> max(Float4<T> i1, T i2) noexcept {
            return {std::max(i1.x, i2), std::max(i1.y, i2),
                    std::max(i1.z, i2), std::max(i1.w, i2)};
        }
    }
}


//@CLION-formatter:off
namespace Noa::Traits {
    template<typename T> struct p_is_float2 : std::false_type {};
    template<typename T> struct p_is_float2<Noa::Float2<T>> : std::true_type {};
    template<typename T> using is_float2 = std::bool_constant<p_is_float2<remove_ref_cv_t<Float>>::value>;
    template<typename Float> NOA_API inline constexpr bool is_float2_v = is_float2<Float>::value;

    template<typename T> struct p_is_float3 : std::false_type {};
    template<typename T> struct p_is_float3<Noa::Float3<T>> : std::true_type {};
    template<typename T> using is_float3 = std::bool_constant<p_is_float3<remove_ref_cv_t<Float>>::value>;
    template<typename Float> NOA_API inline constexpr bool is_float3_v = is_float3<Float>::value;

    template<typename T> struct p_is_float4 : std::false_type {};
    template<typename T> struct p_is_float4<Noa::Float4<T>> : std::true_type {};
    template<typename T> using is_float4 = std::bool_constant<p_is_float4<remove_ref_cv_t<Float>>::value>;
    template<typename Float> NOA_API inline constexpr bool is_float4_v = is_float4<Float>::value;

    template<typename T> using is_floatX = std::bool_constant<is_float4_v<T> || is_float3_v<T> || is_float2_v<T>>;
    template<typename T> NOA_API inline constexpr bool is_floatX_v = is_floatX<T>::value;
}
//@CLION-formatter:on


template<typename FloatX>
struct fmt::formatter<FloatX, std::enable_if_t<Noa::Traits::is_floatX_v<FloatX>, char>>
        : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const FloatX& floats, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(floats.toString(), ctx);
    }
};


template<typename FloatX, typename = std::enable_if_t<Noa::Traits::is_floatX_v<FloatX>>>
std::ostream& operator<<(std::ostream& os, const FloatX& floats) {
    os << floats.toString();
    return os;
}


#define ULP 2
#define EPSILON 1e-6f

namespace Noa::Math {
    template<uint32_t ulp = ULP, typename Float>
    [[nodiscard]] inline constexpr bool isEqual(Float2<Float> float2, Float value,
                                                Float epsilon = EPSILON) {
        return Math::isEqual<ulp>(value, float2.x, epsilon) &&
               Math::isEqual<ulp>(value, float2.y, epsilon);
    }

    template<uint32_t ulp = ULP, typename Float>
    [[nodiscard]] constexpr inline bool isEqual(Float2<Float> first, Float2<Float> second,
                                                Float epsilon = EPSILON) {
        return Math::isEqual<ulp>(first.x, second.x, epsilon) &&
               Math::isEqual<ulp>(first.y, second.y, epsilon);
    }

    template<uint32_t ulp = ULP, typename Float>
    [[nodiscard]] inline constexpr bool isEqual(Float3<Float> first, Float value,
                                                Float epsilon = EPSILON) {
        return Math::isEqual<ulp>(value, first.x, epsilon) &&
               Math::isEqual<ulp>(value, first.y, epsilon) &&
               Math::isEqual<ulp>(value, first.z, epsilon);
    }

    template<uint32_t ulp = ULP, typename Float>
    [[nodiscard]] constexpr inline bool isEqual(Float3<Float> first, Float3<Float> second,
                                                Float epsilon = EPSILON) {
        return Math::isEqual<ulp>(first.x, second.x, epsilon) &&
               Math::isEqual<ulp>(first.y, second.y, epsilon) &&
               Math::isEqual<ulp>(first.z, second.z, epsilon);
    }

    template<uint32_t ulp = ULP, typename Float>
    [[nodiscard]] inline constexpr bool isEqual(Float4<Float> first, Float value,
                                                Float epsilon = EPSILON) {
        return Math::isEqual<ulp>(value, first.x, epsilon) &&
               Math::isEqual<ulp>(value, first.y, epsilon) &&
               Math::isEqual<ulp>(value, first.z, epsilon) &&
               Math::isEqual<ulp>(value, first.w, epsilon);
    }

    template<uint32_t ulp = ULP, typename Float>
    [[nodiscard]] constexpr inline bool isEqual(Float4<Float> first, Float4<Float> second,
                                                Float epsilon = EPSILON) {
        return Math::isEqual<ulp>(first.x, second.x, epsilon) &&
               Math::isEqual<ulp>(first.y, second.y, epsilon) &&
               Math::isEqual<ulp>(first.z, second.z, epsilon) &&
               Math::isEqual<ulp>(first.w, second.w, epsilon);
    }
}

#undef ULP
#undef EPSILON
