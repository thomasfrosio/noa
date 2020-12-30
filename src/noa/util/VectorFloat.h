/**
 * @file VectorFloat.h
 * @brief Static arrays of floating-points.
 * @author Thomas - ffyr2w
 * @date 10/12/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/Traits.h"
#include "noa/util/Math.h"


#define SC(x) static_cast<T>(x)
#define ULP 2
#define EPSILON 1e-6f


namespace Noa {
    template<typename, typename>
    struct Int2;

    /** Static array of 2 floating-points. */
    template<typename T = float, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    struct Float2 {
        T x{0}, y{0};

        // Constructors.
        constexpr Float2() = default;
        constexpr Float2(T xi, T yi) : x(xi), y(yi) {}

        constexpr explicit Float2(T v) : x(v), y(v) {}
        constexpr explicit Float2(T* ptr) : x(ptr[0]), y(ptr[1]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_float_v<U>>>
        constexpr explicit Float2(U* ptr) : x(SC(ptr[0])), y(SC(ptr[1])) {}

        template<typename U>
        constexpr explicit Float2(Float2<U> v) : x(SC(v.x)), y(SC(v.y)) {}

        template<typename U, typename V>
        constexpr explicit Float2(Int2<U, V> v) : x(SC(v.x)), y(SC(v.y)) {}

        // Assignment operators.
        constexpr inline auto& operator=(T v) noexcept {
            x = v;
            y = v;
            return *this;
        }
        constexpr inline auto& operator=(T* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(U* ptr) noexcept {
            x = SC(ptr[0]);
            y = SC(ptr[1]);
            return *this;
        }

        template<typename U>
        constexpr inline auto& operator=(Float2<U> v) noexcept {
            x = SC(v.x);
            y = SC(v.y);
            return *this;
        }

        template<typename U, typename V>
        constexpr inline auto& operator=(Int2<U, V> v) noexcept {
            x = SC(v.x);
            y = SC(v.y);
            return *this;
        }

        //@CLION-formatter:off
        constexpr inline Float2<T> operator*(Float2<T> v) const noexcept { return {x * v.x, y * v.y}; }
        constexpr inline Float2<T> operator/(Float2<T> v) const noexcept { return {x / v.x, y / v.y}; }
        constexpr inline Float2<T> operator+(Float2<T> v) const noexcept { return {x + v.x, y + v.y}; }
        constexpr inline Float2<T> operator-(Float2<T> v) const noexcept { return {x - v.x, y - v.y}; }

        constexpr inline void operator*=(Float2<T> v) noexcept { x *= v.x; y *= v.y; }
        constexpr inline void operator/=(Float2<T> v) noexcept { x /= v.x; y /= v.y; }
        constexpr inline void operator+=(Float2<T> v) noexcept { x += v.x; y += v.y; }
        constexpr inline void operator-=(Float2<T> v) noexcept { x -= v.x; y -= v.y; }

        constexpr inline bool operator>(Float2<T> v) const noexcept { return x > v.x && y > v.y; }
        constexpr inline bool operator<(Float2<T> v) const noexcept { return x < v.x && y < v.y; }
        constexpr inline bool operator>=(Float2<T> v) const noexcept { return x >= v.x && y >= v.y; }
        constexpr inline bool operator<=(Float2<T> v) const noexcept { return x <= v.x && y <= v.y; }
        constexpr inline bool operator==(Float2<T> v) const noexcept { return x == v.x && y == v.y; }
        constexpr inline bool operator!=(Float2<T> v) const noexcept { return x != v.x || y != v.y; }

        constexpr inline Float2<T> operator*(T v) const noexcept { return {x * v, y * v}; }
        constexpr inline Float2<T> operator/(T v) const noexcept { return {x / v, y / v}; }
        constexpr inline Float2<T> operator+(T v) const noexcept { return {x + v, y + v}; }
        constexpr inline Float2<T> operator-(T v) const noexcept { return {x - v, y - v}; }

        constexpr inline void operator*=(T v) noexcept { x *= v; y *= v; }
        constexpr inline void operator/=(T v) noexcept { x /= v; y /= v; }
        constexpr inline void operator+=(T v) noexcept { x += v; y += v; }
        constexpr inline void operator-=(T v) noexcept { x -= v; y -= v; }

        constexpr inline bool operator>(T v) const noexcept { return x > v && y > v; }
        constexpr inline bool operator<(T v) const noexcept { return x < v && y < v; }
        constexpr inline bool operator>=(T v) const noexcept { return x >= v && y >= v; }
        constexpr inline bool operator<=(T v) const noexcept { return x <= v && y <= v; }
        constexpr inline bool operator==(T v) const noexcept { return x == v && y == v; }
        constexpr inline bool operator!=(T v) const noexcept { return x != v || y != v; }

        [[nodiscard]] constexpr inline Float2<T> floor() const { return Float2<T>(std::floor(x), std::floor(y)); }
        [[nodiscard]] constexpr inline Float2<T> ceil() const { return Float2<T>(std::ceil(x), std::ceil(y)); }

        [[nodiscard]] constexpr inline T lengthSq() const noexcept { return x * x + y * y; }
        [[nodiscard]] constexpr inline T length() const { return std::sqrt(lengthSq()); }
        [[nodiscard]] constexpr inline Float2<T> normalize() const { return *this / length(); }

        [[nodiscard]] constexpr inline T sum() const noexcept { return x + y; }
        [[nodiscard]] constexpr inline T prod() const noexcept { return x * y; }

        [[nodiscard]] static constexpr inline size_t size() noexcept { return 2U; }

        [[nodiscard]] constexpr inline T dot(Float2<T> v) const noexcept { return x * v.x + y * v.y; }
        //@CLION-formatter:on

        [[nodiscard]] constexpr inline std::array<T, 2U> toArray() const noexcept {
            return {x, y};
        }

        [[nodiscard]] inline std::string toString() const { return fmt::format("({}, {})", x, y); }

        template<uint32_t ulp = ULP>
        [[nodiscard]] constexpr inline bool isEqual(Float2<T> v, T epsilon = EPSILON) const {
            return Math::isEqual<ulp>(x, v.x, epsilon) && Math::isEqual<ulp>(y, v.y, epsilon);
        }

        template<uint32_t ulp = ULP>
        [[nodiscard]] constexpr inline bool isEqual(T v, T epsilon = EPSILON) const {
            return Math::isEqual<ulp>(x, v, epsilon) && Math::isEqual<ulp>(y, v, epsilon);
        }
    };

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

    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Float2<T>& vec) {
        os << vec.toString();
        return os;
    }
}


namespace Noa {
    template<typename, typename>
    struct Int3;

    /** Static array of 3 floating-points. */
    template<typename T = float, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    struct Float3 {
        T x{0}, y{0}, z{0};

        // Constructors.
        constexpr Float3() = default;
        constexpr Float3(T xi, T yi, T zi) : x(xi), y(yi), z(zi) {}

        constexpr explicit Float3(T v) : x(v), y(v), z(v) {}
        constexpr explicit Float3(T* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr explicit Float3(U* ptr) : x(SC(ptr[0])), y(SC(ptr[1])), z(SC(ptr[2])) {}

        template<typename U>
        constexpr explicit Float3(Float3<U> v) : x(SC(v.x)), y(SC(v.y)), z(SC(v.z)) {}

        template<typename U, typename V>
        constexpr explicit Float3(Int3<U, V> v) : x(SC(v.x)), y(SC(v.y)), z(SC(v.z)) {}

        // Assignment operators.
        constexpr inline auto& operator=(T v) noexcept {
            x = v;
            y = v;
            z = v;
            return *this;
        }
        constexpr inline auto& operator=(T* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            z = ptr[2];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(U* ptr) noexcept {
            x = SC(ptr[0]);
            y = SC(ptr[1]);
            z = SC(ptr[2]);
            return *this;
        }

        template<typename U>
        constexpr inline auto& operator=(Float3<U> v) noexcept {
            x = SC(v.x);
            y = SC(v.y);
            z = SC(v.z);
            return *this;
        }

        template<typename U, typename V>
        constexpr inline auto& operator=(Int3<U, V> v) noexcept {
            x = SC(v.x);
            y = SC(v.y);
            z = SC(v.z);
            return *this;
        }

        //@CLION-formatter:off
        constexpr inline Float3<T> operator*(Float3<T> v) const noexcept { return {x * v.x, y * v.y, z * v.z}; }
        constexpr inline Float3<T> operator/(Float3<T> v) const noexcept { return {x / v.x, y / v.y, z / v.z}; }
        constexpr inline Float3<T> operator+(Float3<T> v) const noexcept { return {x + v.x, y + v.y, z + v.z}; }
        constexpr inline Float3<T> operator-(Float3<T> v) const noexcept { return {x - v.x, y - v.y, z - v.z}; }

        constexpr inline void operator*=(Float3<T> v) noexcept { x *= v.x; y *= v.y; z *= v.z; }
        constexpr inline void operator/=(Float3<T> v) noexcept { x /= v.x; y /= v.y; z /= v.z; }
        constexpr inline void operator+=(Float3<T> v) noexcept { x += v.x; y += v.y; z += v.z; }
        constexpr inline void operator-=(Float3<T> v) noexcept { x -= v.x; y -= v.y; z -= v.z; }

        constexpr inline bool operator>(Float3<T> v) const noexcept { return x > v.x && y > v.y && z > v.z; }
        constexpr inline bool operator<(Float3<T> v) const noexcept { return x < v.x && y < v.y && z < v.z; }
        constexpr inline bool operator>=(Float3<T> v) const noexcept { return x >= v.x && y >= v.y && z >= v.z; }
        constexpr inline bool operator<=(Float3<T> v) const noexcept { return x <= v.x && y <= v.y && z <= v.z; }
        constexpr inline bool operator==(Float3<T> v) const noexcept { return x == v.x && y == v.y && z == v.z; }
        constexpr inline bool operator!=(Float3<T> v) const noexcept { return x != v.x || y != v.y || z != v.z; }

        constexpr inline Float3<T> operator*(T v) const noexcept { return {x * v, y * v, z * v}; }
        constexpr inline Float3<T> operator/(T v) const noexcept { return {x / v, y / v, z / v}; }
        constexpr inline Float3<T> operator+(T v) const noexcept { return {x + v, y + v, z + v}; }
        constexpr inline Float3<T> operator-(T v) const noexcept { return {x - v, y - v, z - v}; }

        constexpr inline void operator*=(T v) noexcept { x *= v; y *= v; z *= v; }
        constexpr inline void operator/=(T v) noexcept { x /= v; y /= v; z /= v; }
        constexpr inline void operator+=(T v) noexcept { x += v; y += v; z += v; }
        constexpr inline void operator-=(T v) noexcept { x -= v; y -= v; z -= v; }

        constexpr inline bool operator>(T v) const noexcept { return x > v && y > v && z > v; }
        constexpr inline bool operator<(T v) const noexcept { return x < v && y < v && z < v; }
        constexpr inline bool operator>=(T v) const noexcept { return x >= v && y >= v && z >= v; }
        constexpr inline bool operator<=(T v) const noexcept { return x <= v && y <= v && z <= v; }
        constexpr inline bool operator==(T v) const noexcept { return x == v && y == v && z == v; }
        constexpr inline bool operator!=(T v) const noexcept { return x != v || y != v || z != v; }

        [[nodiscard]] constexpr inline Float3<T> floor() const { return Float3<T>(std::floor(x), std::floor(y), std::floor(z)); }
        [[nodiscard]] constexpr inline Float3<T> ceil() const { return Float3<T>(std::ceil(x), std::ceil(y), std::ceil(z)); }

        [[nodiscard]] constexpr inline T lengthSq() const noexcept { return x * x + y * y + z * z; }
        [[nodiscard]] constexpr inline T length() const { return std::sqrt(lengthSq()); }
        [[nodiscard]] constexpr inline Float3<T> normalize() const { return *this / length(); }

        [[nodiscard]] constexpr inline T sum() const noexcept { return x + y + z; }
        [[nodiscard]] constexpr inline T prod() const noexcept { return x * y * z; }

        [[nodiscard]] static constexpr inline size_t size() noexcept { return 3U; }

        [[nodiscard]] constexpr inline T dot(Float3<T> v) const { return x * v.x + y * v.y + z * v.z; }
        [[nodiscard]] constexpr inline Float3<T> cross(Float3<T> v) const noexcept {
            return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
        }

        [[nodiscard]] constexpr inline std::array<T, 3U> toArray() const noexcept {
            return {x, y, z};
        }

        [[nodiscard]] inline std::string toString() const { return fmt::format("({}, {}, {})", x, y, z); }
        //@CLION-formatter:on

        template<uint32_t ulp = ULP>
        [[nodiscard]] constexpr inline bool isEqual(Float3<T> v, T epsilon = EPSILON) const {
            return Math::isEqual<ulp>(x, v.x, epsilon) &&
                   Math::isEqual<ulp>(y, v.y, epsilon) &&
                   Math::isEqual<ulp>(z, v.z, epsilon);
        }

        template<uint32_t ulp = ULP>
        [[nodiscard]] constexpr inline bool isEqual(T v, T epsilon = EPSILON) const {
            return Math::isEqual<ulp>(x, v, epsilon) &&
                   Math::isEqual<ulp>(y, v, epsilon) &&
                   Math::isEqual<ulp>(z, v, epsilon);
        }
    };

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

    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Float3<T>& vec) {
        os << vec.toString();
        return os;
    }
}


namespace Noa {
    template<typename, typename>
    struct Int4;

    /** Static array of 4 floating-points. */
    template<typename T = int, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    struct Float4 {
        T x{0}, y{0}, z{0}, w{0};

        // Constructors.
        constexpr Float4() = default;
        constexpr Float4(T xi, T yi, T zi, T wi) : x(xi), y(yi), z(zi), w(wi) {}

        constexpr explicit Float4(T v) : x(v), y(v), z(v), w(v) {}
        constexpr explicit Float4(T* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]), w(ptr[3]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_float_v<U>>>
        constexpr explicit Float4(U* ptr)
                : x(SC(ptr[0])), y(SC(ptr[1])), z(SC(ptr[2])), w(SC(ptr[3])) {}

        template<typename U>
        constexpr explicit Float4(Float4<U> v) : x(SC(v.x)), y(SC(v.y)), z(SC(v.z)), w(SC(v.w)) {}

        template<typename U, typename V>
        constexpr explicit Float4(Int4<U, V> v) : x(SC(v.x)), y(SC(v.y)), z(SC(v.z)), w(SC(v.w)) {}

        // Assignment operators.
        constexpr inline auto& operator=(T v) noexcept {
            x = v;
            y = v;
            z = v;
            w = v;
            return *this;
        }
        constexpr inline auto& operator=(T* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            z = ptr[2];
            w = ptr[3];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(U* ptr) noexcept {
            x = SC(ptr[0]);
            y = SC(ptr[1]);
            z = SC(ptr[2]);
            w = SC(ptr[3]);
            return *this;
        }

        template<typename U>
        constexpr inline auto& operator=(Float4<U> v) noexcept {
            x = SC(v.x);
            y = SC(v.y);
            z = SC(v.z);
            w = SC(v.w);
            return *this;
        }

        template<typename U, typename V>
        constexpr inline auto& operator=(Int4<U, V> v) noexcept {
            x = SC(v.x);
            y = SC(v.y);
            z = SC(v.z);
            w = SC(v.w);
            return *this;
        }

        //@CLION-formatter:off
        constexpr inline Float4<T> operator*(Float4<T> v) const noexcept { return {x * v.x, y * v.y, z * v.z, w * v.w}; }
        constexpr inline Float4<T> operator/(Float4<T> v) const noexcept { return {x / v.x, y / v.y, z / v.z, w / v.w}; }
        constexpr inline Float4<T> operator+(Float4<T> v) const noexcept { return {x + v.x, y + v.y, z + v.z, w + v.w}; }
        constexpr inline Float4<T> operator-(Float4<T> v) const noexcept { return {x - v.x, y - v.y, z - v.z, w - v.w}; }

        constexpr inline void operator*=(Float4<T> v) noexcept { x *= v.x; y *= v.y; z *= v.z; w *= v.w; }
        constexpr inline void operator/=(Float4<T> v) noexcept { x /= v.x; y /= v.y; z /= v.z; w /= v.w; }
        constexpr inline void operator+=(Float4<T> v) noexcept { x += v.x; y += v.y; z += v.z; w += v.w; }
        constexpr inline void operator-=(Float4<T> v) noexcept { x -= v.x; y -= v.y; z -= v.z; w -= v.w; }

        constexpr inline bool operator>(Float4<T> v) const noexcept { return x > v.x && y > v.y && z > v.z && w > v.w; }
        constexpr inline bool operator<(Float4<T> v) const noexcept { return x < v.x && y < v.y && z < v.z && w < v.w; }
        constexpr inline bool operator>=(Float4<T> v) const noexcept { return x >= v.x && y >= v.y && z >= v.z && w >= v.w; }
        constexpr inline bool operator<=(Float4<T> v) const noexcept { return x <= v.x && y <= v.y && z <= v.z && w <= v.w; }
        constexpr inline bool operator==(Float4<T> v) const noexcept { return x == v.x && y == v.y && z == v.z && w == v.w; }
        constexpr inline bool operator!=(Float4<T> v) const noexcept { return x != v.x || y != v.y || z != v.z || w != v.w; }

        constexpr inline Float4<T> operator*(T v) const noexcept { return {x * v, y * v, z * v, w * v}; }
        constexpr inline Float4<T> operator/(T v) const noexcept { return {x / v, y / v, z / v, w / v}; }
        constexpr inline Float4<T> operator+(T v) const noexcept { return {x + v, y + v, z + v, w + v}; }
        constexpr inline Float4<T> operator-(T v) const noexcept { return {x - v, y - v, z - v, w - v}; }

        constexpr inline void operator*=(T v) noexcept { x *= v; y *= v; z *= v; w *= v; }
        constexpr inline void operator/=(T v) noexcept { x /= v; y /= v; z /= v; w /= v; }
        constexpr inline void operator+=(T v) noexcept { x += v; y += v; z += v; w += v; }
        constexpr inline void operator-=(T v) noexcept { x -= v; y -= v; z -= v; w -= v; }

        constexpr inline bool operator>(T v) const noexcept { return x > v && y > v && z > v && w > v; }
        constexpr inline bool operator<(T v) const noexcept { return x < v && y < v && z < v && w < v; }
        constexpr inline bool operator>=(T v) const noexcept { return x >= v && y >= v && z >= v && w >= v; }
        constexpr inline bool operator<=(T v) const noexcept { return x <= v && y <= v && z <= v && w <= v; }
        constexpr inline bool operator==(T v) const noexcept { return x == v && y == v && z == v && w == v; }
        constexpr inline bool operator!=(T v) const noexcept { return x != v || y != v || z != v || w != v; }

        [[nodiscard]] constexpr inline Float4<T> floor() const { return Float4<T>(std::floor(x), std::floor(y), std::floor(z), std::floor(w)); }
        [[nodiscard]] constexpr inline Float4<T> ceil() const { return Float4<T>(std::ceil(x), std::ceil(y), std::ceil(z), std::ceil(w)); }

        [[nodiscard]] constexpr inline T lengthSq() const noexcept { return x * x + y * y + z * z + w * w; }
        [[nodiscard]] constexpr inline T length() const { return std::sqrt(lengthSq()); }
        [[nodiscard]] constexpr inline Float4<T> normalize() const { return *this / length(); }

        [[nodiscard]] static constexpr inline size_t size() noexcept { return 4; }

        [[nodiscard]] constexpr inline T sum() const noexcept { return x + y + z + w; }
        [[nodiscard]] constexpr inline T prod() const noexcept { return x * y * z * w; }

        [[nodiscard]] constexpr inline std::array<T, 4U> toArray() const noexcept {
            return {x, y, z, w};
        }

        [[nodiscard]] inline std::string toString() const { return fmt::format("({}, {}, {}, {})", x, y, z, w); }
        //@CLION-formatter:on

        template<uint32_t ulp = ULP>
        [[nodiscard]] constexpr inline bool isEqual(Float4<T> v, T epsilon = EPSILON) {
            return Math::isEqual<ulp>(x, v.x, epsilon) &&
                   Math::isEqual<ulp>(y, v.y, epsilon) &&
                   Math::isEqual<ulp>(z, v.z, epsilon) &&
                   Math::isEqual<ulp>(w, v.w, epsilon);
        }

        template<uint32_t ulp = ULP>
        [[nodiscard]] constexpr inline bool isEqual(T v, T epsilon = EPSILON) const {
            return Math::isEqual<ulp>(x, v, epsilon) &&
                   Math::isEqual<ulp>(y, v, epsilon) &&
                   Math::isEqual<ulp>(z, v, epsilon) &&
                   Math::isEqual<ulp>(w, v, epsilon);
        }
    };

    template<typename T>
    inline Float4<T> min(Float4<T> i1, Float4<T> i2) noexcept {
        return {std::min(i1.x, i2.x), std::min(i1.y, i2.y),
                std::min(i1.z, i2.z), std::min(i1.w, i2.w)};
    }

    template<typename T>
    inline Float4<T> min(Float4<T> i1, T i2) noexcept {
        return {std::min(i1.x, i2), std::min(i1.y, i2), std::min(i1.z, i2), std::min(i1.w, i2)};
    }

    template<typename T>
    inline Float4<T> max(Float4<T> i1, Float4<T> i2) noexcept {
        return {std::max(i1.x, i2.x), std::max(i1.y, i2.y),
                std::max(i1.z, i2.z), std::max(i1.w, i2.w)};
    }

    template<typename T>
    inline Float4<T> max(Float4<T> i1, T i2) noexcept {
        return {std::max(i1.x, i2), std::max(i1.y, i2), std::max(i1.z, i2), std::max(i1.w, i2)};
    }

    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Float4<T>& vec) {
        os << vec.toString();
        return os;
    }
}

#undef SC
#undef ULP
#undef EPSILON

//@CLION-formatter:off
namespace Noa::Traits {
    template<typename T> struct p_is_float2 : std::false_type {};
    template<typename T> struct p_is_float2<Noa::Float2<T>> : std::true_type {};
    template<typename Float> struct NOA_API is_float2 { static constexpr bool value = p_is_float2<remove_ref_cv_t<Float>>::value; };
    template<typename Float> NOA_API inline constexpr bool is_float2_v = is_float2<Float>::value;

    template<typename T> struct p_is_float3 : std::false_type {};
    template<typename T> struct p_is_float3<Noa::Float3<T>> : std::true_type {};
    template<typename Float> struct NOA_API is_float3 { static constexpr bool value = p_is_float3<remove_ref_cv_t<Float>>::value; };
    template<typename Float> NOA_API inline constexpr bool is_float3_v = is_float3<Float>::value;

    template<typename T> struct p_is_float4 : std::false_type {};
    template<typename T> struct p_is_float4<Noa::Float4<T>> : std::true_type {};
    template<typename Float> struct NOA_API is_float4 { static constexpr bool value = p_is_float4<remove_ref_cv_t<Float>>::value; };
    template<typename Float> NOA_API inline constexpr bool is_float4_v = is_float4<Float>::value;

    template<typename T> struct NOA_API is_vector_float { static constexpr bool value = (is_float4_v<T> || is_float3_v<T> || is_float2_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_vector_float_v = is_vector_float<T>::value;
}
//@CLION-formatter:on


template<typename T>
struct fmt::formatter<T, std::enable_if_t<Noa::Traits::is_vector_float_v<T>, char>>
        : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const T& a, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(a.toString(), ctx);
    }
};
