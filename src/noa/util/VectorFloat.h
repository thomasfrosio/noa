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


#define TO_T(x) static_cast<T>(x)
#define ULP 2
#define EPSILON 1e-6f


namespace Noa {
    /** Static array of 2 floating-points. */
    template<typename T = float, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    struct Float2 {
        T x{0}, y{0};

        constexpr Float2() = default;
        constexpr Float2(T xi, T yi) : x(xi), y(yi) {}

        //@CLION-formatter:off
        constexpr explicit Float2(T v) : x(v), y(v) {}
        constexpr explicit Float2(T* ptr) : x(ptr[0]), y(ptr[1]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_float_v<U>>>
        constexpr explicit Float2(U* ptr) : x(TO_T(ptr[0])), y(TO_T(ptr[1])) {}

        template<typename U>
        constexpr explicit Float2(Float2<U> v) : x(TO_T(v.x)), y(TO_T(v.y)) {}

        // Assignment operators.
        constexpr inline auto& operator=(T v) noexcept { x = v; y = v; return *this; }
        constexpr inline auto& operator=(T* ptr) noexcept {x = ptr[0]; y = ptr[1]; return *this; }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(U* ptr) noexcept { x = TO_T(ptr[0]); y = TO_T(ptr[1]); return *this; }

        template<typename U>
        constexpr inline auto& operator=(Float2<U> v) noexcept { x = TO_T(v.x); y = TO_T(v.y); return *this; }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_int_v<U>>>
        constexpr inline T& operator[](U idx) noexcept { return *(this->data() + idx); }

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

        [[nodiscard]] static constexpr inline size_t size() noexcept { return 2; }

        [[nodiscard]] constexpr inline T dot(Float2<T> v) const noexcept { return x * v.x + y * v.y; }
        //@CLION-formatter:on

        [[nodiscard]] constexpr inline T* data() noexcept { return &x; }
        [[nodiscard]] inline std::string toString() const { return fmt::format("({}, {})", x, y); }

        template<uint32_t ulp = ULP>
        constexpr inline bool isEqual(Float2<T> v, T epsilon = EPSILON) const {
            return Math::isEqual<ulp>(x, v.x, epsilon) && Math::isEqual<ulp>(y, v.y, epsilon);
        }

        template<uint32_t ulp = ULP>
        constexpr inline bool isEqual(T v, T epsilon = EPSILON) const {
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
    /** Static array of 3 floating-points. */
    template<typename T = float, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    struct Float3 {
        T x{0}, y{0}, z{0};

        //@CLION-formatter:off
        constexpr Float3() = default;
        constexpr Float3(T xi, T yi, T zi) : x(xi), y(yi), z(zi) {}

        constexpr explicit Float3(T v) : x(v), y(v), z(v) {}
        constexpr explicit Float3(T* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr explicit Float3(U* ptr) : x(TO_T(ptr[0])), y(TO_T(ptr[1])), z(TO_T(ptr[2])) {}

        template<typename U>
        constexpr explicit Float3(Float3<U> v) : x(TO_T(v.x)), y(TO_T(v.y)), z(TO_T(v.z)) {}

        // Assignment operators.
        constexpr inline auto& operator=(T v) noexcept { x = v; y = v; z = v; return *this; }
        constexpr inline auto& operator=(T* ptr) noexcept {x = ptr[0]; y = ptr[1]; z = ptr[2]; return *this; }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(U* ptr) noexcept { x = TO_T(ptr[0]); y = TO_T(ptr[1]); z = TO_T(ptr[2]); return *this; }

        template<typename U>
        constexpr inline auto& operator=(Float3<U> v) noexcept { x = TO_T(v.x); y = TO_T(v.y); z = TO_T(v.z); return *this; }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_int_v<U>>>
        constexpr inline T& operator[](U idx) noexcept { return *(this->data() + idx); }

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

        [[nodiscard]] static constexpr inline size_t size() noexcept { return 3; }

        [[nodiscard]] constexpr inline T dot(Float3<T> v) const { return x * v.x + y * v.y + z * v.z; }
        [[nodiscard]] constexpr inline Float3<T> cross(Float3<T> v) const noexcept {
            return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
        }

        [[nodiscard]] constexpr inline T* data() noexcept { return &x; }

        [[nodiscard]] inline std::string toString() const { return fmt::format("({}, {}, {})", x, y, z); }
        //@CLION-formatter:on

        template<uint32_t ulp = ULP>
        constexpr inline bool isEqual(Float3<T> v, T epsilon = EPSILON) const {
            return Math::isEqual<ulp>(x, v.x, epsilon) &&
                   Math::isEqual<ulp>(y, v.y, epsilon) &&
                   Math::isEqual<ulp>(z, v.z, epsilon);
        }

        template<uint32_t ulp = ULP>
        constexpr inline bool isEqual(T v, T epsilon = EPSILON) const {
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
    /** Static array of 4 floating-points. */
    template<typename T = int, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    struct Float4 {
        T x{0}, y{0}, z{0}, w{0};

        //@CLION-formatter:off
        constexpr Float4() = default;
        constexpr Float4(T xi, T yi, T zi, T wi) : x(xi), y(yi), z(zi), w(wi) {}

        constexpr explicit Float4(T v) : x(v), y(v), z(v), w(v) {}
        constexpr explicit Float4(T* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]), w(ptr[3]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_float_v<U>>>
        constexpr explicit Float4(U* ptr) : x(TO_T(ptr[0])), y(TO_T(ptr[1])), z(TO_T(ptr[2])), w(TO_T(ptr[3])) {}

        template<typename U>
        constexpr explicit Float4(Float4<U> v) : x(TO_T(v.x)), y(TO_T(v.y)), z(TO_T(v.z)), w(TO_T(v.w)) {}

        // Assignment operators.
        constexpr inline auto& operator=(T v) noexcept { x = v; y = v; z = v; w = v; return *this; }
        constexpr inline auto& operator=(T* ptr) noexcept {x = ptr[0]; y = ptr[1]; z = ptr[2]; w = ptr[3]; return *this; }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(U* ptr) noexcept { x = TO_T(ptr[0]); y = TO_T(ptr[1]); z = TO_T(ptr[2]); w = TO_T(ptr[3]); return *this; }

        template<typename U>
        constexpr inline auto& operator=(Float4<U> v) noexcept { x = TO_T(v.x); y = TO_T(v.y); z = TO_T(v.z); w = TO_T(v.w); return *this; }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_int_v<U>>>
        constexpr inline T& operator[](U idx) noexcept { return *(this->data() + idx); }

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

        [[nodiscard]] constexpr inline T* data() noexcept { return &x; }

        [[nodiscard]] inline std::string toString() const { return fmt::format("({}, {}, {}, {})", x, y, z, w); }
        //@CLION-formatter:on

        template<uint32_t ulp = ULP>
        constexpr inline bool isEqual(Float4<T> v, T epsilon = EPSILON) {
            return Math::isEqual<ulp>(x, v.x, epsilon) &&
                   Math::isEqual<ulp>(y, v.y, epsilon) &&
                   Math::isEqual<ulp>(z, v.z, epsilon) &&
                   Math::isEqual<ulp>(w, v.w, epsilon);
        }

        template<uint32_t ulp = ULP>
        constexpr inline bool isEqual(T v, T epsilon = EPSILON) const {
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

#undef TO_T
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
