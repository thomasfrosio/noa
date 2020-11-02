/**
 * @file Arrays.h
 * @brief Small utility static arrays.
 * @author Thomas - ffyr2w
 * @date 25/10/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/Traits.h"
#include "noa/util/Assert.h"


namespace Noa {
    /**
     * Integer static array of size 2.
     * @tparam T    Integer type.
     */
    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    struct Int2 {
        T x{0}, y{0};

        Int2() = default;

        Int2(T xi, T yi) : x(xi), y(yi) {}

        explicit Int2(T v) : x(v), y(v) {}

        template<typename U>
        explicit Int2(Int2<U> v) : x(static_cast<T>(v.x)), y(static_cast<T>(v.y)) {}

        [[nodiscard]] inline T elements() const noexcept { return x * y; }

        [[nodiscard]] inline T elementsFFT() const noexcept { return (x / 2 + 1) * y; }

        inline T* data() noexcept { return &x; }

        [[nodiscard]] inline std::string toString() const { return fmt::format("{}, {}", x, y); }

        template<typename U>
        inline auto& operator=(Int2<U> v) noexcept {
            x = static_cast<T>(v.x);
            y = static_cast<T>(v.y);
            return *this;
        }

        inline auto& operator=(T v) noexcept {
            x = v;
            y = v;
            return *this;
        }

        inline Int2<T> operator*(Int2<T> v) const noexcept { return {x * v.x, y * v.y}; }

        inline Int2<T> operator/(Int2<T> v) const noexcept { return {x / v.x, y / v.y}; }

        inline Int2<T> operator+(Int2<T> v) const noexcept { return {x + v.x, y + v.y}; }

        inline Int2<T> operator-(Int2<T> v) const noexcept { return {x - v.x, y - v.y}; }

        inline Int2<T> operator*(T v) const noexcept { return {x * v, y * v}; }

        inline Int2<T> operator/(T v) const noexcept { return {x / v, y / v}; }

        inline Int2<T> operator+(T v) const noexcept { return {x + v, y + v}; }

        inline Int2<T> operator-(T v) const noexcept { return {x - v, y - v}; }

        inline void operator*=(Int2<T> v) noexcept {
            x *= v.x;
            y *= v.y;
        }

        inline void operator/=(Int2<T> v) noexcept {
            x /= v.x;
            y /= v.y;
        }

        inline void operator+=(Int2<T> v) noexcept {
            x += v.x;
            y += v.y;
        }

        inline void operator-=(Int2<T> v) noexcept {
            x -= v.x;
            y -= v.y;
        }

        inline void operator*=(T v) noexcept {
            x *= v;
            y *= v;
        }

        inline void operator/=(T v) noexcept {
            x /= v;
            y /= v;
        }

        inline void operator+=(T v) noexcept {
            x += v;
            y += v;
        }

        inline void operator-=(T v) noexcept {
            x -= v;
            y -= v;
        }

        inline bool operator>(Int2<T> v) const noexcept { return x > v.x && y > v.y; }

        inline bool operator<(Int2<T> v) const noexcept { return x < v.x && y < v.y; }

        inline bool operator>=(Int2<T> v) const noexcept { return x >= v.x && y >= v.y; }

        inline bool operator<=(Int2<T> v) const noexcept { return x <= v.x && y <= v.y; }

        inline bool operator==(Int2<T> v) const noexcept { return x == v.x && y == v.y; }

        inline bool operator!=(Int2<T> v) const noexcept { return x != v.x && y != v.y; }

        inline bool operator>(T v) const noexcept { return x > v && y > v; }

        inline bool operator<(T v) const noexcept { return x < v && y < v; }

        inline bool operator>=(T v) const noexcept { return x >= v && y >= v; }

        inline bool operator<=(T v) const noexcept { return x <= v && y <= v; }

        inline bool operator==(T v) const noexcept { return x == v && y == v; }

        inline bool operator!=(T v) const noexcept { return x != v && y != v; }
    };

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int2<T> min(Int2<T> i1, Int2<T> i2) noexcept {
        return {std::min(i1.x, i2.x), std::min(i1.y, i2.y)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int2<T> min(Int2<T> i1, T i2) noexcept {
        return {std::min(i1.x, i2), std::min(i1.y, i2)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int2<T> max(Int2<T> i1, Int2<T> i2) noexcept {
        return {std::max(i1.x, i2.x), std::max(i1.y, i2.y)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int2<T> max(Int2<T> i1, T i2) noexcept {
        return {std::max(i1.x, i2), std::max(i1.y, i2)};
    }
}


namespace Noa {
    /**
     * Integer static array of size 3.
     * @tparam T    Integer type.
     */
    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    struct Int3 {
        T x{0}, y{0}, z{0};

        Int3() = default;

        Int3(T xi, T yi, T zi) : x(xi), y(yi), z(zi) {}

        explicit Int3(T v) : x(v), y(v), z(v) {}

        template<typename U>
        explicit Int3(Int3<U> v) : x(static_cast<T>(v.x)),
                                   y(static_cast<T>(v.y)),
                                   z(static_cast<T>(v.z)) {}

        [[nodiscard]] inline Int3<T> slice() const noexcept { return Int3<T>(x, y, 1); }

        [[nodiscard]] inline T elementsSlice() const noexcept { return x * y; }

        [[nodiscard]] inline T elements() const noexcept { return x * y * z; }

        [[nodiscard]] inline T elementsFFT() const noexcept { return (x / 2 + 1) * y * z; }

        inline T* data() noexcept { return &x; }

        [[nodiscard]] inline std::string toString() const {
            return fmt::format("{}, {}, {}", x, y, z);
        }

        template<typename U>
        inline auto& operator=(Int3<U> v) noexcept {
            x = static_cast<T>(v.x);
            y = static_cast<T>(v.y);
            z = static_cast<T>(v.z);
            return *this;
        }

        inline auto& operator=(T v) noexcept {
            x = v;
            y = v;
            z = v;
            return *this;
        }

        inline Int3<T> operator*(Int3<T> v) const noexcept {
            return {x * v.x, y * v.y, z * v.z};
        }

        inline Int3<T> operator/(Int3<T> v) const noexcept {
            return {x / v.x, y / v.y, z / v.z};
        }

        inline Int3<T> operator+(Int3<T> v) const noexcept {
            return {x + v.x, y + v.y, z + v.z};
        }

        inline Int3<T> operator-(Int3<T> v) const noexcept {
            return {x - v.x, y - v.y, z - v.z};
        }

        inline Int3<T> operator*(T v) const noexcept { return {x * v, y * v, z * v}; }

        inline Int3<T> operator/(T v) const noexcept { return {x / v, y / v, z / v}; }

        inline Int3<T> operator+(T v) const noexcept { return {x + v, y + v, z + v}; }

        inline Int3<T> operator-(T v) const noexcept { return {x - v, y - v, z - v}; }

        inline void operator*=(Int3<T> v) noexcept {
            x *= v.x;
            y *= v.y;
            z *= v.z;
        }

        inline void operator/=(Int3<T> v) noexcept {
            x /= v.x;
            y /= v.y;
            z /= v.z;
        }

        inline void operator+=(Int3<T> v) noexcept {
            x += v.x;
            y += v.y;
            z += v.z;
        }

        inline void operator-=(Int3<T> v) noexcept {
            x -= v.x;
            y -= v.y;
            z -= v.z;
        }

        inline void operator*=(T v) noexcept {
            x *= v;
            y *= v;
            z *= v;
        }

        inline void operator/=(T v) noexcept {
            x /= v;
            y /= v;
            z /= v;
        }

        inline void operator+=(T v) noexcept {
            x += v;
            y += v;
            z += v;
        }

        inline void operator-=(T v) noexcept {
            x -= v;
            y -= v;
            z -= v;
        }

        inline bool operator>(Int3<T> v) const noexcept {
            return x > v.x && y > v.y && z > v.z;
        }

        inline bool operator<(Int3<T> v) const noexcept {
            return x < v.x && y < v.y && z < v.z;
        }

        inline bool operator>=(Int3<T> v) const noexcept {
            return x >= v.x && y >= v.y && z >= v.z;
        }

        inline bool operator<=(Int3<T> v) const noexcept {
            return x <= v.x && y <= v.y && z <= v.z;
        }

        inline bool operator==(Int3<T> v) const noexcept {
            return x == v.x && y == v.y && z == v.z;
        }

        inline bool operator!=(Int3<T> v) const noexcept {
            return x != v.x && y != v.y && y != v.y;
        }

        inline bool operator>(T v) const noexcept { return x > v && y > v && z > v; }

        inline bool operator<(T v) const noexcept { return x < v && y < v && z < v; }

        inline bool operator>=(T v) const noexcept { return x >= v && y >= v && z >= v; }

        inline bool operator<=(T v) const noexcept { return x <= v && y <= v && z <= v; }

        inline bool operator==(T v) const noexcept { return x == v && y == v && z == v; }

        inline bool operator!=(T v) const noexcept { return x != v && y != v && y != v; }
    };

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int3<T> min(Int3<T> v1, Int3<T> v2) noexcept {
        return Int3<T>(std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z));
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int3<T> min(Int3<T> v1, T v2) noexcept {
        return Int3<T>(std::min(v1.x, v2), std::min(v1.y, v2), std::min(v1.z, v2));
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int3<T> max(Int3<T> v1, Int3<T> v2) noexcept {
        return Int3<T>(std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z));
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int3<T> max(Int3<T> v1, T v2) noexcept {
        return Int3<T>(std::max(v1.x, v2), std::max(v1.y, v2), std::max(v1.z, v2));
    }
}


namespace Noa {
    /**
     * Integer static array of size 4.
     * @tparam T    Integer type.
     */
    template<typename T = int, typename = std::enable_if_t<Traits::is_int_v<T>>>
    struct Int4 {
        T x{0}, y{0}, z{0}, w{0};

        Int4() = default;

        Int4(T xi, T yi, T zi, T wi) : x(xi), y(yi), z(zi), w(wi) {}

        explicit Int4(T v) : x(v), y(v), z(v), w(v) {}

        template<typename U>
        explicit Int4(Int4<U> v) : x(static_cast<T>(v.x)),
                                   y(static_cast<T>(v.y)),
                                   z(static_cast<T>(v.z)),
                                   w(static_cast<T>(v.w)) {}

        [[nodiscard]] inline Int4<T> slice() const noexcept { return Int4<T>(x, y, 1, 1); }

        [[nodiscard]] inline T elementsSlice() const noexcept { return x * y; }

        [[nodiscard]] inline T elements() const noexcept { return x * y * z * w; }

        [[nodiscard]] inline T elementsFFT() const noexcept { return (x / 2 + 1) * y * z * w; }

        inline T* data() noexcept { return &x; }

        [[nodiscard]] inline std::string toString() const {
            return fmt::format("{{}, {}, {}, {}}", x, y, z, w);
        }

        template<typename U>
        inline auto& operator=(Int4<U> v) noexcept {
            x = static_cast<T>(v.x);
            y = static_cast<T>(v.y);
            z = static_cast<T>(v.z);
            w = static_cast<T>(v.w);
            return *this;
        }

        inline auto& operator=(T v) noexcept {
            x = v;
            y = v;
            z = v;
            w = v;
            return *this;
        }

        inline Int4<T> operator*(Int4<T> v) const noexcept {
            return {x * v.x, y * v.y, z * v.z, w * v.w};
        }

        inline Int4<T> operator/(Int4<T> v) const noexcept {
            return {x / v.x, y / v.y, z / v.z, w / v.w};
        }

        inline Int4<T> operator+(Int4<T> v) const noexcept {
            return {x + v.x, y + v.y, z + v.z, w + v.w};
        }

        inline Int4<T> operator-(Int4<T> v) const noexcept {
            return {x - v.x, y - v.y, z - v.z, w - v.w};
        }

        inline Int4<T> operator*(T v) const noexcept { return {x * v, y * v, z * v, w * v}; }

        inline Int4<T> operator/(T v) const noexcept { return {x / v, y / v, z / v, w / v}; }

        inline Int4<T> operator+(T v) const noexcept { return {x + v, y + v, z + v, w + v}; }

        inline Int4<T> operator-(T v) const noexcept { return {x - v, y - v, z - v, w - v}; }

        inline void operator*=(Int4<T> v) noexcept {
            x *= v.x;
            y *= v.y;
            z *= v.z;
            w *= v.w;
        }

        inline void operator/=(Int4<T> v) noexcept {
            x /= v.x;
            y /= v.y;
            z /= v.z;
            w /= v.w;
        }

        inline void operator+=(Int4<T> v) noexcept {
            x += v.x;
            y += v.y;
            z += v.z;
            w += v.w;
        }

        inline void operator-=(Int4<T> v) noexcept {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            w -= v.w;
        }

        inline void operator*=(T v) noexcept {
            x *= v;
            y *= v;
            z *= v;
            w *= v;
        }

        inline void operator/=(T v) noexcept {
            x /= v;
            y /= v;
            z /= v;
            w /= v;
        }

        inline void operator+=(T v) noexcept {
            x += v;
            y += v;
            z += v;
            w += v;
        }

        inline void operator-=(T v) noexcept {
            x -= v;
            y -= v;
            z -= v;
            w -= v;
        }

        inline bool operator>(Int4<T> v) const noexcept {
            return x > v.x && y > v.y && z > v.z && w > v.w;
        }

        inline bool operator<(Int4<T> v) const noexcept {
            return x < v.x && y < v.y && z < v.z && w < v.w;
        }

        inline bool operator>=(Int4<T> v) const noexcept {
            return x >= v.x && y >= v.y && z >= v.z && w >= v.w;
        }

        inline bool operator<=(Int4<T> v) const noexcept {
            return x <= v.x && y <= v.y && z <= v.z && w <= v.w;
        }

        inline bool operator==(Int4<T> v) const noexcept {
            return x == v.x && y == v.y && z == v.z && w == v.w;
        }

        inline bool operator!=(Int4<T> v) const noexcept {
            return x != v.x && y != v.y && y != v.z && w != v.w;
        }

        inline bool operator>(T v) const noexcept { return x > v && y > v && z > v && w > v; }

        inline bool operator<(T v) const noexcept { return x < v && y < v && z < v && w < v; }

        inline bool operator>=(T v) const noexcept { return x >= v && y >= v && z >= v && w >= v; }

        inline bool operator<=(T v) const noexcept { return x <= v && y <= v && z <= v && w <= v; }

        inline bool operator==(T v) const noexcept { return x == v && y == v && z == v && w == v; }

        inline bool operator!=(T v) const noexcept { return x != v && y != v && y != v && w != v; }
    };

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int4<T> min(Int4<T> v1, Int4<T> v2) noexcept {
        return Int4<T>(std::min(v1.x, v2.x), std::min(v1.y, v2.y),
                       std::min(v1.z, v2.z), std::min(v1.w, v2.w));
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int4<T> min(Int4<T> v1, T v2) noexcept {
        return Int4<T>(std::min(v1.x, v2), std::min(v1.y, v2),
                       std::min(v1.z, v2), std::min(v1.w, v2));
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int4<T> max(Int4<T> v1, Int4<T> v2) noexcept {
        return Int4<T>(std::max(v1.x, v2.x), std::max(v1.y, v2.y),
                       std::max(v1.z, v2.z), std::min(v1.w, v2.w));
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int4<T> max(Int4<T> v1, T v2) noexcept {
        return Int4<T>(std::max(v1.x, v2), std::max(v1.y, v2),
                       std::max(v1.z, v2), std::min(v1.w, v2));
    }
}


namespace Noa {
    /**
     * Float static array of size 2.
     * @tparam T    Floating point type.
     */
    template<typename T = int, typename = std::enable_if_t<Traits::is_float_v<T>>>
    struct Float2 {
        T x{0}, y{0};

        Float2() = default;

        Float2(T xi, T yi) : x(xi), y(yi) {}

        explicit Float2(T v) : x(v), y(v) {}

        template<typename U>
        explicit Float2(Float2<U> v) : x(static_cast<T>(v.x)), y(static_cast<T>(v.y)) {}

        [[nodiscard]] inline Float2<T> floor() const {
            return Float2<T>(std::floor(x), std::floor(y));
        }

        [[nodiscard]] inline Float2<T> ceil() const {
            return Float2<T>(std::ceil(x), std::ceil(y));
        }

        [[nodiscard]] inline T length() const { return sqrt(x * x + y * y); }

        [[nodiscard]] inline T lengthSq() const { return x * x + y * y; }

        [[nodiscard]] inline Float2<T> normalize() const { return *this / length(); }

        [[nodiscard]] inline T dot(Float2<T> v) const { return x * v.x + y * v.y; }

        [[nodiscard]] inline T cross(Float2<T> v) const { return x * v.y - y * v.x; }

        inline T* data() noexcept { return &x; }

        [[nodiscard]] inline std::string toString() const {
            return fmt::format("{{}, {}}", x, y);
        }

        template<typename U>
        inline auto& operator=(Float2<U> v) noexcept {
            x = static_cast<T>(v.x);
            y = static_cast<T>(v.y);
            return *this;
        }

        inline auto& operator=(T v) noexcept {
            x = v;
            y = v;
            return *this;
        }

        inline Float2<T> operator*(Float2<T> v) const noexcept { return {x * v.x, y * v.y}; }

        inline Float2<T> operator/(Float2<T> v) const noexcept { return {x / v.x, y / v.y}; }

        inline Float2<T> operator+(Float2<T> v) const noexcept { return {x + v.x, y + v.y}; }

        inline Float2<T> operator-(Float2<T> v) const noexcept { return {x - v.x, y - v.y}; }

        inline Float2<T> operator*(T v) const noexcept { return {x * v, y * v}; }

        inline Float2<T> operator/(T v) const noexcept { return {x / v, y / v}; }

        inline Float2<T> operator+(T v) const noexcept { return {x + v, y + v}; }

        inline Float2<T> operator-(T v) const noexcept { return {x - v, y - v}; }

        inline void operator*=(Float2<T> v) noexcept {
            x *= v.x;
            y *= v.y;
        }

        inline void operator/=(Float2<T> v) noexcept {
            x /= v.x;
            y /= v.y;
        }

        inline void operator+=(Float2<T> v) noexcept {
            x += v.x;
            y += v.y;
        }

        inline void operator-=(Float2<T> v) noexcept {
            x -= v.x;
            y -= v.y;
        }

        inline void operator*=(T v) noexcept {
            x *= v;
            y *= v;
        }

        inline void operator/=(T v) noexcept {
            x /= v;
            y /= v;
        }

        inline void operator+=(T v) noexcept {
            x += v;
            y += v;
        }

        inline void operator-=(T v) noexcept {
            x -= v;
            y -= v;
        }

        inline bool operator==(Float2<T> v) const noexcept { return x == v.x && y == v.y; }

        inline bool operator!=(Float2<T> v) const noexcept { return x != v.x && y != v.y; }

        inline bool operator>=(Float2<T> v) const noexcept { return x >= v.x && y >= v.y; }

        inline bool operator<=(Float2<T> v) const noexcept { return x <= v.x && y <= v.y; }

        inline bool operator>(Float2<T> v) const noexcept { return x > v.x && y > v.y; }

        inline bool operator<(Float2<T> v) const noexcept { return x < v.x && y < v.y; }

        inline bool operator==(T v) const noexcept { return x == v && y == v; }

        inline bool operator!=(T v) const noexcept { return x != v && y != v; }

        inline bool operator>(T v) const noexcept { return x > v && y > v; }

        inline bool operator<(T v) const noexcept { return x < v && y < v; }

        inline bool operator>=(T v) const noexcept { return x >= v && y >= v; }

        inline bool operator<=(T v) const noexcept { return x <= v && y <= v; }
    };

    template<typename T, typename = std::enable_if_t<Traits::is_float_v<T>>>
    inline Float2<T> min(Float2<T> v1, Float2<T> v2) noexcept {
        return Float2<T>(std::min(v1.x, v2.x), std::min(v1.y, v2.y));
    }

    template<typename T, typename = std::enable_if_t<Traits::is_float_v<T>>>
    inline Float2<T> min(Float2<T> v1, T v2) noexcept {
        return Float2<T>(std::min(v1.x, v2), std::min(v1.y, v2));
    }

    template<typename T, typename = std::enable_if_t<Traits::is_float_v<T>>>
    inline Float2<T> max(Float2<T> v1, Float2<T> v2) noexcept {
        return Float2<T>(std::max(v1.x, v2.x), std::max(v1.y, v2.y));
    }

    template<typename T, typename = std::enable_if_t<Traits::is_float_v<T>>>
    inline Float2<T> max(Float2<T> v1, T v2) noexcept {
        return Float2<T>(std::max(v1.x, v2), std::max(v1.y, v2));
    }
}


namespace Noa {
    /**
     * Float static array of size 3.
     * @tparam T    Floating point type.
     */
    template<typename T = int, typename = std::enable_if_t<Traits::is_float_v<T>>>
    struct Float3 {
        T x{0}, y{0}, z{0};

        Float3() = default;

        Float3(T xi, T yi, T zi) : x(xi), y(yi), z(zi) {}

        template<typename U>
        explicit Float3(Float3<U> v) : x(static_cast<T>(v.x)),
                                       y(static_cast<T>(v.y)),
                                       z(static_cast<T>(v.z)) {}

        explicit Float3(T v) : x(v), y(v), z(v) {}

        [[nodiscard]] inline Float3<T> floor() const {
            return {std::floor(x), std::floor(y), std::floor(z)};
        }

        [[nodiscard]] inline Float3<T> ceil() const {
            return {std::ceil(x), std::ceil(y), std::ceil(z)};
        }

        [[nodiscard]] inline T length() const { return sqrt(x * x + y * y + z * z); }

        [[nodiscard]] inline T lengthSq() const { return x * x + y * y + z * z; }

        [[nodiscard]] inline Float3<T> normalize() const { return *this / length(); }

        [[nodiscard]] inline T dot(Float3<T> v) const { return x * v.x + y * v.y + z * v.z; }

        [[nodiscard]] inline T cross(Float3<T> v) const { return x * v.y - y * v.x + z * v.z; }

        inline T* data() noexcept { return &x; }

        [[nodiscard]] inline std::string toString() const {
            return fmt::format("{{}, {}, {}}", x, y, z);
        }

        template<typename U>
        inline auto& operator=(Float3<U> v) noexcept {
            x = static_cast<T>(v.x);
            y = static_cast<T>(v.y);
            z = static_cast<T>(v.z);
            return *this;
        }

        inline auto& operator=(T v) noexcept {
            x = v;
            y = v;
            z = v;
            return *this;
        }

        inline Float3<T> operator*(Float3<T> v) const noexcept {
            return {x * v.x, y * v.y, z * v.z};
        }

        inline Float3<T> operator/(Float3<T> v) const noexcept {
            return {x / v.x, y / v.y, z / v.z};
        }

        inline Float3<T> operator+(Float3<T> v) const noexcept {
            return {x + v.x, y + v.y, z + v.z};
        }

        inline Float3<T> operator-(Float3<T> v) const noexcept {
            return {x - v.x, y - v.y, z - v.z};
        }

        inline Float3<T> operator*(T v) const noexcept { return {x * v, y * v, z * v}; }

        inline Float3<T> operator/(T v) const noexcept { return {x / v, y / v, z / v}; }

        inline Float3<T> operator+(T v) const noexcept { return {x + v, y + v, z + v}; }

        inline Float3<T> operator-(T v) const noexcept { return {x - v, y - v, z - v}; }

        inline void operator*=(Float3<T> v) noexcept {
            x *= v.x;
            y *= v.y;
            z *= v.z;
        }

        inline void operator/=(Float3<T> v) noexcept {
            x /= v.x;
            y /= v.y;
            z /= v.z;
        }

        inline void operator+=(Float3<T> v) noexcept {
            x += v.x;
            y += v.y;
            z += v.z;
        }

        inline void operator-=(Float3<T> v) noexcept {
            x -= v.x;
            y -= v.y;
            z -= v.z;
        }

        inline void operator*=(T v) noexcept {
            x *= v;
            y *= v;
            z *= v;
        }

        inline void operator/=(T v) noexcept {
            x /= v;
            y /= v;
            z /= v;
        }

        inline void operator+=(T v) noexcept {
            x += v;
            y += v;
            z += v;
        }

        inline void operator-=(T v) noexcept {
            x -= v;
            y -= v;
            z -= v;
        }

        inline bool operator==(Float3<T> v) const noexcept {
            return x == v.x && y == v.y && z == v.z;
        }

        inline bool operator!=(Float3<T> v) const noexcept {
            return x != v.x && y != v.y && z != v.z;
        }

        inline bool operator>=(Float3<T> v) const noexcept {
            return x >= v.x && y >= v.y && z >= v.z;
        }

        inline bool operator<=(Float3<T> v) const noexcept {
            return x <= v.x && y <= v.y && z <= v.z;
        }

        inline bool operator>(Float3<T> v) const noexcept { return x > v.x && y > v.y && z > v.z; }

        inline bool operator<(Float3<T> v) const noexcept { return x < v.x && y < v.y && z < v.z; }

        inline bool operator==(T v) const noexcept { return x == v && y == v && z == v; }

        inline bool operator!=(T v) const noexcept { return x != v && y != v && z != v; }

        inline bool operator>(T v) const noexcept { return x > v && y > v && z > v; }

        inline bool operator<(T v) const noexcept { return x < v && y < v && z < v; }

        inline bool operator>=(T v) const noexcept { return x >= v && y >= v && z >= v; }

        inline bool operator<=(T v) const noexcept { return x <= v && y <= v && z <= v; }
    };

    template<typename T, typename = std::enable_if_t<Traits::is_float_v<T>>>
    inline Float3<T> min(Float3<T> v1, Float3<T> v2) noexcept {
        return {std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_float_v<T>>>
    inline Float3<T> min(Float3<T> v1, T v2) noexcept {
        return {std::min(v1.x, v2), std::min(v1.y, v2), std::min(v1.z, v2.z)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_float_v<T>>>
    inline Float3<T> max(Float3<T> v1, Float3<T> v2) noexcept {
        return {std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_float_v<T>>>
    inline Float3<T> max(Float3<T> v1, T v2) noexcept {
        return {std::max(v1.x, v2), std::max(v1.y, v2), std::max(v1.z, v2.z)};
    }
}


namespace Noa {
    /**
     * Float static array of size 4.
     * @tparam T    Floating point type.
     */
    template<typename T = int, typename = std::enable_if_t<Traits::is_float_v<T>>>
    struct Float4 {
        T x{0}, y{0}, z{0}, w{0};

        Float4() = default;

        Float4(T xi, T yi, T zi, T wi) : x(xi), y(yi), z(zi), w(wi) {}

        explicit Float4(T v) : x(v), y(v), z(v), w(v) {}

        template<typename U>
        explicit Float4(Float4<U> v) : x(static_cast<T>(v.x)),
                                       y(static_cast<T>(v.y)),
                                       z(static_cast<T>(v.z)),
                                       w(static_cast<T>(v.w)) {}

        [[nodiscard]] inline Float4<T> floor() const {
            return {std::floor(x), std::floor(y), std::floor(z), std::floor(w)};
        }

        [[nodiscard]] inline Float4<T> ceil() const {
            return {std::ceil(x), std::ceil(y), std::ceil(z), std::ceil(w)};
        }

        [[nodiscard]] inline T length() const { return sqrt(x * x + y * y + z * z + w * w); }

        [[nodiscard]] inline T lengthSq() const { return x * x + y * y + z * z + w * w; }

        [[nodiscard]] inline Float4<T> normalize() const { return *this / length(); }

        [[nodiscard]] inline T dot(Float4<T> v) const {
            return x * v.x + y * v.y + z * v.z + w * v.w;
        }

        [[nodiscard]] inline T cross(Float4<T> v) const {
            return x * v.y - y * v.x + z * v.z + w * v.w;
        }

        inline T* data() noexcept { return &x; }

        [[nodiscard]] inline std::string toString() const {
            return fmt::format("{{}, {}, {}, {}}", x, y, z, w);
        }

        template<typename U>
        inline auto& operator=(Float4<U> v) noexcept {
            x = static_cast<T>(v.x);
            y = static_cast<T>(v.y);
            z = static_cast<T>(v.z);
            w = static_cast<T>(v.w);
            return *this;
        }

        inline auto& operator=(T v) noexcept {
            x = v;
            y = v;
            z = v;
            w = v;
            return *this;
        }

        inline Float4<T> operator*(Float4<T> v) const noexcept {
            return {x * v.x, y * v.y, z * v.z, w * v.w};
        }

        inline Float4<T> operator/(Float4<T> v) const noexcept {
            return {x / v.x, y / v.y, z / v.z, w / v.w};
        }

        inline Float4<T> operator+(Float4<T> v) const noexcept {
            return {x + v.x, y + v.y, z + v.z, w + v.w};
        }

        inline Float4<T> operator-(Float4<T> v) const noexcept {
            return {x - v.x, y - v.y, z - v.z, w - v.w};
        }

        inline Float4<T> operator*(T v) const noexcept { return {x * v, y * v, z * v, w * v}; }

        inline Float4<T> operator/(T v) const noexcept { return {x / v, y / v, z / v, w / v}; }

        inline Float4<T> operator+(T v) const noexcept { return {x + v, y + v, z + v, w + v}; }

        inline Float4<T> operator-(T v) const noexcept { return {x - v, y - v, z - v, w - v}; }

        inline void operator*=(Float4<T> v) noexcept {
            x *= v.x;
            y *= v.y;
            z *= v.z;
            w *= v.w;
        }

        inline void operator/=(Float4<T> v) noexcept {
            x /= v.x;
            y /= v.y;
            z /= v.z;
            w /= v.w;
        }

        inline void operator+=(Float4<T> v) noexcept {
            x += v.x;
            y += v.y;
            z += v.z;
            w += v.w;
        }

        inline void operator-=(Float4<T> v) noexcept {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            w -= v.w;
        }

        inline void operator*=(T v) noexcept {
            x *= v;
            y *= v;
            z *= v;
            w *= v;
        }

        inline void operator/=(T v) noexcept {
            x /= v;
            y /= v;
            z /= v;
            w /= v;
        }

        inline void operator+=(T v) noexcept {
            x += v;
            y += v;
            z += v;
            w += v;
        }

        inline void operator-=(T v) noexcept {
            x -= v;
            y -= v;
            z -= v;
            w -= v;
        }

        inline bool operator==(Float4<T> v) const noexcept {
            return x == v.x && y == v.y && z == v.z && w == v.w;
        }

        inline bool operator!=(Float4<T> v) const noexcept {
            return x != v.x && y != v.y && z != v.z && w != v.w;
        }

        inline bool operator>=(Float4<T> v) const noexcept {
            return x >= v.x && y >= v.y && z >= v.z && w >= v.w;
        }

        inline bool operator<=(Float4<T> v) const noexcept {
            return x <= v.x && y <= v.y && z <= v.z && w <= v.w;
        }

        inline bool operator>(Float4<T> v) const noexcept {
            return x > v.x && y > v.y && z > v.z && w > v.w;
        }

        inline bool operator<(Float4<T> v) const noexcept {
            return x < v.x && y < v.y && z < v.z && w < v.w;
        }

        inline bool operator==(T v) const noexcept {
            return x == v && y == v && z == v && w == v;
        }

        inline bool operator!=(T v) const noexcept {
            return x != v && y != v && z != v && w != v;
        }

        inline bool operator>(T v) const noexcept {
            return x > v && y > v && z > v && w > v;
        }

        inline bool operator<(T v) const noexcept {
            return x < v && y < v && z < v && w < v;
        }

        inline bool operator>=(T v) const noexcept {
            return x >= v && y >= v && z >= v && w >= v;
        }

        inline bool operator<=(T v) const noexcept {
            return x <= v && y <= v && z <= v && w <= v;
        }
    };

    template<typename T, typename = std::enable_if_t<Traits::is_float_v<T>>>
    inline Float4<T> min(Float4<T> v1, Float4<T> v2) noexcept {
        return {std::min(v1.x, v2.x), std::min(v1.y, v2.y),
                std::min(v1.z, v2.z), std::min(v1.w, v2.w)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_float_v<T>>>
    inline Float4<T> min(Float4<T> v1, T v2) noexcept {
        return {std::min(v1.x, v2), std::min(v1.y, v2),
                std::min(v1.z, v2), std::min(v1.w, v2)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_float_v<T>>>
    inline Float4<T> max(Float4<T> v1, Float4<T> v2) noexcept {
        return {std::max(v1.x, v2.x), std::max(v1.y, v2.y),
                std::max(v1.z, v2.z), std::max(v1.w, v2.w)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_float_v<T>>>
    inline Float4<T> max(Float4<T> v1, T v2) noexcept {
        return {std::max(v1.x, v2), std::max(v1.y, v2),
                std::max(v1.z, v2), std::max(v1.w, v2)};
    }
}


namespace Noa {
    template<typename Integer = int, typename FloatingPoint>
    inline Int2<Integer> toInt2(Float2<FloatingPoint> v) {
        return {static_cast<Integer>(v.x), static_cast<Integer>(v.y)};
    }

    template<typename Integer = int, typename FloatingPoint>
    inline Int3<Integer> toInt3(Float3<FloatingPoint> v) {
        return {static_cast<Integer>(v.x), static_cast<Integer>(v.y), static_cast<Integer>(v.z)};
    }

    template<typename Integer = int, typename FloatingPoint>
    inline Int4<Integer> toInt4(Float4<FloatingPoint> v) {
        return {static_cast<Integer>(v.x), static_cast<Integer>(v.y),
                static_cast<Integer>(v.z), static_cast<Integer>(v.w)};
    }

    template<typename FloatingPoint = float, typename Integer>
    inline Float2<FloatingPoint> toFloat2(Int2<Integer> v) {
        return {static_cast<FloatingPoint>(v.x), static_cast<FloatingPoint>(v.y)};
    }

    template<typename FloatingPoint = float, typename Integer>
    inline Float3<FloatingPoint> toFloat3(Int3<Integer> v) {
        return {static_cast<FloatingPoint>(v.x), static_cast<FloatingPoint>(v.y),
                static_cast<FloatingPoint>(v.z)};
    }

    template<typename FloatingPoint = float, typename Integer>
    inline Float4<FloatingPoint> toFloat4(Int4<Integer> v) {
        return {static_cast<FloatingPoint>(v.x), static_cast<FloatingPoint>(v.y),
                static_cast<FloatingPoint>(v.z), static_cast<FloatingPoint>(v.w)};
    }
}
