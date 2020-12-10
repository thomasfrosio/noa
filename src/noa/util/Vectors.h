/**
 * @file Vectors.h
 * @brief Small utility static arrays.
 * @author Thomas - ffyr2w
 * @date 25/10/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/Traits.h"


/*
 * Although the IntX vectors support "short" integers ((u)int8_t abd (u)int16_t), in most cases
 * there is an integral promotion performed before arithmetic operations. It then triggers
 * a narrowing conversion when the promoted integer needs to be casted back to these "short" integers.
 * See: https://stackoverflow.com/questions/24371868/why-must-a-short-be-converted-to-an-int-before-arithmetic-operations-in-c-and-c
 *
 * As such, I should simply not use these "short" integers (except when saving space is critical).
 * Warning: Only int32_t, int64_t, uint32_t and uint64_t are tested!
 */


namespace Noa {
    /** Integer static array of size 2. */
    template<typename T = int32_t, typename = std::enable_if_t<Traits::is_int_v<T>>>
    struct Int2 {
        T x{0}, y{0};

        constexpr Int2() = default;
        constexpr Int2(T xi, T yi) : x(xi), y(yi) {}

        constexpr explicit Int2(T v) : x(v), y(v) {}
        constexpr explicit Int2(T* ptr) : x(ptr[0]), y(ptr[1]) {}

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr explicit Int2(U* ptr) : x(static_cast<T>(ptr[0])), y(static_cast<T>(ptr[1])) {}

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr explicit Int2(Int2<U> vec) : x(static_cast<T>(vec.x)), y(static_cast<T>(vec.y)) {}

        constexpr inline auto& operator=(T v) noexcept { x = v; y = v; return *this; }
        constexpr inline auto& operator=(T* ptr) noexcept {x = ptr[0]; y = ptr[1]; return *this; }

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(U* ptr) noexcept {
            x = static_cast<T>(ptr[0]);
            y = static_cast<T>(ptr[1]);
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(Int2<U> vec) noexcept {
            x = static_cast<T>(vec.x);
            y = static_cast<T>(vec.y);
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Traits::is_int_v<U>>>
        constexpr inline T& operator[](U idx) noexcept { return *(this->data() + idx); }

        //@CLION-formatter:off
        constexpr inline Int2<T> operator*(Int2<T> v) const noexcept { return {x * v.x, y * v.y}; }
        constexpr inline Int2<T> operator/(Int2<T> v) const noexcept { return {x / v.x, y / v.y}; }
        constexpr inline Int2<T> operator+(Int2<T> v) const noexcept { return {x + v.x, y + v.y}; }
        constexpr inline Int2<T> operator-(Int2<T> v) const noexcept { return {x - v.x, y - v.y}; }

        constexpr inline void operator*=(Int2<T> v) noexcept { x *= v.x; y *= v.y; }
        constexpr inline void operator/=(Int2<T> v) noexcept { x /= v.x; y /= v.y; }
        constexpr inline void operator+=(Int2<T> v) noexcept { x += v.x; y += v.y; }
        constexpr inline void operator-=(Int2<T> v) noexcept { x -= v.x; y -= v.y; }

        constexpr inline bool operator>(Int2<T> v) const noexcept { return x > v.x && y > v.y; }
        constexpr inline bool operator<(Int2<T> v) const noexcept { return x < v.x && y < v.y; }
        constexpr inline bool operator>=(Int2<T> v) const noexcept { return x >= v.x && y >= v.y; }
        constexpr inline bool operator<=(Int2<T> v) const noexcept { return x <= v.x && y <= v.y; }
        constexpr inline bool operator==(Int2<T> v) const noexcept { return x == v.x && y == v.y; }
        constexpr inline bool operator!=(Int2<T> v) const noexcept { return x != v.x || y != v.y; }

        constexpr inline Int2<T> operator*(T v) const noexcept { return {x * v, y * v}; }
        constexpr inline Int2<T> operator/(T v) const noexcept { return {x / v, y / v}; }
        constexpr inline Int2<T> operator+(T v) const noexcept { return {x + v, y + v}; }
        constexpr inline Int2<T> operator-(T v) const noexcept { return {x - v, y - v}; }

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
        //@CLION-formatter:on

        constexpr inline T* data() noexcept { return &x; }

        [[nodiscard]] constexpr inline size_t size() const noexcept { return 2; }
        [[nodiscard]] constexpr inline T sum() const noexcept { return x + y; }
        [[nodiscard]] constexpr inline T prod() const noexcept { return x * y; }
        [[nodiscard]] constexpr inline T prodFFT() const noexcept { return (x / 2 + 1) * y; }

        [[nodiscard]] inline std::string toString() const { return fmt::format("({}, {})", x, y); }
    };

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    constexpr inline Int2<T> min(Int2<T> i1, Int2<T> i2) noexcept {
        return {std::min(i1.x, i2.x), std::min(i1.y, i2.y)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    constexpr inline Int2<T> min(Int2<T> i1, T i2) noexcept {
        return {std::min(i1.x, i2), std::min(i1.y, i2)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    constexpr inline Int2<T> max(Int2<T> i1, Int2<T> i2) noexcept {
        return {std::max(i1.x, i2.x), std::max(i1.y, i2.y)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    constexpr inline Int2<T> max(Int2<T> i1, T i2) noexcept {
        return {std::max(i1.x, i2), std::max(i1.y, i2)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    std::ostream& operator<<(std::ostream& os, const Int2<T>& vec) {
        os << "(" << vec.x << ", " << vec.y << ")";
        return os;
    }
}


namespace Noa {
    /** Integer static array of size 3. */
    template<typename T = int32_t, typename = std::enable_if_t<Traits::is_int_v<T>>>
    struct Int3 {
        T x{0}, y{0}, z{0};

        constexpr Int3() = default;
        constexpr Int3(T xi, T yi, T zi) : x(xi), y(yi), z(zi) {}

        constexpr explicit Int3(T v) : x(v), y(v), z(v) {}
        constexpr explicit Int3(T* data) : x(data[0]), y(data[1]), z(data[2]) {}

        //@CLION-formatter:off
        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr explicit Int3(U v) : x(static_cast<T>(v)), y(static_cast<T>(v)), z(static_cast<T>(v)) {}

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr explicit Int3(U* data) : x(static_cast<T>(data[0])), y(static_cast<T>(data[1])), z(static_cast<T>(data[2])) {}

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr explicit Int3(Int3<U> vec) : x(static_cast<T>(vec.x)), y(static_cast<T>(vec.y)), z(static_cast<T>(vec.z)) {}

        constexpr inline auto& operator=(T v) noexcept { x = v; y = v; z = v; return *this; }

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(U* v) noexcept {
            x = static_cast<T>(v[0]);
            y = static_cast<T>(v[1]);
            z = static_cast<T>(v[2]);
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(Int3<U> vec) noexcept {
            x = static_cast<T>(vec.x);
            y = static_cast<T>(vec.y);
            z = static_cast<T>(vec.z);
            return *this;
        }

        constexpr inline Int3<T> operator*(Int3<T> v) const noexcept { return {x * v.x, y * v.y, z * v.z}; }
        constexpr inline Int3<T> operator/(Int3<T> v) const noexcept { return {x / v.x, y / v.y, z / v.z}; }
        constexpr inline Int3<T> operator+(Int3<T> v) const noexcept { return {x + v.x, y + v.y, z + v.z}; }
        constexpr inline Int3<T> operator-(Int3<T> v) const noexcept { return {x - v.x, y - v.y, z - v.z}; }

        constexpr inline void operator*=(Int3<T> v) noexcept { x *= v.x; y *= v.y, z *= v.z; }
        constexpr inline void operator/=(Int3<T> v) noexcept { x /= v.x; y /= v.y, z /= v.z; }
        constexpr inline void operator+=(Int3<T> v) noexcept { x += v.x; y += v.y, z += v.z; }
        constexpr inline void operator-=(Int3<T> v) noexcept { x -= v.x; y -= v.y, z -= v.z; }

        constexpr inline bool operator>(Int3<T> v) const noexcept { return x > v.x && y > v.y && z > v.z; }
        constexpr inline bool operator<(Int3<T> v) const noexcept { return x < v.x && y < v.y && z < v.z; }
        constexpr inline bool operator>=(Int3<T> v) const noexcept { return x >= v.x && y >= v.y && z >= v.z; }
        constexpr inline bool operator<=(Int3<T> v) const noexcept { return x <= v.x && y <= v.y && z <= v.z; }
        constexpr inline bool operator==(Int3<T> v) const noexcept { return x == v.x && y == v.y && z == v.z; }
        constexpr inline bool operator!=(Int3<T> v) const noexcept { return x != v.x || y != v.y || z != v.z; }

        constexpr inline Int3<T> operator*(T v) const noexcept { return {x * v, y * v, z * v}; }
        constexpr inline Int3<T> operator/(T v) const noexcept { return {x / v, y / v, z / v}; }
        constexpr inline Int3<T> operator+(T v) const noexcept { return {x + v, y + v, z + v}; }
        constexpr inline Int3<T> operator-(T v) const noexcept { return {x - v, y - v, z - v}; }

        constexpr inline void operator*=(T v) noexcept { x *= v; y *= v, z *= v; }
        constexpr inline void operator/=(T v) noexcept { x /= v; y /= v, z /= v; }
        constexpr inline void operator+=(T v) noexcept { x += v; y += v, z += v; }
        constexpr inline void operator-=(T v) noexcept { x -= v; y -= v, z -= v; }

        constexpr inline bool operator>(T v) const noexcept { return x > v && y > v && z > v; }
        constexpr inline bool operator<(T v) const noexcept { return x < v && y < v && z < v; }
        constexpr inline bool operator>=(T v) const noexcept { return x >= v && y >= v && z >= v; }
        constexpr inline bool operator<=(T v) const noexcept { return x <= v && y <= v && z <= v; }
        constexpr inline bool operator==(T v) const noexcept { return x == v && y == v && z == v; }
        constexpr inline bool operator!=(T v) const noexcept { return x != v || y != v || z != v; }

        constexpr inline T* data() noexcept { return &x; }

        [[nodiscard]] constexpr inline size_t size() const noexcept { return 3; }
        [[nodiscard]] constexpr inline T sum() const noexcept { return x + y + z; }
        [[nodiscard]] constexpr inline T prod() const noexcept { return x * y * z; }
        [[nodiscard]] constexpr inline T prodFFT() const noexcept { return (x / 2 + 1) * y * z; }

        [[nodiscard]] constexpr inline Int3<T> slice() const noexcept { return {x, y, 1}; }
        [[nodiscard]] constexpr inline T prodSlice() const noexcept { return x * y; }

        [[nodiscard]] inline std::string toString() const { return fmt::format("({}, {}, {})", x, y, z); }
        //@CLION-formatter:on
    };

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int3<T> min(Int3<T> i1, Int3<T> i2) noexcept {
        return {std::min(i1.x, i2.x), std::min(i1.y, i2.y), std::min(i1.z, i2.z)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int3<T> min(Int3<T> i1, T i2) noexcept {
        return {std::min(i1.x, i2), std::min(i1.y, i2), std::min(i1.z, i2)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int3<T> max(Int3<T> i1, Int3<T> i2) noexcept {
        return {std::max(i1.x, i2.x), std::max(i1.y, i2.y), std::max(i1.z, i2.z)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int3<T> max(Int3<T> i1, T i2) noexcept {
        return {std::max(i1.x, i2), std::max(i1.y, i2), std::max(i1.z, i2)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    std::ostream& operator<<(std::ostream& os, const Int3<T>& vec) {
        os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
        return os;
    }
}


namespace Noa {
    /** Integer static array of size 4. */
    template<typename T = int32_t, typename = std::enable_if_t<Traits::is_int_v<T>>>
    struct Int4 {
        T x{0}, y{0}, z{0}, w{0};

        constexpr Int4() = default;
        constexpr Int4(T xi, T yi, T zi, T wi) : x(xi), y(yi), z(zi), w(wi) {}

        constexpr explicit Int4(T v) : x(v), y(v), z(v), w(v) {}
        constexpr explicit Int4(T* data) : x(data[0]), y(data[1]), z(data[2]), w(data[3]) {}

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr explicit Int4(U v) : x(static_cast<T>(v)), y(static_cast<T>(v)),
                                       z(static_cast<T>(v)), w(static_cast<T>(v)) {}

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr explicit Int4(U* data) : x(static_cast<T>(data[0])), y(static_cast<T>(data[1])),
                                           z(static_cast<T>(data[2])), w(static_cast<T>(data[3])) {}

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr explicit Int4(Int4<U> vec) : x(static_cast<T>(vec.x)), y(static_cast<T>(vec.y)),
                                               z(static_cast<T>(vec.z)), w(static_cast<T>(vec.w)) {}

        //@CLION-formatter:off
        constexpr inline auto& operator=(T v) noexcept { x = v; y = v; z = v; w = v; return *this; }

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(U* v) noexcept {
            x = static_cast<T>(v[0]);
            y = static_cast<T>(v[1]);
            z = static_cast<T>(v[2]);
            w = static_cast<T>(v[3]);
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(Int4<U> vec) noexcept {
            x = static_cast<T>(vec.x);
            y = static_cast<T>(vec.y);
            z = static_cast<T>(vec.z);
            w = static_cast<T>(vec.w);
            return *this;
        }

        constexpr inline Int4<T> operator*(Int4<T> v) const noexcept { return {x * v.x, y * v.y, z * v.z, w * v.w}; }
        constexpr inline Int4<T> operator/(Int4<T> v) const noexcept { return {x / v.x, y / v.y, z / v.z, w / v.w}; }
        constexpr inline Int4<T> operator+(Int4<T> v) const noexcept { return {x + v.x, y + v.y, z + v.z, w + v.w}; }
        constexpr inline Int4<T> operator-(Int4<T> v) const noexcept { return {x - v.x, y - v.y, z - v.z, w - v.w}; }

        constexpr inline void operator*=(Int4<T> v) noexcept { x *= v.x; y *= v.y, z *= v.z, w *= v.w; }
        constexpr inline void operator/=(Int4<T> v) noexcept { x /= v.x; y /= v.y, z /= v.z, w /= v.w; }
        constexpr inline void operator+=(Int4<T> v) noexcept { x += v.x; y += v.y, z += v.z, w += v.w; }
        constexpr inline void operator-=(Int4<T> v) noexcept { x -= v.x; y -= v.y, z -= v.z, w -= v.w; }

        constexpr inline bool operator>(Int4<T> v) const noexcept { return x > v.x && y > v.y && z > v.z && w > v.w; }
        constexpr inline bool operator<(Int4<T> v) const noexcept { return x < v.x && y < v.y && z < v.z && w < v.w; }
        constexpr inline bool operator>=(Int4<T> v) const noexcept { return x >= v.x && y >= v.y && z >= v.z && w >= v.w; }
        constexpr inline bool operator<=(Int4<T> v) const noexcept { return x <= v.x && y <= v.y && z <= v.z && w <= v.w; }
        constexpr inline bool operator==(Int4<T> v) const noexcept { return x == v.x && y == v.y && z == v.z && w == v.w; }
        constexpr inline bool operator!=(Int4<T> v) const noexcept { return x != v.x || y != v.y || z != v.z || w != v.w; }

        constexpr inline Int4<T> operator*(T v) const noexcept { return {x * v, y * v, z * v, w * v}; }
        constexpr inline Int4<T> operator/(T v) const noexcept { return {x / v, y / v, z / v, w / v}; }
        constexpr inline Int4<T> operator+(T v) const noexcept { return {x + v, y + v, z + v, w + v}; }
        constexpr inline Int4<T> operator-(T v) const noexcept { return {x - v, y - v, z - v, w - v}; }

        constexpr inline void operator*=(T v) noexcept { x *= v; y *= v, z *= v, w *= v; }
        constexpr inline void operator/=(T v) noexcept { x /= v; y /= v, z /= v, w /= v; }
        constexpr inline void operator+=(T v) noexcept { x += v; y += v, z += v, w += v; }
        constexpr inline void operator-=(T v) noexcept { x -= v; y -= v, z -= v, w -= v; }

        constexpr inline bool operator>(T v) const noexcept { return x > v && y > v && z > v && w > v; }
        constexpr inline bool operator<(T v) const noexcept { return x < v && y < v && z < v && w < v; }
        constexpr inline bool operator>=(T v) const noexcept { return x >= v && y >= v && z >= v && w >= v; }
        constexpr inline bool operator<=(T v) const noexcept { return x <= v && y <= v && z <= v && w <= v; }
        constexpr inline bool operator==(T v) const noexcept { return x == v && y == v && z == v && w == v; }
        constexpr inline bool operator!=(T v) const noexcept { return x != v || y != v || z != v || w != v; }

        constexpr inline T* data() noexcept { return &x; }

        [[nodiscard]] constexpr inline size_t size() const noexcept { return 4; }
        [[nodiscard]] constexpr inline T sum() const noexcept { return x + y + z + w; }
        [[nodiscard]] constexpr inline T prod() const noexcept { return x * y * z * w; }
        [[nodiscard]] constexpr inline T prodFFT() const noexcept { return (x / 2 + 1) * y * z * w; }

        [[nodiscard]] constexpr inline Int4<T> slice() const noexcept { return {x, y, 1, 1}; }
        [[nodiscard]] constexpr inline T prodSlice() const noexcept { return x * y; }

        [[nodiscard]] inline std::string toString() const { return fmt::format("({}, {}, {}, {})", x, y, z, w); }
        //@CLION-formatter:on
    };

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int4<T> min(Int4<T> i1, Int4<T> i2) noexcept {
        return {std::min(i1.x, i2.x), std::min(i1.y, i2.y),
                std::min(i1.z, i2.z), std::min(i1.w, i2.w)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int4<T> min(Int4<T> i1, T i2) noexcept {
        return {std::min(i1.x, i2), std::min(i1.y, i2), std::min(i1.z, i2), std::min(i1.w, i2)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int4<T> max(Int4<T> i1, Int4<T> i2) noexcept {
        return {std::max(i1.x, i2.x), std::max(i1.y, i2.y),
                std::max(i1.z, i2.z), std::max(i1.w, i2.w)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Int4<T> max(Int4<T> i1, T i2) noexcept {
        return {std::max(i1.x, i2), std::max(i1.y, i2), std::max(i1.z, i2), std::max(i1.w, i2)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    std::ostream& operator<<(std::ostream& os, const Int4<T>& vec) {
        os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << ")";
        return os;
    }
}


namespace Noa {
    /** Float static array of size 2. */
    template<typename T = float, typename = std::enable_if_t<Traits::is_float_v<T>>>
    struct Float2 {
        T x{0}, y{0};

        Float2() = default;
        Float2(T xi, T yi) : x(xi), y(yi) {}
        explicit Float2(T v) : x(v), y(v) {}
        explicit Float2(T* data) : x(data[0]), y(data[1]) {}

        template<typename U, typename = std::enable_if_t<Traits::is_float_v<U>>>
        explicit Float2(U v) : x(static_cast<T>(v)), y(static_cast<T>(v)) {}

        template<typename U, typename = std::enable_if_t<Traits::is_float_v<U>>>
        explicit Float2(U* data) : x(static_cast<T>(data[0])), y(static_cast<T>(data[1])) {}

        template<typename U, typename = std::enable_if_t<Traits::is_float_v<U>>>
        explicit Float2(Float2<U> v) : x(static_cast<T>(v.x)), y(static_cast<T>(v.y)) {}

        template<typename U, typename = std::enable_if_t<Traits::is_float_v<U>>>
        inline auto& operator=(U* v) noexcept {
            x = static_cast<T>(v[0]);
            y = static_cast<T>(v[1]);
            return *this;
        }

        template<typename U>
        inline auto& operator=(Float2<U> v) noexcept {
            x = static_cast<T>(v.x);
            y = static_cast<T>(v.y);
            return *this;
        }

        //@CLION-formatter:off
        inline auto& operator=(T v) noexcept { x = v; y = v; return *this; }

        inline Float2<T> operator*(Float2<T> v) const noexcept { return {x * v.x, y * v.y}; }
        inline Float2<T> operator/(Float2<T> v) const noexcept { return {x / v.x, y / v.y}; }
        inline Float2<T> operator+(Float2<T> v) const noexcept { return {x + v.x, y + v.y}; }
        inline Float2<T> operator-(Float2<T> v) const noexcept { return {x - v.x, y - v.y}; }

        inline Float2<T> operator*(T v) const noexcept { return {x * v, y * v}; }
        inline Float2<T> operator/(T v) const noexcept { return {x / v, y / v}; }
        inline Float2<T> operator+(T v) const noexcept { return {x + v, y + v}; }
        inline Float2<T> operator-(T v) const noexcept { return {x - v, y - v}; }

        inline void operator*=(Float2<T> v) noexcept { x *= v.x; y *= v.y; }
        inline void operator/=(Float2<T> v) noexcept { x /= v.x; y /= v.y; }
        inline void operator+=(Float2<T> v) noexcept { x += v.x; y += v.y; }
        inline void operator-=(Float2<T> v) noexcept { x -= v.x; y -= v.y; }

        inline void operator*=(T v) noexcept { x *= v; y *= v; }
        inline void operator/=(T v) noexcept { x /= v; y /= v; }
        inline void operator+=(T v) noexcept { x += v; y += v; }
        inline void operator-=(T v) noexcept { x -= v; y -= v; }

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
        //@CLION-formatter:on

        [[nodiscard]] inline Float2<T> floor() const {
            return Float2<T>(std::floor(x), std::floor(y));
        }

        [[nodiscard]] inline Float2<T> ceil() const {
            return Float2<T>(std::ceil(x), std::ceil(y));
        }

        [[nodiscard]] inline T length() const { return std::sqrt(x * x + y * y); }
        [[nodiscard]] inline T lengthSq() const { return x * x + y * y; }

        [[nodiscard]] inline Float2<T> normalize() const { return *this / length(); }
        [[nodiscard]] inline T dot(Float2<T> v) const { return x * v.x + y * v.y; }
        [[nodiscard]] inline T cross(Float2<T> v) const { return x * v.y - y * v.x; }

        inline T* data() noexcept { return &x; }

        [[nodiscard]] inline std::string toString() const {
            return fmt::format("({}, {})", x, y);
        }
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
    /** Float static array of size 3. */
    template<typename T = float, typename = std::enable_if_t<Traits::is_float_v<T>>>
    struct Float3 {
        T x{0}, y{0}, z{0};

        Float3() = default;
        Float3(T xi, T yi, T zi) : x(xi), y(yi), z(zi) {}
        explicit Float3(T v) : x(v), y(v), z(v) {}
        explicit Float3(T* data) : x(data[0]), y(data[1]), z(data[2]) {}

        template<typename U>
        explicit Float3(U* data) : x(static_cast<T>(data[0])),
                                   y(static_cast<T>(data[1])),
                                   z(static_cast<T>(data[2])) {}

        template<typename U>
        explicit Float3(Float3<U> v) : x(static_cast<T>(v.x)),
                                       y(static_cast<T>(v.y)),
                                       z(static_cast<T>(v.z)) {}

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
            return fmt::format("({}, {}, {})", x, y, z);
        }

        template<typename U>
        inline auto& operator=(Float3<U> v) noexcept {
            x = static_cast<T>(v.x);
            y = static_cast<T>(v.y);
            z = static_cast<T>(v.z);
            return *this;
        }

        //@CLION-formatter:off
        inline auto& operator=(T v) noexcept { x = v; y = v; z = v; return *this; }

        inline Float3<T> operator*(Float3<T> v) const noexcept { return {x * v.x, y * v.y, z * v.z}; }
        inline Float3<T> operator/(Float3<T> v) const noexcept { return {x / v.x, y / v.y, z / v.z}; }
        inline Float3<T> operator+(Float3<T> v) const noexcept { return {x + v.x, y + v.y, z + v.z}; }
        inline Float3<T> operator-(Float3<T> v) const noexcept { return {x - v.x, y - v.y, z - v.z}; }

        inline Float3<T> operator*(T v) const noexcept { return {x * v, y * v, z * v}; }
        inline Float3<T> operator/(T v) const noexcept { return {x / v, y / v, z / v}; }
        inline Float3<T> operator+(T v) const noexcept { return {x + v, y + v, z + v}; }
        inline Float3<T> operator-(T v) const noexcept { return {x - v, y - v, z - v}; }

        inline void operator*=(Float3<T> v) noexcept { x *= v.x; y *= v.y; z *= v.z; }
        inline void operator/=(Float3<T> v) noexcept { x /= v.x; y /= v.y; z /= v.z; }
        inline void operator+=(Float3<T> v) noexcept { x += v.x; y += v.y; z += v.z; }
        inline void operator-=(Float3<T> v) noexcept { x -= v.x; y -= v.y; z -= v.z; }

        inline void operator*=(T v) noexcept { x *= v; y *= v; z *= v; }
        inline void operator/=(T v) noexcept { x /= v; y /= v; z /= v; }
        inline void operator+=(T v) noexcept { x += v; y += v; z += v; }
        inline void operator-=(T v) noexcept { x -= v; y -= v; z -= v; }

        inline bool operator==(Float3<T> v) const noexcept { return x == v.x && y == v.y && z == v.z; }
        inline bool operator!=(Float3<T> v) const noexcept { return x != v.x && y != v.y && z != v.z; }
        inline bool operator>=(Float3<T> v) const noexcept { return x >= v.x && y >= v.y && z >= v.z; }
        inline bool operator<=(Float3<T> v) const noexcept { return x <= v.x && y <= v.y && z <= v.z; }
        inline bool operator>(Float3<T> v) const noexcept { return x > v.x && y > v.y && z > v.z; }
        inline bool operator<(Float3<T> v) const noexcept { return x < v.x && y < v.y && z < v.z; }

        inline bool operator==(T v) const noexcept { return x == v && y == v && z == v; }
        inline bool operator!=(T v) const noexcept { return x != v && y != v && z != v; }
        inline bool operator>(T v) const noexcept { return x > v && y > v && z > v; }
        inline bool operator<(T v) const noexcept { return x < v && y < v && z < v; }
        inline bool operator>=(T v) const noexcept { return x >= v && y >= v && z >= v; }
        inline bool operator<=(T v) const noexcept { return x <= v && y <= v && z <= v; }
        //@CLION-formatter:on
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
    /** Integer static array of size 4. */
    template<typename T = int, typename = std::enable_if_t<Traits::is_float_v<T>>>
    struct Float4 {
        T x{0}, y{0}, z{0}, w{0};

        Float4() = default;
        Float4(T xi, T yi, T zi, T wi) : x(xi), y(yi), z(zi), w(wi) {}

        explicit Float4(T v) : x(v), y(v), z(v), w(v) {}
        explicit Float4(T* data) : x(data[0]), y(data[1]), z(data[2]), w(data[3]) {}

        //@CLION-formatter:off
        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        explicit Float4(U v) : x(static_cast<T>(v)), y(static_cast<T>(v)), z(static_cast<T>(v)), w(static_cast<T>(v)) {}

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        explicit Float4(U* data) : x(static_cast<T>(data[0])), y(static_cast<T>(data[1])), z(static_cast<T>(data[2])), w(static_cast<T>(data[3])) {}

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        explicit Float4(Float4<U> vec) : x(static_cast<T>(vec.x)), y(static_cast<T>(vec.y)), z(static_cast<T>(vec.z)), w(static_cast<T>(vec.w)) {}

        inline auto& operator=(T v) noexcept { x = v; y = v; z = v; w = v; return *this; }

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        inline auto& operator=(U* v) noexcept {
            x = static_cast<T>(v[0]);
            y = static_cast<T>(v[1]);
            z = static_cast<T>(v[2]);
            w = static_cast<T>(v[3]);
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v<U>>>
        inline auto& operator=(Float4<U> vec) noexcept {
            x = static_cast<T>(vec.x);
            y = static_cast<T>(vec.y);
            z = static_cast<T>(vec.z);
            w = static_cast<T>(vec.w);
            return *this;
        }

        inline Float4<T> operator*(Float4<T> v) const noexcept { return {x * v.x, y * v.y, z * v.z, w * v.w}; }
        inline Float4<T> operator/(Float4<T> v) const noexcept { return {x / v.x, y / v.y, z / v.z, w / v.w}; }
        inline Float4<T> operator+(Float4<T> v) const noexcept { return {x + v.x, y + v.y, z + v.z, w + v.w}; }
        inline Float4<T> operator-(Float4<T> v) const noexcept { return {x - v.x, y - v.y, z - v.z, w - v.w}; }

        inline void operator*=(Float4<T> v) noexcept { x *= v.x; y *= v.y, z *= v.z, w *= v.w; }
        inline void operator/=(Float4<T> v) noexcept { x /= v.x; y /= v.y, z /= v.z, w /= v.w; }
        inline void operator+=(Float4<T> v) noexcept { x += v.x; y += v.y, z += v.z, w += v.w; }
        inline void operator-=(Float4<T> v) noexcept { x -= v.x; y -= v.y, z -= v.z, w -= v.w; }

        inline bool operator>(Float4<T> v) const noexcept { return x > v.x && y > v.y && z > v.z && w > v.w; }
        inline bool operator<(Float4<T> v) const noexcept { return x < v.x && y < v.y && z < v.z && w < v.w; }
        inline bool operator>=(Float4<T> v) const noexcept { return x >= v.x && y >= v.y && z >= v.z && w >= v.w; }
        inline bool operator<=(Float4<T> v) const noexcept { return x <= v.x && y <= v.y && z <= v.z && w <= v.w; }
        inline bool operator==(Float4<T> v) const noexcept { return x == v.x && y == v.y && z == v.z && w == v.w; }
        inline bool operator!=(Float4<T> v) const noexcept { return x != v.x || y != v.y || z != v.z || w != v.w; }

        inline Float4<T> operator*(T v) const noexcept { return {x * v, y * v, z * v, w * v}; }
        inline Float4<T> operator/(T v) const noexcept { return {x / v, y / v, z / v, w / v}; }
        inline Float4<T> operator+(T v) const noexcept { return {x + v, y + v, z + v, w + v}; }
        inline Float4<T> operator-(T v) const noexcept { return {x - v, y - v, z - v, w - v}; }

        inline void operator*=(T v) noexcept { x *= v; y *= v, z *= v, w *= v; }
        inline void operator/=(T v) noexcept { x /= v; y /= v, z /= v, w /= v; }
        inline void operator+=(T v) noexcept { x += v; y += v, z += v, w += v; }
        inline void operator-=(T v) noexcept { x -= v; y -= v, z -= v, w -= v; }

        inline bool operator>(T v) const noexcept { return x > v && y > v && z > v && w > v; }
        inline bool operator<(T v) const noexcept { return x < v && y < v && z < v && w < v; }
        inline bool operator>=(T v) const noexcept { return x >= v && y >= v && z >= v && w >= v; }
        inline bool operator<=(T v) const noexcept { return x <= v && y <= v && z <= v && w <= v; }
        inline bool operator==(T v) const noexcept { return x == v && y == v && z == v && w == v; }
        inline bool operator!=(T v) const noexcept { return x != v || y != v || z != v || w != v; }

        inline T* data() noexcept { return &x; }

        [[nodiscard]] inline Float4<T> floor() const {
            return {std::floor(x), std::floor(y), std::floor(z), std::floor(w)};
        }

        [[nodiscard]] inline Float4<T> ceil() const {
            return {std::ceil(x), std::ceil(y), std::ceil(z), std::ceil(w)};
        }

        [[nodiscard]] inline T length() const { return sqrt(x * x + y * y + z * z + w * w); }
        [[nodiscard]] inline T lengthSq() const { return x * x + y * y + z * z + w * w; }

        [[nodiscard]] inline Float4<T> normalize() const { return *this / length(); }
        [[nodiscard]] inline T dot(Float4<T> v) const { return x * v.x + y * v.y + z * v.z; }
        [[nodiscard]] inline T cross(Float4<T> v) const { return x * v.y - y * v.x + z * v.z; }

        [[nodiscard]] inline T sum() const noexcept { return x + y + z + w; }
        [[nodiscard]] inline T prod() const noexcept { return x * y * z * w; }
        [[nodiscard]] inline T prodFFT() const noexcept { return (x / 2 + 1) * y * z * w; }

        [[nodiscard]] inline Float4<T> slice() const noexcept { return {x, y, 1, 1}; }
        [[nodiscard]] inline T prodSlice() const noexcept { return x * y; }

        [[nodiscard]] inline std::string toString() const { return fmt::format("({}, {}, {}, {})", x, y, z, w); }
        //@CLION-formatter:on
    };

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Float4<T> min(Float4<T> i1, Float4<T> i2) noexcept {
        return {std::min(i1.x, i2.x), std::min(i1.y, i2.y),
                std::min(i1.z, i2.z), std::min(i1.w, i2.w)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Float4<T> min(Float4<T> i1, T i2) noexcept {
        return {std::min(i1.x, i2), std::min(i1.y, i2), std::min(i1.z, i2), std::min(i1.w, i2)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Float4<T> max(Float4<T> i1, Float4<T> i2) noexcept {
        return {std::max(i1.x, i2.x), std::max(i1.y, i2.y),
                std::max(i1.z, i2.z), std::max(i1.w, i2.w)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    inline Float4<T> max(Float4<T> i1, T i2) noexcept {
        return {std::max(i1.x, i2), std::max(i1.y, i2), std::max(i1.z, i2), std::max(i1.w, i2)};
    }

    template<typename T, typename = std::enable_if_t<Traits::is_int_v<T>>>
    std::ostream& operator<<(std::ostream& os, const Float4<T>& vec) {
        os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << ")";
        return os;
    }
}


//@CLION-formatter:off
namespace Noa::Traits {
    template<typename T> struct p_is_int2 : std::false_type {};
    template<typename T> struct p_is_int2<Noa::Int2<T>> : std::true_type {};
    template<typename Int> struct NOA_API is_int2 { static constexpr bool value = p_is_int2<remove_ref_cv_t<Int>>::value; };
    template<typename Int> NOA_API inline constexpr bool is_int2_v = is_int2<Int>::value;
}

namespace Noa::Traits {
    template<typename T> struct p_is_int3 : std::false_type {};
    template<typename T> struct p_is_int3<Noa::Int3<T>> : std::true_type {};
    template<typename Int> struct NOA_API is_int3 { static constexpr bool value = p_is_int3<remove_ref_cv_t<Int>>::value; };
    template<typename Int> NOA_API inline constexpr bool is_int3_v = is_int3<Int>::value;
}

namespace Noa::Traits {
    template<typename T> struct p_is_int4 : std::false_type {};
    template<typename T> struct p_is_int4<Noa::Int4<T>> : std::true_type {};
    template<typename Int> struct NOA_API is_int4 { static constexpr bool value = p_is_int4<remove_ref_cv_t<Int>>::value; };
    template<typename Int> NOA_API inline constexpr bool is_int4_v = is_int4<Int>::value;
}

namespace Noa::Traits {
    template<typename T> struct NOA_API is_vector_int { static constexpr bool value = (is_int4_v<T> || is_int3_v<T> || is_int2_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_vector_int_v = is_vector_int<T>::value;
}

namespace Noa::Traits {
    template<typename T> struct p_is_float2 : std::false_type {};
    template<typename T> struct p_is_float2<Noa::Float2<T>> : std::true_type {};
    template<typename Float> struct NOA_API is_float2 { static constexpr bool value = p_is_float2<remove_ref_cv_t<Float>>::value; };
    template<typename Float> NOA_API inline constexpr bool is_float2_v = is_float2<Float>::value;
}

namespace Noa::Traits {
    template<typename T> struct p_is_float3 : std::false_type {};
    template<typename T> struct p_is_float3<Noa::Float3<T>> : std::true_type {};
    template<typename Float> struct NOA_API is_float3 { static constexpr bool value = p_is_float3<remove_ref_cv_t<Float>>::value; };
    template<typename Float> NOA_API inline constexpr bool is_float3_v = is_float3<Float>::value;
}

namespace Noa::Traits {
    template<typename T> struct p_is_float4 : std::false_type {};
    template<typename T> struct p_is_float4<Noa::Float4<T>> : std::true_type {};
    template<typename Float> struct NOA_API is_float4 { static constexpr bool value = p_is_float4<remove_ref_cv_t<Float>>::value; };
    template<typename Float> NOA_API inline constexpr bool is_float4_v = is_float4<Float>::value;
}

namespace Noa::Traits {
    template<typename T> struct NOA_API is_vector_float { static constexpr bool value = (is_float4_v<T> || is_float3_v<T> || is_float2_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_vector_float_v = is_vector_float<T>::value;
}

namespace Noa::Traits {
    template<typename T> struct NOA_API is_vector { static constexpr bool value = (is_vector_float_v<T> || is_vector_int_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_vector_v = is_vector<T>::value;
}
//@CLION-formatter:on
