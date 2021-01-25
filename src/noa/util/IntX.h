/**
 * @file VectorInt.h
 * @brief Static arrays of integer.
 * @author Thomas - ffyr2w
 * @date 10/12/2020
 */
#pragma once

#include <string>
#include <array>
#include <type_traits>
#include <spdlog/fmt/fmt.h>

#include "noa/util/traits/BaseTypes.h"
#include "noa/util/string/Format.h"

/**
 * Although the IntX vectors support "short" integers ((u)int8_t abd (u)int16_t), in most cases
 * there is an integral promotion performed before arithmetic operations. It then triggers
 * a narrowing conversion when the promoted integer needs to be casted back to these "short" integers.
 * See: https://stackoverflow.com/questions/24371868/why-must-a-short-be-converted-to-an-int-before-arithmetic-operations-in-c-and-c
 *
 * As such, I should simply not use these "short" integers with these vectors.
 * Warning: Only int32_t, int64_t, uint32_t and uint64_t are tested!
 *
 * @note 29/12/2020 - TF: Since the compiler is allowed to pad, the structures are not necessarily
 *       contiguous. Therefore, remove member function data() and add corresponding constructors.
 */


namespace Noa {
    template<typename, typename>
    struct Float2;

    /** Static array of 3 integers. */
    template<typename Int = int32_t, typename = std::enable_if_t<Noa::Traits::is_int_v<Int>>>
    struct Int2 {
        Int x{0}, y{0};

        // Constructors.
        constexpr Int2() = default;
        constexpr Int2(Int xi, Int yi) : x(xi), y(yi) {}

        constexpr explicit Int2(Int v) : x(v), y(v) {}
        constexpr explicit Int2(Int* ptr) : x(ptr[0]), y(ptr[1]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr explicit Int2(U* ptr) : x(Int(ptr[0])), y(Int(ptr[1])) {}

        template<typename U>
        constexpr explicit Int2(Int2<U> vec) : x(Int(vec.x)), y(Int(vec.y)) {}

        template<typename U, typename V>
        constexpr explicit Int2(Float2<U, V> vec) : x(Int(vec.x)), y(Int(vec.y)) {}

        // Assignment operators.
        inline constexpr auto& operator=(Int v) noexcept {
            x = v;
            y = v;
            return *this;
        }

        inline constexpr auto& operator=(Int* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        inline constexpr auto& operator=(U* ptr) noexcept {
            x = Int(ptr[0]);
            y = Int(ptr[1]);
            return *this;
        }

        template<typename U>
        inline constexpr auto& operator=(Int2<U> vec) noexcept {
            x = Int(vec.x);
            y = Int(vec.y);
            return *this;
        }

        template<typename U, typename V>
        inline constexpr auto& operator=(Float2<U, V> vec) noexcept {
            x = Int(vec.x);
            y = Int(vec.y);
            return *this;
        }

        //@CLION-formatter:off
        inline constexpr Int2<Int> operator*(Int2<Int> v) const noexcept { return {x * v.x, y * v.y}; }
        inline constexpr Int2<Int> operator/(Int2<Int> v) const noexcept { return {x / v.x, y / v.y}; }
        inline constexpr Int2<Int> operator+(Int2<Int> v) const noexcept { return {x + v.x, y + v.y}; }
        inline constexpr Int2<Int> operator-(Int2<Int> v) const noexcept { return {x - v.x, y - v.y}; }

        inline constexpr void operator*=(Int2<Int> v) noexcept { x *= v.x; y *= v.y; }
        inline constexpr void operator/=(Int2<Int> v) noexcept { x /= v.x; y /= v.y; }
        inline constexpr void operator+=(Int2<Int> v) noexcept { x += v.x; y += v.y; }
        inline constexpr void operator-=(Int2<Int> v) noexcept { x -= v.x; y -= v.y; }

        inline constexpr bool operator>(Int2<Int> v) const noexcept { return x > v.x && y > v.y; }
        inline constexpr bool operator<(Int2<Int> v) const noexcept { return x < v.x && y < v.y; }
        inline constexpr bool operator>=(Int2<Int> v) const noexcept { return x >= v.x && y >= v.y; }
        inline constexpr bool operator<=(Int2<Int> v) const noexcept { return x <= v.x && y <= v.y; }
        inline constexpr bool operator==(Int2<Int> v) const noexcept { return x == v.x && y == v.y; }
        inline constexpr bool operator!=(Int2<Int> v) const noexcept { return x != v.x || y != v.y; }

        inline constexpr Int2<Int> operator*(Int v) const noexcept { return {x * v, y * v}; }
        inline constexpr Int2<Int> operator/(Int v) const noexcept { return {x / v, y / v}; }
        inline constexpr Int2<Int> operator+(Int v) const noexcept { return {x + v, y + v}; }
        inline constexpr Int2<Int> operator-(Int v) const noexcept { return {x - v, y - v}; }

        inline constexpr void operator*=(Int v) noexcept { x *= v; y *= v; }
        inline constexpr void operator/=(Int v) noexcept { x /= v; y /= v; }
        inline constexpr void operator+=(Int v) noexcept { x += v; y += v; }
        inline constexpr void operator-=(Int v) noexcept { x -= v; y -= v; }

        inline constexpr bool operator>(Int v) const noexcept { return x > v && y > v; }
        inline constexpr bool operator<(Int v) const noexcept { return x < v && y < v; }
        inline constexpr bool operator>=(Int v) const noexcept { return x >= v && y >= v; }
        inline constexpr bool operator<=(Int v) const noexcept { return x <= v && y <= v; }
        inline constexpr bool operator==(Int v) const noexcept { return x == v && y == v; }
        inline constexpr bool operator!=(Int v) const noexcept { return x != v || y != v; }
        //@CLION-formatter:on

        [[nodiscard]] static inline constexpr size_t size() noexcept {
            return 2U;
        }

        [[nodiscard]] inline constexpr size_t sum() const noexcept {
            return size_t(x) + size_t(y);
        }

        [[nodiscard]] inline constexpr size_t prod() const noexcept {
            return size_t(x) * size_t(y);
        }

        [[nodiscard]] inline constexpr size_t prodFFT() const noexcept {
            return size_t(x / 2 + 1) * size_t(y);
        }

        [[nodiscard]] inline constexpr std::array<Int, 2U> toArray() const noexcept {
            return {x, y};
        }

        [[nodiscard]] inline std::string toString() const {
            return String::format("({}, {})", x, y);
        }
    };

    namespace Math {
        template<typename T>
        inline constexpr Int2<T> min(Int2<T> i1, Int2<T> i2) noexcept {
            return {std::min(i1.x, i2.x), std::min(i1.y, i2.y)};
        }

        template<typename T>
        inline constexpr Int2<T> min(Int2<T> i1, T i2) noexcept {
            return {std::min(i1.x, i2), std::min(i1.y, i2)};
        }

        template<typename T>
        inline constexpr Int2<T> max(Int2<T> i1, Int2<T> i2) noexcept {
            return {std::max(i1.x, i2.x), std::max(i1.y, i2.y)};
        }

        template<typename T>
        inline constexpr Int2<T> max(Int2<T> i1, T i2) noexcept {
            return {std::max(i1.x, i2), std::max(i1.y, i2)};
        }
    }
}

namespace Noa {
    template<typename, typename>
    struct Float3;

    /** Static array of 3 integers. */
    template<typename Int = int32_t, typename = std::enable_if_t<Noa::Traits::is_int_v<Int>>>
    struct Int3 {
        Int x{0}, y{0}, z{0};

        // Constructors.
        constexpr Int3() = default;
        constexpr Int3(Int xi, Int yi, Int zi) : x(xi), y(yi), z(zi) {}

        constexpr explicit Int3(Int v) : x(v), y(v), z(v) {}
        constexpr explicit Int3(Int* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr explicit Int3(U* ptr) : x(Int(ptr[0])), y(Int(ptr[1])), z(Int(ptr[2])) {}

        template<typename U>
        constexpr explicit Int3(Int3<U> vec) : x(Int(vec.x)), y(Int(vec.y)), z(Int(vec.z)) {}

        template<typename U, typename V>
        constexpr explicit Int3(Float3<U, V> vec) : x(Int(vec.x)), y(Int(vec.y)), z(Int(vec.z)) {}

        // Assignment operators.
        inline constexpr auto& operator=(Int v) noexcept {
            x = v;
            y = v;
            z = v;
            return *this;
        }

        inline constexpr auto& operator=(Int* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            z = ptr[2];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        inline constexpr auto& operator=(U* ptr) noexcept {
            x = Int(ptr[0]);
            y = Int(ptr[1]);
            z = Int(ptr[2]);
            return *this;
        }

        template<typename U>
        inline constexpr auto& operator=(Int3<U> vec) noexcept {
            x = Int(vec.x);
            y = Int(vec.y);
            z = Int(vec.z);
            return *this;
        }

        template<typename U, typename V>
        inline constexpr auto& operator=(Float3<U, V> vec) noexcept {
            x = Int(vec.x);
            y = Int(vec.y);
            z = Int(vec.z);
            return *this;
        }

        //@CLION-formatter:off
        inline constexpr Int3<Int> operator*(Int3<Int> v) const noexcept { return {x * v.x, y * v.y, z * v.z}; }
        inline constexpr Int3<Int> operator/(Int3<Int> v) const noexcept { return {x / v.x, y / v.y, z / v.z}; }
        inline constexpr Int3<Int> operator+(Int3<Int> v) const noexcept { return {x + v.x, y + v.y, z + v.z}; }
        inline constexpr Int3<Int> operator-(Int3<Int> v) const noexcept { return {x - v.x, y - v.y, z - v.z}; }

        inline constexpr void operator*=(Int3<Int> v) noexcept { x *= v.x; y *= v.y, z *= v.z; }
        inline constexpr void operator/=(Int3<Int> v) noexcept { x /= v.x; y /= v.y, z /= v.z; }
        inline constexpr void operator+=(Int3<Int> v) noexcept { x += v.x; y += v.y, z += v.z; }
        inline constexpr void operator-=(Int3<Int> v) noexcept { x -= v.x; y -= v.y, z -= v.z; }

        inline constexpr bool operator>(Int3<Int> v) const noexcept { return x > v.x && y > v.y && z > v.z; }
        inline constexpr bool operator<(Int3<Int> v) const noexcept { return x < v.x && y < v.y && z < v.z; }
        inline constexpr bool operator>=(Int3<Int> v) const noexcept { return x >= v.x && y >= v.y && z >= v.z; }
        inline constexpr bool operator<=(Int3<Int> v) const noexcept { return x <= v.x && y <= v.y && z <= v.z; }
        inline constexpr bool operator==(Int3<Int> v) const noexcept { return x == v.x && y == v.y && z == v.z; }
        inline constexpr bool operator!=(Int3<Int> v) const noexcept { return x != v.x || y != v.y || z != v.z; }

        inline constexpr Int3<Int> operator*(Int v) const noexcept { return {x * v, y * v, z * v}; }
        inline constexpr Int3<Int> operator/(Int v) const noexcept { return {x / v, y / v, z / v}; }
        inline constexpr Int3<Int> operator+(Int v) const noexcept { return {x + v, y + v, z + v}; }
        inline constexpr Int3<Int> operator-(Int v) const noexcept { return {x - v, y - v, z - v}; }

        inline constexpr void operator*=(Int v) noexcept { x *= v; y *= v, z *= v; }
        inline constexpr void operator/=(Int v) noexcept { x /= v; y /= v, z /= v; }
        inline constexpr void operator+=(Int v) noexcept { x += v; y += v, z += v; }
        inline constexpr void operator-=(Int v) noexcept { x -= v; y -= v, z -= v; }

        inline constexpr bool operator>(Int v) const noexcept { return x > v && y > v && z > v; }
        inline constexpr bool operator<(Int v) const noexcept { return x < v && y < v && z < v; }
        inline constexpr bool operator>=(Int v) const noexcept { return x >= v && y >= v && z >= v; }
        inline constexpr bool operator<=(Int v) const noexcept { return x <= v && y <= v && z <= v; }
        inline constexpr bool operator==(Int v) const noexcept { return x == v && y == v && z == v; }
        inline constexpr bool operator!=(Int v) const noexcept { return x != v || y != v || z != v; }
        //@CLION-formatter:on

        [[nodiscard]] static inline constexpr size_t size() noexcept {
            return 3U;
        }

        [[nodiscard]] inline constexpr size_t sum() const noexcept {
            return size_t(x) + size_t(y) + size_t(z);
        }

        [[nodiscard]] inline constexpr size_t prod() const noexcept {
            return size_t(x) * size_t(y) * size_t(z);
        }

        [[nodiscard]] inline constexpr size_t prodFFT() const noexcept {
            return size_t(x / 2 + 1) * size_t(y) * size_t(z);
        }

        [[nodiscard]] inline constexpr Int3<Int> slice() const noexcept {
            return {x, y, 1};
        }

        [[nodiscard]] inline constexpr size_t prodSlice() const noexcept {
            return size_t(x) * size_t(y);
        }

        [[nodiscard]] inline constexpr std::array<Int, 3U> toArray() const noexcept {
            return {x, y, z};
        }

        [[nodiscard]] inline std::string toString() const {
            return String::format("({}, {}, {})", x, y, z);
        }
    };

    namespace Math {
        template<typename T>
        inline constexpr Int3<T> min(Int3<T> i1, Int3<T> i2) noexcept {
            return {std::min(i1.x, i2.x), std::min(i1.y, i2.y), std::min(i1.z, i2.z)};
        }

        template<typename T>
        inline constexpr Int3<T> min(Int3<T> i1, T i2) noexcept {
            return {std::min(i1.x, i2), std::min(i1.y, i2), std::min(i1.z, i2)};
        }

        template<typename T>
        inline constexpr Int3<T> max(Int3<T> i1, Int3<T> i2) noexcept {
            return {std::max(i1.x, i2.x), std::max(i1.y, i2.y), std::max(i1.z, i2.z)};
        }

        template<typename T>
        inline constexpr Int3<T> max(Int3<T> i1, T i2) noexcept {
            return {std::max(i1.x, i2), std::max(i1.y, i2), std::max(i1.z, i2)};
        }
    }
}

namespace Noa {
    template<typename, typename>
    struct Float4;

    /** Static array of 3 integers. */
    template<typename Int = int32_t, typename = std::enable_if_t<Noa::Traits::is_int_v<Int>>>
    struct Int4 {
        Int x{0}, y{0}, z{0}, w{0};

        // Constructors.
        constexpr Int4() = default;
        constexpr Int4(Int xi, Int yi, Int zi, Int wi) : x(xi), y(yi), z(zi), w(wi) {}

        constexpr explicit Int4(Int v) : x(v), y(v), z(v), w(v) {}
        constexpr explicit Int4(Int* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]), w(ptr[3]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr explicit Int4(U* ptr)
                : x(Int(ptr[0])), y(Int(ptr[1])), z(Int(ptr[2])), w(Int(ptr[3])) {}

        template<typename U>
        constexpr explicit Int4(Int4<U> vec)
                : x(Int(vec.x)), y(Int(vec.y)), z(Int(vec.z)), w(Int(vec.w)) {}

        template<typename U, typename V>
        constexpr explicit Int4(Float4<U, V> vec)
                : x(Int(vec.x)), y(Int(vec.y)), z(Int(vec.z)), w(Int(vec.w)) {}

        // Assignment operators.
        inline constexpr auto& operator=(Int v) noexcept {
            x = v;
            y = v;
            z = v;
            w = v;
            return *this;
        }
        inline constexpr auto& operator=(Int* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            z = ptr[2];
            w = ptr[3];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        inline constexpr auto& operator=(U* v) noexcept {
            x = Int(v[0]);
            y = Int(v[1]);
            z = Int(v[2]);
            w = Int(v[3]);
            return *this;
        }

        template<typename U>
        inline constexpr auto& operator=(Int4<U> vec) noexcept {
            x = Int(vec.x);
            y = Int(vec.y);
            z = Int(vec.z);
            w = Int(vec.w);
            return *this;
        }

        template<typename U, typename V>
        inline constexpr auto& operator=(Float4<U, V> vec) noexcept {
            x = Int(vec.x);
            y = Int(vec.y);
            z = Int(vec.z);
            w = Int(vec.w);
            return *this;
        }

        //@CLION-formatter:off
        inline constexpr Int4<Int> operator*(Int4<Int> v) const noexcept { return {x * v.x, y * v.y, z * v.z, w * v.w}; }
        inline constexpr Int4<Int> operator/(Int4<Int> v) const noexcept { return {x / v.x, y / v.y, z / v.z, w / v.w}; }
        inline constexpr Int4<Int> operator+(Int4<Int> v) const noexcept { return {x + v.x, y + v.y, z + v.z, w + v.w}; }
        inline constexpr Int4<Int> operator-(Int4<Int> v) const noexcept { return {x - v.x, y - v.y, z - v.z, w - v.w}; }

        inline constexpr void operator*=(Int4<Int> v) noexcept { x *= v.x; y *= v.y, z *= v.z, w *= v.w; }
        inline constexpr void operator/=(Int4<Int> v) noexcept { x /= v.x; y /= v.y, z /= v.z, w /= v.w; }
        inline constexpr void operator+=(Int4<Int> v) noexcept { x += v.x; y += v.y, z += v.z, w += v.w; }
        inline constexpr void operator-=(Int4<Int> v) noexcept { x -= v.x; y -= v.y, z -= v.z, w -= v.w; }

        inline constexpr bool operator>(Int4<Int> v) const noexcept { return x > v.x && y > v.y && z > v.z && w > v.w; }
        inline constexpr bool operator<(Int4<Int> v) const noexcept { return x < v.x && y < v.y && z < v.z && w < v.w; }
        inline constexpr bool operator>=(Int4<Int> v) const noexcept { return x >= v.x && y >= v.y && z >= v.z && w >= v.w; }
        inline constexpr bool operator<=(Int4<Int> v) const noexcept { return x <= v.x && y <= v.y && z <= v.z && w <= v.w; }
        inline constexpr bool operator==(Int4<Int> v) const noexcept { return x == v.x && y == v.y && z == v.z && w == v.w; }
        inline constexpr bool operator!=(Int4<Int> v) const noexcept { return x != v.x || y != v.y || z != v.z || w != v.w; }

        inline constexpr Int4<Int> operator*(Int v) const noexcept { return {x * v, y * v, z * v, w * v}; }
        inline constexpr Int4<Int> operator/(Int v) const noexcept { return {x / v, y / v, z / v, w / v}; }
        inline constexpr Int4<Int> operator+(Int v) const noexcept { return {x + v, y + v, z + v, w + v}; }
        inline constexpr Int4<Int> operator-(Int v) const noexcept { return {x - v, y - v, z - v, w - v}; }

        inline constexpr void operator*=(Int v) noexcept { x *= v; y *= v, z *= v, w *= v; }
        inline constexpr void operator/=(Int v) noexcept { x /= v; y /= v, z /= v, w /= v; }
        inline constexpr void operator+=(Int v) noexcept { x += v; y += v, z += v, w += v; }
        inline constexpr void operator-=(Int v) noexcept { x -= v; y -= v, z -= v, w -= v; }

        inline constexpr bool operator>(Int v) const noexcept { return x > v && y > v && z > v && w > v; }
        inline constexpr bool operator<(Int v) const noexcept { return x < v && y < v && z < v && w < v; }
        inline constexpr bool operator>=(Int v) const noexcept { return x >= v && y >= v && z >= v && w >= v; }
        inline constexpr bool operator<=(Int v) const noexcept { return x <= v && y <= v && z <= v && w <= v; }
        inline constexpr bool operator==(Int v) const noexcept { return x == v && y == v && z == v && w == v; }
        inline constexpr bool operator!=(Int v) const noexcept { return x != v || y != v || z != v || w != v; }
        //@CLION-formatter:on

        [[nodiscard]] static inline constexpr size_t size() noexcept {
            return 4U;
        }

        [[nodiscard]] inline constexpr size_t sum() const noexcept {
            return size_t(x) + size_t(y) + size_t(z) + size_t(w);
        }

        [[nodiscard]] inline constexpr size_t prod() const noexcept {
            return size_t(x) * size_t(y) * size_t(z) * size_t(w);
        }

        [[nodiscard]] inline constexpr size_t prodFFT() const noexcept {
            return size_t(x / 2 + 1) * size_t(y) * size_t(z) * size_t(w);
        }

        [[nodiscard]] inline constexpr Int4<Int> slice() const noexcept {
            return {x, y, 1, 1};
        }

        [[nodiscard]] inline constexpr size_t prodSlice() const noexcept {
            return size_t(x) * size_t(y);
        }

        [[nodiscard]] inline constexpr std::array<Int, 4U> toArray() const noexcept {
            return {x, y, z, w};
        }

        [[nodiscard]] inline std::string toString() const {
            return String::format("({}, {}, {}, {})", x, y, z, w);
        }
    };

    namespace Math {
        template<typename T>
        inline Int4<T> min(Int4<T> i1, Int4<T> i2) noexcept {
            return {std::min(i1.x, i2.x), std::min(i1.y, i2.y),
                    std::min(i1.z, i2.z), std::min(i1.w, i2.w)};
        }

        template<typename T>
        inline Int4<T> min(Int4<T> i1, T i2) noexcept {
            return {std::min(i1.x, i2), std::min(i1.y, i2), std::min(i1.z, i2), std::min(i1.w, i2)};
        }

        template<typename T>
        inline Int4<T> max(Int4<T> i1, Int4<T> i2) noexcept {
            return {std::max(i1.x, i2.x), std::max(i1.y, i2.y),
                    std::max(i1.z, i2.z), std::max(i1.w, i2.w)};
        }

        template<typename T>
        inline Int4<T> max(Int4<T> i1, T i2) noexcept {
            return {std::max(i1.x, i2), std::max(i1.y, i2), std::max(i1.z, i2), std::max(i1.w, i2)};
        }
    }
}


//@CLION-formatter:off
namespace Noa::Traits {
    template<typename> struct p_is_int2 : std::false_type {};
    template<typename T> struct p_is_int2<Noa::Int2<T>> : std::true_type {};
    template<typename T> using is_int2 = std::bool_constant<p_is_int2<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_int2_v = is_int2<T>::value;

    template<typename> struct p_is_int3 : std::false_type {};
    template<typename T> struct p_is_int3<Noa::Int3<T>> : std::true_type {};
    template<typename T> using is_int3 = std::bool_constant<p_is_int3<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_int3_v = is_int3<T>::value;

    template<typename> struct p_is_int4 : std::false_type {};
    template<typename T> struct p_is_int4<Noa::Int4<T>> : std::true_type {};
    template<typename T> using is_int4 = std::bool_constant<p_is_int4<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_int4_v = is_int4<T>::value;

    template<typename T> using is_intX = std::bool_constant<is_int4_v<T> || is_int3_v<T> || is_int2_v<T>>;
    template<typename T> constexpr bool is_intX_v = is_intX<T>::value;

    template<typename> struct p_is_uint2 : std::false_type {};
    template<typename T> struct p_is_uint2<Noa::Int2<T>> : std::bool_constant<is_uint_v<T>> {};
    template<typename T> using is_uint2 = std::bool_constant<p_is_uint2<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_uint2_v = is_uint2<T>::value;

    template<typename> struct p_is_uint3 : std::false_type {};
    template<typename T> struct p_is_uint3<Noa::Int3<T>> : std::bool_constant<is_uint_v<T>> {};
    template<typename T> using is_uint3 = std::bool_constant<p_is_uint3<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_uint3_v = is_uint3<T>::value;

    template<typename> struct p_is_uint4 : std::false_type {};
    template<typename T> struct p_is_uint4<Noa::Int4<T>> : std::bool_constant<is_uint_v<T>> {};
    template<typename T> using is_uint4 = std::bool_constant<p_is_uint4<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_uint4_v = is_uint4<T>::value;

    template<typename T> using is_uintX = std::bool_constant<is_uint4_v<T> || is_uint3_v<T> || is_uint2_v<T>>;
    template<typename T> constexpr bool is_uintX_v = is_uintX<T>::value;
}
//@CLION-formatter:on


template<typename IntX>
struct fmt::formatter<IntX, std::enable_if_t<Noa::Traits::is_intX_v<IntX>, char>>
        : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const IntX& ints, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(ints.toString(), ctx);
    }
};

template<typename Float>
std::ostream& operator<<(std::ostream& os, const Noa::Int2<Float>& ints) {
    os << ints.toString();
    return os;
}

template<typename Float>
std::ostream& operator<<(std::ostream& os, const Noa::Int3<Float>& ints) {
    os << ints.toString();
    return os;
}

template<typename Float>
std::ostream& operator<<(std::ostream& os, const Noa::Int4<Float>& ints) {
    os << ints.toString();
    return os;
}
