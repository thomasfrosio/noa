/**
 * @file VectorInt.h
 * @brief Static arrays of integer.
 * @author Thomas - ffyr2w
 * @date 10/12/2020
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
 * As such, I should simply not use these "short" integers with these vectors.
 * Warning: Only int32_t, int64_t, uint32_t and uint64_t are tested!
 */

#define TO_T(x) static_cast<T>(x)
#define TO_SIZE(x) static_cast<size_t>(x)


namespace Noa {
    /** Static array of 3 integers. */
    template<typename T = int32_t, typename = std::enable_if_t<Noa::Traits::is_int_v<T>>>
    struct Int2 {
        T x{0}, y{0};

        //@CLION-formatter:off
        constexpr Int2() = default;
        constexpr Int2(T xi, T yi) : x(xi), y(yi) {}

        constexpr explicit Int2(T v) : x(v), y(v) {}
        constexpr explicit Int2(T* ptr) : x(ptr[0]), y(ptr[1]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr explicit Int2(U* ptr) : x(TO_T(ptr[0])), y(TO_T(ptr[1])) {}

        template<typename U>
        constexpr explicit Int2(Int2<U> vec) : x(TO_T(vec.x)), y(TO_T(vec.y)) {}

        // Assignment operators.
        constexpr inline auto& operator=(T v) noexcept { x = v; y = v; return *this; }
        constexpr inline auto& operator=(T* ptr) noexcept {x = ptr[0]; y = ptr[1]; return *this; }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(U* ptr) noexcept { x = TO_T(ptr[0]); y = TO_T(ptr[1]); return *this; }

        template<typename U>
        constexpr inline auto& operator=(Int2<U> vec) noexcept { x = TO_T(vec.x); y = TO_T(vec.y); return *this; }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_int_v<U>>>
        constexpr inline T& operator[](U idx) noexcept { return *(this->data() + idx); }

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

        [[nodiscard]] constexpr inline T* data() noexcept { return &x; }
        [[nodiscard]] static constexpr inline size_t size() noexcept { return 2; }
        [[nodiscard]] constexpr inline size_t sum() const noexcept { return TO_SIZE(x) + TO_SIZE(y); }
        [[nodiscard]] constexpr inline size_t prod() const noexcept { return TO_SIZE(x) * TO_SIZE(y); }
        [[nodiscard]] constexpr inline size_t prodFFT() const noexcept { return TO_SIZE(x / 2 + 1) * TO_SIZE(y); }
        [[nodiscard]] inline std::string toString() const { return fmt::format("({}, {})", x, y); }
        //@CLION-formatter:on
    };

    template<typename T>
    constexpr inline Int2<T> min(Int2<T> i1, Int2<T> i2) noexcept {
        return {std::min(i1.x, i2.x), std::min(i1.y, i2.y)};
    }

    template<typename T>
    constexpr inline Int2<T> min(Int2<T> i1, T i2) noexcept {
        return {std::min(i1.x, i2), std::min(i1.y, i2)};
    }

    template<typename T>
    constexpr inline Int2<T> max(Int2<T> i1, Int2<T> i2) noexcept {
        return {std::max(i1.x, i2.x), std::max(i1.y, i2.y)};
    }

    template<typename T>
    constexpr inline Int2<T> max(Int2<T> i1, T i2) noexcept {
        return {std::max(i1.x, i2), std::max(i1.y, i2)};
    }

    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Int2<T>& vec) {
        os << "(" << vec.x << ", " << vec.y << ")";
        return os;
    }
}


namespace Noa {
    /** Static array of 3 integers. */
    template<typename T = int32_t, typename = std::enable_if_t<Noa::Traits::is_int_v<T>>>
    struct Int3 {
        T x{0}, y{0}, z{0};

        //@CLION-formatter:off
        constexpr Int3() = default;
        constexpr Int3(T xi, T yi, T zi) : x(xi), y(yi), z(zi) {}

        constexpr explicit Int3(T v) : x(v), y(v), z(v) {}
        constexpr explicit Int3(T* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr explicit Int3(U* ptr) : x(TO_T(ptr[0])), y(TO_T(ptr[1])), z(TO_T(ptr[2])) {}

        template<typename U>
        constexpr explicit Int3(Int3<U> vec) : x(TO_T(vec.x)), y(TO_T(vec.y)), z(TO_T(vec.z)) {}

        // Assignment operators.
        constexpr inline auto& operator=(T v) noexcept { x = v; y = v; z = v; return *this; }
        constexpr inline auto& operator=(T* ptr) noexcept {x = ptr[0]; y = ptr[1]; z = ptr[2]; return *this; }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(U* ptr) noexcept { x = TO_T(ptr[0]); y = TO_T(ptr[1]); z = TO_T(ptr[2]); return *this; }

        template<typename U>
        constexpr inline auto& operator=(Int3<U> vec) noexcept { x = TO_T(vec.x); y = TO_T(vec.y); z = TO_T(vec.z); return *this; }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_int_v<U>>>
        constexpr inline T& operator[](U idx) noexcept { return *(this->data() + idx); }

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

        [[nodiscard]] constexpr inline T* data() noexcept { return &x; }
        [[nodiscard]] static constexpr inline size_t size() noexcept { return 3; }
        [[nodiscard]] constexpr inline size_t sum() const noexcept { return TO_SIZE(x) + TO_SIZE(y) + TO_SIZE(z); }
        [[nodiscard]] constexpr inline size_t prod() const noexcept { return TO_SIZE(x) * TO_SIZE(y) * TO_SIZE(z); }
        [[nodiscard]] constexpr inline size_t prodFFT() const noexcept { return TO_SIZE(x / 2 + 1) * TO_SIZE(y) * TO_SIZE(z); }

        [[nodiscard]] constexpr inline Int3<T> slice() const noexcept { return {x, y, 1}; }
        [[nodiscard]] constexpr inline size_t prodSlice() const noexcept { return TO_SIZE(x) * TO_SIZE(y); }

        [[nodiscard]] inline std::string toString() const { return fmt::format("({}, {}, {})", x, y, z); }
        //@CLION-formatter:on
    };

    template<typename T>
    inline Int3<T> min(Int3<T> i1, Int3<T> i2) noexcept {
        return {std::min(i1.x, i2.x), std::min(i1.y, i2.y), std::min(i1.z, i2.z)};
    }

    template<typename T>
    inline Int3<T> min(Int3<T> i1, T i2) noexcept {
        return {std::min(i1.x, i2), std::min(i1.y, i2), std::min(i1.z, i2)};
    }

    template<typename T>
    inline Int3<T> max(Int3<T> i1, Int3<T> i2) noexcept {
        return {std::max(i1.x, i2.x), std::max(i1.y, i2.y), std::max(i1.z, i2.z)};
    }

    template<typename T>
    inline Int3<T> max(Int3<T> i1, T i2) noexcept {
        return {std::max(i1.x, i2), std::max(i1.y, i2), std::max(i1.z, i2)};
    }

    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Int3<T>& vec) {
        os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
        return os;
    }
}


namespace Noa {
    /** Static array of 3 integers. */
    template<typename T = int32_t, typename = std::enable_if_t<Noa::Traits::is_int_v<T>>>
    struct Int4 {
        T x{0}, y{0}, z{0}, w{0};

        //@CLION-formatter:off
        constexpr Int4() = default;
        constexpr Int4(T xi, T yi, T zi, T wi) : x(xi), y(yi), z(zi), w(wi) {}

        constexpr explicit Int4(T v) : x(v), y(v), z(v), w(v) {}
        constexpr explicit Int4(T* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]), w(ptr[3]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr explicit Int4(U* ptr) : x(TO_T(ptr[0])), y(TO_T(ptr[1])), z(TO_T(ptr[2])), w(TO_T(ptr[3])) {}

        template<typename U>
        constexpr explicit Int4(Int4<U> vec) : x(TO_T(vec.x)), y(TO_T(vec.y)), z(TO_T(vec.z)), w(TO_T(vec.w)) {}

        // Assignment operators.
        constexpr inline auto& operator=(T v) noexcept { x = v; y = v; z = v; w = v; return *this; }
        constexpr inline auto& operator=(T* ptr) noexcept {x = ptr[0]; y = ptr[1]; z = ptr[2]; w = ptr[3]; return *this; }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        constexpr inline auto& operator=(U* v) noexcept { x = TO_T(v[0]); y = TO_T(v[1]); z = TO_T(v[2]); w = TO_T(v[3]); return *this; }

        template<typename U>
        constexpr inline auto& operator=(Int4<U> vec) noexcept { x = TO_T(vec.x); y = TO_T(vec.y); z = TO_T(vec.z); w = TO_T(vec.w); return *this; }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_int_v<U>>>
        constexpr inline T& operator[](U idx) noexcept { return *(this->data() + idx); }

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

        [[nodiscard]] constexpr inline T* data() noexcept { return &x; }
        [[nodiscard]] static constexpr inline size_t size() noexcept { return 4; }
        [[nodiscard]] constexpr inline size_t sum() const noexcept { return TO_SIZE(x) + TO_SIZE(y) + TO_SIZE(z) + TO_SIZE(w); }
        [[nodiscard]] constexpr inline size_t prod() const noexcept { return TO_SIZE(x) * TO_SIZE(y) * TO_SIZE(z) * TO_SIZE(w); }
        [[nodiscard]] constexpr inline size_t prodFFT() const noexcept { return TO_SIZE(x / 2 + 1) * TO_SIZE(y) * TO_SIZE(z) * TO_SIZE(w); }

        [[nodiscard]] constexpr inline Int4<T> slice() const noexcept { return {x, y, 1, 1}; }
        [[nodiscard]] constexpr inline size_t prodSlice() const noexcept { return TO_SIZE(x) * TO_SIZE(y); }

        [[nodiscard]] inline std::string toString() const { return fmt::format("({}, {}, {}, {})", x, y, z, w); }
        //@CLION-formatter:on
    };

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

    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Int4<T>& vec) {
        os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << ")";
        return os;
    }
}

#undef TO_T
#undef TO_SIZE

//@CLION-formatter:off
namespace Noa::Traits {
    template<typename> struct p_is_int2 : std::false_type {};
    template<typename T> struct p_is_int2<Noa::Int2<T>> : std::true_type {};
    template<typename T> struct NOA_API is_int2 { static constexpr bool value = p_is_int2<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_int2_v = is_int2<T>::value;

    template<typename> struct p_is_int3 : std::false_type {};
    template<typename T> struct p_is_int3<Noa::Int3<T>> : std::true_type {};
    template<typename Int> struct NOA_API is_int3 { static constexpr bool value = p_is_int3<remove_ref_cv_t<Int>>::value; };
    template<typename Int> NOA_API inline constexpr bool is_int3_v = is_int3<Int>::value;

    template<typename> struct p_is_int4 : std::false_type {};
    template<typename T> struct p_is_int4<Noa::Int4<T>> : std::true_type {};
    template<typename Int> struct NOA_API is_int4 { static constexpr bool value = p_is_int4<remove_ref_cv_t<Int>>::value; };
    template<typename Int> NOA_API inline constexpr bool is_int4_v = is_int4<Int>::value;

    template<typename T> struct NOA_API is_vector_int { static constexpr bool value = (is_int4_v<T> || is_int3_v<T> || is_int2_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_vector_int_v = is_vector_int<T>::value;

    template<typename> struct p_is_uint2 : std::false_type {};
    template<typename T> struct p_is_uint2<Noa::Int2<T>> { static constexpr bool value = is_uint_v<T>; };
    template<typename T> struct NOA_API is_uint2 { static constexpr bool value = p_is_uint2<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_uint2_v = is_uint2<T>::value;

    template<typename> struct p_is_uint3 : std::false_type {};
    template<typename T> struct p_is_uint3<Noa::Int3<T>> { static constexpr bool value = is_uint_v<T>; };
    template<typename T> struct NOA_API is_uint3 { static constexpr bool value = p_is_uint3<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_uint3_v = is_uint3<T>::value;

    template<typename> struct p_is_uint4 : std::false_type {};
    template<typename T> struct p_is_uint4<Noa::Int4<T>> { static constexpr bool value = is_uint_v<T>; };
    template<typename T> struct NOA_API is_uint4 { static constexpr bool value = p_is_uint4<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_uint4_v = is_uint4<T>::value;

    template<typename T> struct NOA_API is_vector_uint { static constexpr bool value = (is_uint4_v<T> || is_uint3_v<T> || is_uint2_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_vector_uint_v = is_vector_uint<T>::value;
}
//@CLION-formatter:on


template<typename T>
struct fmt::formatter<T, std::enable_if_t<Noa::Traits::is_vector_int_v<T>, char>>
        : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const T& a, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(a.toString(), ctx);
    }
};
