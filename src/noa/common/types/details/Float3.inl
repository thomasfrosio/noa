#ifndef NOA_INCLUDE_FLOAT3_
#error "This should not be directly included"
#endif

#include "noa/common/Assert.h"

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr T& Float3<T>::operator[](size_t i) {
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

    template<typename T>
    constexpr const T& Float3<T>::operator[](size_t i) const {
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

    // -- (Conversion) Constructors --

    template<typename T>
    template<typename X, typename Y, typename Z>
    constexpr Float3<T>::Float3(X xi, Y yi, Z zi) noexcept
            : x(static_cast<T>(xi)),
              y(static_cast<T>(yi)),
              z(static_cast<T>(zi)) {}

    template<typename T>
    template<typename U>
    constexpr Float3<T>::Float3(U v) noexcept
            : x(static_cast<T>(v)),
              y(static_cast<T>(v)),
              z(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    constexpr Float3<T>::Float3(const Float3<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)) {}

    template<typename T>
    template<typename U>
    constexpr Float3<T>::Float3(const Int3<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)) {}

    template<typename T>
    template<typename U>
    constexpr Float3<T>::Float3(U* ptr)
            : x(static_cast<T>(ptr[0])),
              y(static_cast<T>(ptr[1])),
              z(static_cast<T>(ptr[2])) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Float3<T>& Float3<T>::operator=(U v) noexcept {
        this->x = static_cast<T>(v);
        this->y = static_cast<T>(v);
        this->z = static_cast<T>(v);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float3<T>& Float3<T>::operator=(U* ptr) noexcept {
        this->x = static_cast<T>(ptr[0]);
        this->y = static_cast<T>(ptr[1]);
        this->z = static_cast<T>(ptr[2]);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float3<T>& Float3<T>::operator=(const Float3<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float3<T>& Float3<T>::operator=(const Int3<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float3<T>& Float3<T>::operator+=(const Float3<U>& rhs) noexcept {
        this->x += static_cast<T>(rhs.x);
        this->y += static_cast<T>(rhs.y);
        this->z += static_cast<T>(rhs.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float3<T>& Float3<T>::operator+=(U rhs) noexcept {
        this->x += static_cast<T>(rhs);
        this->y += static_cast<T>(rhs);
        this->z += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float3<T>& Float3<T>::operator-=(const Float3<U>& rhs) noexcept {
        this->x -= static_cast<T>(rhs.x);
        this->y -= static_cast<T>(rhs.y);
        this->z -= static_cast<T>(rhs.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float3<T>& Float3<T>::operator-=(U rhs) noexcept {
        this->x -= static_cast<T>(rhs);
        this->y -= static_cast<T>(rhs);
        this->z -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float3<T>& Float3<T>::operator*=(const Float3<U>& rhs) noexcept {
        this->x *= static_cast<T>(rhs.x);
        this->y *= static_cast<T>(rhs.y);
        this->z *= static_cast<T>(rhs.z);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Float3<T>& Float3<T>::operator*=(U rhs) noexcept {
        this->x *= static_cast<T>(rhs);
        this->y *= static_cast<T>(rhs);
        this->z *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float3<T>& Float3<T>::operator/=(const Float3<U>& rhs) noexcept {
        this->x /= static_cast<T>(rhs.x);
        this->y /= static_cast<T>(rhs.y);
        this->z /= static_cast<T>(rhs.z);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Float3<T>& Float3<T>::operator/=(U rhs) noexcept {
        this->x /= static_cast<T>(rhs);
        this->y /= static_cast<T>(rhs);
        this->z /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T>
    constexpr Float3<T> operator+(const Float3<T>& v) noexcept {
        return v;
    }

    template<typename T>
    constexpr Float3<T> operator-(const Float3<T>& v) noexcept {
        return {-v.x, -v.y, -v.z};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    constexpr Float3<T> operator+(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }
    template<typename T>
    constexpr Float3<T> operator+(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z};
    }
    template<typename T>
    constexpr Float3<T> operator+(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs};
    }

    template<typename T>
    constexpr Float3<T> operator-(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }
    template<typename T>
    constexpr Float3<T> operator-(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z};
    }
    template<typename T>
    constexpr Float3<T> operator-(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs};
    }

    template<typename T>
    constexpr Float3<T> operator*(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
    }
    template<typename T>
    constexpr Float3<T> operator*(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
    }
    template<typename T>
    constexpr Float3<T> operator*(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
    }

    template<typename T>
    constexpr Float3<T> operator/(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z};
    }
    template<typename T>
    constexpr Float3<T> operator/(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z};
    }
    template<typename T>
    constexpr Float3<T> operator/(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
    }

    // -- Comparison Operators --

    template<typename T>
    constexpr Bool3 operator>(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator>(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x > rhs, lhs.y > rhs, lhs.z > rhs};
    }
    template<typename T>
    constexpr Bool3 operator>(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs > rhs.x, lhs > rhs.y, lhs > rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator<(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator<(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x < rhs, lhs.y < rhs, lhs.z < rhs};
    }
    template<typename T>
    constexpr Bool3 operator<(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs < rhs.x, lhs < rhs.y, lhs < rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator>=(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator>=(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x >= rhs, lhs.y >= rhs, lhs.z >= rhs};
    }
    template<typename T>
    constexpr Bool3 operator>=(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator<=(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator<=(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x <= rhs, lhs.y <= rhs, lhs.z <= rhs};
    }
    template<typename T>
    constexpr Bool3 operator<=(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator==(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator==(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs};
    }
    template<typename T>
    constexpr Bool3 operator==(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator!=(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator!=(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs};
    }
    template<typename T>
    constexpr Bool3 operator!=(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z};
    }

    namespace math {
        template<typename T>
        constexpr Float3<T> toRad(const Float3<T>& v) {
            return Float3<T>(toRad(v.x), toRad(v.y), toRad(v.z));
        }

        template<typename T>
        constexpr Float3<T> toDeg(const Float3<T>& v) {
            return Float3<T>(toDeg(v.x), toDeg(v.y), toDeg(v.z));
        }

        template<typename T>
        constexpr Float3<T> floor(const Float3<T>& v) {
            return Float3<T>(floor(v.x), floor(v.y), floor(v.z));
        }

        template<typename T>
        constexpr Float3<T> ceil(const Float3<T>& v) {
            return Float3<T>(ceil(v.x), ceil(v.y), ceil(v.z));
        }

        template<typename T>
        constexpr Float3<T> abs(const Float3<T>& v) {
            return Float3<T>(abs(v.x), abs(v.y), abs(v.z));
        }

        template<typename T>
        constexpr T sum(const Float3<T>& v) noexcept {
            return v.x + v.y + v.z;
        }

        template<typename T>
        constexpr T prod(const Float3<T>& v) noexcept {
            return v.x * v.y * v.z;
        }

        template<typename T>
        constexpr T dot(const Float3<T>& a, const Float3<T>& b) noexcept {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        template<typename T>
        constexpr T innerProduct(const Float3<T>& a, const Float3<T>& b) noexcept {
            return dot(a, b);
        }

        template<typename T>
        constexpr T norm(const Float3<T>& v) {
            return sqrt(dot(v, v));
        }

        template<typename T>
        constexpr T length(const Float3<T>& v) {
            return norm(v);
        }

        template<typename T>
        constexpr Float3<T> normalize(const Float3<T>& v) {
            return v / norm(v);
        }

        template<typename T>
        constexpr Float3<T> cross(const Float3<T>& a, const Float3<T>& b) noexcept {
            return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
        }

        template<typename T>
        constexpr T min(const Float3<T>& v) noexcept {
            return (v.x < v.y) ? min(v.x, v.z) : min(v.y, v.z);
        }

        template<typename T>
        constexpr Float3<T> min(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)};
        }

        template<typename T>
        constexpr Float3<T> min(const Float3<T>& lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs)};
        }

        template<typename T>
        constexpr Float3<T> min(T lhs, const Float3<T>& rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z)};
        }

        template<typename T>
        constexpr T max(const Float3<T>& v) noexcept {
            return (v.x > v.y) ? max(v.x, v.z) : max(v.y, v.z);
        }

        template<typename T>
        constexpr Float3<T> max(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)};
        }

        template<typename T>
        constexpr Float3<T> max(const Float3<T>& lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs)};
        }

        template<typename T>
        constexpr Float3<T> max(T lhs, const Float3<T>& rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z)};
        }

        template<uint ULP, typename T>
        constexpr Bool3 isEqual(const Float3<T>& a, const Float3<T>& b, T e) {
            return {isEqual<ULP>(a.x, b.x, e), isEqual<ULP>(a.y, b.y, e), isEqual<ULP>(a.z, b.z, e)};
        }

        template<uint ULP, typename T>
        constexpr Bool3 isEqual(const Float3<T>& a, T b, T e) {
            return {isEqual<ULP>(a.x, b, e), isEqual<ULP>(a.y, b, e), isEqual<ULP>(a.z, b, e)};
        }

        template<uint ULP, typename T>
        constexpr Bool3 isEqual(T a, const Float3<T>& b, T e) {
            return {isEqual<ULP>(a, b.x, e), isEqual<ULP>(a, b.y, e), isEqual<ULP>(a, b.z, e)};
        }
    }
}
