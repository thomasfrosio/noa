#ifndef NOA_INCLUDE_INT3_
#error "This should not be directly included"
#endif

#include "noa/common/Assert.h"

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr T& Int3<T>::operator[](size_t i) {
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
    constexpr const T& Int3<T>::operator[](size_t i) const {
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
    constexpr Int3<T>::Int3(X xi, Y yi, Z zi) noexcept
            : x(static_cast<T>(xi)),
              y(static_cast<T>(yi)),
              z(static_cast<T>(zi)) {}

    template<typename T>
    template<typename U>
    constexpr Int3<T>::Int3(U v) noexcept
            : x(static_cast<T>(v)),
              y(static_cast<T>(v)),
              z(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    constexpr Int3<T>::Int3(const Int3<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)) {}

    template<typename T>
    template<typename U>
    constexpr Int3<T>::Int3(const Float3<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)) {}

    template<typename T>
    template<typename U>
    constexpr Int3<T>::Int3(U* ptr)
            : x(static_cast<T>(ptr[0])),
              y(static_cast<T>(ptr[1])),
              z(static_cast<T>(ptr[2])) {}

    template<typename T>
    template<typename U, typename V>
    constexpr Int3<T>::Int3(const Int2<U>& v, V oz) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(oz)) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator=(U v) noexcept {
        this->x = static_cast<T>(v);
        this->y = static_cast<T>(v);
        this->z = static_cast<T>(v);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator=(U* ptr) noexcept {
        this->x = static_cast<T>(ptr[0]);
        this->y = static_cast<T>(ptr[1]);
        this->z = static_cast<T>(ptr[2]);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator=(const Int3<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator=(const Float3<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator+=(const Int3<U>& rhs) noexcept {
        this->x += static_cast<T>(rhs.x);
        this->y += static_cast<T>(rhs.y);
        this->z += static_cast<T>(rhs.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator+=(U rhs) noexcept {
        this->x += static_cast<T>(rhs);
        this->y += static_cast<T>(rhs);
        this->z += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator-=(const Int3<U>& rhs) noexcept {
        this->x -= static_cast<T>(rhs.x);
        this->y -= static_cast<T>(rhs.y);
        this->z -= static_cast<T>(rhs.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator-=(U rhs) noexcept {
        this->x -= static_cast<T>(rhs);
        this->y -= static_cast<T>(rhs);
        this->z -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator*=(const Int3<U>& rhs) noexcept {
        this->x *= static_cast<T>(rhs.x);
        this->y *= static_cast<T>(rhs.y);
        this->z *= static_cast<T>(rhs.z);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator*=(U rhs) noexcept {
        this->x *= static_cast<T>(rhs);
        this->y *= static_cast<T>(rhs);
        this->z *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator/=(const Int3<U>& rhs) noexcept {
        this->x /= static_cast<T>(rhs.x);
        this->y /= static_cast<T>(rhs.y);
        this->z /= static_cast<T>(rhs.z);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator/=(U rhs) noexcept {
        this->x /= static_cast<T>(rhs);
        this->y /= static_cast<T>(rhs);
        this->z /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T> constexpr Int3<T> operator+(const Int3<T>& v) noexcept {
        return v;
    }

    template<typename T> constexpr Int3<T> operator-(const Int3<T>& v) noexcept {
        return {-v.x, -v.y, -v.z};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    constexpr Int3<T> operator+(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator+(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator+(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs};
    }

    template<typename T>
    constexpr Int3<T> operator-(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator-(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator-(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs};
    }

    template<typename T>
    constexpr Int3<T> operator*(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator*(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator*(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
    }

    template<typename T>
    constexpr Int3<T> operator/(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator/(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator/(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
    }

    // -- Comparison Operators --

    template<typename T>
    constexpr Bool3 operator>(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator>(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x > rhs, lhs.y > rhs, lhs.z > rhs};
    }
    template<typename T>
    constexpr Bool3 operator>(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs > rhs.x, lhs > rhs.y, lhs > rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator<(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator<(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x < rhs, lhs.y < rhs, lhs.z < rhs};
    }
    template<typename T>
    constexpr Bool3 operator<(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs < rhs.x, lhs < rhs.y, lhs < rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator>=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator>=(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x >= rhs, lhs.y >= rhs, lhs.z >= rhs};
    }
    template<typename T>
    constexpr Bool3 operator>=(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator<=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator<=(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x <= rhs, lhs.y <= rhs, lhs.z <= rhs};
    }
    template<typename T>
    constexpr Bool3 operator<=(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator==(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator==(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs};
    }
    template<typename T>
    constexpr Bool3 operator==(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator!=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator!=(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs};
    }
    template<typename T>
    constexpr Bool3 operator!=(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z};
    }

    template<typename T>
    constexpr T elements(const Int3<T>& v) noexcept {
        return v.x * v.y * v.z;
    }

    template<typename T>
    constexpr T elementsSlice(const Int3<T>& v) noexcept {
        return v.x * v.y;
    }

    template<typename T>
    constexpr T elementsFFT(const Int3<T>& v) noexcept {
        return (v.x / 2 + 1) * v.y * v.z;
    }

    template<typename T>
    constexpr Int3<T> shapeFFT(const Int3<T>& v) noexcept {
        return {v.x / 2 + 1, v.y, v.z};
    }

    template<typename T>
    constexpr Int3<T> slice(const Int3<T>& v) noexcept {
        return {v.x, v.y, 1};
    }

    namespace math {
        template<typename T>
        constexpr T sum(const Int3<T>& v) noexcept {
            return v.x + v.y + v.z;
        }

        template<typename T>
        constexpr T prod(const Int3<T>& v) noexcept {
            return v.x * v.y * v.z;
        }

        template<typename T>
        constexpr T min(const Int3<T>& v) noexcept {
            return (v.x < v.y) ? min(v.x, v.z) : min(v.y, v.z);
        }

        template<typename T>
        constexpr Int3<T> min(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)};
        }

        template<typename T>
        constexpr Int3<T> min(const Int3<T>& lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs)};
        }

        template<typename T>
        constexpr Int3<T> min(T lhs, const Int3<T>& rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z)};
        }

        template<typename T>
        constexpr T max(const Int3<T>& v) noexcept {
            return (v.x > v.y) ? max(v.x, v.z) : max(v.y, v.z);
        }

        template<typename T>
        constexpr Int3<T> max(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)};
        }

        template<typename T>
        constexpr Int3<T> max(const Int3<T>& lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs)};
        }

        template<typename T>
        constexpr Int3<T> max(T lhs, const Int3<T>& rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z)};
        }
    }
}
