#ifndef NOA_INCLUDE_INT4_
#error "This should not be directly included"
#endif

#include "noa/common/Assert.h"

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr T& Int4<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->COUNT);
        switch (i) {
            default:
            case 0:
                return this->x;
            case 1:
                return this->y;
            case 2:
                return this->z;
            case 3:
                return this->w;
        }
    }

    template<typename T>
    constexpr const T& Int4<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->COUNT);
        switch (i) {
            default:
            case 0:
                return this->x;
            case 1:
                return this->y;
            case 2:
                return this->z;
            case 3:
                return this->w;
        }
    }

    // -- (Conversion) Constructors --

    template<typename T>
    template<typename X, typename Y, typename Z, typename W>
    constexpr Int4<T>::Int4(X xi, Y yi, Z zi, W wi) noexcept
            : x(static_cast<T>(xi)),
              y(static_cast<T>(yi)),
              z(static_cast<T>(zi)),
              w(static_cast<T>(wi)) {}

    template<typename T>
    template<typename U>
    constexpr Int4<T>::Int4(U v) noexcept
            : x(static_cast<T>(v)),
              y(static_cast<T>(v)),
              z(static_cast<T>(v)),
              w(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    constexpr Int4<T>::Int4(const Int4<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)),
              w(static_cast<T>(v.w)) {}

    template<typename T>
    template<typename U>
    constexpr Int4<T>::Int4(const Float4<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)),
              w(static_cast<T>(v.w)) {}

    template<typename T>
    template<typename U>
    constexpr Int4<T>::Int4(U* ptr)
            : x(static_cast<T>(ptr[0])),
              y(static_cast<T>(ptr[1])),
              z(static_cast<T>(ptr[2])),
              w(static_cast<T>(ptr[3])) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator=(U v) noexcept {
        this->x = static_cast<T>(v);
        this->y = static_cast<T>(v);
        this->z = static_cast<T>(v);
        this->w = static_cast<T>(v);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator=(U* ptr) noexcept {
        this->x = static_cast<T>(ptr[0]);
        this->y = static_cast<T>(ptr[1]);
        this->z = static_cast<T>(ptr[2]);
        this->w = static_cast<T>(ptr[3]);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator=(const Int4<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        this->w = static_cast<T>(v.w);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator=(const Float4<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        this->w = static_cast<T>(v.w);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator+=(const Int4<U>& rhs) noexcept {
        this->x += static_cast<T>(rhs.x);
        this->y += static_cast<T>(rhs.y);
        this->z += static_cast<T>(rhs.z);
        this->w += static_cast<T>(rhs.w);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator+=(U rhs) noexcept {
        this->x += static_cast<T>(rhs);
        this->y += static_cast<T>(rhs);
        this->z += static_cast<T>(rhs);
        this->w += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator-=(const Int4<U>& rhs) noexcept {
        this->x -= static_cast<T>(rhs.x);
        this->y -= static_cast<T>(rhs.y);
        this->z -= static_cast<T>(rhs.z);
        this->w -= static_cast<T>(rhs.w);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator-=(U rhs) noexcept {
        this->x -= static_cast<T>(rhs);
        this->y -= static_cast<T>(rhs);
        this->z -= static_cast<T>(rhs);
        this->w -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator*=(const Int4<U>& rhs) noexcept {
        this->x *= static_cast<T>(rhs.x);
        this->y *= static_cast<T>(rhs.y);
        this->z *= static_cast<T>(rhs.z);
        this->w *= static_cast<T>(rhs.w);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator*=(U rhs) noexcept {
        this->x *= static_cast<T>(rhs);
        this->y *= static_cast<T>(rhs);
        this->z *= static_cast<T>(rhs);
        this->w *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator/=(const Int4<U>& rhs) noexcept {
        this->x /= static_cast<T>(rhs.x);
        this->y /= static_cast<T>(rhs.y);
        this->z /= static_cast<T>(rhs.z);
        this->w /= static_cast<T>(rhs.w);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator/=(U rhs) noexcept {
        this->x /= static_cast<T>(rhs);
        this->y /= static_cast<T>(rhs);
        this->z /= static_cast<T>(rhs);
        this->w /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T> constexpr Int4<T> operator+(const Int4<T>& v) noexcept {
        return v;
    }

    template<typename T> constexpr Int4<T> operator-(const Int4<T>& v) noexcept {
        return {-v.x, -v.y, -v.z, -v.w};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    constexpr Int4<T> operator+(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator+(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator+(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs};
    }

    template<typename T>
    constexpr Int4<T> operator-(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator-(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator-(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs};
    }

    template<typename T>
    constexpr Int4<T> operator*(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator*(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator*(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs};
    }

    template<typename T>
    constexpr Int4<T> operator/(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator/(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator/(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs};
    }

    // -- Comparison Operators --

    template<typename T>
    constexpr Bool4 operator>(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z, lhs.w > rhs.w};
    }
    template<typename T>
    constexpr Bool4 operator>(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x > rhs, lhs.y > rhs, lhs.z > rhs, lhs.w > rhs};
    }
    template<typename T>
    constexpr Bool4 operator>(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs > rhs.x, lhs > rhs.y, lhs > rhs.z, lhs > rhs.w};
    }

    template<typename T>
    constexpr Bool4 operator<(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z, lhs.w < rhs.w};
    }
    template<typename T>
    constexpr Bool4 operator<(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x < rhs, lhs.y < rhs, lhs.z < rhs, lhs.w < rhs};
    }
    template<typename T>
    constexpr Bool4 operator<(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs < rhs.x, lhs < rhs.y, lhs < rhs.z, lhs < rhs.w};
    }

    template<typename T>
    constexpr Bool4 operator>=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z, lhs.w >= rhs.w};
    }
    template<typename T>
    constexpr Bool4 operator>=(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x >= rhs, lhs.y >= rhs, lhs.z >= rhs, lhs.w >= rhs};
    }
    template<typename T>
    constexpr Bool4 operator>=(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.z, lhs >= rhs.w};
    }

    template<typename T>
    constexpr Bool4 operator<=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z, lhs.w <= rhs.w};
    }
    template<typename T>
    constexpr Bool4 operator<=(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x <= rhs, lhs.y <= rhs, lhs.z <= rhs, lhs.w <= rhs};
    }
    template<typename T>
    constexpr Bool4 operator<=(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.z, lhs <= rhs.w};
    }

    template<typename T>
    constexpr Bool4 operator==(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w};
    }
    template<typename T>
    constexpr Bool4 operator==(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs, lhs.w == rhs};
    }
    template<typename T>
    constexpr Bool4 operator==(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z, lhs == rhs.w};
    }

    template<typename T>
    constexpr Bool4 operator!=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w};
    }
    template<typename T>
    constexpr Bool4 operator!=(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs, lhs.w != rhs};
    }
    template<typename T>
    constexpr Bool4 operator!=(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z, lhs != rhs.w};
    }

    template<typename T>
    constexpr T elements(const Int4<T>& v) noexcept {
        return v.x * v.y * v.z * v.w;
    }

    template<typename T>
    constexpr T elementsSlice(const Int4<T>& v) noexcept {
        return v.x * v.y;
    }

    template<typename T>
    constexpr T elementsFFT(const Int4<T>& v) noexcept {
        return (v.x / 2 + 1) * v.y * v.z * v.w;
    }

    template<typename T>
    constexpr Int4<T> shapeFFT(const Int4<T>& v) noexcept {
        return {v.x / 2 + 1, v.y, v.z, v.w};
    }

    template<typename T>
    constexpr Int4<T> slice(const Int4<T>& v) noexcept {
        return {v.x, v.y, 1, 1};
    }

    namespace math {
        template<typename T>
        constexpr T sum(const Int4<T>& v) noexcept {
            return v.x + v.y + v.z + v.w;
        }

        template<typename T>
        constexpr T prod(const Int4<T>& v) noexcept {
            return v.x * v.y * v.z * v.w;
        }

        template<typename T>
        constexpr T min(const Int4<T>& v) noexcept {
            return min(min(v.x, v.y), min(v.z, v.w));
        }

        template<typename T>
        constexpr Int4<T> min(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z), min(lhs.w, rhs.w)};
        }

        template<typename T>
        constexpr Int4<T> min(const Int4<T>& lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs), min(lhs.w, rhs)};
        }

        template<typename T>
        constexpr Int4<T> min(T lhs, const Int4<T>& rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z), min(lhs, rhs.w)};
        }

        template<typename T>
        constexpr T max(const Int4<T>& v) noexcept {
            return max(max(v.x, v.y), max(v.z, v.w));
        }

        template<typename T>
        constexpr Int4<T> max(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z), max(lhs.w, rhs.w)};
        }

        template<typename T>
        constexpr Int4<T> max(const Int4<T>& lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs), max(lhs.w, rhs)};
        }

        template<typename T>
        constexpr Int4<T> max(T lhs, const Int4<T>& rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z), max(lhs, rhs.w)};
        }
    }
}
