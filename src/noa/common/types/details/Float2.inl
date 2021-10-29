#ifndef NOA_INCLUDE_FLOAT2_
#error "This should not be directly included"
#endif

#include "noa/common/Assert.h"

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr T& Float2<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->COUNT);
        if (i == 1)
            return this->y;
        else
            return this->x;
    }

    template<typename T>
    constexpr const T& Float2<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->COUNT);
        if (i == 1)
            return this->y;
        else
            return this->x;
    }

    // -- (Conversion) Constructors --

    template<typename T>
    template<typename X, typename Y>
    constexpr Float2<T>::Float2(X xi, Y yi) noexcept
            : x(static_cast<T>(xi)),
              y(static_cast<T>(yi)) {}

    template<typename T>
    template<typename U>
    constexpr Float2<T>::Float2(U v) noexcept
            : x(static_cast<T>(v)),
              y(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    constexpr Float2<T>::Float2(const Float2<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)) {}

    template<typename T>
    template<typename U>
    constexpr Float2<T>::Float2(const Int2<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)) {}

    template<typename T>
    template<typename U>
    constexpr Float2<T>::Float2(U* ptr)
            : x(static_cast<T>(ptr[0])),
              y(static_cast<T>(ptr[1])) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Float2<T>& Float2<T>::operator=(U v) noexcept {
        this->x = static_cast<T>(v);
        this->y = static_cast<T>(v);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float2<T>& Float2<T>::operator=(U* ptr) {
        this->x = static_cast<T>(ptr[0]);
        this->y = static_cast<T>(ptr[1]);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float2<T>& Float2<T>::operator=(const Float2<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float2<T>& Float2<T>::operator=(const Int2<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float2<T>& Float2<T>::operator+=(const Float2<U>& rhs) noexcept {
        this->x += static_cast<T>(rhs.x);
        this->y += static_cast<T>(rhs.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float2<T>& Float2<T>::operator+=(U rhs) noexcept {
        this->x += static_cast<T>(rhs);
        this->y += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float2<T>& Float2<T>::operator-=(const Float2<U>& rhs) noexcept {
        this->x -= static_cast<T>(rhs.x);
        this->y -= static_cast<T>(rhs.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float2<T>& Float2<T>::operator-=(U rhs) noexcept {
        this->x -= static_cast<T>(rhs);
        this->y -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float2<T>& Float2<T>::operator*=(const Float2<U>& rhs) noexcept {
        this->x *= static_cast<T>(rhs.x);
        this->y *= static_cast<T>(rhs.y);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Float2<T>& Float2<T>::operator*=(U rhs) noexcept {
        this->x *= static_cast<T>(rhs);
        this->y *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Float2<T>& Float2<T>::operator/=(const Float2<U>& rhs) noexcept {
        this->x /= static_cast<T>(rhs.x);
        this->y /= static_cast<T>(rhs.y);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Float2<T>& Float2<T>::operator/=(U rhs) noexcept {
        this->x /= static_cast<T>(rhs);
        this->y /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T>
    constexpr Float2<T> operator+(const Float2<T>& v) noexcept {
        return v;
    }

    template<typename T>
    constexpr Float2<T> operator-(const Float2<T>& v) noexcept {
        return {-v.x, -v.y};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    constexpr Float2<T> operator+(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y};
    }
    template<typename T>
    constexpr Float2<T> operator+(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y};
    }
    template<typename T>
    constexpr Float2<T> operator+(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs};
    }

    template<typename T>
    constexpr Float2<T> operator-(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y};
    }
    template<typename T>
    constexpr Float2<T> operator-(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y};
    }
    template<typename T>
    constexpr Float2<T> operator-(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs};
    }

    template<typename T>
    constexpr Float2<T> operator*(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y};
    }
    template<typename T>
    constexpr Float2<T> operator*(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y};
    }
    template<typename T>
    constexpr Float2<T> operator*(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs};
    }

    template<typename T>
    constexpr Float2<T> operator/(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y};
    }
    template<typename T>
    constexpr Float2<T> operator/(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y};
    }
    template<typename T>
    constexpr Float2<T> operator/(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs};
    }

    // -- Comparison Operators --

    template<typename T>
    constexpr Bool2 operator>(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x > rhs.x, lhs.y > rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator>(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x > rhs, lhs.y > rhs};
    }
    template<typename T>
    constexpr Bool2 operator>(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs > rhs.x, lhs > rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator<(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x < rhs.x, lhs.y < rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator<(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x < rhs, lhs.y < rhs};
    }
    template<typename T>
    constexpr Bool2 operator<(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs < rhs.x, lhs < rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator>=(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x >= rhs.x, lhs.y >= rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator>=(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x >= rhs, lhs.y >= rhs};
    }
    template<typename T>
    constexpr Bool2 operator>=(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs >= rhs.x, lhs >= rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator<=(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x <= rhs.x, lhs.y <= rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator<=(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x <= rhs, lhs.y <= rhs};
    }
    template<typename T>
    constexpr Bool2 operator<=(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs <= rhs.x, lhs <= rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator==(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator==(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs};
    }
    template<typename T>
    constexpr Bool2 operator==(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator!=(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator!=(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs};
    }
    template<typename T>
    constexpr Bool2 operator!=(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y};
    }

    namespace math {
        template<typename T>
        constexpr Float2<T> floor(const Float2<T>& v) {
            return Float2<T>(floor(v.x), floor(v.y));
        }

        template<typename T>
        constexpr Float2<T> ceil(const Float2<T>& v) {
            return Float2<T>(ceil(v.x), ceil(v.y));
        }

        template<typename T>
        constexpr T dot(const Float2<T>& a, const Float2<T>& b) noexcept {
            return a.x * b.x + a.y * b.y;
        }

        template<typename T>
        constexpr T innerProduct(const Float2<T>& a, const Float2<T>& b) noexcept {
            return dot(a, b);
        }

        template<typename T>
        constexpr T norm(const Float2<T>& v) noexcept {
            return sqrt(dot(v, v));
        }

        template<typename T>
        constexpr T length(const Float2<T>& v) {
            return norm(v);
        }

        template<typename T>
        constexpr Float2<T> normalize(const Float2<T>& v) {
            return v / norm(v);
        }

        template<typename T>
        constexpr T sum(const Float2<T>& v) noexcept {
            return v.x + v.y;
        }

        template<typename T>
        constexpr T prod(const Float2<T>& v) noexcept {
            return v.x * v.y;
        }

        template<typename T>
        constexpr T min(const Float2<T>& v) noexcept {
            return min(v.x, v.y);
        }

        template<typename T>
        constexpr Float2<T> min(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y)};
        }

        template<typename T>
        constexpr Float2<T> min(const Float2<T>& lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs)};
        }

        template<typename T>
        constexpr Float2<T> min(T lhs, const Float2<T>& rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y)};
        }

        template<typename T>
        constexpr T max(const Float2<T>& v) noexcept {
            return max(v.x, v.y);
        }

        template<typename T>
        constexpr Float2<T> max(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y)};
        }

        template<typename T>
        constexpr Float2<T> max(const Float2<T>& lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs)};
        }

        template<typename T>
        constexpr Float2<T> max(T lhs, const Float2<T>& rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y)};
        }

        template<uint ULP, typename T>
        constexpr Bool2 isEqual(const Float2<T>& a, const Float2<T>& b, T e) {
            return {isEqual<ULP>(a.x, b.x, e), isEqual<ULP>(a.y, b.y, e)};
        }

        template<uint ULP, typename T>
        constexpr Bool2 isEqual(const Float2<T>& a, T b, T e) {
            return {isEqual<ULP>(a.x, b, e), isEqual<ULP>(a.y, b, e)};
        }

        template<uint ULP, typename T>
        constexpr Bool2 isEqual(T a, const Float2<T>& b, T e) {
            return {isEqual<ULP>(a, b.x, e), isEqual<ULP>(a, b.y, e)};
        }
    }
}
