#ifndef NOA_INCLUDE_INT2_
#error "This should not be directly included"
#endif

#include "noa/common/Assert.h"

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr T& Int2<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->COUNT);
        if (i == 1)
            return this->y;
        else
            return this->x;
    }

    template<typename T>
    constexpr const T& Int2<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->COUNT);
        if (i == 1)
            return this->y;
        else
            return this->x;
    }

    // -- (Conversion) Constructors --

    template<typename T>
    template<typename X, typename Y>
    constexpr Int2<T>::Int2(X xi, Y yi) noexcept
            : x(static_cast<T>(xi)),
              y(static_cast<T>(yi)) {}

    template<typename T>
    template<typename U>
    constexpr Int2<T>::Int2(U v) noexcept
            : x(static_cast<T>(v)),
              y(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    constexpr Int2<T>::Int2(const Int2<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)) {}

    template<typename T>
    template<typename U>
    constexpr Int2<T>::Int2(const Float2<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)) {}

    template<typename T>
    template<typename U>
    constexpr Int2<T>::Int2(U* ptr)
            : x(static_cast<T>(ptr[0])),
              y(static_cast<T>(ptr[1])) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator=(U v) noexcept {
        this->x = static_cast<T>(v);
        this->y = static_cast<T>(v);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator=(U* ptr) noexcept {
        this->x = static_cast<T>(ptr[0]);
        this->y = static_cast<T>(ptr[1]);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator=(const Int2<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator=(const Float2<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator+=(const Int2<U>& rhs) noexcept {
        this->x += static_cast<T>(rhs.x);
        this->y += static_cast<T>(rhs.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator+=(U rhs) noexcept {
        this->x += static_cast<T>(rhs);
        this->y += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator-=(const Int2<U>& rhs) noexcept {
        this->x -= static_cast<T>(rhs.x);
        this->y -= static_cast<T>(rhs.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator-=(U rhs) noexcept {
        this->x -= static_cast<T>(rhs);
        this->y -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator*=(const Int2<U>& rhs) noexcept {
        this->x *= static_cast<T>(rhs.x);
        this->y *= static_cast<T>(rhs.y);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator*=(U rhs) noexcept {
        this->x *= static_cast<T>(rhs);
        this->y *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator/=(const Int2<U>& rhs) noexcept {
        this->x /= static_cast<T>(rhs.x);
        this->y /= static_cast<T>(rhs.y);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator/=(U rhs) noexcept {
        this->x /= static_cast<T>(rhs);
        this->y /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T> constexpr Int2<T> operator+(const Int2<T>& v) noexcept {
        return v;
    }

    template<typename T> constexpr Int2<T> operator-(const Int2<T>& v) noexcept {
        return {-v.x, -v.y};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    constexpr Int2<T> operator+(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator+(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator+(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs};
    }

    template<typename T>
    constexpr Int2<T> operator-(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator-(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator-(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs};
    }

    template<typename T>
    constexpr Int2<T> operator*(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator*(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator*(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs};
    }

    template<typename T>
    constexpr Int2<T> operator/(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator/(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator/(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs};
    }

    // -- Comparison Operators --

    template<typename T>
    constexpr Bool2 operator>(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x > rhs.x, lhs.y > rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator>(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x > rhs, lhs.y > rhs};
    }
    template<typename T>
    constexpr Bool2 operator>(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs > rhs.x, lhs > rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator<(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x < rhs.x, lhs.y < rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator<(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x < rhs, lhs.y < rhs};
    }
    template<typename T>
    constexpr Bool2 operator<(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs < rhs.x, lhs < rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator>=(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x >= rhs.x, lhs.y >= rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator>=(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x >= rhs, lhs.y >= rhs};
    }
    template<typename T>
    constexpr Bool2 operator>=(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs >= rhs.x, lhs >= rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator<=(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x <= rhs.x, lhs.y <= rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator<=(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x <= rhs, lhs.y <= rhs};
    }
    template<typename T>
    constexpr Bool2 operator<=(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs <= rhs.x, lhs <= rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator==(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator==(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs};
    }
    template<typename T>
    constexpr Bool2 operator==(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator!=(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator!=(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs};
    }
    template<typename T>
    constexpr Bool2 operator!=(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y};
    }

    template<typename T>
    constexpr T elements(const Int2<T>& v) noexcept {
        return v.x * v.y;
    }

    template<typename T>
    constexpr T elementsFFT(const Int2<T>& v) noexcept {
        return (v.x / 2 + 1) * v.y;
    }

    template<typename T>
    constexpr Int2<T> shapeFFT(const Int2<T>& v) noexcept {
        return {v.x / 2 + 1, v.y};
    }

    namespace math {
        template<typename T>
        constexpr T sum(const Int2<T>& v) noexcept {
            return v.x + v.y;
        }

        template<typename T>
        constexpr T prod(const Int2<T>& v) noexcept {
            return v.x * v.y;
        }

        template<typename T>
        constexpr T min(const Int2<T>& v) noexcept {
            return min(v.x, v.y);
        }

        template<typename T>
        constexpr Int2<T> min(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y)};
        }

        template<typename T>
        constexpr Int2<T> min(const Int2<T>& lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs)};
        }

        template<typename T>
        constexpr Int2<T> min(T lhs, const Int2<T>& rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y)};
        }

        template<typename T>
        constexpr T max(const Int2<T>& v) noexcept {
            return max(v.x, v.y);
        }

        template<typename T>
        constexpr Int2<T> max(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y)};
        }

        template<typename T>
        constexpr Int2<T> max(const Int2<T>& lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs)};
        }

        template<typename T>
        constexpr Int2<T> max(T lhs, const Int2<T>& rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y)};
        }
    }
}

namespace noa::traits {
    template<typename> struct p_is_int2 : std::false_type {};
    template<typename T> struct p_is_int2<noa::Int2<T>> : std::true_type {};
    template<typename T> using is_int2 = std::bool_constant<p_is_int2<noa::traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_int2_v = is_int2<T>::value;

    template<typename> struct p_is_uint2 : std::false_type {};
    template<typename T> struct p_is_uint2<noa::Int2<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
    template<typename T> using is_uint2 = std::bool_constant<p_is_uint2<noa::traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_uint2_v = is_uint2<T>::value;

    template<typename T> struct proclaim_is_intX<noa::Int2<T>> : std::true_type {};
    template<typename T> struct proclaim_is_uintX<noa::Int2<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
}
