#ifndef NOA_INCLUDE_COMPLEX_
#error "This should not be directly included"
#endif

// Definitions:
namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr T& Complex<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->COUNT);
        if (i == 1)
            return this->imag;
        else
            return this->real;
    }

    template<typename T>
    constexpr const T& Complex<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->COUNT);
        if (i == 1)
            return this->imag;
        else
            return this->real;
    }

    // -- (Conversion) Constructors --

    template<typename T>
    template<typename U>
    constexpr Complex<T>::Complex(U v) noexcept
            : real(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    constexpr Complex<T>::Complex(U* ptr)
            : real(static_cast<T>(ptr[0])), imag(static_cast<T>(ptr[1])) {}

    template<typename T>
    template<typename R, typename I>
    constexpr Complex<T>::Complex(R re, I im) noexcept
            : real(static_cast<T>(re)), imag(static_cast<T>(im)) {}

    template<typename T>
    template<typename U>
    constexpr Complex<T>::Complex(const Complex<U>& v) noexcept
            : real(static_cast<T>(v.real)), imag(static_cast<T>(v.imag)) {}

    template<typename T>
    template<typename U>
    constexpr Complex<T>::Complex(const Float2<U>& v) noexcept
            : real(static_cast<T>(v.x)), imag(static_cast<T>(v.y)) {}

    template<typename T>
    template<typename U>
    constexpr Complex<T>::Complex(const Int2<U>& v) noexcept
            : real(static_cast<T>(v.x)), imag(static_cast<T>(v.y)) {}

    template<typename T>
    template<typename U>
    constexpr Complex<T>::Complex(const std::complex<U>& v) noexcept
            : real(static_cast<T>(reinterpret_cast<const U(&)[2]>(v)[0])),
              imag(static_cast<T>(reinterpret_cast<const U(&)[2]>(v)[1])) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator=(U v) noexcept {
        this->real = static_cast<T>(v);
        this->imag = 0;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator=(U* ptr) {
        this->real = static_cast<T>(ptr[0]);
        this->imag = static_cast<T>(ptr[1]);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator=(const Complex<U>& v) noexcept {
        this->real = static_cast<T>(v.real);
        this->imag = static_cast<T>(v.imag);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator=(const Float2<U>& v) noexcept {
        this->real = static_cast<T>(v.x);
        this->imag = static_cast<T>(v.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator=(const Int2<U>& v) noexcept {
        this->real = static_cast<T>(v.x);
        this->imag = static_cast<T>(v.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator=(const std::complex<U>& v) noexcept {
        this->real = static_cast<T>(reinterpret_cast<const U(&)[2]>(v)[0]);
        this->imag = static_cast<T>(reinterpret_cast<const U(&)[2]>(v)[1]);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator+=(const Complex<U>& rhs) noexcept {
        this->real += static_cast<T>(rhs.real);
        this->imag += static_cast<T>(rhs.imag);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator-=(const Complex<U>& rhs) noexcept {
        this->real -= static_cast<T>(rhs.real);
        this->imag -= static_cast<T>(rhs.imag);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator*=(const Complex<U>& rhs) noexcept {
        *this = *this * Complex<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator/=(const Complex<U>& rhs) {
        *this = *this / Complex<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator+=(U rhs) noexcept {
        this->real += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator-=(U rhs) noexcept {
        this->real -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator*=(U rhs) noexcept {
        this->real *= static_cast<T>(rhs);
        this->imag *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator/=(U rhs) noexcept {
        this->real /= static_cast<T>(rhs);
        this->imag /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T>
    constexpr Complex<T> operator+(const Complex<T>& v) noexcept {
        return v;
    }

    template<typename T>
    constexpr Complex<T> operator-(const Complex<T>& v) noexcept {
        return {-v.real, -v.imag};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    constexpr Complex<T> operator+(const Complex<T>& lhs, const Complex<T>& rhs) noexcept {
        return {lhs.real + rhs.real, lhs.imag + rhs.imag};
    }
    template<typename T>
    constexpr Complex<T> operator+(T lhs, const Complex<T>& rhs) noexcept {
        return {lhs + rhs.real, rhs.imag};
    }
    template<typename T>
    constexpr Complex<T> operator+(const Complex<T>& lhs, T rhs) noexcept {
        return {lhs.real + rhs, lhs.imag};
    }

    template<typename T>
    constexpr Complex<T> operator-(const Complex<T>& lhs, const Complex<T>& rhs) noexcept {
        return {lhs.real - rhs.real, lhs.imag - rhs.imag};
    }
    template<typename T>
    constexpr Complex<T> operator-(T lhs, const Complex<T>& rhs) noexcept {
        return {lhs - rhs.real, -rhs.imag};
    }
    template<typename T>
    constexpr Complex<T> operator-(const Complex<T>& lhs, T rhs) noexcept {
        return {lhs.real - rhs, lhs.imag};
    }

    template<typename T>
    constexpr Complex<T> operator*(const Complex<T>& lhs, const Complex<T>& rhs) noexcept {
        return {lhs.real * rhs.real - lhs.imag * rhs.imag,
                lhs.real * rhs.imag + lhs.imag * rhs.real};
    }
    template<typename T>
    constexpr Complex<T> operator*(T lhs, const Complex<T>& rhs) noexcept {
        return {lhs * rhs.real, lhs * rhs.imag};
    }
    template<typename T>
    constexpr Complex<T> operator*(const Complex<T>& lhs, T rhs) noexcept {
        return {lhs.real * rhs, lhs.imag * rhs};
    }

    // Adapted from cuComplex.h
    // "This implementation guards against intermediate underflow and overflow
    // by scaling. Such guarded implementations are usually the default for
    // complex library implementations, with some also offering an unguarded,
    // faster version."
    template<typename T>
    constexpr Complex<T> operator/(const Complex<T>& lhs, const Complex<T>& rhs) {
        T s = abs(rhs.real) + abs(rhs.imag);
        T oos = T(1.0) / s;

        T ars = lhs.real * oos;
        T ais = lhs.imag * oos;
        T brs = rhs.real * oos;
        T bis = rhs.imag * oos;

        s = (brs * brs) + (bis * bis);
        oos = T(1.0) / s;

        return {((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos};
    }
    template<typename T>
    constexpr Complex<T> operator/(T lhs, const Complex<T>& rhs) {
        return Complex<T>(lhs) / rhs;
    }
    template<typename T>
    constexpr Complex<T> operator/(const Complex<T>& lhs, T rhs) noexcept {
        return {lhs.real / rhs, lhs.imag / rhs};
    }

    /* --- Equality Operators --- */

    template<typename T>
    constexpr bool operator==(const Complex<T>& lhs, const Complex<T>& rhs) noexcept {
        return lhs.real == rhs.real && lhs.imag == rhs.imag;
    }

    template<typename T>
    constexpr bool operator==(T lhs, const Complex<T>& rhs) noexcept {
        return Complex<T>(lhs) == rhs;
    }

    template<typename T>
    constexpr bool operator==(const Complex<T>& lhs, T rhs) noexcept {
        return lhs == Complex<T>(rhs);
    }

    template<typename T>
    constexpr bool operator!=(const Complex<T>& lhs, const Complex<T>& rhs) noexcept {
        return !(lhs == rhs);
    }

    template<typename T>
    constexpr bool operator!=(T lhs, const Complex<T>& rhs) noexcept {
        return Complex<T>(lhs) != rhs;
    }

    template<typename T>
    constexpr bool operator!=(const Complex<T>& lhs, T rhs) noexcept {
        return lhs != Complex<T>(rhs);
    }

    template<typename T>
    constexpr bool operator==(const Complex<T>& lhs, const std::complex<T>& rhs) noexcept {
        return lhs.real == reinterpret_cast<const T(&)[2]>(rhs)[0] &&
               lhs.imag == reinterpret_cast<const T(&)[2]>(rhs)[1];
    }

    template<typename T>
    constexpr bool operator==(const std::complex<T>& lhs, const Complex<T>& rhs) noexcept {
        return reinterpret_cast<const T(&)[2]>(lhs)[0] == rhs.real &&
               reinterpret_cast<const T(&)[2]>(lhs)[1] == rhs.imag;
    }

    template<typename T>
    constexpr bool operator!=(const Complex<T>& lhs, const std::complex<T>& rhs) noexcept {
        return !(lhs == rhs);
    }

    template<typename T>
    constexpr bool operator!=(const std::complex<T>& lhs, const Complex<T>& rhs) noexcept {
        return !(lhs == rhs);
    }

    namespace math {
        template<typename T>
        T arg(const Complex<T>& x) {
            return atan2(x.imag, x.real);
        }

        template<typename T>
        T abs(const Complex<T>& x) {
            return hypot(x.real, x.imag);
        }

        template<typename T>
        Complex<T> normalize(const Complex<T>& x) {
            T magnitude = abs(x);
            if (magnitude > T{0}) // hum ...
                magnitude = 1 / magnitude;
            return x * magnitude;
        }

        template<>
        NOA_IHD float norm<float>(const Complex<float>& x) {
            if (abs(x.real) < sqrt(FLT_MIN) && abs(x.imag) < sqrt(FLT_MIN)) {
                float a = x.real * 4.0f;
                float b = x.imag * 4.0f;
                return (a * a + b * b) / 16.0f;
            }
            return x.real * x.real + x.imag * x.imag;
        }
        template<>
        NOA_IHD double norm<double>(const Complex<double>& x) {
            if (abs(x.real) < sqrt(DBL_MIN) && abs(x.imag) < sqrt(DBL_MIN)) {
                double a = x.real * 4.0;
                double b = x.imag * 4.0;
                return (a * a + b * b) / 16.0;
            }
            return x.real * x.real + x.imag * x.imag;
        }

        template<typename T>
        constexpr Complex<T> conj(const Complex<T>& x) noexcept {
            return {x.real, -x.imag};
        }

        template<typename T>
        Complex<T> polar(T length, T theta) {
            return {length * cos(theta), length * sin(theta)};
        }

        template<uint ULP, typename T>
        constexpr bool isEqual(const Complex<T>& a, const Complex<T>& b, T e) {
            return isEqual<ULP>(a.real, b.real, e) && isEqual<ULP>(a.imag, b.imag, e);
        }

        template<uint ULP, typename T>
        constexpr bool isEqual(const Complex<T>& a, T b, T e) {
            return isEqual<ULP>(a.real, b, e) && isEqual<ULP>(a.imag, b, e);
        }

        template<uint ULP, typename T>
        constexpr bool isEqual(T a, const Complex<T>& b, T e) {
            return isEqual<ULP>(a, b.real, e) && isEqual<ULP>(a, b.imag, e);
        }
    }
}
