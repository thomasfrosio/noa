/// \file noa/common/types/Complex.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// \brief A complex number that can be used on the device.

#pragma once

#include <complex>
#include <cfloat>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/string/Format.h"
#include "noa/common/types/Bool2.h"

// Array-oriented access
// =====================
//
// Since C++11, it is required for std::complex to have an array-oriented access. It is also defined behavior
// to reinterpret_cast a struct { float|double x, y; } to a float|double*. This does not violate the strict aliasing
// rule. Also, cuComplex, cuDoubleComplex and noa::Complex<> have the same layout.
// As such, std::complex<> or noa::Complex<> can simply be reinterpret_cast<> to cuComplex or cuDoubleComplex whenever
// necessary. Unittests will make sure there's no weird padding and alignment is as expected so that array-oriented
// access is OK.
//
// See: https://en.cppreference.com/w/cpp/numeric/complex
// See: https://en.cppreference.com/w/cpp/language/reinterpret_cast

namespace noa {
    template<typename>
    class Float2;

    template<typename>
    class Int2;

    template<typename T>
    class alignas(sizeof(T) * 2) Complex {
    private:
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
        T m_re{}, m_im{};

    public:
        typedef T value_type;

    public: // Component accesses
        NOA_HD static constexpr size_t elements() noexcept { return 2; }
        NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        constexpr Complex() noexcept = default;
        template<typename U> NOA_HD constexpr Complex(U v) noexcept; // not explicit on purpose, i.e. mimic float
        template<typename U> NOA_HD constexpr explicit Complex(U* ptr);
        template<typename R, typename I> NOA_HD constexpr Complex(R re, I im) noexcept;
        template<typename U> NOA_HD constexpr explicit Complex(const Complex<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Complex(const Float2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Complex(const Int2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Complex(const std::complex<U>& v) noexcept;

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Complex<T>& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator=(U* ptr);
        template<typename U> NOA_HD constexpr Complex<T>& operator=(const Complex<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator=(const Float2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator=(const Int2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator=(const std::complex<U>& v) noexcept;

        template<typename U> NOA_HD constexpr Complex<T>& operator+=(const Complex<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator-=(const Complex<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator*=(const Complex<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator/=(const Complex<U>& rhs);

        template<typename U> NOA_HD constexpr Complex<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator/=(U rhs) noexcept;

        NOA_HD T real() const volatile { return m_re; }
        NOA_HD T imag() const volatile { return m_im; }
        NOA_HD constexpr T real() const { return m_re; }
        NOA_HD constexpr T imag() const { return m_im; }

        NOA_HD void real(T re) volatile { m_re = re; }
        NOA_HD void imag(T im) volatile { m_im = im; }
        NOA_HD constexpr void real(T re) { m_re = re; }
        NOA_HD constexpr void imag(T im) { m_im = im; }
    };

    // -- Unary operators --

    template<typename T> NOA_FHD constexpr Complex<T> operator+(const Complex<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator-(const Complex<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_FHD constexpr Complex<T> operator+(const Complex<T>& lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator+(T lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator+(const Complex<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Complex<T> operator-(const Complex<T>& lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator-(T lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator-(const Complex<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Complex<T> operator*(const Complex<T>& lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator*(T lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator*(const Complex<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Complex<T> operator/(const Complex<T>& lhs, const Complex<T>& rhs);
    template<typename T> NOA_HD constexpr Complex<T> operator/(T lhs, const Complex<T>& rhs);
    template<typename T> NOA_HD constexpr Complex<T> operator/(const Complex<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_FHD constexpr bool operator==(const Complex<T>& lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr bool operator==(const Complex<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr bool operator==(T lhs, const Complex<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr bool operator!=(const Complex<T>& lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr bool operator!=(const Complex<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr bool operator!=(T lhs, const Complex<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr bool operator==(const Complex<T>& lhs, const std::complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr bool operator==(const std::complex<T>& lhs, const Complex<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr bool operator!=(const Complex<T>& lhs, const std::complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr bool operator!=(const std::complex<T>& lhs, const Complex<T>& rhs) noexcept;

    namespace math {
        /// Returns the real part of the complex number \a x.
        template<typename T> NOA_FHD constexpr T real(Complex<T> x) noexcept { return x.real(); }

        /// Returns the imaginary part of the complex number \a x.
        template<typename T> NOA_FHD constexpr T imag(Complex<T> x) noexcept { return x.imag(); }

        /// Returns the phase angle (in radians) of the complex number \a z.
        template<typename T> NOA_FHD T arg(const Complex<T>& x);

        /// Returns the magnitude of the complex number \a x.
        template<typename T> NOA_FHD T abs(const Complex<T>& x);
        template<typename T> NOA_FHD T length(const Complex<T>& x) { return abs(x); }

        /// Returns the length-normalized of the complex number \a x to 1, reducing it to its phase.
        template<typename T> NOA_FHD Complex<T> normalize(const Complex<T>& x);

        /// Returns the squared magnitude of the complex number \a x.
        template<typename T> NOA_IHD T norm(const Complex<T>& x);
        template<typename T> NOA_IHD T lengthSq(const Complex<T>& x) { return norm(x); }

        /// Returns the complex conjugate of \a x.
        template<typename T> NOA_FHD constexpr Complex<T> conj(const Complex<T>& x) noexcept;

        /// Returns a complex number with magnitude \a length (should be positive) and phase angle \a theta.
        template<typename T> NOA_FHD Complex<T> polar(T length, T theta);

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(const Complex<T>& a, const Complex<T>& b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(const Complex<T>& a, T b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(T a, const Complex<T>& b, T e = NOA_EPSILON_);

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace traits {
        template<> struct proclaim_is_complex<Complex<float>> : std::true_type {};
        template<> struct proclaim_is_complex<Complex<double>> : std::true_type {};
    }

    using cfloat_t = Complex<float>;
    using cdouble_t = Complex<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 2> toArray(const Complex<T>& v) noexcept {
        return {v.real(), v.imag()};
    }

    template<> NOA_IH std::string string::typeName<cdouble_t>() { return "cdouble"; }
    template<> NOA_IH std::string string::typeName<cfloat_t>() { return "cfloat"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Complex<T>& z) {
        os << string::format("({:.3f},{:.3f})", z.real(), z.imag());
        return os;
    }
}

// Definitions:
namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr T& Complex<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->elements());
        if (i == 1)
            return this->m_im;
        else
            return this->m_re;
    }

    template<typename T>
    constexpr const T& Complex<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->elements());
        if (i == 1)
            return this->m_im;
        else
            return this->m_re;
    }

    // -- (Conversion) Constructors --

    template<typename T>
    template<typename U>
    constexpr Complex<T>::Complex(U v) noexcept
            : m_re(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    constexpr Complex<T>::Complex(U* ptr)
            : m_re(static_cast<T>(ptr[0])), m_im(static_cast<T>(ptr[1])) {}

    template<typename T>
    template<typename R, typename I>
    constexpr Complex<T>::Complex(R re, I im) noexcept
            : m_re(static_cast<T>(re)), m_im(static_cast<T>(im)) {}

    template<typename T>
    template<typename U>
    constexpr Complex<T>::Complex(const Complex<U>& v) noexcept
            : m_re(static_cast<T>(v.real())), m_im(static_cast<T>(v.imag())) {}

    template<typename T>
    template<typename U>
    constexpr Complex<T>::Complex(const Float2<U>& v) noexcept
            : m_re(static_cast<T>(v.x)), m_im(static_cast<T>(v.y)) {}

    template<typename T>
    template<typename U>
    constexpr Complex<T>::Complex(const Int2<U>& v) noexcept
            : m_re(static_cast<T>(v.x)), m_im(static_cast<T>(v.y)) {}

    template<typename T>
    template<typename U>
    constexpr Complex<T>::Complex(const std::complex<U>& v) noexcept
            : m_re(static_cast<T>(reinterpret_cast<const U(&)[2]>(v)[0])),
              m_im(static_cast<T>(reinterpret_cast<const U(&)[2]>(v)[1])) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator=(U v) noexcept {
        this->m_re = static_cast<T>(v);
        this->m_im = 0;
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator=(U* ptr) {
        this->m_re = static_cast<T>(ptr[0]);
        this->m_im = static_cast<T>(ptr[1]);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator=(const Complex<U>& v) noexcept {
        this->m_re = static_cast<T>(v.real());
        this->m_im = static_cast<T>(v.imag());
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator=(const Float2<U>& v) noexcept {
        this->m_re = static_cast<T>(v.x);
        this->m_im = static_cast<T>(v.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator=(const Int2<U>& v) noexcept {
        this->m_re = static_cast<T>(v.x);
        this->m_im = static_cast<T>(v.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator=(const std::complex<U>& v) noexcept {
        this->m_re = static_cast<T>(reinterpret_cast<const U(&)[2]>(v)[0]);
        this->m_im = static_cast<T>(reinterpret_cast<const U(&)[2]>(v)[1]);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator+=(const Complex<U>& rhs) noexcept {
        this->m_re += static_cast<T>(rhs.real());
        this->m_im += static_cast<T>(rhs.imag());
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator-=(const Complex<U>& rhs) noexcept {
        this->m_re -= static_cast<T>(rhs.real());
        this->m_im -= static_cast<T>(rhs.imag());
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
        this->m_re += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator-=(U rhs) noexcept {
        this->m_re -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator*=(U rhs) noexcept {
        this->m_re *= static_cast<T>(rhs);
        this->m_im *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Complex<T>& Complex<T>::operator/=(U rhs) noexcept {
        this->m_re /= static_cast<T>(rhs);
        this->m_im /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T>
    constexpr Complex<T> operator+(const Complex<T>& v) noexcept {
        return v;
    }

    template<typename T>
    constexpr Complex<T> operator-(const Complex<T>& v) noexcept {
        return {-v.real(), -v.imag()};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    constexpr Complex<T> operator+(const Complex<T>& lhs, const Complex<T>& rhs) noexcept {
        return {lhs.real() + rhs.real(), lhs.imag() + rhs.imag()};
    }
    template<typename T>
    constexpr Complex<T> operator+(T lhs, const Complex<T>& rhs) noexcept {
        return {lhs + rhs.real(), rhs.imag()};
    }
    template<typename T>
    constexpr Complex<T> operator+(const Complex<T>& lhs, T rhs) noexcept {
        return {lhs.real() + rhs, lhs.imag()};
    }

    template<typename T>
    constexpr Complex<T> operator-(const Complex<T>& lhs, const Complex<T>& rhs) noexcept {
        return {lhs.real() - rhs.real(), lhs.imag() - rhs.imag()};
    }
    template<typename T>
    constexpr Complex<T> operator-(T lhs, const Complex<T>& rhs) noexcept {
        return {lhs - rhs.real(), -rhs.imag()};
    }
    template<typename T>
    constexpr Complex<T> operator-(const Complex<T>& lhs, T rhs) noexcept {
        return {lhs.real() - rhs, lhs.imag()};
    }

    template<typename T>
    constexpr Complex<T> operator*(const Complex<T>& lhs, const Complex<T>& rhs) noexcept {
        return {lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
                lhs.real() * rhs.imag() + lhs.imag() * rhs.real()};
    }
    template<typename T>
    constexpr Complex<T> operator*(T lhs, const Complex<T>& rhs) noexcept {
        return {lhs * rhs.real(), lhs * rhs.imag()};
    }
    template<typename T>
    constexpr Complex<T> operator*(const Complex<T>& lhs, T rhs) noexcept {
        return {lhs.real() * rhs, lhs.imag() * rhs};
    }

    // Adapted from cuComplex.h
    // "This implementation guards against intermediate underflow and overflow
    // by scaling. Such guarded implementations are usually the default for
    // complex library implementations, with some also offering an unguarded,
    // faster version."
    template<typename T>
    constexpr Complex<T> operator/(const Complex<T>& lhs, const Complex<T>& rhs) {
        T s = abs(rhs.real()) + abs(rhs.imag());
        T oos = T(1.0) / s;

        T ars = lhs.real() * oos;
        T ais = lhs.imag() * oos;
        T brs = rhs.real() * oos;
        T bis = rhs.imag() * oos;

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
        return {lhs.real() / rhs, lhs.imag() / rhs};
    }

    /* --- Equality Operators --- */

    template<typename T>
    constexpr bool operator==(const Complex<T>& lhs, const Complex<T>& rhs) noexcept {
        return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
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
        return lhs.real() == reinterpret_cast<const T(&)[2]>(rhs)[0] &&
               lhs.imag() == reinterpret_cast<const T(&)[2]>(rhs)[1];
    }

    template<typename T>
    constexpr bool operator==(const std::complex<T>& lhs, const Complex<T>& rhs) noexcept {
        return reinterpret_cast<const T(&)[2]>(lhs)[0] == rhs.real() &&
               reinterpret_cast<const T(&)[2]>(lhs)[1] == rhs.imag();
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
            return atan2(x.imag(), x.real());
        }

        template<typename T>
        T abs(const Complex<T>& x) {
            return hypot(x.real(), x.imag());
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
            if (abs(x.real()) < sqrt(FLT_MIN) && abs(x.imag()) < sqrt(FLT_MIN)) {
                float a = x.real() * 4.0f;
                float b = x.imag() * 4.0f;
                return (a * a + b * b) / 16.0f;
            }
            return x.real() * x.real() + x.imag() * x.imag();
        }
        template<>
        NOA_IHD double norm<double>(const Complex<double>& x) {
            if (abs(x.real()) < sqrt(DBL_MIN) && abs(x.imag()) < sqrt(DBL_MIN)) {
                double a = x.real() * 4.0;
                double b = x.imag() * 4.0;
                return (a * a + b * b) / 16.0;
            }
            return x.real() * x.real() + x.imag() * x.imag();
        }

        template<typename T>
        constexpr Complex<T> conj(const Complex<T>& x) noexcept {
            return {x.real(), -x.imag()};
        }

        template<typename T>
        Complex<T> polar(T length, T theta) {
            return {length * cos(theta), length * sin(theta)};
        }

        template<uint ULP, typename T>
        constexpr bool isEqual(const Complex<T>& a, const Complex<T>& b, T e) {
            return isEqual<ULP>(a.real(), b.real(), e) && isEqual<ULP>(a.imag(), b.imag(), e);
        }

        template<uint ULP, typename T>
        constexpr bool isEqual(const Complex<T>& a, T b, T e) {
            return isEqual<ULP>(a.real(), b, e) && isEqual<ULP>(a.imag(), b, e);
        }

        template<uint ULP, typename T>
        constexpr bool isEqual(T a, const Complex<T>& b, T e) {
            return isEqual<ULP>(a, b.real(), e) && isEqual<ULP>(a, b.imag(), e);
        }
    }
}
