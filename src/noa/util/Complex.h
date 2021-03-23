#pragma once

#include <complex>
#include <cfloat>
#include <type_traits>

#include "noa/Definitions.h"
#include "noa/Math.h"
#include "noa/util/traits/BaseTypes.h"
#include "noa/util/string/Format.h"

/*
 * Array-oriented access
 * =================
 *
 * Since in C++11, it is required for std::complex to have an array-oriented access. It is also defined behavior
 * to reinterpret_cast a struct { float|double x, y; } to a float|double*. This does not violate the strict aliasing
 * rule. Also, cuComplex, cuDoubleComplex and Noa::Complex<> have the same layout.
 * As such, std::complex<> or Noa::Complex<> can simply be reinterpret_cast<> to cuComplex or cuDoubleComplex whenever
 * necessary. Unittests will make sure there's no weird padding and alignment is as expected so that array-oriented
 * access it OK.
 *
 * See: https://en.cppreference.com/w/cpp/numeric/complex
 * See: https://en.cppreference.com/w/cpp/language/reinterpret_cast
 */

namespace Noa {
    template<typename FP>
    struct alignas(sizeof(FP) * 2) Complex {
    private:
        std::enable_if_t<std::is_same_v<FP, float> || std::is_same_v<FP, double>, FP> m_re{}, m_im{};
    public:
        typedef FP value_type;

        // Base constructors.
        NOA_HD constexpr Complex() = default;
        NOA_HD constexpr Complex(const Complex<FP>& c) = default;
        NOA_HD constexpr explicit Complex(FP re) : m_re(re), m_im(0) {};
        NOA_HD constexpr Complex(FP re, FP im) : m_re(re), m_im(im) {};

        // Conversion constructors.
        NOA_HD constexpr explicit Complex(const std::complex<FP>& x);
        template<class U> NOA_HD constexpr explicit Complex(const std::complex<U>& x);
        template<class U> NOA_HD constexpr explicit Complex(const Complex<U>& x);

        // Operator assignments.
        NOA_HD constexpr Complex<FP>& operator=(const Complex<FP>& c) = default;
        NOA_HD constexpr Complex<FP>& operator=(FP x);

        NOA_HD constexpr Complex<FP>& operator+=(const Complex<FP>& x);
        NOA_HD constexpr Complex<FP>& operator-=(const Complex<FP>& x);
        NOA_HD constexpr Complex<FP>& operator*=(const Complex<FP>& x);
        NOA_HD constexpr Complex<FP>& operator/=(const Complex<FP>& x);

        NOA_HD constexpr Complex<FP>& operator+=(FP x);
        NOA_HD constexpr Complex<FP>& operator-=(FP x);
        NOA_HD constexpr Complex<FP>& operator*=(FP x);
        NOA_HD constexpr Complex<FP>& operator/=(FP x);

        NOA_IHD FP real() const volatile { return m_re; }
        NOA_IHD FP imag() const volatile { return m_im; }
        NOA_IHD constexpr FP real() const { return m_re; }
        NOA_IHD constexpr FP imag() const { return m_im; }

        NOA_IHD void real(FP re) volatile { m_re = re; }
        NOA_IHD void imag(FP im) volatile { m_im = im; }
        NOA_IHD constexpr void real(FP re) { m_re = re; }
        NOA_IHD constexpr void imag(FP im) { m_im = im; }
    };

    /* --- Binary Arithmetic Operators --- */

    // Add
    template<typename FP> NOA_HD constexpr Complex<FP> operator+(const Complex<FP>& x, const Complex<FP>& y);
    template<typename FP> NOA_HD constexpr Complex<FP> operator+(FP x, const Complex<FP>& y);
    template<typename FP> NOA_HD constexpr Complex<FP> operator+(const Complex<FP>& x, FP y);

    // Subtract
    template<typename FP> NOA_HD constexpr Complex<FP> operator-(const Complex<FP>& x, const Complex<FP>& y);
    template<typename FP> NOA_HD constexpr Complex<FP> operator-(FP x, const Complex<FP>& y);
    template<typename FP> NOA_HD constexpr Complex<FP> operator-(const Complex<FP>& x, FP y);

    // Multiply
    template<typename FP> NOA_HD constexpr Complex<FP> operator*(const Complex<FP>& x, const Complex<FP>& y);
    template<typename FP> NOA_HD constexpr Complex<FP> operator*(FP x, const Complex<FP>& y);
    template<typename FP> NOA_HD constexpr Complex<FP> operator*(const Complex<FP>& x, FP y);

    // Divide
    template<typename FP> NOA_HD Complex<FP> operator/(const Complex<FP>& x, const Complex<FP>& y);
    template<typename FP> NOA_HD Complex<FP> operator/(FP x, const Complex<FP>& y);
    template<typename FP> NOA_HD Complex<FP> operator/(const Complex<FP>& x, FP y);

    /* --- Equality Operators - Checking for floating-point equality is a bad idea... --- */

    template<typename FP> NOA_HD constexpr bool operator==(const Complex<FP>& x, const Complex<FP>& y);
    template<typename FP> NOA_HD constexpr bool operator==(FP x, const Complex<FP>& y);
    template<typename FP> NOA_HD constexpr bool operator==(const Complex<FP>& x, FP y);

    template<typename FP> NOA_HD constexpr bool operator!=(const Complex<FP>& x, const Complex<FP>& y);
    template<typename FP> NOA_HD constexpr bool operator!=(FP x, const Complex<FP>& y);
    template<typename FP> NOA_HD constexpr bool operator!=(const Complex<FP>& x, FP y);

    template<typename FP> NOA_HD constexpr bool operator==(const Complex<FP>& x, const std::complex<FP>& y);
    template<typename FP> NOA_HD constexpr bool operator==(const std::complex<FP>& x, const Complex<FP>& y);

    template<typename FP> NOA_HD constexpr bool operator!=(const Complex<FP>& x, const std::complex<FP>& y);
    template<typename FP> NOA_HD constexpr bool operator!=(const std::complex<FP>& x, const Complex<FP>& y);

    /* --- Unary Arithmetic Operators --- */

    template<typename FP> NOA_HD constexpr Complex<FP> operator+(const Complex<FP>& x);
    template<typename FP> NOA_HD constexpr Complex<FP> operator-(const Complex<FP>& x);

    template<class T>
    NOA_IH std::string toString(const Complex<T>& z) { return String::format("({},{})", z.real(), z.imag()); }

    template<> NOA_IH const char* String::typeName<Complex<double>>() { return "complex64"; }
    template<> NOA_IH const char* String::typeName<Complex<float>>() { return "complex128"; }

    using cfloat_t = Complex<float>;
    using cdouble_t = Complex<double>;

    namespace Traits {
        template<> struct proclaim_is_complex<cfloat_t> : std::true_type {};
        template<> struct proclaim_is_complex<cdouble_t> : std::true_type {};
    }

    namespace Math {
        /// Returns the real part of the complex number @a x.
        template<typename T> NOA_FHD constexpr T real(Complex<T> x) { return x.real(); }

        /// Returns the imaginary part of the complex number @a x.
        template<typename T> NOA_FHD constexpr T imag(Complex<T> x) { return x.imag(); }

        /// Returns the phase angle (in radians) of the complex number @a z.
        template<typename T> NOA_HD T arg(const Complex<T>& x);

        /// Returns the magnitude of the complex number @a x.
        template<typename T> NOA_HD T abs(const Complex<T>& x);
        template<typename T> NOA_FHD T length(const Complex<T>& x) { return abs(x); }

        /** Returns the length-normalized of the complex number @a x to 1, reducing it to its phase. */
        template<typename T> NOA_HD Complex<T> normalize(const Complex<T>& x);

        /** Returns the squared magnitude of the complex number @a x. */
        template<typename T> NOA_HD T norm(const Complex<T>& x);
        template<typename T> NOA_FHD T lengthSq(const Complex<T>& x) { return norm(x); }

        /// Returns the complex conjugate of @a x.
        template<typename T> NOA_HD constexpr Complex<T> conj(const Complex<T>& x);

        /// Returns a complex number with magnitude @a length (should be positive) and phase angle @a theta.
        template<typename T> NOA_HD Complex<T> polar(T length, T theta);
    }

    // IMPLEMENTATION:
    template<typename FP>
    NOA_FHD constexpr Complex<FP>::Complex(const std::complex<FP>& x)
            : m_re(reinterpret_cast<const FP(&)[2]>(x)[0]),
              m_im(reinterpret_cast<const FP(&)[2]>(x)[1]) {}

    template<typename FP>
    template<typename U>
    NOA_FHD constexpr Complex<FP>::Complex(const std::complex<U>& x)
            : m_re(static_cast<FP>(reinterpret_cast<const U(&)[2]>(x)[0])),
              m_im(static_cast<FP>(reinterpret_cast<const U(&)[2]>(x)[1])) {}

    template<typename FP>
    template<class U>
    NOA_FHD constexpr Complex<FP>::Complex(const Complex<U>& x) : m_re(FP(x.m_re)), m_im(FP(x.m_im)) {}

    // Operator assignments.
    template<typename FP>
    NOA_FHD constexpr Complex<FP>& Complex<FP>::operator=(FP x) {
        m_re = x;
        m_im = FP(0);
        return *this;
    }

    template<typename FP>
    NOA_FHD constexpr Complex<FP>& Complex<FP>::operator+=(const Complex<FP>& x) {
        *this = *this + x;
        return *this;
    }

    template<typename FP>
    NOA_FHD constexpr Complex<FP>& Complex<FP>::operator-=(const Complex<FP>& x) {
        *this = *this - x;
        return *this;
    }

    template<typename FP>
    NOA_FHD constexpr Complex<FP>& Complex<FP>::operator*=(const Complex<FP>& x) {
        *this = *this * x;
        return *this;
    }

    template<typename FP>
    NOA_FHD constexpr Complex<FP>& Complex<FP>::operator/=(const Complex<FP>& x) {
        *this = *this / x;
        return *this;
    }

    template<typename FP>
    NOA_FHD constexpr Complex<FP>& Complex<FP>::operator+=(FP x) {
        *this = *this + x;
        return *this;
    }

    template<typename FP>
    NOA_FHD constexpr Complex<FP>& Complex<FP>::operator-=(FP x) {
        *this = *this - x;
        return *this;
    }

    template<typename FP>
    NOA_FHD constexpr Complex<FP>& Complex<FP>::operator*=(FP x) {
        *this = *this * x;
        return *this;
    }

    template<typename FP>
    NOA_FHD constexpr Complex<FP>& Complex<FP>::operator/=(FP x) {
        *this = *this / x;
        return *this;
    }

    /* --- Equality Operators --- */

    template<typename FP>
    NOA_FHD constexpr bool operator==(const Complex<FP>& x, const Complex<FP>& y) {
        return x.real() == y.real() && x.imag() == y.imag();
    }

    template<typename FP>
    NOA_FHD constexpr bool operator==(FP x, const Complex<FP>& y) { return Complex<FP>(x) == y; }

    template<typename FP>
    NOA_FHD constexpr bool operator==(const Complex<FP>& x, FP y) { return x == Complex<FP>(y); }

    template<typename FP>
    NOA_FHD constexpr bool operator!=(const Complex<FP>& x, const Complex<FP>& y) { return !(x == y); }

    template<typename FP>
    NOA_FHD constexpr bool operator!=(FP x, const Complex<FP>& y) { return Complex<FP>(x) != y; }

    template<typename FP>
    NOA_FHD constexpr bool operator!=(const Complex<FP>& x, FP y) { return x != Complex<FP>(y); }

    template<typename FP>
    NOA_FHD constexpr bool operator==(const Complex<FP>& x, const std::complex<FP>& y) {
        return x.real() == reinterpret_cast<const FP(&)[2]>(y)[0] && x.imag() == reinterpret_cast<const FP(&)[2]>(y)[1];
    }

    template<typename FP>
    NOA_FHD constexpr bool operator==(const std::complex<FP>& x, const Complex<FP>& y) {
        return reinterpret_cast<const FP(&)[2]>(x)[0] == y.real() && reinterpret_cast<const FP(&)[2]>(x)[1] == y.imag();
    }

    template<typename FP>
    NOA_FHD constexpr bool operator!=(const Complex<FP>& x, const std::complex<FP>& y) { return !(x == y); }

    template<typename FP>
    NOA_FHD constexpr bool operator!=(const std::complex<FP>& x, const Complex<FP>& y) { return !(x == y); }

    /* --- Binary Arithmetic Operators --- */

    // Add
    template<class FP>
    NOA_FHD constexpr Complex<FP> operator+(const Complex<FP>& x, const Complex<FP>& y) {
        return Complex<FP>(x.real() + y.real(), x.imag() + y.imag());
    }
    template<class FP>
    NOA_FHD constexpr Complex<FP> operator+(FP x, const Complex<FP>& y) {
        return Complex<FP>(x + y.real(), y.imag());
    }
    template<class FP>
    NOA_FHD constexpr Complex<FP> operator+(const Complex<FP>& x, FP y) {
        return Complex<FP>(x.real() + y, x.imag());
    }

    // Subtract
    template<class FP>
    NOA_FHD constexpr Complex<FP> operator-(const Complex<FP>& x, const Complex<FP>& y) {
        return Complex<FP>(x.real() - y.real(), x.imag() - y.imag());
    }
    template<class FP>
    NOA_FHD constexpr Complex<FP> operator-(FP x, const Complex<FP>& y) {
        return Complex<FP>(x - y.real(), -y.imag());
    }
    template<class FP>
    NOA_FHD constexpr Complex<FP> operator-(const Complex<FP>& x, FP y) {
        return Complex<FP>(x.real() - y, x.imag());
    }

    // Multiply
    template<class FP>
    NOA_FHD constexpr Complex<FP> operator*(const Complex<FP>& x, const Complex<FP>& y) {
        return Complex<FP>(x.real() * y.real() - x.imag() * y.imag(),
                           x.real() * y.imag() + x.imag() * y.real());
    }
    template<class FP>
    NOA_FHD constexpr Complex<FP> operator*(FP x, const Complex<FP>& y) {
        return Complex<FP>(x * y.real(), x * y.imag());
    }
    template<class FP>
    NOA_FHD constexpr Complex<FP> operator*(const Complex<FP>& x, FP y) {
        return Complex<FP>(x.real() * y, x.imag() * y);
    }

    // Divide
    /*
     * Adapted from cuComplex.h
     * "This implementation guards against intermediate underflow and overflow
     * by scaling. Such guarded implementations are usually the default for
     * complex library implementations, with some also offering an unguarded,
     * faster version."
     */
    template<class FP>
    NOA_HD Complex<FP> operator/(const Complex<FP>& x, const Complex<FP>& y) {
        FP s = abs(y.real()) + abs(y.imag());
        FP oos = FP(1.0) / s;

        FP ars = x.real() * oos;
        FP ais = x.imag() * oos;
        FP brs = y.real() * oos;
        FP bis = y.imag() * oos;

        s = (brs * brs) + (bis * bis);
        oos = FP(1.0) / s;

        return Complex<FP>(((ars * brs) + (ais * bis)) * oos,
                           ((ais * brs) - (ars * bis)) * oos);
    }
    template<class FP>
    NOA_FHD Complex<FP> operator/(FP x, const Complex<FP>& y) {
        return Complex<FP>(x) / y;
    }
    template<class FP>
    NOA_FHD Complex<FP> operator/(const Complex<FP>& x, FP y) {
        return Complex<FP>(x.real() / y, x.imag() / y);
    }

    /* --- Unary Arithmetic Operators --- */

    template<typename FP> NOA_FHD constexpr Complex<FP> operator+(const Complex<FP>& x) { return x; }
    template<typename FP> NOA_FHD constexpr Complex<FP> operator-(const Complex<FP>& x) { return x * -FP(1); }

    namespace Math {
        template<typename T> NOA_FHD T arg(const Complex<T>& x) { return atan2(x.imag(), x.real()); }
        template<typename T> NOA_FHD T abs(const Complex<T>& x) { return hypot(x.real(), x.imag()); }

        template<typename T> NOA_FHD Complex<T> normalize(const Complex<T>& x) {
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
        NOA_FHD constexpr Complex<T> conj(const Complex<T>& x) {
            return Complex<T>(x.real(), -x.imag());
        }

        template<typename T>
        NOA_FHD Complex<T> polar(T length, T theta) {
            return Complex<T>(length * cos(theta), length * sin(theta));
        }
    }
}

template<typename T>
struct fmt::formatter<Noa::Complex<T>> : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const Noa::Complex<T>& z, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(Noa::toString(z), ctx);
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Noa::Complex<T>& z) {
    os << Noa::toString(z);
    return os;
}
