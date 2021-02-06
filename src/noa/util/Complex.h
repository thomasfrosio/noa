#pragma once

#include <complex>
#include <cfloat>

#include "noa/Define.h"
#include "noa/util/Math.h"

/*
 * Notes:
 *  -   Converting a std::complex<> into a Noa::Complex<> can be done on the device, since in C++11 it is required for
 *      std::complex to have an array-oriented access. See https://en.cppreference.com/w/cpp/numeric/complex
 *
 *  -   Reinterpreting a Noa::Complex<> into its underlying type is defined behavior.
 *      See https://en.cppreference.com/w/cpp/language/reinterpret_cast.
 *      Unittests will make sure there's no weird padding and alignment is as expected.
 *      As such, it can assumed that the example below works:
 *      {
 *          Noa::Complex<float> cfloat_array[256];
 *          auto* underlying_data = reinterpret_cast<float*>(cfloat_array);
 *          // as mentioned above, this cast is defined and does not violate the strict aliasing rule:
 *          // access elements of underlying_data as if it was an array of 256*2 floats.
 *      }
 *
 *  -   The implementation (adapted from nvidia/thrust) assumes int is 32bits and little-endian.
 *      This is unlikely to be a issue, but it is worth mentioning.
 *      Unittests will make sure this is respected.
 */

namespace Noa {
    template<typename FP>
    struct alignas(sizeof(FP) * 2) Complex {
    private:
        std::enable_if_t<std::is_same_v<FP, float> || std::is_same_v<FP, double>, FP> m_re{}, m_im{};
    public:
        // Base constructors.
        NOA_HD constexpr Complex() = default;
        NOA_HD constexpr Complex(const Complex<FP>& c) = default;
        NOA_HD constexpr explicit Complex(FP re) : m_re(re), m_im(0) {};
        NOA_HD constexpr Complex(FP re, FP im) : m_re(re), m_im(im) {};

        // Conversion constructors.
        NOA_HD constexpr explicit Complex(const std::complex<FP>& x);
        template<class U>
        NOA_HD constexpr explicit Complex(const std::complex<U>& x);
        template<class U>
        NOA_HD constexpr explicit Complex(const Complex<U>& x);

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

    //@CLION-formatter:off
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
    //@CLION-formatter:on

    /* --- Unary Arithmetic Operators --- */
    template<typename FP>
    NOA_HD constexpr Complex<FP> operator+(const Complex<FP>& x);
    template<typename FP>
    NOA_HD constexpr Complex<FP> operator-(const Complex<FP>& x);

    template<class T>
    NOA_IH std::string toString(const Complex<T>& z) { return String::format("({},{})", z.real(), z.imag()); }

    namespace Math {
        /* --- carith.h --- */
        /** Returns the phase angle (in radians) of the complex number @a z. */
        NOA_HD double arg(const Complex<double>& x);
        NOA_HD float arg(const Complex<float>& x);

        /** Returns the magnitude of the complex number @a x. */
        NOA_HD double abs(const Complex<double>& x);
        NOA_HD float abs(const Complex<float>& x);

        /** Returns the complex conjugate of @a x. */
        NOA_HD constexpr Complex<double> conj(const Complex<double>& x);
        NOA_HD constexpr Complex<float> conj(const Complex<float>& x);

        /** Returns the magnitude of the complex number @a x. */
        NOA_IHD double length(const Complex<double>& x) { return abs(x); }
        NOA_IHD float length(const Complex<float>& x) { return abs(x); }

        /** Returns the squared magnitude of the complex number @a x. */
        NOA_HD double norm(const Complex<double>& x);
        NOA_HD float norm(const Complex<float>& x);

        /** Returns the magnitude of the complex number @a x. */
        NOA_IHD double lengthSq(const Complex<double>& x) { return norm(x); }
        NOA_IHD float lengthSq(const Complex<float>& x) { return norm(x); }

        /** Returns a complex number with magnitude @a length (should be positive) and phase angle @a theta. */
        NOA_HD Complex<double> polar(double length, double phase);
        NOA_HD Complex<float> polar(float length, float theta);

        /** Returns a complex number with magnitude @a length (should be positive) and phase angle @a theta. */
        NOA_HD Complex<double> proj(const Complex<double>& x);
        NOA_HD Complex<float> proj(const Complex<float>& x);

        /** Computes base-e exponential of @a z. */
        NOA_HD Complex<double> exp(const Complex<double>& z);
        NOA_HD Complex<float> exp(const Complex<float>& z);

        /** Computes the complex natural logarithm with the branch cuts along the negative real axis. */
        NOA_HD Complex<double> log(const Complex<double>& z);
        NOA_HD Complex<float> log(const Complex<float>& z);

        /** Computes the complex common logarithm with the branch cuts along the negative real axis. */
        NOA_HD Complex<double> log10(const Complex<double>& z);
        NOA_HD Complex<float> log10(const Complex<float>& z);

        /** Computes the complex power, that is `exp(y*log(x))`, one or both arguments may be a complex number. */
        NOA_HD Complex<double> pow(const Complex<double>& x, const Complex<double>& y);
        NOA_HD Complex<double> pow(const Complex<double>& x, double y);
        NOA_HD Complex<double> pow(double x, const Complex<double>& y);
        NOA_HD Complex<float> pow(const Complex<float>& x, const Complex<float>& y);
        NOA_HD Complex<float> pow(const Complex<float>& x, float y);
        NOA_HD Complex<float> pow(float x, const Complex<float>& y);

        /**  Returns the square root of @a z using the principal branch, whose cuts are along the negative real axis. */
        NOA_HD Complex<double> sqrt(const Complex<double>& z);
        NOA_HD Complex<float> sqrt(const Complex<float>& z);

        // For now, let's just ignore these since they are quite complicated
        // and I [ffyr2w] am not sure they are even used in cryoEM.
        //template<class FP> NOA_HD Complex<FP> sin(const Complex<FP>& c);
        //template<class FP> NOA_HD Complex<FP> cos(const Complex<FP>& c);
        //template<class FP> NOA_HD Complex<FP> tan(const Complex<FP>& c);
        //template<class FP> NOA_HD Complex<FP> asin(const Complex<FP>& c);
        //template<class FP> NOA_HD Complex<FP> acos(const Complex<FP>& c);
        //template<class FP> NOA_HD Complex<FP> atan(const Complex<FP>& c);

        /** Returns the real part of the complex number @a x. */
        NOA_IHD double real(Complex<double> x) { return x.real(); }
        NOA_IHD float real(Complex<float> x) { return x.real(); }

        /** Returns the imaginary part of the complex number @a x. */
        NOA_IHD double imag(Complex<double> x) { return x.imag(); }
        NOA_IHD float imag(Complex<float> x) { return x.imag(); }
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

#include "noa/util/complex/Complex-inl.h"
