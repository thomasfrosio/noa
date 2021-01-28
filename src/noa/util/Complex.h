#pragma once

#include <complex>
#include <cfloat>

#include "noa/Define.h"
#include "noa/util/Math.h"

//@CLION-formatter:off
namespace Noa {
    template<typename FP>
    struct alignas(sizeof(FP) * 2) Complex {
    private:
        std::enable_if_t<std::is_same_v<FP, float> || std::is_same_v<FP, double>, FP> m_re{}, m_im{};
    public:
        // Base constructors.
        NOA_DH constexpr Complex() = default;
        NOA_DH constexpr explicit Complex(FP re) : m_re(re), m_im(0) {};
        NOA_DH constexpr Complex(FP re, FP im) : m_re(re), m_im(im) {};
        NOA_DH constexpr Complex(const Complex<FP>& c) = default;

        // Conversion constructors.
        NOA_DH constexpr explicit Complex(const std::complex<FP>& c) : m_re(c.real()), m_im(c.imag()) {}
        template<class U> NOA_DH constexpr explicit Complex(const std::complex<U>& c) : m_re(FP(c.real())), m_im(FP(c.imag())) {}
        template<class U> NOA_DH constexpr explicit Complex(const Complex<U>& c) : m_re(FP(c.m_re)), m_im(FP(c.m_im)) {}

        // Operator assignments.
        NOA_DH constexpr Complex<FP>& operator=(FP re);
        NOA_DH constexpr Complex<FP>& operator=(const Complex<FP>& c);

        NOA_DH constexpr Complex<FP>& operator+=(const Complex<FP>& c);
        NOA_DH constexpr Complex<FP>& operator-=(const Complex<FP>& c);
        NOA_DH constexpr Complex<FP>& operator*=(const Complex<FP>& c);
        NOA_DH constexpr Complex<FP>& operator/=(const Complex<FP>& c);

        NOA_DH constexpr Complex<FP>& operator+=(FP v);
        NOA_DH constexpr Complex<FP>& operator-=(FP v);
        NOA_DH constexpr Complex<FP>& operator*=(FP v);
        NOA_DH constexpr Complex<FP>& operator/=(FP v);

        NOA_DH constexpr inline FP real() const volatile { return m_re; }
        NOA_DH constexpr inline FP real() const { return m_re; }
        NOA_DH constexpr inline FP imag() const volatile { return m_im; }
        NOA_DH constexpr inline FP imag() const { return m_im; }

        NOA_DH constexpr inline void real(FP re) const volatile { m_re = re; }
        NOA_DH constexpr inline void real(FP re) const { m_re = re; }
        NOA_DH constexpr inline void imag(FP im) const volatile { m_im = im; }
        NOA_DH constexpr inline void imag(FP im) const { m_im = im; }
    };

    /* --- Binary Arithmetic Operators --- */
    // Add
    template<class FP> NOA_DH constexpr Complex<FP> operator+(const Complex<FP>& c1, const Complex<FP>& c2);
    template<class FP> NOA_DH constexpr Complex<FP> operator+(FP v, const Complex<FP>& c);
    template<class FP> NOA_DH constexpr Complex<FP> operator+(const Complex<FP>& c, FP v);

    // Subtract
    template<class FP> NOA_DH constexpr Complex<FP> operator-(const Complex<FP>& c1, const Complex<FP>& c2);
    template<class FP> NOA_DH constexpr Complex<FP> operator-(FP v, const Complex<FP>& c);
    template<class FP> NOA_DH constexpr Complex<FP> operator-(const Complex<FP>& c, FP v);

    // Multiply
    template<class FP> NOA_DH constexpr Complex<FP> operator*(const Complex<FP>& c1, const Complex<FP>& c2);
    template<class FP> NOA_DH constexpr Complex<FP> operator*(FP v, const Complex<FP>& c);
    template<class FP> NOA_DH constexpr Complex<FP> operator*(const Complex<FP>& c, FP v);

    // Divide
    template<class FP> NOA_DH constexpr Complex<FP> operator/(const Complex<FP>& c1, const Complex<FP>& c2);
    template<class FP> NOA_DH constexpr Complex<FP> operator/(FP v, const Complex<FP>& c);
    template<class FP> NOA_DH constexpr Complex<FP> operator/(const Complex<FP>& c, FP v);

    /* --- Equality Operators - These are mostly useless --- */
    template<class FP> NOA_DH constexpr bool operator==(const Complex<FP>& c1, const Complex<FP>& c2);
    template<class FP> NOA_DH constexpr bool operator==(FP re, const Complex<FP>& c);
    template<class FP> NOA_DH constexpr bool operator==(const Complex<FP>& c, FP re);

    template<class FP> NOA_DH constexpr bool operator!=(const Complex<FP>& c1, const Complex<FP>& c2);
    template<class FP> NOA_DH constexpr bool operator!=(FP re, const Complex<FP>& c);
    template<class FP> NOA_DH constexpr bool operator!=(const Complex<FP>& c, FP re);

    namespace Math {
        // Overloads
        NOA_DH double abs(const Complex<double>& z);
        NOA_DH float abs(const Complex<float>& z);

        NOA_DH Complex<double> exp(const Complex<double>& z);
        NOA_DH Complex<float> exp(const Complex<float>& z);

        NOA_DH Complex<double> sqrt(const Complex<double>& z);
        NOA_DH Complex<float> sqrt(const Complex<float>& z);

        NOA_DH Complex<double> log(const Complex<double>& z);
        NOA_DH Complex<float> log(const Complex<float>& z);

        NOA_DH Complex<double> log10(const Complex<double>& z);
        NOA_DH Complex<float> log10(const Complex<float>& z);

        NOA_DH Complex<double> pow(const Complex<double>& z1, const Complex<double>& z2);
        NOA_DH Complex<double> pow(const Complex<double>& z, double v);
        NOA_DH Complex<double> pow(double v, const Complex<double>& z);

        NOA_DH Complex<float> pow(const Complex<float>& z1, const Complex<float>& z2);
        NOA_DH Complex<float> pow(const Complex<float>& z, float v);
        NOA_DH Complex<float> pow(float v, const Complex<float>& z);

        // For now, let's just ignore these since they are quite complicated
        // and I [ffyr2w] am not sure they are even used.
        //template<class FP> NOA_DH Complex<FP> sin(const Complex<FP>& c);
        //template<class FP> NOA_DH Complex<FP> cos(const Complex<FP>& c);
        //template<class FP> NOA_DH Complex<FP> tan(const Complex<FP>& c);
        //template<class FP> NOA_DH Complex<FP> asin(const Complex<FP>& c);
        //template<class FP> NOA_DH Complex<FP> acos(const Complex<FP>& c);
        //template<class FP> NOA_DH Complex<FP> atan(const Complex<FP>& c);

        // Supplement
        NOA_DH double arg(const Complex<double>& z);
        NOA_DH float arg(const Complex<float>& z);

        NOA_DH double norm(const Complex<double>& z);
        NOA_DH float norm(const Complex<float>& z);

        NOA_DH constexpr Complex<double> conj(const Complex<double>& z);
        NOA_DH constexpr Complex<float> conj(const Complex<float>& z);

        NOA_DH Complex<double> polar(double mag, double phase);
        NOA_DH Complex<float> polar(float mag, float phase);
    }
}
//@CLION-formatter:on

namespace Noa {
    template<class FP>
    NOA_DH constexpr inline Complex<FP>& Complex<FP>::operator=(FP real) {
        m_re = real;
        return *this;
    }

    template<class FP>
    NOA_DH constexpr inline Complex<FP>& Complex<FP>::operator=(const Complex<FP>& c) {
        m_re = c.m_re;
        m_im = c.m_im;
        return *this;
    }

    template<typename FP>
    NOA_DH constexpr inline Complex<FP>& Complex<FP>::operator+=(const Complex<FP>& c) {
        *this = *this + c;
        return *this;
    }

    template<typename FP>
    NOA_DH constexpr inline Complex<FP>& Complex<FP>::operator-=(const Complex<FP>& c) {
        *this = *this - c;
        return *this;
    }

    template<typename FP>
    NOA_DH constexpr inline Complex<FP>& Complex<FP>::operator*=(const Complex<FP>& c) {
        *this = *this * c;
        return *this;
    }

    template<typename FP>
    NOA_DH constexpr inline Complex<FP>& Complex<FP>::operator/=(const Complex<FP>& c) {
        *this = *this / c;
        return *this;
    }

    template<typename FP>
    NOA_DH constexpr inline Complex<FP>& Complex<FP>::operator+=(FP v) {
        *this = *this + v;
        return *this;
    }

    template<typename FP>
    NOA_DH constexpr inline Complex<FP>& Complex<FP>::operator-=(FP v) {
        *this = *this - v;
        return *this;
    }

    template<typename FP>
    NOA_DH constexpr inline Complex<FP>& Complex<FP>::operator*=(FP v) {
        *this = *this * v;
        return *this;
    }

    template<typename FP>
    NOA_DH constexpr inline Complex<FP>& Complex<FP>::operator/=(FP v) {
        *this = *this / v;
        return *this;
    }
}

#include "noa/util/complex/carith.h"
#include "noa/util/complex/cexp.h"
#include "noa/util/complex/clog.h"
#include "noa/util/complex/cpow.h"
#include "noa/util/complex/csqrt.h"
