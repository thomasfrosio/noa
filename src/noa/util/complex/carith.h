// Adapted from thrust/complex.h.
// See licence/nvidia_thrust.txt.

#pragma once

#include <cfloat>
#include "cuComplex.h"

namespace Noa {
    template<class T>
    struct Complex;

    /* --- Binary Arithmetic Operators --- */

    // Add
    template<class FP>
    NOA_DH constexpr inline Complex<FP> operator+(const Complex<FP>& x, const Complex<FP>& y) {
        return Complex<FP>(x.real() + y.real(), x.imag() + y.imag());
    }
    template<class FP>
    NOA_DH constexpr inline Complex<FP> operator+(FP x, const Complex<FP>& y) {
        return Complex<FP>(x + y.real(), y.imag());
    }
    template<class FP>
    NOA_DH constexpr inline Complex<FP> operator+(const Complex<FP>& x, FP y) {
        return Complex<FP>(x.real() + y, x.imag());
    }

    // Subtract
    template<class FP>
    NOA_DH constexpr inline Complex<FP> operator-(const Complex<FP>& x, const Complex<FP>& y) {
        return Complex<FP>(x.real() - y.real(), x.imag() - y.imag());
    }
    template<class FP>
    NOA_DH constexpr inline Complex<FP> operator-(FP x, const Complex<FP>& y) {
        return Complex<FP>(x - y.real(), -y.imag());
    }
    template<class FP>
    NOA_DH constexpr inline Complex<FP> operator-(const Complex<FP>& x, FP y) {
        return Complex<FP>(x.real() - y, x.imag());
    }

    // Multiply
    template<class FP>
    NOA_DH constexpr inline Complex<FP> operator*(const Complex<FP>& x, const Complex<FP>& y) {
        return Complex<FP>(x.real() * y.real() - x.imag() * y.imag(),
                           x.real() * y.imag() + x.imag() * y.real());
    }
    template<class FP>
    NOA_DH constexpr inline Complex<FP> operator*(FP x, const Complex<FP>& y) {
        return Complex<FP>(x * y.real(), x * y.imag());
    }
    template<class FP>
    NOA_DH constexpr inline Complex<FP> operator*(const Complex<FP>& x, FP y) {
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
    NOA_DH Complex<FP> operator/(const Complex<FP>& x, const Complex<FP>& y) {
        FP s = Math::abs(y.real()) + Math::abs(y.imag());
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
    NOA_DH inline Complex<FP> operator/(FP x, const Complex<FP>& y) {
        return Complex<FP>(x) / y;
    }
    template<class FP>
    NOA_DH inline Complex<FP> operator/(const Complex<FP>& x, FP y) {
        return Complex<FP>(x.real() / y, x.imag() / y);
    }

    /* --- Unary Arithmetic Operators --- */
    template<typename FP> NOA_DH inline constexpr Complex<FP> operator+(const Complex<FP>& x) { return x; }
    template<typename FP> NOA_DH inline constexpr Complex<FP> operator-(const Complex<FP>& x) {return x * -FP(1); }
}

namespace Noa::Math {
    NOA_DH inline double abs(const Complex<double>& x) {
        return hypot(x.real(), x.imag());
    }
    NOA_DH inline float abs(const Complex<float>& x) {
        return hypot(x.real(), x.imag());
    }

    NOA_DH inline double arg(const Complex<double>& x) {
        return atan2(x.imag(), x.real());
    }
    NOA_DH inline float arg(const Complex<float>& x) {
        return atan2(x.imag(), x.real());
    }

    NOA_DH constexpr inline Complex<double> conj(const Complex<double>& x) {
        return Complex<double>(x.real(), -x.imag());
    }
    NOA_DH constexpr inline Complex<float> conj(const Complex<float>& x) {
        return Complex<float>(x.real(), -x.imag());
    }

    NOA_DH inline float norm(const Complex<float>& x) {
        if (abs(x.real()) < sqrt(FLT_MIN) && abs(x.imag()) < sqrt(FLT_MIN)) {
            float a = x.real() * 4.0f;
            float b = x.imag() * 4.0f;
            return (a * a + b * b) / 16.0f;
        }
        return x.real() * x.real() + x.imag() * x.imag();
    }
    NOA_DH inline double norm(const Complex<double>& x) {
        if (abs(x.real()) < sqrt(DBL_MIN) && abs(x.imag()) < sqrt(DBL_MIN)) {
            double a = x.real() * 4.0;
            double b = x.imag() * 4.0;
            return (a * a + b * b) / 16.0;
        }
        return x.real() * x.real() + x.imag() * x.imag();
    }

    NOA_DH inline Complex<double> polar(double length, double theta) {
        return Complex<double>(length * cos(theta), length * sin(theta));
    }
    NOA_DH inline Complex<float> polar(float length, float theta) {
        return Complex<float>(length * cos(theta), length * sin(theta));
    }
}
