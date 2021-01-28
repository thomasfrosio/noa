// Adapted from thrust/complex.h.
// See licence/nvidia_thrust.txt.

#pragma once

#include <cfloat>

#include "noa/util/Complex.h"

namespace Noa {
    /* --- Binary Arithmetic Operators --- */

    // Add
    template<class FP>
    NOA_DH constexpr Complex<FP> operator+(const Complex<FP>& c1, const Complex<FP>& c2) {
        return Complex<FP>(c1.real() + c2.real(), c1.imag() + c2.imag());
    }

    template<class FP>
    NOA_DH constexpr Complex<FP> operator+(FP v, const Complex<FP>& c) {
        return Complex<FP>(v + c.real(), c.imag());
    }

    template<class FP>
    NOA_DH constexpr Complex<FP> operator+(const Complex<FP>& c, FP v) {
        return Complex<FP>(c.real() + v, c.imag());
    }

    // Subtract
    template<class FP>
    NOA_DH constexpr Complex<FP> operator-(const Complex<FP>& c1, const Complex<FP>& c2) {
        return Complex<FP>(c1.real() - c2.real(), c1.imag() - c2.imag());
    }

    template<class FP>
    NOA_DH constexpr Complex<FP> operator-(FP v, const Complex<FP>& c) {
        return Complex<FP>(v - c.real(), c.imag());
    }

    template<class FP>
    NOA_DH constexpr Complex<FP> operator-(const Complex<FP>& c, FP v) {
        return Complex<FP>(c.real() - v, c.imag());
    }

    // Multiply
    template<class FP>
    NOA_DH constexpr Complex<FP> operator*(const Complex<FP>& c1, const Complex<FP>& c2) {
        return Complex<FP>(c1.real() * c2.real() - c1.imag() * c2.imag(),
                           c1.real() * c2.imag() + c1.imag() * c2.real());
    }

    template<class FP>
    NOA_DH constexpr Complex<FP> operator*(FP v, const Complex<FP>& c) {
        return Complex<FP>(v * c.real(), v * c.imag());
    }

    template<class FP>
    NOA_DH constexpr Complex<FP> operator*(const Complex<FP>& c, FP v) {
        return Complex<FP>(c.real() * v, c.imag() * v);
    }

    // Divide
    template<class FP>
    NOA_DH constexpr Complex<FP> operator/(const Complex<FP>& x, const Complex<FP>& y) {
        // Adapted from cuComplex.h
        FP s = Math::abs(x.real()) + Math::abs(y.imag());
        FP oos = FP(1) / s;

        FP ars = x.real() * oos;
        FP ais = x.imag() * oos;
        FP brs = y.real() * oos;
        FP bis = y.imag() * oos;

        s = (brs * brs) + (bis * bis);
        oos = FP(1) / s;

        return Complex<FP>(((ars * brs) + (ais * bis)) * oos,
                           ((ais * brs) - (ars * bis)) * oos);
    }

    template<class FP>
    NOA_DH constexpr Complex<FP> operator/(FP v, const Complex<FP>& c) {
        return Complex<FP>(v) / c;
    }

    template<class FP>
    NOA_DH constexpr Complex<FP> operator/(const Complex<FP>& c, FP v) {
        return Complex<FP>(c.real() / v, c.imag() / v);
    }

    /* --- Equality Operators --- */

    template<typename FP>
    NOA_DH constexpr bool operator==(const Complex<FP>& c1, const Complex<FP>& c2) {
        return c1.real() == c2.real() && c1.imag() == c2.imag();
    }

    template<typename FP>
    NOA_DH constexpr bool operator==(FP re, const Complex<FP>& c) { return Complex<FP>(re) == c; }

    template<typename FP>
    NOA_DH constexpr bool operator==(const Complex<FP>& c, FP re) { return c == Complex<FP>(re); }

    template<typename FP>
    NOA_DH constexpr bool operator!=(const Complex<FP>& c1, const Complex<FP>& c2) { return !(c1 == c2); }

    template<typename FP>
    NOA_DH constexpr bool operator!=(FP re, const Complex<FP>& c) { return Complex<FP>(re) != c; }

    template<typename FP>
    NOA_DH constexpr bool operator!=(const Complex<FP>& c, FP re) { return c != Complex<FP>(re); }
}

namespace Noa::Math {
    NOA_DH inline double abs(const Complex<double>& z) {
        return hypot(z.real(), z.imag());
    }

    NOA_DH inline float abs(const Complex<float>& z) {
        return hypot(z.real(), z.imag());
    }

    NOA_DH inline double arg(const Complex<double>& z) {
        return atan2(z.imag(), z.real());
    }

    NOA_DH inline float arg(const Complex<float>& z) {
        return atan2(z.imag(), z.real());
    }

    NOA_DH inline float norm(const Complex<float>& z) {
        if (abs(z.real()) < sqrt(FLT_MIN) && abs(z.imag()) < sqrt(FLT_MIN)) {
            float a = z.real() * 4.0f;
            float b = z.imag() * 4.0f;
            return (a * a + b * b) / 16.0f;
        }

        return z.real() * z.real() + z.imag() * z.imag();
    }

    NOA_DH inline double norm(const Complex<double>& z) {
        if (abs(z.real()) < sqrt(DBL_MIN) && abs(z.imag()) < sqrt(DBL_MIN)) {
            double a = z.real() * 4.0;
            double b = z.imag() * 4.0;
            return (a * a + b * b) / 16.0;
        }

        return z.real() * z.real() + z.imag() * z.imag();
    }

    NOA_DH constexpr inline Complex<double> conj(const Complex<double>& z) {
        return Complex<double>(z.real(), -z.imag());
    }

    NOA_DH constexpr inline Complex<float> conj(const Complex<float>& z) {
        return Complex<float>(z.real(), -z.imag());
    }

    NOA_DH inline Complex<double> polar(double mag, double phase) {
        return Complex<double>(mag * cos(phase), mag * sin(phase));
    }

    NOA_DH inline Complex<float> polar(float mag, float phase) {
        return Complex<float>(mag * cos(phase), mag * sin(phase));
    }
}
