#pragma once

namespace Noa {
    template<typename FP>
    struct Complex;

    template<typename FP>
    NOA_DH inline constexpr Complex<FP>::Complex(const std::complex<FP>& x)
            : m_re(reinterpret_cast<const FP(&)[2]>(x)[0]),
              m_im(reinterpret_cast<const FP(&)[2]>(x)[1]) {}

    template<typename FP>
    template<typename U>
    NOA_DH inline constexpr Complex<FP>::Complex(const std::complex<U>& x)
            : m_re(static_cast<FP>(reinterpret_cast<const U(&)[2]>(x)[0])),
              m_im(static_cast<FP>(reinterpret_cast<const U(&)[2]>(x)[1])) {}

    template<typename FP>
    template<class U>
    NOA_DH inline constexpr Complex<FP>::Complex(const Complex<U>& x) : m_re(FP(x.m_re)), m_im(FP(x.m_im)) {}

    // Operator assignments.
    template<typename FP>
    NOA_DH inline constexpr Complex<FP>& Complex<FP>::operator=(FP x) {
        m_re = x;
        m_im = FP(0);
        return *this;
    }

    template<typename FP>
    NOA_DH inline constexpr Complex<FP>& Complex<FP>::operator+=(const Complex<FP>& x) {
        *this = *this + x;
        return *this;
    }

    template<typename FP>
    NOA_DH inline constexpr Complex<FP>& Complex<FP>::operator-=(const Complex<FP>& x) {
        *this = *this - x;
        return *this;
    }

    template<typename FP>
    NOA_DH inline constexpr Complex<FP>& Complex<FP>::operator*=(const Complex<FP>& x) {
        *this = *this * x;
        return *this;
    }

    template<typename FP>
    NOA_DH inline constexpr Complex<FP>& Complex<FP>::operator/=(const Complex<FP>& x) {
        *this = *this / x;
        return *this;
    }

    template<typename FP>
    NOA_DH inline constexpr Complex<FP>& Complex<FP>::operator+=(FP x) {
        *this = *this + x;
        return *this;
    }

    template<typename FP>
    NOA_DH inline constexpr Complex<FP>& Complex<FP>::operator-=(FP x) {
        *this = *this - x;
        return *this;
    }

    template<typename FP>
    NOA_DH inline constexpr Complex<FP>& Complex<FP>::operator*=(FP x) {
        *this = *this * x;
        return *this;
    }

    template<typename FP>
    NOA_DH inline constexpr Complex<FP>& Complex<FP>::operator/=(FP x) {
        *this = *this / x;
        return *this;
    }

    /* --- Equality Operators --- */

    template<typename FP>
    NOA_DH inline constexpr bool operator==(const Complex<FP>& x, const Complex<FP>& y) {
        return x.real() == y.real() && x.imag() == y.imag();
    }

    template<typename FP>
    NOA_DH inline constexpr bool operator==(FP x, const Complex<FP>& y) { return Complex<FP>(x) == y; }

    template<typename FP>
    NOA_DH inline constexpr bool operator==(const Complex<FP>& x, FP y) { return x == Complex<FP>(y); }

    template<typename FP>
    NOA_DH inline constexpr bool operator!=(const Complex<FP>& x, const Complex<FP>& y) { return !(x == y); }

    template<typename FP>
    NOA_DH inline constexpr bool operator!=(FP x, const Complex<FP>& y) { return Complex<FP>(x) != y; }

    template<typename FP>
    NOA_DH inline constexpr bool operator!=(const Complex<FP>& x, FP y) { return x != Complex<FP>(y); }

    template<typename FP>
    NOA_DH inline constexpr bool operator==(const Complex<FP>& x, const std::complex<FP>& y) {
        return x.real() == reinterpret_cast<const FP(&)[2]>(y)[0] && x.imag() == reinterpret_cast<const FP(&)[2]>(y)[1];
    }

    template<typename FP>
    NOA_DH inline constexpr bool operator==(const std::complex<FP>& x, const Complex<FP>& y) {
        return reinterpret_cast<const FP(&)[2]>(x)[0] == y.real() && reinterpret_cast<const FP(&)[2]>(x)[1] == y.imag();
    }

    template<typename FP>
    NOA_DH inline constexpr bool operator!=(const Complex<FP>& x, const std::complex<FP>& y) { return !(x == y); }

    template<typename FP>
    NOA_DH inline constexpr bool operator!=(const std::complex<FP>& x, const Complex<FP>& y) { return !(x == y); }
}

#include "noa/util/complex/carith.h"
#include "noa/util/complex/cproj.h"
#include "noa/util/complex/cexp.h"
#include "noa/util/complex/clog.h"
#include "noa/util/complex/cpow.h"
#include "noa/util/complex/csqrt.h"
