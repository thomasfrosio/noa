// Adapted from thrust/complex.h.
// See licence/nvidia_thrust.txt.

#pragma once

namespace Noa::Math {
    NOA_DH inline Complex<double> pow(const Complex<double>& x, const Complex<double>& y) {
        return Math::exp(Math::log(x) * y);
    }

    NOA_DH inline Complex<double> pow(const Complex<double>& x, double y) {
        return Math::exp(Math::log(x) * y);
    }

    NOA_DH inline Complex<double> pow(double x, const Complex<double>& y) {
        return Math::exp(Math::log(x) * y);
    }

    NOA_DH inline Complex<float> pow(const Complex<float>& x, const Complex<float>& y) {
        return Math::exp(Math::log(x) * y);
    }

    NOA_DH inline Complex<float> pow(const Complex<float>& x, float y) {
        return Math::exp(Math::log(x) * y);
    }

    NOA_DH inline Complex<float> pow(float x, const Complex<float>& y) {
        return Math::exp(Math::log(x) * y);
    }
}
