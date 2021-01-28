// Adapted from thrust/complex.h.
// See licence/nvidia_thrust.txt.

#pragma once

#include "noa/util/Complex.h"

namespace Noa::Math {
    NOA_DH inline Complex<double> pow(const Complex<double>& z1, const Complex<double>& z2) {
        return Math::exp(Math::log(z1) * z2);
    }

    NOA_DH inline Complex<double> pow(const Complex<double>& z, double v) {
        return Math::exp(Math::log(z) * v);
    }

    NOA_DH inline Complex<double> pow(double v, const Complex<double>& z) {
        return Math::exp(Math::log(v) * z);
    }

    NOA_DH inline Complex<float> pow(const Complex<float>& z1, const Complex<float>& z2) {
        return Math::exp(Math::log(z1) * z2);
    }

    NOA_DH inline Complex<float> pow(const Complex<float>& z, float v) {
        return Math::exp(Math::log(z) * v);
    }

    NOA_DH inline Complex<float> pow(float v, const Complex<float>& z) {
        return Math::exp(Math::log(v) * z);
    }
}
