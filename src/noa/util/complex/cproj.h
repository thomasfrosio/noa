// Adapted from thrust/complex.h.
// See licence/nvidia_thrust.txt.

#pragma once

#include "noa/Define.h"
#include "noa/util/Math.h"
#include "noa/util/complex/math_private.h"

// Implementation for Math::proj(Complex<X>)
namespace Noa::Math::Details::Complex {
    NOA_IHD Noa::Complex<float> cprojf(const Noa::Complex<float>& z) {
        if (!Math::isInf(z.real()) && !Math::isInf(z.imag())) {
            return z;
        } else {
            // std::numeric_limits<T>::infinity() doesn't run on the GPU
            return Noa::Complex<float>(infinity<float>(), Math::copysign(0.0f, z.imag()));
        }
    }

    NOA_IHD Noa::Complex<double> cproj(const Noa::Complex<double>& z) {
        if (!Math::isInf(z.real()) && !Math::isInf(z.imag())) {
            return z;
        } else {
            // std::numeric_limits<T>::infinity() doesn't run on the GPU
            return Noa::Complex<double>(infinity<double>(), Noa::Math::copysign(0.0, z.imag()));
        }
    }
}

namespace Noa::Math {
    NOA_IHD Noa::Complex<float> proj(const Noa::Complex<float>& x) {
        return Details::Complex::cprojf(x);
    }

    NOA_IHD Noa::Complex<double> proj(const Noa::Complex<double>& x) {
        return Details::Complex::cproj(x);
    }
}
