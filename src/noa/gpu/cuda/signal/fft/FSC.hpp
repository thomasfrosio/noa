#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::signal::fft {
    template<noa::fft::Remap REMAP, typename Real,
             typename = std::enable_if_t<noa::traits::is_any_v<Real, f32, f64>>>
    void isotropic_fsc(
            const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
            const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
            Real* fsc,
            const Shape4<i64>& shape,
            Stream& stream);

    template<noa::fft::Remap REMAP, typename Real,
             typename = std::enable_if_t<noa::traits::is_any_v<Real, f32, f64>>>
    void anisotropic_fsc(
            const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
            const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
            Real* fsc,
            const Shape4<i64>& shape,
            const Vec3<f32>* normalized_cone_directions,
            i64 cone_count, f32 cone_aperture,
            Stream& stream);
}
