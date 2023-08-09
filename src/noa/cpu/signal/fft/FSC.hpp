#pragma once

#include "noa/core/Types.hpp"

namespace noa::cpu::signal::fft {
    template<noa::fft::Remap REMAP, typename Real,
             typename = std::enable_if_t<nt::is_any_v<Real, f32, f64>>>
    void isotropic_fsc(const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
                       const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
                       Real* fsc,
                       const Shape4<i64>& shape,
                       i64 threads);

    template<noa::fft::Remap REMAP, typename Real,
             typename = std::enable_if_t<nt::is_any_v<Real, f32, f64>>>
    void anisotropic_fsc(const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
                         const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
                         Real* fsc,
                         const Shape4<i64>& shape,
                         const Vec3<f32>* normalized_cone_directions,
                         i64 cone_count, f32 cone_aperture,
                         i64 threads);
}
