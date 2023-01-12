#pragma once

#include "noa/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::signal::fft {
    template<noa::fft::Remap REMAP, typename Real,
             typename = std::enable_if_t<noa::traits::is_any_v<Real, float, double>>>
    void isotropicFSC(const shared_t<Complex<Real>[]>& lhs, dim4_t lhs_strides,
                      const shared_t<Complex<Real>[]>& rhs, dim4_t rhs_strides,
                      const shared_t<Real[]>& fsc,
                      dim4_t shape,
                      cuda::Stream& stream);

    template<noa::fft::Remap REMAP, typename Real,
             typename = std::enable_if_t<noa::traits::is_any_v<Real, float, double>>>
    void anisotropicFSC(const shared_t<Complex<Real>[]>& lhs, dim4_t lhs_strides,
                        const shared_t<Complex<Real>[]>& rhs, dim4_t rhs_strides,
                        const shared_t<Real[]>& fsc,
                        dim4_t shape,
                        const shared_t<float3_t[]>& normalized_cone_directions,
                        dim_t cone_count, float cone_aperture,
                        cuda::Stream& stream);
}
