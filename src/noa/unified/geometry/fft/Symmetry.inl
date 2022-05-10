#pragma once

#ifndef NOA_UNIFIED_FFT_SYMMETRY_
#error "This is a private header"
#endif

#include "noa/cpu/geometry/fft/Symmetry.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Symmetry.h"
#endif

namespace noa::geometry::fft {
    template<Remap REMAP, typename T>
    void symmetrize2D(const Array<T>& input, const Array<T>& output,
                      const Symmetry& symmetry, float2_t shift,
                      float cutoff, InterpMode interp_mode, bool normalize) {
        transform2D<REMAP>(input, output, float22_t{}, symmetry, shift, cutoff, interp_mode, normalize);
    }

    template<Remap REMAP, typename T>
    void symmetrize3D(const Array<T>& input, const Array<T>& output,
                      const Symmetry& symmetry, float3_t shift,
                      float cutoff, InterpMode interp_mode, bool normalize) {
        transform3D<REMAP>(input, output, float33_t{}, symmetry, shift, cutoff, interp_mode, normalize);
    }
}
