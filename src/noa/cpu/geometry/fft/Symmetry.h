#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/geometry/fft/Transform.h"

namespace noa::cpu::geometry::fft {
    // Symmetrizes a non-redundant 2D (batched) FFT.
    // TODO ADD TESTS!
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, T>>>
    void symmetrize2D(const shared_t<T[]>& input, dim4_t input_strides,
                      const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                      const Symmetry& symmetry, float2_t shift,
                      float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        transform2D<REMAP>(input, input_strides, output, output_strides, shape, float22_t{}, symmetry,
                           shift, cutoff, interp_mode, normalize, stream);
    }

    // Symmetrizes a non-redundant 3D (batched) FFT.
    // TODO ADD TESTS!
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_transform_sym_v<REMAP, T>>>
    void symmetrize3D(const shared_t<T[]>& input, dim4_t input_strides,
                      const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                      const Symmetry& symmetry, float3_t shift,
                      float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        transform3D<REMAP>(input, input_strides, output, output_strides, shape, float33_t{}, symmetry,
                           shift, cutoff, interp_mode, normalize, stream);
    }
}
