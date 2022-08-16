#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Exception.h"
#include "noa/common/geometry/Symmetry.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry {
    using Symmetry = ::noa::geometry::Symmetry;

    // Symmetrizes the 2D (batched) input array.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void symmetrize2D(const shared_t<T[]>& input, size4_t input_strides,
                      const shared_t<T[]>& output, size4_t output_strides,
                      size4_t shape, const Symmetry& symmetry, float2_t center,
                      InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream);

    // Symmetrizes the 3D (batched) input array.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void symmetrize3D(const shared_t<T[]>& input, size4_t input_strides,
                      const shared_t<T[]>& output, size4_t output_strides,
                      size4_t shape, const Symmetry& symmetry, float3_t center,
                      InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream);
}
