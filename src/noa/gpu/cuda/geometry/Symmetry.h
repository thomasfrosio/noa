#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/common/geometry/Symmetry.h"

namespace noa::cuda::geometry {
    using Symmetry = ::noa::geometry::Symmetry;

    // Symmetrizes the 2D (batched) input array.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void symmetrize2D(const shared_t<T[]>& input, dim4_t input_strides,
                      const shared_t<T[]>& output, dim4_t output_strides,
                      dim4_t shape, const Symmetry& symmetry, float2_t center,
                      InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream);

    // Symmetrizes the 2D texture.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void symmetrize2D(const shared_t<cudaArray>& array,
                      const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                      const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                      const Symmetry& symmetry, float2_t center, bool normalize, Stream& stream);

    // Symmetrizes the 3D (batched) input array.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void symmetrize3D(const shared_t<T[]>& input, dim4_t input_strides,
                      const shared_t<T[]>& output, dim4_t output_strides,
                      dim4_t shape, const Symmetry& symmetry, float3_t center,
                      InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream);

    // Symmetrizes the 3D texture.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void symmetrize3D(const shared_t<cudaArray>& array,
                      const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                      const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                      const Symmetry& symmetry, float3_t center, bool normalize, Stream& stream);
}
