#pragma once

#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::geometry::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_polar_xform_v = traits::is_any_v<T, float, cfloat_t> && REMAP == HC2FC;
}

namespace noa::cuda::geometry::fft {
    using Remap = noa::fft::Remap;

    // Transforms 2D FFT(s) to (log-)polar coordinates.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_polar_xform_v<REMAP, T>>>
    void cartesian2polar(const shared_t<T[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         const shared_t<T[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, InterpMode interp, Stream& stream);

    // Transforms 2D FFT(s) to (log-)polar coordinates.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void cartesian2polar(const shared_t<cudaArray>& array,
                         const shared_t<cudaTextureObject_t>& cartesian,
                         InterpMode cartesian_interp, dim2_t cartesian_shape,
                         const shared_t<T[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, Stream& stream);
}
