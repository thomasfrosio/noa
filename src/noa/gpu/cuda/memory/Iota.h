#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory {
    // Returns a tiled sequence [0, elements).
    template<typename T, typename = std::enable_if_t<traits::is_restricted_scalar_v<T>>>
    void iota(const shared_t<T[]>& src, dim_t elements, dim_t tile, Stream& stream);

    // Returns a tiled sequence [0, elements), in the rightmost order.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_scalar_v<T>>>
    void iota(const shared_t<T[]>& src, dim4_t strides, dim4_t shape, dim4_t tile, Stream& stream);
}
