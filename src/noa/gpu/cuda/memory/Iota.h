#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory {
    // Returns a tiled sequence [0, elements).
    template<typename T, typename = std::enable_if_t<traits::is_restricted_scalar_v<T>>>
    void iota(const shared_t<T[]>& src, size_t elements, size_t tile, Stream& stream);

    // Returns a tiled sequence [0, elements), in the rightmost order.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_scalar_v<T>>>
    void iota(const shared_t<T[]>& src, size4_t strides, size4_t shape, size4_t tile, Stream& stream);
}
