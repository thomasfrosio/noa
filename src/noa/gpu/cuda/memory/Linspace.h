#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory {
    template<typename T>
    std::tuple<size_t, T, T> linspaceStep(size_t elements, T start, T stop, bool endpoint = true) {
        const size_t count = elements - static_cast<size_t>(endpoint);
        const T delta = stop - start;
        const T step = delta / static_cast<T>(count);
        return {count, delta, step};
    }

    // Returns evenly spaced values within a given interval.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T> && !traits::is_bool_v<T>>>
    T linspace(const shared_t<T[]>& src, size_t elements,
               T start, T stop, bool endpoint, Stream& stream);

    // Returns evenly spaced values within a given interval, in the rightmost order.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T> && !traits::is_bool_v<T>>>
    T linspace(const shared_t<T[]>& src, size4_t strides, size4_t shape,
               T start, T stop, bool endpoint, Stream& stream);

}
