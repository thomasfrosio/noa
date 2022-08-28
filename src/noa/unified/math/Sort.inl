#pragma once

#ifndef NOA_UNIFIED_SORT_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/math/Sort.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Sort.h"
#endif

namespace noa::math {
    template<typename T, typename>
    void sort(const Array<T>& array, bool ascending, int axis) {
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::sort(array.share(), array.strides(), array.shape(), ascending, axis, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::math::details::is_valid_sort_v<T>)
                return cuda::math::sort(array.share(), array.strides(), array.shape(), ascending, axis, stream.cuda());
            else
                NOA_THROW("The CUDA backend doesn't support this type: {}");
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
