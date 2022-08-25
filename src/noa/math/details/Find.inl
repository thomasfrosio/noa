#pragma once

#ifndef NOA_UNIFIED_MATH_FIND_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/math/Find.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Find.h"
#endif

namespace noa::math {
    template<typename S, typename T, typename U, typename>
    void find(S searcher, const Array<T>& input, const Array<U>& offsets, bool batch, bool swap_layout) {
        const Device device = input.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(offsets.dereferenceable(), "The output offsets should be accessible to the CPU");
            if (offsets.device() != device)
                offsets.eval();
            cpu::math::find(searcher, input.share(), input.strides(), input.shape(),
                            offsets.share(), batch, swap_layout, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::find(searcher, input.share(), input.strides(), input.shape(),
                             offsets.share(), batch, swap_layout, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename offset_t, typename S, typename T, typename>
    offset_t find(S searcher, const Array<T>& input, bool swap_layout) {
        const Device device = input.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            return cpu::math::find(searcher, input.share(), input.strides(), input.shape(), swap_layout, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::find(searcher, input.share(), input.strides(), input.shape(), swap_layout, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
