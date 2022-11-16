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
        NOA_CHECK(!input.empty(), "Empty array detected");

        [[maybe_unused]] const dim_t required_size = batch ? input.shape()[0] : 1; // TODO >=1 ?
        NOA_CHECK(indexing::isVector(offsets.shape()) && offsets.contiguous() && offsets.elements() == required_size,
                   "The output offsets should be specified as a contiguous vector of {} elements, "
                   "but got strides:{} and shape:{}", required_size, offsets.strides(), offsets.shape());

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
    [[nodiscard]] offset_t find(S searcher, const Array<T>& input, bool swap_layout) {
        NOA_CHECK(!input.empty(), "Empty array detected");

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
