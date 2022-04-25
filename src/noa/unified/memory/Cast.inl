#pragma once

#ifndef NOA_UNIFIED_CAST_
#error "This is an internal header"
#endif

#include "noa/cpu/memory/Cast.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Cast.h"
#endif

namespace noa::memory {
    template<typename T, typename U>
    void cast(const Array<T>& input, const Array<U>& output, bool clamp) {
        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device{output.device()};
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  device, input.device());

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::cast(input.share(), input_stride, output.share(), output.stride(),
                              output.shape(), clamp, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::cast(input.share(), input_stride, output.share(), output.stride(),
                               output.shape(), clamp, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
