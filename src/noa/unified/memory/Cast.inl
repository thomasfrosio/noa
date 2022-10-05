#pragma once

#ifndef NOA_UNIFIED_CAST_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/memory/Cast.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Cast.h"
#endif

namespace noa::memory {
    template<typename T, typename U>
    void cast(const Array<T>& input, const Array<U>& output, bool clamp) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");
        NOA_CHECK(!indexing::isOverlap(input, output), "The input and output arrays should not overlap");

        dim4_t input_strides = input.strides();
        if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device(output.device());
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  device, input.device());

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::cast(input.share(), input_strides, output.share(), output.strides(),
                              output.shape(), clamp, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (cuda::memory::details::is_valid_cast_v<T, U>) {
                cuda::memory::cast(input.share(), input_strides, output.share(), output.strides(),
                                   output.shape(), clamp, stream.cuda());
            } else {
                // TODO Add nvrtc to support all types.
                NOA_THROW("This cast ({} -> {}) is not supported by the CUDA backend",
                          string::human<T>(), string::human<U>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
