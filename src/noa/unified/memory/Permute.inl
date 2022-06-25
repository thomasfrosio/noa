#pragma once

#ifndef NOA_UNIFIED_TRANSPOSE_
#error "This is an internal header"
#endif

#include "noa/cpu/memory/Permute.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Permute.h"
#endif

namespace noa::memory {
    template<typename T, typename>
    void permute(const Array<T>& input, const Array<T>& output, uint4_t permutation) {
        size4_t input_stride = input.stride();
        size4_t input_shape = input.shape();
        for (size_t i = 0; i < 4; ++i) {
            const size_t d = permutation[i];
            if (input.shape()[d] == 1 && output.shape()[i] != 1) {
                input_stride[d] = 0; // broadcast this dimension
                input_shape[d] = output.shape()[i];
            } else if (input.shape()[d] != output.shape()[i]) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          indexing::reorder(input.shape(), permutation), output.shape());
            }
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::permute(input.share(), input_stride, input_shape,
                                 output.share(), output.stride(),
                                 permutation, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::permute(input.share(), input_stride, input_shape,
                                  output.share(), output.stride(),
                                  permutation, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
