#pragma once

#ifndef NOA_UNIFIED_TRANSPOSE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/memory/Permute.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Permute.h"
#endif

namespace noa::memory {
    template<typename T, typename>
    void permute(const Array<T>& input, const Array<T>& output, dim4_t permutation) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");

        dim4_t input_strides = input.strides();
        dim4_t input_shape = input.shape();
        for (dim_t i = 0; i < 4; ++i) {
            const dim_t d = permutation[i];
            if (input.shape()[d] == 1 && output.shape()[i] != 1) {
                input_strides[d] = 0; // broadcast this dimension
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
            cpu::memory::permute(input.share(), input_strides, input_shape,
                                 output.share(), output.strides(),
                                 permutation, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::permute(input.share(), input_strides, input_shape,
                                  output.share(), output.strides(),
                                  permutation, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
