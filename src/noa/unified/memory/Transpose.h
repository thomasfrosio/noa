#pragma once

#include "noa/cpu/memory/Transpose.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Transpose.h"
#endif

#include "noa/unified/Array.h"

namespace noa::memory {
    /// Transposes, in memory, the axes of an array.
    /// \tparam T           Any data type.
    /// \param[in] input    Array to permute.
    /// \param[out] output  Permuted array. Its shape and stride should be permuted already.
    /// \param permutation  Rightmost permutation. Axes are numbered from 0 to 3, 3 being the innermost dimension.
    /// \note For in-place permutations, only 0123, 0213, 0132, and 0321 are supported. Anything else throws an error.
    /// \note The in-place 0213 permutation requires the axis 1 and 2 to have the same size.
    ///       The in-place 0132 permutation requires the axis 3 and 2 to have the same size.
    ///       The in-place 0321 permutation requires the axis 3 and 1 to have the same size.
    template<typename T, typename = std::enable_if_t<noa::traits::is_data_v<T>>>
    void transpose(const Array<T>& input, const Array<T>& output, uint4_t permutation) {
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

        const Device device{output.device()};
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::transpose(input.share(), input_stride, input_shape,
                                   output.share(), output.stride(),
                                   permutation, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::transpose(input.share(), input_stride, input_shape,
                                    output.share(), output.stride(),
                                    permutation, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
