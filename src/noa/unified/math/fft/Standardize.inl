#pragma once

#ifndef NOA_UNIFIED_STANDARDIZE_
#error "This is a private header"
#endif

#include "noa/cpu/math/fft/Standardize.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/fft/Standardize.h"
#endif

namespace noa::math::fft {
    template<Remap REMAP, typename T, typename>
    void standardize(const Array<T>& input, const Array<T>& output, size4_t shape, Norm norm) {
        if constexpr (REMAP == Remap::F2F || REMAP == Remap::FC2FC) {
            NOA_CHECK(all(input.shape() == shape) && all(output.shape() == shape),
                      "The input {} and output {} redundant FFTs don't match the expected logical shape {}",
                      input.shape(), output.shape(), shape);
        } else {
            NOA_CHECK(all(input.shape() == shape.fft()) && all(output.shape() == shape.fft()),
                      "The input {} and output {} non-redundant FFTs don't match the expected physical shape {}",
                      input.shape(), output.shape(), shape.fft());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::fft::standardize<REMAP>(input.share(), input.stride(),
                                               output.share(), output.stride(),
                                               shape, norm, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::fft::standardize<REMAP>(input.share(), input.stride(),
                                                output.share(), output.stride(),
                                                shape, norm, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
