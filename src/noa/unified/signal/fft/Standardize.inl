#pragma once

#ifndef NOA_UNIFIED_STANDARDIZE_
#error "This is a private header"
#endif

#include "noa/cpu/signal/fft/Standardize.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/Standardize.h"
#endif

namespace noa::signal::fft {
    template<Remap REMAP, typename T, typename>
    void standardize(const Array<T>& input, const Array<T>& output, size4_t shape, Norm norm) {
        constexpr bool IS_FULL = REMAP == Remap::F2F || REMAP == Remap::FC2FC;
        const size4_t actual_shape = IS_FULL ? shape : shape.fft();
        NOA_CHECK(all(input.shape() == actual_shape) && all(output.shape() == actual_shape),
                  "The input {} and output {} {}redundant FFTs don't match the expected logical shape {}",
                  input.shape(), output.shape(), IS_FULL ? "" : "non-", actual_shape);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::standardize<REMAP>(input.share(), input.strides(),
                                                 output.share(), output.strides(),
                                                 shape, norm, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::standardize<REMAP>(input.share(), input.strides(),
                                                  output.share(), output.strides(),
                                                  shape, norm, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
