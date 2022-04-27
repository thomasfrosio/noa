#pragma once

#ifndef NOA_UNIFIED_REMAP_
#error "This is a private header"
#endif

#include "noa/cpu/fft/Remap.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Remap.h"
#endif

namespace noa::fft {
    template<typename T, typename>
    void remap(Remap remap, const Array<T>& input, const Array<T>& output, size4_t shape) {
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "The non-redundant FFT with a shape of [logical:{}, pitch:{}] is expected, but got pitch of {}",
                  shape, shape.fft(), output.shape());

        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::fft::remap(remap, input.share(), input_stride,
                            output.share(), output.stride(),
                            shape, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::fft::remap(remap, input.share(), input_stride,
                             output.share(), output.stride(),
                             shape, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
