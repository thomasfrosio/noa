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
        const auto remap_ = static_cast<uint8_t>(remap);
        const bool is_src_full = remap_ & Layout::SRC_FULL;
        const bool is_dst_full = remap_ & Layout::DST_FULL;

        NOA_CHECK(all(input.shape() == (is_src_full ? shape : shape.fft())),
                  "Given the {} remap, the input FFT is expected to have a physical shape of {}, but got {}",
                  remap, is_src_full ? shape : shape.fft(), input.shape());
        NOA_CHECK(all(output.shape() == (is_dst_full ? shape : shape.fft())),
                  "Given the {} remap, the output FFT is expected to have a physical shape of {}, but got {}",
                  remap, is_dst_full ? shape : shape.fft(), output.shape());

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::fft::remap(remap, input.share(), input.stride(),
                            output.share(), output.stride(),
                            shape, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::fft::remap(remap, input.share(), input.stride(),
                             output.share(), output.stride(),
                             shape, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    Array<T> remap(Remap remap, const Array<T>& input, size4_t shape) {
        using enum_t = std::underlying_type_t<Layout>;
        const size4_t output_shape = static_cast<enum_t>(remap) & Layout::DST_FULL ? shape : shape.fft();
        Array<T> output(shape, input.options());
        remap(remap, input, output, shape);
        return output;
    }
}
