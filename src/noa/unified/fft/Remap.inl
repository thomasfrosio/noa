#pragma once

#ifndef NOA_UNIFIED_REMAP_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/fft/Remap.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Remap.h"
#endif

namespace noa::fft {
    template<typename T, typename>
    void remap(Remap remap, const Array<T>& input, const Array<T>& output, dim4_t shape) {
        const auto remap_ = static_cast<uint8_t>(remap);
        const bool is_src_full = remap_ & Layout::SRC_FULL;
        const bool is_dst_full = remap_ & Layout::DST_FULL;

        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");
        NOA_CHECK(all(input.shape() == (is_src_full ? shape : shape.fft())),
                  "Given the {} remap, the input FFT is expected to have a physical shape of {}, but got {}",
                  remap, is_src_full ? shape : shape.fft(), input.shape());
        NOA_CHECK(all(output.shape() == (is_dst_full ? shape : shape.fft())),
                  "Given the {} remap, the output FFT is expected to have a physical shape of {}, but got {}",
                  remap, is_dst_full ? shape : shape.fft(), output.shape());

        NOA_CHECK(!indexing::isOverlap(input, output) ||
                  (remap == fft::H2HC &&
                   input.get() == output.get() &&
                   (shape[2] == 1 || !(shape[2] % 2)) &&
                   (shape[1] == 1 || !(shape[1] % 2))),
                  "In-place remapping is only available with {} and when the depth and height dimensions "
                  "have an even number of elements, but got remap {} and shape {}", fft::H2HC, remap, shape);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::fft::remap(remap, input.share(), input.strides(),
                            output.share(), output.strides(),
                            shape, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::fft::remap(remap, input.share(), input.strides(),
                             output.share(), output.strides(),
                             shape, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    [[nodiscard]] Array<T> remap(Remap remap, const Array<T>& input, dim4_t shape) {
        using enum_t = std::underlying_type_t<Layout>;
        const dim4_t output_shape = static_cast<enum_t>(remap) & Layout::DST_FULL ? shape : shape.fft();
        Array<T> output(output_shape, input.options());
        fft::remap(remap, input, output, shape);
        return output;
    }
}
