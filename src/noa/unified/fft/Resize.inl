#pragma once

#ifndef NOA_UNIFIED_RESIZE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/fft/Resize.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Resize.h"
#endif

// TODO H2HC, HC2H, F2FC and FC2F could be useful to have and should be simple to add.

namespace noa::fft {
    template<Remap REMAP, typename T, typename>
    void resize(const Array<T>& input, dim4_t input_shape, const Array<T>& output, dim4_t output_shape) {
        using enum_t = std::underlying_type_t<Layout>;
        constexpr auto REMAP_ = static_cast<enum_t>(REMAP);
        constexpr bool IS_FULL = REMAP_ & Layout::SRC_FULL;

        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");
        NOA_CHECK(!indexing::isOverlap(input, output), "Input and output arrays should not overlap");
        NOA_CHECK(input.shape()[0] == output.shape()[0], "The batch dimension cannot be resized");

        NOA_CHECK(all(input.shape() == (IS_FULL ? input_shape : input_shape.fft())),
                  "Given the {} remap, the input FFT is expected to have a physical shape of {}, but got {}",
                  REMAP, IS_FULL ? input_shape : input_shape.fft(), input.shape());
        NOA_CHECK(all(output.shape() == (IS_FULL ? output_shape : output_shape.fft())),
                  "Given the {} remap, the output FFT is expected to have a physical shape of {}, but got {}",
                  REMAP, IS_FULL ? output_shape : output_shape.fft(), output.shape());

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::fft::resize<REMAP>(input.share(), input.strides(), input_shape,
                                    output.share(), output.strides(), output_shape, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::fft::resize<REMAP>(input.share(), input.strides(), input_shape,
                                     output.share(), output.strides(), output_shape, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    Array<T> resize(const Array<T>& input, dim4_t input_shape, dim4_t output_shape) {
        using enum_t = std::underlying_type_t<Layout>;
        constexpr auto REMAP_ = static_cast<enum_t>(REMAP);
        Array<T> output(REMAP_ & Layout::DST_FULL ? output_shape : output_shape.fft(), input.options());
        fft::resize<REMAP>(input, input_shape, output, output_shape);
        return output;
    }
}
