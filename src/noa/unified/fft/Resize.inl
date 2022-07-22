#pragma once

#ifndef NOA_UNIFIED_RESIZE_
#error "This is a private header"
#endif

#include "noa/cpu/fft/Resize.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Resize.h"
#endif

namespace noa::fft {
    template<Remap REMAP, typename T, typename>
    void resize(const Array<T>& input, size4_t input_shape, const Array<T>& output, size4_t output_shape) {
        using enum_t = std::underlying_type_t<Layout>;
        constexpr auto REMAP_ = static_cast<enum_t>(REMAP);
        constexpr bool IS_SRC_FULL = REMAP_ & Layout::SRC_FULL;
        constexpr bool IS_DST_FULL = REMAP_ & Layout::DST_FULL;
        static_assert(IS_SRC_FULL == IS_DST_FULL);
        // TODO: HC2HC, H2HC, HC2H, FC2FC, F2FC and FC2F could be useful to have and should be simple to add.

        NOA_CHECK(all(input.shape() == (IS_SRC_FULL ? input_shape : input_shape.fft())),
                  "Given the {} remap, the input FFT is expected to have a physical shape of {}, but got {}",
                  REMAP, IS_SRC_FULL ? input_shape : input_shape.fft(), input.shape());
        NOA_CHECK(all(output.shape() == (IS_SRC_FULL ? output_shape : output_shape.fft())),
                  "Given the {} remap, the output FFT is expected to have a physical shape of {}, but got {}",
                  REMAP, IS_SRC_FULL ? output_shape : output_shape.fft(), output.shape());

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
    Array<T> resize(const Array<T>& input, size4_t input_shape, size4_t output_shape) {
        using enum_t = std::underlying_type_t<Layout>;
        constexpr auto REMAP_ = static_cast<enum_t>(REMAP);
        Array<T> output(REMAP_ & Layout::DST_FULL ? output_shape : output_shape.fft(), input.options());
        resize(input, input_shape, output, output_shape);
        return output;
    }
}
