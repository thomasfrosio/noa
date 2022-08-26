#pragma once

#ifndef NOA_UNIFIED_RESIZE_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/memory/Resize.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Resize.h"
#endif

namespace noa::memory {
    std::pair<int4_t, int4_t> borders(size4_t input_shape, size4_t output_shape) {
        return cpu::memory::borders(input_shape, output_shape);
    }

    template<typename T, typename>
    void resize(const Array<T>& input, const Array<T>& output,
                int4_t border_left, int4_t border_right,
                BorderMode border_mode, T border_value) {
        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);
        NOA_CHECK(all(int4_t(output.shape()) == int4_t(input.shape()) + border_left + border_right),
                  "The output shape {} does not math the expected shape (input:{}, left:{}, right:{})",
                  output.shape(), input.shape(), border_left, border_right);
        NOA_CHECK(input.get() != output.get(), "In-place resizing is not allowed");

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::resize(input.share(), input.strides(), input.shape(),
                                border_left, border_right,
                                output.share(), output.strides(),
                                border_mode, border_value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::resize(input.share(), input.strides(), input.shape(),
                                 border_left, border_right,
                                 output.share(), output.strides(),
                                 border_mode, border_value, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    Array<T> resize(const Array<T>& input,
                    int4_t border_left, int4_t border_right,
                    BorderMode border_mode, T border_value) {
        const int4_t output_shape = int4_t(input.shape()) + border_left + border_right;
        NOA_CHECK(all(output_shape > 0),
                  "Cannot resize [left:{}, right:{}] an array of shape {} into an array of shape {}",
                  border_left, border_right, input.shape(), output_shape);
        Array<T> output(size4_t(output_shape), input.options());
        resize(input, output, border_left, border_right, border_mode, border_value);
        return output;
    }

    template<typename T, typename>
    void resize(const Array<T>& input, const Array<T>& output,
                BorderMode border_mode, T border_value) {
        auto[border_left, border_right] = borders(input.shape(), output.shape());
        resize(input, output, border_left, border_right, border_mode, border_value);
    }
}
