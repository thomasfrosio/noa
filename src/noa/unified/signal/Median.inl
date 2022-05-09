#pragma once

#ifndef NOA_UNIFIED_MEDIAN_
#error "This is a private header"
#endif

#include "noa/cpu/signal/Median.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/Median.h"
#endif

namespace noa::signal {
    template<typename T, typename>
    void median1(const Array<T>& input, const Array<T>& output,
                 size_t window_size, BorderMode border_mode) {
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
            cpu::signal::median1(input.share(), input_stride,
                                 output.share(), output.stride(), output.shape(),
                                 border_mode, window_size, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::median1(input.share(), input_stride,
                                  output.share(), output.stride(), output.shape(),
                                  border_mode, window_size, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void median2(const Array<T>& input, const Array<T>& output,
                 size_t window_size, BorderMode border_mode) {
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
            cpu::signal::median2(input.share(), input_stride,
                                 output.share(), output.stride(), output.shape(),
                                 border_mode, window_size, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::median2(input.share(), input_stride,
                                  output.share(), output.stride(), output.shape(),
                                  border_mode, window_size, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void median3(const Array<T>& input, const Array<T>& output,
                 size_t window_size, BorderMode border_mode) {
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
            cpu::signal::median3(input.share(), input_stride,
                                 output.share(), output.stride(), output.shape(),
                                 border_mode, window_size, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::median3(input.share(), input_stride,
                                  output.share(), output.stride(), output.shape(),
                                  border_mode, window_size, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
