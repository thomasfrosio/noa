#pragma once

#ifndef NOA_UNIFIED_MEDIAN_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/signal/Median.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/Median.h"
#endif

namespace noa::signal {
    template<typename T, typename>
    void median1(const Array<T>& input, const Array<T>& output,
                 dim_t window_size, BorderMode border_mode) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");
        NOA_CHECK(!indexing::isOverlap(input, output), "The input and output array should not overlap");

        dim4_t input_strides = input.strides();
        if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        NOA_CHECK(border_mode == BORDER_ZERO || output.shape()[3] >= window_size / 2 + 1,
                  "With BORDER_REFLECT and a window of {}, the width should be >= than {}, but got {}",
                  window_size, window_size / 2 + 1, output.shape()[3]);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::median1(input.share(), input_strides,
                                 output.share(), output.strides(), output.shape(),
                                 border_mode, window_size, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::median1(input.share(), input_strides,
                                  output.share(), output.strides(), output.shape(),
                                  border_mode, window_size, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void median2(const Array<T>& input, const Array<T>& output,
                 dim_t window_size, BorderMode border_mode) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");
        NOA_CHECK(!indexing::isOverlap(input, output), "The input and output array should not overlap");

        dim4_t input_strides = input.strides();
        if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        NOA_CHECK(border_mode == BORDER_ZERO ||
                  (output.shape()[3] >= window_size / 2 + 1 && output.shape()[2] >= window_size / 2 + 1),
                  "With BORDER_REFLECT and a window of {}, the height and width should be >= than {}, but got ({}, {})",
                  window_size, window_size / 2 + 1, output.shape()[2], output.shape()[3]);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::median2(input.share(), input_strides,
                                 output.share(), output.strides(), output.shape(),
                                 border_mode, window_size, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::median2(input.share(), input_strides,
                                  output.share(), output.strides(), output.shape(),
                                  border_mode, window_size, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void median3(const Array<T>& input, const Array<T>& output,
                 dim_t window_size, BorderMode border_mode) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");
        NOA_CHECK(!indexing::isOverlap(input, output), "The input and output array should not overlap");

        dim4_t input_strides = input.strides();
        if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        NOA_CHECK(border_mode == BORDER_ZERO || all(dim3_t(output.shape().get(1)) >= window_size / 2 + 1),
                  "With BORDER_REFLECT and a window of {}, the depth, height and width should be >= than {}, but got {}",
                  window_size, window_size / 2 + 1, dim3_t(output.shape().get(1)));

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::median3(input.share(), input_strides,
                                 output.share(), output.strides(), output.shape(),
                                 border_mode, window_size, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::median3(input.share(), input_strides,
                                  output.share(), output.strides(), output.shape(),
                                  border_mode, window_size, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
