#pragma once

#ifndef NOA_UNIFIED_BANDPASS_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/signal/fft/Bandpass.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/Bandpass.h"
#endif

namespace noa::signal::fft {
    using noa::fft::Remap;

    template<Remap REMAP, typename T, typename>
    void lowpass(const Array<T>& input, const Array<T>& output, size4_t shape, float cutoff, float width) {
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        size4_t input_strides = input.strides();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          input.shape(), output.shape());
            }
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), device);
            NOA_CHECK(REMAP == Remap::H2H || REMAP == Remap::HC2HC || input.get() != output.get(),
                      "In-place remapping is not allowed");
        }

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::lowpass<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(),
                    shape, cutoff, width, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::lowpass<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(),
                    shape, cutoff, width, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void highpass(const Array<T>& input, const Array<T>& output, size4_t shape, float cutoff, float width) {
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        size4_t input_strides = input.strides();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          input.shape(), output.shape());
            }
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), device);
            NOA_CHECK(REMAP == Remap::H2H || REMAP == Remap::HC2HC || input.get() != output.get(),
                      "In-place remapping is not allowed");
        }

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::highpass<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(),
                    shape, cutoff, width, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::highpass<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(),
                    shape, cutoff, width, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void bandpass(const Array<T>& input, const Array<T>& output, size4_t shape,
                  float cutoff1, float cutoff2, float width1, float width2) {
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        size4_t input_strides = input.strides();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          input.shape(), output.shape());
            }
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), device);
            NOA_CHECK(REMAP == Remap::H2H || REMAP == Remap::HC2HC || input.get() != output.get(),
                      "In-place remapping is not allowed");
        }

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::bandpass<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(),
                    shape, cutoff1, cutoff2, width1, width2, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::bandpass<REMAP>(
                    input.share(), input_strides,
                    output.share(), output.strides(),
                    shape, cutoff1, cutoff2, width1, width2, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
