#pragma once

#ifndef NOA_UNIFIED_COMPLEX_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/math/Complex.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Complex.h"
#endif

namespace noa::math {
    template<typename T, typename>
    void decompose(const Array<Complex<T>>& input, const Array<T>& real, const Array<T>& imag) {
        const size4_t output_shape = real.shape();
        NOA_CHECK(all(output_shape == imag.shape()),
                  "The real and imaginary arrays should have the same shape, but got real:{} and imag:{}",
                  output_shape, imag.shape());

        size4_t input_strides = input.strides();
        if (!indexing::broadcast(input.shape(), input_strides, output_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output_shape);
        }

        const Device device = real.device();
        NOA_CHECK(device == input.device() && device == imag.device(),
                  "The input and output arrays must be on the same device, but got input:{}, real:{} and imag:{}",
                  input.device(), device, imag.device());

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::decompose(input.share(), input_strides,
                                 real.share(), real.strides(),
                                 imag.share(), imag.strides(),
                                 output_shape, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::decompose(input.share(), input_strides,
                                  real.share(), real.strides(),
                                  imag.share(), imag.strides(),
                                  output_shape, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void real(const Array<Complex<T>>& input, const Array<T>& real) {
        size4_t input_strides = input.strides();
        if (!indexing::broadcast(input.shape(), input_strides, real.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), real.shape());
        }

        const Device device = real.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and real:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::real(input.share(), input_strides,
                            real.share(), real.strides(),
                            real.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::real(input.share(), input_strides,
                             real.share(), real.strides(),
                             real.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void imag(const Array<Complex<T>>& input, const Array<T>& imag) {
        size4_t input_strides = input.strides();
        if (!indexing::broadcast(input.shape(), input_strides, imag.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), imag.shape());
        }

        const Device device = imag.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and imag:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::imag(input.share(), input_strides,
                            imag.share(), imag.strides(),
                            imag.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::imag(input.share(), input_strides,
                             imag.share(), imag.strides(),
                             imag.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void complex(const Array<T>& real, const Array<T>& imag, const Array<Complex<T>>& output) {
        const size4_t output_shape = output.shape();
        size4_t real_strides = real.strides();
        if (!indexing::broadcast(real.shape(), real_strides, output_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      real.shape(), output_shape);
        }
        size4_t imag_strides = imag.strides();
        if (!indexing::broadcast(imag.shape(), imag_strides, output_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      imag.shape(), output_shape);
        }

        const Device device = output.device();
        NOA_CHECK(device == real.device() && device == imag.device(),
                  "The input and output arrays must be on the same device, but got real:{}, imag:{} and output:{}",
                  real.device(), imag.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::complex(real.share(), real_strides,
                               imag.share(), imag_strides,
                               output.shape(), output.strides(),
                               output_shape, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::complex(real.share(), real_strides,
                                imag.share(), imag_strides,
                                output.shape(), output.strides(),
                                output_shape, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}