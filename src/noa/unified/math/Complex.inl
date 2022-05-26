#pragma once

#ifndef NOA_UNIFIED_COMPLEX_
#error "This is a private header"
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

        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, output_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output_shape);
        }

        const Device device = real.device();
        NOA_CHECK(device == input.device() && device == imag.device(),
                  "The input and output arrays must be on the same device, but got input:{}, real:{} and imag:{}",
                  input.device(), device, imag.device());

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::decompose(input.share(), input_stride,
                                 real.share(), real.stride(),
                                 imag.share(), imag.stride(),
                                 output_shape, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::decompose(input.share(), input_stride,
                                  real.share(), real.stride(),
                                  imag.share(), imag.stride(),
                                  output_shape, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void real(const Array<Complex<T>>& input, const Array<T>& real) {
        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, real.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), real.shape());
        }

        const Device device = real.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and real:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::real(input.share(), input_stride,
                            real.share(), real.stride(),
                            real.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::real(input.share(), input_stride,
                             real.share(), real.stride(),
                             real.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void imag(const Array<Complex<T>>& input, const Array<T>& imag) {
        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, imag.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), imag.shape());
        }

        const Device device = imag.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and imag:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::imag(input.share(), input_stride,
                            imag.share(), imag.stride(),
                            imag.shape(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::imag(input.share(), input_stride,
                             imag.share(), imag.stride(),
                             imag.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void complex(const Array<T>& real, const Array<T>& imag, const Array<Complex<T>>& output) {
        const size4_t output_shape = output.shape();
        size4_t real_stride = real.stride();
        if (!indexing::broadcast(real.shape(), real_stride, output_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      real.shape(), output_shape);
        }
        size4_t imag_stride = imag.stride();
        if (!indexing::broadcast(imag.shape(), imag_stride, output_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      imag.shape(), output_shape);
        }

        const Device device = output.device();
        NOA_CHECK(device == real.device() && device == imag.device(),
                  "The input and output arrays must be on the same device, but got real:{}, imag:{} and output:{}",
                  real.device(), imag.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::complex(real.share(), real_stride,
                               imag.share(), imag_stride,
                               output.shape(), output.stride(),
                               output_shape, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::math::complex(real.share(), real_stride,
                                imag.share(), imag_stride,
                                output.shape(), output.stride(),
                                output_shape, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}