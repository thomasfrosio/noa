#pragma once

#ifndef NOA_UNIFIED_TRANSFORM_
#error "This is a private header"
#endif

#include "noa/cpu/fft/Transforms.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Transforms.h"
#endif

namespace noa::fft {
    size_t nextFastSize(size_t size) {
        #ifdef NOA_ENABLE_CUDA
        return noa::cuda::fft::fastSize(size);
        #else
        return noa::cpu::fft::fastSize(size);
        #endif
    }

    template<typename T>
    Int4<T> nextFastShape(Int4<T> shape) {
        #ifdef NOA_ENABLE_CUDA
        return noa::cuda::fft::fastShape(shape);
        #else
        return noa::cpu::fft::fastShape(shape);
        #endif
    }

    template<typename T, typename>
    Array<T> alias(const Array<Complex<T>>& input, size4_t shape) {
        Array<T> tmp = input.template as<T>();
        return Array<T>(tmp.share(), shape, tmp.stride(), tmp.options());
    }

    template<typename T, typename>
    void r2c(const Array<T>& input, const Array<Complex<T>>& output, Norm norm) {
        NOA_CHECK(all(output.shape() == input.shape().fft()),
                  "Given the real input with a shape of {}, the non-redundant shape of the complex output "
                  "should be {}, but got {}", input.shape(), input.shape().fft(), output.shape());

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::fft::r2c(input.share(), input.stride(),
                          output.share(), output.stride(),
                          input.shape(), cpu::fft::ESTIMATE | cpu::fft::PRESERVE_INPUT,
                          norm, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::fft::r2c(input.share(), input.stride(),
                           output.share(), output.stride(),
                           input.shape(), norm, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    Array<Complex<T>> r2c(const Array<T>& input, Norm norm) {
        Array<Complex<T>> output(input.shape().fft(), input.options());
        r2c(input, output, norm);
        return output;
    }

    template<typename T, typename>
    void c2r(const Array<Complex<T>>& input, const Array<T>& output, Norm norm) {
        NOA_CHECK(all(input.shape() == output.shape().fft()),
                  "Given the real output with a shape of {}, the non-redundant shape of the complex input "
                  "should be {}, but got {}", output.shape(), output.shape().fft(), input.shape());

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::fft::c2r(input.share(), input.stride(),
                          output.share(), output.stride(),
                          output.shape(), cpu::fft::ESTIMATE,
                          norm, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::fft::c2r(input.share(), input.stride(),
                           output.share(), output.stride(),
                           output.shape(), norm, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    Array<T> c2r(const Array<Complex<T>>& input, size4_t shape, Norm norm) {
        Array<T> output(shape, input.options());
        c2r(input, output, norm);
        return output;
    }

    template<typename T, typename>
    void c2c(const Array<Complex<T>>& input, const Array<Complex<T>>& output, Sign sign, Norm norm) {
        NOA_CHECK(all(input.shape() == output.shape()),
                  "The input and output shape should match (no broadcasting allowed), but got input {} and output {}",
                  input.shape(), output.shape());

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::fft::c2c(input.share(), input.stride(),
                          output.share(), output.stride(),
                          input.shape(), sign, cpu::fft::ESTIMATE | cpu::fft::PRESERVE_INPUT,
                          norm, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::fft::c2c(input.share(), input.stride(),
                           output.share(), output.stride(),
                           input.shape(), sign, norm, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    Array<Complex<T>> c2c(const Array<Complex<T>>& input, Sign sign, Norm norm) {
        Array<Complex<T>> output(input.shape(), input.options());
        c2c(input, output, sign, norm);
        return output;
    }
}
