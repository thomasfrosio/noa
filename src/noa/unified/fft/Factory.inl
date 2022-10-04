#pragma once

#ifndef NOA_FFT_FACTORY_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/fft/Transforms.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Transforms.h"
#endif

namespace noa::fft {
    dim_t nextFastSize(dim_t size) {
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
    Array<T> alias(const Array<Complex<T>>& input, dim4_t shape) {
        NOA_CHECK(all(input.shape() == shape.fft()),
                  "Given the {} logical shape, the non-redundant input should have a shape of {}, but got {}",
                  shape, input.shape(), shape.fft());
        Array<T> tmp = input.template as<T>();
        return Array<T>(tmp.share(), shape, tmp.strides(), tmp.options());
    }

    template<typename T, typename>
    std::pair<Array<T>, Array<Complex<T>>> zeros(dim4_t shape, ArrayOption option) {
        Array out1 = memory::zeros<Complex<T>>(shape.fft(), option);
        Array out0 = fft::alias(out1, shape);
        return {out0, out1};
    }

    template<typename T, typename>
    std::pair<Array<T>, Array<Complex<T>>> ones(dim4_t shape, ArrayOption option) {
        Array out1 = memory::ones<Complex<T>>(shape.fft(), option);
        Array out0 = fft::alias(out1, shape);
        return {out0, out1};
    }

    template<typename T, typename>
    std::pair<Array<T>, Array<Complex<T>>> empty(dim4_t shape, ArrayOption option) {
        Array out1 = memory::empty<Complex<T>>(shape.fft(), option);
        Array out0 = fft::alias(out1, shape);
        return {out0, out1};
    }
}
