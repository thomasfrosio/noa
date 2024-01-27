#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/cpu/fft/Plan.hpp"

namespace noa::cpu::fft {
    template<typename T>
    void r2c(T* input, Complex<T>* output, const Shape4<i64>& shape, u32 flag, i64 n_threads) {
        Plan(input, output, shape, flag, n_threads).execute();
    }

    template<typename T>
    void r2c(
            T* input, const Strides4<i64>& input_strides,
            Complex<T>* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, u32 flag, i64 n_threads
    ) {
        Plan(input, input_strides, output, output_strides, shape, flag, n_threads).execute();
    }

    template<typename T>
    void r2c(T* data, const Shape4<i64>& shape, u32 flag, i64 n_threads) {
        r2c(data, reinterpret_cast<Complex<T>*>(data), shape, flag, n_threads);
    }

    template<typename T>
    void r2c(T* data, const Strides4<i64>& strides, const Shape4<i64>& shape, u32 flag, i64 n_threads) {
        // Since it is in-place, the physical width (in real elements):
        //  1: is even, since complex elements take 2 real elements.
        //  2: has at least 1 (if odd) or 2 (if even) extract real element.
        NOA_ASSERT(!(strides.physical_shape()[2] % 2));
        NOA_ASSERT(strides.physical_shape()[2] >= shape[3] + 1 + static_cast<i64>(!(shape[3] % 2)));

        const auto complex_strides = Strides4<i64>{strides[0] / 2, strides[1] / 2, strides[2] / 2, strides[3]};
        r2c(data, strides, reinterpret_cast<Complex<T>*>(data), complex_strides, shape, flag, n_threads);
    }

    template<typename T>
    void c2r(Complex<T>* input, T* output, const Shape4<i64>& shape, u32 flag, i64 n_threads) {
        Plan(input, output, shape, flag, n_threads).execute();
    }

    template<typename T>
    void c2r(
            Complex<T>* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, u32 flag, i64 n_threads
    ) {
        Plan(input, input_strides, output, output_strides, shape, flag, n_threads).execute();
    }

    template<typename T>
    void c2r(Complex<T>* data, const Shape4<i64>& shape, u32 flag, i64 n_threads) {
        c2r(data, reinterpret_cast<T*>(data), shape, flag, n_threads);
    }

    template<typename T>
    void c2r(Complex<T>* data, const Strides4<i64>& strides, const Shape4<i64>& shape, u32 flag, i64 n_threads) {
        const auto real_strides = Strides4<i64>{strides[0] * 2, strides[1] * 2, strides[2] * 2, strides[3]};
        c2r(data, strides, reinterpret_cast<T*>(data), real_strides, shape, flag, n_threads);
    }

    template<typename T>
    void c2c(
            Complex<T>* input, Complex<T>* output, const Shape4<i64>& shape,
            noa::fft::Sign sign, u32 flag, i64 n_threads
    ) {
        Plan(input, output, shape, sign, flag, n_threads).execute();
    }

    template<typename T>
    void c2c(
            Complex<T>* input, const Strides4<i64>& input_strides,
            Complex<T>* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, noa::fft::Sign sign, u32 flag, i64 n_threads
    ) {
        Plan(input, input_strides, output, output_strides, shape, sign, flag, n_threads).execute();
    }

    template<typename T>
    void c2c(Complex<T>* data, const Shape4<i64>& shape, noa::fft::Sign sign, u32 flag, i64 n_threads) {
        c2c(data, data, shape, sign, flag, n_threads);
    }

    template<typename T>
    void c2c(
            Complex<T>* data, const Strides4<i64>& strides, const Shape4<i64>& shape,
            noa::fft::Sign sign, u32 flag, i64 n_threads
    ) {
        c2c(data, strides, data, strides, shape, sign, flag, n_threads);
    }
}
#endif
