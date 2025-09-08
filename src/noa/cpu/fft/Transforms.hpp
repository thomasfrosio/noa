#pragma once

#include "noa/cpu/fft/Plan.hpp"

namespace noa::cpu::fft {
    template<typename T>
    void r2c(
        T* input, const Strides4<i64>& input_strides,
        Complex<T>* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, u32 flag, bool plan_only, i64 n_threads
    ) {
        auto plan = Plan(input, input_strides, output, output_strides, shape, flag, n_threads);
        if (not plan_only)
            plan.execute();
    }

    template<typename T>
    void r2c(T* data, const Strides4<i64>& strides, const Shape4<i64>& shape, u32 flag, bool plan_only, i64 n_threads) {
        // Since it is in-place, the physical width (in real elements):
        //  1: is even, since complex elements take 2 real elements.
        //  2: has at least 1 (if odd) or 2 (if even) extra real-element.
        check(is_even(strides.physical_shape()[2]));
        check(strides.physical_shape()[2] >= shape[3] + 1 + static_cast<i64>(is_even(shape[3])));

        const auto complex_strides = Strides4<i64>{strides[0] / 2, strides[1] / 2, strides[2] / 2, strides[3]};
        r2c(data, strides, reinterpret_cast<Complex<T>*>(data), complex_strides, shape, flag, plan_only, n_threads);
    }

    template<typename T>
    void c2r(
        Complex<T>* input, const Strides4<i64>& input_strides,
        T* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, u32 flag, bool plan_only, i64 n_threads
    ) {
        auto plan = Plan(input, input_strides, output, output_strides, shape, flag, n_threads);
        if (not plan_only)
            plan.execute();
    }

    template<typename T>
    void c2r(Complex<T>* data, const Strides4<i64>& strides, const Shape4<i64>& shape, u32 flag, bool plan_only, i64 n_threads) {
        const auto real_strides = Strides4<i64>{strides[0] * 2, strides[1] * 2, strides[2] * 2, strides[3]};
        c2r(data, strides, reinterpret_cast<T*>(data), real_strides, shape, flag, plan_only, n_threads);
    }

    template<typename T>
    void c2c(
        Complex<T>* input, const Strides4<i64>& input_strides,
        Complex<T>* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, noa::fft::Sign sign, u32 flag, bool plan_only, i64 n_threads
    ) {
        auto plan = Plan(input, input_strides, output, output_strides, shape, sign, flag, n_threads);
        if (not plan_only)
            plan.execute();
    }

    template<typename T>
    void c2c(
        Complex<T>* data, const Strides4<i64>& strides, const Shape4<i64>& shape,
        noa::fft::Sign sign, u32 flag, bool plan_only, i64 n_threads
    ) {
        c2c(data, strides, data, strides, shape, sign, flag, plan_only, n_threads);
    }
}
