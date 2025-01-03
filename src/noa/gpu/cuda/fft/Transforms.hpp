#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Enums.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/fft/Plan.hpp"

namespace noa::cuda::fft {
    template<typename T>
    void r2c(T* input, Complex<T>* output, const Shape4<i64>& shape, bool cache_plan, Stream& stream) {
        Plan<T>(Type::R2C, shape, stream.device(), cache_plan)
                .execute(input, output, stream);
    }

    template<typename T>
    void r2c(
        T* input, const Strides4<i64>& input_strides,
        Complex<T>* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, bool cache_plan, Stream& stream
    ) {
        Plan<T>(Type::R2C, input_strides, output_strides, shape, stream.device(), cache_plan)
                .execute(input, output, stream);
    }

    template<typename T>
    void r2c(T* data, const Shape4<i64>& shape, bool cache_plan, Stream& stream) {
        r2c(data, reinterpret_cast<Complex<T>*>(data), shape, cache_plan, stream);
    }

    template<typename T>
    void r2c(
        T* data, const Strides4<i64>& strides, const Shape4<i64>& shape,
        bool cache_plan, Stream& stream
    ) {
        // Since it is in-place, the physical width (in real elements):
        //  1: is even, since complex elements take 2 real elements.
        //  2: has at least 1 (if odd) or 2 (if even) extract real element.
        NOA_ASSERT(is_even(strides.physical_shape()[2]));
        NOA_ASSERT(strides.physical_shape()[2] >= shape[3] + 1 + static_cast<i64>(is_even(shape[3])));

        const auto complex_strides = Strides{strides[0] / 2, strides[1] / 2, strides[2] / 2, strides[3]};
        r2c(data, strides, reinterpret_cast<Complex<T>*>(data), complex_strides, shape, cache_plan, stream);
    }

    template<typename T>
    void c2r(
        Complex<T>* input, const Strides4<i64>& input_strides,
        T* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, bool cache_plan, Stream& stream
    ) {
        Plan<T>(Type::C2R, input_strides, output_strides, shape, stream.device(), cache_plan)
                .execute(input, output, stream);
    }

    template<typename T>
    void c2r(Complex<T>* input, T* output, const Shape4<i64>& shape, bool cache_plan, Stream& stream) {
        Plan<T>(Type::C2R, shape, stream.device(), cache_plan).execute(input, output, stream);
    }

    template<typename T>
    void c2r(Complex<T>* data, const Shape4<i64>& shape, bool cache_plan, Stream& stream) {
        c2r(data, reinterpret_cast<T*>(data), shape, cache_plan, stream);
    }

    template<typename T>
    void c2r(
        Complex<T>* data, const Strides4<i64>& strides,
        const Shape4<i64>& shape, bool cache_plan, Stream& stream
    ) {
        const auto real_strides = Strides{strides[0] * 2, strides[1] * 2, strides[2] * 2, strides[3]};
        c2r(data, strides, reinterpret_cast<T*>(data), real_strides, shape, cache_plan, stream);
    }

    template<typename T>
    void c2c(
        Complex<T>* input, const Strides4<i64>& input_strides,
        Complex<T>* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, noa::fft::Sign sign, bool cache_plan, Stream& stream
    ) {
        Plan<T>(Type::C2C, input_strides, output_strides, shape, stream.device(), cache_plan)
                .execute(input, output, sign, stream);
    }

    template<typename T>
    void c2c(
        Complex<T>* input, Complex<T>* output, const Shape4<i64>& shape,
        noa::fft::Sign sign, bool cache_plan, Stream& stream
    ) {
        Plan<T>(Type::C2C, shape, stream.device(), cache_plan).execute(input, output, sign, stream);
    }

    template<typename T>
    void c2c(Complex<T>* data, const Shape4<i64>& shape, noa::fft::Sign sign, bool cache_plan, Stream& stream) {
        c2c(data, data, shape, sign, cache_plan, stream);
    }

    template<typename T>
    void c2c(
        Complex<T>* data, const Strides4<i64>& strides, const Shape4<i64>& shape,
        noa::fft::Sign sign, bool cache_plan, Stream& stream
    ) {
        c2c(data, strides, data, strides, shape, sign, cache_plan, stream);
    }
}
