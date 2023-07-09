#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/Ewise.hpp"

namespace noa::cuda::math {
    // Extracts the real and imaginary part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_real_v<T>>>
    void decompose(const Complex<T>* input, Strides4<i64> input_strides,
                   T* real, Strides4<i64> real_strides,
                   T* imag, Strides4<i64> imag_strides,
                   Shape4<i64> shape, Stream& stream);

    // Extracts the real part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_real_v<T>>>
    inline void real(const Complex<T>* input, const Strides4<i64>& input_strides,
                     T* real, const Strides4<i64>& real_strides,
                     const Shape4<i64>& shape, Stream& stream) {
        noa::cuda::ewise_unary(input, input_strides, real, real_strides, shape, noa::real_t{}, stream);
    }

    // Extracts the imaginary part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_real_v<T>>>
    inline void imag(const Complex<T>* input, const Strides4<i64>& input_strides,
                     T* imag, const Strides4<i64>& imag_strides,
                     const Shape4<i64>& shape, Stream& stream) {
        noa::cuda::ewise_unary(input, input_strides, imag, imag_strides, shape, noa::imag_t{}, stream);
    }

    // Fuses the real and imaginary components.
    template<typename T, typename = std::enable_if_t<traits::is_real_v<T>>>
    void complex(const T* real, const Strides4<i64>& real_strides,
                 const T* imag, const Strides4<i64>& imag_strides,
                 Complex<T>* output, const Strides4<i64>& output_strides,
                 const Shape4<i64>& shape, Stream& stream);
}
