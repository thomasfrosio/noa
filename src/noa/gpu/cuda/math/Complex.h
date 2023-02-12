#pragma once

#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/Ewise.h"

namespace noa::cuda::math {
    // Extracts the real and imaginary part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_real_v<T>>>
    void decompose(const Shared<Complex<T>[]>& input, Strides4<i64> input_strides,
                   const Shared<T[]>& real, Strides4<i64> real_strides,
                   const Shared<T[]>& imag, Strides4<i64> imag_strides,
                   Shape4<i64> shape, Stream& stream);

    // Extracts the real part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_real_v<T>>>
    inline void real(const Shared<Complex<T>[]>& input, const Strides4<i64>& input_strides,
                     const Shared<T[]>& real, const Strides4<i64>& real_strides,
                     const Shape4<i64>& shape, Stream& stream) {
        cuda::ewise_unary(input, input_strides, real, real_strides, shape, noa::real_t{}, stream);
    }

    // Extracts the imaginary part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_real_v<T>>>
    inline void imag(const Shared<Complex<T>[]>& input, const Strides4<i64>& input_strides,
                     const Shared<T[]>& imag, const Strides4<i64>& imag_strides,
                     const Shape4<i64>& shape, Stream& stream) {
        cuda::ewise_unary(input, input_strides, imag, imag_strides, shape, noa::imag_t{}, stream);
    }

    // Fuses the real and imaginary components.
    template<typename T, typename = std::enable_if_t<traits::is_real_v<T>>>
    void complex(const Shared<T[]>& real, const Strides4<i64>& real_strides,
                 const Shared<T[]>& imag, const Strides4<i64>& imag_strides,
                 const Shared<Complex<T>[]>& output, const Strides4<i64>& output_strides,
                 const Shape4<i64>& shape, Stream& stream);
}
