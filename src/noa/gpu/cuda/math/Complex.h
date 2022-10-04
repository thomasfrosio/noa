#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/math/Ewise.h"

namespace noa::cuda::math {
    // Extracts the real and imaginary part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void decompose(const shared_t<Complex<T>[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& real, dim4_t real_strides,
                   const shared_t<T[]>& imag, dim4_t imag_strides,
                   dim4_t shape, Stream& stream);

    // Extracts the real part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    inline void real(const shared_t<Complex<T>[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& real, dim4_t real_strides,
                     dim4_t shape, Stream& stream) {
        cuda::math::ewise(input, input_strides, real, real_strides, shape, noa::math::real_t{}, stream);
    }

    // Extracts the imaginary part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    inline void imag(const shared_t<Complex<T>[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& imag, dim4_t imag_strides,
                     dim4_t shape, Stream& stream) {
        cuda::math::ewise(input, input_strides, imag, imag_strides, shape, noa::math::imag_t{}, stream);
    }

    // Fuses the real and imaginary components.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void complex(const shared_t<T[]>& real, dim4_t real_strides,
                 const shared_t<T[]>& imag, dim4_t imag_strides,
                 const shared_t<Complex<T>[]>& output, dim4_t output_strides,
                 dim4_t shape, Stream& stream);
}
