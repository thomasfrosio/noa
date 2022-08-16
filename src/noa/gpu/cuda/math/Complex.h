#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/math/Ewise.h"

namespace noa::cuda::math {
    // Extracts the real and imaginary part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void decompose(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                   const shared_t<T[]>& real, size4_t real_strides,
                   const shared_t<T[]>& imag, size4_t imag_strides,
                   size4_t shape, Stream& stream);

    // Extracts the real part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    inline void real(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                     const shared_t<T[]>& real, size4_t real_strides,
                     size4_t shape, Stream& stream) {
        cuda::math::ewise(input, input_strides, real, real_strides, shape, noa::math::real_t{}, stream);
    }

    // Extracts the imaginary part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    inline void imag(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                     const shared_t<T[]>& imag, size4_t imag_strides,
                     size4_t shape, Stream& stream) {
        cuda::math::ewise(input, input_strides, imag, imag_strides, shape, noa::math::imag_t{}, stream);
    }

    // Fuses the real and imaginary components.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void complex(const shared_t<T[]>& real, size4_t real_strides,
                 const shared_t<T[]>& imag, size4_t imag_strides,
                 const shared_t<Complex<T>[]>& output, size4_t output_strides,
                 size4_t shape, Stream& stream);
}
