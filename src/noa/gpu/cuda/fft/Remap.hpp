#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::fft {
    // Remaps FFT(s).
    template<typename T, typename = std::enable_if_t<noa::traits::is_real_or_complex_v<T>>>
    void remap(noa::fft::Remap remap,
               const T* input, Strides4<i64> input_strides,
               T* output, Strides4<i64> output_strides,
               Shape4<i64> shape, Stream& stream);
}
