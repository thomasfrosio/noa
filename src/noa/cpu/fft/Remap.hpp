#pragma once

#include "noa/core/Types.hpp"

namespace noa::cpu::fft {
    template<typename T, typename = std::enable_if_t<nt::is_real_or_complex_v<T>>>
    void remap(noa::fft::Remap remap,
               const T* input, Strides4<i64> input_strides,
               T* output, Strides4<i64> output_strides,
               Shape4<i64> shape, i64 threads);
}
