#pragma once

#include "noa/core/Types.hpp"

namespace noa::cpu::fft::details {
    using Remap = ::noa::fft::Remap;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_resize =
            (noa::traits::is_real_or_complex_v<T>) &&
            (REMAP == Remap::H2H || REMAP == Remap::F2F ||
             REMAP == Remap::HC2HC || REMAP == Remap::FC2FC);
}

namespace noa::cpu::fft {
    // Crops or zero-pads a FFT.
    template<noa::fft::Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_resize<REMAP, T>>>
    void resize(const T* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
                T* output, Strides4<i64> output_strides, Shape4<i64> output_shape,
                i64 threads);
}
