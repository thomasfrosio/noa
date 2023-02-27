#pragma once

#include "noa/core/Types.hpp"

namespace noa::cpu::signal::fft::details {
    using Remap = noa::fft::Remap;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_std_v =
            noa::traits::is_any_v<T, c32, c64> &&
            (REMAP == Remap::H2H || REMAP == Remap::HC2HC || REMAP == Remap::F2F || REMAP == Remap::FC2FC);
}

namespace noa::cpu::signal::fft {
    // Standardizes (mean=0, stddev=1) a real-space signal, in Fourier space.
    template<noa::fft::Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_std_v<REMAP, T>>>
    void standardize_ifft(const T* input, const Strides4<i64>& input_strides,
                          T* output, const Strides4<i64>& output_strides,
                          const Shape4<i64>& shape, noa::fft::Norm norm, i64 threads);
}
