#pragma once

#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::signal::fft::details {
    using Remap = noa::fft::Remap;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_std_v =
            traits::is_any_v<T, cfloat_t, cdouble_t> &&
    (REMAP == Remap::H2H || REMAP == Remap::HC2HC || REMAP == Remap::F2F || REMAP == Remap::FC2FC);
}

namespace noa::cpu::signal::fft {
    using Remap = noa::fft::Remap;
    using Norm = noa::fft::Norm;

    // Standardizes (mean=0, stddev=1) a real-space signal, in Fourier space.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_std_v<REMAP, T>>>
    void standardize(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides,
                     size4_t shape, Norm norm, Stream& stream);
}
