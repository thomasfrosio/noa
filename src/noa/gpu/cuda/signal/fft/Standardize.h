#pragma once

#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::signal::fft::details {
    using Remap = noa::fft::Remap;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_std_v =
            traits::is_any_v<T, cfloat_t, cdouble_t> &&
            (REMAP == Remap::H2H || REMAP == Remap::HC2HC || REMAP == Remap::F2F || REMAP == Remap::FC2FC);
}

namespace noa::cuda::signal::fft {
    using Remap = noa::fft::Remap;
    using Norm = noa::fft::Norm;

    // Standardizes (mean=0, stddev=1) a real-space signal, in Fourier space.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_std_v<REMAP, T>>>
    void standardize(const shared_t<T[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& output, dim4_t output_strides,
                     dim4_t shape, Norm norm, Stream& stream);
}
