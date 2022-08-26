#pragma once

#include "noa/unified/Array.h"
#include "noa/unified/fft/Transform.h"

namespace noa::signal::fft::details {
    using Remap = noa::fft::Remap;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_std_v =
            traits::is_complex_v<T> &&
            (REMAP == Remap::H2H || REMAP == Remap::HC2HC || REMAP == Remap::F2F || REMAP == Remap::FC2FC);
}

namespace noa::signal::fft {
    using Remap = noa::fft::Remap;
    using Norm = noa::fft::Norm;

    /// Standardizes (mean=0, stddev=1) a real-space signal, by modifying its Fourier coefficients.
    /// \tparam REMAP       Remapping operator. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam T           cfloat_t or cdouble_t.
    /// \param[in] input    Input FFT.
    /// \param[out] output  Output FFT. Can be equal to \p input.
    ///                     The C2R transform of \p output has its mean set to 0 and its stddev set to 1.
    /// \param shape        BDHW logical shape of \p input and \p output.
    /// \param norm         Normalization mode of \p input.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_std_v<REMAP, T>>>
    void standardize(const Array<T>& input, const Array<T>& output, size4_t shape, Norm norm = noa::fft::NORM_DEFAULT);
}

#define NOA_UNIFIED_STANDARDIZE_
#include "noa/unified/signal/fft/Standardize.inl"
#undef NOA_UNIFIED_STANDARDIZE_
