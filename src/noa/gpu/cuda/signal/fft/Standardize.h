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

    /// Standardizes (mean=0, stddev=1) a real-space signal, in Fourier space.
    /// \tparam REMAP           Remapping operator. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam T               cfloat_t or cdouble_t.
    /// \param[in] input        On the \b device. Input FFT.
    /// \param input_strides    BDHW strides of \p input.
    /// \param[out] output      On the \b device. Output FFT. Can be equal to \p input.
    ///                         The C2R transform of \p output has its mean set to 0 and its stddev set to 1.
    /// \param output_strides   BDHW strides of \p output.
    /// \param shape            BDHW logical shape of \p input and \p output.
    /// \param norm             Normalization mode of \p input.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_std_v<REMAP, T>>>
    void standardize(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides,
                     size4_t shape, Norm norm, Stream& stream);
}
