#pragma once

#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::math::fft {
    using Remap = noa::fft::Remap;
    using Norm = noa::fft::Norm;

    /// Standardizes (mean=0, stddev=1) a real-space signal, in Fourier space.
    /// \tparam REMAP           Remapping operator. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam T               cfloat_t or cdouble_t.
    /// \param[in] input        On the \b host. Input FFT.
    /// \param input_stride     Rightmost stride of \p input.
    /// \param[out] output      On the \b host. Output FFT. Can be equal to \p input.
    ///                         The C2R transform of \p output has its mean set to 0 and its stddev set to 1.
    /// \param output_stride    Rightmost stride of \p output.
    /// \param shape            Rightmost (logical) shape of \p input and \p output.
    /// \param norm             Normalization mode of \p input.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<Remap REMAP, typename T>
    void standardize(const shared_t<T[]>& input, size4_t input_stride,
                     const shared_t<T[]>& output, size4_t output_stride,
                     size4_t shape, Norm norm, Stream& stream);
}
