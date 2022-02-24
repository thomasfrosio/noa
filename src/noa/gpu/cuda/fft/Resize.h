/// \file noa/gpu/cuda/fft/Resize.h
/// \brief Fourier crop/ and pad.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::fft::details {
    template<typename T>
    NOA_HOST void cropH2H(const T* input, size4_t input_stride, size4_t input_shape,
                          T* output, size4_t output_stride, size4_t output_shape, Stream& stream);
    template<typename T>
    NOA_HOST void cropF2F(const T* input, size4_t input_stride, size4_t input_shape,
                          T* output, size4_t output_stride, size4_t output_shape, Stream& stream);
    template<typename T>
    NOA_HOST void padH2H(const T* input, size4_t input_stride, size4_t input_shape,
                         T* output, size4_t output_stride, size4_t output_shape, Stream& stream);
    template<typename T>
    NOA_HOST void padF2F(const T* input, size4_t input_stride, size4_t input_shape,
                         T* output, size4_t output_stride, size4_t output_shape, Stream& stream);
}

namespace noa::cuda::fft {
    using Remap = ::noa::fft::Remap;

    /// Crops or zero-pads an FFT.
    /// \tparam REMAP           FFT Remap. Only H2H and F2F are currently supported.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b device. FFT to resize.
    /// \param input_stride     Rightmost strides of \p input.
    /// \param input_shape      Rightmost shape of \p input.
    /// \param[out] output      On the \b device. Resized FFT.
    /// \param output_stride    Rightmost strides of \p output.
    /// \param output_shape     Rightmost shape of \p output.
    ///                         All dimensions should either be <= or >= than \p input_shape.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function runs asynchronously with respect to the host and may return before completion.
    /// \note The batch dimension cannot be resized.
    template<Remap REMAP, typename T>
    NOA_IH void resize(const T* input, size4_t input_stride, size4_t input_shape,
                       T* output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if (all(input_shape >= output_shape)) {
            if constexpr (REMAP == Remap::H2H)
                details::cropH2H<T>(input, input_stride, input_shape, output, output_stride, output_shape, stream);
            else if constexpr (REMAP == Remap::F2F)
                details::cropF2F<T>(input, input_stride, input_shape, output, output_stride, output_shape, stream);
            else
                static_assert(noa::traits::always_false_v<T>);
        } else {
            if constexpr (REMAP == Remap::H2H)
                details::padH2H<T>(input, input_stride, input_shape, output, output_stride, output_shape, stream);
            else if constexpr (REMAP == Remap::F2F)
                details::padF2F<T>(input, input_stride, input_shape, output, output_stride, output_shape, stream);
            else
                static_assert(noa::traits::always_false_v<T>);
        }
    }
}
