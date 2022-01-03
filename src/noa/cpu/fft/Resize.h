/// \file noa/cpu/fft/Resize.h
/// \brief Fourier crop and pad.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::fft::details {
    template<typename T>
    NOA_HOST void cropH2H(const T* inputs, size3_t input_pitch, size3_t input_shape,
                           T* outputs, size3_t output_pitch, size3_t output_shape, size_t batches);
    template<typename T>
    NOA_HOST void cropF2F(const T* inputs, size3_t input_pitch, size3_t input_shape,
                           T* outputs, size3_t output_pitch, size3_t output_shape, size_t batches);
    template<typename T>
    NOA_HOST void padH2H(const T* inputs, size3_t input_pitch, size3_t input_shape,
                          T* outputs, size3_t output_pitch, size3_t output_shape, size_t batches);
    template<typename T>
    NOA_HOST void padF2F(const T* inputs, size3_t input_pitch, size3_t input_shape,
                          T* outputs, size3_t output_pitch, size3_t output_shape, size_t batches);
}

namespace noa::cpu::fft {
    using Remap = ::noa::fft::Remap;

    /// Crops or zero-pads an FFT.
    /// \tparam REMAP           FFT Remap. Only H2H and F2F are currently supported.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. FFT to resize.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param input_shape      Logical {fast, medium, slow} shape of \p inputs.
    /// \param[out] outputs     On the \b host. Resized FFT.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param output_shape     Logical {fast, medium, slow} shape of \p outputs.
    ///                         All dimensions should either be <= or >= than \p input_shape.
    /// \param batches          Number of batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<Remap REMAP, typename T>
    NOA_IH void resize(const T* inputs, size3_t input_pitch, size3_t input_shape,
                       T* outputs, size3_t output_pitch, size3_t output_shape,
                       size_t batches, Stream& stream) {
        if (all(input_shape >= output_shape)) {
            if constexpr (REMAP == Remap::H2H)
                stream.enqueue(details::cropH2H<T>, inputs, input_pitch, input_shape,
                               outputs, output_pitch, output_shape, batches);
            else if constexpr (REMAP == Remap::F2F)
                stream.enqueue(details::cropF2F<T>, inputs, input_pitch, input_shape,
                               outputs, output_pitch, output_shape, batches);
            else
                static_assert(noa::traits::always_false_v<T>);
        } else {
            if constexpr (REMAP == Remap::H2H)
                stream.enqueue(details::padH2H<T>, inputs, input_pitch, input_shape,
                               outputs, output_pitch, output_shape, batches);
            else if constexpr (REMAP == Remap::F2F)
                stream.enqueue(details::padF2F<T>, inputs, input_pitch, input_shape,
                               outputs, output_pitch, output_shape, batches);
            else
                static_assert(noa::traits::always_false_v<T>);
        }
    }
}
