/// \file noa/gpu/cuda/fft/Resize.h
/// \brief Fourier crop/ and pad.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::fft {
    /// Crops non-redundant FFTs.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Non-redundant non-centered FFT.
    /// \param input_pitch      Pitch of \p inputs, in number of \p T elements.
    /// \param input_shape      Logical {fast, medium, slow} shape of \p inputs.
    /// \param[out] outputs     On the \b device. Cropped non-redundant non-centered FFT.
    /// \param output_pitch     Pitch of \p outputs, in number of \p T elements.
    /// \param output_shape     Logical {fast, medium, slow} shape of \p \p outputs.
    ///                         All dimensions should be less or equal than the dimensions of \p input_shape.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note See noa::fft::Layout for more details about FFT layouts.
    /// \note \p inputs and \p outputs should not overlap.
    /// \note This function runs asynchronously with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void crop(const T* inputs, size_t input_pitch, size3_t input_shape,
                       T* outputs, size_t output_pitch, size3_t output_shape,
                       size_t batches, Stream& stream);

    /// Crops redundant FFTs.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Redundant non-centered FFT.
    /// \param input_pitch      Pitch of \p in, in number of \p T elements.
    /// \param input_shape      Logical {fast, medium, slow} shape of \p inputs.
    /// \param[out] outputs     On the \b device. Cropped redundant non-centered FFT.
    /// \param output_pitch     Pitch of \p out, in number of \p T elements.
    /// \param output_shape     Logical {fast, medium, slow} shape of \p outputs.
    ///                         All dimensions should be less or equal than the dimensions of \p input_shape.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note See noa::fft::Layout for more details about FFT layouts.
    /// \note \p inputs and \p outputs should not overlap.
    /// \note This function runs asynchronously with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void cropFull(const T* inputs, size_t input_pitch, size3_t input_shape,
                           T* outputs, size_t output_pitch, size3_t output_shape,
                           size_t batches, Stream& stream);

    /// Pads non-redundant FFTs with zeros.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Non-redundant non-centered FFT.
    /// \param input_pitch      Pitch of \p in, in number of elements.
    /// \param input_shape      Logical {fast, medium, slow} shape of \p inputs.
    /// \param[out] outputs     On the \b device. Padded non-redundant non-centered FFT.
    /// \param output_pitch     Pitch of \p out, in number of elements.
    /// \param output_shape     Logical {fast, medium, slow} shape of \p outputs.
    ///                         All dimensions should be greater or equal than the dimensions of \p input_shape.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note See noa::fft::Layout for more details about FFT layouts.
    /// \note \p inputs and \p outputs should not overlap.
    /// \note This function runs asynchronously with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void pad(const T* inputs, size_t input_pitch, size3_t input_shape,
                      T* outputs, size_t output_pitch, size3_t output_shape,
                      size_t batches, Stream& stream);

    /// Pads redundant FFTs with zeros.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Redundant non-centered FFT.
    /// \param input_pitch      Pitch of \p in, in number of \p T elements.
    /// \param input_shape      Logical {fast, medium, slow} shape of \p inputs.
    /// \param[out] outputs     On the \b device. Padded redundant non-centered FFT.
    /// \param output_pitch     Pitch of \p out, in number of \p T elements.
    /// \param output_shape     Logical {fast, medium, slow} shape of \p outputs.
    ///                         All dimensions should be greater or equal than the dimensions of \p input_shape.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note See noa::fft::Layout for more details about FFT layouts.
    /// \note \p inputs and \p outputs should not overlap.
    /// \note This function runs asynchronously with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void padFull(const T* inputs, size_t input_pitch, size3_t input_shape,
                          T* outputs, size_t output_pitch, size3_t output_shape,
                          size_t batches, Stream& stream);
}
