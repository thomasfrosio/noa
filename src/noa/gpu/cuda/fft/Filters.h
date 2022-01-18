/// \file noa/gpu/cuda/fft/Filters.h
/// \brief low-, high-, band-pass filters.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::fft {
    using noa::fft::Remap;

    /// Lowpass FFTs.
    /// \tparam REMAP           Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Non-redundant non-centered FFT to filter. One per batch.
    ///                         If nullptr, the filter is directly written into \p outputs and \p T must be real.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Filtered non-redundant non-centered FFT. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape.
    /// \param batches          Number of contiguous batches to filter.
    /// \param cutoff           Frequency cutoff, in cycle/pix, usually from 0 (DC) to 0.5 (Nyquist).
    ///                         At this frequency, the pass starts to roll-off.
    /// \param width            Width of the Hann window, in cycle/pix, usually from 0 to 0.5.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note \p inputs can be equal to \p outputs iff there's no remapping.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T>
    NOA_HOST void lowpass(const T* inputs, size3_t input_pitch,
                          T* outputs, size3_t output_pitch,
                          size3_t shape, size_t batches,
                          float cutoff, float width,
                          Stream& stream);

    /// Highpass FFTs.
    /// \tparam REMAP           Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Non-redundant non-centered FFT to filter. One per batch.
    ///                         If nullptr, the filter is directly written into \p outputs and \p T must be real.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Filtered non-redundant non-centered FFT. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape.
    /// \param batches          Number of contiguous batches to filter.
    /// \param cutoff           Frequency cutoff, in cycle/pix, usually from 0 (DC) to 0.5 (Nyquist).
    ///                         At this frequency, the pass is fully recovered.
    /// \param width            Width of the Hann window, in cycle/pix, usually from 0 to 0.5.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note \p inputs can be equal to \p outputs iff there's no remapping.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T>
    NOA_HOST void highpass(const T* inputs, size3_t input_pitch,
                           T* outputs, size3_t output_pitch,
                           size3_t shape, size_t batches,
                           float cutoff, float width,
                           Stream& stream);

    /// Bandpass FFTs.
    /// \tparam REMAP           Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Non-redundant non-centered FFT to filter. One per batch.
    ///                         If nullptr, the filter is directly written into \p outputs and \p T must be real.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Filtered non-redundant non-centered FFT. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape.
    /// \param batches          Number of contiguous batches to filter.
    /// \param cutoff1          First frequency cutoff, in cycle/pix, usually from 0 (DC) to \p cutoff2.
    ///                         At this frequency, the pass is fully recovered.
    /// \param cutoff2          Second frequency cutoff, in cycle/pix, usually from \p cutoff1 to 0.5 (Nyquist).
    ///                         At this frequency, the pass starts to roll-off.
    /// \param width1           Frequency width, in cycle/pix, of the Hann window between 0 and \p cutoff1.
    /// \param width2           Frequency width, in cycle/pix, of the Hann window between \p cutoff2 and 0.5.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note \p inputs can be equal to \p outputs iff there's no remapping.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T>
    NOA_HOST void bandpass(const T* inputs, size3_t input_pitch,
                           T* outputs, size3_t output_pitch,
                           size3_t shape, size_t batches,
                           float cutoff1, float cutoff2, float width1, float width2,
                           Stream& stream);
}
