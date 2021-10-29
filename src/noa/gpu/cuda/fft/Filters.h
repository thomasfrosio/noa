/// \file noa/gpu/cuda/fft/Filters.h
/// \brief low-, high-, band-pass filters.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::fft {
    /// Lowpass filters FFT(s).
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Non-redundant non-centered FFT to filter. One per batch.
    /// \param inputs_pitch     Pitch, in \p T elements, of \a inputs.
    /// \param[out] outputs     On the \b device. Filtered non-redundant non-centered FFT. One per batch.
    /// \param outputs_pitch    Pitch, in \p T elements, of \a outputs.
    /// \param shape            Logical {fast, medium, slow} shape.
    /// \param batches          Number of contiguous batches to filter.
    /// \param freq_cutoff      Frequency cutoff, usually from 0 to 0.5.
    ///                         At this frequency, the pass starts to roll-off.
    /// \param freq_width       Width of the Hann window, in frequencies, usually from 0 to 0.5.
    /// \note \p inputs can be equal to \p outputs.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void lowpass(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                          size3_t shape, size_t batches,
                          float freq_cutoff, float freq_width, Stream& stream);

    /// Computes a lowpass filter.
    /// \tparam T                       float, double.
    /// \param[out] output_lowpass      On the \b device. Non-redundant non-centered lowpass filter.
    /// \param output_lowpass_pitch     Pitch, in \p T elements, of \a output_lowpass.
    /// \param shape                    Logical {fast, medium, slow} shape.
    /// \param freq_cutoff              Frequency cutoff, usually from 0 to 0.5.
    ///                                 At this frequency, the pass starts to roll-off.
    /// \param freq_width               Width of the Hann window, in frequencies, usually from 0 to 0.5.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void lowpass(T* output_lowpass, size_t output_lowpass_pitch, size3_t shape,
                          float freq_cutoff, float freq_width, Stream& stream);

    /// Highpass filters FFT(s).
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Non-redundant non-centered FFT to filter. One per batch.
    /// \param inputs_pitch     Pitch, in \p T elements, of \a inputs.
    /// \param[out] outputs     On the \b device. Filtered non-redundant non-centered FFT. One per batch.
    /// \param outputs_pitch    Pitch, in \p T elements, of \a outputs.
    /// \param shape            Logical {fast, medium, slow} shape.
    /// \param batches          Number of contiguous batches to filter.
    /// \param freq_cutoff      Frequency cutoff, usually from 0 to 0.5.
    ///                         At this frequency, the pass is fully recovered.
    /// \param freq_width       Width of the Hann window, in frequencies, usually from 0 to 0.5.
    /// \note \p inputs can be equal to \p outputs.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void highpass(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                           size3_t shape, size_t batches,
                           float freq_cutoff, float freq_width, Stream& stream);

    /// Computes a highpass filter.
    /// \tparam T                       float, double.
    /// \param[out] output_lowpass      On the \b device. Non-redundant non-centered lowpass filter.
    /// \param output_highpass_pitch    Pitch, in \p T elements, of \a output_highpass.
    /// \param shape                    Logical {fast, medium, slow} shape.
    /// \param freq_cutoff              Frequency cutoff, usually from 0 to 0.5.
    ///                                 At this frequency, the pass is fully recovered.
    /// \param freq_width               Width of the Hann window, in frequencies, usually from 0 to 0.5.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void highpass(T* output_highpass, size_t output_highpass_pitch, size3_t shape,
                           float freq_cutoff, float freq_width, Stream& stream);

    /// Bandpass filters FFT(s).
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Non-redundant non-centered FFT to filter. One per batch.
    /// \param inputs_pitch     Pitch, in \p T elements, of \a inputs.
    /// \param[out] outputs     On the \b device. Filtered non-redundant non-centered FFT. One per batch.
    /// \param outputs_pitch    Pitch, in \p T elements, of \a outputs.
    /// \param shape            Logical {fast, medium, slow} shape.
    /// \param batches          Number of contiguous batches to filter.
    /// \param freq_cutoff_1    First frequency cutoff, usually from 0 to 0.5.
    ///                         At this frequency, the pass is fully recovered.
    /// \param freq_cutoff_2    Second frequency cutoff, usually from 0 to 0.5.
    ///                         At this frequency, the pass starts to roll-off.
    /// \param freq_width_1     Frequency width of the Hann window between 0 and \p freq_cutoff_1.
    /// \param freq_width_2     Frequency width of the Hann window between \p freq_cutoff_2 and 0.5.
    /// \note \p inputs can be equal to \p outputs.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void bandpass(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                           size3_t shape, size_t batches,
                           float freq_cutoff_1, float freq_cutoff_2, float freq_width_1, float freq_width_2,
                           Stream& stream);

    /// Computes a bandpass filter.
    /// \tparam T                       float, double.
    /// \param[out] output_bandpass     On the \b device. Non-redundant non-centered bandpass filter.
    /// \param output_bandpass_pitch    Pitch, in \p T elements, of \a output_bandpass.
    /// \param shape                    Logical {fast, medium, slow} shape.
    /// \param freq_cutoff_1            First frequency cutoff, usually from 0 to 0.5.
    ///                                 At this frequency, the pass is fully recovered.
    /// \param freq_cutoff_2            Second frequency cutoff, usually from 0 to 0.5.
    ///                                 At this frequency, the pass starts to roll-off.
    /// \param freq_width_1             Frequency width of the Hann window between 0 and \p freq_cutoff_1.
    /// \param freq_width_2             Frequency width of the Hann window between \p freq_cutoff_2 and 0.5.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void bandpass(T* output_bandpass, size_t output_bandpass_pitch, size3_t shape,
                           float freq_cutoff_1, float freq_cutoff_2, float freq_width_1, float freq_width_2,
                           Stream& stream);
}
