/// \file noa/cpu/fft/Filters.h
/// \brief low-, high-, band-pass filters.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::cpu::fft {
    /// Lowpass filters FFT(s).
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs   On the \b host. Non-redundant non-centered FFT to filter. One per batch.
    /// \param[out] outputs On the \b host. Filtered non-redundant non-centered FFT. One per batch.
    /// \param shape        Logical {fast, medium, slow} shape.
    /// \param batches      Number of contiguous batches to filter.
    /// \param freq_cutoff  Frequency cutoff, usually from 0 to 0.5.
    ///                     At this frequency, the pass starts to roll-off.
    /// \param freq_width   Width of the Hann window, in frequencies, usually from 0 to 0.5.
    /// \note \p inputs can be equal to \p outputs.
    template<typename T>
    NOA_HOST void lowpass(const T* inputs, T* outputs, size3_t shape, size_t batches,
                          float freq_cutoff, float freq_width);

    /// Computes a lowpass filter.
    /// \tparam T                   float, double.
    /// \param[out] output_lowpass  On the \b host. Non-redundant non-centered lowpass filter.
    /// \param shape                Logical {fast, medium, slow} shape.
    /// \param freq_cutoff          Frequency cutoff, usually from 0 to 0.5.
    ///                             At this frequency, the pass starts to roll-off.
    /// \param freq_width           Width of the Hann window, in frequencies, usually from 0 to 0.5.
    template<typename T>
    NOA_HOST void lowpass(T* output_lowpass, size3_t shape, float freq_cutoff, float freq_width);

    /// Highpass filters FFT(s).
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs   On the \b host. Non-redundant non-centered FFT to filter. One per batch.
    /// \param[out] outputs On the \b host. Filtered non-redundant non-centered FFT. One per batch.
    /// \param shape        Logical {fast, medium, slow} shape.
    /// \param batches      Number of contiguous batches to filter.
    /// \param freq_cutoff  Frequency cutoff, usually from 0 to 0.5.
    ///                     At this frequency, the pass is fully recovered.
    /// \param freq_width   Width of the Hann window, in frequencies, usually from 0 to 0.5.
    /// \note \p inputs can be equal to \p outputs.
    template<typename T>
    NOA_HOST void highpass(const T* inputs, T* outputs, size3_t shape, size_t batches,
                           float freq_cutoff, float freq_width);

    /// Computes a highpass filter.
    /// \tparam T                   float, double.
    /// \param[out] output_highpass On the \b host. Non-redundant non-centered lowpass filter.
    /// \param shape                Logical {fast, medium, slow} shape.
    /// \param freq_cutoff          Frequency cutoff, usually from 0 to 0.5.
    ///                             At this frequency, the pass is fully recovered.
    /// \param freq_width           Width of the Hann window, in frequencies, usually from 0 to 0.5.
    template<typename T>
    NOA_HOST void highpass(T* output_highpass, size3_t shape, float freq_cutoff, float freq_width);

    /// Bandpass filters FFT(s).
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Non-redundant non-centered FFT to filter. One per batch.
    /// \param[out] outputs     On the \b host. Filtered non-redundant non-centered FFT. One per batch.
    /// \param shape            Logical {fast, medium, slow} shape.
    /// \param batches          Number of contiguous batches to filter.
    /// \param freq_cutoff_1    First frequency cutoff, usually from 0 to 0.5.
    ///                         At this frequency, the pass is fully recovered.
    /// \param freq_cutoff_2    Second frequency cutoff, usually from 0 to 0.5.
    ///                         At this frequency, the pass starts to roll-off.
    /// \param freq_width_1     Frequency width of the Hann window between 0 and \p freq_cutoff_1.
    /// \param freq_width_2     Frequency width of the Hann window between \p freq_cutoff_2 and 0.5.
    /// \note \p inputs can be equal to \p outputs.
    template<typename T>
    NOA_HOST void bandpass(const T* inputs, T* outputs, size3_t shape, size_t batches,
                           float freq_cutoff_1, float freq_cutoff_2, float freq_width_1, float freq_width_2);

    /// Computes a bandpass filter.
    /// \tparam T                   float, double.
    /// \param[out] output_bandpass On the \b host. Non-redundant non-centered bandpass filter.
    /// \param shape                Logical {fast, medium, slow} shape.
    /// \param freq_cutoff_1        First frequency cutoff, usually from 0 to 0.5.
    ///                             At this frequency, the pass is fully recovered.
    /// \param freq_cutoff_2        Second frequency cutoff, usually from 0 to 0.5.
    ///                             At this frequency, the pass starts to roll-off.
    /// \param freq_width_1         Frequency width of the Hann window between 0 and \p freq_cutoff_1.
    /// \param freq_width_2         Frequency width of the Hann window between \p freq_cutoff_2 and 0.5.
    template<typename T>
    NOA_HOST void bandpass(T* output_bandpass, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                           float freq_width_1, float freq_width_2);
}
