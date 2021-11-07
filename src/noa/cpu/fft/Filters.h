/// \file noa/cpu/fft/Filters.h
/// \brief low-, high-, band-pass filters.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

// TODO(TF) Add remap (H2H, H2HC, HC2HC, HC2H) as template parameter.

namespace noa::cpu::fft {
    /// Lowpass FFTs.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs   On the \b host. Non-redundant non-centered FFT to filter. One per batch.
    ///                     If nullptr, the filter is directly written into \p outputs and \p T must be real.
    /// \param[out] outputs On the \b host. Filtered non-redundant non-centered FFT. One per batch.
    /// \param shape        Logical {fast, medium, slow} shape.
    /// \param batches      Number of contiguous batches to filter.
    /// \param freq_cutoff  Frequency cutoff, in cycle/pix, usually from 0 (DC) to 0.5 (Nyquist).
    ///                     At this frequency, the pass starts to roll-off.
    /// \param freq_width   Width of the Hann window, in cycle/pix, usually from 0 to 0.5.
    /// \note \p inputs can be equal to \p outputs.
    template<typename T>
    NOA_HOST void lowpass(const T* inputs, T* outputs, size3_t shape, size_t batches,
                          float freq_cutoff, float freq_width);

    /// Highpass FFTs.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs   On the \b host. Non-redundant non-centered FFT to filter. One per batch.
    ///                     If nullptr, the filter is directly written into \p outputs and \p T must be real.
    /// \param[out] outputs On the \b host. Filtered non-redundant non-centered FFT. One per batch.
    /// \param shape        Logical {fast, medium, slow} shape.
    /// \param batches      Number of contiguous batches to filter.
    /// \param freq_cutoff  Frequency cutoff, in cycle/pix, usually from 0 (DC) to 0.5 (Nyquist).
    ///                     At this frequency, the pass is fully recovered.
    /// \param freq_width   Width of the Hann window, in cycle/pix, usually from 0 to 0.5.
    /// \note \p inputs can be equal to \p outputs.
    template<typename T>
    NOA_HOST void highpass(const T* inputs, T* outputs, size3_t shape, size_t batches,
                           float freq_cutoff, float freq_width);

    /// Bandpass FFTs.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Non-redundant non-centered FFT to filter. One per batch.
    ///                         If nullptr, the filter is directly written into \p outputs and \p T must be real.
    /// \param[out] outputs     On the \b host. Filtered non-redundant non-centered FFT. One per batch.
    /// \param shape            Logical {fast, medium, slow} shape.
    /// \param batches          Number of contiguous batches to filter.
    /// \param freq_cutoff_1    First frequency cutoff, in cycle/pix, usually from 0 (DC) to \p freq_cutoff_2.
    ///                         At this frequency, the pass is fully recovered.
    /// \param freq_cutoff_2    Second frequency cutoff, in cycle/pix, usually from \p freq_cutoff_1 to 0.5 (Nyquist).
    ///                         At this frequency, the pass starts to roll-off.
    /// \param freq_width_1     Frequency width, in cycle/pix, of the Hann window between 0 and \p freq_cutoff_1.
    /// \param freq_width_2     Frequency width, in cycle/pix, of the Hann window between \p freq_cutoff_2 and 0.5.
    /// \note \p inputs can be equal to \p outputs.
    template<typename T>
    NOA_HOST void bandpass(const T* inputs, T* outputs, size3_t shape, size_t batches,
                           float freq_cutoff_1, float freq_cutoff_2, float freq_width_1, float freq_width_2);
}
