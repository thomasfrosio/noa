/// \file noa/cpu/fft/Filters.h
/// \brief low-, high-, band-pass filters.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

// These filters are all using a raised-cosine (Hann) window. The cutoffs and window width are specified in fractional
// reciprocal lattice units from 0 to 0.5. Anything outside this range is still valid.
//
// For instance, given a 64x64 image with a pixel size of 1.4 A/pixel. To lowpass filter this image at a resolution
// of 8 A, the frequency cutoff should be 1.4 / 8 = 0.175. Note that multiplying this normalized value by the
// dimension of the image gives us the number of oscillations in the real-space image at this frequency (or the
// resolution shell in Fourier space), i.e. 0.175 * 64 = 22.4. Naturally, the Nyquist frequency is at 0.5 in fractional
// reciprocal lattice units and, for this example, at the 64th shell.

namespace noa::cpu::fft {
    /// Applies a lowpass filter to the input array(s).
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs   On the \b host. Contiguous input non-redundant, non-centered transforms. One per batch.
    /// \param[out] outputs On the \b host. Contiguous output non-redundant, non-centered transforms. One per batch.
    /// \param shape        Logical {fast, medium, slow} shape.
    /// \param freq_cutoff  Frequency cutoff, from 0 to 0.5. At this frequency, the pass starts to roll-off.
    /// \param freq_width   Width of the Hann window, in frequencies, from 0 to 0.5.
    /// \param batches      Number of batches.
    /// \note If \p inputs == \p outputs, applies it in-place.
    template<typename T>
    NOA_HOST void lowpass(const T* inputs, T* outputs, size3_t shape,
                          float freq_cutoff, float freq_width, uint batches);

    /// Computes a lowpass filter.
    /// \tparam T                   float, double.
    /// \param[out] output_lowpass  On the \b host. Contiguous output non-redundant, non-centered lowpass filter.
    /// \param shape                Logical {fast, medium, slow} shape.
    /// \param freq_cutoff          Frequency cutoff, from 0 to 0.5. At this frequency, the pass starts to roll-off.
    /// \param freq_width           Width of the Hann window, in frequencies, from 0 to 0.5.
    template<typename T>
    NOA_HOST void lowpass(T* output_lowpass, size3_t shape, float freq_cutoff, float freq_width);

    /// Applies a highpass filter to the input array(s).
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs   On the \b host. Contiguous input non-redundant, non-centered transforms. One per batch.
    /// \param[out] outputs On the \b host. Contiguous output non-redundant, non-centered transforms. One per batch.
    /// \param shape        Logical {fast, medium, slow} shape.
    /// \param freq_cutoff  Frequency cutoff, from 0 to 0.5. At this frequency, the pass is fully recovered.
    /// \param freq_width   Width of the Hann window, in frequencies, from 0 to 0.5.
    /// \param batches      Number of batches.
    /// \note If \p inputs == \p outputs, applies it in-place.
    template<typename T>
    NOA_HOST void highpass(const T* inputs, T* outputs, size3_t shape,
                           float freq_cutoff, float freq_width, uint batches);

    /// Computes a highpass filter.
    /// \tparam T                   float, double.
    /// \param[out] output_highpass On the \b host. Contiguous output non-redundant, non-centered lowpass filter.
    /// \param shape                Logical {fast, medium, slow} shape.
    /// \param freq_cutoff          Frequency cutoff, from 0 to 0.5. At this frequency, the pass is fully recovered.
    /// \param freq_width           Width of the Hann window, in frequencies, from 0 to 0.5.
    template<typename T>
    NOA_HOST void highpass(T* output_highpass, size3_t shape, float freq_cutoff, float freq_width);

    /// Applies a bandpass filter to the input array(s).
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Contiguous input non-redundant, non-centered transforms. One per batch.
    /// \param[out] outputs     On the \b host. Contiguous output non-redundant, non-centered transforms. One per batch.
    /// \param shape            Logical {fast, medium, slow} shape.
    /// \param freq_cutoff_1    First frequency cutoff, from 0 to 0.5. At this frequency, the pass is fully recovered.
    /// \param freq_cutoff_2    Second frequency cutoff, from 0 to 0.5. At this frequency, the pass starts to roll-off.
    /// \param freq_width_1     Width of the Hann window between 0 and \p freq_cutoff_1, in frequencies, from 0 to 0.5.
    /// \param freq_width_2     Width of the Hann window between \p freq_cutoff_2 and 0.5, in frequencies, from 0 to 0.5.
    /// \param batches          Number of batches.
    /// \note If \p inputs == \p outputs, applies it in-place.
    template<typename T>
    NOA_HOST void bandpass(const T* inputs, T* outputs, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                           float freq_width_1, float freq_width_2, uint batches);

    /// Computes a bandpass filter.
    /// \tparam T                   float, double.
    /// \param[out] output_bandpass On the \b host. Contiguous output non-redundant, non-centered lowpass filter.
    /// \param shape                Logical {fast, medium, slow} shape.
    /// \param freq_cutoff_1        First frequency cutoff, from 0 to 0.5. At this frequency, the pass is fully recovered.
    /// \param freq_cutoff_2        Second frequency cutoff, from 0 to 0.5. At this frequency, the pass starts to roll-off.
    /// \param freq_width_1         Width of the Hann window between 0 and \p freq_cutoff_1, in frequencies, from 0 to 0.5.
    /// \param freq_width_2         Width of the Hann window between \p freq_cutoff_2 and 0.5, in frequencies, from 0 to 0.5.
    template<typename T>
    NOA_HOST void bandpass(T* output_bandpass, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                           float freq_width_1, float freq_width_2);
}
