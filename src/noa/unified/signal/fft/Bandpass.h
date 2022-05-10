#pragma once

#include "noa/unified/Array.h"

namespace noa::signal::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_pass_v = (traits::is_float_v<T> || traits::is_complex_v<T>) &&
                                     (REMAP == H2H || REMAP == H2HC || REMAP == HC2H || REMAP == HC2HC);
}

namespace noa::signal::fft {
    using noa::fft::Remap;

    /// Lowpass FFTs.
    /// \tparam REMAP       Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input    FFT to filter. If empty, the filter is written into \p output and \p T must be real.
    /// \param[out] output  Filtered FFT.
    /// \param shape        Rightmost logical shape.
    /// \param cutoff       Frequency cutoff, in cycle/pix, usually from 0 (DC) to 0.5 (Nyquist).
    ///                     At this frequency, the pass starts to roll-off.
    /// \param width        Width of the Hann window, in cycle/pix, usually from 0 to 0.5.
    /// \note \p input can be equal to \p output iff there's no remapping, i.e. with H2H or HC2HC.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_pass_v<REMAP, T>>>
    void lowpass(const Array<T>& input, const Array<T>& output, size4_t shape, float cutoff, float width);

    /// Highpass FFTs.
    /// \tparam REMAP       Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input    FFT to filter. If empty, the filter is written into \p output and \p T must be real.
    /// \param[out] output  Filtered FFT.
    /// \param shape        Rightmost logical shape.
    /// \param cutoff       Frequency cutoff, in cycle/pix, usually from 0 (DC) to 0.5 (Nyquist).
    ///                     At this frequency, the pass is fully recovered.
    /// \param width        Width of the Hann window, in cycle/pix, usually from 0 to 0.5.
    /// \note \p input can be equal to \p output iff there's no remapping, i.e. with H2H or HC2HC.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_pass_v<REMAP, T>>>
    void highpass(const Array<T>& input, const Array<T>& output, size4_t shape, float cutoff, float width);

    /// Bandpass FFTs.
    /// \tparam REMAP       Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input    FFT to filter. If empty, the filter is written into \p output and \p T must be real.
    /// \param[out] output  Filtered FFT.
    /// \param shape        Rightmost logical shape.
    /// \param cutoff1      First frequency cutoff, in cycle/pix, usually from 0 (DC) to \p cutoff2.
    ///                     At this frequency, the pass is fully recovered.
    /// \param cutoff2      Second frequency cutoff, in cycle/pix, usually from \p cutoff1 to 0.5 (Nyquist).
    ///                     At this frequency, the pass starts to roll-off.
    /// \param width1       Frequency width, in cycle/pix, of the Hann window between 0 and \p cutoff1.
    /// \param width2       Frequency width, in cycle/pix, of the Hann window between \p cutoff2 and 0.5.
    /// \note \p input can be equal to \p output iff there's no remapping, i.e. with H2H or HC2HC.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_pass_v<REMAP, T>>>
    void bandpass(const Array<T>& input, const Array<T>& output, size4_t shape,
                  float cutoff1, float cutoff2, float width1, float width2);
}

#define NOA_UNIFIED_BANDPASS_
#include "noa/unified/signal/fft/Bandpass.inl"
#undef NOA_UNIFIED_BANDPASS_
