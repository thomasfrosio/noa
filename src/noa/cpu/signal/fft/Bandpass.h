/// \file noa/cpu/signal/fft/Bandpass.h
/// \brief low-, high-, band-pass filters.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::signal::fft {
    using noa::fft::Remap;

    /// Lowpass FFTs.
    /// \tparam REMAP           Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant non-centered FFT to filter.
    ///                         If nullptr, the filter is directly written into \p output and \p T must be real.
    /// \param input_stride     Rightmost strides of \p input.
    /// \param[out] output      On the \b host. Filtered non-redundant non-centered FFT.
    /// \param output_stride    Rightmost strides of \p output.
    /// \param shape            Rightmost logical shape.
    /// \param cutoff           Frequency cutoff, in cycle/pix, usually from 0 (DC) to 0.5 (Nyquist).
    ///                         At this frequency, the pass starts to roll-off.
    /// \param width            Width of the Hann window, in cycle/pix, usually from 0 to 0.5.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note \p input can be equal to \p output iff there's no remapping.
    /// \note Depending on the stream, this function can be asynchronous and may return before completion.
    template<Remap REMAP, typename T>
    void lowpass(const shared_t<T[]>& input, size4_t input_stride,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                 float cutoff, float width, Stream& stream);

    /// Highpass FFTs.
    /// \tparam REMAP           Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant non-centered FFT to filter.
    ///                         If nullptr, the filter is directly written into \p output and \p T must be real.
    /// \param input_stride     Rightmost strides of \p input.
    /// \param[out] output      On the \b host. Filtered non-redundant non-centered FFT.
    /// \param output_stride    Rightmost strides of \p output.
    /// \param shape            Rightmost logical shape.
    /// \param cutoff           Frequency cutoff, in cycle/pix, usually from 0 (DC) to 0.5 (Nyquist).
    ///                         At this frequency, the pass is fully recovered.
    /// \param width            Width of the Hann window, in cycle/pix, usually from 0 to 0.5.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note \p input can be equal to \p output iff there's no remapping.
    /// \note Depending on the stream, this function can be asynchronous and may return before completion.
    template<Remap REMAP, typename T>
    void highpass(const shared_t<T[]>& input, size4_t input_stride,
                  const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                  float cutoff, float width, Stream& stream);

    /// Bandpass FFTs.
    /// \tparam REMAP           Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant non-centered FFT to filter.
    ///                         If nullptr, the filter is directly written into \p output and \p T must be real.
    /// \param input_stride     Rightmost strides of \p input.
    /// \param[out] output      On the \b host. Filtered non-redundant non-centered FFT.
    /// \param output_stride    Rightmost strides of \p output.
    /// \param shape            Rightmost logical shape.
    /// \param cutoff1          First frequency cutoff, in cycle/pix, usually from 0 (DC) to \p cutoff2.
    ///                         At this frequency, the pass is fully recovered.
    /// \param cutoff2          Second frequency cutoff, in cycle/pix, usually from \p cutoff1 to 0.5 (Nyquist).
    ///                         At this frequency, the pass starts to roll-off.
    /// \param width1           Frequency width, in cycle/pix, of the Hann window between 0 and \p cutoff1.
    /// \param width2           Frequency width, in cycle/pix, of the Hann window between \p cutoff2 and 0.5.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note \p input can be equal to \p output iff there's no remapping.
    /// \note Depending on the stream, this function can be asynchronous and may return before completion.
    template<Remap REMAP, typename T>
    void bandpass(const shared_t<T[]>& input, size4_t input_stride,
                  const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                  float cutoff1, float cutoff2, float width1, float width2, Stream& stream);
}
