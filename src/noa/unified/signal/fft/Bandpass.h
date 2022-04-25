#pragma once

#include "noa/cpu/signal/fft/Bandpass.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/Bandpass.h"
#endif

#include "noa/unified/Array.h"

namespace noa::signal::fft {
    using noa::fft::Remap;

    /// Lowpass FFTs.
    /// \tparam REMAP           Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input        FFT to filter. If empty, the filter is written into \p output and \p T must be real.
    /// \param[out] output      Filtered FFT.
    /// \param shape            Rightmost logical shape.
    /// \param cutoff           Frequency cutoff, in cycle/pix, usually from 0 (DC) to 0.5 (Nyquist).
    ///                         At this frequency, the pass starts to roll-off.
    /// \param width            Width of the Hann window, in cycle/pix, usually from 0 to 0.5.
    /// \note \p input can be equal to \p output iff there's no remapping, i.e. with H2H or HC2HC.
    template<Remap REMAP, typename T>
    void lowpass(const Array<T>& input, const Array<T>& output, size4_t shape, float cutoff, float width) {
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "The non-redundant FFT with a shape of [logical:{}, pitch:{}] is expected, but got pitch of {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        size4_t input_stride = input.stride();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          input.shape(), output.shape());
            }
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), device);
        }

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::lowpass<REMAP>(input.share(), input_stride,
                                             output.share(), output.stride(),
                                             shape, cutoff, width, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::lowpass<REMAP>(input.share(), input_stride,
                                              output.share(), output.stride(),
                                              shape, cutoff, width, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Highpass FFTs.
    /// \tparam REMAP           Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input        FFT to filter. If empty, the filter is written into \p output and \p T must be real.
    /// \param[out] output      Filtered FFT.
    /// \param shape            Rightmost logical shape.
    /// \param cutoff           Frequency cutoff, in cycle/pix, usually from 0 (DC) to 0.5 (Nyquist).
    ///                         At this frequency, the pass is fully recovered.
    /// \param width            Width of the Hann window, in cycle/pix, usually from 0 to 0.5.
    /// \note \p input can be equal to \p output iff there's no remapping, i.e. with H2H or HC2HC.
    template<Remap REMAP, typename T>
    void highpass(const Array<T>& input, const Array<T>& output, size4_t shape, float cutoff, float width) {
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "The non-redundant FFT with a shape of [logical:{}, pitch:{}] is expected, but got pitch of {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        size4_t input_stride = input.stride();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          input.shape(), output.shape());
            }
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), device);
        }

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::highpass<REMAP>(input.share(), input_stride,
                                              output.share(), output.stride(),
                                              shape, cutoff, width, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::highpass<REMAP>(input.share(), input_stride,
                                               output.share(), output.stride(),
                                               shape, cutoff, width, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Bandpass FFTs.
    /// \tparam REMAP           Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input        FFT to filter. If empty, the filter is written into \p output and \p T must be real.
    /// \param[out] output      Filtered FFT.
    /// \param shape            Rightmost logical shape.
    /// \param cutoff1          First frequency cutoff, in cycle/pix, usually from 0 (DC) to \p cutoff2.
    ///                         At this frequency, the pass is fully recovered.
    /// \param cutoff2          Second frequency cutoff, in cycle/pix, usually from \p cutoff1 to 0.5 (Nyquist).
    ///                         At this frequency, the pass starts to roll-off.
    /// \param width1           Frequency width, in cycle/pix, of the Hann window between 0 and \p cutoff1.
    /// \param width2           Frequency width, in cycle/pix, of the Hann window between \p cutoff2 and 0.5.
    /// \note \p input can be equal to \p output iff there's no remapping, i.e. with H2H or HC2HC.
    template<Remap REMAP, typename T>
    void bandpass(const Array<T>& input, const Array<T>& output, size4_t shape,
                  float cutoff1, float cutoff2, float width1, float width2) {
        NOA_CHECK(all(output.shape() == shape.fft()),
                  "The non-redundant FFT with a shape of [logical:{}, pitch:{}] is expected, but got pitch of {}",
                  shape, shape.fft(), output.shape());

        const Device device = output.device();
        size4_t input_stride = input.stride();
        if (!input.empty()) {
            if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
                NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                          input.shape(), output.shape());
            }
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), device);
        }

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::bandpass<REMAP>(input.share(), input_stride,
                                              output.share(), output.stride(),
                                              shape, cutoff1, cutoff2, width1, width2, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::bandpass<REMAP>(input.share(), input_stride,
                                               output.share(), output.stride(),
                                               shape, cutoff1, cutoff2, width1, width2, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
