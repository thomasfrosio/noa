#pragma once

#include "noa/cpu/signal/fft/Bandpass.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/Bandpass.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::signal::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP>
    constexpr bool is_valid_pass_remap_v = (REMAP == H2H || REMAP == H2HC || REMAP == HC2H || REMAP == HC2HC);

    template<Remap REMAP, typename Input, typename Output>
    void check_bandpass_parameters(const Input& input, const Output& output, const Shape4<i64>& shape) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        NOA_CHECK(noa::all(output.shape() == shape.rfft()),
                  "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
                  shape, shape.rfft(), output.shape());

        if (!input.is_empty()) {
            NOA_CHECK(output.device() == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), output.device());
            NOA_CHECK(REMAP == Remap::H2H || REMAP == Remap::HC2HC || input.get() != output.get(),
                      "In-place remapping is not allowed");
        }
    }
}

namespace noa::signal::fft {
    using noa::fft::Remap;

    /// Lowpass FFTs.
    /// \tparam REMAP       Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \param[in] input    FFT to filter. If empty and real, the filter is written into \p output.
    /// \param[out] output  Filtered FFT.
    /// \param shape        BDHW logical shape.
    /// \param cutoff       Frequency cutoff, in cycle/pix, usually from 0 (DC) to 0.5 (Nyquist).
    ///                     At this frequency, the pass starts to roll-off.
    /// \param width        Width of the Hann window, in cycle/pix.
    /// \note \p input can be equal to \p output iff there's no remapping, i.e. with H2H or HC2HC.
    template<Remap REMAP, typename Output,
             typename Input = View<const noa::traits::value_type_t<Output>>, typename = std::enable_if_t<
             noa::traits::are_array_or_view_of_real_or_complex_v<Input, Output> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             details::is_valid_pass_remap_v<REMAP>>>
    void lowpass(const Input& input, const Output& output, const Shape4<i64>& shape, f32 cutoff, f32 width) {
        details::check_bandpass_parameters<REMAP>(input, output, shape);

        const Device device = output.device();
        auto input_strides = input.strides();
        if (!input.is_empty() && !indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::signal::fft::lowpass<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(),
                        shape, cutoff, width, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::fft::lowpass<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(),
                    shape, cutoff, width, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Highpass FFTs.
    /// \tparam REMAP       Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \param[in] input    FFT to filter. If empty and real, the filter is written into \p output.
    /// \param[out] output  Filtered FFT.
    /// \param shape        BDHW logical shape.
    /// \param cutoff       Frequency cutoff, in cycle/pix, usually from 0 (DC) to 0.5 (Nyquist).
    ///                     At this frequency, the pass is fully recovered.
    /// \param width        Width of the Hann window, in cycle/pix.
    /// \note \p input can be equal to \p output iff there's no remapping, i.e. with H2H or HC2HC.
    template<Remap REMAP, typename Output,
             typename Input = View<const noa::traits::value_type_t<Output>>, typename = std::enable_if_t<
             noa::traits::are_array_or_view_of_real_or_complex_v<Input, Output> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             details::is_valid_pass_remap_v<REMAP>>>
    void highpass(const Input& input, const Output& output, const Shape4<i64>& shape, f32 cutoff, f32 width) {
        details::check_bandpass_parameters<REMAP>(input, output, shape);

        const Device device = output.device();
        auto input_strides = input.strides();
        if (!input.is_empty() && !indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::signal::fft::highpass<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(),
                        shape, cutoff, width, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::fft::highpass<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(),
                    shape, cutoff, width, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Bandpass FFTs.
    /// \tparam REMAP       Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \param[in] input    FFT to filter. If empty and real, the filter is written into \p output.
    /// \param[out] output  Filtered FFT.
    /// \param shape        BDHW logical shape.
    /// \param cutoff_high  First frequency cutoff, in cycle/pix, usually from 0 (DC) to \p cutoff_low.
    ///                     At this frequency, the pass is fully recovered.
    /// \param cutoff_low   Second frequency cutoff, in cycle/pix, usually from \p cutoff_high to 0.5 (Nyquist).
    ///                     At this frequency, the pass starts to roll-off.
    /// \param width_high   Frequency width, in cycle/pix, of the Hann window between 0 and \p cutoff_high.
    /// \param width_low    Frequency width, in cycle/pix, of the Hann window between \p cutoff_low and 0.5.
    /// \note \p input can be equal to \p output iff there's no remapping, i.e. with H2H or HC2HC.
    template<Remap REMAP, typename Output,
             typename Input = View<const noa::traits::value_type_t<Output>>, typename = std::enable_if_t<
             noa::traits::are_array_or_view_of_real_or_complex_v<Input, Output> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             details::is_valid_pass_remap_v<REMAP>>>
    void bandpass(const Input& input, const Output& output, const Shape4<i64>& shape,
                  f32 cutoff_high, f32 cutoff_low, f32 width_high, f32 width_low) {
        details::check_bandpass_parameters<REMAP>(input, output, shape);

        const Device device = output.device();
        auto input_strides = input.strides();
        if (!input.is_empty() && !indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::signal::fft::bandpass<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(),
                        shape, cutoff_high, cutoff_low, width_high, width_low, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::fft::bandpass<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(),
                    shape, cutoff_high, cutoff_low, width_high, width_low, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
