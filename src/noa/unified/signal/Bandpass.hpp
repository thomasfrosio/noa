#pragma once

#include "noa/core/signal/Bandpass.hpp"
#include "noa/core/fft/RemapInterface.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa::signal {
    /// Lowpass or highpass filter parameters, specified in fftfreq.
    struct SinglePassParameters {
        /// Frequency cutoff, in cycle/pix.
        /// At this frequency, the lowpass starts to roll-off, and the highpass is fully recovered.
        f64 cutoff;

        /// Width of the Hann window, in cycle/pix.
        f64 width;
    };

    /// Bandpass filter parameters, specified in fftfreq.
    struct DoublePassParameters {
        f64 highpass_cutoff;
        f64 highpass_width;
        f64 lowpass_cutoff;
        f64 lowpass_width;
    };
}

namespace noa::signal::guts {
    template<noa::fft::Remap REMAP>
    constexpr bool is_valid_pass_remap_v =
            (REMAP == noa::fft::Remap::H2H or REMAP == noa::fft::Remap::H2HC or
             REMAP == noa::fft::Remap::HC2H or REMAP == noa::fft::Remap::HC2HC);

    template<noa::fft::Remap REMAP, typename Input, typename Output>
    void check_bandpass_parameters(const Input& input, const Output& output, const Shape4<i64>& shape) {
        check(not output.is_empty(), "Empty array detected");
        check(all(output.shape() == shape.rfft()),
              "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
              shape, shape.rfft(), output.shape());

        if (not input.is_empty()) {
            check(output.device() == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), output.device());
            check(REMAP == noa::fft::Remap::H2H or REMAP == noa::fft::Remap::HC2HC or input.get() != output.get(),
                  "In-place remapping is not allowed");
        }
    }
}

namespace noa::signal {
    /// Lowpass FFTs.
    /// \tparam REMAP       Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \param[in] input    FFT to filter. If empty and real, the filter is written into \p output.
    /// \param[out] output  Filtered FFT.
    /// \param shape        BDHW logical shape.
    /// \param parameters   Lowpass filter parameters.
    /// \note \p input can be equal to \p output iff there's no remapping, i.e. with H2H or HC2HC.
    template<noa::fft::RemapInterface REMAP, typename Output, typename Input = View<const nt::value_type_t<Output>>>
    requires (nt::are_varray_of_real_or_complex_v<Output, Input> and guts::is_valid_pass_remap_v<REMAP.remap>)
    void lowpass(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const SinglePassParameters& parameters
    ) {
        guts::check_bandpass_parameters<REMAP.remap>(input, output, shape);

        const Device device = output.device();
        auto input_strides = input.strides();
        if (not input.is_empty() and not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }

        using input_value_t = nt::mutable_value_type_t<Input>;
        using input_real_t = nt::value_type_t<input_value_t>;
        using output_value_t = nt::value_type_t<Output>;
        constexpr auto LOWPASS = guts::PassType::LOWPASS;
        const auto cutoff = static_cast<input_real_t>(parameters.cutoff);
        const auto width = static_cast<input_real_t>(parameters.width);

        using iaccessor_t = AccessorI64<const input_value_t, 4>;
        using oaccessor_t = AccessorI64<output_value_t, 4>;
        auto iaccessor = iaccessor_t(input.get(), input.strides());
        auto oaccessor = oaccessor_t(output.get(), output.strides());

        if (width > 1e-6f) {
            using op_t = Bandpass<REMAP.remap, LOWPASS, true, i64, input_real_t, iaccessor_t, oaccessor_t>;
            iwise(shape.rfft(), device, op_t(iaccessor, oaccessor, shape, cutoff, width), input, output);
        } else {
            using op_t = Bandpass<REMAP.remap, LOWPASS, false, i64, input_real_t, iaccessor_t, oaccessor_t>;
            iwise(shape.rfft(), device, op_t(iaccessor, oaccessor, shape, cutoff), input, output);
        }
    }

    /// Highpass FFTs.
    /// \tparam REMAP       Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \param[in] input    FFT to filter. If empty and real, the filter is written into \p output.
    /// \param[out] output  Filtered FFT.
    /// \param shape        BDHW logical shape.
    /// \param parameters   Highpass filter parameters.
    /// \note \p input can be equal to \p output iff there's no remapping, i.e. with H2H or HC2HC.
    template<noa::fft::RemapInterface REMAP, typename Output, typename Input = View<const nt::value_type_t<Output>>>
    requires (nt::are_varray_of_real_or_complex_v<Output, Input> and guts::is_valid_pass_remap_v<REMAP.remap>)
    void highpass(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const SinglePassParameters& parameters
    ) {
        guts::check_bandpass_parameters<REMAP.remap>(input, output, shape);

        const Device device = output.device();
        auto input_strides = input.strides();
        if (not input.is_empty() and not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }

        using input_value_t = nt::mutable_value_type_t<Input>;
        using input_real_t = nt::value_type_t<input_value_t>;
        using output_value_t = nt::value_type_t<Output>;
        constexpr auto HIGHPASS = guts::PassType::HIGHPASS;
        const auto cutoff = static_cast<input_real_t>(parameters.cutoff);
        const auto width = static_cast<input_real_t>(parameters.width);

        using iaccessor_t = AccessorI64<const input_value_t, 4>;
        using oaccessor_t = AccessorI64<output_value_t, 4>;
        auto iaccessor = iaccessor_t(input.get(), input.strides());
        auto oaccessor = oaccessor_t(output.get(), output.strides());

        if (width > 1e-6f) {
            using op_t = Bandpass<REMAP.remap, HIGHPASS, true, i64, input_real_t, iaccessor_t, oaccessor_t>;
            iwise(shape.rfft(), device, op_t(iaccessor, oaccessor, shape, cutoff, width), input, output);
        } else {
            using op_t = Bandpass<REMAP.remap, HIGHPASS, false, i64, input_real_t, iaccessor_t, oaccessor_t>;
            iwise(shape.rfft(), device, op_t(iaccessor, oaccessor, shape, cutoff), input, output);
        }
    }

    /// Bandpass FFTs.
    /// \tparam REMAP       Remapping. One of H2H, H2HC, HC2H, HC2HC.
    /// \param[in] input    FFT to filter. If empty and real, the filter is written into \p output.
    /// \param[out] output  Filtered FFT.
    /// \param shape        BDHW logical shape.
    /// \param parameters   Bandpass filter parameters.
    /// \note \p input can be equal to \p output iff there's no remapping, i.e. with H2H or HC2HC.
    template<noa::fft::RemapInterface REMAP, typename Output, typename Input = View<const nt::value_type_t<Output>>>
    requires (nt::are_varray_of_real_or_complex_v<Output, Input> and guts::is_valid_pass_remap_v<REMAP.remap>)
    void bandpass(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const DoublePassParameters& parameters
    ) {
        guts::check_bandpass_parameters<REMAP.remap>(input, output, shape);

        const Device device = output.device();
        auto input_strides = input.strides();
        if (not input.is_empty() and not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }

        using input_value_t = nt::mutable_value_type_t<Input>;
        using input_real_t = nt::value_type_t<input_value_t>;
        using output_value_t = nt::value_type_t<Output>;
        constexpr auto BANDPASS = guts::PassType::BANDPASS;
        const auto highpass_cutoff = static_cast<input_real_t>(parameters.highpass_cutoff);
        const auto lowpass_cutoff = static_cast<input_real_t>(parameters.lowpass_cutoff);
        const auto highpass_width = static_cast<input_real_t>(parameters.highpass_width);
        const auto lowpass_width = static_cast<input_real_t>(parameters.lowpass_width);

        using iaccessor_t = AccessorI64<const input_value_t, 4>;
        using oaccessor_t = AccessorI64<output_value_t, 4>;
        auto iaccessor = iaccessor_t(input.get(), input.strides());
        auto oaccessor = oaccessor_t(output.get(), output.strides());

        if (highpass_cutoff > 1e-6f || lowpass_cutoff > 1e-6f) {
            using op_t = Bandpass<REMAP.remap, BANDPASS, true, i64, input_real_t, iaccessor_t, oaccessor_t>;
            auto op = op_t(iaccessor, oaccessor, shape, highpass_cutoff, lowpass_cutoff, highpass_width, lowpass_width);
            iwise(shape.rfft(), device, op, input, output);
        } else {
            using op_t = Bandpass<REMAP.remap, BANDPASS, false, i64, input_real_t, iaccessor_t, oaccessor_t>;
            auto op = op_t(iaccessor, oaccessor, shape, highpass_cutoff, lowpass_cutoff);
            iwise(shape.rfft(), device, op, input, output);
        }
    }
}
