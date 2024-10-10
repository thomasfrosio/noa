#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/signal/Bandpass.hpp"
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
    template<Remap REMAP, typename Input, typename Output>
    void check_bandpass_parameters(const Input& input, const Output& output, const Shape4<i64>& shape) {
        check(not output.is_empty(), "Empty array detected");

        const auto expected_output_shape = REMAP.is_xx2hx() ? shape.rfft() : shape;
        check(vall(Equal{}, output.shape(), expected_output_shape),
              "Given the logical shape {} and {} remap, the expected output shape should be {}, but got {}",
              shape, REMAP, expected_output_shape, output.shape());

        if (not input.is_empty()) {
            check(output.device() == input.device(),
                  "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
                  input.device(), output.device());

            const auto expected_input_shape = REMAP.is_hx2xx() ? shape.rfft() : shape;
            check(vall(Equal{}, input.shape(), expected_input_shape),
                  "Given the logical shape {} and {} remap, the expected input shape should be {}, but got {}",
                  shape, REMAP, expected_input_shape, input.shape());

            check(not REMAP.has_layout_change() or not ni::are_overlapped(input, output),
                  "In-place remapping is not allowed");
        }
    }
}

namespace noa::signal {
    /// Lowpass FFTs.
    /// \param[in] input    FFT to filter. If empty and real, the filter is written into \p output.
    /// \param[out] output  Filtered FFT.
    /// \param shape        BDHW logical shape.
    /// \param parameters   Lowpass filter parameters.
    /// \note \p input can be equal to \p output iff there's no remapping.
    template<Remap REMAP,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void lowpass(
        Input&& input,
        Output&& output,
        const Shape4<i64>& shape,
        const SinglePassParameters& parameters
    ) {
        guts::check_bandpass_parameters<REMAP>(input, output, shape);

        const Device device = output.device();
        auto input_strides = input.strides();
        if (not input.is_empty() and not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }

        using input_real_t = nt::mutable_value_type_twice_t<Input>;
        using coord_t = std::conditional_t<(sizeof(input_real_t) < 4), f32, input_real_t>;
        const auto cutoff = static_cast<coord_t>(parameters.cutoff);

        using iaccessor_t = AccessorI64<nt::const_value_type_t<Input>, 4>;
        using oaccessor_t = AccessorI64<nt::value_type_t<Output>, 4>;
        auto iaccessor = iaccessor_t(input.get(), input_strides);
        auto oaccessor = oaccessor_t(output.get(), output.strides());

        if (parameters.width > 1e-6) {
            const auto width = static_cast<coord_t>(parameters.width);
            using op_t = guts::Bandpass<REMAP, guts::PassType::LOWPASS, true, i64, coord_t, iaccessor_t, oaccessor_t>;
            iwise(output.shape(), device, op_t(iaccessor, oaccessor, shape, cutoff, width),
                  std::forward<Input>(input), std::forward<Output>(output));
        } else {
            using op_t = guts::Bandpass<REMAP, guts::PassType::LOWPASS, false, i64, coord_t, iaccessor_t, oaccessor_t>;
            iwise(output.shape(), device, op_t(iaccessor, oaccessor, shape, cutoff),
                  std::forward<Input>(input), std::forward<Output>(output));
        }
    }

    /// Highpass FFTs.
    /// \param[in] input    FFT to filter. If empty and real, the filter is written into \p output.
    /// \param[out] output  Filtered FFT.
    /// \param shape        BDHW logical shape.
    /// \param parameters   Highpass filter parameters.
    /// \note \p input can be equal to \p output iff there's no remapping.
    template<Remap REMAP,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void highpass(
        Input&& input,
        Output&& output,
        const Shape4<i64>& shape,
        const SinglePassParameters& parameters
    ) {
        guts::check_bandpass_parameters<REMAP>(input, output, shape);

        const Device device = output.device();
        auto input_strides = input.strides();
        if (not input.is_empty() and not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }

        using input_real_t = nt::mutable_value_type_twice_t<Input>;
        using coord_t = std::conditional_t<(sizeof(input_real_t) < 4), f32, input_real_t>;
        const auto cutoff = static_cast<coord_t>(parameters.cutoff);

        using iaccessor_t = AccessorI64<nt::const_value_type_t<Input>, 4>;
        using oaccessor_t = AccessorI64<nt::value_type_t<Output>, 4>;
        auto iaccessor = iaccessor_t(input.get(), input_strides);
        auto oaccessor = oaccessor_t(output.get(), output.strides());

        if (parameters.width > 1e-6) {
            const auto width = static_cast<coord_t>(parameters.width);
            using op_t = guts::Bandpass<REMAP, guts::PassType::HIGHPASS, true, i64, coord_t, iaccessor_t, oaccessor_t>;
            iwise(output.shape(), device, op_t(iaccessor, oaccessor, shape, cutoff, width),
                  std::forward<Input>(input), std::forward<Output>(output));
        } else {
            using op_t = guts::Bandpass<REMAP, guts::PassType::HIGHPASS, false, i64, coord_t, iaccessor_t, oaccessor_t>;
            iwise(output.shape(), device, op_t(iaccessor, oaccessor, shape, cutoff),
                  std::forward<Input>(input), std::forward<Output>(output));
        }
    }

    /// Bandpass FFTs.
    /// \param[in] input    FFT to filter. If empty and real, the filter is written into \p output.
    /// \param[out] output  Filtered FFT.
    /// \param shape        BDHW logical shape.
    /// \param parameters   Bandpass filter parameters.
    /// \note \p input can be equal to \p output iff there's no remapping.
    template<Remap REMAP,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void bandpass(
        Input&& input,
        Output&& output,
        const Shape4<i64>& shape,
        const DoublePassParameters& parameters
    ) {
        guts::check_bandpass_parameters<REMAP>(input, output, shape);

        const Device device = output.device();
        auto input_strides = input.strides();
        if (not input.is_empty() and not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }

        using input_real_t = nt::mutable_value_type_twice_t<Input>;
        using coord_t = std::conditional_t<(sizeof(input_real_t) < 4), f32, input_real_t>;
        const auto highpass_cutoff = static_cast<coord_t>(parameters.highpass_cutoff);
        const auto lowpass_cutoff = static_cast<coord_t>(parameters.lowpass_cutoff);

        using iaccessor_t = AccessorI64<nt::const_value_type_t<Input>, 4>;
        using oaccessor_t = AccessorI64<nt::value_type_t<Output>, 4>;
        auto iaccessor = iaccessor_t(input.get(), input_strides);
        auto oaccessor = oaccessor_t(output.get(), output.strides());

        if (parameters.highpass_cutoff > 1e-6 or parameters.lowpass_cutoff > 1e-6) {
            const auto highpass_width = static_cast<coord_t>(parameters.highpass_width);
            const auto lowpass_width = static_cast<coord_t>(parameters.lowpass_width);
            using op_t = guts::Bandpass<REMAP, guts::PassType::BANDPASS, true, i64, coord_t, iaccessor_t, oaccessor_t>;
            auto op = op_t(iaccessor, oaccessor, shape, highpass_cutoff, lowpass_cutoff, highpass_width, lowpass_width);
            iwise(output.shape(), device, op, std::forward<Input>(input), std::forward<Output>(output));
        } else {
            using op_t = guts::Bandpass<REMAP, guts::PassType::BANDPASS, false, i64, coord_t, iaccessor_t, oaccessor_t>;
            auto op = op_t(iaccessor, oaccessor, shape, highpass_cutoff, lowpass_cutoff);
            iwise(output.shape(), device, op, std::forward<Input>(input), std::forward<Output>(output));
        }
    }
}
