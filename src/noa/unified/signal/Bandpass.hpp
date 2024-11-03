#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/signal/Bandpass.hpp"
#include "noa/unified/signal/FilterSpectrum.hpp"

namespace noa::signal {
    /// Lowpass or highpass filter parameters, specified in fftfreq.
    struct Lowpass {
        /// Frequency cutoff, in cycle/pix.
        /// At this frequency, the lowpass starts to roll-off, and the highpass is fully recovered.
        f64 cutoff;

        /// Width of the Hann window, in cycle/pix.
        f64 width;
    };
    using Highpass = Lowpass;

    /// Bandpass filter parameters, specified in fftfreq.
    struct Bandpass {
        f64 highpass_cutoff;
        f64 highpass_width;
        f64 lowpass_cutoff;
        f64 lowpass_width;
    };
}

namespace noa::signal {
    /// Lowpass FFTs.
    /// \param[in] input    Spectrum to filter. If empty, the filter is written into the output.
    /// \param[out] output  Filtered spectrum. Can be equal to the input (in-place filtering) if there's no remapping.
    ///                     If real and the filtered input is complex, the power spectrum of the filter input is saved.
    /// \param shape        BDHW logical shape.
    /// \param pass         Lowpass filter parameters.
    template<Remap REMAP,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void lowpass(
        Input&& input,
        Output&& output,
        const Shape4<i64>& shape,
        const Lowpass& pass
    ) {
        using coord_t = guts::filter_spectrum_default_coord_t<Input>;
        const auto cutoff = static_cast<coord_t>(pass.cutoff);

        if (pass.width > 1e-6) {
            const auto width = static_cast<coord_t>(pass.width);
            const auto filter = guts::Bandpass<guts::BandpassType::LOWPASS, true, coord_t>(cutoff, width);
            filter_spectrum<REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape, filter);
        } else {
            const auto filter = guts::Bandpass<guts::BandpassType::LOWPASS, false, coord_t>(cutoff);
            filter_spectrum<REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape, filter);
        }
    }

    /// Highpass FFTs.
    /// \param[in] input    Spectrum to filter. If empty, the filter is written into the output.
    /// \param[out] output  Filtered spectrum. Can be equal to the input (in-place filtering) if there's no remapping.
    ///                     If real and the filtered input is complex, the power spectrum of the filter input is saved.
    /// \param shape        BDHW logical shape.
    /// \param pass         Highpass filter parameters.
    template<Remap REMAP,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void highpass(
        Input&& input,
        Output&& output,
        const Shape4<i64>& shape,
        const Highpass& pass
    ) {
        using coord_t = guts::filter_spectrum_default_coord_t<Input>;
        const auto cutoff = static_cast<coord_t>(pass.cutoff);

        if (pass.width > 1e-6) {
            const auto width = static_cast<coord_t>(pass.width);
            const auto filter = guts::Bandpass<guts::BandpassType::HIGHPASS, true, coord_t>(cutoff, width);
            filter_spectrum<REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape, filter);
        } else {
            const auto filter = guts::Bandpass<guts::BandpassType::HIGHPASS, false, coord_t>(cutoff);
            filter_spectrum<REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape, filter);
        }
    }

    /// Bandpass FFTs.
    /// \param[in] input    Spectrum to filter. If empty, the filter is written into the output.
    /// \param[out] output  Filtered spectrum. Can be equal to the input (in-place filtering) if there's no remapping.
    ///                     If real and the filtered input is complex, the power spectrum of the filter input is saved.
    /// \param shape        BDHW logical shape.
    /// \param pass         Bandpass filter parameters.
    template<Remap REMAP,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Input = View<nt::const_value_type_t<Output>>>
    requires (nt::varray_decay_with_spectrum_types<Input, Output> and REMAP.is_hx2hx() or REMAP.is_fx2fx())
    void bandpass(
        Input&& input,
        Output&& output,
        const Shape4<i64>& shape,
        const Bandpass& pass
    ) {
        using coord_t = guts::filter_spectrum_default_coord_t<Input>;
        const auto highpass_cutoff = static_cast<coord_t>(pass.highpass_cutoff);
        const auto lowpass_cutoff = static_cast<coord_t>(pass.lowpass_cutoff);

        if (pass.highpass_cutoff > 1e-6 or pass.lowpass_cutoff > 1e-6) {
            const auto highpass_width = static_cast<coord_t>(pass.highpass_width);
            const auto lowpass_width = static_cast<coord_t>(pass.lowpass_width);
            using filter_t = guts::Bandpass<guts::BandpassType::BANDPASS, true, coord_t>;
            auto filter = filter_t(highpass_cutoff, lowpass_cutoff, highpass_width, lowpass_width);
            filter_spectrum<REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape, filter);
        } else {
            using filter_t = guts::Bandpass<guts::BandpassType::BANDPASS, false, coord_t>;
            auto filter = filter_t(highpass_cutoff, lowpass_cutoff);
            filter_spectrum<REMAP>(std::forward<Input>(input), std::forward<Output>(output), shape, filter);
        }
    }
}
