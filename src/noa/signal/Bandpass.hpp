#pragma once

#include "noa/fft/core/Layout.hpp"
#include "noa/signal/FilterSpectrum.hpp"

namespace noa::signal::details {
    enum class BandpassType {
        LOWPASS,
        HIGHPASS,
        BANDPASS
    };

    template<BandpassType PASS, bool SOFT, nt::real Coord>
    class Bandpass {
    public:
        using enum BandpassType;
        using coord_type = Coord;
        using cutoff_type = std::conditional_t<PASS != BANDPASS, Vec<coord_type, 1>, Vec<coord_type, 2>>;
        using width_type = std::conditional_t<SOFT, cutoff_type, Empty>;

    public:
        constexpr explicit Bandpass(
            coord_type cutoff,
            coord_type width = coord_type{}
        ) requires (PASS != BANDPASS) {
            if constexpr (SOFT) {
                m_cutoff[0] = cutoff;
                m_width[0] = width;
            } else {
                m_cutoff[0] = cutoff * cutoff;
            }
        }

        constexpr Bandpass(
            coord_type cutoff_high,
            coord_type cutoff_low,
            coord_type width_high = coord_type{},
            coord_type width_low = coord_type{}
        ) requires (PASS == BANDPASS) {
            if constexpr (SOFT) {
                m_cutoff[0] = cutoff_high;
                m_cutoff[1] = cutoff_low;
                m_width[0] = width_high;
                m_width[1] = width_low;
            } else {
                m_cutoff[0] = cutoff_high * cutoff_high;
                m_cutoff[1] = cutoff_low * cutoff_low;
            }
        }

        template<usize R> requires (R >= 1 and R <= 3)
        constexpr auto operator()(const Vec<coord_type, R>& fftfreq_r, auto) const -> coord_type {
            const coord_type fftfreq_sqd = dot(fftfreq_r, fftfreq_r);

            if constexpr (SOFT) {
                const auto fftfreq = sqrt(fftfreq_sqd);
                if constexpr (PASS == LOWPASS or PASS == HIGHPASS) {
                    return get_soft_window_<PASS>(m_cutoff[0], m_width[0], fftfreq);
                } else {
                    return get_soft_window_<HIGHPASS>(m_cutoff[0], m_width[0], fftfreq) *
                           get_soft_window_<LOWPASS>(m_cutoff[1], m_width[1], fftfreq);
                }
            } else {
                if constexpr (PASS == LOWPASS or PASS == HIGHPASS) {
                    return get_hard_window_<PASS>(m_cutoff[0], fftfreq_sqd);
                } else {
                    return get_hard_window_<HIGHPASS>(m_cutoff[0], fftfreq_sqd) *
                           get_hard_window_<LOWPASS>(m_cutoff[1], fftfreq_sqd);
                }
            }
        }

    private:
        template<BandpassType FILTER_TYPE>
        static constexpr coord_type get_soft_window_(
            coord_type frequency_cutoff,
            coord_type frequency_width,
            coord_type frequency
        ) {
            constexpr coord_type PI = Constant<coord_type>::PI;
            coord_type filter;
            if constexpr (FILTER_TYPE == LOWPASS) {
                if (frequency <= frequency_cutoff) {
                    filter = 1;
                } else if (frequency_cutoff + frequency_width <= frequency) {
                    filter = 0;
                } else {
                    const auto tmp = cos(PI * (frequency_cutoff - frequency) / frequency_width);
                    filter = (coord_type{1} + tmp) * coord_type{0.5};
                }
            } else if constexpr (FILTER_TYPE == HIGHPASS) {
                if (frequency_cutoff <= frequency) {
                    filter = 1;
                } else if (frequency <= frequency_cutoff - frequency_width) {
                    filter = 0;
                } else {
                    const auto tmp = cos(PI * (frequency - frequency_cutoff) / frequency_width);
                    filter = (coord_type{1} + tmp) * coord_type{0.5};
                }
            } else {
                static_assert(nt::always_false<coord_type>);
            }
            return filter;
        }

        template<BandpassType FILTER_TYPE>
        static constexpr coord_type get_hard_window_(
            coord_type frequency_cutoff_sqd,
            coord_type frequency_sqd
        ) {
            coord_type filter;
            if constexpr (FILTER_TYPE == LOWPASS) {
                if (frequency_cutoff_sqd < frequency_sqd)
                    filter = 0;
                else
                    filter = 1;
            } else if constexpr (FILTER_TYPE == HIGHPASS) {
                if (frequency_sqd < frequency_cutoff_sqd)
                    filter = 0;
                else
                    filter = 1;
            } else {
                static_assert(nt::always_false<coord_type>);
            }
            return filter;
        }

    private:
        cutoff_type m_cutoff;
        width_type m_width;
    };
}

namespace noa::signal {
    /// Lowpass or highpass filter parameters, specified in fftfreq.
    struct Lowpass {
        /// fftfreq cutoff.
        /// At this frequency, the lowpass starts to roll-off, and the highpass is fully recovered.
        f64 cutoff;

        /// Width of the Hann window, in fftfreq.
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
    /// \tparam R:
    ///     Rank of the transform.
    /// \param[in] input:
    ///     Spectrum to filter.
    ///     If empty, the filter is written into the output.
    /// \param[out] output:
    ///     Filtered spectrum.
    ///     Can be equal to the input (in-place filtering) if there's no remapping.
    ///     If real and the filtered input is complex, the power spectrum of the filter input is saved.
    /// \param shape    Logical shape.
    /// \param pass     Lowpass filter parameters.
    /// \param options  Spectrum options.
    template<nf::Layout REMAP, usize R = 3, typename Output, typename Input = Output, usize N>
        requires details::filter_spectrum_able<REMAP, R, Input, Output, N>
    void lowpass(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        const Lowpass& pass,
        FilterSpectrumOptions options = {}
    ) {
        using coord_t = details::filter_spectrum_default_coord_t<Input>;
        const auto cutoff = static_cast<coord_t>(pass.cutoff);

        if (pass.width > 1e-6) {
            const auto width = static_cast<coord_t>(pass.width);
            const auto filter = details::Bandpass<details::BandpassType::LOWPASS, true, coord_t>(cutoff, width);
            filter_spectrum<REMAP, R>(std::forward<Input>(input), std::forward<Output>(output), shape, filter, options);
        } else {
            const auto filter = details::Bandpass<details::BandpassType::LOWPASS, false, coord_t>(cutoff);
            filter_spectrum<REMAP, R>(std::forward<Input>(input), std::forward<Output>(output), shape, filter, options);
        }
    }

    template<nf::Layout REMAP, typename Output, typename Input = Output, usize N>
    void lowpass_1d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        const Lowpass& pass,
        FilterSpectrumOptions options = {}
    ) {
        lowpass<REMAP, 1>(input, output, shape, pass, options);
    }

    template<nf::Layout REMAP, typename Output, typename Input = Output, usize N>
    void lowpass_2d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        const Lowpass& pass,
        FilterSpectrumOptions options = {}
    ) {
        lowpass<REMAP, 2>(input, output, shape, pass, options);
    }

    template<nf::Layout REMAP, typename Output, typename Input = Output, usize N>
    void lowpass_3d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        const Lowpass& pass,
        FilterSpectrumOptions options = {}
    ) {
        lowpass<REMAP, 3>(input, output, shape, pass, options);
    }

    /// Highpass FFTs.
    /// \tparam R:
    ///     Rank of the transform.
    /// \param[in] input:
    ///     Spectrum to filter.
    ///     If empty, the filter is written into the output.
    /// \param[out] output:
    ///     Filtered spectrum.
    ///     Can be equal to the input (in-place filtering) if there's no remapping.
    ///     If real and the filtered input is complex, the power spectrum of the filter input is saved.
    /// \param shape    Logical shape.
    /// \param pass     Highpass filter parameters.
    /// \param options  Spectrum options.
    template<nf::Layout REMAP, usize R = 3, typename Output, typename Input = Output, usize N>
        requires details::filter_spectrum_able<REMAP, R, Input, Output, N>
    void highpass(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        const Highpass& pass,
        FilterSpectrumOptions options = {}
    ) {
        using coord_t = details::filter_spectrum_default_coord_t<Input>;
        const auto cutoff = static_cast<coord_t>(pass.cutoff);

        if (pass.width > 1e-6) {
            const auto width = static_cast<coord_t>(pass.width);
            const auto filter = details::Bandpass<details::BandpassType::HIGHPASS, true, coord_t>(cutoff, width);
            filter_spectrum<REMAP, R>(std::forward<Input>(input), std::forward<Output>(output), shape, filter, options);
        } else {
            const auto filter = details::Bandpass<details::BandpassType::HIGHPASS, false, coord_t>(cutoff);
            filter_spectrum<REMAP, R>(std::forward<Input>(input), std::forward<Output>(output), shape, filter, options);
        }
    }

    template<nf::Layout REMAP, typename Output, typename Input = Output, usize N>
    void highpass_1d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        const Highpass& pass,
        FilterSpectrumOptions options = {}
    ) {
        highpass<REMAP, 1>(input, output, shape, pass, options);
    }

    template<nf::Layout REMAP, typename Output, typename Input = Output, usize N>
    void highpass_2d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        const Highpass& pass,
        FilterSpectrumOptions options = {}
    ) {
        highpass<REMAP, 2>(input, output, shape, pass, options);
    }

    template<nf::Layout REMAP, typename Output, typename Input = Output, usize N>
    void highpass_3d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        const Highpass& pass,
        FilterSpectrumOptions options = {}
    ) {
        highpass<REMAP, 3>(input, output, shape, pass, options);
    }

    /// Bandpass FFTs.
    /// \tparam R:
    ///     Rank of the transform.
    /// \param[in] input:
    ///     Spectrum to filter.
    ///     If empty, the filter is written into the output.
    /// \param[out] output:
    ///     Filtered spectrum.
    ///     Can be equal to the input (in-place filtering) if there's no remapping.
    ///     If real and the filtered input is complex, the power spectrum of the filter input is saved.
    /// \param shape    Logical shape.
    /// \param pass     Bandpass filter parameters.
    /// \param options  Spectrum options.
    template<nf::Layout REMAP, usize R = 3, typename Output, typename Input = Output, usize N>
        requires details::filter_spectrum_able<REMAP, R, Input, Output, N>
    void bandpass(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        const Bandpass& pass,
        FilterSpectrumOptions options = {}
    ) {
        using coord_t = details::filter_spectrum_default_coord_t<Input>;
        const auto highpass_cutoff = static_cast<coord_t>(pass.highpass_cutoff);
        const auto lowpass_cutoff = static_cast<coord_t>(pass.lowpass_cutoff);

        if (pass.highpass_cutoff > 1e-6 or pass.lowpass_cutoff > 1e-6) {
            const auto highpass_width = static_cast<coord_t>(pass.highpass_width);
            const auto lowpass_width = static_cast<coord_t>(pass.lowpass_width);
            using filter_t = details::Bandpass<details::BandpassType::BANDPASS, true, coord_t>;
            auto filter = filter_t(highpass_cutoff, lowpass_cutoff, highpass_width, lowpass_width);
            filter_spectrum<REMAP, R>(std::forward<Input>(input), std::forward<Output>(output), shape, filter, options);
        } else {
            using filter_t = details::Bandpass<details::BandpassType::BANDPASS, false, coord_t>;
            auto filter = filter_t(highpass_cutoff, lowpass_cutoff);
            filter_spectrum<REMAP, R>(std::forward<Input>(input), std::forward<Output>(output), shape, filter, options);
        }
    }

    template<nf::Layout REMAP, typename Output, typename Input = Output, usize N>
    void bandpass_1d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        const Bandpass& pass,
        FilterSpectrumOptions options = {}
    ) {
        bandpass<REMAP, 1>(input, output, shape, pass, options);
    }

    template<nf::Layout REMAP, typename Output, typename Input = Output, usize N>
    void bandpass_2d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        const Bandpass& pass,
        FilterSpectrumOptions options = {}
    ) {
        bandpass<REMAP, 2>(input, output, shape, pass, options);
    }

    template<nf::Layout REMAP, typename Output, typename Input = Output, usize N>
    void bandpass_3d(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        const Bandpass& pass,
        FilterSpectrumOptions options = {}
    ) {
        bandpass<REMAP, 3>(input, output, shape, pass, options);
    }
}
