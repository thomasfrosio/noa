#pragma once

#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa::signal::guts {
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
        using coord3_type = Vec3<coord_type>;
        using cutoff_type = std::conditional_t<PASS != BANDPASS, Vec1<coord_type>, Vec2<coord_type>>;
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

        constexpr auto operator()(const Vec<coord_type, 3>& fftfreq_3d, auto) const -> coord_type {
            const coord_type fftfreq_sqd = dot(fftfreq_3d, fftfreq_3d);

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
                static_assert(nt::always_false<>);
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
                static_assert(nt::always_false<>);
            }
            return filter;
        }

    private:
        cutoff_type m_cutoff;
        width_type m_width;
    };
}
