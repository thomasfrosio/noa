#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::signal {
    enum class PassType {
        LOWPASS,
        HIGHPASS,
        BANDPASS
    };

    template<fft::Remap REMAP, PassType PASS, bool SOFT,
             typename Index, typename Coord,
             typename InputAccessor, typename OutputAccessor>
    requires (std::is_integral_v<Index> and
              std::is_floating_point_v<Coord> and
              nt::are_accessor_pure_nd_v<4, InputAccessor, OutputAccessor>)
    class Bandpass {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_SRC_CENTERED = u8_REMAP & Layout::SRC_CENTERED;
        static_assert(not ((u8_REMAP & Layout::SRC_FULL) or (u8_REMAP & Layout::DST_FULL)));

        using index_type = Index;
        using coord_type = Coord;
        using coord3_type = Vec3<coord_type>;
        using shape2_type = Shape2<index_type>;
        using shape4_type = Shape4<index_type>;

        using input_accessor_type = InputAccessor;
        using output_accessor_type = OutputAccessor;
        using input_real_type = nt::value_type_twice_t<input_accessor_type>;
        using output_value_type = nt::value_type_t<output_accessor_type>;
        using cutoff_type = std::conditional_t<PASS != PassType::BANDPASS, Vec1<coord_type>, Vec2<coord_type>>;
        using width_type = std::conditional_t<SOFT, cutoff_type, Empty>;

    public:
        template<typename Void = void> requires (std::is_void_v<Void> and PASS != PassType::BANDPASS)
        Bandpass(
                const input_accessor_type& input,
                const output_accessor_type& output,
                const shape4_type& shape,
                coord_type cutoff,
                coord_type width = coord_type{}
        ): m_input(input), m_output(output),
           m_norm(coord_type{1} / coord3_type::from_vec(shape.pop_front().vec)),
           m_dh_shape(shape.filter(1, 2)) {
            if constexpr (SOFT) {
                m_cutoff[0] = cutoff;
                m_width[0] = width;
            } else {
                m_cutoff[0] = cutoff * cutoff;
            }
        }

        template<typename Void = void>
        requires (std::is_void_v<Void> and PASS == PassType::BANDPASS)
        Bandpass(
                const input_accessor_type& input,
                const output_accessor_type& output,
                const shape4_type& shape,
                coord_type cutoff_high,
                coord_type cutoff_low,
                coord_type width_high = coord_type{},
                coord_type width_low = coord_type{}
        ) : m_input(input), m_output(output),
            m_norm(coord_type{1} / coord3_type::from_vec(shape.pop_front().vec)),
            m_dh_shape(shape.filter(1, 2))
        {
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

        NOA_HD constexpr void operator()(index_type ii, index_type ij, index_type ik, index_type il) const noexcept {
            // Compute the filter value for the current frequency:
            auto frequency = coord3_type::from_values(
                    noa::fft::index2frequency<IS_SRC_CENTERED>(ij, m_dh_shape[0]),
                    noa::fft::index2frequency<IS_SRC_CENTERED>(ik, m_dh_shape[1]),
                    il);
            frequency *= m_norm;
            const auto frequency_sqd = dot(frequency, frequency);
            const auto filter = static_cast<input_real_type>(get_pass_(frequency_sqd));

            // Compute the index of the current frequency in the output:
            const auto oj = noa::fft::remap_index<REMAP>(ij, m_dh_shape[0]);
            const auto ol = noa::fft::remap_index<REMAP>(ik, m_dh_shape[1]);
            const auto value = m_input ? m_input(ii, ij, ik, il) * filter : filter;
            m_output(ii, oj, ol, il) = static_cast<output_value_type>(value);
        }

    private:
        NOA_HD constexpr coord_type get_pass_(coord_type frequency_sqd) const noexcept {
            if constexpr (SOFT) {
                const auto frequency = sqrt(frequency_sqd);
                if constexpr (PASS == PassType::LOWPASS || PASS == PassType::HIGHPASS) {
                    return get_soft_window_<PASS>(m_cutoff[0], m_width[0], frequency);
                } else {
                    return get_soft_window_<PassType::HIGHPASS>(m_cutoff[0], m_width[0], frequency) *
                           get_soft_window_<PassType::LOWPASS>(m_cutoff[1], m_width[1], frequency);
                }
            } else {
                if constexpr (PASS == PassType::LOWPASS || PASS == PassType::HIGHPASS) {
                    return get_hard_window_<PASS>(m_cutoff[0], frequency_sqd);
                } else {
                    return get_hard_window_<PassType::HIGHPASS>(m_cutoff[0], frequency_sqd) *
                           get_hard_window_<PassType::LOWPASS>(m_cutoff[1], frequency_sqd);
                }
            }
        }

        template<PassType FILTER_TYPE>
        NOA_HD static constexpr coord_type get_soft_window_(
                coord_type frequency_cutoff,
                coord_type frequency_width,
                coord_type frequency
        ) {
            constexpr coord_type PI = Constant<coord_type>::PI;
            coord_type filter;
            if constexpr (FILTER_TYPE == PassType::LOWPASS) {
                if (frequency <= frequency_cutoff) {
                    filter = 1;
                } else if (frequency_cutoff + frequency_width <= frequency) {
                    filter = 0;
                } else {
                    const auto tmp = cos(PI * (frequency_cutoff - frequency) / frequency_width);
                    filter = (coord_type{1} + tmp) * coord_type{0.5};
                }
            } else if constexpr (FILTER_TYPE == PassType::HIGHPASS) {
                if (frequency_cutoff <= frequency) {
                    filter = 1;
                } else if (frequency <= frequency_cutoff - frequency_width) {
                    filter = 0;
                } else {
                    const auto tmp = cos(PI * (frequency - frequency_cutoff) / frequency_width);
                    filter = (coord_type{1} + tmp) * coord_type{0.5};
                }
            } else {
                static_assert(nt::always_false_v<coord_type>);
            }
            return filter;
        }

        template<PassType FILTER_TYPE>
        NOA_HD static constexpr coord_type get_hard_window_(
                coord_type frequency_cutoff_sqd,
                coord_type frequency_sqd
        ) {
            coord_type filter;
            if constexpr (FILTER_TYPE == PassType::LOWPASS) {
                if (frequency_cutoff_sqd < frequency_sqd)
                    filter = 0;
                else
                    filter = 1;
            } else if constexpr (FILTER_TYPE == PassType::HIGHPASS) {
                if (frequency_sqd < frequency_cutoff_sqd)
                    filter = 0;
                else
                    filter = 1;
            } else {
                static_assert(nt::always_false_v<coord_type>);
            }
            return filter;
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        coord3_type m_norm;
        shape2_type m_dh_shape;
        cutoff_type m_cutoff;
        width_type m_width;
    };
}
