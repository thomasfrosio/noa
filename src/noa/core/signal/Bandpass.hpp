#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::signal::guts {
    enum class PassType {
        LOWPASS,
        HIGHPASS,
        BANDPASS
    };

    template<Remap REMAP, PassType PASS, bool SOFT,
             nt::integer Index,
             nt::real Coord,
             nt::readable_nd_optional<4> Input,
             nt::writable_nd<4> Output>
    requires (nt::accessor_pure<Input> and (REMAP.is_hx2hx() or REMAP.is_fx2fx()))
    class Bandpass {
    public:
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();
        static constexpr bool IS_SRC_CENTERED = REMAP.is_xc2xx();
        using enum PassType;

        using index_type = Index;
        using coord_type = Coord;
        using coord3_type = Vec3<coord_type>;
        using shape_nd_type = Shape<index_type, 3 - REMAP.is_hx2hx()>;
        using shape4_type = Shape4<index_type>;

        using input_type = Input;
        using output_type = Output;
        using input_real_type = nt::value_type_twice_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using cutoff_type = std::conditional_t<PASS != BANDPASS, Vec1<coord_type>, Vec2<coord_type>>;
        using width_type = std::conditional_t<SOFT, cutoff_type, Empty>;

    public:
        constexpr Bandpass(
            const input_type& input,
            const output_type& output,
            const shape4_type& shape,
            coord_type cutoff,
            coord_type width = coord_type{}
        ) requires (PASS != BANDPASS) :
            m_input(input),
            m_output(output),
            m_norm(coord_type{1} / coord3_type::from_vec(shape.pop_front().vec)),
            m_dh_shape(shape.filter(1, 2, 3).template pop_back<REMAP.is_hx2hx()>())
        {
            if constexpr (SOFT) {
                m_cutoff[0] = cutoff;
                m_width[0] = width;
            } else {
                m_cutoff[0] = cutoff * cutoff;
            }
        }

        constexpr Bandpass(
            const input_type& input,
            const output_type& output,
            const shape4_type& shape,
            coord_type cutoff_high,
            coord_type cutoff_low,
            coord_type width_high = coord_type{},
            coord_type width_low = coord_type{}
        ) requires (PASS == BANDPASS) :
            m_input(input), m_output(output),
            m_norm(coord_type{1} / coord3_type::from_vec(shape.pop_front().vec)),
            m_dh_shape(shape.filter(1, 2, 3).template pop_back<REMAP.is_hx2hx()>())
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

        constexpr void operator()(index_type ii, index_type ij, index_type ik, index_type il) const {
            // Compute the filter value for the current frequency:
            const auto frequency = noa::fft::index2frequency<IS_SRC_CENTERED, IS_RFFT>(Vec{ij, ik, il}, m_dh_shape);
            const auto fftfreq = coord3_type::from_vec(frequency) * m_norm;
            const auto fftfreq_sqd = dot(fftfreq, fftfreq);
            const auto filter = static_cast<input_real_type>(get_pass_(fftfreq_sqd));

            // Compute the index of the current frequency in the output:
            const auto output_indices = noa::fft::remap_indices<REMAP>(Vec{ij, ik, il}, m_dh_shape);
            auto& output = m_output(output_indices.push_front(ii));
            if (m_input)
                output = cast_or_abs_squared<output_value_type>(m_input(ii, ij, ik, il) * filter);
            else
                output = static_cast<output_value_type>(filter);
        }

    private:
        constexpr coord_type get_pass_(coord_type frequency_sqd) const {
            if constexpr (SOFT) {
                const auto frequency = sqrt(frequency_sqd);
                if constexpr (PASS == LOWPASS or PASS == HIGHPASS) {
                    return get_soft_window_<PASS>(m_cutoff[0], m_width[0], frequency);
                } else {
                    return get_soft_window_<HIGHPASS>(m_cutoff[0], m_width[0], frequency) *
                           get_soft_window_<LOWPASS>(m_cutoff[1], m_width[1], frequency);
                }
            } else {
                if constexpr (PASS == LOWPASS or PASS == HIGHPASS) {
                    return get_hard_window_<PASS>(m_cutoff[0], frequency_sqd);
                } else {
                    return get_hard_window_<HIGHPASS>(m_cutoff[0], frequency_sqd) *
                           get_hard_window_<LOWPASS>(m_cutoff[1], frequency_sqd);
                }
            }
        }

        template<PassType FILTER_TYPE>
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

        template<PassType FILTER_TYPE>
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
        input_type m_input;
        output_type m_output;
        coord3_type m_norm;
        shape_nd_type m_dh_shape;
        cutoff_type m_cutoff;
        width_type m_width;
    };
}
