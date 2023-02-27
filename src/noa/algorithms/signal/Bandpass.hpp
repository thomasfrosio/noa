#pragma once

#include "noa/core/Types.hpp"
#include "noa/algorithms/Utilities.hpp"

namespace noa::algorithm::signal {
    enum class PassType {
        LOWPASS,
        HIGHPASS,
        BANDPASS
    };

    template<fft::Remap REMAP, PassType PASS, bool SOFT,
             typename Index, typename Offset, typename Value, typename Coord>
    class Bandpass {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_SRC_CENTERED = u8_REMAP & Layout::SRC_CENTERED;
        static_assert(!(u8_REMAP & Layout::SRC_FULL || u8_REMAP & Layout::DST_FULL));

        using index_type = Index;
        using offset_type = Offset;
        using value_type = Value;
        using coord_type = Coord;
        using coord3_type = Vec3<coord_type>;
        using shape2_type = Shape2<index_type>;
        using index3_type = Vec3<index_type>;
        using shape4_type = Shape4<index_type>;
        using real_type = noa::traits::value_type_t<value_type>;

        using input_accessor_type = Accessor<const value_type, 4, offset_type>;
        using output_accessor_type = Accessor<value_type, 4, offset_type>;
        using cutoff_type = std::conditional_t<PASS != PassType::BANDPASS, Vec1<coord_type>, Vec2<coord_type>>;
        using width_type = std::conditional_t<SOFT, cutoff_type, Empty>;

    public:
        template<typename Void = void, typename = std::enable_if_t<std::is_void_v<Void> && PASS != PassType::BANDPASS>>
        Bandpass(const input_accessor_type& input,
                 const output_accessor_type& output,
                 const shape4_type& shape,
                 coord_type cutoff, coord_type width = coord_type{})
                : m_input(input), m_output(output),
                  m_dh_shape(shape.filter(1, 2)) {

            // If odd, subtract 1 to keep Nyquist at 0.5:
            const auto l_shape = shape.pop_front().vec();
            m_norm = coord_type{1} / coord3_type(l_shape / 2 * 2 + index3_type(l_shape == 1));

            if constexpr (SOFT) {
                m_cutoff[0] = cutoff;
                m_width[0] = width;
            } else {
                m_cutoff[0] = cutoff * cutoff;
            }
        }

        template<typename Void = void, typename = std::enable_if_t<std::is_void_v<Void> && PASS == PassType::BANDPASS>>
        Bandpass(const input_accessor_type& input,
                 const output_accessor_type& output,
                 const shape4_type& shape,
                 coord_type cutoff_high,
                 coord_type cutoff_low,
                 coord_type width_high = coord_type{},
                 coord_type width_low = coord_type{})
                : m_input(input), m_output(output),
                  m_dh_shape(shape.filter(1, 2)) {

            // If odd, subtract 1 to keep Nyquist at 0.5:
            const auto l_shape = shape.pop_front().vec();
            m_norm = coord_type{1} / coord3_type(l_shape / 2 * 2 + index3_type(l_shape == 1));

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
            coord3_type frequency(index2frequency<IS_SRC_CENTERED>(ij, m_dh_shape[0]),
                                  index2frequency<IS_SRC_CENTERED>(ik, m_dh_shape[1]),
                                  il);
            frequency *= m_norm;
            const auto frequency_sqd = noa::math::dot(frequency, frequency);
            const auto filter = static_cast<real_type>(get_pass_(frequency_sqd));

            // Compute the index of the current frequency in the output:
            const auto oj = to_output_index<REMAP>(ij, m_dh_shape[0]);
            const auto ol = to_output_index<REMAP>(ik, m_dh_shape[1]);
            m_output(ii, oj, ol, il) = m_input ? m_input(ii, ij, ik, il) * filter : filter;
        }

    private:
        NOA_HD constexpr coord_type get_pass_(coord_type frequency_sqd) const noexcept {
            if constexpr (SOFT) {
                const auto frequency = noa::math::sqrt(frequency_sqd);
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
                coord_type frequency) {
            constexpr coord_type PI = noa::math::Constant<coord_type>::PI;
            coord_type filter;
            if constexpr (FILTER_TYPE == PassType::LOWPASS) {
                if (frequency <= frequency_cutoff) {
                    filter = 1;
                } else if (frequency_cutoff + frequency_width <= frequency) {
                    filter = 0;
                } else {
                    const auto tmp = noa::math::cos(PI * (frequency_cutoff - frequency) / frequency_width);
                    filter = (coord_type{1} + tmp) * coord_type{0.5};
                }
            } else if constexpr (FILTER_TYPE == PassType::HIGHPASS) {
                if (frequency_cutoff <= frequency) {
                    filter = 1;
                } else if (frequency <= frequency_cutoff - frequency_width) {
                    filter = 0;
                } else {
                    const auto tmp = noa::math::cos(PI * (frequency - frequency_cutoff) / frequency_width);
                    filter = (coord_type{1} + tmp) * coord_type{0.5};
                }
            } else {
                static_assert(noa::traits::always_false_v<value_type>);
            }
            return filter;
        }

        template<PassType FILTER_TYPE>
        NOA_HD static constexpr coord_type get_hard_window_(
                coord_type frequency_cutoff_sqd,
                coord_type frequency_sqd) {
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
                static_assert(noa::traits::always_false_v<value_type>);
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

    template<fft::Remap REMAP, bool SOFT,
             typename Index, typename Offset, typename Value, typename Coord>
    auto bandpass(const Value* input, const Strides4<Offset>& input_strides,
                  Value* output, const Strides4<Offset>& output_strides,
                  const Shape4<Index>& shape,
                  Coord cutoff_high, Coord cutoff_low,
                  Coord width_high, Coord width_low) {
        const auto input_accessor = Accessor<const Value, 4, Offset>(input, input_strides);
        const auto output_accessor = Accessor<Value, 4, Offset>(output, output_strides);
        return Bandpass<REMAP, PassType::BANDPASS, SOFT, Index, Offset, Value, Coord>(
                input_accessor, output_accessor, shape, cutoff_high, cutoff_low, width_high, width_low);
    }

    template<fft::Remap REMAP, bool SOFT,
             typename Index, typename Offset, typename Value, typename Coord>
    auto lowpass(const Value* input, const Strides4<Offset>& input_strides,
                 Value* output, const Strides4<Offset>& output_strides,
                 const Shape4<Index>& shape,
                 Coord cutoff, Coord width) {
        const auto input_accessor = Accessor<const Value, 4, Offset>(input, input_strides);
        const auto output_accessor = Accessor<Value, 4, Offset>(output, output_strides);
        return Bandpass<REMAP, PassType::LOWPASS, SOFT, Index, Offset, Value, Coord>(
                input_accessor, output_accessor, shape, cutoff, width);
    }

    template<fft::Remap REMAP, bool SOFT,
             typename Index, typename Offset, typename Value, typename Coord>
    auto highpass(const Value* input, const Strides4<Offset>& input_strides,
                  Value* output, const Strides4<Offset>& output_strides,
                  const Shape4<Index>& shape,
                  Coord cutoff, Coord width) {
        const auto input_accessor = Accessor<const Value, 4, Offset>(input, input_strides);
        const auto output_accessor = Accessor<Value, 4, Offset>(output, output_strides);
        return Bandpass<REMAP, PassType::HIGHPASS, SOFT, Index, Offset, Value, Coord>(
                input_accessor, output_accessor, shape, cutoff, width);
    }
}
