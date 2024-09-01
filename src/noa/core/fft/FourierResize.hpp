#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa::fft::guts {
    enum class FourierResizeMode {
        PAD_H2H,
        PAD_F2F,
        CROP_H2H,
        CROP_F2F
    };

    template<FourierResizeMode MODE,
             nt::sinteger Index,
             nt::readable_nd<4> Input,
             nt::writable_nd<4> Output>
    class FourierResize {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::compatible_or_spectrum_types<input_value_type, output_value_type>);

        using dh_shape_type = std::conditional_t<
                MODE == FourierResizeMode::CROP_H2H or MODE == FourierResizeMode::PAD_H2H,
                Shape2<index_type>, Empty>;
        using dhw_vec_type = std::conditional_t<
                MODE == FourierResizeMode::CROP_F2F or MODE == FourierResizeMode::PAD_F2F,
                Shape3<index_type>, Empty>;

        constexpr FourierResize(
            const input_type& input,
            const output_type& output,
            const Shape3<index_type>& input_shape,
            const Shape3<index_type>& output_shape
        ) : m_input(input),
            m_output(output)
        {
            if constexpr (MODE == FourierResizeMode::CROP_H2H) {
                m_input_shape = input_shape.pop_back();
                m_output_shape = output_shape.pop_back();

            } else if constexpr (MODE == FourierResizeMode::PAD_H2H) {
                m_input_shape = input_shape.pop_back();
                m_output_shape = output_shape.pop_back();

            } else if constexpr (MODE == FourierResizeMode::CROP_F2F) {
                m_offset = input_shape - output_shape;
                m_limit = (output_shape + 1) / 2;

            } else if constexpr (MODE == FourierResizeMode::PAD_F2F) {
                m_offset = output_shape - input_shape;
                m_limit = (input_shape + 1) / 2;
            }
        }

        constexpr void operator()(index_type i, index_type j, index_type k, index_type l) const {
            if constexpr (MODE == FourierResizeMode::CROP_H2H) {
                const auto ij = j < (m_output_shape[0] + 1) / 2 ? j : j + m_input_shape[0] - m_output_shape[0];
                const auto ik = k < (m_output_shape[1] + 1) / 2 ? k : k + m_input_shape[1] - m_output_shape[1];
                m_output(i, j, k, l) = cast_or_abs_squared<output_value_type>(m_input(i, ij, ik, l));

            } else if constexpr (MODE == FourierResizeMode::PAD_H2H) {
                const auto oj = j < (m_input_shape[0] + 1) / 2 ? j : j + m_output_shape[0] - m_input_shape[0];
                const auto ok = k < (m_input_shape[1] + 1) / 2 ? k : k + m_output_shape[1] - m_input_shape[1];
                m_output(i, oj, ok, l) = cast_or_abs_squared<output_value_type>(m_input(i, j, k, l));

            } else if constexpr (MODE == FourierResizeMode::CROP_F2F) {
                const auto ij = j < m_limit[0] ? j : j + m_offset[0];
                const auto ik = k < m_limit[1] ? k : k + m_offset[1];
                const auto il = l < m_limit[2] ? l : l + m_offset[2];
                m_output(i, j, k, l) =  cast_or_abs_squared<output_value_type>(m_input(i, ij, ik, il));

            } else if constexpr (MODE == FourierResizeMode::PAD_F2F) {
                const auto oj = j < m_limit[0] ? j : j + m_offset[0];
                const auto ok = k < m_limit[1] ? k : k + m_offset[1];
                const auto ol = l < m_limit[2] ? l : l + m_offset[2];
                m_output(i, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(i, j, k, l));

            } else {
                static_assert(nt::always_false<Index>);
            }
        }

    private:
        input_type m_input;
        output_type m_output;
        NOA_NO_UNIQUE_ADDRESS dh_shape_type m_input_shape{};
        NOA_NO_UNIQUE_ADDRESS dh_shape_type m_output_shape{};
        NOA_NO_UNIQUE_ADDRESS dhw_vec_type m_offset{};
        NOA_NO_UNIQUE_ADDRESS dhw_vec_type m_limit{};
    };
}
