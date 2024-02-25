#pragma once

#include "noa/core/Types.hpp"

namespace noa::fft {
    enum class ResizeMode {
        PAD_H2H,
        PAD_F2F,
        CROP_H2H,
        CROP_F2F
    };

    template<ResizeMode MODE, typename Index, typename InputAccessor, typename OutputAccessor>
    requires (nt::are_accessor_pure_nd<4, InputAccessor, OutputAccessor>::value and nt::is_sint_v<Index>)
    class FourierResize {
    public:
        using index_type = Index;
        using input_accessor_type = InputAccessor;
        using output_accessor_type = OutputAccessor;
        using input_value_type = input_accessor_type::mutable_value_type;
        using output_value_type = output_accessor_type::value_type;

        static_assert(nt::are_complex_v<input_value_type, output_value_type> or
                      nt::are_real_v<input_value_type, output_value_type> or
                      (nt::is_complex_v<input_value_type> and nt::is_real_v<output_value_type>));

        using dh_shape_type = std::conditional_t<
                MODE == ResizeMode::CROP_H2H or MODE == ResizeMode::PAD_H2H,
                Shape2<index_type>, Empty>;
        using dhw_vec_type = std::conditional_t<
                MODE == ResizeMode::CROP_F2F or MODE == ResizeMode::PAD_F2F,
                Shape3<index_type>, Empty>;

        FourierResize(
                const input_accessor_type& input,
                const output_accessor_type& output,
                const Shape3<index_type>& input_shape,
                const Shape3<index_type>& output_shape
        ) : m_input(input),
            m_output(output)
        {
            if constexpr (MODE == ResizeMode::CROP_H2H) {
                m_input_shape = input_shape.pop_back();
                m_output_shape = output_shape.pop_back();

            } else if constexpr (MODE == ResizeMode::PAD_H2H) {
                m_input_shape = input_shape.pop_back();
                m_output_shape = output_shape.pop_back();

            } else if constexpr (MODE == ResizeMode::CROP_F2F) {
                m_offset = input_shape - output_shape;
                m_limit = (output_shape + 1) / 2;

            } else if constexpr (MODE == ResizeMode::PAD_F2F) {
                m_offset = output_shape - input_shape;
                m_limit = (input_shape + 1) / 2;
            }
        }

        NOA_HD constexpr void operator()(index_type i, index_type j, index_type k, index_type l) const noexcept {
            if constexpr (MODE == ResizeMode::CROP_H2H) {
                const auto ij = j < (m_output_shape[0] + 1) / 2 ? j : j + m_input_shape[0] - m_output_shape[0];
                const auto ik = k < (m_output_shape[1] + 1) / 2 ? k : k + m_input_shape[1] - m_output_shape[1];
                m_output(i, j, k, l) = to_output_(m_input(i, ij, ik, l));

            } else if constexpr (MODE == ResizeMode::PAD_H2H) {
                const auto oj = j < (m_input_shape[0] + 1) / 2 ? j : j + m_output_shape[0] - m_input_shape[0];
                const auto ok = k < (m_input_shape[1] + 1) / 2 ? k : k + m_output_shape[1] - m_input_shape[1];
                m_output(i, oj, ok, l) = to_output_(m_input(i, j, k, l));

            } else if constexpr (MODE == ResizeMode::CROP_F2F) {
                const auto ij = j < m_limit[0] ? j : j + m_offset[0];
                const auto ik = k < m_limit[1] ? k : k + m_offset[1];
                const auto il = l < m_limit[2] ? l : l + m_offset[2];
                m_output(i, j, k, l) =  to_output_(m_input(i, ij, ik, il));

            } else if constexpr (MODE == ResizeMode::PAD_F2F) {
                const auto oj = j < m_limit[0] ? j : j + m_offset[0];
                const auto ok = k < m_limit[1] ? k : k + m_offset[1];
                const auto ol = l < m_limit[2] ? l : l + m_offset[2];
                m_output(i, oj, ok, ol) = to_output_(m_input(i, j, k, l));

            } else {
                static_assert(nt::always_false_v<input_value_type>);
            }
        }

    private:
        NOA_HD constexpr output_value_type to_output_(const input_value_type& value) {
            if constexpr (nt::is_complex_v<input_value_type> and nt::is_real_v<output_value_type>)
                return static_cast<output_value_type>(abs_squared(value));
            else
                return static_cast<output_value_type>(value);
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        NOA_NO_UNIQUE_ADDRESS dh_shape_type m_input_shape{};
        NOA_NO_UNIQUE_ADDRESS dh_shape_type m_output_shape{};
        NOA_NO_UNIQUE_ADDRESS dhw_vec_type m_offset{};
        NOA_NO_UNIQUE_ADDRESS dhw_vec_type m_limit{};
    };
}
