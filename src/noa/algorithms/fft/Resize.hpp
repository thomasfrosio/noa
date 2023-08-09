#pragma once

#include "noa/core/Types.hpp"

namespace noa::algorithm::fft {
    enum class ResizeMode {
        PAD_H2H,
        PAD_F2F,
        CROP_H2H,
        CROP_F2F
    };

    // FFT resize.
    template<ResizeMode MODE, typename Value, typename Index, typename Offset>
    class Resize {
    public:
        using value_type = Value;
        using index_type = Index;
        using offset_type = Offset;
        using input_accessor_type = AccessorRestrict<const value_type, 4, offset_type>;
        using output_accessor_type = AccessorRestrict<value_type, 4, offset_type>;

        using dh_shape_type = std::conditional_t<
                MODE == ResizeMode::CROP_H2H || MODE == ResizeMode::PAD_H2H,
                Shape2<index_type>, Empty>;
        using dhw_vec_type = std::conditional_t<
                MODE == ResizeMode::CROP_F2F || MODE == ResizeMode::PAD_F2F,
                Shape3<index_type>, Empty>;

        Resize(const input_accessor_type input,
               const output_accessor_type output,
               const Shape3<index_type> input_shape,
               const Shape3<index_type> output_shape)
                : m_input(input),
                  m_output(output) {
            NOA_ASSERT(input.get() != output.get());

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
                m_output(i, j, k, l) = m_input(i, ij, ik, l);

            } else if constexpr (MODE == ResizeMode::PAD_H2H) {
                const auto oj = j < (m_input_shape[0] + 1) / 2 ? j : j + m_output_shape[0] - m_input_shape[0];
                const auto ok = k < (m_input_shape[1] + 1) / 2 ? k : k + m_output_shape[1] - m_input_shape[1];
                m_output(i, oj, ok, l) = m_input(i, j, k, l);

            } else if constexpr (MODE == ResizeMode::CROP_F2F) {
                const auto ij = j < m_limit[0] ? j : j + m_offset[0];
                const auto ik = k < m_limit[1] ? k : k + m_offset[1];
                const auto il = l < m_limit[2] ? l : l + m_offset[2];
                m_output(i, j, k, l) =  m_input(i, ij, ik, il);

            } else if constexpr (MODE == ResizeMode::PAD_F2F) {
                const auto oj = j < m_limit[0] ? j : j + m_offset[0];
                const auto ok = k < m_limit[1] ? k : k + m_offset[1];
                const auto ol = l < m_limit[2] ? l : l + m_offset[2];
                m_output(i, oj, ok, ol) = m_input(i, j, k, l);

            } else {
                static_assert(nt::always_false_v<value_type>);
            }
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        NOA_NO_UNIQUE_ADDRESS dh_shape_type m_input_shape{};
        NOA_NO_UNIQUE_ADDRESS dh_shape_type m_output_shape{};
        NOA_NO_UNIQUE_ADDRESS dhw_vec_type m_offset{};
        NOA_NO_UNIQUE_ADDRESS dhw_vec_type m_limit{};
    };

    template<ResizeMode MODE, typename Value, typename Index, typename Offset>
    auto resize(const Value* input, const Strides4<Offset>& input_strides, const Shape4<Index>& input_shape,
                Value* output, const Strides4<Offset>& output_strides, const Shape4<Index>& output_shape) {
        const auto input_accessor = AccessorRestrict<const Value, 4, Offset>(input, input_strides);
        const auto output_accessor = AccessorRestrict<Value, 4, Offset>(output, output_strides);
        using kernel_type = Resize<MODE, Value, Index, Offset>;
        auto kernel = kernel_type(input_accessor, output_accessor, input_shape.pop_front(), output_shape.pop_front());

        // We always loop through the smallest shape. This implies that for padding, the padded elements
        // in the output are NOT set and the backend should make sure these are set to zeros at some point.
        Shape4<Index> iwise_shape;
        if constexpr (MODE == ResizeMode::CROP_H2H) {
            iwise_shape = output_shape.rfft();
        } else if constexpr (MODE == ResizeMode::PAD_H2H) {
            iwise_shape = input_shape.rfft();
        } else if constexpr (MODE == ResizeMode::CROP_F2F) {
            iwise_shape = output_shape;
        } else if constexpr (MODE == ResizeMode::PAD_F2F) {
            iwise_shape = input_shape;
        } else {
            static_assert(nt::always_false_v<Value>);
        }

        return std::pair{kernel, iwise_shape};
    }
}
