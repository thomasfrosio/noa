#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/indexing/Subregion.hpp"

namespace noa {
    /// 4d index-wise operator to resize an array, out of place.
    /// \details The "border_left" and "border_right", specify the number of elements to crop (negative value)
    ///          or pad (positive value) on the left or right side of each dimension. Padded elements are handled
    ///          according to the Border. The input and output arrays should not overlap.
    template<Border MODE, typename Index, typename InputAccessor, typename OutputAccessor>
    requires (nt::are_accessor_nd<4, InputAccessor, OutputAccessor>::value)
    class Resize {
    public:
        static_assert(MODE != Border::NOTHING);
        static_assert(std::is_signed_v<Index>);

        using input_accessor_type = InputAccessor;
        using output_accessor_type = OutputAccessor;
        using output_value_type = output_accessor_type;
        using index_type = Index;
        using index4_type = Vec4<Index>;
        using shape4_type = Shape4<Index>;
        using output_value_or_empty_type = std::conditional_t<MODE == Border::VALUE, output_value_type, Empty>;

        static constexpr bool IS_BOUNDLESS = MODE != Border::VALUE and MODE != Border::ZERO;
        using index4_or_empty_type = std::conditional_t<IS_BOUNDLESS, Empty, index4_type>;

    public:
        Resize(const input_accessor_type& input_accessor,
               const output_accessor_type& output_accessor,
               const shape4_type& input_shape,
               const shape4_type& output_shape,
               const index4_type& border_left,
               const index4_type& border_right,
               output_value_type cvalue
        ) : m_input(input_accessor),
            m_output(output_accessor),
            m_input_shape(input_shape),
            m_crop_left(min(border_left, index_type{0}) * -1),
            m_pad_left(max(border_left, index_type{0}))
        {
            if constexpr (MODE == Border::VALUE || MODE == Border::ZERO) {
                const auto pad_right = max(border_right, index_type{0});
                m_right = output_shape.vec() - pad_right;
            } else {
                (void) border_right;
                (void) output_shape;
            }

            if constexpr (MODE == Border::VALUE)
                m_cvalue = cvalue;
            else
                (void) cvalue;
        }

        NOA_HD constexpr void operator()(const index4_type& output_indices) const noexcept {
            const auto input_indices = output_indices - m_pad_left + m_crop_left;

            if constexpr (MODE == Border::VALUE || MODE == Border::ZERO) {
                const bool is_withing_input =
                        output_indices[0] >= m_pad_left[0] and output_indices[0] < m_right[0] and
                        output_indices[1] >= m_pad_left[1] and output_indices[1] < m_right[1] and
                        output_indices[2] >= m_pad_left[2] and output_indices[2] < m_right[2] and
                        output_indices[3] >= m_pad_left[3] and output_indices[3] < m_right[3];
                if constexpr (MODE == Border::VALUE) {
                    m_output(output_indices) =
                            is_withing_input ?
                            static_cast<output_value_type>(m_input(input_indices)) : m_cvalue;
                } else {
                    m_output(output_indices) =
                            is_withing_input ?
                            static_cast<output_value_type>(m_input(input_indices)) : output_value_type{};
                }
            } else { // CLAMP || PERIODIC || MIRROR || REFLECT
                const index4_type indices_bounded{
                        ni::offset_at<MODE>(input_indices[0], m_input_shape[0]),
                        ni::offset_at<MODE>(input_indices[1], m_input_shape[1]),
                        ni::offset_at<MODE>(input_indices[2], m_input_shape[2]),
                        ni::offset_at<MODE>(input_indices[3], m_input_shape[3]),
                };
                m_output(output_indices) = static_cast<output_value_type>(m_input(indices_bounded));
            }
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        shape4_type m_input_shape;
        index4_type m_crop_left;
        index4_type m_pad_left;
        NOA_NO_UNIQUE_ADDRESS index4_or_empty_type m_right;
        NOA_NO_UNIQUE_ADDRESS output_value_or_empty_type m_cvalue;
    };

    /// Computes the common subregions between the input and output.
    /// These can then be used to copy the input subregion into the output subregion.
    [[nodiscard]] inline auto extract_common_subregion(
            const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
            const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Vec4<i64>& border_left, const Vec4<i64>& border_right
    ) -> Pair<ni::SubregionIndexer, ni::SubregionIndexer> {
        // Exclude the regions in the input that don't end up in the output.
        const auto crop_left = min(border_left, i64{0}) * -1;
        const auto crop_right = min(border_right, i64{0}) * -1;
        const auto cropped_input = ni::SubregionIndexer(input_shape, input_strides)
                .extract_subregion(ni::Slice{crop_left[0], input_shape[0] - crop_right[0]},
                                   ni::Slice{crop_left[1], input_shape[1] - crop_right[1]},
                                   ni::Slice{crop_left[2], input_shape[2] - crop_right[2]},
                                   ni::Slice{crop_left[3], input_shape[3] - crop_right[3]});

        // Exclude the regions in the output that are not from the input.
        const auto pad_left = max(border_left, i64{0});
        const auto pad_right = max(border_right, i64{0});
        const auto cropped_output = ni::SubregionIndexer(output_shape, output_strides)
                .extract_subregion(ni::Slice{pad_left[0], output_shape[0] - pad_right[0]},
                                   ni::Slice{pad_left[1], output_shape[1] - pad_right[1]},
                                   ni::Slice{pad_left[2], output_shape[2] - pad_right[2]},
                                   ni::Slice{pad_left[3], output_shape[3] - pad_right[3]});

        // One can now copy cropped_input -> cropped_output.
        NOA_ASSERT(all(cropped_input.shape == cropped_output.shape));
        return {cropped_input, cropped_output};
    }
}
