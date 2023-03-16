#pragma once

#include "noa/core/Types.hpp"

namespace noa::algorithm::memory {
    // Sets the number of element(s) to pad/crop for each border of each dimension to get from input_shape to
    // output_shape, while keeping the centers of the input and output array (defined as ``shape / 2``) aligned.
    inline auto borders(const Shape4<i64>& input_shape, const Shape4<i64>& output_shape) {
        const auto diff = output_shape - input_shape;
        const auto border_left = output_shape / 2 - input_shape / 2;
        const auto border_right = diff - border_left;
        return std::pair{border_left.vec(), border_right.vec()};
    }

    // Resize an array.
    // - We pass the "border_left" and "border_right", specifying the number of elements
    //   to crop (neg value) or pad (pos value) on the left or right side of each dimension.
    // - Padded elements are handled according to the BorderMode.
    template<BorderMode MODE, typename Index, typename Offset, typename Value>
    class Resize {
    public:
        static_assert(MODE != BorderMode::NOTHING);
        static_assert(std::is_signed_v<Index>);
        using value_type = Value;
        using index_type = Index;
        using offset_type = Offset;
        using index4_type = Vec4<Index>;
        using shape4_type = Shape4<Index>;
        using value_or_empty_type = std::conditional_t<MODE == BorderMode::VALUE, Value, Empty>;
        using input_accessor_type = AccessorRestrict<const value_type, 4, offset_type>;
        using output_accessor_type = AccessorRestrict<value_type, 4, offset_type>;

        static constexpr bool IS_BOUNDLESS = MODE != BorderMode::VALUE && MODE != BorderMode::ZERO;
        using index4_or_empty_type = std::conditional_t<IS_BOUNDLESS, Empty, index4_type>;

    public:
        Resize(const input_accessor_type& input_accessor,
               const output_accessor_type& output_accessor,
               const shape4_type& input_shape,
               const shape4_type& output_shape,
               const index4_type& border_left,
               const index4_type& border_right,
               value_type cvalue)
                : m_input(input_accessor),
                  m_output(output_accessor),
                  m_input_shape(input_shape),
                  m_crop_left(noa::math::min(border_left, index_type{0}) * -1),
                  m_pad_left(noa::math::max(border_left, index_type{0})) {

            if constexpr (MODE == BorderMode::VALUE || MODE == BorderMode::ZERO) {
                const auto pad_right = noa::math::max(border_right, index_type{0});
                m_right = output_shape.vec() - pad_right;
            } else {
                (void) border_right;
                (void) output_shape;
            }

            if constexpr (MODE == BorderMode::VALUE)
                m_cvalue = cvalue;
            else
                (void) cvalue;
        }

        NOA_HD constexpr void operator()(index_type oi, index_type oj, index_type ok, index_type ol) const {
            const auto input_index = index4_type{oi, oj, ok, ol} - m_pad_left + m_crop_left;

            if constexpr (MODE == BorderMode::VALUE || MODE == BorderMode::ZERO) {
                const bool is_withing_input =
                        oi >= m_pad_left[0] && oi < m_right[0] &&
                        oj >= m_pad_left[1] && oj < m_right[1] &&
                        ok >= m_pad_left[2] && ok < m_right[2] &&
                        ol >= m_pad_left[3] && ol < m_right[3];
                if constexpr (MODE == BorderMode::VALUE)
                    m_output(oi, oj, ok, ol) = is_withing_input ? m_input(input_index) : m_cvalue;
                else
                    m_output(oi, oj, ok, ol) = is_withing_input ? m_input(input_index) : value_type{};

            } else { // CLAMP || PERIODIC || MIRROR || REFLECT
                const index_type ii_bounded = noa::indexing::at<MODE>(input_index[0], m_input_shape[0]);
                const index_type ij_bounded = noa::indexing::at<MODE>(input_index[1], m_input_shape[1]);
                const index_type ik_bounded = noa::indexing::at<MODE>(input_index[2], m_input_shape[2]);
                const index_type il_bounded = noa::indexing::at<MODE>(input_index[3], m_input_shape[3]);
                m_output(oi, oj, ok, ol) = m_input(ii_bounded, ij_bounded, ik_bounded, il_bounded);
            }
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        shape4_type m_input_shape;
        index4_type m_crop_left;
        index4_type m_pad_left;
        NOA_NO_UNIQUE_ADDRESS index4_or_empty_type m_right;
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_cvalue;
    };

    template<BorderMode MODE, typename Index, typename Offset, typename Value>
    auto resize(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                const Vec4<i64>& border_left, const Vec4<i64>& border_right, Value cvalue = Value{}) {
        return Resize<MODE, Index, Offset, Value>(
                AccessorRestrict<const Value, 4, Offset>(input, input_strides.as_safe<Offset>()),
                AccessorRestrict<Value, 4, Offset>(output, output_strides.as_safe<Offset>()),
                input_shape.as_safe<Index>(), output_shape.as_safe<Index>(),
                border_left.as_safe<Index>(), border_right.as_safe<Index>(),
                cvalue);
    }

    [[nodiscard]] inline auto extract_common_subregion(
            const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
            const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Vec4<i64>& border_left, const Vec4<i64>& border_right
    ) -> std::pair<indexing::Subregion, indexing::Subregion> {

        // Exclude the regions in the input that don't end up in the output.
        const auto crop_left = noa::math::min(border_left, i64{0}) * -1;
        const auto crop_right = noa::math::min(border_right, i64{0}) * -1;
        const auto cropped_input = noa::indexing::Subregion(input_shape, input_strides)
                .extract(noa::indexing::slice_t{crop_left[0], input_shape[0] - crop_right[0]},
                         noa::indexing::slice_t{crop_left[1], input_shape[1] - crop_right[1]},
                         noa::indexing::slice_t{crop_left[2], input_shape[2] - crop_right[2]},
                         noa::indexing::slice_t{crop_left[3], input_shape[3] - crop_right[3]});

        // Exclude the regions in the output that are not from the input.
        const auto pad_left = noa::math::max(border_left, i64{0});
        const auto pad_right = noa::math::max(border_right, i64{0});
        const auto cropped_output = noa::indexing::Subregion(output_shape, output_strides)
                .extract(noa::indexing::slice_t{pad_left[0], output_shape[0] - pad_right[0]},
                         noa::indexing::slice_t{pad_left[1], output_shape[1] - pad_right[1]},
                         noa::indexing::slice_t{pad_left[2], output_shape[2] - pad_right[2]},
                         noa::indexing::slice_t{pad_left[3], output_shape[3] - pad_right[3]});

        // One can now copy cropped_input -> cropped_output.
        NOA_ASSERT(noa::all(cropped_input.shape == cropped_output.shape));
        return {cropped_input, cropped_output};
    }
}
