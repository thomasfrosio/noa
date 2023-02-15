#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory::details {
    using namespace noa;

    template<typename Input, typename Lhs, typename Operator,
             typename ExtractedValue, typename ExtractedOffset>
    constexpr bool is_valid_extract_unary_v =
            traits::is_any_v<ExtractedValue, i32, i64, u32, u64, f16, f32, f64> &&
            traits::are_all_same_v<Input, Lhs, ExtractedValue> &&
            traits::is_any_v<ExtractedOffset, i32, i64, u32, u64> &&
            std::is_same_v<Operator, logical_not_t>;

    template<typename Input, typename Lhs, typename Rhs, typename Operator,
             typename ExtractedValue, typename ExtractedOffset>
    constexpr bool is_valid_extract_binary_v =
            traits::is_any_v<ExtractedValue, i32, i64, u32, u64, f16, f32, f64> &&
            traits::are_all_same_v<Input, Lhs, Rhs, ExtractedValue> &&
            traits::is_any_v<ExtractedOffset, i32, i64, u32, u64> &&
            traits::is_any_v<Operator, equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t>;

    template<typename Input, typename ExtractedOffset, typename ExtractedValue>
    constexpr bool is_valid_insert_v =
            traits::is_any_v<ExtractedValue, i32, i64, u32, u64, f16, f32, f64> &&
            std::is_same_v<Input, ExtractedValue> &&
            traits::is_any_v<ExtractedOffset, i32, i64, u32, u64>;
}

namespace noa::cuda::memory {
    template<typename T, typename I>
    struct Extracted {
        Shared<T[]> values{};
        Shared<I[]> offsets{};
        i64 count{};
    };

    // Extracts elements (and/or offsets) from the input array based on a unary bool operator.
    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename UnaryOp,
             typename = std::enable_if_t<details::is_valid_extract_unary_v<
                     Input, Lhs, UnaryOp, ExtractedValue, ExtractedOffset>>>
    auto extract_unary(
            const Input* input, const Strides4<i64>& input_strides,
            const Lhs* lhs, const Strides4<i64>& lhs_strides, const Shape4<i64>& shape,
            UnaryOp unary_op, bool extract_values, bool extract_offsets, Stream& stream)
    -> Extracted<ExtractedValue, ExtractedOffset>;

    // Extracts elements (and/or offsets) from the input array based on a binary bool operator.
    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename Rhs, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_extract_binary_v<
                     Input, Lhs, Rhs, BinaryOp, ExtractedValue, ExtractedOffset>>>
    auto extract_binary(
            const Input* input, Strides4<i64> input_strides,
            const Lhs* lhs, Strides4<i64> lhs_strides,
            const Rhs* rhs, Strides4<i64> rhs_strides, Shape4<i64> shape,
            BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream)
    -> Extracted<ExtractedValue, ExtractedOffset>;

    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename Rhs, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_extract_binary_v<
                     Input, Lhs, Rhs, BinaryOp, ExtractedValue, ExtractedOffset>>>
    auto extract_binary(
            const Input* input, const Strides4<i64>& input_strides,
            const Lhs* lhs, const Strides4<i64>& lhs_strides,
            Rhs rhs,
            const Shape4<i64>& shape, BinaryOp binary_op,
            bool extract_values, bool extract_offsets, Stream& stream)
    -> Extracted<ExtractedValue, ExtractedOffset>;

    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename Rhs, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_extract_binary_v<
                     Input, Lhs, Rhs, BinaryOp, ExtractedValue, ExtractedOffset>>>
    auto extract_binary(
            const Input* input, const Strides4<i64>& input_strides,
            Lhs lhs,
            const Rhs* rhs, const Strides4<i64>& rhs_strides,
            const Shape4<i64>& shape, BinaryOp binary_op,
            bool extract_values, bool extract_offsets, Stream& stream)
    -> Extracted<ExtractedValue, ExtractedOffset>;

    // TODO Add to unified API.
    template<typename Input, typename Offset, typename Output,
             typename = std::enable_if_t<details::is_valid_insert_v<Input, Offset, Output>>>
    void extract_elements(
            const Input* input,
            const Offset* offsets,
            Output* output,
            i64 elements, Stream& stream);

    template<typename ExtractedValue, typename ExtractedOffset, typename Output,
             typename = std::enable_if_t<details::is_valid_insert_v<ExtractedValue, ExtractedOffset, Output>>>
    void insert_elements(
            const Extracted<ExtractedValue, ExtractedOffset>& extracted,
            Output* output, Stream& stream);
}
