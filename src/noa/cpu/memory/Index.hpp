#pragma once

#include "noa/core/Types.hpp"
#include "noa/cpu/memory/Copy.hpp"
#include "noa/cpu/memory/AllocatorHeap.hpp"

namespace noa::cpu::memory {
    template<typename T, typename I>
    struct Extracted {
        Shared<T[]> values{};
        Shared<I[]> offsets{};
        i64 count{};
    };
}

namespace noa::cpu::memory::details {
    // TODO Modify the allocator of std::vector so that we can release its memory.
    template<typename T, typename O>
    Extracted<T, O> prepare_extracted(std::vector<T>& elements, std::vector<O>& offsets) {
        Extracted<T, O> extracted{};
        extracted.count = static_cast<i64>(noa::math::max(elements.size(), offsets.size()));
        if (!elements.empty()) {
            extracted.values = AllocatorHeap<T>::allocate(extracted.count);
            copy(elements.data(), extracted.values.get(), extracted.count);
            elements.clear();
        }
        if (!offsets.empty()) {
            extracted.offsets = AllocatorHeap<O>::allocate(extracted.count);
            copy(offsets.data(), extracted.offsets.get(), extracted.count);
            offsets.clear();
        }
        return extracted;
    }
}

namespace noa::cpu::memory {
    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename UnaryOp>
    auto extract_unary(
            const Input* input, Strides4<i64> input_strides,
            const Lhs* lhs, Strides4<i64> lhs_strides, Shape4<i64> shape,
            UnaryOp&& unary_op, bool extract_values, bool extract_offsets)
    -> Extracted<ExtractedValue, ExtractedOffset> {

        NOA_ASSERT((input || !extract_values) && lhs && all(shape > 0 ));

        const auto order = noa::indexing::order(input_strides, shape);
        if (noa::any(order != Vec4<i64>{0, 1, 2, 3})) {
            input_strides = noa::indexing::reorder(input_strides, order);
            lhs_strides = noa::indexing::reorder(lhs_strides, order);
            shape = noa::indexing::reorder(shape, order);
        }

        const auto input_accessor = AccessorContiguous<const Input, 1, i64>(input);
        const auto lhs_accessor = Accessor<const Lhs, 4, i64>(lhs, lhs_strides);
        std::vector<ExtractedValue> values;
        std::vector<ExtractedOffset> offsets;

        for (i64 i = 0; i < shape[0]; ++i) {
            for (i64 j = 0; j < shape[1]; ++j) {
                for (i64 k = 0; k < shape[2]; ++k) {
                    for (i64 l = 0; l < shape[3]; ++l) {
                        if (unary_op(lhs_accessor(i, j, k, l))) {
                            const auto offset = noa::indexing::at(i, j, k, l, input_strides);
                            if (extract_values)
                                values.emplace_back(static_cast<ExtractedValue>(input_accessor[offset]));
                            if (extract_offsets)
                                offsets.emplace_back(static_cast<ExtractedOffset>(offset));
                        }
                    }
                }
            }
        }
        return details::prepare_extracted(values, offsets);
    }

    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename Rhs, typename BinaryOp>
    auto extract_binary(
            const Input* input, Strides4<i64> input_strides,
            const Lhs* lhs, Strides4<i64> lhs_strides,
            const Rhs* rhs, Strides4<i64> rhs_strides, Shape4<i64> shape,
            BinaryOp&& binary_op, bool extract_values, bool extract_offsets)
    -> Extracted<ExtractedValue, ExtractedOffset> {

        NOA_ASSERT((input || !extract_values) && lhs && rhs && all(shape > 0 ));

        const auto order = noa::indexing::order(input_strides, shape);
        if (noa::any(order != Vec4<i64>{0, 1, 2, 3})) {
            input_strides = noa::indexing::reorder(input_strides, order);
            lhs_strides = noa::indexing::reorder(lhs_strides, order);
            rhs_strides = noa::indexing::reorder(rhs_strides, order);
            shape = noa::indexing::reorder(shape, order);
        }

        const auto input_accessor = AccessorContiguous<const Input, 1, i64>(input);
        const auto lhs_accessor = Accessor<const Lhs, 4, i64>(lhs, lhs_strides);
        const auto rhs_accessor = Accessor<const Rhs, 4, i64>(rhs, rhs_strides);
        std::vector<ExtractedValue> values;
        std::vector<ExtractedOffset> offsets;

        for (i64 i = 0; i < shape[0]; ++i) {
            for (i64 j = 0; j < shape[1]; ++j) {
                for (i64 k = 0; k < shape[2]; ++k) {
                    for (i64 l = 0; l < shape[3]; ++l) {
                        if (binary_op(lhs_accessor(i, j, k, l), rhs_accessor(i, j, k, l))) {
                            const auto offset = noa::indexing::at(i, j, k, l, input_strides);
                            if (extract_values)
                                values.emplace_back(static_cast<ExtractedValue>(input_accessor[offset]));
                            if (extract_offsets)
                                offsets.emplace_back(static_cast<ExtractedOffset>(offset));
                        }
                    }
                }
            }
        }
        return details::prepare_extracted(values, offsets);
    }

    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename Rhs, typename BinaryOp>
    auto extract_binary(
            const Input* input, const Strides4<i64>& input_strides,
            const Lhs* lhs, const Strides4<i64>& lhs_strides,
            Rhs rhs,
            const Shape4<i64>& shape, BinaryOp&& binary_op,
            bool extract_values, bool extract_offsets) {
        auto unary_op = [=, op = std::forward<BinaryOp>(binary_op)](Lhs lhs_value) {
            return op(lhs_value, rhs);
        };
        return extract_unary<ExtractedValue, ExtractedOffset>(
                input, input_strides, lhs, lhs_strides, shape,
                unary_op, extract_values, extract_offsets);
    }

    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename Rhs, typename BinaryOp>
    auto extract_binary(
            const Input* input, const Strides4<i64>& input_strides,
            Lhs lhs,
            const Rhs* rhs, const Strides4<i64>& rhs_strides,
            const Shape4<i64>& shape, BinaryOp&& binary_op,
            bool extract_values, bool extract_offsets) {
        auto unary_op = [=, op = std::forward<BinaryOp>(binary_op)](Rhs rhs_value) {
            return op(lhs, rhs_value);
        };
        return extract_unary<ExtractedValue, ExtractedOffset>(
                input, input_strides, rhs, rhs_strides, shape,
                unary_op, extract_values, extract_offsets);
    }

    // TODO Add to unified API.
    template<typename Input, typename Offset, typename Output>
    void extract_elements(const Input* input, const Offset* offsets, Output* output, i64 elements) {
        NOA_ASSERT(input && output && offsets);
        for (i64 idx = 0; idx < elements; ++idx, ++offsets, ++output)
            *output = static_cast<Output>(input[*offsets]);
    }

    template<typename ExtractedValue, typename ExtractedOffset, typename Output>
    void insert_elements(const ExtractedValue* extracted_values,
                         const ExtractedOffset* extracted_offsets,
                         i64 elements,
                         Output* output) {
        NOA_ASSERT(extracted_values && extracted_offsets && output);
        for (i64 idx = 0; idx < elements; ++idx, ++extracted_values, ++extracted_offsets)
            output[*extracted_offsets] = static_cast<Output>(*extracted_values);
    }
}
