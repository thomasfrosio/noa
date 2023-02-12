#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"
#include "noa/cpu/memory/Copy.hpp"
#include "noa/cpu/memory/PtrHost.hpp"

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
        extracted.count = noa::math::max(elements.size(), offsets.size());
        if (!elements.empty()) {
            extracted.values = PtrHost<T>::alloc(extracted.count);
            copy(elements.data(), extracted.values.get(), extracted.count);
            elements.clear();
        }
        if (!offsets.empty()) {
            extracted.offsets = PtrHost<O>::alloc(extracted.count);
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
            const Shared<Input[]>& input, Strides4<i64> input_strides,
            const Shared<Lhs[]>& lhs, Strides4<i64> lhs_strides, Shape4<i64> shape,
            UnaryOp&& unary_op, bool extract_values, bool extract_offsets, Stream& stream)
    -> Extracted<ExtractedValue, ExtractedOffset> {

        NOA_ASSERT((input || !extract_values) && lhs && all(shape > 0 ));

        const auto order = noa::indexing::order(input_strides, shape);
        if (noa::any(order != Vec4<i64>{0, 1, 2, 3})) {
            input_strides = noa::indexing::reorder(input_strides, order);
            lhs_strides = noa::indexing::reorder(lhs_strides, order);
            shape = noa::indexing::reorder(shape, order);
        }

        const auto input_accessor = AccessorContiguous<const Input, 1, i64>(input.get());
        const auto lhs_accessor = Accessor<const Lhs, 4, i64>(lhs.get(), lhs_strides);
        std::vector<ExtractedValue> values;
        std::vector<ExtractedOffset> offsets;
        stream.synchronize();

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
            const Shared<Input[]>& input, Strides4<i64> input_strides,
            const Shared<Lhs[]>& lhs, Strides4<i64> lhs_strides,
            const Shared<Rhs[]>& rhs, Strides4<i64> rhs_strides, Shape4<i64> shape,
            BinaryOp&& binary_op, bool extract_values, bool extract_offsets, Stream& stream)
    -> Extracted<ExtractedValue, ExtractedOffset> {

        NOA_ASSERT((input || !extract_values) && lhs && rhs && all(shape > 0 ));

        const auto order = noa::indexing::order(input_strides, shape);
        if (noa::any(order != Vec4<i64>{0, 1, 2, 3})) {
            input_strides = noa::indexing::reorder(input_strides, order);
            lhs_strides = noa::indexing::reorder(lhs_strides, order);
            rhs_strides = noa::indexing::reorder(rhs_strides, order);
            shape = noa::indexing::reorder(shape, order);
        }

        const auto input_accessor = AccessorContiguous<const Input, 1, i64>(input.get());
        const auto lhs_accessor = Accessor<const Lhs, 4, i64>(lhs.get(), lhs_strides);
        const auto rhs_accessor = Accessor<const Rhs, 4, i64>(rhs.get(), rhs_strides);
        std::vector<ExtractedValue> values;
        std::vector<ExtractedOffset> offsets;
        stream.synchronize();

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
            const Shared<Input[]>& input, const Strides4<i64>& input_strides,
            const Shared<Lhs[]>& lhs, const Strides4<i64>& lhs_strides, Rhs rhs,
            const Shape4<i64>& shape, BinaryOp&& binary_op,
            bool extract_values, bool extract_offsets, Stream& stream) {
        auto unary_op = [=, op = std::forward<BinaryOp>(binary_op)](Lhs lhs_value) {
            return binary_op(lhs_value, rhs);
        };
        return extract_unary<ExtractedValue, ExtractedOffset>(
                input, input_strides, lhs, lhs_strides, shape,
                unary_op, extract_values, extract_offsets, stream);
    }

    template<typename ExtractedValue, typename ExtractedOffset,
             typename Input, typename Lhs, typename Rhs, typename BinaryOp>
    auto extract_binary(
            const Shared<Input[]>& input, const Strides4<i64>& input_strides,
            Lhs lhs, const Shared<Rhs[]>& rhs, const Strides4<i64>& rhs_strides,
            const Shape4<i64>& shape, BinaryOp&& binary_op,
            bool extract_values, bool extract_offsets, Stream& stream) {
        auto unary_op = [=, op = std::forward<BinaryOp>(binary_op)](Rhs rhs_value) {
            return binary_op(lhs, rhs_value);
        };
        return extract_unary<ExtractedValue, ExtractedOffset>(
                input, input_strides, rhs, rhs_strides, shape,
                unary_op, extract_values, extract_offsets, stream);
    }

    // TODO Add to unified API.
    template<typename Input, typename Offset, typename Output>
    void extract_elements(
            const Shared<Input[]>& input,
            const Shared<Offset[]>& offsets,
            const Shared<Output[]>& output,
            i64 elements, Stream& stream) {
        NOA_ASSERT(input && output && offsets);
        stream.enqueue([=]() {
            const auto* input_ptr = input.get();
            const auto* offsets_ptr = offsets.get();
            auto* output_ptr = output.get();
            for (i64 idx = 0; idx < elements; ++idx, ++offsets_ptr, ++output_ptr)
                *output_ptr = static_cast<Output>(input_ptr[*offsets_ptr]);
        });
    }

    template<typename ExtractedValue, typename ExtractedOffset, typename Output>
    void insert_elements(
            const Extracted<ExtractedValue, ExtractedOffset>& extracted,
            const Shared<Output[]>& output, Stream& stream) {
        NOA_ASSERT(extracted.values && extracted.offsets && output);
        stream.enqueue([=]() {
            const auto* extracted_values = extracted.values.get();
            const auto* extracted_offsets = extracted.offsets.get();
            auto* output_ptr = output.get();
            for (i64 idx = 0; idx < extracted.count; ++idx, ++extracted_values, ++extracted_offsets)
                output_ptr[*extracted_offsets] = static_cast<Output>(*extracted_values);
        });
    }
}
