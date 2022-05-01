#pragma once

#ifndef NOA_INDEX_INL_
#error "This is an internal header; it should not be included."
#endif

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"

namespace noa::cpu::memory::details {
    // NOTE: The count is the only reason these functions require a synchronization and have to be synchronous...
    template<typename T, typename I>
    Extracted<T, I> prepareExtracted(std::vector<T>& elements, std::vector<I>& offsets) {
        Extracted<T, I> extracted{};
        extracted.count = noa::math::max(elements.size(), offsets.size());
        if (!elements.empty()) {
            extracted.values = PtrHost<T>::alloc(extracted.count);
            copy(elements.data(), extracted.values.get(), extracted.count);
            elements.clear();
        }
        if (!offsets.empty()) {
            extracted.offsets = PtrHost<I>::alloc(extracted.count);
            copy(offsets.data(), extracted.offsets.get(), extracted.count);
            offsets.clear();
        }
        return extracted;
    }
}

namespace noa::cpu::memory {
    size4_t atlasLayout(size4_t subregion_shape, int4_t* origins) {
        using namespace noa::math;
        const auto col = static_cast<size_t>(ceil(sqrt(static_cast<float>(subregion_shape[0]))));
        const size_t row = (subregion_shape[0] + col - 1) / col;
        const size4_t atlas_shape{1, subregion_shape[1], row * subregion_shape[2], col * subregion_shape[3]};
        for (size_t y = 0; y < row; ++y) {
            for (size_t x = 0; x < col; ++x) {
                const size_t idx = y * col + x;
                if (idx >= subregion_shape[0])
                    break;
                origins[idx] = {0, 0, y * subregion_shape[2], x * subregion_shape[3]};
            }
        }
        return atlas_shape;
    }

    template<typename value_t, typename offset_, typename T, typename U, typename UnaryOp>
    Extracted<value_t, offset_> extract(const shared_t<T[]>& input, size4_t input_stride,
                                        const shared_t<U[]>& lhs, size4_t lhs_stride, size4_t shape,
                                        UnaryOp unary_op, bool extract_values, bool extract_offsets, Stream& stream) {
        const T* input_ = input.get();
        const U* lhs_ = lhs.get();
        std::vector<value_t> values;
        std::vector<offset_> offsets;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        if (unary_op(lhs_[indexing::at(i, j, k, l, lhs_stride)])) {
                            const size_t offset = indexing::at(i, j, k, l, input_stride);
                            if (extract_values)
                                values.emplace_back(static_cast<value_t>(input_[offset]));
                            if (extract_offsets)
                                offsets.emplace_back(static_cast<offset_>(offset));
                        }
                    }
                }
            }
        }
        return details::prepareExtracted(values, offsets);
    }

    template<typename value_t, typename offset_, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, offset_> extract(const shared_t<T[]>& input, size4_t input_stride,
                                        const shared_t<U[]>& lhs, size4_t lhs_stride, V rhs, size4_t shape,
                                        BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream) {
        const T* input_ = input.get();
        const U* lhs_ = lhs.get();
        std::vector<value_t> values;
        std::vector<offset_> offsets;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        if (binary_op(lhs_[indexing::at(i, j, k, l, lhs_stride)], rhs)) {
                            const size_t offset = indexing::at(i, j, k, l, input_stride);
                            if (extract_values)
                                values.emplace_back(static_cast<value_t>(input_[offset]));
                            if (extract_offsets)
                                offsets.emplace_back(static_cast<offset_>(offset));
                        }
                    }
                }
            }
        }
        return details::prepareExtracted(values, offsets);
    }

    template<typename value_t, typename offset_, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, offset_> extract(const shared_t<T[]>& input, size4_t input_stride,
                                        U lhs, const shared_t<V[]>& rhs, size4_t rhs_stride, size4_t shape,
                                        BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream) {
        const T* input_ = input.get();
        const V* rhs_ = rhs.get();
        std::vector<value_t> values;
        std::vector<offset_> offsets;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        if (binary_op(lhs, rhs_[indexing::at(i, j, k, l, rhs_stride)])) {
                            const size_t offset = indexing::at(i, j, k, l, input_stride);
                            if (extract_values)
                                values.emplace_back(static_cast<value_t>(input_[offset]));
                            if (extract_offsets)
                                offsets.emplace_back(static_cast<offset_>(offset));
                        }
                    }
                }
            }
        }
        return details::prepareExtracted(values, offsets);
    }

    template<typename value_t, typename offset_, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, offset_> extract(const shared_t<T[]>& input, size4_t input_stride,
                                        const shared_t<U[]>& lhs, size4_t lhs_stride,
                                        const shared_t<V[]>& rhs, size4_t rhs_stride,
                                        size4_t shape, BinaryOp binary_op, bool extract_values, bool extract_offsets,
                                        Stream& stream) {
        const T* input_ = input.get();
        const U* lhs_ = lhs.get();
        const V* rhs_ = rhs.get();
        std::vector<value_t> values;
        std::vector<offset_> offsets;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        if (binary_op(lhs_[indexing::at(i, j, k, l, lhs_stride)],
                                      rhs_[indexing::at(i, j, k, l, rhs_stride)])) {
                            const size_t offset = indexing::at(i, j, k, l, input_stride);
                            if (extract_values)
                                values.emplace_back(static_cast<value_t>(input_[offset]));
                            if (extract_offsets)
                                offsets.emplace_back(static_cast<offset_>(offset));
                        }
                    }
                }
            }
        }
        return details::prepareExtracted(values, offsets);
    }

    template<typename T, typename U, typename V>
    void extract(const shared_t<T[]>& input, const shared_t<U[]>& offsets,
                 const shared_t<V[]>& output, size_t elements, Stream& stream){
        stream.enqueue([=]() {
            const auto* input_ = input.get();
            const auto* offsets_ = offsets.get();
            auto* output_ = output.get();
            for (size_t idx = 0; idx < elements; ++idx, ++offsets_, ++output_)
                *output_ = static_cast<V>(input_[*offsets_]);
        });
    }

    template<typename value_t, typename offset_, typename T>
    void insert(const Extracted<value_t, offset_>& extracted, const shared_t<T[]>& output, Stream& stream) {
        stream.enqueue([=]() {
            const auto* elements = extracted.values.get();
            const auto* offsets = extracted.offsets.get();
            auto* output_ = output.get();
            for (size_t idx = 0; idx < extracted.count; ++idx, ++elements, ++offsets)
                output_[*offsets] = static_cast<value_t>(*elements);
        });
    }
}
