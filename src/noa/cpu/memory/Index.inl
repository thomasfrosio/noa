#pragma once

#ifndef NOA_INDEX_INL_
#error "This is an internal header; it should not be included."
#endif

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"

namespace noa::cpu::memory::details {
    // NOTE: The count is the only reason these functions require a synchronization and have to be synchronous...
    template<typename T, typename I>
    Extracted<T, I> prepareExtracted(std::vector<T>& elements, std::vector<I>& indexes) {
        Extracted<T, I> extracted{};
        extracted.count = noa::math::max(elements.size(), indexes.size());
        if (!elements.empty()) {
            extracted.values = PtrHost<T>::alloc(extracted.count);
            copy(elements.data(), extracted.values.get(), extracted.count);
            elements.clear();
        }
        if (!indexes.empty()) {
            extracted.indexes = PtrHost<I>::alloc(extracted.count);
            copy(indexes.data(), extracted.indexes.get(), extracted.count);
            indexes.clear();
        }
        return extracted;
    }
}

namespace noa::cpu::memory {
    size4_t atlasLayout(size4_t subregion_shape, int4_t* origins) {
        const auto col = static_cast<size_t>(math::ceil(math::sqrt(static_cast<float>(subregion_shape[0]))));
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

    template<typename value_t, typename index_t, typename T, typename U, typename UnaryOp>
    Extracted<value_t, index_t> extract(const shared_t<T[]>& input, size4_t input_stride,
                                        const shared_t<U[]>& lhs, size4_t lhs_stride, size4_t shape,
                                        UnaryOp unary_op, bool extract_values, bool extract_indexes, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const T* input_ = input.get();
        const U* lhs_ = lhs.get();
        std::vector<value_t> values;
        std::vector<index_t> indexes;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        if (unary_op(lhs_[indexing::at(i, j, k, l, lhs_stride)])) {
                            const size_t offset = indexing::at(i, j, k, l, input_stride);
                            if (extract_values)
                                values.emplace_back(static_cast<value_t>(input_[offset]));
                            if (extract_indexes)
                                indexes.emplace_back(static_cast<index_t>(offset));
                        }
                    }
                }
            }
        }
        return details::prepareExtracted(values, indexes);
    }

    template<typename value_t, typename index_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, index_t> extract(const shared_t<T[]>& input, size4_t input_stride,
                                        const shared_t<U[]>& lhs, size4_t lhs_stride, V rhs, size4_t shape,
                                        BinaryOp binary_op, bool extract_values, bool extract_indexes, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const T* input_ = input.get();
        const U* lhs_ = lhs.get();
        std::vector<value_t> values;
        std::vector<index_t> indexes;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        if (binary_op(lhs_[indexing::at(i, j, k, l, lhs_stride)], rhs)) {
                            const size_t offset = indexing::at(i, j, k, l, input_stride);
                            if (extract_values)
                                values.emplace_back(static_cast<value_t>(lhs_[offset]));
                            if (extract_indexes)
                                indexes.emplace_back(static_cast<index_t>(offset));
                        }
                    }
                }
            }
        }
        return details::prepareExtracted(values, indexes);
    }

    template<typename value_t, typename index_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, index_t> extract(const shared_t<T[]>& input, size4_t input_stride,
                                        U lhs, const shared_t<V[]>& rhs, size4_t rhs_stride, size4_t shape,
                                        BinaryOp binary_op, bool extract_values, bool extract_indexes, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const T* input_ = input.get();
        const V* rhs_ = rhs.get();
        std::vector<value_t> values;
        std::vector<index_t> indexes;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        if (binary_op(lhs, rhs_[indexing::at(i, j, k, l, rhs_stride)])) {
                            const size_t offset = indexing::at(i, j, k, l, input_stride);
                            if (extract_values)
                                values.emplace_back(static_cast<value_t>(rhs_[offset]));
                            if (extract_indexes)
                                indexes.emplace_back(static_cast<index_t>(offset));
                        }
                    }
                }
            }
        }
        return details::prepareExtracted(values, indexes);
    }

    template<typename value_t, typename index_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, index_t> extract(const shared_t<T[]>& input, size4_t input_stride,
                                        const shared_t<U[]>& lhs, size4_t lhs_stride,
                                        const shared_t<V[]>& rhs, size4_t rhs_stride,
                                        size4_t shape, BinaryOp binary_op, bool extract_values, bool extract_indexes,
                                        Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const T* input_ = input.get();
        const U* lhs_ = lhs.get();
        const V* rhs_ = rhs.get();
        std::vector<value_t> values;
        std::vector<index_t> indexes;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        if (binary_op(lhs_[indexing::at(i, j, k, l, lhs_stride)],
                                      rhs_[indexing::at(i, j, k, l, rhs_stride)])) {
                            const size_t offset = indexing::at(i, j, k, l, input_stride);
                            if (extract_values)
                                values.emplace_back(static_cast<value_t>(lhs_[offset]));
                            if (extract_indexes)
                                indexes.emplace_back(static_cast<index_t>(offset));
                        }
                    }
                }
            }
        }
        return details::prepareExtracted(values, indexes);
    }

    template<typename value_t, typename index_t, typename T>
    void insert(const Extracted<value_t, index_t>& extracted, const shared_t<T[]>& output, Stream& stream) {
        stream.enqueue([=]() {
            const value_t* elements = extracted.values.get();
            const index_t* indexes = extracted.indexes.get();
            for (size_t idx = 0; idx < extracted.count; ++idx, ++elements, ++indexes)
                output.get()[*indexes] = static_cast<value_t>(*elements);
        });
    }
}
