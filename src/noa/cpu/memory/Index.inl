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
            extracted.elements = PtrHost<T>::alloc(extracted.count);
            copy(elements.data(), extracted.elements.get(), extracted.count);
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

    template<typename T, typename I, typename UnaryOp>
    Extracted<T, I> extract(const shared_t<T[]>& input, size4_t stride, size4_t shape,
                            UnaryOp unary_op, bool extract_elements, bool extract_indexes, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const T* input_ptr = input.get();
        std::vector<T> elements_buffer;
        std::vector<I> indexes_buffer;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const size_t offset = indexing::at(i, j, k, l, stride);
                        if (unary_op(input_ptr[offset])) {
                            if (extract_elements)
                                elements_buffer.emplace_back(input_ptr[offset]);
                            if (extract_indexes)
                                indexes_buffer.emplace_back(static_cast<I>(offset));
                        }
                    }
                }
            }
        }
        return details::prepareExtracted(elements_buffer, indexes_buffer);
    }

    template<typename T, typename I, typename U, typename BinaryOp, typename>
    Extracted<T, I> extract(const shared_t<T[]>& lhs, size4_t lhs_stride, size4_t lhs_shape, U rhs,
                            BinaryOp binary_op, bool extract_elements, bool extract_indexes,
                            Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const T* lhs_ = lhs.get();
        std::vector<T> elements_buffer;
        std::vector<I> indexes_buffer;
        stream.synchronize();

        for (size_t i = 0; i < lhs_shape[0]; ++i) {
            for (size_t j = 0; j < lhs_shape[1]; ++j) {
                for (size_t k = 0; k < lhs_shape[2]; ++k) {
                    for (size_t l = 0; l < lhs_shape[3]; ++l) {
                        const size_t offset = indexing::at(i, j, k, l, lhs_stride);
                        if (binary_op(lhs_[offset], rhs)) {
                            if (extract_elements)
                                elements_buffer.emplace_back(lhs_[offset]);
                            if (extract_indexes)
                                indexes_buffer.emplace_back(static_cast<I>(offset));
                        }
                    }
                }
            }
        }
        return details::prepareExtracted(elements_buffer, indexes_buffer);
    }

    template<typename T, typename I, typename U, typename BinaryOp, typename>
    Extracted<T, I> extract(T lhs, const shared_t<U[]>& rhs, size4_t rhs_stride, size4_t rhs_shape,
                            BinaryOp binary_op, bool extract_elements, bool extract_indexes,
                            Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const T* rhs_ = rhs.get();
        std::vector<T> elements_buffer;
        std::vector<I> indexes_buffer;
        stream.synchronize();

        for (size_t i = 0; i < rhs_shape[0]; ++i) {
            for (size_t j = 0; j < rhs_shape[1]; ++j) {
                for (size_t k = 0; k < rhs_shape[2]; ++k) {
                    for (size_t l = 0; l < rhs_shape[3]; ++l) {
                        const size_t offset = indexing::at(i, j, k, l, rhs_stride);
                        if (binary_op(lhs, rhs_[offset])) {
                            if (extract_elements)
                                elements_buffer.emplace_back(rhs_[offset]);
                            if (extract_indexes)
                                indexes_buffer.emplace_back(static_cast<I>(offset));
                        }
                    }
                }
            }
        }
        return details::prepareExtracted(elements_buffer, indexes_buffer);
    }

    template<typename T, typename I, typename U, typename BinaryOp>
    Extracted<T, I> extract(const shared_t<T[]>& lhs, size4_t lhs_stride,
                            const shared_t<U[]>& rhs, size4_t rhs_stride,
                            size4_t shape, BinaryOp binary_op,
                            bool extract_elements, bool extract_indexes, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const T* lhs_ = lhs.get();
        const U* rhs_ = rhs.get();
        std::vector<T> elements_buffer;
        std::vector<I> indexes_buffer;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const size_t iffset = indexing::at(i, j, k, l, lhs_stride);
                        const size_t affset = indexing::at(i, j, k, l, rhs_stride);
                        if (binary_op(lhs_[iffset], rhs_[affset])) {
                            if (extract_elements)
                                elements_buffer.emplace_back(lhs_[iffset]);
                            if (extract_indexes)
                                indexes_buffer.emplace_back(static_cast<I>(iffset));
                        }
                    }
                }
            }
        }
        return details::prepareExtracted(elements_buffer, indexes_buffer);
    }

    template<typename T, typename I>
    void insert(const Extracted<T, I>& extracted, const shared_t<T[]>& output, Stream& stream) {
        stream.enqueue([=]() {
            const T* elements = extracted.elements.get();
            const I* indexes = extracted.indexes.get();
            for (size_t idx = 0; idx < extracted.count; ++idx, ++elements, ++indexes)
                output.get()[*indexes] = *elements;
        });
    }
}
