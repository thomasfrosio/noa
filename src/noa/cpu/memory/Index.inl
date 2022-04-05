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
    Extracted<T, I> extract(const shared_t<const T[]>& input, size4_t stride, size4_t shape,
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
    Extracted<T, I> extract(const shared_t<const T[]>& input, size4_t stride, size4_t shape, U value,
                            BinaryOp binary_op, bool extract_elements, bool extract_indexes,
                            Stream& stream) {
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
                        if (binary_op(input_ptr[offset], value)) {
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

    template<typename T, typename I, typename U, typename BinaryOp>
    Extracted<T, I> extract(const shared_t<const T[]>& input, size4_t stride, size4_t shape,
                            const shared_t<const U[]>& values, BinaryOp binary_op,
                            bool extract_elements, bool extract_indexes, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const T* input_ptr = input.get();
        const U* values_ptr = values.get();
        std::vector<T> elements_buffer;
        std::vector<I> indexes_buffer;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const size_t offset = indexing::at(i, j, k, l, stride);
                        if (binary_op(input_ptr[offset], values_ptr[i])) {
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

    template<typename T, typename I, typename U, typename BinaryOp>
    Extracted<T, I> extract(const shared_t<const T[]>& input, size4_t input_stride,
                            const shared_t<const U[]>& array, size4_t array_stride,
                            size4_t shape, BinaryOp binary_op,
                            bool extract_elements, bool extract_indexes, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const T* input_ptr = input.get();
        const U* array_ptr = array.get();
        std::vector<T> elements_buffer;
        std::vector<I> indexes_buffer;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const size_t iffset = indexing::at(i, j, k, l, input_stride);
                        const size_t affset = indexing::at(i, j, k, l, array_stride);
                        if (binary_op(input_ptr[iffset], array_ptr[affset])) {
                            if (extract_elements)
                                elements_buffer.emplace_back(input_ptr[iffset]);
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
    void insert(const Extracted<T, I>& extracted, shared_t<T[]>& output, Stream& stream) {
        stream.enqueue([=]() {
            const T* elements = extracted.elements.get();
            const I* indexes = extracted.indexes.get();
            for (size_t idx = 0; idx < extracted.count; ++idx, ++elements, ++indexes)
                output.get()[*indexes] = *elements;
        });
    }
}
