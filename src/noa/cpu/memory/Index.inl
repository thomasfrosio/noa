#pragma once

#ifndef NOA_INDEX_INL_
#error "This is an internal header; it should not be included."
#endif

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"

// I couldn't find a way to immediately return the output pointer since the size is unknown when the function
// is called and asking the callee to synchronize before using the output (even passing by value) is pointless.
// Thus, the extract functions will synchronize the stream...

namespace noa::cpu::memory::details {
    // TODO Since in CUDA we don't have the equivalent of std::vector (or a device allocator compatible with
    //      STL containers), the extract functions return C arrays. This should be revisited at some point.
    template<typename T>
    T* releaseVector(std::vector<T>& vector) {
        PtrHost<T> sequence;
        if (!vector.empty()) {
            sequence.reset(vector.size());
            copy(vector.data(), sequence.get(), vector.size());
            vector.clear();
        }
        return sequence.release(); // not super safe...
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

    template<bool EXTRACT, typename I, typename T, typename UnaryOp>
    std::tuple<T*, I*, size_t> extract(const T* input, size4_t stride, size4_t shape,
                                       UnaryOp unary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        std::vector<T> elements_buffer; // std::vector<std::pair<T,I>> instead?
        std::vector<I> indexes_buffer;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const size_t offset = at(i, j, k, l, stride);
                        if (unary_op(input[offset])) {
                            if constexpr (EXTRACT)
                                elements_buffer.emplace_back(input[offset]);
                            if constexpr(!noa::traits::is_same_v<I, void>)
                                indexes_buffer.emplace_back(static_cast<I>(offset));
                        }
                    }
                }
            }
        }
        const size_t extracted = noa::math::max(elements_buffer.size(), indexes_buffer.size());
        return {details::releaseVector(elements_buffer), details::releaseVector(indexes_buffer), extracted};
    }

    template<bool EXTRACT, typename I, typename T, typename U, typename BinaryOp>
    std::tuple<T*, I*, size_t> extract(const T* input, size4_t stride, size4_t shape, U values,
                                       BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        std::vector<T> elements_buffer;
        std::vector<I> indexes_buffer;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            const std::remove_pointer_t<U> value;
            if constexpr (std::is_pointer_v<U>)
                value = values[i];
            else
                value = values;

            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const size_t offset = at(i, j, k, l, stride);
                        if (binary_op(input[offset], value)) {
                            if constexpr (EXTRACT)
                                elements_buffer.emplace_back(input[offset]);
                            if constexpr(!noa::traits::is_same_v<I, void>)
                                indexes_buffer.emplace_back(static_cast<I>(offset));
                        }
                    }
                }
            }
        }
        const size_t extracted = noa::math::max(elements_buffer.size(), indexes_buffer.size());
        return {details::releaseVector(elements_buffer), details::releaseVector(indexes_buffer), extracted};
    }

    template<bool EXTRACT, typename I, typename T, typename U, typename BinaryOp>
    std::tuple<T*, I*, size_t> extract(const T* input, size4_t input_stride,
                                       const U* array, size4_t array_stride,
                                       size4_t shape, BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        std::vector<T> elements_buffer;
        std::vector<I> indexes_buffer;
        stream.synchronize();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const size_t iffset = at(i, j, k, l, input_stride);
                        const size_t affset = at(i, j, k, l, array_stride);
                        if (binary_op(input[iffset], array[affset])) {
                            if constexpr (EXTRACT)
                                elements_buffer.emplace_back(input[iffset]);
                            if constexpr(!noa::traits::is_same_v<I, void>)
                                indexes_buffer.emplace_back(static_cast<I>(iffset));
                        }
                    }
                }
            }
        }
        const size_t extracted = noa::math::max(elements_buffer.size(), indexes_buffer.size());
        return {details::releaseVector(elements_buffer), details::releaseVector(indexes_buffer), extracted};
    }

    template<typename T, typename I>
    void insert(const T* sequence_values, const I* sequence_indexes, size_t sequence_size,
                T* output, Stream& stream) {
        stream.enqueue([=]() mutable {
            for (size_t idx = 0; idx < sequence_size; ++idx, ++sequence_values, ++sequence_indexes)
                output[*sequence_indexes] = *sequence_values;
        });
    }
}
