#pragma once

#ifndef NOA_INDEX_INL_
#error "This is an internal header; it should not be included."
#endif

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"

// I couldn't find a way to immediately return the output pointer since the size is unknown when the function
// is called and asking the callee to synchronize before using the output (even passing by value) is pointless.
// Thus, the extract functions will synchronize the stream...

// TODO Since in CUDA we don't have the equivalent of std::vector (or a device allocator compatible with STL containers),
//      the extract functions shall return raw C arrays. This should be revisited at some point.

namespace noa::cpu::memory {
    size3_t atlasLayout(size3_t subregion_shape, size_t subregion_count, int3_t* origins) {
        const auto col = static_cast<size_t>(math::ceil(math::sqrt(static_cast<float>(subregion_count))));
        const size_t row = (subregion_count + col - 1) / col;
        const size3_t atlas_shape(col * subregion_shape.x, row * subregion_shape.y, subregion_shape.z);
        for (size_t y = 0; y < row; ++y) {
            for (size_t x = 0; x < col; ++x) {
                const size_t idx = y * col + x;
                if (idx >= subregion_count)
                    break;
                origins[idx] = {x * subregion_shape.x, y * subregion_shape.y, 0};
            }
        }
        return atlas_shape;
    }

    template<bool EXTRACT, typename I, typename T, typename UnaryOp>
    std::tuple<T*, I*, size_t> extract(const T* inputs, size3_t pitch, size3_t shape, size_t batches,
                                       UnaryOp unary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const size_t base = elements(pitch);
        std::vector<T> elements_buffer; // std::vector<std::pair<T,I>> instead?
        std::vector<I> indexes_buffer;
        stream.synchronize();

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = 0; x < shape.x; ++x) {
                        const size_t offset = batch * base + index(x, y, z, pitch);
                        if (unary_op(inputs[offset])) {
                            if constexpr (EXTRACT)
                                elements_buffer.emplace_back(inputs[offset]);
                            if constexpr(!noa::traits::is_same_v<I, void>)
                                indexes_buffer.emplace_back(static_cast<I>(offset));
                        }
                    }
                }
            }
        }
        PtrHost<T> seq_elements;
        PtrHost<I> seq_indexes;
        if (!elements_buffer.empty()) {
            seq_elements.reset(elements_buffer.size());
            copy(elements_buffer.data(), seq_elements.get(), elements_buffer.size());
        }
        if (!indexes_buffer.empty()) {
            seq_indexes.reset(indexes_buffer.size());
            copy(indexes_buffer.data(), seq_indexes.get(), indexes_buffer.size());
        }
        const size_t extracted = noa::math::max(elements_buffer.size(), indexes_buffer.size());
        return {seq_elements.release(), seq_indexes.release(), extracted};
    }

    template<bool EXTRACT, typename I, typename T, typename U, typename BinaryOp>
    std::tuple<T*, I*, size_t> extract(const T* inputs, size3_t input_pitch, size3_t shape, U values,
                                       size_t batches, BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const size_t base = elements(input_pitch);
        std::vector<T> elements_buffer;
        std::vector<I> indexes_buffer;
        stream.synchronize();

        for (size_t batch = 0; batch < batches; ++batch) {
            const std::remove_pointer_t<U> value;
            if constexpr (std::is_pointer_v<U>)
                value = values[batch];
            else
                value = values;

            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = 0; x < shape.x; ++x) {
                        const size_t offset = batch * base + index(x, y, z, input_pitch);
                        if (binary_op(inputs[offset], value)) {
                            if constexpr (EXTRACT)
                                elements_buffer.emplace_back(inputs[offset]);
                            if constexpr(!noa::traits::is_same_v<I, void>)
                                indexes_buffer.emplace_back(static_cast<I>(offset));
                        }
                    }
                }
            }
        }
        PtrHost<T> seq_elements;
        PtrHost<I> seq_indexes;
        if (!elements_buffer.empty()) {
            seq_elements.reset(elements_buffer.size());
            copy(elements_buffer.data(), seq_elements.get(), elements_buffer.size());
        }
        if (!indexes_buffer.empty()) {
            seq_indexes.reset(indexes_buffer.size());
            copy(indexes_buffer.data(), seq_indexes.get(), indexes_buffer.size());
        }
        const size_t extracted = noa::math::max(elements_buffer.size(), indexes_buffer.size());
        return {seq_elements.release(), seq_indexes.release(), extracted};
    }

    template<bool EXTRACT, typename I, typename T, typename U, typename BinaryOp>
    std::tuple<T*, I*, size_t> extract(const T* inputs, size3_t input_pitch,
                                       const U* arrays, size3_t array_pitch,
                                       size3_t shape, size_t batches, BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const size_t base_input = elements(input_pitch);
        const size_t base_array = elements(array_pitch);
        std::vector<T> elements_buffer;
        std::vector<I> indexes_buffer;
        stream.synchronize();

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = 0; x < shape.x; ++x) {
                        const size_t iffset = batch * base_input + index(x, y, z, input_pitch);
                        const size_t affset = batch * base_array + index(x, y, z, array_pitch);
                        if (binary_op(inputs[iffset], arrays[affset])) {
                            if constexpr (EXTRACT)
                                elements_buffer.emplace_back(inputs[iffset]);
                            if constexpr(!noa::traits::is_same_v<I, void>)
                                indexes_buffer.emplace_back(static_cast<I>(iffset));
                        }
                    }
                }
            }
        }
        PtrHost<T> seq_elements;
        PtrHost<I> seq_indexes;
        if (!elements_buffer.empty()) {
            seq_elements.reset(elements_buffer.size());
            copy(elements_buffer.data(), seq_elements.get(), elements_buffer.size());
        }
        if (!indexes_buffer.empty()) {
            seq_indexes.reset(indexes_buffer.size());
            copy(indexes_buffer.data(), seq_indexes.get(), indexes_buffer.size());
        }
        const size_t extracted = noa::math::max(elements_buffer.size(), indexes_buffer.size());
        return {seq_elements.release(), seq_indexes.release(), extracted};
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
