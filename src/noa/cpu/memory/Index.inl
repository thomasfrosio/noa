#pragma once

#ifndef NOA_INDEX_INL_
#error "This is an internal header. Include the corresponding .h file instead"
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
    template<typename T, typename>
    dim4_t atlasLayout(dim4_t subregion_shape, T* origins) {
        NOA_ASSERT(origins && all(subregion_shape > 0));
        using namespace noa::math;
        const auto col = static_cast<dim_t>(ceil(sqrt(static_cast<float>(subregion_shape[0]))));
        const dim_t row = (subregion_shape[0] + col - 1) / col;
        const dim4_t atlas_shape{1, subregion_shape[1], row * subregion_shape[2], col * subregion_shape[3]};
        for (dim_t y = 0; y < row; ++y) {
            for (dim_t x = 0; x < col; ++x) {
                const dim_t idx = y * col + x;
                if (idx >= subregion_shape[0])
                    break;
                if constexpr (traits::is_int4_v<T>)
                    origins[idx] = {0, 0, y * subregion_shape[2], x * subregion_shape[3]};
                else
                    origins[idx] = {y * subregion_shape[2], x * subregion_shape[3]};
            }
        }
        return atlas_shape;
    }

    template<typename value_t, typename offset_, typename T, typename U, typename UnaryOp>
    Extracted<value_t, offset_> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                        const shared_t<U[]>& lhs, dim4_t lhs_strides, dim4_t shape,
                                        UnaryOp unary_op, bool extract_values, bool extract_offsets, Stream& stream) {
        NOA_ASSERT((input || !extract_values) && lhs && all(shape > 0 ));
        const dim4_t order = indexing::order(input_strides, shape);
        input_strides = indexing::reorder(input_strides, order);
        lhs_strides = indexing::reorder(lhs_strides, order);
        shape = indexing::reorder(shape, order);

        const T* input_ = input.get();
        const U* lhs_ = lhs.get();
        std::vector<value_t> values;
        std::vector<offset_> offsets;
        stream.synchronize();

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {
                    for (dim_t l = 0; l < shape[3]; ++l) {
                        if (unary_op(lhs_[indexing::at(i, j, k, l, lhs_strides)])) {
                            const dim_t offset = indexing::at(i, j, k, l, input_strides);
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
    Extracted<value_t, offset_> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                        const shared_t<U[]>& lhs, dim4_t lhs_strides, V rhs, dim4_t shape,
                                        BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream) {
        NOA_ASSERT((input || !extract_values) && lhs && all(shape > 0 ));
        const dim4_t order = indexing::order(input_strides, shape);
        input_strides = indexing::reorder(input_strides, order);
        lhs_strides = indexing::reorder(lhs_strides, order);
        shape = indexing::reorder(shape, order);

        const T* input_ = input.get();
        const U* lhs_ = lhs.get();
        std::vector<value_t> values;
        std::vector<offset_> offsets;
        stream.synchronize();

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {
                    for (dim_t l = 0; l < shape[3]; ++l) {
                        if (binary_op(lhs_[indexing::at(i, j, k, l, lhs_strides)], rhs)) {
                            const dim_t offset = indexing::at(i, j, k, l, input_strides);
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
    Extracted<value_t, offset_> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                        U lhs, const shared_t<V[]>& rhs, dim4_t rhs_strides, dim4_t shape,
                                        BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream) {
        NOA_ASSERT((input || !extract_values) && rhs && all(shape > 0 ));
        const dim4_t order = indexing::order(input_strides, shape);
        input_strides = indexing::reorder(input_strides, order);
        rhs_strides = indexing::reorder(rhs_strides, order);
        shape = indexing::reorder(shape, order);

        const T* input_ = input.get();
        const V* rhs_ = rhs.get();
        std::vector<value_t> values;
        std::vector<offset_> offsets;
        stream.synchronize();

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {
                    for (dim_t l = 0; l < shape[3]; ++l) {
                        if (binary_op(lhs, rhs_[indexing::at(i, j, k, l, rhs_strides)])) {
                            const dim_t offset = indexing::at(i, j, k, l, input_strides);
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
    Extracted<value_t, offset_> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                        const shared_t<U[]>& lhs, dim4_t lhs_strides,
                                        const shared_t<V[]>& rhs, dim4_t rhs_strides,
                                        dim4_t shape, BinaryOp binary_op, bool extract_values, bool extract_offsets,
                                        Stream& stream) {
        NOA_ASSERT((input || !extract_values) && lhs && rhs && all(shape > 0 ));
        const dim4_t order = indexing::order(input_strides, shape);
        input_strides = indexing::reorder(input_strides, order);
        lhs_strides = indexing::reorder(lhs_strides, order);
        rhs_strides = indexing::reorder(rhs_strides, order);
        shape = indexing::reorder(shape, order);

        const T* input_ = input.get();
        const U* lhs_ = lhs.get();
        const V* rhs_ = rhs.get();
        std::vector<value_t> values;
        std::vector<offset_> offsets;
        stream.synchronize();

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {
                    for (dim_t l = 0; l < shape[3]; ++l) {
                        if (binary_op(lhs_[indexing::at(i, j, k, l, lhs_strides)],
                                      rhs_[indexing::at(i, j, k, l, rhs_strides)])) {
                            const dim_t offset = indexing::at(i, j, k, l, input_strides);
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
                 const shared_t<V[]>& output, dim_t elements, Stream& stream){
        NOA_ASSERT(input && output && offsets);
        stream.enqueue([=]() {
            const auto* input_ = input.get();
            const auto* offsets_ = offsets.get();
            auto* output_ = output.get();
            for (dim_t idx = 0; idx < elements; ++idx, ++offsets_, ++output_)
                *output_ = static_cast<V>(input_[*offsets_]);
        });
    }

    template<typename value_t, typename offset_, typename T>
    void insert(const Extracted<value_t, offset_>& extracted, const shared_t<T[]>& output, Stream& stream) {
        NOA_ASSERT(extracted.values && extracted.offsets && output);
        stream.enqueue([=]() {
            const auto* elements = extracted.values.get();
            const auto* offsets = extracted.offsets.get();
            auto* output_ = output.get();
            for (dim_t idx = 0; idx < extracted.count; ++idx, ++elements, ++offsets)
                output_[*offsets] = static_cast<value_t>(*elements);
        });
    }
}
