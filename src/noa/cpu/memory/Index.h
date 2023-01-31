/// \file noa/cpu/memory/Index.h
/// \brief Remapping functions.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

// -- Using coordinates -- //
namespace noa::cpu::memory {
    // Extracts one or multiple ND (1 <= N <= 3) subregions at various locations in the input array.
    template<typename Value, typename = std::enable_if_t<traits::is_restricted_data_v<Value>>>
    void extract_subregions(const shared_t<Value[]>& input, dim4_t input_strides, dim4_t input_shape,
                            const shared_t<Value[]>& subregions, dim4_t subregion_strides, dim4_t subregion_shape,
                            const shared_t<int4_t[]>& origins, BorderMode border_mode, Value border_value,
                            Stream& stream);

    // Inserts into the output array one or multiple ND (1 <= N <= 3) subregions at various locations.
    template<typename Value, typename = std::enable_if_t<traits::is_restricted_data_v<Value>>>
    void insert_subregions(const shared_t<Value[]>& subregions, dim4_t subregion_strides, dim4_t subregion_shape,
                           const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                           const shared_t<int4_t[]>& origins, Stream& stream);
}

// -- Using a sequence of linear indexes -- //
namespace noa::cpu::memory {
    template<typename T, typename I>
    struct Extracted {
        shared_t<T[]> values{};
        shared_t<I[]> offsets{};
        dim_t count{};
    };

    // Extracts elements (and/or offsets) from the input array based on an unary bool operator.
    template<typename value_t, typename offset_t, typename T, typename U, typename UnaryOp>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                         const shared_t<U[]>& lhs, dim4_t lhs_strides, dim4_t shape,
                                         UnaryOp unary_op, bool extract_values, bool extract_offsets, Stream& stream);

    // Extracts elements (and/or offsets) from the input array based on a binary bool operator.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                         const shared_t<U[]>& lhs, dim4_t lhs_strides, V rhs, dim4_t shape,
                                         BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream);

    // Extracts elements (and/or offsets) from the input array based on a binary bool operator.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                         U lhs, const shared_t<V[]>& rhs, dim4_t rhs_strides, dim4_t shape,
                                         BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream);

    // Extracts elements (and/or offsets) from the input array based on a binary bool operator.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                         const shared_t<U[]>& lhs, dim4_t lhs_strides,
                                         const shared_t<V[]>& rhs, dim4_t rhs_strides,
                                         dim4_t shape, BinaryOp binary_op, bool extract_values, bool extract_offsets,
                                         Stream& stream);

    // Extracts elements from the input array at particular offsets.
    // TODO Add to unified API.
    template<typename T, typename U, typename V>
    void extract(const shared_t<T[]>& input, const shared_t<U[]>& offsets,
                 const shared_t<V[]>& output, dim_t elements, Stream& stream);

    // Inserts elements into output.
    template<typename value_t, typename offset_t, typename T>
    void insert(const Extracted<value_t, offset_t>& extracted, const shared_t<T[]>& output, Stream& stream);
}

#define NOA_INDEX_INL_
#include "noa/cpu/memory/Index.inl"
#undef NOA_INDEX_INL_
