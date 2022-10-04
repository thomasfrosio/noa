#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Functors.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory::details {
    using namespace noa::math;
    using namespace noa::traits;

    template<typename in_t, typename lhs_t, typename out_t, typename idx_t, typename op_t>
    constexpr bool is_valid_extract_unary_v =
            (is_any_v<out_t, int32_t, int64_t, uint32_t, uint64_t> || is_float_v<out_t>) &&
            are_all_same_v<in_t, lhs_t, out_t> && is_any_v<idx_t, uint32_t, uint64_t> &&
            std::is_same_v<op_t, logical_not_t>;

    template<typename in_t, typename lhs_t, typename rhs_t, typename out_t, typename idx_t, typename op_t>
    constexpr bool is_valid_extract_binary_v =
            (is_any_v<out_t, int32_t, int64_t, uint32_t, uint64_t> || is_float_v<out_t>) &&
            are_all_same_v<in_t, lhs_t, rhs_t, out_t> && is_any_v<idx_t, uint32_t, uint64_t> &&
            is_any_v<op_t, equal_t, not_equal_t, less_t, less_equal_t, greater_t, greater_equal_t>;

    template<typename in_t, typename idx_t, typename out_t>
    constexpr bool is_valid_insert_v =
            (is_any_v<out_t, int32_t, int64_t, uint32_t, uint64_t> || is_float_v<out_t>) &&
            std::is_same_v<in_t, out_t> && is_any_v<idx_t, uint32_t, uint64_t>;
}

// -- Using coordinates -- //
namespace noa::cuda::memory {
    // Extracts one or multiple ND (1 <= N <= 3) subregions at various locations in the input array.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void extract(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<T[]>& subregions, dim4_t subregion_strides, dim4_t subregion_shape,
                 const shared_t<int4_t[]>& origins, BorderMode border_mode, T border_value,
                 Stream& stream);

    // Inserts into the output array one or multiple ND (1 <= N <= 3) subregions at various locations.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void insert(const shared_t<T[]>& subregions, dim4_t subregion_strides, dim4_t subregion_shape,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                const shared_t<int4_t[]>& origins, Stream& stream);

    // Gets the atlas layout (shape + subregion origins).
    template<typename T, typename = std::enable_if_t<traits::is_int2_v<T> || traits::is_int4_v<T>>>
    inline dim4_t atlasLayout(dim4_t subregion_shape, T* origins) {
        // This a copied from noa/cpu/memory/Index.inl
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
}

// -- Using a sequence of linear indexes -- //
namespace noa::cuda::memory {
    template<typename T, typename I>
    struct Extracted {
        shared_t<T[]> values{};
        shared_t<I[]> offsets{};
        dim_t count{};
    };

    // Extracts elements (and/or offsets) from the input array based on an unary bool operator.
    template<typename value_t, typename offset_t, typename T, typename U, typename UnaryOp,
             typename = std::enable_if_t<details::is_valid_extract_unary_v<T, U, value_t, offset_t, UnaryOp>>>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                         const shared_t<U[]>& lhs, dim4_t lhs_strides, dim4_t shape,
                                         UnaryOp unary_op, bool extract_values, bool extract_offsets, Stream& stream);

    // Extracts elements (and/or offsets) from the input array based on a binary bool operator.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_extract_binary_v<T, U, V, value_t, offset_t, BinaryOp>>>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                         const shared_t<U[]>& lhs, dim4_t lhs_strides, V rhs, dim4_t shape,
                                         BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream);

    // Extracts elements (and/or offsets) from the input array based on a binary bool operator.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_extract_binary_v<T, U, V, value_t, offset_t, BinaryOp>>>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                         U lhs, const shared_t<V[]>& rhs, dim4_t rhs_strides, dim4_t shape,
                                         BinaryOp binary_op, bool extract_values, bool extract_offsets, Stream& stream);

    // Extracts elements (and/or offsets) from the input array based on a binary bool operator.
    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<details::is_valid_extract_binary_v<T, U, V, value_t, offset_t, BinaryOp>>>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                         const shared_t<U[]>& lhs, dim4_t lhs_strides,
                                         const shared_t<V[]>& rhs, dim4_t rhs_strides,
                                         dim4_t shape, BinaryOp binary_op, bool extract_values, bool extract_offsets,
                                         Stream& stream);

    // Extracts elements from the input array at particular offsets and saves them
    // contiguously in the output in the same order as specified in the output.
    // TODO Add to unified API.
    template<typename T, typename U, typename V,
            typename = std::enable_if_t<details::is_valid_insert_v<T, U, V>>>
    void extract(const shared_t<T[]>& input, const shared_t<U[]>& offsets,
                 const shared_t<V[]>& output, dim_t elements, Stream& stream);

    // Inserts elements (back) into output.
    template<typename value_t, typename offset_t, typename T,
             typename = std::enable_if_t<details::is_valid_insert_v<value_t, offset_t, T>>>
    void insert(const Extracted<value_t, offset_t>& extracted, const shared_t<T[]>& output, Stream& stream);
}
