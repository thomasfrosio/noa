#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::details {
    template<typename ReduceOp, typename Value, typename Offset>
    constexpr bool is_valid_find_v =
            traits::is_any_v<Value, u32, u64, i32, i64, f16, f32, f64> &&
            traits::is_any_v<Offset, u32, u64, i32, i64> &&
            traits::is_any_v<ReduceOp, noa::first_min_t, noa::first_max_t, noa::last_min_t, noa::last_max_t>;
}

namespace noa::cuda {
    template<typename ReduceOp, typename Value, typename Offset,
             typename = std::enable_if_t<details::is_valid_find_v<ReduceOp, Value, Offset>>>
    void find_offsets(ReduceOp reduce_op, const Shared<Value[]>& input,
                      const Strides4<i64>& strides, const Shape4<i64>& shape,
                      const Shared<Offset[]>& offsets, bool reduce_batch, bool swap_layout, Stream& stream);

    template<typename ReduceOp, typename Value,
             typename = std::enable_if_t<details::is_valid_find_v<ReduceOp, Value, i64>>>
    i64 find_offset(ReduceOp reduce_op, const Shared<Value[]>& input,
                    const Strides4<i64>& strides, const Shape4<i64>& shape,
                    bool swap_layout, Stream& stream);

    template<typename ReduceOp, typename Value,
             typename = std::enable_if_t<details::is_valid_find_v<ReduceOp, Value, i64>>>
    i64 find_offset(ReduceOp reduce_op, const Shared<Value[]>& input, i64 elements, Stream& stream);
}
