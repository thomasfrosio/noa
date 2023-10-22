#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"
#include "noa/cpu/utils/ReduceUnary.hpp"

#if defined(NOA_IS_OFFLINE)
#include <algorithm>

namespace noa::cpu::guts {
    template<typename ReduceOp, typename Value, typename Offset>
    constexpr bool is_valid_find_v =
            nt::is_any_v<Value, u32, u64, i32, i64, f16, f32, f64> &&
            nt::is_any_v<Offset, u32, u64, i32, i64> &&
            nt::is_any_v<ReduceOp, noa::first_min_t, noa::first_max_t, noa::last_min_t, noa::last_max_t>;

    template<typename ReduceOp, typename Value>
    constexpr Value get_initial_reduce() {
        if constexpr (nt::is_any_v<ReduceOp, noa::first_min_t, noa::last_min_t>)
            return std::numeric_limits<Value>::max();
        else
            return std::numeric_limits<Value>::min();
    }
}

namespace noa::cpu {
    template<typename ReduceOp, typename Value, typename Offset,
             typename = std::enable_if_t<guts::is_valid_find_v<ReduceOp, Value, Offset>>>
    void find_offsets(
            ReduceOp reduce_op, const Value* input,
            const Strides4<i64>& strides, const Shape4<i64>& shape,
            Offset* offsets, bool reduce_batch, bool swap_layout, i64 threads
    ) {
        const auto preprocess_op = [](Value value, i64 offset) { return Pair{value, static_cast<Offset>(offset)}; };
        const auto postprocess_op = [](const Pair<Value, Offset>& pair) { return pair.second; };

        NOA_ASSERT(is_safe_cast<Offset>(noa::offset_at((shape - 1).vec(), strides)));
        constexpr Value INITIAL_REDUCE = guts::get_initial_reduce<ReduceOp, Value>();
        noa::cpu::reduce_unary(
                input, strides, shape,
                offsets, Strides1<i64>{1}, Pair<Value, Offset>{INITIAL_REDUCE, 0},
                preprocess_op, reduce_op, postprocess_op,
                threads, reduce_batch, swap_layout);
    }

    template<typename ReduceOp, typename Value,
             typename = std::enable_if_t<guts::is_valid_find_v<ReduceOp, Value, i64>>>
    i64 find_offset(
            ReduceOp reduce_op, const Value* input,
            const Strides4<i64>& strides, const Shape4<i64>& shape,
            bool swap_layout, i64 threads
    ) {
        const auto preprocess_op = [](Value value, i64 offset) { return Pair{value, offset}; };
        const auto postprocess_op = [](const Pair<Value, i64>& pair) { return pair.second; };
        constexpr Value INITIAL_REDUCE = guts::get_initial_reduce<ReduceOp, Value>();

        i64 offset{};
        reduce_unary(
                input, strides, shape,
                &offset, Strides1<i64>{1}, Pair<Value, i64>{INITIAL_REDUCE, 0},
                preprocess_op, reduce_op, postprocess_op,
                threads, true, swap_layout);
        return offset;
    }

    template<typename ReduceOp, typename Value,
             typename = std::enable_if_t<guts::is_valid_find_v<ReduceOp, Value, i64>>>
    i64 find_offset(ReduceOp reduce_op, const Value* input, i64 elements, i64 threads) {
        NOA_ASSERT(input);
        if constexpr (std::is_same_v<ReduceOp, noa::first_min_t>) {
            const Value* ptr = std::min_element(input, input + elements);
            return static_cast<i64>(ptr - input);
        } else if constexpr (std::is_same_v<ReduceOp, noa::first_max_t>) {
            const Value* ptr = std::max_element(input, input + elements);
            return static_cast<i64>(ptr - input);
        } else {
            // Not sure how to take the last occurrence using the STL, probably with reverse iterator?
            const auto preprocess_op = [](Value value, i64 offset) { return Pair{value, offset}; };
            const auto postprocess_op = [](const Pair<Value, i64>& pair) { return pair.second; };
            constexpr Value INITIAL_REDUCE = guts::get_initial_reduce<ReduceOp, Value>();

            const auto shape = Shape4<i64>{1, 1, 1, elements};
            const auto strides = shape.strides();

            i64 offset{};
            reduce_unary(
                    input, strides, shape,
                    &offset, Strides1<i64>{1}, Pair<Value, i64>{INITIAL_REDUCE, 0},
                    preprocess_op, reduce_op, postprocess_op,
                    threads);
            return offset;
        }
    }
}
#endif
