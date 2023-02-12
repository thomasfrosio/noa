#include <algorithm>
#include "noa/core/utils/Pair.hpp"
#include "noa/cpu/Find.hpp"
#include "noa/cpu/utils/ReduceUnary.hpp"

namespace {
    template<typename ReduceOp, typename Value>
    constexpr Value get_initial_reduce() {
        if constexpr (noa::traits::is_any_v<ReduceOp, noa::first_min_t, noa::last_min_t>)
            return noa::math::Limits<Value>::max();
        else
            return noa::math::Limits<Value>::min();
    }
}

namespace noa::cpu {
    template<typename ReduceOp, typename Value, typename Offset, typename>
    void find_offsets(
            ReduceOp reduce_op, const Shared<Value[]>& input,
            const Strides4<i64>& strides, const Shape4<i64>& shape,
            const Shared<Offset[]>& offsets, bool reduce_batch, bool swap_layout, Stream& stream
    ) {
        const auto threads = stream.threads();
        stream.enqueue([=]() {
            const auto preprocess_op = [](Value value, i64 offset) { return Pair{value, offset}; };
            const auto postprocess_op = [](const Pair<Value, i64>& pair) { return static_cast<Offset>(pair.second); };

            constexpr Value INITIAL_REDUCE = get_initial_reduce<ReduceOp, Value>();
            noa::cpu::utils::reduce_unary_4d(
                    input.get(), strides, shape,
                    offsets.get(), Strides1<i64>{1}, Pair<Value, i64>{INITIAL_REDUCE, 0},
                    preprocess_op, reduce_op, postprocess_op,
                    threads, reduce_batch, swap_layout);
        });
    }

    template<typename ReduceOp, typename Value, typename>
    i64 find_offset(
            ReduceOp reduce_op, const Shared<Value[]>& input,
            const Strides4<i64>& strides, const Shape4<i64>& shape,
            bool swap_layout, Stream& stream
    ) {
        const auto preprocess_op = [](Value value, i64 offset) { return Pair{value, offset}; };
        const auto postprocess_op = [](const Pair<Value, i64>& pair) { return pair.second; };
        constexpr Value INITIAL_REDUCE = get_initial_reduce<ReduceOp, Value>();

        i64 offset{};
        stream.synchronize();
        noa::cpu::utils::reduce_unary_4d(
                input.get(), strides, shape,
                &offset, Strides1<i64>{1}, Pair<Value, i64>{INITIAL_REDUCE, 0},
                preprocess_op, reduce_op, postprocess_op,
                stream.threads(), true, swap_layout);
        return offset;
    }

    template<typename ReduceOp, typename Value, typename>
    i64 find_offset(ReduceOp reduce_op, const Shared<Value[]>& input, i64 elements, Stream& stream) {
        NOA_ASSERT(input);
        stream.synchronize();
        if constexpr (std::is_same_v<ReduceOp, noa::first_min_t>) {
            Value* ptr = std::min_element(input.get(), input.get() + elements);
            return static_cast<i64>(ptr - input.get());
        } else if constexpr (std::is_same_v<ReduceOp, noa::first_max_t>) {
            Value* ptr = std::max_element(input.get(), input.get() + elements);
            return static_cast<i64>(ptr - input.get());
        } else {
            // Not sure how to take the last occurrence using the STL, probably with reverse iterator?
            const auto preprocess_op = [](Value value, i64 offset) { return Pair{value, offset}; };
            const auto postprocess_op = [](const Pair<Value, i64>& pair) { return pair.second; };
            constexpr Value INITIAL_REDUCE = get_initial_reduce<ReduceOp, Value>();

            const auto shape = Shape4<i64>{1, 1, 1, elements};
            const auto strides = shape.strides();

            i64 offset{};
            noa::cpu::utils::reduce_unary_4d(
                    input.get(), strides, shape,
                    &offset, Strides1<i64>{1}, Pair<Value, i64>{INITIAL_REDUCE, 0},
                    preprocess_op, reduce_op, postprocess_op,
                    stream.threads());
            return offset;
        }
    }

    #define NOA_INSTANTIATE_FIND_OFFSETS(R, T, U)                           \
    template void find_offsets<R, T, U, void>(                              \
        R, const Shared<T[]>&, const Strides4<i64>&, const Shape4<i64>&,    \
        const Shared<U[]>&, bool, bool, Stream&)

    #define NOA_INSTANTIATE_FIND_OFFSETS_ALL_(R, T) \
    NOA_INSTANTIATE_FIND_OFFSETS(R, T, u32);        \
    NOA_INSTANTIATE_FIND_OFFSETS(R, T, u64);        \
    NOA_INSTANTIATE_FIND_OFFSETS(R, T, i32);        \
    NOA_INSTANTIATE_FIND_OFFSETS(R, T, i64)

    #define NOA_INSTANTIATE_FIND_OFFSET(R, T)                               \
    template i64 find_offset<R, T, void>(                                   \
        R, const Shared<T[]>&, const Strides4<i64>&, const Shape4<i64>&,    \
        bool, Stream&);                                                     \
    template i64 find_offset<R, T, void>(                                   \
        R, const Shared<T[]>&, i64, Stream&)

    #define NOA_INSTANTIATE_FIND_(T)                        \
    NOA_INSTANTIATE_FIND_OFFSETS_ALL_(noa::first_min_t, T); \
    NOA_INSTANTIATE_FIND_OFFSETS_ALL_(noa::first_max_t, T); \
    NOA_INSTANTIATE_FIND_OFFSETS_ALL_(noa::last_min_t, T);  \
    NOA_INSTANTIATE_FIND_OFFSETS_ALL_(noa::last_max_t, T);  \
    NOA_INSTANTIATE_FIND_OFFSET(noa::first_min_t, T);       \
    NOA_INSTANTIATE_FIND_OFFSET(noa::first_max_t, T);       \
    NOA_INSTANTIATE_FIND_OFFSET(noa::last_min_t, T);        \
    NOA_INSTANTIATE_FIND_OFFSET(noa::last_max_t, T)

    NOA_INSTANTIATE_FIND_(u32);
    NOA_INSTANTIATE_FIND_(i32);
    NOA_INSTANTIATE_FIND_(u64);
    NOA_INSTANTIATE_FIND_(i64);
    NOA_INSTANTIATE_FIND_(f16);
    NOA_INSTANTIATE_FIND_(f32);
    NOA_INSTANTIATE_FIND_(f64);
}
