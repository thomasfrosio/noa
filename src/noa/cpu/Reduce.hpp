#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"
#include "noa/cpu/utils/ReduceUnary.hpp"

namespace noa::cpu {
    template<typename Input, typename Reduced, typename Output, typename Index,
             typename PreprocessOp = noa::copy_t,
             typename ReduceOp,
             typename PostProcessOp = noa::copy_t,
             typename = std::enable_if_t<!std::is_const_v<Reduced> && !std::is_const_v<Output>>>
    constexpr void reduce_unary(
            const Input* input, const Strides4<Index>& strides, const Shape4<Index>& shape,
            Output* output,
            Reduced initial_reduce,
            PreprocessOp&& preprocess_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op,
            bool reduce_batch,
            bool swap_layout,
            i64 threads) {
        noa::cpu::utils::reduce_unary(
                input, strides, shape,
                output, Strides1<i64>{1},
                initial_reduce, preprocess_op, reduce_op, post_process_op,
                threads, reduce_batch, swap_layout);
    }

    template<typename Input, typename Reduced, typename Index,
             typename PreprocessOp = noa::copy_t,
             typename ReduceOp,
             typename PostProcessOp = noa::copy_t,
             typename = std::enable_if_t<!std::is_const_v<Reduced>>>
    constexpr auto reduce_unary(
            const Input* input, const Strides4<Index>& strides, const Shape4<Index>& shape,
            Reduced initial_reduce,
            PreprocessOp&& preprocess_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op,
            bool swap_layout,
            i64 threads) {
        using return_type = std::invoke_result_t<PostProcessOp, Reduced>;
        return_type output{};
        noa::cpu::utils::reduce_unary(
                input, strides, shape,
                &output, Strides1<i64>{1},
                initial_reduce, preprocess_op, reduce_op, post_process_op,
                threads, true, swap_layout);
        return output;
    }
}
