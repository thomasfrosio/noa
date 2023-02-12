#pragma once

#include <omp.h>
#include "noa/core/Types.hpp"

// Input -> PreprocessOp(Input) -> Reduced -> PostprocessOp(Reduced) -> Output

// Parallel reductions:
namespace noa::cpu::utils::details {
    static constexpr i64 REDUCTION_PARALLEL_THRESHOLD = 16'777'216; // 4096x4096

    template<typename Input, typename Reduced, typename Index,
             typename PreprocessOp, typename ReduceOp, typename PostProcessOp>
    auto reduce_unary_4d_parallel(
            Accessor<Input, 4, Index> input,
            Shape4<Index> shape,
            Reduced initial_reduce,
            PreprocessOp preprocess_op,
            ReduceOp reduce_op,
            PostProcessOp&& post_process_op,
            i64 threads) {
        auto final_reduce = initial_reduce;

        // The operators that are used for the reduction itself are passed as firstprivate.
        // This copy-initializes in the parallel block, so that each thread ends up with a copy.
        // Practically, the difference with "shared" is that the operators can have a per-thread
        // state that can be modified before (using its .initialize() member function) or during
        // (using the operator()) the reduction. Just before the end of the parallel block,
        // the reduction operator can be called one last time in a thread-safe environment
        // using its .closure() member function.
        #pragma omp parallel default(none) num_threads(threads) \
        shared(input, shape, initial_reduce, final_reduce) \
        firstprivate(reduce_op, preprocess_op)
        {
            const i64 thread_id = omp_get_thread_num();
            Reduced local_reduce = initial_reduce;
            if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, ReduceOp>)
                reduce_op.initialize(thread_id);

            #pragma omp for collapse(4)
            for (Index i = 0; i < shape[0]; ++i) {
                for (Index j = 0; j < shape[1]; ++j) {
                    for (Index k = 0; k < shape[2]; ++k) {
                        for (Index l = 0; l < shape[3]; ++l) {
                            if constexpr (noa::traits::is_detected_v<
                                    noa::traits::has_binary_operator, PreprocessOp, Input, Index>) {
                                const auto offset = noa::indexing::at(i, j, k, l, input.strides());
                                auto& value = input.get()[offset];
                                local_reduce = reduce_op(local_reduce, preprocess_op(value, offset));
                            } else {
                                local_reduce = reduce_op(local_reduce, preprocess_op(input(i, j, k, l)));
                            }
                        }
                    }
                }
            }
            #pragma omp critical
            {
                final_reduce = reduce_op(final_reduce, local_reduce);
                if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, ReduceOp>)
                    reduce_op.closure(thread_id);
            }
        }
        return post_process_op(final_reduce);
    }

    template<typename Input, typename Reduced, typename Index,
             typename PreprocessOp, typename ReduceOp, typename PostProcessOp>
    auto reduce_unary_1d_parallel(
            Input* input,
            Index size,
            Reduced initial_reduce,
            PreprocessOp&& preprocess_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op,
            i64 threads) {
        auto final_reduce = initial_reduce;

        #pragma omp parallel default(none) num_threads(threads) \
        shared(input, size, initial_reduce, final_reduce) \
        firstprivate(reduce_op, preprocess_op)
        {
            const i64 thread_id = omp_get_thread_num();
            Reduced local_reduce = initial_reduce;
            if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, ReduceOp>)
                reduce_op.initialize(thread_id);

            #pragma omp for
            for (Index i = 0; i < size; ++i) {
                if constexpr (noa::traits::is_detected_v<noa::traits::has_binary_operator, PreprocessOp, Input, Index>)
                    local_reduce = reduce_op(local_reduce, preprocess_op(input[i], i));
                else
                    local_reduce = reduce_op(local_reduce, preprocess_op(input[i]));
            }

            #pragma omp critical
            {
                final_reduce = reduce_op(final_reduce, local_reduce);
                if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, ReduceOp>)
                    reduce_op.closure(thread_id);
            }
        }
        return post_process_op(final_reduce);
    }
}

// Serial reductions:
namespace noa::cpu::utils::details {
    template<typename Input, typename Reduced, typename Index,
             typename PreprocessOp, typename ReduceOp, typename PostProcessOp>
    auto reduce_unary_4d_serial(
            Accessor<Input, 4, Index> input,
            Shape4<Index> shape,
            Reduced initial_reduce,
            PreprocessOp&& preprocess_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op) {
        auto reduce = initial_reduce;
        if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, ReduceOp>)
            reduce_op.initialize(0);
        for (Index i = 0; i < shape[0]; ++i) {
            for (Index j = 0; j < shape[1]; ++j) {
                for (Index k = 0; k < shape[2]; ++k) {
                    for (Index l = 0; l < shape[3]; ++l) {
                        if constexpr (noa::traits::is_detected_v<
                                noa::traits::has_binary_operator, PreprocessOp, Input, Index>) {
                            const auto offset = noa::indexing::at(i, j, k, l, input.strides());
                            auto& value = input.get()[offset];
                            reduce = reduce_op(reduce, preprocess_op(value, offset));
                        } else {
                            reduce = reduce_op(reduce, preprocess_op(input(i, j, k, l)));
                        }
                    }
                }
            }
        }
        if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, ReduceOp>)
            reduce_op.closure(0);
        return post_process_op(reduce);
    }

    template<typename Input, typename Reduced, typename Index,
             typename PreprocessOp, typename ReduceOp, typename PostProcessOp>
    auto reduce_unary_1d_serial(
            Input* input,
            Index size,
            Reduced initial_reduce,
            PreprocessOp&& preprocess_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op) {
        auto reduce = initial_reduce;
        if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, ReduceOp>)
            reduce_op.initialize(0);
        for (Index i = 0; i < size; ++i) {
            if constexpr (noa::traits::is_detected_v<
                    noa::traits::has_binary_operator, PreprocessOp, Input, Index>)
                reduce = reduce_op(reduce, preprocess_op(input[i], i));
            else
                reduce = reduce_op(reduce, preprocess_op(input[i]));
        }
        if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, ReduceOp>)
            reduce_op.closure(0);
        return post_process_op(reduce);
    }
}

namespace noa::cpu::utils {
    // Generic element-wise 4D reduction.
    template<typename Input, typename Reduced, typename Output, typename Index,
             typename PreprocessOp = noa::copy_t,
             typename ReduceOp,
             typename PostProcessOp = noa::copy_t,
             typename = std::enable_if_t<!std::is_const_v<Reduced> && !std::is_const_v<Output>>>
    constexpr void reduce_unary_4d(
            Input* input, Strides4<Index> strides, Shape4<Index> shape,
            Output* output, Strides1<Index> output_stride,
            Reduced initial_reduce,
            PreprocessOp&& preprocess_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op,
            i64 threads = 1,
            bool reduce_batch = true,
            bool swap_layout = true) {

        static_assert((traits::is_detected_exact_v<Reduced, traits::has_unary_operator, PreprocessOp, Input> ||
                       traits::is_detected_exact_v<Reduced, traits::has_binary_operator, PreprocessOp, Input, Index>));
        static_assert(traits::is_detected_exact_v<Reduced, traits::has_binary_operator, ReduceOp, Reduced, Reduced>);
        static_assert(traits::is_detected_convertible_v<Output, traits::has_unary_operator, PostProcessOp, Reduced>);

        if (noa::any(shape <= 0))
            return;
        NOA_ASSERT(input && output);

        // Rearrange to rightmost order.
        if (swap_layout) {
            if (reduce_batch) {
                const auto order = noa::indexing::order(strides, shape);
                shape = noa::indexing::reorder(shape, order);
                strides = noa::indexing::reorder(strides, order);
            } else {
                const auto order_3d = noa::indexing::order(strides.pop_front(), shape.pop_front()) + 1;
                const auto order = order_3d.push_front(0);
                shape = noa::indexing::reorder(shape, order);
                strides = noa::indexing::reorder(strides, order);
            }
        }

        const Index batches_to_reduce = reduce_batch ? 1 : shape[0];
        const Index elements = reduce_batch ? shape.elements() : shape.pop_front().elements();
        const Vec4<bool> is_contiguous = noa::indexing::is_contiguous(strides, shape);
        const bool parallel = threads > 1 && elements > details::REDUCTION_PARALLEL_THRESHOLD;

        if ((reduce_batch || is_contiguous[0]) && noa::all(is_contiguous.pop_front())) {
            for (i64 i = 0; i < batches_to_reduce; ++i) {
                Input* input_ptr = input + noa::indexing::at(i, strides);
                Output* output_ptr = output + noa::indexing::at(i, output_stride);
                if (parallel) {
                    *output_ptr = static_cast<Output>(
                            noa::cpu::utils::details::reduce_unary_1d_parallel(
                                    input_ptr, elements, initial_reduce,
                                    preprocess_op, reduce_op, post_process_op, threads
                            ));
                } else {
                    *output_ptr = static_cast<Output>(
                            noa::cpu::utils::details::reduce_unary_1d_serial(
                                    input_ptr, elements, initial_reduce,
                                    preprocess_op, reduce_op, post_process_op
                            ));
                }
            }
        } else {
            if (!reduce_batch)
                shape[0] = 1;
            for (i64 i = 0; i < batches_to_reduce; ++i) {
                Input* input_ptr = input + noa::indexing::at(i, strides);
                Output* output_ptr = output + noa::indexing::at(i, output_stride);
                const auto input_accessor = Accessor<Input, 4, Index>(input_ptr, strides);
                if (parallel) {
                    *output_ptr = static_cast<Output>(
                            noa::cpu::utils::details::reduce_unary_4d_parallel(
                                    input_accessor, shape, initial_reduce,
                                    preprocess_op, reduce_op, post_process_op, threads
                            ));
                } else {
                    *output_ptr = static_cast<Output>(
                            noa::cpu::utils::details::reduce_unary_4d_serial(
                                    input_accessor, shape, initial_reduce,
                                    preprocess_op, reduce_op, post_process_op
                            ));
                }
            }
        }
    }
}
