#pragma once

#include <omp.h>
#include "noa/core/Types.hpp"
#include "noa/cpu/utils/ReduceUnary.hpp"

// This is very similar than the reduce_unary functions, except that the preprocess-operator
// has to take two input values, combine them and return a Reduced value. Also, passing an
// offset to the preprocess-operator as third argument is currently not supported.
// (Lhs, Rhs) -> PreProcessOp(Lhs, Rhs) -> Reduced -> PostprocessOp(Reduced) -> Output

// Parallel reductions:
namespace noa::cpu::utils::details {
    template<typename Lhs, typename Rhs, typename Reduced, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    auto reduce_binary_4d_parallel(
            Accessor<Lhs, 4, Index> lhs,
            Accessor<Rhs, 4, Index> rhs,
            Shape4<Index> shape,
            Reduced initial_reduce,
            PreProcessOp pre_process_op,
            ReduceOp reduce_op,
            PostProcessOp&& post_process_op,
            i64 threads) {
        auto final_reduce = initial_reduce;

        #pragma omp parallel default(none) num_threads(threads) \
        shared(lhs, rhs, shape, initial_reduce, final_reduce) \
        firstprivate(reduce_op, pre_process_op)
        {
            const i64 thread_id = omp_get_thread_num();
            Reduced local_reduce = initial_reduce;
            if constexpr (nt::is_detected_v<nt::has_initialize, ReduceOp>)
                reduce_op.initialize(thread_id);

            #pragma omp for collapse(4)
            for (Index i = 0; i < shape[0]; ++i) {
                for (Index j = 0; j < shape[1]; ++j) {
                    for (Index k = 0; k < shape[2]; ++k) {
                        for (Index l = 0; l < shape[3]; ++l) {
                            const Reduced preprocessed = pre_process_op(lhs(i, j, k, l), rhs(i, j, k, l));
                            local_reduce = reduce_op(local_reduce, preprocessed);
                        }
                    }
                }
            }
            #pragma omp critical
            {
                final_reduce = reduce_op(final_reduce, local_reduce);
                if constexpr (nt::is_detected_v<nt::has_closure, ReduceOp>)
                    reduce_op.closure(thread_id);
            }
        }
        return post_process_op(final_reduce);
    }

    template<typename Lhs, typename Rhs, typename Reduced, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    auto reduce_binary_1d_parallel(
            Lhs* lhs, Rhs* rhs, Index size,
            Reduced initial_reduce,
            PreProcessOp&& pre_process_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op,
            i64 threads) {
        auto final_reduce = initial_reduce;

        #pragma omp parallel default(none) num_threads(threads) \
        shared(lhs, rhs, size, initial_reduce, final_reduce)    \
        firstprivate(reduce_op, pre_process_op)
        {
            const i64 thread_id = omp_get_thread_num();
            Reduced local_reduce = initial_reduce;
            if constexpr (nt::is_detected_v<nt::has_initialize, ReduceOp>)
                reduce_op.initialize(thread_id);

            #pragma omp for
            for (Index i = 0; i < size; ++i)
                local_reduce = reduce_op(local_reduce, pre_process_op(lhs[i], rhs[i]));

            #pragma omp critical
            {
                final_reduce = reduce_op(final_reduce, local_reduce);
                if constexpr (nt::is_detected_v<nt::has_closure, ReduceOp>)
                    reduce_op.closure(thread_id);
            }
        }
        return post_process_op(final_reduce);
    }
}

// Serial reductions:
namespace noa::cpu::utils::details {
    template<typename Lhs, typename Rhs, typename Reduced, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    auto reduce_binary_4d_serial(
            Accessor<Lhs, 4, Index> lhs,
            Accessor<Rhs, 4, Index> rhs,
            Shape4<Index> shape,
            Reduced initial_reduce,
            PreProcessOp&& pre_process_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op) {
        auto reduce = initial_reduce;
        if constexpr (nt::is_detected_v<nt::has_initialize, ReduceOp>)
            reduce_op.initialize(0);
        for (Index i = 0; i < shape[0]; ++i)
            for (Index j = 0; j < shape[1]; ++j)
                for (Index k = 0; k < shape[2]; ++k)
                    for (Index l = 0; l < shape[3]; ++l)
                        reduce = reduce_op(reduce, pre_process_op(lhs(i, j, k, l), rhs(i, j, k, l)));
        if constexpr (nt::is_detected_v<nt::has_closure, ReduceOp>)
            reduce_op.closure(0);
        return post_process_op(reduce);
    }

    template<typename Lhs, typename Rhs, typename Reduced, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    auto reduce_binary_1d_serial(
            Lhs* lhs, Rhs* rhs, Index size,
            Reduced initial_reduce,
            PreProcessOp&& pre_process_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op) {
        auto reduce = initial_reduce;
        if constexpr (nt::is_detected_v<nt::has_initialize, ReduceOp>)
            reduce_op.initialize(0);
        for (Index i = 0; i < size; ++i)
            reduce = reduce_op(reduce, pre_process_op(lhs[i], rhs[i]));
        if constexpr (nt::is_detected_v<nt::has_closure, ReduceOp>)
            reduce_op.closure(0);
        return post_process_op(reduce);
    }
}

namespace noa::cpu::utils {
    // Generic element-wise 4D reduction.
    template<typename Lhs, typename Rhs, typename Reduced, typename Output, typename Index,
             typename PreProcessOp = noa::copy_t,
             typename ReduceOp,
             typename PostProcessOp = noa::copy_t,
             typename = std::enable_if_t<!std::is_const_v<Reduced> && !std::is_const_v<Output>>>
    constexpr void reduce_binary(
            Lhs* lhs, Strides4<Index> lhs_strides,
            Rhs* rhs, Strides4<Index> rhs_strides,
            Shape4<Index> shape,
            Output* output, Strides1<Index> output_stride,
            Reduced initial_reduce,
            PreProcessOp&& pre_process_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op,
            i64 threads = 1,
            bool reduce_batch = true,
            bool swap_layout = true) {

        static_assert(traits::is_detected_exact_v<Reduced, traits::has_binary_operator, PreProcessOp, Lhs, Rhs>);
        static_assert(traits::is_detected_exact_v<Reduced, traits::has_binary_operator, ReduceOp, Reduced, Reduced>);
        static_assert(traits::is_detected_convertible_v<Output, traits::has_unary_operator, PostProcessOp, Reduced>);

        if (noa::any(shape <= 0))
            return;
        NOA_ASSERT(lhs && rhs && output);

        // Rearrange to rightmost order.
        if (swap_layout) {
            if (reduce_batch) {
                const auto lhs_order = noa::indexing::order(lhs_strides, shape);
                const auto rhs_order = noa::indexing::order(rhs_strides, shape);
                if (noa::all(lhs_order == rhs_order) && noa::any(lhs_order != Vec4<i64>{0, 1, 2, 3})) {
                    shape = noa::indexing::reorder(shape, lhs_order);
                    lhs_strides = noa::indexing::reorder(lhs_strides, lhs_order);
                    rhs_strides = noa::indexing::reorder(rhs_strides, lhs_order);
                }
            } else {
                const auto lhs_order_3d = noa::indexing::order(lhs_strides.pop_front(), shape.pop_front()) + 1;
                const auto rhs_order_3d = noa::indexing::order(rhs_strides.pop_front(), shape.pop_front()) + 1;
                if (noa::all(lhs_order_3d == rhs_order_3d) && noa::any(lhs_order_3d != Vec3<i64>{1, 2, 3})) {
                    const auto order = lhs_order_3d.push_front(0);
                    shape = noa::indexing::reorder(shape, order);
                    lhs_strides = noa::indexing::reorder(lhs_strides, order);
                    rhs_strides = noa::indexing::reorder(rhs_strides, order);
                }
            }
        }

        const Index batches_to_reduce = reduce_batch ? 1 : shape[0];
        const Index elements = reduce_batch ? shape.elements() : shape.pop_front().elements();
        const Vec4<bool> is_contiguous = noa::indexing::is_contiguous(lhs_strides, shape) &&
                                         noa::indexing::is_contiguous(rhs_strides, shape);
        const bool parallel = threads > 1 && elements > details::REDUCTION_PARALLEL_THRESHOLD;

        if ((reduce_batch || is_contiguous[0]) && noa::all(is_contiguous.pop_front())) {
            for (i64 i = 0; i < batches_to_reduce; ++i) {
                Lhs* lhs_ptr = lhs + noa::indexing::at(i, lhs_strides);
                Rhs* rhs_ptr = rhs + noa::indexing::at(i, rhs_strides);
                Output* output_ptr = output + noa::indexing::at(i, output_stride);
                if (parallel) {
                    *output_ptr = static_cast<Output>(
                            noa::cpu::utils::details::reduce_binary_1d_parallel(
                                    lhs_ptr, rhs_ptr, elements, initial_reduce,
                                    pre_process_op, reduce_op, post_process_op, threads
                            ));
                } else {
                    *output_ptr = static_cast<Output>(
                            noa::cpu::utils::details::reduce_binary_1d_serial(
                                    lhs_ptr, rhs_ptr, elements, initial_reduce,
                                    pre_process_op, reduce_op, post_process_op
                            ));
                }
            }
        } else {
            if (!reduce_batch)
                shape[0] = 1;
            for (i64 i = 0; i < batches_to_reduce; ++i) {
                Lhs* lhs_ptr = lhs + noa::indexing::at(i, lhs_strides);
                Rhs* rhs_ptr = rhs + noa::indexing::at(i, rhs_strides);
                Output* output_ptr = output + noa::indexing::at(i, output_stride);
                const auto lhs_accessor = Accessor<Lhs, 4, Index>(lhs_ptr, lhs_strides);
                const auto rhs_accessor = Accessor<Rhs, 4, Index>(rhs_ptr, rhs_strides);
                if (parallel) {
                    *output_ptr = static_cast<Output>(
                            noa::cpu::utils::details::reduce_binary_4d_parallel(
                                    lhs_accessor, rhs_accessor, shape, initial_reduce,
                                    pre_process_op, reduce_op, post_process_op, threads
                            ));
                } else {
                    *output_ptr = static_cast<Output>(
                            noa::cpu::utils::details::reduce_binary_4d_serial(
                                    lhs_accessor, rhs_accessor, shape, initial_reduce,
                                    pre_process_op, reduce_op, post_process_op
                            ));
                }
            }
        }
    }
}
