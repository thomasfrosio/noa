#pragma once

#include <omp.h>
#include "noa/core/Types.hpp"

/// Unary: Operator(Input)->Output

/// \b Reordering: we try to find contiguity in the arrays by rearranging them to the rightmost order.
/// Empty and broadcast dimensions are moved to the left, and are ignored for the rest of the function.
/// The input can be broadcast onto the output shape. However, the output should not broadcast
/// a non-empty dimension of the input. This is not valid. However, here broadcast dimensions
/// in the output are treated as empty, so the corresponding input dimension isn't used and no
/// rules are technically broken (because correctness is guaranteed).

/// \b Const: The constness of the input is not enforced here. That way, the operator can take the input
/// as reference, i.e. Operator(Input&)->Output, and modify it on the fly, as well as its output.
/// If this is not intended, the caller should make the input type const for clarity (performance-wise
/// it doesn't really matter). The output cannot be const.

/// \b Cast: The output value of the operator is explicitly cast to the output type using static_cast.
/// This is still safe, as it compiles only if types are "compatibles". Casting is often implied here
/// and this is mostly to simplify the operators, especially for cases where the output type is not explicit.

/// \p Parallel: The element-wise operation can be parallel. The operator is copied to every thread,
/// so the operator() call can be mutable without resulting in false-sharing and degraded performance.
/// Moreover, before starting the parallel loop, each thread will call (if it exists) the op.initialize(size_t)
/// member function (size_t is the thread ID), allowing per-thread initialization of the operator.

/// GCC and Clang are able to see and optimize through these function. The operators are correctly
/// inlined and the 1D cases can be strongly optimized using SIMD or memset/memcopy/memmove calls.
/// Parallelization turns most of these optimizations off, as well as non-contiguous arrays.
/// Note that passing 1 (the default value) to the "threads" parameter should reduce the amount of
/// generated code because the parallel version can be omitted.

namespace noa::cpu::utils::details {
    // Parallelization is expensive. Turn it on only for large arrays.
    // TODO Parallelization is even more expensive for the 1D case where it prevents a lot of inner-loop
    //      optimizations. We could have another, more stringent, heuristic here for the 1D case.
    constexpr i64 EWISE_UNARY_PARALLEL_THRESHOLD = 16'777'216; // 4096x4096

    template<typename Input, typename Output, typename Index, typename Operator>
    void ewise_unary_4d_parallel(
            Accessor<Input, 4, Index> input,
            Accessor<Output, 4, Index> output,
            Shape4<Index> shape, Operator op, i64 threads) {

        // We could use shared(op), but:
        // - We assume op is cheap to copy (often, it is empty), so the one-per-thread call to
        //   the copy constructor with firstprivate(op) is assumed to be not-significant.
        // - If op() is read-only, then shared(op) and firstprivate(op) should be exactly the
        //   same performance wise (other than the initial copy constructor with firstprivate).
        //   If op() modifies its state, then shared(op) creates a false-share and invalids the cache,
        //   resulting in worse performance. However, with firstprivate(op) it doesn't happen because
        //   each thread has its own copy of op.
        // - firstprivate(op) allows more complex operation, because now op can have a per-thread
        //   state, which is correctly copy-initialized from the original op. This was originally
        //   developed for random number generation, where the op() call was writing to op member
        //   variable and needed to be initialized for every thread.
        #pragma omp parallel default(none) num_threads(threads) shared(input, output, shape) firstprivate(op)
        {
            if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
                op.initialize(static_cast<i64>(omp_get_thread_num()));

            #pragma omp for collapse(4)
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
                    for (Index k = 0; k < shape[2]; ++k)
                        for (Index l = 0; l < shape[3]; ++l)
                            output(i, j, k, l) = static_cast<Output>(op(input(i, j, k, l)));

            if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
                op.closure(static_cast<i64>(omp_get_thread_num()));
        }
    }

    template<typename Input, typename Output, typename Index, typename Operator>
    void ewise_unary_4d_serial(
            Accessor<Input, 4, Index> input,
            Accessor<Output, 4, Index> output,
            Shape4<Index> shape, Operator&& op) {
        if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
            op.initialize(0);
        for (Index i = 0; i < shape[0]; ++i)
            for (Index j = 0; j < shape[1]; ++j)
                for (Index k = 0; k < shape[2]; ++k)
                    for (Index l = 0; l < shape[3]; ++l)
                        output(i, j, k, l) = static_cast<Output>(op(input(i, j, k, l)));
    }

    template<bool PARALLEL, typename Value, typename Index, typename Operator,
             typename = std::enable_if_t<!std::is_const_v<Value>>>
    void ewise_unary_1d(Value* input_output, Index size, Operator&& op, i64 threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) num_threads(threads) shared(input_output, size, op)
            for (Index i = 0; i < size; ++i)
                input_output[i] = static_cast<Value>(op(input_output[i]));
        } else {
            (void) threads;
            for (Index i = 0; i < size; ++i)
                input_output[i] = static_cast<Value>(op(input_output[i]));
        }
    }

    template<bool PARALLEL, typename Input, typename Output, typename Index, typename Operator>
    void ewise_unary_1d(
            Input* input,
            Output* output,
            Index size, Operator&& op, i64 threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) num_threads(threads) shared(input, output, size, op)
            for (Index i = 0; i < size; ++i)
                output[i] = static_cast<Output>(op(input[i]));
        } else {
            (void) threads;
            for (Index i = 0; i < size; ++i)
                output[i] = static_cast<Output>(op(input[i]));
        }
    }

    template<bool PARALLEL, typename Input, typename Output, typename Index, typename Operator>
    void ewise_unary_1d_restrict(
            Input* __restrict input,
            Output* __restrict output,
            Index size, Operator&& op, i64 threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) num_threads(threads) shared(input, output, size, op)
            for (Index i = 0; i < size; ++i)
                output[i] = static_cast<Output>(op(input[i]));
        } else {
            (void) threads;
            for (Index i = 0; i < size; ++i)
                output[i] = static_cast<Output>(op(input[i]));
        }
    }
}

namespace noa::cpu::utils {
    template<typename Input, typename Output,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> && !std::is_const_v<Output>>>
    constexpr void ewise_unary(
            Input* input, Strides4<Index> input_strides,
            Output* output, Strides4<Index> output_strides,
            Shape4<Index> shape, Operator&& op, Int threads = Int{1}) {
        // Rearrange to rightmost order.
        shape = noa::indexing::effective_shape(shape, output_strides);
        const auto order = noa::indexing::order(output_strides, shape);
        if (noa::all(order != Vec4<Index>{0, 1, 2, 3})) {
            shape = noa::indexing::reorder(shape, order);
            input_strides = noa::indexing::reorder(input_strides, order);
            output_strides = noa::indexing::reorder(output_strides, order);
        }

        const Index elements = shape.elements();
        if (!elements)
            return;
        NOA_ASSERT(input && output);

        // Serial vs Parallel. We must instantiate both versions here, otherwise it seems that
        // the compiler is more inclined to simplify and only generate the parallel version.
        const i64 threads_omp =
                elements <= details::EWISE_UNARY_PARALLEL_THRESHOLD ?
                1 : clamp_cast<i64>(threads);
        const bool serial = threads_omp <= 1;

        // If contiguous, go to special 1D case to hopefully trigger optimizations.
        // The in-place case is also treated as a special case so that 1) it generates
        // better code for the in-place case, and 2) it generates better code for the
        // out-of-place case since we can guarantee there's no aliasing.
        const bool is_contiguous =
                noa::indexing::are_contiguous(input_strides, shape) &&
                noa::indexing::are_contiguous(output_strides, shape);
        if (is_contiguous) {
            // Input and output can be of the same type, meaning that the input is not const.
            // In this case, we can simplify the ewise operation to a single array.
            if constexpr (std::is_same_v<std::remove_cv_t<Input>, Output>) {
                const bool are_equal = static_cast<const void*>(input) == static_cast<const void*>(output);
                if (are_equal) {
                    if (serial) {
                        details::ewise_unary_1d<false>(
                                output, elements, std::forward<Operator>(op), 1);
                    } else {
                        details::ewise_unary_1d<true>(
                                output, elements, std::forward<Operator>(op), threads_omp);
                    }
                } else {
                    if (serial) {
                        details::ewise_unary_1d_restrict<false>(
                                input, output, elements, std::forward<Operator>(op), 1);
                    } else {
                        details::ewise_unary_1d_restrict<true>(
                                input, output, elements, std::forward<Operator>(op), threads_omp);
                    }
                }
            } else {
                if (serial) {
                    details::ewise_unary_1d<false>(
                            input, output, elements, std::forward<Operator>(op), 1);
                } else {
                    details::ewise_unary_1d<true>(
                            input, output, elements, std::forward<Operator>(op), threads_omp);
                }
            }
        } else {
            // Not contiguous. Run 4 nested loops. Optimizations regarding the element-wise
            // loops are likely to be turned off because of the dynamic strides.
            const auto input_accessor = Accessor<Input, 4, Index>(input, input_strides);
            const auto output_accessor = Accessor<Output, 4, Index>(output, output_strides);
            if (serial) {
                details::ewise_unary_4d_serial(
                        input_accessor, output_accessor, shape,
                        std::forward<Operator>(op));
            } else {
                details::ewise_unary_4d_parallel(
                        input_accessor, output_accessor, shape,
                        std::forward<Operator>(op), threads_omp);
            }
        }
    }

    // Shortcut for inplace ewise.
    template<typename Value, typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> && !std::is_const_v<Value>>>
    constexpr void ewise_unary(
            Value* array, const Strides4<Index>& strides,
            Shape4<Index> shape, Operator&& op, Int threads = Int{1}) {
        ewise_unary(array, strides, array, strides, shape, std::forward<Operator>(op), threads);
    }
}
