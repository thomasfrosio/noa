#pragma once

#include "noa/common/Types.h"

// We try to find contiguity in the arrays by rearranging them to the rightmost order.
// Empty and broadcast dimensions are moved to the left, and are ignored for the rest of the function.
// The input can be broadcast onto the output shape. However, the output should not broadcast
// a non-empty dimension of this input. This is not valid. However, here broadcast dimensions
// in the output are treated as empty, so the corresponding input dimension isn't used and no
// rules are technically broken (because correctness is guaranteed).

// The constness of the input is not enforced here. That way, the operator can take the input
// as reference and modify it on the fly, as well as its output. If this is not intended,
// the caller should make the input type const. The output cannot be const.

// GCC and Clang are able to see and optimize through this function. The operators are correctly
// inlined and the 1D cases can be strongly optimized using SIMD or memset/memcopy/memmove calls.
// Parallelization turns most of these optimizations off, as well as non-contiguous arrays.
// Note that passing 1 or ignoring the "threads" parameter reduce the binary size since the
// parallel version can be omitted.

namespace noa::cpu::utils::details {
    // Parallelization is expensive. Turn it on only for large arrays.
    // TODO Parallelization is even more expensive for the 1D case where it prevents a lot of inner-loop
    //      optimizations. We could have another, more stringent, heuristic here for the 1D case.
    constexpr int64_t EWISE_UNARY_PARALLEL_THRESHOLD = 16'777'216; // 4096x4096

    template<bool PARALLEL, typename InputValue, typename OutputValue, typename Index, typename Operator>
    void ewiseUnary4D(Accessor<InputValue, 4, Index> input,
                      Accessor<OutputValue, 4, Index> output,
                      const dim4_t& shape, Operator&& op, int64_t threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(4) num_threads(threads) shared(input, output, shape, op)
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
                    for (Index k = 0; k < shape[2]; ++k)
                        for (Index l = 0; l < shape[3]; ++l)
                            output(i, j, k, l) = op(input(i, j, k, l));
        } else {
            (void) threads;
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
                    for (Index k = 0; k < shape[2]; ++k)
                        for (Index l = 0; l < shape[3]; ++l)
                            output(i, j, k, l) = op(input(i, j, k, l));
        }
    }

    template<bool PARALLEL, typename Value, typename Index, typename Operator,
             typename = std::enable_if_t<!std::is_const_v<Value>>>
    void ewiseUnary1D(Value* input_output, Index size, Operator&& op, int64_t threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) num_threads(threads) shared(input_output, size, op)
            for (Index i = 0; i < size; ++i)
                input_output[i] = op(input_output[i]);
        } else {
            (void) threads;
            for (Index i = 0; i < size; ++i)
                input_output[i] = op(input_output[i]);
        }
    }

    template<bool PARALLEL, typename InputValue, typename OutputValue, typename Index, typename Operator>
    void ewiseUnary1D(InputValue* input,
                      OutputValue* output,
                      Index size, Operator&& op, int64_t threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) num_threads(threads) shared(input, output, size, op)
            for (Index i = 0; i < size; ++i)
                output[i] = op(input[i]);
        } else {
            (void) threads;
            for (Index i = 0; i < size; ++i)
                output[i] = op(input[i]);
        }
    }

    template<bool PARALLEL, typename InputValue, typename OutputValue, typename Index, typename Operator>
    void ewiseUnary1DRestrict(InputValue* __restrict input,
                              OutputValue* __restrict output,
                              Index size, Operator&& op, int64_t threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) num_threads(threads) shared(input, output, size, op)
            for (Index i = 0; i < size; ++i)
                output[i] = op(input[i]);
        } else {
            (void) threads;
            for (Index i = 0; i < size; ++i)
                output[i] = op(input[i]);
        }
    }
}

namespace noa::cpu::utils {
    template<typename InputValue, typename OutputValue,
             typename Index, typename Operator, typename Int = int64_t,
             typename = std::enable_if_t<std::is_integral_v<Int> && !std::is_const_v<OutputValue>>>
    constexpr void ewiseUnary(InputValue* input, Int4<Index> input_strides,
                              OutputValue* output, Int4<Index> output_strides,
                              Int4<Index> shape, Operator&& op, Int threads = Int{1}) {
        // Rearrange to rightmost order.
        shape = noa::indexing::effectiveShape(shape, output_strides);
        const auto order = noa::indexing::order(output_strides, shape);
        if (noa::all(order != Int4<Index>{0, 1, 2, 3})) {
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
        const int64_t threads_omp =
                elements <= details::EWISE_UNARY_PARALLEL_THRESHOLD ?
                1 : clamp_cast<int64_t>(threads);
        const bool serial = threads_omp <= 1;

        // If contiguous, go to special 1D case to hopefully trigger optimizations.
        // The in-place case is also treated as a special case so that 1) it generates
        // better code for the in-place case, and 2) it generates better code for the
        // out-of-place case since we can guarantee there's no aliasing.
        const bool is_contiguous =
                noa::indexing::areContiguous(input_strides, shape) &&
                noa::indexing::areContiguous(output_strides, shape);
        if (is_contiguous) {
            // Input and output can be of the same type, meaning that the input is not const.
            // In this case, we can simplify the ewise operation to a single array.
            if constexpr (std::is_same_v<std::remove_cv_t<InputValue>, OutputValue>) {
                const bool are_equal = static_cast<void*>(input) == static_cast<void*>(output);
                if (are_equal) {
                    if (serial) {
                        details::ewiseUnary1D<false>(
                                output, elements, std::forward<Operator>(op), 1);
                    } else {
                        details::ewiseUnary1D<true>(
                                output, elements, std::forward<Operator>(op), threads_omp);
                    }
                } else {
                    if (serial) {
                        details::ewiseUnary1DRestrict<false>(
                                input, output, elements, std::forward<Operator>(op), 1);
                    } else {
                        details::ewiseUnary1DRestrict<true>(
                                input, output, elements, std::forward<Operator>(op), threads_omp);
                    }
                }
            } else {
                if (serial) {
                    details::ewiseUnary1D<false>(
                            input, output, elements, std::forward<Operator>(op), 1);
                } else {
                    details::ewiseUnary1D<true>(
                            input, output, elements, std::forward<Operator>(op), threads_omp);
                }
            }
        } else {
            // Not contiguous. Run 4 nested loops. Optimizations regarding the element-wise
            // loops are likely to be turned off because of the dynamic strides.
            const auto input_accessor = Accessor<InputValue, 4, Index>(input, input_strides);
            const auto output_accessor = Accessor<OutputValue, 4, Index>(output, output_strides);
            if (serial) {
                details::ewiseUnary4D<false>(
                        input_accessor, output_accessor, shape,
                        std::forward<Operator>(op), 1);
            } else {
                details::ewiseUnary4D<true>(
                        input_accessor, output_accessor, shape,
                        std::forward<Operator>(op), threads_omp);
            }
        }
    }
}
