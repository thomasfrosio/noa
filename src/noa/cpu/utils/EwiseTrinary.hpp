#pragma once

#include "noa/core/Types.hpp"
#include "noa/cpu/utils/EwiseUnary.hpp"
#include "noa/cpu/utils/EwiseBinary.hpp"

namespace noa::cpu::utils::details {
    // Parallelization is expensive. Turn it on only for large arrays.
    constexpr i64 EWISE_TRINARY_PARALLEL_THRESHOLD = 16'777'216; // 4096x4096

    template<typename Lhs, typename Mhs, typename Rhs,
             typename Output, typename Index, typename Operator>
    void ewise_trinary_4d_parallel(
            Accessor<Lhs, 4, Index> lhs,
            Accessor<Mhs, 4, Index> mhs,
            Accessor<Rhs, 4, Index> rhs,
            Accessor<Output, 4, Index> output,
            Shape4<Index> shape, Operator&& op, i64 threads) {
        #pragma omp parallel default(none) num_threads(threads) shared(lhs, mhs, rhs, output, shape) firstprivate(op)
        {
            if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
                op.initialize(omp_get_thread_num());

            #pragma omp for
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
                    for (Index k = 0; k < shape[2]; ++k)
                        for (Index l = 0; l < shape[3]; ++l)
                            output(i, j, k, l) = static_cast<Output>(op(lhs(i, j, k, l), mhs(i, j, k, l), rhs(i, j, k, l)));

            if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
                op.closure(omp_get_thread_num());
        }
    }

    template<typename Lhs, typename Mhs, typename Rhs,
             typename Output, typename Index, typename Operator>
    void ewise_trinary_4d_serial(
            Accessor<Lhs, 4, Index> lhs,
            Accessor<Mhs, 4, Index> mhs,
            Accessor<Rhs, 4, Index> rhs,
            Accessor<Output, 4, Index> output,
            Shape4<Index> shape, Operator&& op) {
        if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
            op.initialize(0);
        for (Index i = 0; i < shape[0]; ++i)
            for (Index j = 0; j < shape[1]; ++j)
                for (Index k = 0; k < shape[2]; ++k)
                    for (Index l = 0; l < shape[3]; ++l)
                        output(i, j, k, l) = static_cast<Output>(op(lhs(i, j, k, l), mhs(i, j, k, l), rhs(i, j, k, l)));
        if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
            op.closure(0);
    }

    template<typename Lhs, typename Mhs, typename Rhs,
             typename Output, typename Index, typename Operator>
    void ewise_trinary_1d_parallel(Lhs* lhs, Mhs* mhs, Rhs* rhs, Output* output,
                                   Index size, Operator&& op, i64 threads) {
        #pragma omp parallel default(none) num_threads(threads) shared(lhs, mhs, rhs, output, size) firstprivate(op)
        {
            if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
                op.initialize(omp_get_thread_num());

            #pragma omp for
            for (Index i = 0; i < size; ++i)
                output[i] = static_cast<Output>(op(lhs[i], mhs[i], rhs[i]));

            if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
                op.closure(omp_get_thread_num());
        }
    }

    template<typename Lhs, typename Mhs, typename Rhs,
             typename Output, typename Index, typename Operator>
    void ewise_trinary_1d_serial(Lhs* lhs, Mhs* mhs, Rhs* rhs, Output* output,
                                 Index size, Operator&& op) {
        if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
            op.initialize(0);
        for (Index i = 0; i < size; ++i)
            output[i] = static_cast<Output>(op(lhs[i], mhs[i], rhs[i]));
        if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
            op.closure(0);
    }

    template<typename Lhs, typename Mhs, typename Rhs,
             typename Output, typename Index, typename Operator>
    void ewise_trinary_1d_restrict_parallel(
            Lhs* __restrict lhs,
            Mhs* __restrict mhs,
            Rhs* __restrict rhs,
            Output* __restrict output,
            Index size, Operator&& op, i64 threads) {
        #pragma omp parallel default(none) num_threads(threads) shared(lhs, mhs, rhs, output, size) firstprivate(op)
        {
            if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
                op.initialize(omp_get_thread_num());

            #pragma omp for
            for (Index i = 0; i < size; ++i)
                output[i] = static_cast<Output>(op(lhs[i], mhs[i], rhs[i]));

            if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
                op.closure(omp_get_thread_num());
        }
    }

    template<typename Lhs, typename Mhs, typename Rhs,
             typename Output, typename Index, typename Operator>
    void ewise_trinary_1d_restrict_serial(
            Lhs* __restrict lhs,
            Mhs* __restrict mhs,
            Rhs* __restrict rhs,
            Output* __restrict output,
            Index size, Operator&& op) {
        if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
            op.initialize(0);
        for (Index i = 0; i < size; ++i)
            output[i] = static_cast<Output>(op(lhs[i], mhs[i], rhs[i]));
        if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
            op.closure(0);
    }
}

namespace noa::cpu::utils {
    template<typename Lhs, typename Mhs, typename Rhs, typename Output,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> && !std::is_const_v<Output>>>
    constexpr void ewise_trinary(
            Lhs* lhs, Strides4<Index> lhs_strides,
            Mhs* mhs, Strides4<Index> mhs_strides,
            Rhs* rhs, Strides4<Index> rhs_strides,
            Output* output, Strides4<Index> output_strides,
            Shape4<Index> shape, Operator&& op, Int threads = Int{1}) {
        // Rearrange to rightmost order.
        shape = noa::indexing::effective_shape(shape, output_strides);
        const auto order = noa::indexing::order(output_strides, shape);
        if (noa::all(order != Vec4<Index>{0, 1, 2, 3})) {
            shape = noa::indexing::reorder(shape, order);
            lhs_strides = noa::indexing::reorder(lhs_strides, order);
            mhs_strides = noa::indexing::reorder(mhs_strides, order);
            rhs_strides = noa::indexing::reorder(rhs_strides, order);
            output_strides = noa::indexing::reorder(output_strides, order);
        }

        const Index elements = shape.elements();
        if (!elements)
            return;
        NOA_ASSERT(lhs && mhs && rhs && output);

        const i64 threads_omp =
                elements <= details::EWISE_TRINARY_PARALLEL_THRESHOLD ?
                1 : clamp_cast<i64>(threads);
        const bool serial = threads_omp <= 1;

        const bool is_contiguous =
                noa::indexing::are_contiguous(lhs_strides, shape) &&
                noa::indexing::are_contiguous(mhs_strides, shape) &&
                noa::indexing::are_contiguous(rhs_strides, shape) &&
                noa::indexing::are_contiguous(output_strides, shape);
        if (is_contiguous) {
            constexpr bool ARE_SAME_TYPE = noa::traits::are_all_same_v<
                    std::remove_cv_t<Lhs>, std::remove_cv_t<Mhs>,
                    std::remove_cv_t<Rhs>, Output>;
            if constexpr (ARE_SAME_TYPE) {
                const bool are_equal = static_cast<const void*>(lhs) == static_cast<const void*>(output) &&
                                       static_cast<const void*>(mhs) == static_cast<const void*>(output) &&
                                       static_cast<const void*>(rhs) == static_cast<const void*>(output);
                if (!are_equal) {
                    if (serial) {
                        details::ewise_trinary_1d_restrict_serial(
                                lhs, mhs, rhs, output, elements, std::forward<Operator>(op));
                    } else {
                        details::ewise_trinary_1d_restrict_parallel(
                                lhs, mhs, rhs, output, elements, std::forward<Operator>(op), threads_omp);
                    }
                    return;
                }
            }
            if (serial) {
                details::ewise_trinary_1d_serial(
                        lhs, mhs, rhs, output, elements, std::forward<Operator>(op));
            } else {
                details::ewise_trinary_1d_parallel(
                        lhs, mhs, rhs, output, elements, std::forward<Operator>(op), threads_omp);
            }
        } else {
            const auto lhs_accessor = Accessor<Lhs, 4, Index>(lhs, lhs_strides);
            const auto mhs_accessor = Accessor<Mhs, 4, Index>(mhs, mhs_strides);
            const auto rhs_accessor = Accessor<Rhs, 4, Index>(rhs, rhs_strides);
            const auto output_accessor = Accessor<Output, 4, Index>(output, output_strides);
            if (threads_omp <= 1) {
                details::ewise_trinary_4d_serial(
                        lhs_accessor, mhs_accessor, rhs_accessor, output_accessor,
                        shape, std::forward<Operator>(op));
            } else {
                details::ewise_trinary_4d_parallel(
                        lhs_accessor, mhs_accessor, rhs_accessor, output_accessor,
                        shape, std::forward<Operator>(op), threads_omp);
            }
        }
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Output,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<Output> &&
                                         !std::is_pointer_v<Rhs>>>
    constexpr void ewise_trinary(
            Lhs* lhs, const Strides4<Index>& lhs_strides,
            Mhs* mhs, const Strides4<Index>& mhs_strides,
            Rhs rhs,
            Output* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewise_binary(lhs, lhs_strides, mhs, mhs_strides, output, output_strides, shape,
                     [=, op_ = std::forward<Operator>(op)](auto& lhs_value, auto& mhs_value) {
                         return op_(lhs_value, mhs_value, rhs);
                     },
                     threads);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Output,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<Output> &&
                                         !std::is_pointer_v<Rhs>>>
    constexpr void ewise_trinary(
            Lhs* lhs, const Strides4<Index>& lhs_strides,
            Mhs mhs,
            Rhs* rhs, const Strides4<Index>& rhs_strides,
            Output* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewise_binary(lhs, lhs_strides, rhs, rhs_strides, output, output_strides, shape,
                     [=, op_ = std::forward<Operator>(op)](auto& lhs_value, auto& rhs_value) {
                         return op_(lhs_value, mhs, rhs_value);
                     },
                     threads);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Output,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<Output> &&
                                         !std::is_pointer_v<Rhs>>>
    constexpr void ewise_trinary(
            Lhs lhs,
            Mhs* mhs, const Strides4<Index>& mhs_strides,
            Rhs* rhs, const Strides4<Index>& rhs_strides,
            Output* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewise_binary(mhs, mhs_strides, rhs, rhs_strides, output, output_strides, shape,
                     [=, op_ = std::forward<Operator>(op)](auto& mhs_value, auto& rhs_value) {
                         return op_(lhs, mhs_value, rhs_value);
                     },
                     threads);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Output,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<Output> &&
                                         !std::is_pointer_v<Lhs>>>
    constexpr void ewise_trinary(
            Lhs* lhs, const Strides4<Index>& lhs_strides,
            Mhs mhs,
            Rhs rhs,
            Output* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewise_unary(lhs, lhs_strides, output, output_strides, shape,
                    [=, op_ = std::forward<Operator>(op)](auto& lhs_value) { return op_(lhs_value, mhs, rhs); },
                    threads);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Output,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<Output> &&
                                         !std::is_pointer_v<Lhs>>>
    constexpr void ewise_trinary(
            Lhs lhs,
            Mhs* mhs, const Strides4<Index>& mhs_strides,
            Rhs rhs,
            Output* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewise_unary(mhs, mhs_strides, output, output_strides, shape,
                    [=, op_ = std::forward<Operator>(op)](auto& mhs_value) { return op_(lhs, mhs_value, rhs); },
                    threads);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Output,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<Output> &&
                                         !std::is_pointer_v<Lhs>>>
    constexpr void ewise_trinary(
            Lhs lhs,
            Mhs mhs,
            Rhs* rhs, const Strides4<Index>& rhs_strides,
            Output* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewise_unary(rhs, rhs_strides, output, output_strides, shape,
                    [=, op_ = std::forward<Operator>(op)](auto& rhs_value) { return op_(lhs, mhs, rhs_value); },
                    threads);
    }
}
