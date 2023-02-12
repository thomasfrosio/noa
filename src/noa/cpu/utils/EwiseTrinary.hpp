#pragma once

#include "noa/core/Types.hpp"
#include "noa/cpu/utils/EwiseUnary.hpp"
#include "noa/cpu/utils/EwiseBinary.hpp"

namespace noa::cpu::utils::details {
    // Parallelization is expensive. Turn it on only for large arrays.
    constexpr i64 EWISE_TRINARY_PARALLEL_THRESHOLD = 16'777'216; // 4096x4096

    template<bool PARALLEL, typename LhsValue, typename RhsValue,
             typename OutputValue, typename Index, typename Operator>
    void ewise_trinary_4d(
            Accessor<LhsValue, 4, Index> lhs,
            Accessor<RhsValue, 4, Index> mhs,
            Accessor<RhsValue, 4, Index> rhs,
            Accessor<OutputValue, 4, Index> output,
            Shape4<Index> shape, Operator&& op, i64 threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(4) num_threads(threads) \
                    shared(lhs, mhs, rhs, output, shape, op)
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
                    for (Index k = 0; k < shape[2]; ++k)
                        for (Index l = 0; l < shape[3]; ++l)
                            output(i, j, k, l) =
                                    static_cast<OutputValue>(
                                            op(lhs(i, j, k, l),
                                               mhs(i, j, k, l),
                                               rhs(i, j, k, l)));
        } else {
            (void) threads;
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
                    for (Index k = 0; k < shape[2]; ++k)
                        for (Index l = 0; l < shape[3]; ++l)
                            output(i, j, k, l) =
                                    static_cast<OutputValue>(
                                            op(lhs(i, j, k, l),
                                               mhs(i, j, k, l),
                                               rhs(i, j, k, l)));
        }
    }

    template<bool PARALLEL, typename LhsValue, typename MhsValue, typename RhsValue,
             typename OutputValue, typename Index, typename Operator>
    void ewise_trinary_1d(
            LhsValue* lhs,
            MhsValue* mhs,
            RhsValue* rhs,
            OutputValue* output,
            Index size, Operator&& op, i64 threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) num_threads(threads) \
                    shared(lhs, mhs, rhs, output, size, op)
            for (Index i = 0; i < size; ++i)
                output[i] = static_cast<OutputValue>(op(lhs[i], mhs[i], rhs[i]));
        } else {
            (void) threads;
            for (Index i = 0; i < size; ++i)
                output[i] = static_cast<OutputValue>(op(lhs[i], mhs[i], rhs[i]));
        }
    }

    template<bool PARALLEL, typename LhsValue, typename MhsValue, typename RhsValue,
             typename OutputValue, typename Index, typename Operator>
    void ewise_trinary_1d_restrict(
            LhsValue* __restrict lhs,
            MhsValue* __restrict mhs,
            RhsValue* __restrict rhs,
            OutputValue* __restrict output,
            Index size, Operator&& op, i64 threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) num_threads(threads) \
                    shared(lhs, mhs, rhs, output, size, op)
            for (Index i = 0; i < size; ++i)
                output[i] = static_cast<OutputValue>(op(lhs[i], mhs[i], rhs[i]));
        } else {
            (void) threads;
            for (Index i = 0; i < size; ++i)
                output[i] = static_cast<OutputValue>(op(lhs[i], mhs[i], rhs[i]));
        }
    }
}

namespace noa::cpu::utils {
    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> && !std::is_const_v<OutputValue>>>
    constexpr void ewise_trinary(
            LhsValue* lhs, Strides4<Index> lhs_strides,
            MhsValue* mhs, Strides4<Index> mhs_strides,
            RhsValue* rhs, Strides4<Index> rhs_strides,
            OutputValue* output, Strides4<Index> output_strides,
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
                    std::remove_cv_t<LhsValue>, std::remove_cv_t<MhsValue>,
                    std::remove_cv_t<RhsValue>, OutputValue>;
            if constexpr (ARE_SAME_TYPE) {
                const bool are_equal = static_cast<const void*>(lhs) == static_cast<const void*>(output) &&
                                       static_cast<const void*>(mhs) == static_cast<const void*>(output) &&
                                       static_cast<const void*>(rhs) == static_cast<const void*>(output);
                if (!are_equal) {
                    if (serial) {
                        details::ewise_trinary_1d_restrict<false>(
                                lhs, mhs, rhs, output, elements, std::forward<Operator>(op), 1);
                    } else {
                        details::ewise_trinary_1d_restrict<true>(
                                lhs, mhs, rhs, output, elements, std::forward<Operator>(op), threads_omp);
                    }
                    return;
                }
            }
            if (serial) {
                details::ewise_trinary_1d<false>(
                        lhs, mhs, rhs, output, elements, std::forward<Operator>(op), 1);
            } else {
                details::ewise_trinary_1d<true>(
                        lhs, mhs, rhs, output, elements, std::forward<Operator>(op), threads_omp);
            }
        } else {
            const auto lhs_accessor = Accessor<LhsValue, 4, Index>(lhs, lhs_strides);
            const auto mhs_accessor = Accessor<LhsValue, 4, Index>(mhs, mhs_strides);
            const auto rhs_accessor = Accessor<LhsValue, 4, Index>(rhs, rhs_strides);
            const auto output_accessor = Accessor<OutputValue, 4, Index>(output, output_strides);
            if (threads_omp <= 1) {
                details::ewise_trinary_4d<false>(
                        lhs_accessor, mhs_accessor, rhs_accessor, output_accessor,
                        shape, std::forward<Operator>(op), 1);
            } else {
                details::ewise_trinary_4d<true>(
                        lhs_accessor, mhs_accessor, rhs_accessor, output_accessor,
                        shape, std::forward<Operator>(op), threads_omp);
            }
        }
    }

    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<OutputValue> &&
                                         !std::is_pointer_v<RhsValue>>>
    constexpr void ewise_trinary(
            LhsValue* lhs, const Strides4<Index>& lhs_strides,
            MhsValue* mhs, const Strides4<Index>& mhs_strides,
            RhsValue rhs,
            OutputValue* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewise_binary(lhs, lhs_strides, mhs, mhs_strides, output, output_strides, shape,
                     [=, op_ = std::forward<Operator>(op)](auto& lhs_value, auto& mhs_value) {
                         return op_(lhs_value, mhs_value, rhs);
                     },
                     threads);
    }

    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<OutputValue> &&
                                         !std::is_pointer_v<RhsValue>>>
    constexpr void ewise_trinary(
            LhsValue* lhs, const Strides4<Index>& lhs_strides,
            MhsValue mhs,
            RhsValue* rhs, const Strides4<Index>& rhs_strides,
            OutputValue* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewise_binary(lhs, lhs_strides, rhs, rhs_strides, output, output_strides, shape,
                     [=, op_ = std::forward<Operator>(op)](auto& lhs_value, auto& rhs_value) {
                         return op_(lhs_value, mhs, rhs_value);
                     },
                     threads);
    }

    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<OutputValue> &&
                                         !std::is_pointer_v<RhsValue>>>
    constexpr void ewise_trinary(
            LhsValue lhs,
            MhsValue* mhs, const Strides4<Index>& mhs_strides,
            RhsValue* rhs, const Strides4<Index>& rhs_strides,
            OutputValue* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewise_binary(mhs, mhs_strides, rhs, rhs_strides, output, output_strides, shape,
                     [=, op_ = std::forward<Operator>(op)](auto& mhs_value, auto& rhs_value) {
                         return op_(lhs, mhs_value, rhs_value);
                     },
                     threads);
    }

    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<OutputValue> &&
                                         !std::is_pointer_v<LhsValue>>>
    constexpr void ewise_trinary(
            LhsValue* lhs, const Strides4<Index>& lhs_strides,
            MhsValue mhs,
            RhsValue rhs,
            OutputValue* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewise_unary(lhs, lhs_strides, output, output_strides, shape,
                    [=, op_ = std::forward<Operator>(op)](auto& lhs_value) { return op_(lhs_value, mhs, rhs); },
                    threads);
    }

    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<OutputValue> &&
                                         !std::is_pointer_v<LhsValue>>>
    constexpr void ewise_trinary(
            LhsValue lhs,
            MhsValue* mhs, const Strides4<Index>& mhs_strides,
            RhsValue rhs,
            OutputValue* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewise_unary(mhs, mhs_strides, output, output_strides, shape,
                    [=, op_ = std::forward<Operator>(op)](auto& mhs_value) { return op_(lhs, mhs_value, rhs); },
                    threads);
    }

    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<OutputValue> &&
                                         !std::is_pointer_v<LhsValue>>>
    constexpr void ewise_trinary(
            LhsValue lhs,
            MhsValue mhs,
            RhsValue* rhs, const Strides4<Index>& rhs_strides,
            OutputValue* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewise_unary(rhs, rhs_strides, output, output_strides, shape,
                    [=, op_ = std::forward<Operator>(op)](auto& rhs_value) { return op_(lhs, mhs, rhs_value); },
                    threads);
    }
}
