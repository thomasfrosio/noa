#pragma once

#include "noa/common/Types.h"
#include "noa/cpu/utils/EwiseUnary.h"
#include "noa/cpu/utils/EwiseBinary.h"

namespace noa::cpu::utils::details {
    // Parallelization is expensive. Turn it on only for large arrays.
    constexpr int64_t EWISE_TRINARY_PARALLEL_THRESHOLD = 16'777'216; // 4096x4096

    template<bool PARALLEL, typename LhsValue, typename RhsValue,
             typename OutputValue, typename Index, typename Operator>
    void ewiseTrinary4D(const Accessor<LhsValue, 4, Index>& lhs,
                        const Accessor<RhsValue, 4, Index>& mhs,
                        const Accessor<RhsValue, 4, Index>& rhs,
                        const Accessor<OutputValue, 4, Index>& output,
                        const dim4_t& shape, Operator&& op, int64_t threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(4) num_threads(threads) \
                    shared(lhs, mhs, rhs, output, shape, op)
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
                    for (Index k = 0; k < shape[2]; ++k)
                        for (Index l = 0; l < shape[3]; ++l)
                            output(i, j, k, l) = op(lhs(i, j, k, l),
                                                    mhs(i, j, k, l),
                                                    rhs(i, j, k, l));
        } else {
            (void) threads;
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
                    for (Index k = 0; k < shape[2]; ++k)
                        for (Index l = 0; l < shape[3]; ++l)
                            output(i, j, k, l) = op(lhs(i, j, k, l),
                                                    mhs(i, j, k, l),
                                                    rhs(i, j, k, l));
        }
    }

    template<bool PARALLEL, typename LhsValue, typename MhsValue, typename RhsValue,
             typename OutputValue, typename Index, typename Operator>
    void ewiseTrinary1D(LhsValue* lhs,
                        MhsValue* mhs,
                        RhsValue* rhs,
                        OutputValue* output,
                        Index size, Operator&& op, int64_t threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) num_threads(threads) \
                    shared(lhs, mhs, rhs, output, size, op)
            for (Index i = 0; i < size; ++i)
                output[i] = op(lhs[i], mhs[i], rhs[i]);
        } else {
            (void) threads;
            for (Index i = 0; i < size; ++i)
                output[i] = op(lhs[i], mhs[i], rhs[i]);
        }
    }

    template<bool PARALLEL, typename LhsValue, typename MhsValue, typename RhsValue,
             typename OutputValue, typename Index, typename Operator>
    void ewiseTrinary1DRestrict(LhsValue* __restrict lhs,
                                MhsValue* __restrict mhs,
                                RhsValue* __restrict rhs,
                                OutputValue* __restrict output,
                                Index size, Operator&& op, int64_t threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) num_threads(threads) \
                    shared(lhs, mhs, rhs, output, size, op)
            for (Index i = 0; i < size; ++i)
                output[i] = op(lhs[i], mhs[i], rhs[i]);
        } else {
            (void) threads;
            for (Index i = 0; i < size; ++i)
                output[i] = op(lhs[i], mhs[i], rhs[i]);
        }
    }
}

namespace noa::cpu::utils {
    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = int64_t,
             typename = std::enable_if_t<std::is_integral_v<Int> && !std::is_const_v<OutputValue>>>
    constexpr void ewiseUnary(LhsValue* lhs, Int4<Index> lhs_strides,
                              MhsValue* mhs, Int4<Index> mhs_strides,
                              RhsValue* rhs, Int4<Index> rhs_strides,
                              OutputValue* output, Int4<Index> output_strides,
                              Int4<Index> shape, Operator&& op, Int threads = Int{1}) {
        // Rearrange to rightmost order.
        shape = noa::indexing::effectiveShape(shape, output_strides);
        const auto order = noa::indexing::order(output_strides, shape);
        if (noa::all(order != Int4<Index>{0, 1, 2, 3})) {
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

        const int64_t threads_omp =
                elements <= details::EWISE_TRINARY_PARALLEL_THRESHOLD ?
                1 : clamp_cast<int64_t>(threads);
        const bool serial = threads_omp <= 1;

        const bool is_contiguous =
                noa::indexing::areContiguous(lhs_strides, shape) &&
                noa::indexing::areContiguous(mhs_strides, shape) &&
                noa::indexing::areContiguous(rhs_strides, shape) &&
                noa::indexing::areContiguous(output_strides, shape);
        if (is_contiguous) {
            constexpr bool ARE_SAME_TYPE = noa::traits::are_all_same_v<
                    std::remove_cv_t<LhsValue>, std::remove_cv_t<MhsValue>,
                    std::remove_cv_t<RhsValue>, OutputValue>;
            if constexpr (ARE_SAME_TYPE) {
                const bool are_equal = static_cast<void*>(lhs) == static_cast<void*>(output) &&
                                       static_cast<void*>(mhs) == static_cast<void*>(output) &&
                                       static_cast<void*>(rhs) == static_cast<void*>(output);
                if (!are_equal) {
                    if (serial) {
                        details::ewiseTrinary1DRestrict<false>(
                                lhs, mhs, rhs, output, elements, std::forward<Operator>(op), 1);
                    } else {
                        details::ewiseTrinary1DRestrict<true>(
                                lhs, mhs, rhs, output, elements, std::forward<Operator>(op), threads_omp);
                    }
                    return;
                }
            }
            if (serial) {
                details::ewiseTrinary1D<false>(
                        lhs, mhs, rhs, output, elements, std::forward<Operator>(op), 1);
            } else {
                details::ewiseTrinary1D<true>(
                        lhs, mhs, rhs, output, elements, std::forward<Operator>(op), threads_omp);
            }
        } else {
            const auto lhs_accessor = Accessor<LhsValue, 4, Index>(lhs, lhs_strides);
            const auto mhs_accessor = Accessor<LhsValue, 4, Index>(mhs, mhs_strides);
            const auto rhs_accessor = Accessor<LhsValue, 4, Index>(rhs, rhs_strides);
            const auto output_accessor = Accessor<OutputValue, 4, Index>(output, output_strides);
            if (threads_omp <= 1) {
                details::ewiseTrinary4D<false>(
                        lhs_accessor, mhs_accessor, rhs_accessor, output_accessor,
                        shape, std::forward<Operator>(op), 1);
            } else {
                details::ewiseTrinary4D<true>(
                        lhs_accessor, mhs_accessor, rhs_accessor, output_accessor,
                        shape, std::forward<Operator>(op), threads_omp);
            }
        }
    }

    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = int64_t,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<OutputValue> &&
                                         !std::is_pointer_v<RhsValue>>>
    constexpr void ewiseUnary(LhsValue* lhs, const Int4<Index>& lhs_strides,
                              MhsValue* mhs, const Int4<Index>& mhs_strides,
                              RhsValue rhs,
                              OutputValue* output, const Int4<Index>& output_strides,
                              const Int4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewiseBinary(lhs, lhs_strides, mhs, mhs_strides, output, output_strides, shape,
                    [=, op_ = std::forward<Operator>(op)](auto& lhs_value, auto& mhs_value) {
                        return op_(lhs_value, mhs_value, rhs);
                    },
                    threads);
    }

    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = int64_t,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<OutputValue> &&
                                         !std::is_pointer_v<RhsValue>>>
    constexpr void ewiseUnary(LhsValue* lhs, const Int4<Index>& lhs_strides,
                              MhsValue mhs,
                              RhsValue* rhs, const Int4<Index>& rhs_strides,
                              OutputValue* output, const Int4<Index>& output_strides,
                              const Int4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewiseBinary(lhs, lhs_strides, rhs, rhs_strides, output, output_strides, shape,
                    [=, op_ = std::forward<Operator>(op)](auto& lhs_value, auto& rhs_value) {
                        return op_(lhs_value, mhs, rhs_value);
                    },
                    threads);
    }

    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = int64_t,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<OutputValue> &&
                                         !std::is_pointer_v<RhsValue>>>
    constexpr void ewiseUnary(LhsValue lhs,
                              MhsValue* mhs, const Int4<Index>& mhs_strides,
                              RhsValue* rhs, const Int4<Index>& rhs_strides,
                              OutputValue* output, const Int4<Index>& output_strides,
                              const Int4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewiseBinary(mhs, mhs_strides, rhs, rhs_strides, output, output_strides, shape,
                    [=, op_ = std::forward<Operator>(op)](auto& mhs_value, auto& rhs_value) {
                        return op_(lhs, mhs_value, rhs_value);
                    },
                    threads);
    }

    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = int64_t,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<OutputValue> &&
                                         !std::is_pointer_v<LhsValue>>>
    constexpr void ewiseUnary(LhsValue* lhs, const Int4<Index>& lhs_strides,
                              MhsValue mhs,
                              RhsValue rhs,
                              OutputValue* output, const Int4<Index>& output_strides,
                              const Int4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewiseUnary(lhs, lhs_strides, output, output_strides, shape,
                   [=, op_ = std::forward<Operator>(op)] (auto& lhs_value) { return op_(lhs_value, mhs, rhs); },
                   threads);
    }

    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = int64_t,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<OutputValue> &&
                                         !std::is_pointer_v<LhsValue>>>
    constexpr void ewiseUnary(LhsValue lhs,
                              MhsValue* mhs, const Int4<Index>& mhs_strides,
                              RhsValue rhs,
                              OutputValue* output, const Int4<Index>& output_strides,
                              const Int4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewiseUnary(mhs, mhs_strides, output, output_strides, shape,
                   [=, op_ = std::forward<Operator>(op)] (auto& mhs_value) { return op_(lhs, mhs_value, rhs); },
                   threads);
    }

    template<typename LhsValue, typename MhsValue, typename RhsValue, typename OutputValue,
             typename Index, typename Operator, typename Int = int64_t,
             typename = std::enable_if_t<std::is_integral_v<Int> &&
                                         !std::is_const_v<OutputValue> &&
                                         !std::is_pointer_v<LhsValue>>>
    constexpr void ewiseUnary(LhsValue lhs,
                              MhsValue mhs,
                              RhsValue* rhs, const Int4<Index>& rhs_strides,
                              OutputValue* output, const Int4<Index>& output_strides,
                              const Int4<Index>& shape, Operator&& op, Int threads = Int{1}) {
        ewiseUnary(rhs, rhs_strides, output, output_strides, shape,
                   [=, op_ = std::forward<Operator>(op)] (auto& rhs_value) { return op_(lhs, mhs, rhs_value); },
                   threads);
    }
}
