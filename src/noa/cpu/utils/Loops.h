#pragma once

#include "noa/common/Types.h"

// Very simple (and naive) set of index-wise nested loops.
// The performance should be identical to using these loops in-place without operators.
// Speaking of performance, it is apparently crucial to keep the serial version in a separate compile-time branch,
// otherwise some compilers (e.g. g++) seem to merge the OpenMP and serial loops, which is bad for performance
// because the OpenMP version collapses the loops so cannot do constant evaluated expressions to outer-loops.
// For the OpenMP version, the nested loops are collapsed. Unfortunately, the collapse parameter needs to be a
// constant-evaluated value (cannot be a template variable).
// OpenMP has a if() option, but this is at runtime, so at this point, the loops will be collapsed and the
// non-collapsed version will be faster in a single-thread scenario. As such, these functions below move this
// if() option at compile time by instantiating both collapsed and uncollapsed versions. OpenMP is getting a lot
// of improvements in the new compilers, so may be worth revisiting this at some point.

namespace noa::cpu::utils::details {
    constexpr int64_t IWISE_PARALLEL_THRESHOLD = 1'048'576; // 1024x1024

    template<bool PARALLEL, typename dimension_t, typename operator_t>
    void iwise4D(const Int4<dimension_t>& start,
                 const Int4<dimension_t>& end,
                 operator_t&& op,
                 int64_t threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(4) num_threads(threads) shared(start, end, op)
            for (dimension_t i = start[0]; i < end[0]; ++i)
                for (dimension_t j = start[1]; j < end[1]; ++j)
                    for (dimension_t k = start[2]; k < end[2]; ++k)
                        for (dimension_t l = start[3]; l < end[3]; ++l)
                            op(i, j, k, l);
        } else {
            (void) threads;
            for (dimension_t i = start[0]; i < end[0]; ++i)
                for (dimension_t j = start[1]; j < end[1]; ++j)
                    for (dimension_t k = start[2]; k < end[2]; ++k)
                        for (dimension_t l = start[3]; l < end[3]; ++l)
                            op(i, j, k, l);
        }
    }

    template<bool PARALLEL, typename dimension_t, typename operator_t>
    void iwise4D(const Int4<dimension_t>& shape,
                 operator_t&& op,
                 int64_t threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(4) num_threads(threads) shared(shape, op)
            for (dimension_t i = 0; i < shape[0]; ++i)
                for (dimension_t j = 0; j < shape[1]; ++j)
                    for (dimension_t k = 0; k < shape[2]; ++k)
                        for (dimension_t l = 0; l < shape[3]; ++l)
                            op(i, j, k, l);
        } else {
            (void) threads;
            for (dimension_t i = 0; i < shape[0]; ++i)
                for (dimension_t j = 0; j < shape[1]; ++j)
                    for (dimension_t k = 0; k < shape[2]; ++k)
                        for (dimension_t l = 0; l < shape[3]; ++l)
                            op(i, j, k, l);
        }
    }

    template<bool PARALLEL, typename dimension_t, typename operator_t>
    void iwise3D(const Int3<dimension_t>& start,
                 const Int3<dimension_t>& end,
                 operator_t&& op,
                 int64_t threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(3) num_threads(threads) shared(start, end, op)
            for (dimension_t i = start[0]; i < end[0]; ++i)
                for (dimension_t j = start[1]; j < end[1]; ++j)
                    for (dimension_t k = start[2]; k < end[2]; ++k)
                        op(i, j, k);
        } else {
            (void) threads;
            for (dimension_t i = start[0]; i < end[0]; ++i)
                for (dimension_t j = start[1]; j < end[1]; ++j)
                    for (dimension_t k = start[2]; k < end[2]; ++k)
                        op(i, j, k);
        }
    }

    template<bool PARALLEL, typename dimension_t, typename operator_t>
    void iwise3D(const Int3<dimension_t>& shape,
                 operator_t&& op,
                 int64_t threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(3) num_threads(threads) shared(shape, op)
            for (dimension_t i = 0; i < shape[0]; ++i)
                for (dimension_t j = 0; j < shape[1]; ++j)
                    for (dimension_t k = 0; k < shape[2]; ++k)
                        op(i, j, k);
        } else {
            (void) threads;
            for (dimension_t i = 0; i < shape[0]; ++i)
                for (dimension_t j = 0; j < shape[1]; ++j)
                    for (dimension_t k = 0; k < shape[2]; ++k)
                        op(i, j, k);
        }
    }

    template<bool PARALLEL, typename dimension_t, typename operator_t>
    void iwise2D(const Int2<dimension_t>& start,
                 const Int2<dimension_t>& end,
                 operator_t&& op,
                 int64_t threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(2) num_threads(threads) shared(start, end, op)
            for (dimension_t i = start[0]; i < end[0]; ++i)
                for (dimension_t j = start[1]; j < end[1]; ++j)
                    op(i, j);
        } else {
            (void) threads;
            for (dimension_t i = start[0]; i < end[0]; ++i)
                for (dimension_t j = start[1]; j < end[1]; ++j)
                    op(i, j);
        }
    }

    template<bool PARALLEL, typename dimension_t, typename operator_t>
    void iwise2D(const Int2<dimension_t>& shape,
                 operator_t&& op,
                 int64_t threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(2) num_threads(threads) shared(shape, op)
            for (dimension_t i = 0; i < shape[0]; ++i)
                for (dimension_t j = 0; j < shape[1]; ++j)
                    op(i, j);
        } else {
            (void) threads;
            for (dimension_t i = 0; i < shape[0]; ++i)
                for (dimension_t j = 0; j < shape[1]; ++j)
                    op(i, j);
        }
    }
}

namespace noa::cpu::utils {
    // Dispatches the operator across 4 dimensions, in the rightmost order (innermost loop is the rightmost dimension).
    // The operator will be called with every (i,j,k,l) combination. If multiple threads are passed, the order of these
    // combinations is undefined (although the rightmost order is still respected within a thread), and the operator
    // should ensure there's no data race. Furthermore, in the multithreading case, the operator is copied to every
    // thread, which can add a big performance cost if the operator has expensive copies.

    template<typename dimension_t, typename operator_t, typename int_t = int64_t,
             typename = std::enable_if_t<std::is_integral_v<int_t>>>
    void iwise4D(const Int4<dimension_t>& start,
                 const Int4<dimension_t>& end,
                 operator_t&& op,
                 int_t threads = int_t{1}) {
        const int64_t elements = (long4_t(end) - long4_t(start)).elements();
        const int64_t threads_ = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : clamp_cast<int64_t>(threads);
        if (threads_ <= 1)
            details::iwise4D<false>(start, end, std::forward<operator_t>(op), threads_);
        else
            details::iwise4D<true>(start, end, std::forward<operator_t>(op), threads_);
    }

    template<typename dimension_t, typename operator_t, typename int_t = int64_t,
             typename = std::enable_if_t<std::is_integral_v<int_t>>>
    void iwise4D(const Int4<dimension_t>& shape,
                 operator_t&& op,
                 int_t threads = int_t{1}) {
        const int64_t elements = long4_t(shape).elements();
        const int64_t threads_ = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : clamp_cast<int64_t>(threads);
        if (threads_ <= 1)
            details::iwise4D<false>(shape, std::forward<operator_t>(op), threads_);
        else
            details::iwise4D<true>(shape, std::forward<operator_t>(op), threads_);
    }

    template<typename dimension_t, typename operator_t, typename int_t = int64_t,
             typename = std::enable_if_t<std::is_integral_v<int_t>>>
    void iwise3D(const Int3<dimension_t>& start,
                 const Int3<dimension_t>& end,
                 operator_t&& op,
                 int_t threads = int_t{1}) {
        const int64_t elements = (long3_t(end) - long3_t(start)).elements();
        const int64_t threads_ = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : clamp_cast<int64_t>(threads);
        if (threads_ <= 1)
            details::iwise3D<false>(start, end, std::forward<operator_t>(op), threads_);
        else
            details::iwise3D<true>(start, end, std::forward<operator_t>(op), threads_);
    }

    template<typename dimension_t, typename operator_t, typename int_t = int64_t,
             typename = std::enable_if_t<std::is_integral_v<int_t>>>
    void iwise3D(const Int3<dimension_t>& shape,
                 operator_t&& op,
                 int_t threads = int_t{1}) {
        const int64_t elements = long3_t(shape).elements();
        const int64_t threads_ = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : clamp_cast<int64_t>(threads);
        if (threads_ <= 1)
            details::iwise3D<false>(shape, std::forward<operator_t>(op), threads);
        else
            details::iwise3D<true>(shape, std::forward<operator_t>(op), threads);
    }

    template<typename dimension_t, typename operator_t, typename int_t = int64_t,
             typename = std::enable_if_t<std::is_integral_v<int_t>>>
    void iwise2D(const Int2<dimension_t>& start,
                 const Int2<dimension_t>& end,
                 operator_t&& op,
                 int_t threads = int_t{1}) {
        const int64_t elements = (long2_t(end) - long2_t(start)).elements();
        const int64_t threads_ = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : clamp_cast<int64_t>(threads);
        if (threads_ <= 1)
            details::iwise2D<false>(start, end, std::forward<operator_t>(op), threads_);
        else
            details::iwise2D<true>(start, end, std::forward<operator_t>(op), threads_);
    }

    template<typename dimension_t, typename operator_t, typename int_t = int64_t,
             typename = std::enable_if_t<std::is_integral_v<int_t>>>
    void iwise2D(const Int2<dimension_t>& shape,
                 operator_t&& op,
                 int_t threads = int_t{1}) {
        const int64_t elements = long2_t(shape).elements();
        const int64_t threads_ = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : clamp_cast<int64_t>(threads);
        if (threads_ <= 1)
            details::iwise2D<false>(shape, std::forward<operator_t>(op), threads_);
        else
            details::iwise2D<true>(shape, std::forward<operator_t>(op), threads_);
    }
}
