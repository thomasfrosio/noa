#pragma once

#include "noa/core/Types.hpp"

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
    constexpr i64 IWISE_PARALLEL_THRESHOLD = 1'048'576; // 1024x1024

    template<bool PARALLEL, typename Index, typename Operator>
    void iwise_4d(
            const Vec4<Index>& start,
            const Vec4<Index>& end,
            Operator&& op,
            i64 threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(4) num_threads(threads) shared(start, end, op)
            for (Index i = start[0]; i < end[0]; ++i)
                for (Index j = start[1]; j < end[1]; ++j)
                    for (Index k = start[2]; k < end[2]; ++k)
                        for (Index l = start[3]; l < end[3]; ++l)
                            op(i, j, k, l);
        } else {
            (void) threads;
            for (Index i = start[0]; i < end[0]; ++i)
                for (Index j = start[1]; j < end[1]; ++j)
                    for (Index k = start[2]; k < end[2]; ++k)
                        for (Index l = start[3]; l < end[3]; ++l)
                            op(i, j, k, l);
        }
    }

    template<bool PARALLEL, typename Index, typename Operator>
    void iwise_4d(
            const Shape4<Index>& shape,
            Operator&& op,
            i64 threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(4) num_threads(threads) shared(shape, op)
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
                    for (Index k = 0; k < shape[2]; ++k)
                        for (Index l = 0; l < shape[3]; ++l)
                            op(i, j, k, l);
        } else {
            (void) threads;
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
                    for (Index k = 0; k < shape[2]; ++k)
                        for (Index l = 0; l < shape[3]; ++l)
                            op(i, j, k, l);
        }
    }

    template<bool PARALLEL, typename Index, typename Operator>
    void iwise_3d(
            const Vec3<Index>& start,
            const Vec3<Index>& end,
            Operator&& op,
            i64 threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(3) num_threads(threads) shared(start, end, op)
            for (Index i = start[0]; i < end[0]; ++i)
                for (Index j = start[1]; j < end[1]; ++j)
                    for (Index k = start[2]; k < end[2]; ++k)
                        op(i, j, k);
        } else {
            (void) threads;
            for (Index i = start[0]; i < end[0]; ++i)
                for (Index j = start[1]; j < end[1]; ++j)
                    for (Index k = start[2]; k < end[2]; ++k)
                        op(i, j, k);
        }
    }

    template<bool PARALLEL, typename Index, typename Operator>
    void iwise_3d(
            const Shape3<Index>& shape,
            Operator&& op,
            i64 threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(3) num_threads(threads) shared(shape, op)
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
                    for (Index k = 0; k < shape[2]; ++k)
                        op(i, j, k);
        } else {
            (void) threads;
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
                    for (Index k = 0; k < shape[2]; ++k)
                        op(i, j, k);
        }
    }

    template<bool PARALLEL, typename Index, typename Operator>
    void iwise_2d(
            const Vec2<Index>& start,
            const Vec2<Index>& end,
            Operator&& op,
            i64 threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(2) num_threads(threads) shared(start, end, op)
            for (Index i = start[0]; i < end[0]; ++i)
                for (Index j = start[1]; j < end[1]; ++j)
                    op(i, j);
        } else {
            (void) threads;
            for (Index i = start[0]; i < end[0]; ++i)
                for (Index j = start[1]; j < end[1]; ++j)
                    op(i, j);
        }
    }

    template<bool PARALLEL, typename Index, typename Operator>
    void iwise_2d(
            const Shape2<Index>& shape,
            Operator&& op,
            i64 threads) {
        if constexpr (PARALLEL) {
            #pragma omp parallel for default(none) collapse(2) num_threads(threads) shared(shape, op)
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
                    op(i, j);
        } else {
            (void) threads;
            for (Index i = 0; i < shape[0]; ++i)
                for (Index j = 0; j < shape[1]; ++j)
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

    template<typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int>>>
    void iwise_4d(
            const Vec4<Index>& start,
            const Vec4<Index>& end,
            Operator&& op,
            Int threads = Int{1}) {
        const i64 elements = (end.template as<i64>() - start.template as<i64>()).elements();
        const i64 threads_ = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : clamp_cast<i64>(threads);
        if (threads_ <= 1)
            details::iwise_4d<false>(start, end, std::forward<Operator>(op), threads_);
        else
            details::iwise_4d<true>(start, end, std::forward<Operator>(op), threads_);
    }

    template<typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int>>>
    void iwise_4d(
            const Shape4<Index>& shape,
            Operator&& op,
            Int threads = Int{1}) {
        const i64 elements = shape.template as<i64>().elements();
        const i64 threads_ = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : clamp_cast<i64>(threads);
        if (threads_ <= 1)
            details::iwise_4d<false>(shape, std::forward<Operator>(op), threads_);
        else
            details::iwise_4d<true>(shape, std::forward<Operator>(op), threads_);
    }

    template<typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int>>>
    void iwise_3d(
            const Vec3<Index>& start,
            const Vec3<Index>& end,
            Operator&& op,
            Int threads = Int{1}) {
        const i64 elements = (end.template as<i64>() - start.template as<i64>()).elements();
        const i64 threads_ = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : clamp_cast<i64>(threads);
        if (threads_ <= 1)
            details::iwise_3d<false>(start, end, std::forward<Operator>(op), threads_);
        else
            details::iwise_3d<true>(start, end, std::forward<Operator>(op), threads_);
    }

    template<typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int>>>
    void iwise_3d(
            const Shape3<Index>& shape,
            Operator&& op,
            Int threads = Int{1}) {
        const i64 elements = shape.template as<i64>().elements();
        const i64 threads_ = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : clamp_cast<i64>(threads);
        if (threads_ <= 1)
            details::iwise_3d<false>(shape, std::forward<Operator>(op), threads_);
        else
            details::iwise_3d<true>(shape, std::forward<Operator>(op), threads_);
    }

    template<typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int>>>
    void iwise_2d(
            const Vec2<Index>& start,
            const Vec2<Index>& end,
            Operator&& op,
            Int threads = Int{1}) {
        const i64 elements = (end.template as<i64>() - start.template as<i64>()).elements();
        const i64 threads_ = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : clamp_cast<i64>(threads);
        if (threads_ <= 1)
            details::iwise_2d<false>(start, end, std::forward<Operator>(op), threads_);
        else
            details::iwise_2d<true>(start, end, std::forward<Operator>(op), threads_);
    }

    template<typename Index, typename Operator, typename Int = i64,
             typename = std::enable_if_t<std::is_integral_v<Int>>>
    void iwise_2d(
            const Shape2<Index>& shape,
            Operator&& op,
            Int threads = Int{1}) {
        const i64 elements = shape.template as<i64>().elements();
        const i64 threads_ = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : clamp_cast<i64>(threads);
        if (threads_ <= 1)
            details::iwise_2d<false>(shape, std::forward<Operator>(op), threads_);
        else
            details::iwise_2d<true>(shape, std::forward<Operator>(op), threads_);
    }
}
