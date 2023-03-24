#pragma once

#include <omp.h>
#include "noa/core/Types.hpp"

namespace noa::cpu::utils::details {
    constexpr i64 IWISE_PARALLEL_THRESHOLD = 1'048'576; // 1024x1024

    template<typename Index, typename Operator>
    void iwise_4d_parallel(const Vec4<Index>& start, const Vec4<Index>& end, Operator&& op, i64 threads) {
        #pragma omp parallel default(none) num_threads(threads) shared(start, end) firstprivate(op)
        {
            if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
                op.initialize(omp_get_thread_num());

            #pragma omp for collapse(4)
            for (Index i = start[0]; i < end[0]; ++i)
                for (Index j = start[1]; j < end[1]; ++j)
                    for (Index k = start[2]; k < end[2]; ++k)
                        for (Index l = start[3]; l < end[3]; ++l)
                            op(i, j, k, l);

            if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
                op.closure(omp_get_thread_num());
        }
    }

    template<typename Index, typename Operator>
    void iwise_4d_serial(const Vec4<Index>& start, const Vec4<Index>& end, Operator&& op) {
        if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
            op.initialize(0);
        for (Index i = start[0]; i < end[0]; ++i)
            for (Index j = start[1]; j < end[1]; ++j)
                for (Index k = start[2]; k < end[2]; ++k)
                    for (Index l = start[3]; l < end[3]; ++l)
                        op(i, j, k, l);
        if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
            op.closure(0);
    }

    template<typename Index, typename Operator>
    void iwise_4d_parallel(const Shape4<Index>& shape,Operator&& op, i64 threads) {
        iwise_4d_parallel(Vec4<Index>{0}, shape.vec(), std::forward<Operator>(op), threads);
    }

    template<typename Index, typename Operator>
    void iwise_4d_serial(const Shape4<Index>& shape,Operator&& op) {
        iwise_4d_serial(Vec4<Index>{0}, shape.vec(), std::forward<Operator>(op));
    }

    template<typename Index, typename Operator>
    void iwise_3d_parallel(const Vec3<Index>& start, const Vec3<Index>& end, Operator&& op, i64 threads) {
        #pragma omp parallel default(none) num_threads(threads) shared(start, end) firstprivate(op)
        {
            if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
                op.initialize(omp_get_thread_num());

            #pragma omp for collapse(3)
            for (Index i = start[0]; i < end[0]; ++i)
                for (Index j = start[1]; j < end[1]; ++j)
                    for (Index k = start[2]; k < end[2]; ++k)
                        op(i, j, k);

            if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
                op.closure(omp_get_thread_num());
        }
    }

    template<typename Index, typename Operator>
    void iwise_3d_serial(const Vec3<Index>& start, const Vec3<Index>& end, Operator&& op) {
        if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
            op.initialize(0);
        for (Index i = start[0]; i < end[0]; ++i)
            for (Index j = start[1]; j < end[1]; ++j)
                for (Index k = start[2]; k < end[2]; ++k)
                    op(i, j, k);
        if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
            op.closure(0);
    }

    template<typename Index, typename Operator>
    void iwise_3d_parallel(const Shape3<Index>& shape,Operator&& op, i64 threads) {
        iwise_3d_parallel(Vec3<Index>{0}, shape.vec(), std::forward<Operator>(op), threads);
    }

    template<typename Index, typename Operator>
    void iwise_3d_serial(const Shape3<Index>& shape,Operator&& op) {
        iwise_3d_serial(Vec3<Index>{0}, shape.vec(), std::forward<Operator>(op));
    }

    template<typename Index, typename Operator>
    void iwise_2d_parallel(const Vec2<Index>& start, const Vec2<Index>& end, Operator&& op, i64 threads) {
        #pragma omp parallel default(none) num_threads(threads) shared(start, end) firstprivate(op)
        {
            if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
                op.initialize(omp_get_thread_num());

            #pragma omp for collapse(2)
            for (Index i = start[0]; i < end[0]; ++i)
                for (Index j = start[1]; j < end[1]; ++j)
                    op(i, j);

            if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
                op.closure(omp_get_thread_num());
        }
    }

    template<typename Index, typename Operator>
    void iwise_2d_serial(const Vec2<Index>& start, const Vec2<Index>& end, Operator&& op) {
        if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
            op.initialize(0);
        for (Index i = start[0]; i < end[0]; ++i)
            for (Index j = start[1]; j < end[1]; ++j)
                op(i, j);
        if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
            op.closure(0);
    }

    template<typename Index, typename Operator>
    void iwise_2d_parallel(const Shape2<Index>& shape,Operator&& op, i64 threads) {
        iwise_2d_parallel(Vec2<Index>{0}, shape.vec(), std::forward<Operator>(op), threads);
    }

    template<typename Index, typename Operator>
    void iwise_2d_serial(const Shape2<Index>& shape,Operator&& op) {
        iwise_2d_serial(Vec2<Index>{0}, shape.vec(), std::forward<Operator>(op));
    }

    template<typename Index, typename Operator>
    void iwise_1d_parallel(const Vec1<Index>& start, const Vec1<Index>& end, Operator&& op, i64 threads) {
        #pragma omp parallel default(none) num_threads(threads) shared(start, end) firstprivate(op)
        {
            if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
                op.initialize(omp_get_thread_num());

            #pragma omp for
            for (Index i = start[0]; i < end[0]; ++i)
                op(i);

            if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
                op.closure(omp_get_thread_num());
        }
    }

    template<typename Index, typename Operator>
    void iwise_1d_serial(const Vec1<Index>& start, const Vec1<Index>& end, Operator&& op) {
        if constexpr (noa::traits::is_detected_v<noa::traits::has_initialize, Operator>)
            op.initialize(0);
        for (Index i = start[0]; i < end[0]; ++i)
            op(i);
        if constexpr (noa::traits::is_detected_v<noa::traits::has_closure, Operator>)
            op.closure(0);
    }

    template<typename Index, typename Operator>
    void iwise_1d_parallel(const Shape1<Index>& shape,Operator&& op, i64 threads) {
        iwise_1d_parallel(Vec1<Index>{0}, shape.vec(), std::forward<Operator>(op), threads);
    }

    template<typename Index, typename Operator>
    void iwise_1d_serial(const Shape1<Index>& shape,Operator&& op) {
        iwise_1d_serial(Vec1<Index>{0}, shape.vec(), std::forward<Operator>(op));
    }
}

namespace noa::cpu::utils {
    // Dispatches the operator across 4 dimensions, in the rightmost order (innermost loop is the rightmost dimension).
    // The operator will be called with every (i,j,k,l) combination. If multiple threads are passed, the order of these
    // combinations is undefined (although the rightmost order is still respected within a thread), and the operator
    // should ensure there's no data race. Furthermore, in the multithreading case, the operator is copied to every
    // thread, which can add a big performance cost if the operator has expensive copies.

    template<typename Index, typename Operator>
    void iwise_4d(const Vec4<Index>& start, const Vec4<Index>& end, Operator&& op, i64 threads = 1) {
        const i64 elements = Shape4<i64>(end.template as<i64>() - start.template as<i64>()).elements();
        const i64 actual_threads = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : threads;
        if (actual_threads <= 1)
            details::iwise_4d_serial(start, end, std::forward<Operator>(op));
        else
            details::iwise_4d_parallel(start, end, std::forward<Operator>(op), actual_threads);
    }

    template<typename Index, typename Operator>
    void iwise_4d(const Shape4<Index>& shape, Operator&& op, i64 threads = 1) {
        const i64 elements = shape.template as<i64>().elements();
        const i64 actual_threads = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : threads;
        if (actual_threads <= 1)
            details::iwise_4d_serial(shape, std::forward<Operator>(op));
        else
            details::iwise_4d_parallel(shape, std::forward<Operator>(op), actual_threads);
    }

    template<typename Index, typename Operator>
    void iwise_3d(const Vec3<Index>& start, const Vec3<Index>& end, Operator&& op, i64 threads = 1) {
        const i64 elements = Shape3<i64>(end.template as<i64>() - start.template as<i64>()).elements();
        const i64 actual_threads = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : threads;
        if (actual_threads <= 1)
            details::iwise_3d_serial(start, end, std::forward<Operator>(op));
        else
            details::iwise_3d_parallel(start, end, std::forward<Operator>(op), actual_threads);
    }

    template<typename Index, typename Operator>
    void iwise_3d(const Shape3<Index>& shape, Operator&& op, i64 threads = 1) {
        const i64 elements = shape.template as<i64>().elements();
        const i64 actual_threads = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : threads;
        if (actual_threads <= 1)
            details::iwise_3d_serial(shape, std::forward<Operator>(op));
        else
            details::iwise_3d_parallel(shape, std::forward<Operator>(op), actual_threads);
    }

    template<typename Index, typename Operator>
    void iwise_2d(const Vec2<Index>& start, const Vec2<Index>& end, Operator&& op, i64 threads = 1) {
        const i64 elements = Shape2<i64>(end.template as<i64>() - start.template as<i64>()).elements();
        const i64 actual_threads = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : threads;
        if (actual_threads <= 1)
            details::iwise_2d_serial(start, end, std::forward<Operator>(op));
        else
            details::iwise_2d_parallel(start, end, std::forward<Operator>(op), actual_threads);
    }

    template<typename Index, typename Operator>
    void iwise_2d(const Shape2<Index>& shape, Operator&& op, i64 threads = 1) {
        const i64 elements = shape.template as<i64>().elements();
        const i64 actual_threads = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : threads;
        if (actual_threads <= 1)
            details::iwise_2d_serial(shape, std::forward<Operator>(op));
        else
            details::iwise_2d_parallel(shape, std::forward<Operator>(op), actual_threads);
    }

    template<typename Index, typename Operator>
    void iwise_1d(const Vec1<Index>& start, const Vec1<Index>& end, Operator&& op, i64 threads = 1) {
        const i64 elements = Shape1<i64>(end.template as<i64>() - start.template as<i64>()).elements();
        const i64 actual_threads = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : threads;
        if (actual_threads <= 1)
            details::iwise_1d_serial(start, end, std::forward<Operator>(op));
        else
            details::iwise_1d_parallel(start, end, std::forward<Operator>(op), actual_threads);
    }

    template<typename Index, typename Operator>
    void iwise_1d(const Shape1<Index>& shape, Operator&& op, i64 threads = 1) {
        const i64 elements = shape.template as<i64>().elements();
        const i64 actual_threads = elements <= details::IWISE_PARALLEL_THRESHOLD ? 1 : threads;
        if (actual_threads <= 1)
            details::iwise_1d_serial(shape, std::forward<Operator>(op));
        else
            details::iwise_1d_parallel(shape, std::forward<Operator>(op), actual_threads);
    }
}
