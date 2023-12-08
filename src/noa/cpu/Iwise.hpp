#pragma once

#include "noa/core/Types.hpp"

#if defined(NOA_IS_OFFLINE)
#include <omp.h>

namespace noa::cpu {
    /// Dispatches an index-wise operator across N-dimensional (parallel) for-loops.
    /// \note This is done in the rightmost order, i.e. the innermost loop is assumed to be the rightmost dimension.
    ///       In the parallel case, the order is not specified (although the rightmost order is still respected
    ///       within a thread), and the operator should ensure there's no data race. Furthermore, still in the
    ///       multithreaded case, the operator is copied to every thread, which can add a big performance hit
    ///       if the operator is expensive to copy.
    class Iwise {
    public:
        /// Default threshold to trigger the multi-threaded implementation.
        /// This is a large number of elements to make sure there's enough work for each thread.
        static constexpr i64 PARALLEL_ELEMENTS_THRESHOLD = 1'048'576; // 1024x1024

        /// Launches the loop.
        /// \param start    Starting indices, usually 0.
        /// \param end      Ending indices, usually the shape
        /// \param op       Operator. In the multi-threaded case, it is copied to every thread.
        ///                 (optional) op.init(thread_index) is called by each thread before the loop.
        ///                 op(vec_type) or op(index_type...) is then called within the loop.
        ///                 (optional) op.final(thread_index) is called by each thread after the loop.
        /// \param threads  Maximum number of threads. Note that passing a literal 1 (the default value)
        ///                 should reduce the amount of generated code because the parallel version can
        ///                 be omitted. In this case, it is even better to set the THRESHOLD to 1,
        ///                 turning off the multi-threaded implementation entirely.
        ///
        /// \note GCC and Clang are able to see and optimize through this. The operators are correctly inlined and the
        ///       1d cases can be strongly optimized using SIMD or memset/memcopy/memmove calls. Parallelization can
        ///       turn some of these optimizations off, as well as non-contiguous arrays.
        template<i64 THRESHOLD = PARALLEL_ELEMENTS_THRESHOLD,
                 size_t N, typename Index, typename Operator>
        static constexpr void launch(
                const Vec<Index, N>& start,
                const Vec<Index, N>& end,
                Operator&& op,
                i64 threads = 1
        ) {
            if constexpr (THRESHOLD > 1) {
                const i64 elements = Shape<i64, N>::from_vec(end.template as<i64>() - start.template as<i64>()).elements();
                const i64 actual_threads = elements <= THRESHOLD ? 1 : threads;
                if (actual_threads <= 1)
                    return iwise_parallel_(start, end, std::forward<Operator>(op), actual_threads);
            }
            iwise_serial_(start, end, std::forward<Operator>(op));
        }

        template<i64 THRESHOLD = PARALLEL_ELEMENTS_THRESHOLD, size_t N, typename Index, typename Operator>
        static constexpr void launch(const Shape<Index, N>& shape, Operator&& op, i64 threads = 1) {
            launch<THRESHOLD>(Vec<Index, N>{}, shape.vec, std::forward<Operator>(op), threads);
        }

    private:
        static constexpr void call_(auto& op, auto... indices) {
            if constexpr (requires { op(Vec{indices...}); })
                op(Vec{indices...});
            else if constexpr (requires { op(indices...); })
                op(indices...);
            else
                static_assert(nt::always_false_v<decltype(op)>);
        };

        template<size_t N, typename Index, typename Operator>
        static void iwise_parallel_(const Vec<Index, N>& start, const Vec<Index, N>& end, Operator&& op, i64 threads) {
            // firstprivate(op) vs shared(op):
            //  - We assume op is cheap to copy, so the once-per-thread call to the copy constructor with
            //    firstprivate(op) is assumed to be non-significant compared to the rest of the function.
            //  - If op() is read-only, then shared(op) and firstprivate(op) should be exactly the same performance
            //    wise (other than the initial copy constructor with firstprivate). If op() modifies its state
            //    (which admittedly is quite rare), then shared(op) creates a false-share and invalidates the cache,
            //    resulting in worse performance. With firstprivate(op) it doesn't happen because each thread has its
            //    own copy of op.
            //  - firstprivate(op) allows more complex operations, because op can have a per-thread state, which is
            //    correctly copy-initialized from the original op. This was originally developed for random number
            //    generation, where the op() call was writing to op member variable and needed to be initialized for
            //    every thread.
            #pragma omp parallel default(none) num_threads(threads) shared(start, end) firstprivate(op)
            {
                if constexpr (requires { op.init(omp_get_thread_num()); })
                    op.init(omp_get_thread_num());

                if constexpr (N == 4) {
                    #pragma omp for collapse(4)
                    for (Index i = start[0]; i < end[0]; ++i)
                        for (Index j = start[1]; j < end[1]; ++j)
                            for (Index k = start[2]; k < end[2]; ++k)
                                for (Index l = start[3]; l < end[3]; ++l)
                                    call_(op, i, j, k, l);

                } else if constexpr (N == 3) {
                    #pragma omp for collapse(3)
                    for (Index i = start[0]; i < end[0]; ++i)
                        for (Index j = start[1]; j < end[1]; ++j)
                            for (Index k = start[2]; k < end[2]; ++k)
                                call_(op, i, j, k);

                } else if constexpr (N == 2) {
                    #pragma omp for collapse(2)
                    for (Index i = start[0]; i < end[0]; ++i)
                        for (Index j = start[1]; j < end[1]; ++j)
                            call_(op, i, j);

                } else if constexpr (N == 1) {
                    #pragma omp for collapse(1)
                    for (Index i = start[0]; i < end[0]; ++i)
                        call_(op, i);
                }

                if constexpr (requires { op.final(omp_get_thread_num()); })
                    op.final(omp_get_thread_num());
            }
        }

        template<size_t N, typename Index, typename Operator>
        static constexpr void iwise_serial_(const Vec<Index, N>& start, const Vec<Index, N>& end, Operator&& op) {
            if constexpr (requires { op.init(0); })
                op.init(0);

            if constexpr (N == 4) {
                for (Index i = start[0]; i < end[0]; ++i)
                    for (Index j = start[1]; j < end[1]; ++j)
                        for (Index k = start[2]; k < end[2]; ++k)
                            for (Index l = start[3]; l < end[3]; ++l)
                                call_(op, i, j, k, l);

            } else if constexpr (N == 3) {
                for (Index i = start[0]; i < end[0]; ++i)
                    for (Index j = start[1]; j < end[1]; ++j)
                        for (Index k = start[2]; k < end[2]; ++k)
                            call_(op, i, j, k);

            } else if constexpr (N == 2) {
                for (Index i = start[0]; i < end[0]; ++i)
                    for (Index j = start[1]; j < end[1]; ++j)
                        call_(op, i, j);

            } else if constexpr (N == 1) {
                for (Index i = start[0]; i < end[0]; ++i)
                    call_(op, i);
            }

            if constexpr (requires { op.final(0); })
                op.final(0);
        }
    };
}
#endif
