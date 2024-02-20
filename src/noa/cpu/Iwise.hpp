#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include <omp.h>
#include "noa/core/utils/Interfaces.hpp"
#include "noa/core/types/Shape.hpp"

namespace noa::cpu::guts {
    class Iwise {
    public:
        template<size_t N, typename Index, typename Operator>
        static void parallel(const Shape<Index, N>& shape, Operator& op, i64 n_threads) {
            // firstprivate(op) vs shared(op):
            //  - We assume op is cheap to copy, so the once-per-thread call to the copy constructor with
            //    firstprivate(op) is assumed to be non-significant compared to the rest of the function.
            //    Notice that the operator is taken by lvalue reference here, whereas the serial function takes
            //    it by value. This is to have the same number of copies/moves (per thread) for both the
            //    parallel and serial version.
            //  - If op() is read-only, then shared(op) and firstprivate(op) should be exactly the same performance
            //    wise (other than the initial copy constructor with firstprivate). If op() modifies its state
            //    (which admittedly is quite rare), then shared(op) creates a false-share and invalidates the cache,
            //    resulting in worse performance. With firstprivate(op) it doesn't happen because each thread has its
            //    own copy of op.
            //  - firstprivate(op) allows more complex operations, because op can have a per-thread state, which is
            //    correctly copy-initialized from the original op. This was originally developed for random number
            //    generation, where the op() call was writing to an op member variable and needed to be initialized
            //    for every thread.
            #pragma omp parallel default(none) num_threads(n_threads) shared(shape) firstprivate(op)
            {
                using interface = ng::IwiseInterface;
                interface::init(op, omp_get_thread_num());

                if constexpr (N == 4) {
                    #pragma omp for collapse(4)
                    for (Index i = 0; i < shape[0]; ++i)
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                for (Index l = 0; l < shape[3]; ++l)
                                    interface::call(op, i, j, k, l);

                } else if constexpr (N == 3) {
                    #pragma omp for collapse(3)
                    for (Index i = 0; i < shape[0]; ++i)
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                interface::call(op, i, j, k);

                } else if constexpr (N == 2) {
                    #pragma omp for collapse(2)
                    for (Index i = 0; i < shape[0]; ++i)
                        for (Index j = 0; j < shape[1]; ++j)
                            interface::call(op, i, j);

                } else if constexpr (N == 1) {
                    #pragma omp for collapse(1)
                    for (Index i = 0; i < shape[0]; ++i)
                        interface::call(op, i);
                }

                interface::final(op, omp_get_thread_num());
            }
        }

        template<size_t N, typename Index, typename Operator>
        static constexpr void serial(const Shape<Index, N>& shape, Operator op) {
            using interface = ng::IwiseInterface;
            interface::init(op, 0);

            if constexpr (N == 4) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            for (Index l = 0; l < shape[3]; ++l)
                                interface::call(op, i, j, k, l);

            } else if constexpr (N == 3) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            interface::call(op, i, j, k);

            } else if constexpr (N == 2) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        interface::call(op, i, j);

            } else if constexpr (N == 1) {
                for (Index i = 0; i < shape[0]; ++i)
                    interface::call(op, i);
            }

            interface::final(op, 0);
        }
    };
}

namespace noa::cpu {
    struct IwiseConfig {
        i64 parallel_threshold = 1'048'576; // 1024x1024
    };

    /// Dispatches an index-wise operator across N-dimensional (parallel) for-loops.
    /// \tparam PARALLEL_THRESHOLD  Numbers of elements above which the multithreaded implementation is chosen.
    /// \param start                Shape of the N-dimensional loop.
    /// \param op                   Valid index-wise operator (see core interface). The operator is forwarded to
    ///                             the loop, which takes it by value (it is either moved or copied once by the time
    ///                             it reaches the loop). In the multi-threaded case, it is copied to every thread.
    /// \param n_threads            Maximum number of threads. Note that passing a literal 1 (the default value)
    ///                             should reduce the amount of generated code because the parallel version can
    ///                             be optimized away. In this case, it is even better to set the THRESHOLD to 1,
    ///                             turning off the multi-threaded implementation entirely.
    ///
    /// \note This is done in the rightmost order, i.e. the innermost loop is assumed to be the rightmost dimension.
    ///       In the parallel case, the order is not specified (although the rightmost order is still respected
    ///       within a thread), and the operator should ensure there's no data race. Furthermore, still in the
    ///       multithreaded case, the operator is copied to every thread, which can add a big performance hit
    ///       if the operator is expensive to copy.
    ///
    /// \note GCC and Clang are able to see and optimize through this. The operators are correctly inlined and the
    ///       1d cases can be strongly optimized using SIMD or memset/memcopy/memmove calls. Parallelization,
    ///       as well as non-contiguous arrays, can turn some of these optimizations off.
    template<IwiseConfig config = IwiseConfig{}, size_t N, typename Index, typename Op>
    constexpr void iwise(const Shape<Index, N>& shape, Op&& op, i64 n_threads = 1) {
        if constexpr (config.parallel_threshold > 1) {
            const i64 n_elements = shape.template as<i64>().elements();
            const i64 actual_n_threads = n_elements <= config.parallel_threshold ? 1 : n_threads;
            if (actual_n_threads > 1)
                return guts::Iwise::parallel(shape, op, actual_n_threads); // take op by lvalue reference
        }
        guts::Iwise::serial(shape, std::forward<Op>(op));
    }
}
#endif
