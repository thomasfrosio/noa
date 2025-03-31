#pragma once

#include <omp.h>
#include "noa/core/Interfaces.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Shape.hpp"

namespace noa::cpu::guts {
    class Iwise {
    public:
        template<size_t N, typename Index, typename Operator>
        [[gnu::noinline]] static void parallel(const Shape<Index, N>& shape, Operator op, i64 n_threads) {
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
        [[gnu::noinline]] static constexpr void serial(const Shape<Index, N>& shape, Operator op) {
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
    template<i64 ElementsPerThread = 1'048'576>
    struct IwiseConfig {
        static constexpr i64 n_elements_per_thread = ElementsPerThread;
    };

    template<typename Config = IwiseConfig<>, size_t N, typename Index, typename Op>
    constexpr void iwise(const Shape<Index, N>& shape, Op&& op, i64 n_threads = 1) {
        if constexpr (Config::n_elements_per_thread > 1) {
            const i64 n_elements = shape.template as<i64>().n_elements();
            i64 actual_n_threads = n_elements <= Config::n_elements_per_thread ? 1 : n_threads;
            if (actual_n_threads > 1)
                actual_n_threads = min(n_threads, n_elements / Config::n_elements_per_thread);

            if (actual_n_threads > 1)
                return guts::Iwise::parallel(shape, std::forward<Op>(op), actual_n_threads);
        }
        guts::Iwise::serial(shape, std::forward<Op>(op));
    }
}
