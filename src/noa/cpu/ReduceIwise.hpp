#pragma once

#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/Interfaces.hpp"

namespace noa::cpu::guts {
    template<bool ZipReduced, bool ZipOutput>
    class ReduceIwise {
    public:
        using interface = ng::ReduceIwiseInterface<ZipReduced, ZipOutput>;

        template<typename Op, typename Reduced, typename Output, typename Index, size_t N>
        [[gnu::noinline]] static void parallel(
            const Vec<Index, N>& shape, Op op,
            Reduced reduced, Output& output, i64 n_threads
        ) {
            #pragma omp parallel default(none) num_threads(n_threads) shared(shape, reduced) firstprivate(op)
            {
                auto local_reduce = reduced;

                if constexpr (N == 4) {
                    #pragma omp for collapse(4)
                    for (Index i = 0; i < shape[0]; ++i)
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                for (Index l = 0; l < shape[3]; ++l)
                                    interface::init(op, local_reduce, i, j, k, l);

                } else if constexpr (N == 3) {
                    #pragma omp for collapse(3)
                    for (Index i = 0; i < shape[0]; ++i)
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                interface::init(op, local_reduce, i, j, k);

                } else if constexpr (N == 2) {
                    #pragma omp for collapse(2)
                    for (Index i = 0; i < shape[0]; ++i)
                        for (Index j = 0; j < shape[1]; ++j)
                            interface::init(op, local_reduce, i, j);

                } else if constexpr (N == 1) {
                    #pragma omp for collapse(1)
                    for (Index i = 0; i < shape[0]; ++i)
                        interface::init(op, local_reduce, i);
                }

                #pragma omp critical
                {
                    interface::join(op, local_reduce, reduced);
                }
            }
            interface::final(op, reduced, output, 0);
        }

        template<typename Op, typename Reduced, typename Output, typename Index, size_t N>
        [[gnu::noinline]] static constexpr void serial(
            const Vec<Index, N>& shape, Op op,
            Reduced reduced, Output& output
        ) {
            if constexpr (N == 4) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            for (Index l = 0; l < shape[3]; ++l)
                                interface::init(op, reduced, i, j, k, l);

            } else if constexpr (N == 3) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            interface::init(op, reduced, i, j, k);

            } else if constexpr (N == 2) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        interface::init(op, reduced, i, j);

            } else if constexpr (N == 1) {
                for (Index i = 0; i < shape[0]; ++i)
                    interface::init(op, reduced, i);
            }

            interface::final(op, reduced, output, 0);
        }
    };
}

namespace noa::cpu {
    template<bool ZipReduced = false, bool ZipOutput = false, i64 ElementsPerThread = 1'048'576>
    struct ReduceIwiseConfig {
        static constexpr bool zip_reduced = ZipReduced;
        static constexpr bool zip_output = ZipOutput;
        static constexpr i64 n_elements_per_thread = ElementsPerThread;
    };

    template<typename Config = ReduceIwiseConfig<>,
             typename Op, typename Reduced, typename Output, typename Index, size_t N>
    requires (nt::tuple_of_accessor_value<std::decay_t<Reduced>> and
              nt::tuple_of_accessor_nd_or_empty<Output, 1>)
    constexpr void reduce_iwise(
        const Shape<Index, N>& shape,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        i64 n_threads = 1
    ) {
        using reduce_iwise_t = guts::ReduceIwise<Config::zip_reduced, Config::zip_output>;
        if constexpr (Config::n_elements_per_thread > 1) {
            const i64 n_elements = shape.template as<i64>().n_elements();
            i64 actual_n_threads = n_elements <= Config::n_elements_per_thread ? 1 : n_threads;
            if (actual_n_threads > 1)
                actual_n_threads = min(n_threads, n_elements / Config::n_elements_per_thread);

            if (actual_n_threads > 1) {
                return reduce_iwise_t::parallel(
                    shape.vec, std::forward<Op>(op), std::forward<Reduced>(reduced), output, actual_n_threads);
            }
        }
        reduce_iwise_t::serial(shape.vec, std::forward<Op>(op), std::forward<Reduced>(reduced), output);
    }
}
