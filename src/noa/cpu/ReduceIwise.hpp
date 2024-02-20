#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include <omp.h>
#include "noa/core/Types.hpp"
#include "noa/core/utils/Interfaces.hpp"

namespace noa::cpu::guts {
    /// Index-wise reduction.
    template<bool ZipReduced, bool ZipOutput>
    class ReduceIwise {
    public:
        using interface = ng::ReduceIwiseInterface<ZipReduced, ZipOutput>;

        template<typename Op, typename Reduced, typename Output, typename Index, size_t N>
        static void parallel(
                const Vec<Index, N>& shape, Op& op,
                Reduced reduced, Output& output, i64 n_threads
        ) {
            #pragma omp parallel default(none) num_threads(n_threads) shared(shape, reduced) firstprivate(op)
            {
                // Local copy of the reduced values.
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
                    // Join the reduced values of each thread together.
                    interface::join(op, local_reduce, reduced);
                }
            }
            interface::final(op, reduced, output, 0);
        }

        template<typename Op, typename Reduced, typename Output, typename Index, size_t N>
        static constexpr void serial(
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
    struct ReduceIwiseConfig {
        bool zip_reduced = false;
        bool zip_output = false;
        i64 parallel_threshold = 1'048'576; // 1024x1024
    };

    /// Dispatches a index-wise reduction operator across N-dimensional (parallel) for-loops.
    /// \param start        Starting indices, usually 0.
    /// \param end          Ending indices, usually the shape.
    /// \param op           Valid reduce index-wise operator (see core interface). The operator is forwarded to
    ///                     the kernel, which takes it by value (it is either moved or copied once by the time
    ///                     it reaches the kernel). In the multi-threaded case, it is copied to every thread.
    /// \param reduced      Tuple of accessor-values, initialized with the values to start the reduction.
    ///                     These are moved/copied into the kernel, or copied to every thread in the parallel case.
    /// \param output       Tuple of accessors or accessor-values. These are taken by reference to the kernel
    ///                     and never moved/copied, so the caller can read the updated values from accessor-values.
    /// \param n_threads    Maximum number of threads. Note that passing a literal 1 (the default value)
    ///                     should reduce the amount of generated code because the parallel version can
    ///                     be omitted. In this case, it is even better to set \p PARALLEL_THRESHOLD to 1,
    ///                     turning off the multi-threaded implementation entirely.
    template<ReduceIwiseConfig config = ReduceIwiseConfig{},
             typename Op, typename Reduced, typename Output, typename Index, size_t N>
    requires (nt::is_tuple_of_accessor_value_v<Reduced> and nt::is_tuple_of_accessor_v<Output>)
    constexpr void reduce_iwise(
            const Shape<Index, N>& shape,
            Op&& op,
            Reduced&& reduced,
            Output& output,
            i64 n_threads = 1
    )  {
        using core = guts::ReduceIwise<config.zip_reduced, config.zip_output>;
        if constexpr (config.parallel_threshold > 1) {
            const i64 actual_n_threads = shape.template as<i64>().elements() <= config.parallel_threshold ? 1 : n_threads;
            if (actual_n_threads > 1)
                return core::parallel(shape.vec, op, std::forward<Reduced>(reduced), output, actual_n_threads);
        }
        core::serial(shape.vec, std::forward<Op>(op), std::forward<Reduced>(reduced), output);
    }
}
#endif
