#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/utils/IwiseOp.hpp"

#include <omp.h>

/*
 * noa::reduce_iwise(shape, device, noa::wrap(f64{}, i32{}), f32{}, op);
 * struct Op {
 *      void operator()(indices..., f64&, i32&); // wrap
 *      void operator()(indices..., Tuple<f64&, i32&>); // zip
 * };
 */

namespace noa::cpu {
    /// Index-wise reduction.
    template<guts::ReduceIwiseChecker c>
    class ReduceIwise {
    public:
        using checker = decltype(c);
        using index_type = checker::index_type;
        using vec_type = Vec<index_type, checker::N_DIMENIONS>;
        using shape_type = Shape<index_type, checker::N_DIMENIONS>;
        using shape_i64_type = Shape<i64, checker::N_DIMENIONS>;

        /// Default threshold to trigger the multi-threaded implementation.
        /// This is a large number of elements to make sure there's enough work for each thread.
        static constexpr i64 PARALLEL_ELEMENTS_THRESHOLD = 1'048'576; // 1024x1024

        /// Launches the loop.
        /// \param start    Starting indices, usually 0.
        /// \param end      Ending indices, usually the shape.
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
                 typename Operator,
                 typename... ReducedAccessors,
                 typename... OutputAccessors>
        requires nt::are_accessor_value_v<ReducedAccessors...> &&
                 nt::are_accessor_value_v<OutputAccessors...>
        constexpr void execute(
                const vec_type& start,
                const vec_type& end,
                Operator&& op,
                const Tuple<ReducedAccessors...>& reduced_accessors,
                const Tuple<OutputAccessors...>& output_accessors,
                i64 threads = 1
        )  {
            if constexpr (THRESHOLD > 1) {
                const auto shape = shape_i64_type::from_vec(end.template as<i64>() - start.template as<i64>());
                const i64 actual_threads = shape.elements() <= THRESHOLD ? 1 : threads;
                if (actual_threads <= 1) {
                    return reduce_iwise_parallel_(start, end, std::forward<Operator>(op),
                                                  reduced_accessors, output_accessors, actual_threads);
                }
            }
            reduce_iwise_serial_(start, end, std::forward<Operator>(op), reduced_accessors, output_accessors);
        }

        template<i64 THRESHOLD = PARALLEL_ELEMENTS_THRESHOLD,
                 typename Operator,
                 typename... ReducedAccessors,
                 typename... OutputAccessors>
        constexpr void execute(
                const shape_type& shape, Operator&& op,
                const Tuple<ReducedAccessors...>& reduced_accessors,
                const Tuple<OutputAccessors...>& output_accessors,
                i64 threads = 1
        ) {
            execute<THRESHOLD>(
                    vec_type{}, shape.vec, std::forward<Operator>(op),
                    reduced_accessors, output_accessors, threads);
        }

    private:
        template<typename Operator, typename ReducedAccessors, typename OutputAccessors>
        static void reduce_iwise_parallel_(
                const vec_type& start, const vec_type& end, Operator&& op,
                ReducedAccessors& reduced, OutputAccessors& output, i64 threads
        ) {
            #pragma omp parallel default(none) num_threads(threads) shared(start, end, reduced, output) firstprivate(op)
            {
                // Local copy of the reduced values.
                auto local_reduce = reduced;

                if constexpr (checker::N_DIMENSIONS == 4) {
                    #pragma omp for collapse(4)
                    for (index_type i = start[0]; i < end[0]; ++i)
                        for (index_type j = start[1]; j < end[1]; ++j)
                            for (index_type k = start[2]; k < end[2]; ++k)
                                for (index_type l = start[3]; l < end[3]; ++l)
                                    guts::ReduceIwiseOpWrapper::init<false, false>(op, local_reduce, i, j, k, l);

                } else if constexpr (checker::N_DIMENSIONS == 3) {
                    #pragma omp for collapse(3)
                    for (index_type i = start[0]; i < end[0]; ++i)
                        for (index_type j = start[1]; j < end[1]; ++j)
                            for (index_type k = start[2]; k < end[2]; ++k)
                                guts::ReduceIwiseOpWrapper::init<false, false>(op, local_reduce, i, j, k);

                } else if constexpr (checker::N_DIMENSIONS == 2) {
                    #pragma omp for collapse(2)
                    for (index_type i = start[0]; i < end[0]; ++i)
                        for (index_type j = start[1]; j < end[1]; ++j)
                            guts::ReduceIwiseOpWrapper::init<false, false>(op, local_reduce, i, j);

                } else if constexpr (checker::N_DIMENSIONS == 1) {
                    #pragma omp for collapse(1)
                    for (index_type i = start[0]; i < end[0]; ++i)
                        guts::ReduceIwiseOpWrapper::init<false, false>(op, local_reduce, i);
                }

                #pragma omp critical
                {
                    // Join the reduced values of each thread together.
                    guts::ReduceIwiseOpWrapper::join<false, false>(op, local_reduce, reduced);
                }
            }
            guts::ReduceIwiseOpWrapper::final<false, false>(op, reduced, output);
        }

        template<typename Operator, typename ReducedAccessors, typename OutputAccessors>
        static constexpr void reduce_iwise_serial_(
                const vec_type& start, const vec_type& end, Operator& op,
                ReducedAccessors& reduced, OutputAccessors& output
        ) {
            if constexpr (checker::N_DIMENSIONS == 4) {
                for (index_type i = start[0]; i < end[0]; ++i)
                    for (index_type j = start[1]; j < end[1]; ++j)
                        for (index_type k = start[2]; k < end[2]; ++k)
                            for (index_type l = start[3]; l < end[3]; ++l)
                                guts::ReduceIwiseOpWrapper::init<false, false>(op, reduced, i, j, k, l);

            } else if constexpr (checker::N_DIMENSIONS == 3) {
                for (index_type i = start[0]; i < end[0]; ++i)
                    for (index_type j = start[1]; j < end[1]; ++j)
                        for (index_type k = start[2]; k < end[2]; ++k)
                            guts::ReduceIwiseOpWrapper::init<false, false>(op, reduced, i, j, k);

            } else if constexpr (checker::N_DIMENSIONS == 2) {
                for (index_type i = start[0]; i < end[0]; ++i)
                    for (index_type j = start[1]; j < end[1]; ++j)
                        guts::ReduceIwiseOpWrapper::init<false, false>(op, reduced, i, j);

            } else if constexpr (checker::N_DIMENSIONS == 1) {
                for (index_type i = start[0]; i < end[0]; ++i)
                    guts::ReduceIwiseOpWrapper::init<false, false>(op, reduced, i);
            }

            guts::ReduceIwiseOpWrapper::final<false, false>(op, reduced, output);
        }
    };
}
