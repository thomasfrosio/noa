#pragma once

#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/cpu/ReduceIwise.hpp"
#include "noa/runtime/cpu/ComputeHandle.hpp"

namespace noa::cpu::details {
    template<bool ZipReduced, bool ZipOutput>
    class ReduceAxesIwise {
    public:
        using interface = nd::ReduceIwiseInterface<ZipReduced, ZipOutput>;

        template<usize R, usize N, typename Index, typename Op>
        NOA_NOINLINE static void single_axis(
            const Shape<Index, N>& shape,
            Op op, auto reduced, auto output,
            isize n_threads
        ) {
            if (n_threads > 1) {
                #pragma omp parallel default(none) num_threads(n_threads) shared(shape, reduced, output) firstprivate(op)
                {
                    constexpr auto ci = ComputeHandle<Index, true>{};
                    if constexpr (N == 6) {
                        #pragma omp for collapse(5)
                        for (Index i = 0; i < shape[0]; ++i) {
                            for (Index j = 0; j < shape[1]; ++j) {
                                for (Index k = 0; k < shape[2]; ++k) {
                                    for (Index l = 0; l < shape[3]; ++l) {
                                        for (Index m = 0; m < shape[4]; ++m) {
                                            interface::init(ci, op, i, j, k, l, m);
                                            auto local = reduced;
                                            for (Index n = 0; n < shape[5]; ++n) {
                                                if constexpr (R == 0) {
                                                    interface::call(ci, op, local, n, i, j, k, l, m);
                                                } else if constexpr (R == 1) {
                                                    interface::call(ci, op, local, i, n, j, k, l, m);
                                                } else if constexpr (R == 2) {
                                                    interface::call(ci, op, local, i, j, n, k, l, m);
                                                } else if constexpr (R == 3) {
                                                    interface::call(ci, op, local, i, j, k, n, l, m);
                                                } else if constexpr (R == 4) {
                                                    interface::call(ci, op, local, i, j, k, l, n, m);
                                                } else if constexpr (R == 5) {
                                                    interface::call(ci, op, local, i, j, k, l, m, n);
                                                } else {
                                                    static_assert(nt::always_false<Op>);
                                                }
                                            }
                                            interface::deinit(ci, op, i, j, k, l, m);
                                            interface::post(op, local, output, i, j, k, l, m);
                                        }
                                    }
                                }
                            }
                        }
                    } else if constexpr (N == 5) {
                        #pragma omp for collapse(4)
                        for (Index i = 0; i < shape[0]; ++i) {
                            for (Index j = 0; j < shape[1]; ++j) {
                                for (Index k = 0; k < shape[2]; ++k) {
                                    for (Index l = 0; l < shape[3]; ++l) {
                                        interface::init(ci, op, i, j, k, l);
                                        auto local = reduced;
                                        for (Index m = 0; m < shape[4]; ++m) {
                                            if constexpr (R == 0) {
                                                interface::call(ci, op, local, m, i, j, k, l);
                                            } else if constexpr (R == 1) {
                                                interface::call(ci, op, local, i, m, j, k, l);
                                            } else if constexpr (R == 2) {
                                                interface::call(ci, op, local, i, j, m, k, l);
                                            } else if constexpr (R == 3) {
                                                interface::call(ci, op, local, i, j, k, m, l);
                                            } else if constexpr (R == 4) {
                                                interface::call(ci, op, local, i, j, k, l, m);
                                            } else {
                                                static_assert(nt::always_false<Op>);
                                            }
                                        }
                                        interface::deinit(ci, op, i, j, k, l);
                                        interface::post(op, local, output, i, j, k, l);
                                    }
                                }
                            }
                        }
                    } else if constexpr (N == 4) {
                        #pragma omp for collapse(3)
                        for (Index i = 0; i < shape[0]; ++i) {
                            for (Index j = 0; j < shape[1]; ++j) {
                                for (Index k = 0; k < shape[2]; ++k) {
                                    interface::init(ci, op, i, j, k);
                                    auto local = reduced;
                                    for (Index l = 0; l < shape[3]; ++l) {
                                        if constexpr (R == 0) {
                                            interface::call(ci, op, local, l, i, j, k);
                                        } else if constexpr (R == 1) {
                                            interface::call(ci, op, local, i, l, j, k);
                                        } else if constexpr (R == 2) {
                                            interface::call(ci, op, local, i, j, l, k);
                                        } else if constexpr (R == 3) {
                                            interface::call(ci, op, local, i, j, k, l);
                                        } else {
                                            static_assert(nt::always_false<Op>);
                                        }
                                    }
                                    interface::deinit(ci, op, i, j, k);
                                    interface::post(op, local, output, i, j, k);
                                }
                            }
                        }
                    } else if constexpr (N == 3) {
                        #pragma omp for collapse(2)
                        for (Index i = 0; i < shape[0]; ++i) {
                            for (Index j = 0; j < shape[1]; ++j) {
                                interface::init(ci, op, i, j);
                                auto local = reduced;
                                for (Index k = 0; k < shape[2]; ++k) {
                                    if constexpr (R == 0) {
                                        interface::call(ci, op, local, k, i, j);
                                    } else if constexpr (R == 1) {
                                        interface::call(ci, op, local, i, k, j);
                                    } else if constexpr (R == 2) {
                                        interface::call(ci, op, local, i, j, k);
                                    } else {
                                        static_assert(nt::always_false<Op>);
                                    }
                                }
                                interface::deinit(ci, op, i, j);
                                interface::post(op, local, output, i, j);
                            }
                        }
                    } else if constexpr (N == 2) {
                        #pragma omp for
                        for (Index i = 0; i < shape[0]; ++i) {
                            interface::init(ci, op, i);
                            auto local = reduced;
                            for (Index j = 0; j < shape[1]; ++j) {
                                if constexpr (R == 0) {
                                    interface::call(ci, op, local, j, i);
                                } else {
                                    static_assert(nt::always_false<Op>);
                                }
                            }
                            interface::deinit(ci, op, i);
                            interface::post(op, local, output, i);
                        }
                    } else {
                        static_assert(nt::always_false<Op>);
                    }
                }
            } else {
                constexpr auto ci = ComputeHandle<Index, false>{};
                if constexpr (N == 6) {
                    for (Index i = 0; i < shape[0]; ++i) {
                        for (Index j = 0; j < shape[1]; ++j) {
                            for (Index k = 0; k < shape[2]; ++k) {
                                for (Index l = 0; l < shape[3]; ++l) {
                                    for (Index m = 0; m < shape[4]; ++m) {
                                        interface::init(ci, op, i, j, k, l, m);
                                        auto local = reduced;
                                        for (Index n = 0; n < shape[5]; ++n) {
                                            if constexpr (R == 0) {
                                                interface::call(ci, op, local, n, i, j, k, l, m);
                                            } else if constexpr (R == 1) {
                                                interface::call(ci, op, local, i, n, j, k, l, m);
                                            } else if constexpr (R == 2) {
                                                interface::call(ci, op, local, i, j, n, k, l, m);
                                            } else if constexpr (R == 3) {
                                                interface::call(ci, op, local, i, j, k, n, l, m);
                                            } else if constexpr (R == 4) {
                                                interface::call(ci, op, local, i, j, k, l, n, m);
                                            } else if constexpr (R == 5) {
                                                interface::call(ci, op, local, i, j, k, l, m, n);
                                            } else {
                                                static_assert(nt::always_false<Op>);
                                            }
                                        }
                                        interface::deinit(ci, op, i, j, k, l, m);
                                        interface::post(op, local, output, i, j, k, l, m);
                                    }
                                }
                            }
                        }
                    }
                } else if constexpr (N == 5) {
                    for (Index i = 0; i < shape[0]; ++i) {
                        for (Index j = 0; j < shape[1]; ++j) {
                            for (Index k = 0; k < shape[2]; ++k) {
                                for (Index l = 0; l < shape[3]; ++l) {
                                    interface::init(ci, op, i, j, k, l);
                                    auto local = reduced;
                                    for (Index m = 0; m < shape[4]; ++m) {
                                        if constexpr (R == 0) {
                                            interface::call(ci, op, local, m, i, j, k, l);
                                        } else if constexpr (R == 1) {
                                            interface::call(ci, op, local, i, m, j, k, l);
                                        } else if constexpr (R == 2) {
                                            interface::call(ci, op, local, i, j, m, k, l);
                                        } else if constexpr (R == 3) {
                                            interface::call(ci, op, local, i, j, k, m, l);
                                        } else if constexpr (R == 4) {
                                            interface::call(ci, op, local, i, j, k, l, m);
                                        } else {
                                            static_assert(nt::always_false<Op>);
                                        }
                                    }
                                    interface::deinit(ci, op, i, j, k, l);
                                    interface::post(op, local, output, i, j, k, l);
                                }
                            }
                        }
                    }
                } else if constexpr (N == 4) {
                    for (Index i = 0; i < shape[0]; ++i) {
                        for (Index j = 0; j < shape[1]; ++j) {
                            for (Index k = 0; k < shape[2]; ++k) {
                                interface::init(ci, op, i, j, k);
                                auto local = reduced;
                                for (Index l = 0; l < shape[3]; ++l) {
                                    if constexpr (R == 0) {
                                        interface::call(ci, op, local, l, i, j, k);
                                    } else if constexpr (R == 1) {
                                        interface::call(ci, op, local, i, l, j, k);
                                    } else if constexpr (R == 2) {
                                        interface::call(ci, op, local, i, j, l, k);
                                    } else if constexpr (R == 3) {
                                        interface::call(ci, op, local, i, j, k, l);
                                    } else {
                                        static_assert(nt::always_false<Op>);
                                    }
                                }
                                interface::deinit(ci, op, i, j, k);
                                interface::post(op, local, output, i, j, k);
                            }
                        }
                    }
                } else if constexpr (N == 3) {
                    for (Index i = 0; i < shape[0]; ++i) {
                        for (Index j = 0; j < shape[1]; ++j) {
                            interface::init(ci, op, i, j);
                            auto local = reduced;
                            for (Index k = 0; k < shape[2]; ++k) {
                                if constexpr (R == 0) {
                                    interface::call(ci, op, local, k, i, j);
                                } else if constexpr (R == 1) {
                                    interface::call(ci, op, local, i, k, j);
                                } else if constexpr (R == 2) {
                                    interface::call(ci, op, local, i, j, k);
                                } else {
                                    static_assert(nt::always_false<Op>);
                                }
                            }
                            interface::deinit(ci, op, i, j);
                            interface::post(op, local, output, i, j);
                        }
                    }
                } else if constexpr (N == 2) {
                    for (Index i = 0; i < shape[0]; ++i) {
                        interface::init(ci, op, i);
                        auto local = reduced;
                        for (Index j = 0; j < shape[1]; ++j) {
                            if constexpr (R == 0) {
                                interface::call(ci, op, local, j, i);
                            } else {
                                static_assert(nt::always_false<Op>);
                            }
                        }
                        interface::deinit(ci, op, i);
                        interface::post(op, local, output, i);
                    }
                } else {
                    static_assert(nt::always_false<Op>);
                }
            }
        }

        enum ReductionMode {
            SerialReduction,
            ParallelReduction,
        };

        template<ReductionMode MODE, typename Index, usize N>
        NOA_NOINLINE static void batch_reduction_nd_parallel(const Shape<Index, N>& shape, auto op, auto reduced, auto output, isize n_threads) {
            auto original_reduced = reduced;
            #pragma omp parallel default(none) num_threads(n_threads) shared(shape, reduced, output, original_reduced) firstprivate(op)
            {
                constexpr auto ci = ComputeHandle<Index, true>{};
                if constexpr (MODE == ReductionMode::SerialReduction) {
                    #pragma omp for
                    for (Index i = 0; i < shape[0]; ++i) {
                        interface::init(ci, op, i);
                        auto local = reduced;
                        if constexpr (N == 6) {
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    for (Index l = 0; l < shape[3]; ++l)
                                        for (Index m = 0; m < shape[4]; ++m)
                                            for (Index n = 0; n < shape[5]; ++n)
                                                interface::call(ci, op, local, i, j, k, l, m, n);
                        } else if constexpr (N == 5) {
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    for (Index l = 0; l < shape[3]; ++l)
                                        for (Index m = 0; m < shape[4]; ++m)
                                            interface::call(ci, op, local, i, j, k, l, m);
                        } else if constexpr (N == 4) {
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    for (Index l = 0; l < shape[3]; ++l)
                                        interface::call(ci, op, local, i, j, k, l);
                        } else if constexpr (N == 3) {
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    interface::call(ci, op, local, i, j, k);
                        } else if constexpr (N == 2) {
                            for (Index j = 0; j < shape[1]; ++j)
                                interface::call(ci, op, local, i, j);
                        }
                        interface::deinit(ci, op, i);
                        interface::post(op, local, output, i);
                    }
                } else if constexpr (MODE == ReductionMode::ParallelReduction) {
                    for (Index i = 0; i < shape[0]; ++i) {
                        interface::init(ci, op, i);
                        auto local = reduced;
                        if constexpr (N == 6) {
                            #pragma omp for collapse(5)
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    for (Index l = 0; l < shape[3]; ++l)
                                        for (Index m = 0; m < shape[4]; ++m)
                                            for (Index n = 0; n < shape[5]; ++n)
                                                interface::call(ci, op, local, i, j, k, l, m, n);
                        } else if constexpr (N == 5) {
                            #pragma omp for collapse(4)
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    for (Index l = 0; l < shape[3]; ++l)
                                        for (Index m = 0; m < shape[4]; ++m)
                                            interface::call(ci, op, local, i, j, k, l, m);
                        } else if constexpr (N == 4) {
                            #pragma omp for collapse(3)
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    for (Index l = 0; l < shape[3]; ++l)
                                        interface::call(ci, op, local, i, j, k, l);
                        } else if constexpr (N == 3) {
                            #pragma omp for collapse(2)
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    interface::call(ci, op, local, i, j, k);
                        } else if constexpr (N == 2) {
                            #pragma omp for
                            for (Index j = 0; j < shape[1]; ++j)
                                interface::call(ci, op, local, i, j);
                        }
                        interface::deinit(ci, op, i);

                        #pragma omp critical
                        interface::join(op, local, reduced);

                        #pragma omp barrier
                        #pragma omp single
                        {
                            interface::post(op, reduced, output, i);
                            reduced = original_reduced;
                        }
                    }
                } else {
                    static_assert(nt::always_false<Index>);
                }
            }
        }

        template<typename Index, usize N>
        NOA_NOINLINE static constexpr void batch_reduction_nd_serial(const Shape<Index, N>& shape, auto op, auto reduced, auto output) {
            constexpr auto ci = ComputeHandle<Index, false>{};
            for (Index i = 0; i < shape[0]; ++i) {
                interface::init(ci, op, i);
                auto local = reduced;
                if constexpr (N == 6) {
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            for (Index l = 0; l < shape[3]; ++l)
                                for (Index m = 0; m < shape[4]; ++m)
                                    for (Index n = 0; n < shape[5]; ++n)
                                        interface::call(ci, op, local, i, j, k, l, m, n);
                } else if constexpr (N == 5) {
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            for (Index l = 0; l < shape[3]; ++l)
                                for (Index m = 0; m < shape[4]; ++m)
                                    interface::call(ci, op, local, i, j, k, l, m);
                } else if constexpr (N == 4) {
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            for (Index l = 0; l < shape[3]; ++l)
                                interface::call(ci, op, local, i, j, k, l);
                } else if constexpr (N == 3) {
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            interface::call(ci, op, local, i, j, k);
                } else if constexpr (N == 2) {
                    for (Index j = 0; j < shape[1]; ++j)
                        interface::call(ci, op, local, i, j);
                }
                interface::deinit(ci, op, i);
                interface::post(op, local, output, i);
            }
        }
    };
}

namespace noa::cpu {
    template<bool ZipReduced = false, bool ZipOutput = false, isize ElementsPerThread = 1'048'576>
    struct ReduceAxesIwiseConfig {
        static constexpr bool zip_reduced = ZipReduced;
        static constexpr bool zip_output = ZipOutput;
        static constexpr isize n_elements_per_thread = ElementsPerThread;
    };

    template<typename Config = ReduceAxesIwiseConfig<>,
            typename Op, usize N, typename Reduced, typename Output, typename Index>
    requires (nt::tuple_of_accessor_pure_nd_or_empty<std::decay_t<Output>, N> and
              nt::tuple_of_accessor_value_or_empty<std::decay_t<Reduced>>)
    NOA_NOINLINE constexpr void reduce_axes_iwise(
        const Shape<Index, N>& input_shape,
        const Shape<Index, N>& output_shape,
        Op&& op,
        Reduced&& reduced,
        Output&& output,
        i32 n_threads = 1
    ) {
        const auto axes_to_reduce = input_shape.cmp_ne(output_shape);
        const auto axes_empty_or_to_reduce = output_shape.cmp_eq(1) or axes_to_reduce;

        if (axes_empty_or_to_reduce == true) { // reduce to a single value
            auto output_1d = nd::reconfig_accessors<nd::AccessorConfig{.enforce_contiguous = true}>(output, 0);
            return reduce_iwise<Config>(input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
        }
        NOA_ASSERT(N >= 2);

        if constexpr (N >= 2) {
            using reduce_axes_iwise_t = details::ReduceAxesIwise<Config::zip_reduced, Config::zip_output>;
            const auto shape = input_shape.template as<isize>();
            const auto n_batches = clamp_cast<i32>(shape[0]);
            const isize n_elements = shape.n_elements();
            const bool is_small = n_elements <= Config::n_elements_per_thread;

            i32 actual_n_threads = is_small ? 1 : n_threads;
            if (actual_n_threads > 1)
                actual_n_threads = min(n_threads, clamp_cast<i32>(n_elements / Config::n_elements_per_thread));

            // Batch reduction.
            if (axes_empty_or_to_reduce.pop_front() == true) {
                auto output_1d = nd::reconfig_accessors(output, 0);
                if (is_small) {
                    reduce_axes_iwise_t::batch_reduction_nd_serial(input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d);
                } else if (n_batches < actual_n_threads) {
                    reduce_axes_iwise_t::template batch_reduction_nd_parallel<reduce_axes_iwise_t::ParallelReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, actual_n_threads);
                } else {
                    reduce_axes_iwise_t::template batch_reduction_nd_parallel<reduce_axes_iwise_t::SerialReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, actual_n_threads);
                }
                return;
            }

            // Single axis reduction.
            if constexpr (N == 6) {
                if (axes_to_reduce[5]) {
                    auto output_5d = nd::reconfig_accessors(output, 0, 1, 2, 3, 4);
                    reduce_axes_iwise_t::template single_axis<5>(
                        input_shape.filter(0, 1, 2, 3, 4, 5), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_5d, actual_n_threads);
                } else if (axes_to_reduce[4]) {
                    auto output_5d = nd::reconfig_accessors(output, 0, 1, 2, 3, 5);
                    reduce_axes_iwise_t::template single_axis<4>(
                        input_shape.filter(0, 1, 2, 3, 5, 4), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_5d, actual_n_threads);
                } else if (axes_to_reduce[3]) {
                    auto output_5d = nd::reconfig_accessors(output, 0, 1, 2, 4, 5);
                    reduce_axes_iwise_t::template single_axis<3>(
                        input_shape.filter(0, 1, 2, 4, 5, 3), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_5d, actual_n_threads);
                } else if (axes_to_reduce[2]) {
                    auto output_5d = nd::reconfig_accessors(output, 0, 1, 3, 4, 5);
                    reduce_axes_iwise_t::template single_axis<2>(
                        input_shape.filter(0, 1, 3, 4, 5, 2), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_5d, actual_n_threads);
                } else if (axes_to_reduce[1]) {
                    auto output_5d = nd::reconfig_accessors(output, 0, 2, 3, 4, 5);
                    reduce_axes_iwise_t::template single_axis<1>(
                        input_shape.filter(0, 2, 3, 4, 5, 1), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_5d, actual_n_threads);
                } else {
                    auto output_5d = nd::reconfig_accessors(output, 1, 2, 3, 4, 5);
                    reduce_axes_iwise_t::template single_axis<0>(
                        input_shape.filter(1, 2, 3, 4, 5, 0), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_5d, actual_n_threads);
                }
            } else if constexpr (N == 5) {
                if (axes_to_reduce[4]) {
                    auto output_4d = nd::reconfig_accessors(output, 0, 1, 2, 3);
                    reduce_axes_iwise_t::template single_axis<4>(
                        input_shape.filter(0, 1, 2, 3, 4), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_4d, actual_n_threads);
                } else if (axes_to_reduce[3]) {
                    auto output_4d = nd::reconfig_accessors(output, 0, 1, 2, 4);
                    reduce_axes_iwise_t::template single_axis<3>(
                        input_shape.filter(0, 1, 2, 4, 3), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_4d, actual_n_threads);
                } else if (axes_to_reduce[2]) {
                    auto output_4d = nd::reconfig_accessors(output, 0, 1, 3, 4);
                    reduce_axes_iwise_t::template single_axis<2>(
                        input_shape.filter(0, 1, 3, 4, 2), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_4d, actual_n_threads);
                } else if (axes_to_reduce[1]) {
                    auto output_4d = nd::reconfig_accessors(output, 0, 2, 3, 4);
                    reduce_axes_iwise_t::template single_axis<1>(
                        input_shape.filter(0, 2, 3, 4, 1), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_4d, actual_n_threads);
                } else {
                    auto output_4d = nd::reconfig_accessors(output, 1, 2, 3, 4);
                    reduce_axes_iwise_t::template single_axis<0>(
                        input_shape.filter(1, 2, 3, 4, 0), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_4d, actual_n_threads);
                }
            } else if constexpr (N == 4) {
                if (axes_to_reduce[3]) {
                    auto output_3d = nd::reconfig_accessors(output, 0, 1, 2);
                    reduce_axes_iwise_t::template single_axis<3>(
                        input_shape, std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_3d, actual_n_threads);
                } else if (axes_to_reduce[2]) {
                    auto output_3d = nd::reconfig_accessors(output, 0, 1, 3);
                    reduce_axes_iwise_t::template single_axis<2>(
                        input_shape.filter(0, 1, 3, 2), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_3d, actual_n_threads);
                } else if (axes_to_reduce[1]) {
                    auto output_3d = nd::reconfig_accessors(output, 0, 2, 3);
                    reduce_axes_iwise_t::template single_axis<1>(
                        input_shape.filter(0, 2, 3, 1), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_3d, actual_n_threads);
                } else {
                    auto output_3d = nd::reconfig_accessors(output, 1, 2, 3);
                    reduce_axes_iwise_t::template single_axis<0>(
                        input_shape.filter(1, 2, 3, 0), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_3d, actual_n_threads);
                }
            } else if constexpr (N == 3) {
                if (axes_to_reduce[2]) {
                    auto output_2d = nd::reconfig_accessors(output, 0, 1);
                    reduce_axes_iwise_t::template single_axis<2>(
                        input_shape, std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_2d, actual_n_threads);
                } else if (axes_to_reduce[1]) {
                    auto output_2d = nd::reconfig_accessors(output, 0, 2);
                    reduce_axes_iwise_t::template single_axis<1>(
                        input_shape.filter(0, 2, 1), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_2d, actual_n_threads);
                } else {
                    auto output_2d = nd::reconfig_accessors(output, 1, 2);
                    reduce_axes_iwise_t::template single_axis<0>(
                        input_shape.filter(1, 2, 0), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_2d, actual_n_threads);
                }
            } else if constexpr (N == 2) {
                // axes_to_reduce[1] == false and axes_to_reduce[0] == true
                auto output_1d = nd::reconfig_accessors(output, 1);
                reduce_axes_iwise_t::template single_axis<0>(
                    input_shape.filter(1, 0), std::forward<Op>(op),
                    std::forward<Reduced>(reduced), output_1d, actual_n_threads);
            }
        }
    }
}
