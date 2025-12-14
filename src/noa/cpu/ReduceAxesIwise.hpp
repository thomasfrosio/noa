#pragma once

#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/Interfaces.hpp"
#include "noa/cpu/ReduceIwise.hpp"

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
                    if constexpr (N == 4) {
                        #pragma omp for collapse(3)
                        for (Index i = 0; i < shape[0]; ++i) {
                            for (Index j = 0; j < shape[1]; ++j) {
                                for (Index k = 0; k < shape[2]; ++k) {
                                    auto local = reduced;
                                    for (Index l = 0; l < shape[3]; ++l) {
                                        if constexpr (R == 0) {
                                            interface::init(op, local, l, i, j, k);
                                        } else if constexpr (R == 1) {
                                            interface::init(op, local, i, l, j, k);
                                        } else if constexpr (R == 2) {
                                            interface::init(op, local, i, j, l, k);
                                        } else if constexpr (R == 3) {
                                            interface::init(op, local, i, j, k, l);
                                        } else {
                                            static_assert(nt::always_false<Op>);
                                        }
                                    }
                                    interface::final(op, local, output, i, j, k);
                                }
                            }
                        }
                    } else if constexpr (N == 3) {
                        #pragma omp for collapse(2)
                        for (Index i = 0; i < shape[0]; ++i) {
                            for (Index j = 0; j < shape[1]; ++j) {
                                auto local = reduced;
                                for (Index k = 0; k < shape[2]; ++k) {
                                    if constexpr (R == 0) {
                                        interface::init(op, local, k, i, j);
                                    } else if constexpr (R == 1) {
                                        interface::init(op, local, i, k, j);
                                    } else if constexpr (R == 2) {
                                        interface::init(op, local, i, j, k);
                                    } else {
                                        static_assert(nt::always_false<Op>);
                                    }
                                }
                                interface::final(op, local, output, i, j);
                            }
                        }
                    } else if constexpr (N == 2) {
                        #pragma omp for
                        for (Index i = 0; i < shape[0]; ++i) {
                            auto local = reduced;
                            for (Index j = 0; j < shape[1]; ++j) {
                                if constexpr (R == 0) {
                                    interface::init(op, local, j, i);
                                } else {
                                    static_assert(nt::always_false<Op>);
                                }
                            }
                            interface::final(op, local, output, i);
                        }
                    } else {
                        static_assert(nt::always_false<Op>);
                    }
                }
            } else {
                if constexpr (N == 4) {
                    for (Index i = 0; i < shape[0]; ++i) {
                        for (Index j = 0; j < shape[1]; ++j) {
                            for (Index k = 0; k < shape[2]; ++k) {
                                auto local = reduced;
                                for (Index l = 0; l < shape[3]; ++l) {
                                    if constexpr (R == 0) {
                                        interface::init(op, local, l, i, j, k);
                                    } else if constexpr (R == 1) {
                                        interface::init(op, local, i, l, j, k);
                                    } else if constexpr (R == 2) {
                                        interface::init(op, local, i, j, l, k);
                                    } else if constexpr (R == 3) {
                                        interface::init(op, local, i, j, k, l);
                                    } else {
                                        static_assert(nt::always_false<Op>);
                                    }
                                }
                                interface::final(op, local, output, i, j, k);
                            }
                        }
                    }
                } else if constexpr (N == 3) {
                    for (Index i = 0; i < shape[0]; ++i) {
                        for (Index j = 0; j < shape[1]; ++j) {
                            auto local = reduced;
                            for (Index k = 0; k < shape[2]; ++k) {
                                if constexpr (R == 0) {
                                    interface::init(op, local, k, i, j);
                                } else if constexpr (R == 1) {
                                    interface::init(op, local, i, k, j);
                                } else if constexpr (R == 2) {
                                    interface::init(op, local, i, j, k);
                                } else {
                                    static_assert(nt::always_false<Op>);
                                }
                            }
                            interface::final(op, local, output, i, j);
                        }
                    }
                } else if constexpr (N == 2) {
                    for (Index i = 0; i < shape[0]; ++i) {
                        auto local = reduced;
                        for (Index j = 0; j < shape[1]; ++j) {
                            if constexpr (R == 0) {
                                interface::init(op, local, j, i);
                            } else {
                                static_assert(nt::always_false<Op>);
                            }
                        }
                        interface::final(op, local, output, i);
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

        template<ReductionMode MODE, typename Index>
        NOA_NOINLINE static void parallel_4d(const Shape<Index, 4>& shape, auto op, auto reduced, auto output, isize n_threads) {
            auto original_reduced = reduced;
            #pragma omp parallel default(none) num_threads(n_threads) shared(shape, reduced, output, original_reduced) firstprivate(op)
            {
                if constexpr (MODE == ReductionMode::SerialReduction) {
                    #pragma omp for
                    for (Index i = 0; i < shape[0]; ++i) {
                        auto local = reduced;
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                for (Index l = 0; l < shape[3]; ++l)
                                    interface::init(op, local, i, j, k, l);
                        interface::final(op, local, output, i);
                    }
                } else if constexpr (MODE == ReductionMode::ParallelReduction) {
                    for (Index i = 0; i < shape[0]; ++i) {
                        auto local = reduced;
                        #pragma omp for collapse(3)
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                for (Index l = 0; l < shape[3]; ++l)
                                    interface::init(op, local, i, j, k, l);
                        #pragma omp critical
                        interface::join(op, local, reduced);

                        #pragma omp barrier
                        #pragma omp single
                        {
                            interface::final(op, reduced, output, i);
                            reduced = original_reduced;
                        }
                    }
                }
            }
        }

        template<typename Index>
        NOA_NOINLINE static constexpr void serial_4d(const Shape<Index, 4>& shape, auto op, auto reduced, auto output) {
            for (Index i = 0; i < shape[0]; ++i) {
                auto local = reduced;
                for (Index j = 0; j < shape[1]; ++j)
                    for (Index k = 0; k < shape[2]; ++k)
                        for (Index l = 0; l < shape[3]; ++l)
                            interface::init(op, local, i, j, k, l);
                interface::final(op, local, output, i);
            }
        }

        template<ReductionMode MODE, typename Index>
        NOA_NOINLINE static void parallel_3d(const Shape<Index, 3>& shape, auto op, auto reduced, auto output, isize n_threads) {
            auto original_reduced = reduced;
            #pragma omp parallel default(none) num_threads(n_threads) shared(shape, reduced, output, original_reduced) firstprivate(op)
            {
                if constexpr (MODE == ReductionMode::SerialReduction) {
                    #pragma omp for
                    for (Index i = 0; i < shape[0]; ++i) {
                        auto local = reduced;
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                interface::init(op, local, i, j, k);
                        interface::final(op, local, output, i);
                    }
                } else if constexpr (MODE == ReductionMode::ParallelReduction) {
                    for (Index i = 0; i < shape[0]; ++i) {
                        auto local = reduced;
                        #pragma omp for collapse(2)
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                interface::init(op, local, i, j, k);
                        #pragma omp critical
                        interface::join(op, local, reduced);

                        #pragma omp barrier
                        #pragma omp single
                        {
                            interface::final(op, reduced, output, i);
                            reduced = original_reduced;
                        }
                    }
                } else {
                    static_assert(nt::always_false<Index>);
                }
            }
        }

        template<typename Index>
        NOA_NOINLINE static constexpr void serial_3d(const Shape<Index, 3>& shape, auto op, auto reduced, auto output) {
            for (Index d = 0; d < shape[0]; ++d) {
                auto local = reduced;
                for (Index h = 0; h < shape[1]; ++h)
                    for (Index w = 0; w < shape[2]; ++w)
                        interface::init(op, local, d, h, w);
                interface::final(op, local, output, d);
            }
        }

        template<ReductionMode MODE, typename Index>
        NOA_NOINLINE static void parallel_2d(const Shape<Index, 2>& shape, auto op, auto reduced, auto output, isize n_threads) {
            auto original_reduced = reduced;
            #pragma omp parallel default(none) num_threads(n_threads) shared(shape, reduced, output, original_reduced) firstprivate(op)
            {
                if constexpr (MODE == ReductionMode::SerialReduction) {
                    #pragma omp for
                    for (Index i = 0; i < shape[0]; ++i) {
                        auto local = reduced;
                        for (Index j = 0; j < shape[1]; ++j)
                            interface::init(op, local, i, j);
                        interface::final(op, local, output, i);
                    }
                } else if constexpr (MODE == ReductionMode::ParallelReduction) {
                    for (Index i = 0; i < shape[0]; ++i) {
                        auto local = reduced;
                        #pragma omp for
                        for (Index j = 0; j < shape[1]; ++j)
                            interface::init(op, local, i, j);
                        #pragma omp critical
                        interface::join(op, local, reduced);

                        #pragma omp barrier
                        #pragma omp single
                        {
                            interface::final(op, reduced, output, i);
                            reduced = original_reduced;
                        }
                    }
                } else {
                    static_assert(nt::always_false<Index>);
                }
            }
        }

        template<typename Index>
        NOA_NOINLINE static constexpr void serial_2d(const Shape<Index, 2>& shape, auto op, auto reduced, auto output) {
            for (Index h = 0; h < shape[0]; ++h) {
                auto local = reduced;
                for (Index w = 0; w < shape[1]; ++w)
                    interface::init(op, local, h, w);
                interface::final(op, local, output, h);
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
              nt::tuple_of_accessor_value<std::decay_t<Reduced>>)
    NOA_NOINLINE constexpr void reduce_axes_iwise(
        const Shape<Index, N>& input_shape,
        const Shape<Index, N>& output_shape,
        Op&& op,
        Reduced&& reduced,
        Output&& output,
        i32 n_threads = 1
    ) {
        const auto axes_to_reduce = input_shape.cmp_ne(output_shape);
        check((axes_to_reduce and output_shape.cmp_ne(1)) == false,
              "Dimensions should match the input shape, or be 1, "
              "indicating the dimension should be reduced to one element. "
              "Got shape input={}, output={}", input_shape, output_shape);
        check(axes_to_reduce.any_eq(true),
              "No reduction to compute. Got shape input={}, output={}. Please use iwise instead.",
              input_shape, output_shape);

        const auto axes_empty_or_to_reduce = output_shape.cmp_eq(1) or axes_to_reduce;
        if (axes_empty_or_to_reduce == true) { // reduce to a single value
            constexpr auto config = nd::AccessorConfig<1>{.enforce_contiguous = true, .filter = {0}};
            auto output_1d = nd::reconfig_accessors<config>(output);
            return reduce_iwise<Config>(
                input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
        }

        using reduce_axes_iwise_t = details::ReduceAxesIwise<Config::zip_reduced, Config::zip_output>;
        const auto shape = input_shape.template as<isize>();
        const auto n_batches = clamp_cast<i32>(shape[0]);
        const isize n_elements_to_reduce = input_shape.template as<isize>().n_elements();
        const bool is_small = n_elements_to_reduce <= Config::n_elements_per_thread;

        if constexpr (N == 4) {
            if (axes_empty_or_to_reduce.pop_front() == true) { // reduce to one value per batch
                auto output_1d = nd::reconfig_accessors<nd::AccessorConfig<1>{.filter = {0}}>(output);
                if (is_small or n_threads <= 1) {
                    reduce_axes_iwise_t::serial_4d(input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d);
                } else if (n_batches < n_threads) {
                    reduce_axes_iwise_t::template parallel_4d<reduce_axes_iwise_t::ParallelReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
                } else {
                    reduce_axes_iwise_t::template parallel_4d<reduce_axes_iwise_t::SerialReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
                }
            } else {
                if (axes_to_reduce[3]) {
                    auto output_3d = nd::reconfig_accessors<nd::AccessorConfig<3>{.filter = {0, 1, 2}}>(output);
                    reduce_axes_iwise_t::template single_axis<3>(
                        input_shape, std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_3d, n_threads);
                } else if (axes_to_reduce[2]) {
                    auto output_3d = nd::reconfig_accessors<nd::AccessorConfig<3>{.filter = {0, 1, 3}}>(output);
                    reduce_axes_iwise_t::template single_axis<2>(
                        input_shape.filter(0, 1, 3, 2), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_3d, n_threads);
                } else if (axes_to_reduce[1]) {
                    auto output_3d = nd::reconfig_accessors<nd::AccessorConfig<3>{.filter = {0, 2, 3}}>(output);
                    reduce_axes_iwise_t::template single_axis<1>(
                        input_shape.filter(0, 2, 3, 1), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_3d, n_threads);
                } else {
                    auto output_3d = nd::reconfig_accessors<nd::AccessorConfig<3>{.filter = {1, 2, 3}}>(output);
                    reduce_axes_iwise_t::template single_axis<0>(
                        input_shape.filter(1, 2, 3, 0), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_3d, n_threads);
                }
            }
        } else if constexpr (N == 3) {
            if (axes_empty_or_to_reduce.pop_front() == true) { // reduce to one value per batch
                auto output_1d = nd::reconfig_accessors<nd::AccessorConfig<1>{.filter={0}}>(output);
                if (is_small or n_threads <= 1) {
                    reduce_axes_iwise_t::serial_3d(input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d);
                } else if (n_batches < n_threads) {
                    reduce_axes_iwise_t::template parallel_3d<reduce_axes_iwise_t::ParallelReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
                } else {
                    reduce_axes_iwise_t::template parallel_3d<reduce_axes_iwise_t::SerialReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
                }
            } else {
                if (axes_to_reduce[2]) {
                    auto output_2d = nd::reconfig_accessors<nd::AccessorConfig<2>{.filter = {0, 1}}>(output);
                    reduce_axes_iwise_t::template single_axis<2>(
                        input_shape, std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_2d, n_threads);
                } else if (axes_to_reduce[1]) {
                    auto output_2d = nd::reconfig_accessors<nd::AccessorConfig<2>{.filter = {0, 2}}>(output);
                    reduce_axes_iwise_t::template single_axis<1>(
                        input_shape.filter(0, 2, 1), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_2d, n_threads);
                } else {
                    auto output_2d = nd::reconfig_accessors<nd::AccessorConfig<2>{.filter = {1, 2}}>(output);
                    reduce_axes_iwise_t::template single_axis<0>(
                        input_shape.filter(1, 2, 0), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_2d, n_threads);
                }
            }
        } else if constexpr (N == 2) {
            if (axes_to_reduce[1]) {
                auto output_1d = nd::reconfig_accessors<nd::AccessorConfig<1>{.filter = {0}}>(output);
                if (is_small or n_threads <= 1) {
                    reduce_axes_iwise_t::serial_2d(input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d);
                } else if (n_batches < n_threads) {
                    reduce_axes_iwise_t::template parallel_2d<reduce_axes_iwise_t::ParallelReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
                } else {
                    reduce_axes_iwise_t::template parallel_2d<reduce_axes_iwise_t::SerialReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
                }
            } else {
                auto output_1d = nd::reconfig_accessors<nd::AccessorConfig<1>{.filter = {1}}>(output);
                reduce_axes_iwise_t::template single_axis<0>(
                    input_shape.filter(1, 0), std::forward<Op>(op),
                    std::forward<Reduced>(reduced), output_1d, n_threads);
            }
        }
    }
}
