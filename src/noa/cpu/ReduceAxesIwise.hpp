#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/Interfaces.hpp"
#include "noa/cpu/ReduceIwise.hpp"

namespace noa::cpu::guts {
    template<bool ZipReduced, bool ZipOutput>
    class ReduceAxesIwise {
    public:
        using interface = ng::ReduceIwiseInterface<ZipReduced, ZipOutput>;

        template<size_t R, size_t N, typename Index, typename Op>
        static void single_axis(
            const Shape<Index, N>& shape,
            Op op, auto reduced, auto output,
            i64 n_threads
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
        static void parallel_4d(const Shape4<Index>& shape, auto op, auto reduced, auto output, i64 n_threads) {
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
        static constexpr void serial_4d(const Shape4<Index>& shape, auto op, auto reduced, auto output) {
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
        static void parallel_3d(const Shape3<Index>& shape, auto op, auto reduced, auto output, i64 n_threads) {
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
        static constexpr void serial_3d(const Shape3<Index>& shape, auto op, auto reduced, auto output) {
            for (Index d = 0; d < shape[0]; ++d) {
                auto local = reduced;
                for (Index h = 0; h < shape[1]; ++h)
                    for (Index w = 0; w < shape[2]; ++w)
                        interface::init(op, local, d, h, w);
                interface::final(op, local, output, d);
            }
        }

        template<ReductionMode MODE, typename Index>
        static void parallel_2d(const Shape2<Index>& shape, auto op, auto reduced, auto output, i64 n_threads) {
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
        static constexpr void serial_2d(const Shape2<Index>& shape, auto op, auto reduced, auto output) {
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
    template<bool ZipReduced = false, bool ZipOutput = false, i64 ElementsPerThread = 1'048'576>
    struct ReduceAxesIwiseConfig {
        static constexpr bool zip_reduced = ZipReduced;
        static constexpr bool zip_output = ZipOutput;
        static constexpr i64 n_elements_per_thread = ElementsPerThread;
    };

    template<typename Config = ReduceAxesIwiseConfig<>,
            typename Op, size_t N, typename Reduced, typename Output, typename Index>
    requires (nt::tuple_of_accessor_pure<std::decay_t<Output>> and
              nt::tuple_of_accessor_nd<std::decay_t<Output>, N> and
              nt::tuple_of_accessor_value<std::decay_t<Reduced>>)
    constexpr void reduce_axes_iwise(
        const Shape<Index, N>& input_shape,
        const Shape<Index, N>& output_shape,
        Op&& op,
        Reduced&& reduced,
        Output&& output,
        i64 n_threads = 1
    ) {
        const Vec<bool, N> axes_to_reduce = input_shape != output_shape;
        if (any(axes_to_reduce and (output_shape != 1))) {
            panic("Dimensions should match the input shape, or be 1, "
                  "indicating the dimension should be reduced to one element. "
                  "Got shape input={}, output={}", input_shape, output_shape);
        } else if (all(axes_to_reduce == false)) {
            panic("No reduction to compute. Got shape input={}, output={}. Please use iwise instead.",
                  input_shape, output_shape);
        }

        const auto axes_empty_or_to_reduce = output_shape == 1 or axes_to_reduce;
        if (all(axes_empty_or_to_reduce)) { // reduce to a single value
            constexpr auto config = ng::AccessorConfig<1>{.enforce_contiguous = true, .filter = {0}};
            auto output_1d = ng::reconfig_accessors<config>(output);
            return reduce_iwise<Config>(
                input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
        }

        using reducer = guts::ReduceAxesIwise<Config::zip_reduced, Config::zip_output>;
        const auto shape = input_shape.template as<i64>();
        const auto n_batches = shape[0];
        const i64 n_elements_to_reduce = input_shape.template as<i64>().n_elements();
        const bool is_small = n_elements_to_reduce <= Config::n_elements_per_thread;

        if constexpr (N == 4) {
            if (all(axes_empty_or_to_reduce.pop_front())) { // reduce to one value per batch
                auto output_1d = ng::reconfig_accessors<ng::AccessorConfig<1>{.filter = {0}}>(output);
                if (is_small or n_threads <= 1) {
                    reducer::serial_4d(input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d);
                } else if (n_batches < n_threads) {
                    reducer::template parallel_4d<reducer::ParallelReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
                } else {
                    reducer::template parallel_4d<reducer::SerialReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
                }
            } else {
                if (axes_to_reduce[3]) {
                    auto output_3d = ng::reconfig_accessors<ng::AccessorConfig<3>{.filter = {0, 1, 2}}>(output);
                    reducer::template single_axis<3>(
                        input_shape, std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_3d, n_threads);
                } else if (axes_to_reduce[2]) {
                    auto output_3d = ng::reconfig_accessors<ng::AccessorConfig<3>{.filter = {0, 1, 3}}>(output);
                    reducer::template single_axis<2>(
                        input_shape.filter(0, 1, 3, 2), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_3d, n_threads);
                } else if (axes_to_reduce[1]) {
                    auto output_3d = ng::reconfig_accessors<ng::AccessorConfig<3>{.filter = {0, 2, 3}}>(output);
                    reducer::template single_axis<1>(
                        input_shape.filter(0, 2, 3, 1), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_3d, n_threads);
                } else {
                    auto output_3d = ng::reconfig_accessors<ng::AccessorConfig<3>{.filter = {1, 2, 3}}>(output);
                    reducer::template single_axis<0>(
                        input_shape.filter(1, 2, 3, 0), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_3d, n_threads);
                }
            }
        } else if constexpr (N == 3) {
            if (all(axes_empty_or_to_reduce.pop_front())) { // reduce to one value per batch
                auto output_1d = ng::reconfig_accessors<ng::AccessorConfig<1>{.filter={0}}>(output);
                if (is_small or n_threads <= 1) {
                    reducer::serial_3d(input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d);
                } else if (n_batches < n_threads) {
                    reducer::template parallel_3d<reducer::ParallelReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
                } else {
                    reducer::template parallel_3d<reducer::SerialReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
                }
            } else {
                if (axes_to_reduce[2]) {
                    auto output_2d = ng::reconfig_accessors<ng::AccessorConfig<2>{.filter = {0, 1}}>(output);
                    reducer::template single_axis<2>(
                        input_shape, std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_2d, n_threads);
                } else if (axes_to_reduce[1]) {
                    auto output_2d = ng::reconfig_accessors<ng::AccessorConfig<2>{.filter = {0, 2}}>(output);
                    reducer::template single_axis<1>(
                        input_shape.filter(0, 2, 1), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_2d, n_threads);
                } else {
                    auto output_2d = ng::reconfig_accessors<ng::AccessorConfig<2>{.filter = {1, 2}}>(output);
                    reducer::template single_axis<0>(
                        input_shape.filter(1, 2, 0), std::forward<Op>(op),
                        std::forward<Reduced>(reduced), output_2d, n_threads);
                }
            }
        } else if constexpr (N == 2) {
            if (axes_to_reduce[1]) {
                auto output_1d = ng::reconfig_accessors<ng::AccessorConfig<1>{.filter = {0}}>(output);
                if (is_small or n_threads <= 1) {
                    reducer::serial_2d(input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d);
                } else if (n_batches < n_threads) {
                    reducer::template parallel_2d<reducer::ParallelReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
                } else {
                    reducer::template parallel_2d<reducer::SerialReduction>(
                        input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, n_threads);
                }
            } else {
                auto output_1d = ng::reconfig_accessors<ng::AccessorConfig<1>{.filter = {1}}>(output);
                reducer::template single_axis<0>(
                    input_shape.filter(1, 0), std::forward<Op>(op),
                    std::forward<Reduced>(reduced), output_1d, n_threads);
            }
        }
    }
}
#endif
