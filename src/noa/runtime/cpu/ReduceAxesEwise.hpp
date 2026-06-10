#pragma once

#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/cpu/ReduceEwise.hpp"
#include "noa/runtime/cpu/ComputeHandle.hpp"

namespace noa::cpu::details {
    template<bool ZipInput, bool ZipReduced, bool ZipOutput>
    class ReduceAxesEwise {
    public:
        using interface = nd::ReduceEwiseInterface<ZipInput, ZipReduced, ZipOutput>;

        template<i32 MODE, typename Op, typename Input, typename Reduced, typename Output, typename Index, usize N>
        NOA_NOINLINE static void run(
            const Shape<Index, N>& shape, Op op,
            Input input, Reduced reduced, Output output, i32 threads
        ) {
            auto original_reduced = reduced;
            #pragma omp parallel default(none) num_threads(threads) shared(shape, input, reduced, output, original_reduced) firstprivate(op)
            {
                constexpr auto ci = ComputeHandle<Index, true>{};
                if constexpr (MODE == 3) {
                    // Parallel reduction.
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
                                                interface::call(ci, op, input, local, i, j, k, l, m, n);
                        } else if constexpr (N == 5) {
                            #pragma omp for collapse(4)
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    for (Index l = 0; l < shape[3]; ++l)
                                        for (Index m = 0; m < shape[4]; ++m)
                                            interface::call(ci, op, input, local, i, j, k, l, m);
                        } else if constexpr (N == 4) {
                            #pragma omp for collapse(3)
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    for (Index l = 0; l < shape[3]; ++l)
                                        interface::call(ci, op, input, local, i, j, k, l);
                        } else if constexpr (N == 3) {
                            #pragma omp for collapse(2)
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    interface::call(ci, op, input, local, i, j, k);
                        } else if constexpr (N == 2) {
                            #pragma omp for
                            for (Index j = 0; j < shape[1]; ++j)
                                interface::call(ci, op, input, local, i, j);
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
                } else if constexpr (MODE == 2) {
                    // Batch reduction.
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
                                                interface::call(ci, op, input, local, i, j, k, l, m, n);
                        } else if constexpr (N == 5) {
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    for (Index l = 0; l < shape[3]; ++l)
                                        for (Index m = 0; m < shape[4]; ++m)
                                            interface::call(ci, op, input, local, i, j, k, l, m);
                        } else if constexpr (N == 4) {
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    for (Index l = 0; l < shape[3]; ++l)
                                        interface::call(ci, op, input, local, i, j, k, l);
                        } else if constexpr (N == 3) {
                            for (Index j = 0; j < shape[1]; ++j)
                                for (Index k = 0; k < shape[2]; ++k)
                                    interface::call(ci, op, input, local, i, j, k);
                        } else if constexpr (N == 2) {
                            for (Index j = 0; j < shape[1]; ++j)
                                interface::call(ci, op, input, local, i, j);
                        }
                        interface::deinit(ci, op, i);
                        interface::post(op, local, output, i);
                    }
                } else if constexpr (MODE == 1) {
                    // Single axis reduction.
                    if constexpr (N == 6) {
                        #pragma omp for collapse(5)
                        for (Index i = 0; i < shape[0]; ++i) {
                            for (Index j = 0; j < shape[1]; ++j) {
                                for (Index k = 0; k < shape[2]; ++k) {
                                    for (Index l = 0; l < shape[3]; ++l) {
                                        for (Index m = 0; m < shape[4]; ++m) {
                                            interface::init(ci, op, i, j, k, l, m);
                                            auto local = reduced;
                                            for (Index n = 0; n < shape[5]; ++n)
                                                interface::call(ci, op, input, local, i, j, k, l, m, n);
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
                                        for (Index m = 0; m < shape[4]; ++m)
                                            interface::call(ci, op, input, local, i, j, k, l, m);
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
                                    for (Index l = 0; l < shape[3]; ++l)
                                        interface::call(ci, op, input, local, i, j, k, l);
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
                                for (Index k = 0; k < shape[2]; ++k)
                                    interface::call(ci, op, input, local, i, j, k);
                                interface::deinit(ci, op, i, j);
                                interface::post(op, local, output, i, j);
                            }
                        }
                    } else if constexpr (N == 2) {
                        #pragma omp for
                        for (Index i = 0; i < shape[0]; ++i) {
                            interface::init(ci, op, i);
                            auto local = reduced;
                            for (Index j = 0; j < shape[1]; ++j)
                                interface::call(ci, op, input, local, i, j);
                            interface::deinit(ci, op, i);
                            interface::post(op, local, output, i);
                        }
                    }
                } else {
                    static_assert(nt::always_false<Op>);
                }
            }
        }
    };
}

namespace noa::cpu {
    template<bool ZipInput = false, bool ZipReduced = false, bool ZipOutput = false, isize ElementsPerThread = 1'048'576>
    struct ReduceAxesEwiseConfig {
        static constexpr bool zip_input = ZipInput;
        static constexpr bool zip_reduced = ZipReduced;
        static constexpr bool zip_output = ZipOutput;
        static constexpr isize n_elements_per_thread = ElementsPerThread;
    };

    template<typename Config = ReduceAxesEwiseConfig<>,
             typename Op, typename Input, typename Reduced, typename Output, typename Index, usize N>
    requires (nt::tuple_of_accessor_nd<std::decay_t<Input>, N> and
              not nt::tuple_of_accessor_value<std::decay_t<Input>> and // at least one varray
              nt::tuple_of_accessor_pure_nd<std::decay_t<Output>, N> and
              nt::tuple_of_accessor_value_or_empty<std::decay_t<Reduced>>)
    NOA_NOINLINE constexpr void reduce_axes_ewise(
        const Shape<Index, N>& input_shape,
        const Shape<Index, N>& output_shape,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        i32 n_threads = 1
    ) {
        using reduce_axes_ewise_t = details::ReduceAxesEwise<Config::zip_input, Config::zip_reduced, Config::zip_output>;
        const auto axes_to_reduce = input_shape.cmp_ne(output_shape);
        const auto axes_empty_or_to_reduce = output_shape.cmp_eq(1) or axes_to_reduce;

        // Reduce all dimensions into a single value.
        if (axes_empty_or_to_reduce == true) {
            // The output has a single value, so its stride doesn't matter, and taking any axis works.
            auto output_1d = nd::reconfig_accessors<nd::AccessorConfig{.enforce_contiguous = true}>(output, 0);
            reduce_ewise(
                input_shape,
                std::forward<Op>(op),
                std::forward<Input>(input),
                std::forward<Reduced>(reduced),
                output_1d, n_threads
            );
            return;
        }
        NOA_ASSERT(N != 1);

        if constexpr (N >= 2) {
            // Find the first non-empty axis.
            usize first_non_empty{N - 1};
            for (usize i = 0; i < N - 1; ++i) {
                if (input_shape[i] > 1) {
                    first_non_empty = i;
                    break;
                }
            }

            // Batch reduction is when all axes after first non-empty axis are empty or reduced.
            bool batched_reduction{true};
            for (usize i = first_non_empty + 1; i < N; ++i)
                if (not axes_empty_or_to_reduce[i])
                    batched_reduction = false;

            if (batched_reduction) {
                const auto n_reductions = output_shape[first_non_empty];
                const auto actual_n_threads = min(clamp_cast<i32>(n_reductions), n_threads);
                isize n_elements_per_reduction{1};
                for (usize i = first_non_empty + 1; i < N; ++i)
                    n_elements_per_reduction *= static_cast<isize>(input_shape[i]);

                auto output_1d = nd::reconfig_accessors(std::forward<Output>(output), first_non_empty);
                const bool parallel_reduction =
                    n_elements_per_reduction > Config::n_elements_per_thread and
                    clamp_cast<i32>(n_reductions) < n_threads;

                // Optimize for 2d case.
                const bool is_2d = first_non_empty == N - 2;
                const auto shape_2d = Shape<Index, 2>{n_reductions, n_elements_per_reduction};
                constexpr auto CONTIGUOUS_RESTRICT_2D = nd::AccessorConfig<2>{
                    .enforce_contiguous = true,
                    .enforce_restrict = true,
                };

                // SAFETY: If the operator has enabled vectorization, this function de facto
                // assumes that the none of inputs and outputs are not aliasing.
                const bool is_restrict =
                    nt::enable_vectorization_v<Op> or
                    not nd::are_accessors_aliased(input, output);

                if (parallel_reduction) {
                    if (is_2d and is_restrict) {
                        auto input_2d = nd::reconfig_accessors<CONTIGUOUS_RESTRICT_2D>(
                            std::forward<Input>(input), N - 2, N - 1);
                        reduce_axes_ewise_t::template run<3>(
                            shape_2d,
                            std::forward<Op>(op),
                            std::move(input_2d),
                            std::forward<Reduced>(reduced),
                            std::move(output_1d),
                            actual_n_threads);
                    } else {
                        reduce_axes_ewise_t::template run<3>(
                            input_shape,
                            std::forward<Op>(op),
                            std::forward<Input>(input),
                            std::forward<Reduced>(reduced),
                            std::move(output_1d),
                            actual_n_threads);
                    }
                } else {
                    if (is_2d and is_restrict) {
                        auto input_2d = nd::reconfig_accessors<CONTIGUOUS_RESTRICT_2D>(
                            std::forward<Input>(input), N - 2, N - 1);
                        reduce_axes_ewise_t::template run<2>(
                            shape_2d,
                            std::forward<Op>(op),
                            std::move(input_2d),
                            std::forward<Reduced>(reduced),
                            std::move(output_1d),
                            actual_n_threads);
                    } else {
                        reduce_axes_ewise_t::template run<2>(
                            input_shape,
                            std::forward<Op>(op),
                            std::forward<Input>(input),
                            std::forward<Reduced>(reduced),
                            std::move(output_1d),
                            actual_n_threads);
                    }
                }
                return;
            }
            NOA_ASSERT(sum(axes_to_reduce.template as<i32>()) == 1);

            // Single axis reduction.
            // First copy|move the input and output since they'll need to be reordered.
            auto input_ = std::forward<Input>(input);
            auto output_ = std::forward<Output>(output);

            // Move the reduced dimension to the rightmost dimension (width).
            const auto order = noa::squeeze_empty_dimensions_left(axes_to_reduce.template as<i32>() + 1);
            auto reordered_shape = input_shape.permute(order);
            if (order != Vec<i32, N>::arange())
                nd::permute_accessors(order, input_, output_);

            // Exclude the reduced width from the output.
            auto output_nd = nd::reconfig_accessors(std::move(output_), Vec<usize, N - 1>::arange());

            // Single-threaded reductions while distributing reductions across threads.
            const auto n_reductions = reordered_shape.pop_back().template as<isize>().n_elements();
            const auto actual_n_threads = min(clamp_cast<i32>(n_reductions), n_threads);
            reduce_axes_ewise_t::template run<1>(
                reordered_shape,
                std::forward<Op>(op),
                std::move(input_),
                std::forward<Reduced>(reduced),
                std::move(output_nd),
                actual_n_threads
            );
        }
    }
}
