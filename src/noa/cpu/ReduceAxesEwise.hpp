#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include <omp.h>
#include "noa/core/Types.hpp"
#include "noa/core/utils/Interfaces.hpp"
#include "noa/cpu/ReduceEwise.hpp"

namespace noa::cpu::guts {
    template<bool ZipInput, bool ZipReduced, bool ZipOutput>
    class ReduceAxesEwise {
    public:
        using interface = ng::ReduceEwiseInterface<ZipInput, ZipReduced, ZipOutput>;

        template<i32 MODE, typename Op, typename Input, typename Reduced, typename Output, typename Index, size_t N>
        static void parallel(
                const Shape<Index, N>& shape, Op& op,
                Input input, Reduced reduced, Output output, i64 threads
        ) {
            auto original_reduced = reduced;
            #pragma omp parallel default(none) num_threads(threads) shared(shape, input, reduced, output, original_reduced) firstprivate(op)
            {
                if constexpr (MODE == 3 and N == 4) {
                    // The first 3 rightmost dimensions to reduce contain many elements,
                    // and there are fewer batches than threads.
                    for (Index i = 0; i < shape[0]; ++i) {
                        auto local = reduced;
                        #pragma omp for collapse(3)
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                for (Index l = 0; l < shape[3]; ++l)
                                    interface::init(op, input, local, i, j, k, l);

                        #pragma omp critical
                        interface::join(op, local, reduced);

                        #pragma omp barrier
                        #pragma omp single
                        {
                            interface::final(op, reduced, output, i);
                            reduced = original_reduced;
                        }
                    }
                } else if constexpr (MODE == 3 and N == 2) {
                    // Same as above, but the 3 dimensions to reduce are collapsed
                    // for cases where the inputs are contiguous.
                    for (Index i = 0; i < shape[0]; ++i) {
                        auto local = reduced;
                        #pragma omp for
                        for (Index j = 0; j < shape[1]; ++j)
                            interface::init(op, input, local, i, j);

                        #pragma omp critical
                        interface::join(op, local, reduced);

                        #pragma omp barrier
                        #pragma omp single
                        {
                            interface::final(op, reduced, output, i);
                            reduced = original_reduced;
                        }
                    }
                } else if constexpr (MODE == 2 and N == 4) {
                    // There are more batches than threads, so distribute to one thread per reduction,
                    // thus making the reduction serial.
                    #pragma omp for
                    for (Index i = 0; i < shape[0]; ++i) {
                        auto local = reduced;
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                for (Index l = 0; l < shape[3]; ++l)
                                    interface::init(op, input, local, i, j, k, l);
                        interface::final(op, local, output, i);
                    }
                } else if constexpr (MODE == 2 and N == 2) {
                    // Same as above, but the 3 dimensions to reduce are collapsed
                    // for cases where the inputs are contiguous.
                    #pragma omp for
                    for (Index i = 0; i < shape[0]; ++i) {
                        auto local = reduced;
                        for (Index j = 0; j < shape[1]; ++j)
                            interface::init(op, input, local, i, j);
                        interface::final(op, local, output, i);
                    }
                } else if constexpr (MODE == 1 and N == 4) {
                    // Reduce one axis.
                    // Assuming there are more elements in the 3 leftmost dimensions
                    // than there are threads, so distribute the reductions.
                    #pragma omp for collapse(3)
                    for (Index i = 0; i < shape[0]; ++i) {
                        for (Index j = 0; j < shape[1]; ++j) {
                            for (Index k = 0; k < shape[2]; ++k) {
                                auto local = reduced;
                                for (Index l = 0; l < shape[3]; ++l)
                                    interface::init(op, input, local, i, j, k, l);
                                interface::final(op, local, output, i, j, k);
                            }
                        }
                    }
                } else {
                    static_assert(nt::always_false_v<Op>);
                }
            }
        }

        template<i32 MODE, typename Op, typename Input, typename Reduced, typename Output, typename Index>
        static constexpr void serial(
                const Shape4<Index>& shape, Op op,
                Input input, Reduced reduced, Output output
        ) {
            if constexpr (MODE == 1) {
                for (Index i = 0; i < shape[0]; ++i) {
                    for (Index j = 0; j < shape[1]; ++j) {
                        for (Index k = 0; k < shape[2]; ++k) {
                            auto local = reduced;
                            for (Index l = 0; l < shape[3]; ++l)
                                interface::init(op, input, local, i, j, k, l);
                            interface::final(op, local, output, i, j, k);
                        }
                    }
                }
            } else {
                static_assert(nt::always_false_v<Op>);
            }
        }
    };

    template<typename Index>
    constexpr auto get_reduced_axes(
            const Shape4<Index>& input_shape,
            const Shape4<Index>& output_shape
    ) -> Vec4<bool> {
        const auto axes_to_reduce = input_shape != output_shape;
        if (any(axes_to_reduce and (output_shape != 1))) {
            panic("Dimensions should match the input shape, or be 1, "
                  "indicating the dimension should be reduced to one element. "
                  "Got shape input={}, output={}", input_shape, output_shape);
        }
        return axes_to_reduce;
    }
}

namespace noa::cpu {
    template<bool ZipInput = false, bool ZipReduced = false, bool ZipOutput = false, i64 ParallelThreshold = 1'048'576>
    struct ReduceAxesEwiseConfig {
        static constexpr bool zip_input = ZipInput;
        static constexpr bool zip_reduced = ZipReduced;
        static constexpr bool zip_output = ZipOutput;
        static constexpr i64 parallel_threshold = ParallelThreshold;
    };

    template<typename Config = ReduceAxesEwiseConfig<>,
             typename Op, typename Input, typename Reduced, typename Output, typename Index>
    requires (nt::is_tuple_of_accessor_v<Input> and
              not nt::is_tuple_of_accessor_value_v<Input> and // at least one varray
              nt::is_tuple_of_accessor_pure_v<Output> and
              nt::is_tuple_of_accessor_value_v<Reduced>)
    constexpr void reduce_axes_ewise(
            const Shape4<Index>& input_shape,
            const Shape4<Index>& output_shape,
            Op&& op,
            Input&& input,
            Reduced&& reduced,
            Output&& output,
            i64 n_threads = 1
    )  {
        using reduce_axes_ewise_core = guts::ReduceAxesEwise<Config::zip_input, Config::zip_reduced, Config::zip_output>;
        const Vec axes_to_reduce = guts::get_reduced_axes(input_shape, output_shape);
        const bool are_aliased = ng::are_accessors_aliased(input, output);

        const auto axes_empty_or_to_reduce = output_shape == 1 or axes_to_reduce;
        if (all(axes_empty_or_to_reduce)) { // reduce to a single value
            // Reduce to a single value.
            constexpr auto to_1d = ng::AccessorConfig<1>{.enforce_contiguous=true, .filter={0}};
            auto output_1d = ng::reconfig_accessors<to_1d>(output);
            return reduce_ewise(
                    input_shape,
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    output_1d, n_threads);

        } else if (all(axes_empty_or_to_reduce.pop_front())) { // reduce to one value per batch
            const auto n_batches = output_shape[0];
            const auto n_elements_to_reduce = input_shape.pop_front().template as<i64>().elements();
            const bool are_contiguous = all(ni::is_contiguous(input, input_shape).pop_front());
            const auto actual_n_threads = min(n_batches, n_threads);

            const auto shape_2d = Shape2<Index>{n_batches, n_elements_to_reduce};
            constexpr auto contiguous_restrict_2d = ng::AccessorConfig<2>{
                    .enforce_contiguous=true,
                    .enforce_restrict=true,
                    .filter={0, 3},
            };

            // Extract the batch from the output(s).
            auto output_1d = ng::reconfig_accessors
                    <ng::AccessorConfig<1>{.filter={0}}>
                    (std::forward<Output>(output));

            if (n_elements_to_reduce > Config::parallel_threshold and n_batches < n_threads) {
                if (are_contiguous and not are_aliased) {
                    auto input_2d = ng::reconfig_accessors<contiguous_restrict_2d>(std::forward<Input>(input));
                    reduce_axes_ewise_core::template parallel<3>(
                            shape_2d,
                            op, // taken by lvalue reference
                            std::move(input_2d),
                            std::forward<Reduced>(reduced),
                            std::move(output_1d),
                            actual_n_threads);
                } else {
                    reduce_axes_ewise_core::template parallel<3>(
                            input_shape,
                            op, // taken by lvalue reference
                            std::forward<Input>(input),
                            std::forward<Reduced>(reduced),
                            std::move(output_1d),
                            actual_n_threads);
                }
            } else {
                if (are_contiguous and not are_aliased) {
                    auto input_2d = ng::reconfig_accessors<contiguous_restrict_2d>(std::forward<Input>(input));
                    reduce_axes_ewise_core::template parallel<2>(
                            shape_2d,
                            op, // taken by lvalue reference
                            std::move(input_2d),
                            std::forward<Reduced>(reduced),
                            std::move(output_1d),
                            actual_n_threads);
                } else {
                    reduce_axes_ewise_core::template parallel<2>(
                            input_shape,
                            op, // taken by lvalue reference
                            std::forward<Input>(input),
                            std::forward<Reduced>(reduced),
                            std::move(output_1d),
                            actual_n_threads);
                }
            }
            return;
        }

        const i32 nb_axes_to_reduce = sum(axes_to_reduce.template as<i32>());
        check(nb_axes_to_reduce == 1,
              "Reducing more than one axis at a time is currently limited to a reduction that would "
              "result in one value per batch, i.e. the DHW dimensions should empty after reduction. "
              "Got input_shape={}, output_shape={}, axes_to_reduce={}",
              input_shape, output_shape, axes_to_reduce);

        // First copy|move the input and output since they'll need to be reordered.
        auto input_ = std::forward<Input>(input);
        auto output_ = std::forward<Output>(output);

        // Move the reduced dimension to the rightmost dimension.
        const auto order = ni::squeeze_left(axes_to_reduce.template as<i32>() + 1);
        auto reordered_shape = input_shape.reorder(order);
        if (any(order != Vec{0, 1, 2, 3})) {
            input_.for_each([&order](auto& accessor) { accessor.reorder(order); });
            output_.for_each([&order](auto& accessor) { accessor.reorder(order); });
        }

        auto output_3d = ng::reconfig_accessors
                <ng::AccessorConfig<3>{.filter={0, 1, 2}}>
                (std::move(output_));

        // This function distributes the threads on the dimensions that are not reduced.
        // In other words, the reduction is done by the same thread.
        const i64 n_iterations = reordered_shape.pop_back().template as<i64>().elements();
        const i64 actual_n_threads = n_iterations > 1024 ? n_threads : 1; // TODO Improve this heuristic
        const bool is_contiguous = ni::is_contiguous(input_, reordered_shape)[3];

        if (is_contiguous and not are_aliased) {
            constexpr auto contiguous_restrict = ng::AccessorConfig<0>{
                    .enforce_contiguous=true,
                    .enforce_restrict=true,
            };
            auto contiguous_input = ng::reconfig_accessors<contiguous_restrict>(std::move(input_));
            if (actual_n_threads > 1) {
                reduce_axes_ewise_core::template parallel<1>(
                        reordered_shape,
                        op, // taken by lvalue reference
                        std::move(contiguous_input),
                        std::forward<Reduced>(reduced),
                        std::move(output_3d),
                        actual_n_threads);
            } else {
                reduce_axes_ewise_core::template serial<1>(
                        reordered_shape,
                        std::forward<Op>(op),
                        std::move(contiguous_input),
                        std::forward<Reduced>(reduced),
                        std::move(output_3d));
            }
        } else {
            if (actual_n_threads > 1) {
                reduce_axes_ewise_core::template parallel<1>(
                        reordered_shape,
                        op, // taken by lvalue reference
                        std::move(input_),
                        std::forward<Reduced>(reduced),
                        std::move(output_3d),
                        actual_n_threads);
            } else {
                reduce_axes_ewise_core::template serial<1>(
                        reordered_shape,
                        std::forward<Op>(op),
                        std::move(input_),
                        std::forward<Reduced>(reduced),
                        std::move(output_3d));
            }
        }
    }
}
#endif
