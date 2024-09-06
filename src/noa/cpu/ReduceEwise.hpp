#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/Types.hpp"
#include "noa/core/Interfaces.hpp"

namespace noa::cpu::guts {
    template<bool ZipInput, bool ZipReduced, bool ZipOutput>
    class ReduceEwise {
    public:
        using interface = ng::ReduceEwiseInterface<ZipInput, ZipReduced, ZipOutput>;

        template<typename Op, typename Input, typename Reduced, typename Output, typename Index, size_t N>
        static void parallel(
            const Shape<Index, N>& shape, Op op,
            Input input, Reduced reduced, Output& output, i64 n_threads
        ) {
            #pragma omp parallel default(none) num_threads(n_threads) shared(shape, input, reduced) firstprivate(op)
            {
                // Local copy of the reduced values.
                auto local_reduce = reduced;

                if constexpr (N == 4) {
                    #pragma omp for collapse(4)
                    for (Index i = 0; i < shape[0]; ++i)
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                for (Index l = 0; l < shape[3]; ++l)
                                    interface::init(op, input, local_reduce, i, j, k, l);
                } else if constexpr (N == 1) {
                    #pragma omp for collapse(1)
                    for (Index i = 0; i < shape[0]; ++i)
                        interface::init(op, input, local_reduce, i);
                } else {
                    static_assert(nt::always_false<Op>);
                }

                #pragma omp critical
                {
                    // Join the reduced values of each thread together.
                    interface::join(op, local_reduce, reduced);
                }
            }
            interface::final(op, reduced, output, 0);
        }

        template<typename Op, typename Input, typename Reduced, typename Output, typename Index, size_t N>
        static void serial(
            const Shape<Index, N>& shape, Op op,
            Input input, Reduced reduced, Output& output
        ) {
            if constexpr (N == 4) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            for (Index l = 0; l < shape[3]; ++l)
                                interface::init(op, input, reduced, i, j, k, l);
            } else if constexpr (N == 1) {
                for (Index i = 0; i < shape[0]; ++i)
                    interface::init(op, input, reduced, i);
            } else {
                static_assert(nt::always_false<Op>);
            }

            interface::final(op, reduced, output, 0);
        }
    };
}

namespace noa::cpu {
    template<bool ZipInput = false, bool ZipReduced = false, bool ZipOutput = false, i64 ElementsPerThread = 1'048'576>
    struct ReduceEwiseConfig {
        static constexpr bool zip_input = ZipInput;
        static constexpr bool zip_reduced = ZipReduced;
        static constexpr bool zip_output = ZipOutput;
        static constexpr i64 n_elements_per_thread = ElementsPerThread;
    };

    template<typename Config = ReduceEwiseConfig<>,
             typename Input, typename Reduced, typename Output, typename Index, typename Op>
    requires (nt::are_tuple_of_accessor_v<std::decay_t<Input>, Output> and
              not nt::tuple_of_accessor_value<std::decay_t<Input>> and // at least one varray
              nt::tuple_of_accessor_value<std::decay_t<Reduced>>)
    void reduce_ewise(
        const Shape4<Index>& shape,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output& output,
        i64 n_threads = 1
    ) {
        // Check contiguity.
        const bool are_all_contiguous = ni::are_contiguous(input, shape);
        const i64 n_elements = shape.template as<i64>().n_elements();
        i64 actual_n_threads = n_elements <= Config::n_elements_per_thread ? 1 : n_threads;
        if (actual_n_threads > 1)
            actual_n_threads = min(n_threads, n_elements / Config::n_elements_per_thread);

        using reduce_ewise_core = guts::ReduceEwise<Config::zip_input, Config::zip_reduced, Config::zip_output>;

        // FIXME We could try collapse contiguous dimensions to still have a contiguous loop.
        // FIXME In most cases, the inputs are not expected to be aliases of each other, so only
        //       optimise for the 1d-contig restrict case? remove 1d-contig non-restrict case
        if (are_all_contiguous) {
            auto shape_1d = Shape1<Index>::from_value(n_elements);
            if (ng::are_accessors_aliased(input, output)) {
                constexpr auto contiguous_1d = ng::AccessorConfig<1>{
                    .enforce_contiguous = true,
                    .enforce_restrict = false,
                    .filter = {3},
                };
                auto input_1d = ng::reconfig_accessors<contiguous_1d>(std::forward<Input>(input));
                if (actual_n_threads > 1) {
                    reduce_ewise_core::parallel(
                        shape_1d,
                        std::forward<Op>(op),
                        std::move(input_1d),
                        std::forward<Reduced>(reduced),
                        output, actual_n_threads);
                } else {
                    reduce_ewise_core::serial(
                        shape_1d,
                        std::forward<Op>(op),
                        std::move(input_1d),
                        std::forward<Reduced>(reduced),
                        output);
                }
            } else {
                constexpr auto contiguous_restrict_1d = ng::AccessorConfig<1>{
                    .enforce_contiguous = true,
                    .enforce_restrict = true,
                    .filter = {3},
                };
                auto input_1d = ng::reconfig_accessors<contiguous_restrict_1d>(std::forward<Input>(input));
                if (actual_n_threads > 1) {
                    reduce_ewise_core::parallel(
                        shape_1d,
                        std::forward<Op>(op),
                        std::move(input_1d),
                        std::forward<Reduced>(reduced),
                        output, actual_n_threads);
                } else {
                    reduce_ewise_core::serial(
                        shape_1d,
                        std::forward<Op>(op),
                        std::move(input_1d),
                        std::forward<Reduced>(reduced),
                        output);
                }
            }
        } else {
            if (actual_n_threads > 1) {
                reduce_ewise_core::parallel(
                    shape,
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    output, actual_n_threads);
            } else {
                reduce_ewise_core::serial(
                    shape,
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    output);
            }
        }
    }
}
#endif
