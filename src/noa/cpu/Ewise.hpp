#pragma once

#include <omp.h>
#include "noa/core/Interfaces.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/indexing/Layout.hpp"
#include "noa/core/types/Accessor.hpp"

namespace noa::cpu::guts {
    template<bool ZipInput, bool ZipOutput>
    class Ewise {
    public:
        using interface = ng::EwiseInterface<ZipInput, ZipOutput>;

        // Take the input and output by value, a reference will be passed to each thread.
        // Take the operator by reference since a copy will be passed to each thread.
        template<size_t N, typename Index, typename Op, typename Input, typename Output>
        NOA_NOINLINE static void parallel(const Shape<Index, N>& shape, Op op, Input input, Output output, i64 n_threads) {
            #pragma omp parallel default(none) num_threads(n_threads) shared(shape, input, output) firstprivate(op)
            {
                interface::init(op, omp_get_thread_num());

                if constexpr (N == 4) {
                    #pragma omp for collapse(4)
                    for (Index i = 0; i < shape[0]; ++i)
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                for (Index l = 0; l < shape[3]; ++l)
                                    interface::call(op, input, output, i, j, k, l);

                } else if constexpr (N == 1) {
                    #pragma omp for collapse(1)
                    for (Index i = 0; i < shape[0]; ++i)
                        interface::call(op, input, output, i);

                } else {
                    static_assert(nt::always_false<Op>);
                }

                interface::final(op, omp_get_thread_num());
            }
        }

        template<size_t N, typename Index, typename Op, typename Input, typename Output>
        NOA_NOINLINE static constexpr void serial(const Shape<Index, N>& shape, Op op, Input input, Output output) {
            interface::init(op, 0);

            if constexpr (N == 4) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            for (Index l = 0; l < shape[3]; ++l)
                                interface::call(op, input, output, i, j, k, l);

            } else if constexpr (N == 1) {
                for (Index i = 0; i < shape[0]; ++i)
                    interface::call(op, input, output, i);

            } else {
                static_assert(nt::always_false<Op>);
            }

            interface::final(op, 0);
        }
    };
}

namespace noa::cpu {
    template<bool ZipInput = false, bool ZipOutput = false, i64 ElementsPerThread = 1'048'576>
    struct EwiseConfig {
        static constexpr bool zip_input = ZipInput;
        static constexpr bool zip_output = ZipOutput;
        static constexpr i64 n_elements_per_thread = ElementsPerThread;
    };

    template<typename Config = EwiseConfig<>,
             typename Input, typename Output, typename Index, typename Op>
    requires (nt::tuple_of_accessor_nd_or_empty<std::decay_t<Input>, 4> and
              nt::tuple_of_accessor_pure_nd_or_empty<std::decay_t<Output>, 4>)
    constexpr void ewise(
        const Shape4<Index>& shape,
        Op&& op,
        Input&& input,
        Output&& output,
        i64 n_threads = 1
    ) {
        // Check contiguity.
        // TODO We could try collapse contiguous dimensions first.
        const bool are_all_contiguous =
            ni::are_contiguous(input, shape) and
            ni::are_contiguous(output, shape);

        const i64 elements = shape.template as<i64>().n_elements();
        i64 actual_n_threads = elements <= Config::n_elements_per_thread ? 1 : n_threads;
        if (actual_n_threads > 1)
            actual_n_threads = min(n_threads, elements / Config::n_elements_per_thread);

        using ewise_t = guts::Ewise<Config::zip_input, Config::zip_output>;

        if (are_all_contiguous) {
            auto shape_1d = Shape1<Index>{shape.n_elements()};
            if (not nt::enable_vectorization_v<Op> and ng::are_accessors_aliased(input, output)) {
                constexpr auto accessor_config_1d = ng::AccessorConfig<1>{
                    .enforce_contiguous=true,
                    .enforce_restrict=false,
                    .filter={3},
                };
                auto input_1d = ng::reconfig_accessors<accessor_config_1d>(std::forward<Input>(input));
                auto output_1d = ng::reconfig_accessors<accessor_config_1d>(output);
                if (actual_n_threads > 1)
                    ewise_t::parallel(shape_1d, std::forward<Op>(op), std::move(input_1d), output_1d, actual_n_threads);
                else
                    ewise_t::serial(shape_1d, std::forward<Op>(op), std::move(input_1d), output_1d);
            } else {
                constexpr auto accessor_config_1d = ng::AccessorConfig<1>{
                    .enforce_contiguous=true,
                    .enforce_restrict=true,
                    .filter={3},
                };
                auto input_1d = ng::reconfig_accessors<accessor_config_1d>(std::forward<Input>(input));
                auto output_1d = ng::reconfig_accessors<accessor_config_1d>(output);
                if (actual_n_threads > 1)
                    ewise_t::parallel(shape_1d, std::forward<Op>(op), std::move(input_1d), output_1d, actual_n_threads);
                else
                    ewise_t::serial(shape_1d, std::forward<Op>(op), std::move(input_1d), output_1d);
            }
        } else {
            if (actual_n_threads > 1)
                ewise_t::parallel(shape, std::forward<Op>(op), std::forward<Input>(input), output, actual_n_threads);
            else
                ewise_t::serial(shape, std::forward<Op>(op), std::forward<Input>(input), output);
        }
    }
}
