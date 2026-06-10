#pragma once

#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/cpu/ComputeHandle.hpp"

namespace noa::cpu::details {
    template<bool ZipInput, bool ZipOutput>
    class Ewise {
    public:
        using interface = nd::EwiseInterface<ZipInput, ZipOutput>;

        // Take the input and output by value, a reference will be passed to each thread.
        // Take the operator by reference since a copy will be passed to each thread.
        template<usize N, typename Index, typename Op, typename Input, typename Output>
        NOA_NOINLINE static void parallel(const Shape<Index, N>& shape, Op op, Input input, Output output, i32 n_threads) {
            #pragma omp parallel default(none) num_threads(n_threads) shared(shape, input, output) firstprivate(op)
            {
                constexpr auto ci = ComputeHandle<Index, true>{};
                interface::init(ci, op);

                if constexpr (N == 6) {
                    #pragma omp for collapse(6)
                    for (Index i = 0; i < shape[0]; ++i)
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                for (Index l = 0; l < shape[3]; ++l)
                                    for (Index m = 0; m < shape[4]; ++m)
                                        for (Index n = 0; n < shape[5]; ++n)
                                            interface::call(ci, op, input, output, i, j, k, l, m, n);
                } else if constexpr (N == 5) {
                    #pragma omp for collapse(5)
                    for (Index i = 0; i < shape[0]; ++i)
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                for (Index l = 0; l < shape[3]; ++l)
                                    for (Index m = 0; m < shape[4]; ++m)
                                        interface::call(ci, op, input, output, i, j, k, l, m);
                } else if constexpr (N == 4) {
                    #pragma omp for collapse(4)
                    for (Index i = 0; i < shape[0]; ++i)
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                for (Index l = 0; l < shape[3]; ++l)
                                    interface::call(ci, op, input, output, i, j, k, l);
                } else if constexpr (N == 3) {
                    #pragma omp for collapse(3)
                    for (Index i = 0; i < shape[0]; ++i)
                        for (Index j = 0; j < shape[1]; ++j)
                            for (Index k = 0; k < shape[2]; ++k)
                                interface::call(ci, op, input, output, i, j, k);
                } else if constexpr (N == 2) {
                    #pragma omp for collapse(2)
                    for (Index i = 0; i < shape[0]; ++i)
                        for (Index j = 0; j < shape[1]; ++j)
                            interface::call(ci, op, input, output, i, j);
                } else if constexpr (N == 1) {
                    #pragma omp for
                    for (Index i = 0; i < shape[0]; ++i)
                        interface::call(ci, op, input, output, i);
                } else {
                    static_assert(nt::always_false<Op>);
                }

                interface::deinit(ci, op);
            }
        }

        template<usize N, typename Index, typename Op, typename Input, typename Output>
        NOA_NOINLINE static constexpr void serial(const Shape<Index, N>& shape, Op op, Input input, Output output) {
            constexpr auto ci = ComputeHandle<Index, true>{};
            interface::init(ci, op);

            if constexpr (N == 6) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            for (Index l = 0; l < shape[3]; ++l)
                                for (Index m = 0; m < shape[4]; ++m)
                                    for (Index n = 0; n < shape[5]; ++n)
                                        interface::call(ci, op, input, output, i, j, k, l, m, n);
            } else if constexpr (N == 5) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            for (Index l = 0; l < shape[3]; ++l)
                                for (Index m = 0; m < shape[4]; ++m)
                                    interface::call(ci, op, input, output, i, j, k, l, m);
            } else if constexpr (N == 4) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            for (Index l = 0; l < shape[3]; ++l)
                                interface::call(ci, op, input, output, i, j, k, l);
            } else if constexpr (N == 3) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            interface::call(ci, op, input, output, i, j, k);
            } else if constexpr (N == 2) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        interface::call(ci, op, input, output, i, j);
            } else if constexpr (N == 1) {
                for (Index i = 0; i < shape[0]; ++i)
                    interface::call(ci, op, input, output, i);
            } else {
                static_assert(nt::always_false<Op>);
            }

            interface::deinit(ci, op);
        }
    };
}

namespace noa::cpu {
    template<bool ZipInput = false, bool ZipOutput = false, isize ElementsPerThread = 1'048'576>
    struct EwiseConfig {
        static constexpr bool zip_input = ZipInput;
        static constexpr bool zip_output = ZipOutput;
        static constexpr isize n_elements_per_thread = ElementsPerThread;
    };

    template<typename Config = EwiseConfig<>,
             typename Input, typename Output, typename Index, typename Op, usize N>
    requires (N >= 1 and
              nt::tuple_of_accessor_nd_or_empty<std::decay_t<Input>, N> and
              nt::tuple_of_accessor_pure_nd_or_empty<std::decay_t<Output>, N>)
    constexpr void ewise(
        const Shape<Index, N>& shape,
        Op&& op,
        Input&& input,
        Output&& output,
        i32 n_threads = 1
    ) {
        // Assume shape is optimized.
        const isize elements = shape.template as<isize>().n_elements();
        i32 actual_n_threads = elements <= Config::n_elements_per_thread ? 1 : n_threads;
        if (actual_n_threads > 1)
            actual_n_threads = min(n_threads, clamp_cast<i32>(elements / Config::n_elements_per_thread));

        using ewise_t = details::Ewise<Config::zip_input, Config::zip_output>;

        if (nd::are_accessors_contiguous(shape, input, output)) {
            auto shape_1d = Shape<Index, 1>{shape.n_elements()};
            if constexpr (nt::enable_vectorization_v<Op>) {
                if (not nd::are_accessors_aliased(input, output)) {
                    constexpr auto accessor_config_1d = nd::AccessorConfig<1>{
                        .enforce_contiguous=true,
                        .enforce_restrict=true,
                    };
                    auto input_1d = nd::reconfig_accessors<accessor_config_1d>(std::forward<Input>(input), N - 1);
                    auto output_1d = nd::reconfig_accessors<accessor_config_1d>(output, N - 1);
                    if (actual_n_threads > 1)
                        ewise_t::parallel(shape_1d, std::forward<Op>(op), std::move(input_1d), output_1d, actual_n_threads);
                    else
                        ewise_t::serial(shape_1d, std::forward<Op>(op), std::move(input_1d), output_1d);
                    return;
                }
            }
            constexpr auto accessor_config_1d = nd::AccessorConfig<1>{
                .enforce_contiguous=true,
                .enforce_restrict=false,
            };
            auto input_1d = nd::reconfig_accessors<accessor_config_1d>(std::forward<Input>(input), N - 1);
            auto output_1d = nd::reconfig_accessors<accessor_config_1d>(output, N - 1);
            if (actual_n_threads > 1)
                ewise_t::parallel(shape_1d, std::forward<Op>(op), std::move(input_1d), output_1d, actual_n_threads);
            else
                ewise_t::serial(shape_1d, std::forward<Op>(op), std::move(input_1d), output_1d);
        } else {
            if (actual_n_threads > 1)
                ewise_t::parallel(shape, std::forward<Op>(op), std::forward<Input>(input), output, actual_n_threads);
            else
                ewise_t::serial(shape, std::forward<Op>(op), std::forward<Input>(input), output);
        }
    }
}
