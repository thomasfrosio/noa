#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/utils/Interfaces.hpp"
#include "noa/gpu/cuda/AllocatorDevice.hpp"
#include "noa/gpu/cuda/kernels/ReduceEwise.cuh"

namespace noa::cuda::guts {
    template<typename Config,
             typename Op, typename Index,
             typename Input, typename Reduced, typename Output>
    void launch_reduce_ewise_small_2d(
            Op&& op,
            Input&& input,
            Reduced&& reduced,
            Output&& output,
            Shape2<Index> shape,
            Stream& stream
    ) {
        using op_t = std::decay_t<Op>;
        using reduced_t = std::decay_t<Reduced>;
        using output_t = std::decay_t<Output>;

        const u32 vector_size = maximum_vector_size(
                input, Config::n_elements_per_thread, Config::block_size, Shape3<Index>{shape[0], 1, 1});

        const auto launch_config = LaunchConfig{
            .n_blocks=static_cast<u32>(shape[0]),
            .n_threads=Config::block_size
        };

        if (vector_size > 1) {
            constexpr auto to_contiguous_2d = ng::AccessorConfig<2>{
                    .enforce_contiguous=true,
                    .enforce_restrict=false,
                    .filter={0, 3},
            };
            auto input_2d = ng::reconfig_accessors<to_contiguous_2d>(std::forward<Input>(input));

            if (vector_size == 8) {
                using kernel_config = ReduceEwise2dConfig<Config, 8, true>;
                stream.enqueue(
                        reduce_ewise_2d<kernel_config, op_t, Index, decltype(input_2d), reduced_t, output_t>,
                        launch_config, std::forward<Op>(op), std::move(input_2d), shape[1],
                        std::forward<Reduced>(reduced), std::forward<Output>(output)
                );
            } else if (vector_size == 4) {
                using kernel_config = ReduceEwise2dConfig<Config, 4, true>;
                stream.enqueue(
                        reduce_ewise_2d<kernel_config, op_t, Index, decltype(input_2d), reduced_t, output_t>,
                        launch_config, std::forward<Op>(op), std::move(input_2d), shape[1],
                        std::forward<Reduced>(reduced), std::forward<Output>(output)
                );
            } else {
                using kernel_config = ReduceEwise2dConfig<Config, 2, true>;
                stream.enqueue(
                        reduce_ewise_2d<kernel_config, op_t, Index, decltype(input_2d), reduced_t, output_t>,
                        launch_config, std::forward<Op>(op), std::move(input_2d), shape[1],
                        std::forward<Reduced>(reduced), std::forward<Output>(output)
                );
            }
        } else {
            constexpr auto to_2d = ng::AccessorConfig<2>{
                    .enforce_contiguous=false,
                    .enforce_restrict=false,
                    .filter={0, 3},
            };
            auto input_2d = ng::reconfig_accessors<to_2d>(std::forward<Input>(input));
            using kernel_config = ReduceEwise2dConfig<Config, 1, true>;
            stream.enqueue(
                    reduce_ewise_2d<kernel_config, op_t, Index, decltype(input_2d), reduced_t, output_t>,
                    launch_config, std::forward<Op>(op), std::move(input_2d), shape[1],
                    std::forward<Reduced>(reduced), std::forward<Output>(output)
            );
        }
    }

    template<typename Config,
             typename Input, typename Reduced, typename Output,
             typename Index, typename Op>
    void launch_reduce_ewise_small_4d(
            Op&& op,
            Input&& input,
            Reduced&& reduced,
            Output&& output,
            Shape4<Index> shape,
            bool is_per_batch,
            Stream& stream
    ) {
        // In this config, the input cannot be easily interpreted as a 1d array.
        // As such, the 3 outermost dimensions are batched in a set of rows. Each block reduces at least one row.
        // If the innermost dimension is contiguous, blocks can use vectorize loads to read their row(s).
        using op_t = std::decay_t<Op>;
        using input_t = std::decay_t<Input>;
        using reduced_t = std::decay_t<Reduced>;
        using output_t = std::decay_t<Output>;

        // Compute the grid and block dimensions.
        constexpr u32 n_threads_x = Constant::WARP_SIZE; // TODO better heuristic?
        const u32 n_threads_y = max(Config::block_size / n_threads_x, u32{1});
        const auto n_rows = shape[2] * shape[1] * (is_per_batch ? 1 : shape[0]);
        const u32 n_blocks_x = 1; // one block to reduce n_rows
        const u32 n_blocks_y = is_per_batch ? static_cast<u32>(shape[0]) : 1;
        const auto first_launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x, n_blocks_y),
                .n_threads=dim3(n_threads_x, n_threads_y),
        };

        // Compute the vector size for "input".
        const u32 input_vector_size = maximum_vector_size(
                input, Config::n_elements_per_thread, Config::block_size, shape.pop_back());

        const auto shape_dhw = shape.pop_front();
        if (input_vector_size > 1) {
            constexpr auto to_contiguous = ng::AccessorConfig<0>{.enforce_contiguous=true};
            auto contig_input = ng::reconfig_accessors<to_contiguous>(std::forward<Input>(input));
            using contig_input_t = decltype(contig_input);

            if (input_vector_size == 2) {
                using kernel_config = ReduceEwise4dConfig<Config, n_threads_x, 2, true>;
                stream.enqueue(
                        reduce_ewise_4d<kernel_config, op_t, Index, contig_input_t, reduced_t, output_t>,
                        first_launch_config, std::forward<Op>(op), std::move(contig_input),
                        std::forward<Reduced>(reduced), shape_dhw, n_rows, output
                );
            } else if (input_vector_size == 4) {
                using kernel_config = ReduceEwise4dConfig<Config, n_threads_x, 4, true>;
                stream.enqueue(
                        reduce_ewise_4d<kernel_config, op_t, Index, contig_input_t, reduced_t, output_t>,
                        first_launch_config, std::forward<Op>(op), std::move(contig_input),
                        std::forward<Reduced>(reduced), shape_dhw, n_rows, output
                );
            } else { // clamp to 8
                using kernel_config = ReduceEwise4dConfig<Config, n_threads_x, 8, true>;
                stream.enqueue(
                        reduce_ewise_4d<kernel_config, op_t, Index, contig_input_t, reduced_t, output_t>,
                        first_launch_config, std::forward<Op>(op), std::move(contig_input),
                        std::forward<Reduced>(reduced), shape_dhw, n_rows, output
                );
            }
        } else {
            using kernel_config = ReduceEwise4dConfig<Config, n_threads_x, 1, true>;
            stream.enqueue(
                    reduce_ewise_4d<kernel_config, op_t, Index, input_t, reduced_t, output_t>,
                    first_launch_config, std::forward<Op>(op), std::forward<Input>(input),
                    std::forward<Reduced>(reduced), shape_dhw, n_rows, output
            );
        }
    }

    template<typename Joined>
    constexpr u32 get_joined_vector_size() {
        u32 joined_vector_size{16};
        auto get_vector_size = [&]<typename A>(std::type_identity<A>) {
            using value_t = typename A::value_type;
            constexpr u32 i_vector_size =
                    is_power_of_2(sizeof(value_t)) ?
                    clamp(u32{16 / sizeof(value_t)}, u32{1}, u32{8}) : 1u;
            joined_vector_size = min(joined_vector_size, i_vector_size);
        };

        [&]<typename...T>(nt::TypeList<T...>) {
            (get_vector_size(std::type_identity<T>{}), ...);
        }(nt::type_list_t<Joined>{});

        return joined_vector_size;
    }

    // Allocate the joined buffers and set the accessors.
    template<typename Joined>
    auto get_joined_buffer(u32 n_blocks_x, u32 n_blocks_y, Joined& joined, u32 vector_size, Stream& stream) {
        return joined.map([&]<typename A>(A& accessor) {
            const u32 pitch = next_multiple_of(n_blocks_x, vector_size);
            auto buffer = AllocatorDevice<typename A::value_type>::allocate_async(pitch * n_blocks_y, stream);
            accessor = A(buffer.get(), Strides2<typename A::index_type>{pitch, 1});
            return buffer;
        });
    }

    template<typename Config,
             typename Op, typename Index,
             typename Input, typename Reduced, typename Output>
    void launch_reduce_ewise_large_2d(
            Op&& op,
            Input&& input,
            Reduced&& reduced,
            Output&& output,
            Shape2<Index> shape,
            Stream& stream
    ) {
        // In this config, the inputs can be interpreted as 1d arrays. If the innermost dimension is contiguous,
        // i.e. if all elements to reduce are contiguous, we can vectorize loads for the first kernel.
        using op_t = std::decay_t<Op>;
        using reduced_t = std::decay_t<Reduced>;
        using output_t = std::decay_t<Output>;

        // Launch config:
        // Limit the number of blocks, otherwise the second kernel would have too much work to do.
        auto n_blocks_x = static_cast<u32>(divide_up(shape[1], static_cast<Index>(Config::block_work_size)));
        n_blocks_x = min(n_blocks_x, Config::max_grid_size);
        const auto n_blocks_y = static_cast<u32>(shape[0]);
        const auto first_launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x, n_blocks_y),
                .n_threads=Config::block_size
        };

        using joined_t = joined_tuple_t<2, Index, Reduced>;
        joined_t joined; // Tuple<AccessorRestrictContiguous<T, 2, Index>,...>
        constexpr u32 joined_vector_size = get_joined_vector_size<joined_t>();
        [[maybe_unused]] auto joined_buffer = get_joined_buffer(
                n_blocks_x, n_blocks_y, joined, joined_vector_size, stream);

        // Compute the vector size for "input".
        const u32 input_vector_size = maximum_vector_size(
                input, Config::n_elements_per_thread, Config::block_size, Shape3<Index>{shape[0], 1, 1});

        // First kernel.
        if (Config::n_elements_per_thread > 1 and input_vector_size > 1) {
            constexpr auto to_contiguous_2d = ng::AccessorConfig<2>{
                    .enforce_contiguous=true,
                    .enforce_restrict=false,
                    .filter={0, 3},
            };
            auto input_2d = ng::reconfig_accessors<to_contiguous_2d>(std::forward<Input>(input));
            if (input_vector_size == 2) {
                using kernel_config = ReduceEwise2dConfig<Config, 4>;
                stream.enqueue(
                        reduce_ewise_2d<kernel_config, op_t, Index, decltype(input_2d), reduced_t, joined_t>,
                        first_launch_config, op, std::move(input_2d), shape[1], reduced, joined
                );
            } else if (input_vector_size == 4) {
                using kernel_config = ReduceEwise2dConfig<Config, 4>;
                stream.enqueue(
                        reduce_ewise_2d<kernel_config, op_t, Index, decltype(input_2d), reduced_t, joined_t>,
                        first_launch_config, op, std::move(input_2d), shape[1], reduced, joined
                );
            } else if (input_vector_size == 8) {
                using kernel_config = ReduceEwise2dConfig<Config, 8>;
                stream.enqueue(
                        reduce_ewise_2d<kernel_config, op_t, Index, decltype(input_2d), reduced_t, joined_t>,
                        first_launch_config, op, std::move(input_2d), shape[1], reduced, joined
                );
            } else if (input_vector_size == 16) {
                using kernel_config = ReduceEwise2dConfig<Config, 16>;
                stream.enqueue(
                        reduce_ewise_2d<kernel_config, op_t, Index, decltype(input_2d), reduced_t, joined_t>,
                        first_launch_config, op, std::move(input_2d), shape[1], reduced, joined
                );
            }
        } else {
            constexpr auto to_2d = ng::AccessorConfig<2>{
                    .enforce_contiguous=false,
                    .enforce_restrict=false,
                    .filter={0, 3},
            };
            auto input_2d = ng::reconfig_accessors<to_2d>(std::forward<Input>(input));
            using kernel_config = ReduceEwise2dConfig<Config, 1>;
            stream.enqueue(
                    reduce_ewise_2d<kernel_config, op_t, Index, decltype(input_2d), reduced_t, joined_t>,
                    first_launch_config, op, std::move(input_2d), shape[1], reduced, joined
            );
        }

        // Second kernel.
        using kernel_config = ReduceEwise2dConfig<Config, joined_vector_size>;
        const auto second_launch_config = LaunchConfig{.n_blocks=n_blocks_y, .n_threads=Config::block_size};
        stream.enqueue(
                reduce_ewise_second<kernel_config, op_t, Index, joined_t, reduced_t, output_t>, second_launch_config,
                std::forward<Op>(op), joined, n_blocks_x, std::forward<Reduced>(reduced), std::forward<Output>(output)
        );
    }

    template<typename Config,
             typename Input, typename Reduced, typename Output,
             typename Index, typename Op>
    void launch_reduce_ewise_large_4d(
            Op&& op,
            Input&& input,
            Reduced&& reduced,
            Output&& output,
            Shape4<Index> shape,
            bool is_per_batch,
            Stream& stream
    ) {
        // In this config, the input cannot be easily interpreted as a 1d array.
        // As such, the 3 outermost dimensions are batched in a set of rows. Each block reduces at least one row.
        // If the innermost dimension is contiguous, blocks can use vectorize loads to read their row(s).
        using op_t = std::decay_t<Op>;
        using input_t = std::decay_t<Input>;
        using reduced_t = std::decay_t<Reduced>;
        using output_t = std::decay_t<Output>;

        // Compute the grid and block dimensions.
        const u32 n_threads_x = shape[3] > 512 ? 256 : 64; // TODO better heuristic?
        const u32 n_threads_y = max(Config::block_size / n_threads_x, u32{1});
        const auto n_rows = shape[2] * shape[1] * (is_per_batch ? 1 : shape[0]);
        const u32 n_blocks_x = min(static_cast<u32>(divide_up(n_rows, static_cast<Index>(n_threads_y))), Config::max_grid_size);
        const u32 n_blocks_y = is_per_batch ? static_cast<u32>(shape[0]) : 1;
        const auto first_launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x, n_blocks_y),
                .n_threads=dim3(n_threads_x, n_threads_y),
        };

        using joined_t = joined_tuple_t<2, Index, Reduced>;
        joined_t joined; // Tuple<AccessorRestrictContiguous<T, 2, Index>,...>
        constexpr u32 joined_vector_size = get_joined_vector_size<joined_t>();
        [[maybe_unused]] auto joined_buffer = get_joined_buffer(
                n_blocks_x, n_blocks_y, joined, joined_vector_size, stream);

        // Compute the vector size for "input".
        const u32 input_vector_size = maximum_vector_size(
                input, Config::n_elements_per_thread, Config::block_size, shape.pop_back());

        const auto shape_dhw = shape.pop_front();
        if (input_vector_size > 1) {
            constexpr auto to_contiguous = ng::AccessorConfig<0>{.enforce_contiguous=true};
            auto contig_input = ng::reconfig_accessors<to_contiguous>(std::forward<Input>(input));
            using contig_input_t = decltype(contig_input);

            // TODO Reduce the number of instantiations by adding support of a runtime block_size_x.
            //      That would mean using dynamic shared memory in block_reduce, which is fine...
            if (n_threads_x == 256) {
                if (input_vector_size == 2) {
                    using kernel_config = ReduceEwise4dConfig<Config, 256, 2>;
                    stream.enqueue(
                            reduce_ewise_4d<kernel_config, op_t, Index, contig_input_t, reduced_t, joined_t>,
                            first_launch_config, op, std::move(contig_input), reduced, shape_dhw, n_rows, joined
                    );
                } else if (input_vector_size == 4) {
                    using kernel_config = ReduceEwise4dConfig<Config, 256, 4>;
                    stream.enqueue(
                            reduce_ewise_4d<kernel_config, op_t, Index, contig_input_t, reduced_t, joined_t>,
                            first_launch_config, op, std::move(contig_input), reduced, shape_dhw, n_rows, joined
                    );
                } else { // clamp to 8
                    using kernel_config = ReduceEwise4dConfig<Config, 256, 8>;
                    stream.enqueue(
                            reduce_ewise_4d<kernel_config, op_t, Index, contig_input_t, reduced_t, joined_t>,
                            first_launch_config, op, std::move(contig_input), reduced, shape_dhw, n_rows, joined
                    );
                }
            } else {
                if (input_vector_size == 2) {
                    using kernel_config = ReduceEwise4dConfig<Config, 64, 2>;
                    stream.enqueue(
                            reduce_ewise_4d<kernel_config, op_t, Index, contig_input_t, reduced_t, joined_t>,
                            first_launch_config, op, std::move(contig_input), reduced, shape_dhw, n_rows, joined
                    );
                } else if (input_vector_size == 4) {
                    using kernel_config = ReduceEwise4dConfig<Config, 64, 4>;
                    stream.enqueue(
                            reduce_ewise_4d<kernel_config, op_t, Index, contig_input_t, reduced_t, joined_t>,
                            first_launch_config, op, std::move(contig_input), reduced, shape_dhw, n_rows, joined
                    );
                } else { // clamp to 8
                    using kernel_config = ReduceEwise4dConfig<Config, 64, 8>;
                    stream.enqueue(
                            reduce_ewise_4d<kernel_config, op_t, Index, contig_input_t, reduced_t, joined_t>,
                            first_launch_config, op, std::move(contig_input), reduced, shape_dhw, n_rows, joined
                    );
                }
            }
        } else {
            if (n_threads_x == 256) {
                using kernel_config = ReduceEwise4dConfig<Config, 256, 1>;
                stream.enqueue(
                        reduce_ewise_4d<kernel_config, op_t, Index, input_t, reduced_t, joined_t>,
                        first_launch_config, op, std::forward<Input>(input), reduced, shape_dhw, n_rows, joined
                );
            } else {
                using kernel_config = ReduceEwise4dConfig<Config, 64, 1>;
                stream.enqueue(
                        reduce_ewise_4d<kernel_config, op_t, Index, input_t, reduced_t, joined_t>,
                        first_launch_config, op, std::forward<Input>(input), reduced, shape_dhw, n_rows, joined
                );
            }
        }

        // Second kernel.
        using kernel_config = ReduceEwise2dConfig<Config, joined_vector_size>;
        const auto second_launch_config = LaunchConfig{.n_blocks=n_blocks_y, .n_threads=Config::block_size};
        stream.enqueue(
                reduce_ewise_second<kernel_config, op_t, Index, joined_t, reduced_t, output_t>, second_launch_config,
                std::forward<Op>(op), joined, n_blocks_x, std::forward<Reduced>(reduced), std::forward<Output>(output)
        );
    }
}

namespace noa::cuda {
    template<typename Config = ReduceEwiseConfig<>,
             typename Input, typename Reduced, typename Output, typename Index, typename Op>
    requires (nt::is_tuple_of_accessor_v<Input> and
              nt::is_tuple_of_accessor_ndim_v<4, Input> and
              not nt::is_tuple_of_accessor_value_v<Input> and // at least one varray
              nt::is_tuple_of_accessor_pure_v<Output> and
              nt::is_tuple_of_accessor_ndim_v<1, Output> and
              nt::is_tuple_of_accessor_value_v<Reduced>)
    void reduce_ewise(
            const Shape4<Index>& shape,
            Op&& op,
            Input&& input,
            Reduced&& reduced,
            Output& output,
            Stream& stream
    ) {
        const auto elements = safe_cast<Index>(shape.template as<i64>().elements());
        const Vec4<bool> is_contiguous = ni::is_contiguous(input, shape);

        constexpr auto SMALL_THRESHOLD = Config::n_elements_per_thread * Config::block_size * 4;
        if (all(is_contiguous.pop_back())) {
            if (elements <= SMALL_THRESHOLD) {
                guts::launch_reduce_ewise_small_2d<Config>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), Shape2<Index>{1, elements}, stream);
            } else {
                guts::launch_reduce_ewise_large_2d<Config>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), Shape2<Index>{1, elements}, stream);
            }
        } else {
            if (elements <= SMALL_THRESHOLD) {
                guts::launch_reduce_ewise_small_4d<Config>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), shape, false, stream);
            } else {
                guts::launch_reduce_ewise_large_4d<Config>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), shape, false, stream);
            }
        }
    }
}
