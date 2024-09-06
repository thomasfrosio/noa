#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Interfaces.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/gpu/cuda/Allocators.hpp"
#include "noa/gpu/cuda/kernels/Block.cuh"

// These reduction kernels are adapted from different sources, but the main logic comes from:
//  - https://github.com/NVIDIA/cuda-samples/tree/master/Samples/reduction
//  - https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

namespace noa::cuda::guts {
    // 1d grid of 1d blocks.
    // Each block writes one element in "joined", which should thus have as many elements as there are blocks.
    template<typename Config, u32 VectorSize, bool IsFinal = false>
    struct ReduceEwise2dConfig {
        using interface = Config::interface;
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = Config::block_size;
        static constexpr u32 vector_size = VectorSize;
        static constexpr u32 n_elements_per_thread = max(VectorSize, Config::n_elements_per_thread);
        static constexpr u32 block_work_size = block_size * n_elements_per_thread;
        static constexpr bool is_final = IsFinal;
    };

    template<typename Config, u32 BlockSizeX, u32 VectorSize, bool IsFinal = false>
    struct ReduceEwise4dConfig {
        using interface = Config::interface;
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = Config::block_size;
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = max(block_size / block_size_x, 1u);
        static constexpr u32 vector_size_x = VectorSize;
        static constexpr u32 n_elements_per_thread_x = max(vector_size_x, Config::n_elements_per_thread);
        static constexpr u32 block_work_size_x = block_size_x * n_elements_per_thread_x;
        static constexpr bool is_final = IsFinal;
    };

    // Reduce element-wise 1d or 2d input accessors.
    // 2d grid (y is the batch) of 1d blocks.
    template<typename Config, typename Op, typename Index, typename Input, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_ewise_2d(Op op, Input input, Index n_elements_per_batch, Reduced reduced, Output output) {
        const Index batch = blockIdx.y;
        const Index bid = blockIdx.x;
        const Index tid = threadIdx.x;
        const Index starting_index = Config::block_work_size * bid;
        const Index grid_work_size = Config::block_work_size * gridDim.x;

        auto input_1d = std::move(input).map([batch]<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T>) {
                return std::forward<T>(accessor);
            } else if constexpr (nt::is_accessor_nd_v<T, 2>) {
                // Offset the input accessor to the current batch,
                // so that it can be used later to reset the pointer.
                accessor.offset_inplace(batch);
                return accessor[0]; // 1d AccessorReference
            } else {
                static_assert(nt::always_false<T>);
            }
        });

        for (Index cid = starting_index; cid < n_elements_per_batch; cid += grid_work_size) {
            input_1d.for_each_enumerate([&input, cid]<size_t I>(auto& accessor) {
                accessor.reset_pointer(input[Tag<I>{}].get());
                accessor.offset_inplace(cid);
            });
            block_reduce_ewise_1d_init
                <Config::block_size, Config::n_elements_per_thread, Config::vector_size, Config::interface>
                (op, input_1d, n_elements_per_batch - cid, reduced, tid);
        }

        if constexpr (Config::is_final) {
            // There's one block per batch, so compute the reduced value for the block
            // and save it in the output at the batch index.
            block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, batch);
        } else {
            // The output is the "joined" buffer, which is a buffer with one value per block and per batch.
            // These values will then be reduced by the second reduction kernel (see below).
            block_reduce_join<Config::interface, Config::block_size>(op, reduced, output, tid, batch, bid);
        }
    }

    // Here the input is organized has a series of rows. Given the original DHW shape of the input and the row index,
    // we can derive the BDH indices. Each dimension can have an arbitrary stride, but if the rows themselves are
    // contiguous (if the W stride is 1), then vectorized load/stores can be used to load/store elements from the rows.
    //
    // This kernel explicitly supports per-batch reductions (see reduce_axes_ewise), in which case the grid should be
    // 2d (gridDim.x is the number of blocks to reduce the rows of a given batch and gridDim.y is the number of batches)
    // and joined should be 2d Accessors, where the outer dimension is the batch.
    template<typename Config, typename Op, typename Index,
             typename Input, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_ewise_4d(
        Op op,
        Input input,
        Reduced reduced,
        Shape3<Index> shape_dhw,
        Index n_rows,
        Output output
    ) {
        const Index batch = blockIdx.y;
        const Index bid = blockIdx.x;
        const Index n_blocks_per_batch = gridDim.x;
        const Index n_rows_per_block = blockDim.y;
        const Index n_rows_per_grid = n_blocks_per_batch * n_rows_per_block;
        const Index initial_row = bid * n_rows_per_block + threadIdx.y;

        auto input_1d = std::move(input).map([batch]<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T>) {
                return std::forward<T>(accessor);
            } else {
                // Offset the input accessor to the current batch,
                // so that it can be used later to reset the pointer.
                accessor.offset_inplace(batch);
                return accessor[0][0][0]; // 1d AccessorReference
            }
        });

        for (Index row = initial_row; row < n_rows; row += n_rows_per_grid) { // for every row (within a batch)
            // If there batched are fused (gridDim.y==0), bdh[0] is always 0.
            Vec3<Index> bdh = ni::offset2index(row, shape_dhw[0], shape_dhw[1]);

            for (Index cid = 0; cid < shape_dhw[2]; cid += Config::block_work_size_x) { // consume the row
                input_1d.for_each_enumerate([&input, &bdh, &cid]<size_t I>(auto& accessor_1d) {
                    if constexpr (not nt::is_accessor_value_v<decltype(accessor_1d)>) {
                        auto& accessor = input[Tag<I>{}];
                        auto new_pointer = accessor.offset_pointer(accessor.get(), bdh[0], bdh[1], bdh[2], cid);
                        accessor_1d.reset_pointer(new_pointer);
                    }
                });
                block_reduce_ewise_1d_init
                    <Config::block_size_x, Config::n_elements_per_thread_x, Config::vector_size_x, Config::interface>
                    (op, input_1d, shape_dhw[2] - cid, reduced, static_cast<Index>(threadIdx.x));
            }
        }

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        if constexpr (Config::is_final)
            block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, batch);
        else
            block_reduce_join<Config::interface, Config::block_size>(op, reduced, output, tid, batch, bid);
    }

    // One 1d block per batch to finish joining the reduced values and compute the final output.
    template<typename Config, typename Op, typename Index,
             typename Joined, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_ewise_second(
        Op op,
        Joined joined, // Tuple of 2d Accessor(s) corresponding to Reduced
        Index n_elements,
        Reduced reduced, // Tuple of AccessorValue(s)
        Output output // Tuple of 1d Accessor(s)
    ) {
        const Index batch = blockIdx.x;
        const Index tid = threadIdx.x;

        auto joined_1d = joined.map([&](auto& accessor) { return accessor[batch]; });
        for (Index cid = 0; cid < n_elements; cid += Config::block_work_size) {
            block_reduce_ewise_1d_join
                <Config::block_size, Config::n_elements_per_thread, Config::vector_size, Config::interface>
                (op, joined_1d, n_elements - cid, reduced, tid);
            joined_1d.for_each([](auto& accessor) { accessor.offset_inplace(Config::block_work_size); });
        }

        block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, batch);
    }
}

#ifdef NOA_IS_OFFLINE
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

        u32 vector_size{1};
        if constexpr ((nt::has_allow_vectorization_v<Op> or ng::are_accessors_const<Input>())
                      and std::decay_t<Input>::SIZE <= 4) {
            vector_size = maximum_vector_size(
                input, Config::n_elements_per_thread, Config::block_size, Shape3<Index>{shape[0], 1, 1});
        }

        const u32 n_blocks_x = 1; // one block to reduce shape[1]
        const u32 n_blocks_y = static_cast<u32>(shape[0]);
        const auto launch_config = LaunchConfig{
            .n_blocks = dim3(n_blocks_x, n_blocks_y),
            .n_threads = Config::block_size
        };

        if (vector_size > 1) {
            constexpr auto to_contiguous_2d = ng::AccessorConfig<2>{
                .enforce_contiguous = true,
                .enforce_restrict = false,
                .filter = {0, 3},
            };
            auto input_2d = ng::reconfig_accessors<to_contiguous_2d>(std::forward<Input>(input));

            if (vector_size == 2) {
                using kernel_config = ReduceEwise2dConfig<Config, 2, true>;
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
                using kernel_config = ReduceEwise2dConfig<Config, 8, true>;
                stream.enqueue(
                    reduce_ewise_2d<kernel_config, op_t, Index, decltype(input_2d), reduced_t, output_t>,
                    launch_config, std::forward<Op>(op), std::move(input_2d), shape[1],
                    std::forward<Reduced>(reduced), std::forward<Output>(output)
                );
            }
        } else {
            constexpr auto to_2d = ng::AccessorConfig<2>{
                .enforce_contiguous = false,
                .enforce_restrict = false,
                .filter = {0, 3},
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
        constexpr u32 n_threads_x = Constant::WARP_SIZE;
        const u32 n_threads_y = max(Config::block_size / n_threads_x, u32{1});
        const auto n_rows = shape[2] * shape[1] * (is_per_batch ? 1 : shape[0]);
        const u32 n_blocks_x = 1; // one block to reduce n_rows
        const u32 n_blocks_y = is_per_batch ? static_cast<u32>(shape[0]) : 1;
        const auto launch_config = LaunchConfig{
            .n_blocks = dim3(n_blocks_x, n_blocks_y),
            .n_threads = dim3(n_threads_x, n_threads_y),
        };

        u32 input_vector_size{1};
        if constexpr ((nt::has_allow_vectorization_v<Op> or ng::are_accessors_const<Input>()) and input_t::SIZE <= 4) {
            input_vector_size = maximum_vector_size(
                input, Config::n_elements_per_thread, Config::block_size, shape.pop_back());
        }

        const auto shape_dhw = shape.pop_front();
        if (input_vector_size > 1) {
            constexpr auto to_contiguous = ng::AccessorConfig<0>{.enforce_contiguous = true};
            auto contig_input = ng::reconfig_accessors<to_contiguous>(std::forward<Input>(input));
            using contig_input_t = decltype(contig_input);

            if (input_vector_size == 2) {
                using kernel_config = ReduceEwise4dConfig<Config, n_threads_x, 2, true>;
                stream.enqueue(
                    reduce_ewise_4d<kernel_config, op_t, Index, contig_input_t, reduced_t, output_t>,
                    launch_config, std::forward<Op>(op), std::move(contig_input),
                    std::forward<Reduced>(reduced), shape_dhw, n_rows, output
                );
            } else if (input_vector_size == 4) {
                using kernel_config = ReduceEwise4dConfig<Config, n_threads_x, 4, true>;
                stream.enqueue(
                    reduce_ewise_4d<kernel_config, op_t, Index, contig_input_t, reduced_t, output_t>,
                    launch_config, std::forward<Op>(op), std::move(contig_input),
                    std::forward<Reduced>(reduced), shape_dhw, n_rows, output
                );
            } else {
                using kernel_config = ReduceEwise4dConfig<Config, n_threads_x, 8, true>;
                stream.enqueue(
                    reduce_ewise_4d<kernel_config, op_t, Index, contig_input_t, reduced_t, output_t>,
                    launch_config, std::forward<Op>(op), std::move(contig_input),
                    std::forward<Reduced>(reduced), shape_dhw, n_rows, output
                );
            }
        } else {
            using kernel_config = ReduceEwise4dConfig<Config, n_threads_x, 1, true>;
            stream.enqueue(
                reduce_ewise_4d<kernel_config, op_t, Index, input_t, reduced_t, output_t>,
                launch_config, std::forward<Op>(op), std::forward<Input>(input),
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
            .n_blocks = dim3(n_blocks_x, n_blocks_y),
            .n_threads = Config::block_size
        };

        using joined_t = joined_tuple_t<2, Index, Reduced>;
        joined_t joined; // Tuple<AccessorRestrictContiguous<T, 2, Index>,...>
        constexpr u32 joined_vector_size = get_joined_vector_size<joined_t>();
        [[maybe_unused]] auto joined_buffer = get_joined_buffer(
            n_blocks_x, n_blocks_y, joined, joined_vector_size, stream);

        u32 input_vector_size{1};
        if constexpr ((nt::has_allow_vectorization_v<Op> or ng::are_accessors_const<Input>())
                      and std::decay_t<Input>::SIZE <= 4) {
            input_vector_size = maximum_vector_size(
                input, Config::n_elements_per_thread, Config::block_size, Shape3<Index>{shape[0], 1, 1});
        }

        // First kernel.
        if (Config::n_elements_per_thread > 1 and input_vector_size > 1) {
            constexpr auto to_contiguous_2d = ng::AccessorConfig<2>{
                .enforce_contiguous = true,
                .enforce_restrict = false,
                .filter = {0, 3},
            };
            auto input_2d = ng::reconfig_accessors<to_contiguous_2d>(std::forward<Input>(input));
            if (input_vector_size == 2) {
                using kernel_config = ReduceEwise2dConfig<Config, 2>;
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
            } else {
                using kernel_config = ReduceEwise2dConfig<Config, 8>;
                stream.enqueue(
                    reduce_ewise_2d<kernel_config, op_t, Index, decltype(input_2d), reduced_t, joined_t>,
                    first_launch_config, op, std::move(input_2d), shape[1], reduced, joined
                );
            }
        } else {
            constexpr auto to_2d = ng::AccessorConfig<2>{
                .enforce_contiguous = false,
                .enforce_restrict = false,
                .filter = {0, 3},
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
        const auto second_launch_config = LaunchConfig{.n_blocks = n_blocks_y, .n_threads = Config::block_size};
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
        u32 n_threads_x = shape[3] > 512 ? 256 : 64; // TODO better heuristic?
        if (not is_multiple_of(Config::block_size, n_threads_x))
            n_threads_x = Constant::WARP_SIZE;
        const u32 n_threads_y = max(Config::block_size / n_threads_x, u32{1});
        const auto n_rows = shape[2] * shape[1] * (is_per_batch ? 1 : shape[0]);
        const u32 n_blocks_x = min(static_cast<u32>(divide_up(n_rows, static_cast<Index>(n_threads_y))), Config::max_grid_size);
        const u32 n_blocks_y = is_per_batch ? static_cast<u32>(shape[0]) : 1;
        const auto first_launch_config = LaunchConfig{
            .n_blocks = dim3(n_blocks_x, n_blocks_y),
            .n_threads = dim3(n_threads_x, n_threads_y),
        };

        using joined_t = joined_tuple_t<2, Index, Reduced>;
        joined_t joined; // Tuple<AccessorRestrictContiguous<T, 2, Index>,...>
        constexpr u32 joined_vector_size = get_joined_vector_size<joined_t>();
        [[maybe_unused]] auto joined_buffer = get_joined_buffer(
            n_blocks_x, n_blocks_y, joined, joined_vector_size, stream);

        // Compute the vector size for "input".
        u32 input_vector_size{1};
        if constexpr ((nt::has_allow_vectorization_v<Op> or ng::are_accessors_const<Input>()) and input_t::SIZE <= 4) {
            input_vector_size = maximum_vector_size(
                input, Config::n_elements_per_thread, Config::block_size, shape.pop_back());
        }

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
        const auto second_launch_config = LaunchConfig{.n_blocks = n_blocks_y, .n_threads = Config::block_size};
        stream.enqueue(
            reduce_ewise_second<kernel_config, op_t, Index, joined_t, reduced_t, output_t>, second_launch_config,
            std::forward<Op>(op), joined, n_blocks_x, std::forward<Reduced>(reduced), std::forward<Output>(output)
        );
    }
}

namespace noa::cuda {
    template<bool ZipInput = false,
             bool ZipReduced = false,
             bool ZipOutput = false,
             u32 ElementsPerThread = 8,
             u32 BlockSize = 512,
             u32 MaxGridSize = 4096>
    struct ReduceEwiseConfig {
        static_assert(is_multiple_of(BlockSize, Constant::WARP_SIZE) and BlockSize <= Limits::MAX_THREADS);
        using interface = ng::ReduceEwiseInterface<ZipInput, ZipReduced, ZipOutput>;
        static constexpr u32 max_grid_size = MaxGridSize;
        static constexpr u32 block_size = BlockSize;
        static constexpr u32 n_elements_per_thread = ElementsPerThread;
        static constexpr u32 block_work_size = block_size * n_elements_per_thread;
    };

    template<typename Config = ReduceEwiseConfig<>,
             typename Input, typename Reduced, typename Output, typename Index, typename Op>
    requires (nt::tuple_of_accessor<Input> and
              nt::tuple_of_accessor_nd<Input, 4> and
              not nt::tuple_of_accessor_value<Input> and // at least one varray
              nt::tuple_of_accessor_pure<Output> and
              nt::tuple_of_accessor_nd<Output, 1> and
              nt::tuple_of_accessor_value<Reduced>)
    void reduce_ewise(
        const Shape4<Index>& shape,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream
    ) {
        const auto n_elements = safe_cast<Index>(shape.template as<i64>().elements());
        const Vec4<bool> is_contiguous = ni::is_contiguous(input, shape);

        constexpr auto SMALL_THRESHOLD = Config::n_elements_per_thread * Config::block_size * 4;
        if (all(is_contiguous.pop_back())) {
            if (n_elements <= SMALL_THRESHOLD) {
                guts::launch_reduce_ewise_small_2d<Config>(
                    std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), Shape2<Index>{1, n_elements}, stream);
            } else {
                guts::launch_reduce_ewise_large_2d<Config>(
                    std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), Shape2<Index>{1, n_elements}, stream);
            }
        } else {
            if (n_elements <= SMALL_THRESHOLD) {
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
#endif
