#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/runtime/core/Config.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/core/types/Accessor.hpp"
#include "noa/runtime/core/types/Vec.hpp"
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Block.cuh"

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wduplicated-branches"
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(push, 0)
#endif

// These reduction kernels are adapted from different sources, but the main logic comes from:
//  - https://github.com/NVIDIA/cuda-samples/tree/master/Samples/reduction
//  - https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

namespace noa::cuda::details {
    // 1d grid of 1d blocks.
    // Each block writes one element in "joined", which should thus have as many elements as there are blocks.
    template<typename Config, u32 MaxVectorSize, bool IsFinal = false>
    struct ReduceEwise2dConfig {
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = Config::block_size;
        static constexpr u32 n_elements_per_thread = max(MaxVectorSize, Config::n_elements_per_thread);
        static constexpr u32 block_work_size = block_size * n_elements_per_thread;
        static constexpr bool is_final = IsFinal;
    };

    template<typename Config, u32 BlockSizeX, u32 MaxVectorSize, bool IsFinal = false>
    struct ReduceEwise4dConfig {
        using interface = Config::interface;
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = Config::block_size;
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = max(block_size / block_size_x, 1u);
        static constexpr u32 n_elements_per_thread_x = max(MaxVectorSize, Config::n_elements_per_thread);
        static constexpr u32 block_work_size_x = block_size_x * n_elements_per_thread_x;
        static constexpr bool is_final = IsFinal;
    };

    // Reduce element-wise 1d or 2d input accessors.
    // 2d grid (y is the batch) of 1d blocks.
    template<typename Block, typename Interface, typename Op, typename Index,
             typename Input, typename InputAlignedBuffers, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_ewise_2d(Op op, Input input, Index n_elements_per_batch, Reduced reduced, Output output) {
        const Index batch = blockIdx.y;
        const Index bid = blockIdx.x;
        const Index tid = threadIdx.x;
        const Index starting_index = Block::block_work_size * bid;
        const Index grid_work_size = Block::block_work_size * gridDim.x;

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
                <Block::block_size, Block::n_elements_per_thread, InputAlignedBuffers, Interface>
                (op, input_1d, n_elements_per_batch - cid, reduced, tid);
        }

        if constexpr (Block::is_final) {
            // There's one block per batch, so compute the reduced value for the block
            // and save it in the output at the batch index.
            block_reduce_join_and_final<Interface, Block::block_size>(op, reduced, output, tid, batch);
        } else {
            // The output is the "joined" buffer, which is a buffer with one value per block and per batch.
            // These values will then be reduced by the second reduction kernel (see below).
            block_reduce_join<Interface, Block::block_size>(op, reduced, output, tid, batch, bid);
        }
    }

    // Here the input is organized has a series of rows. Given the original DHW shape of the input and the row index,
    // we can derive the BDH indices. Each dimension can have an arbitrary stride, but if the rows themselves are
    // contiguous (if the W stride is 1), then vectorized load/stores can be used to load/store elements from the rows.
    //
    // This kernel explicitly supports per-batch reductions (see reduce_axes_ewise), in which case the grid should be
    // 2d (gridDim.x is the number of blocks to reduce the rows of a given batch and gridDim.y is the number of batches)
    // and joined should be 2d Accessors, where the outer dimension is the batch.
    template<typename Block, typename Interface, typename Op, typename Index,
             typename Input, typename InputAlignedBuffers, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_ewise_4d(
        Op op,
        Input input,
        Reduced reduced,
        Shape<Index, 3> shape_dhw,
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
            Vec<Index, 3> bdh = offset2index(row, shape_dhw[0], shape_dhw[1]);

            for (Index cid = 0; cid < shape_dhw[2]; cid += Block::block_work_size_x) { // consume the row
                input_1d.for_each_enumerate([&input, &bdh, &cid]<size_t I>(auto& accessor_1d) {
                    if constexpr (not nt::is_accessor_value_v<decltype(accessor_1d)>) {
                        auto& accessor = input[Tag<I>{}];
                        auto new_pointer = accessor.offset_pointer(accessor.get(), bdh[0], bdh[1], bdh[2], cid);
                        accessor_1d.reset_pointer(new_pointer);
                    }
                });
                block_reduce_ewise_1d_init
                    <Block::block_size_x, Block::n_elements_per_thread_x, InputAlignedBuffers, Interface>
                    (op, input_1d, shape_dhw[2] - cid, reduced, static_cast<Index>(threadIdx.x));
            }
        }

        const Index tid = threadIdx.y * Block::block_size_x + threadIdx.x;
        if constexpr (Block::is_final)
            block_reduce_join_and_final<Interface, Block::block_size>(op, reduced, output, tid, batch);
        else
            block_reduce_join<Interface, Block::block_size>(op, reduced, output, tid, batch, bid);
    }

    // One 1d block per batch to finish joining the reduced values and compute the final output.
    template<typename Block, typename Interface, typename Op, typename Index,
             typename Joined, typename JoinedAlignedBuffers, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
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
        for (Index cid = 0; cid < n_elements; cid += Block::block_work_size) {
            block_reduce_ewise_1d_join
                <Block::block_size, Block::n_elements_per_thread, JoinedAlignedBuffers, Interface>
                (op, joined_1d, n_elements - cid, reduced, tid);
            joined_1d.for_each([](auto& accessor) {
                constexpr auto OFFSET = Block::block_work_size;
                accessor.offset_inplace(OFFSET);
            });
        }

        block_reduce_join_and_final<Interface, Block::block_size>(op, reduced, output, tid, batch);
    }
}

namespace noa::cuda::details {
    // nvcc bug - this could be a lambda, but nvcc <=12.6 is broken...
    template<size_t ALIGNMENT, typename Config,
             typename Op, typename Index,
             typename Input, typename Reduced, typename Output>
    void launch_reduce_ewise_small_2d_(
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Shape<Index, 2> shape,
        Stream& stream
    ) {
        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        using Block = ReduceEwise2dConfig<Config, VEC_SIZE, true>;

        constexpr auto TO_2D = nd::AccessorConfig<2>{
            .enforce_contiguous = is_vectorized<InputVec>(),
            .enforce_restrict = false,
            .filter = {0, 3},
        };
        auto input_2d = nd::reconfig_accessors<TO_2D>(std::forward<Input>(input));

        using Interface = Config::interface;
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using OutputDecay = std::decay_t<Output>;
        using Input2D = decltype(input_2d);

        const auto grid_y = GridY(shape[0], 1);
        for (u32 y{}; y < grid_y.n_launches(); ++y) {
            nd::offset_accessors(Vec{grid_y.offset_additive(y)}, input_2d, output);
            const auto config = LaunchConfig{
                .n_blocks = dim3(1, grid_y.n_blocks(y)),
                .n_threads = dim3(Config::block_size, 1)
            };
            stream.enqueue(
                reduce_ewise_2d<Block, Interface, OpDecay, Index, Input2D, InputVec, ReducedDecay, OutputDecay>,
                config, op, input_2d, shape[1], reduced, output
            );
        }
    }

    template<typename Config,
             typename Op, typename Index,
             typename Input, typename Reduced, typename Output>
    void launch_reduce_ewise_small_2d(
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Shape<Index, 2> shape,
        Stream& stream
    ) {
        if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
            size_t alignment = min_address_alignment(input, Shape<Index, 3>{shape[0], 1, 1});
            if (alignment == 16) {
                return launch_reduce_ewise_small_2d_<16, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    shape, stream
                );
            } else if (alignment == 8) {
                return launch_reduce_ewise_small_2d_<8, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    shape, stream
                );
            } else if (alignment == 4) {
                return launch_reduce_ewise_small_2d_<4, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    shape, stream
                );
            } else if (alignment == 2) {
                return launch_reduce_ewise_small_2d_<2, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    shape, stream
                );
            }
        }
        details::launch_reduce_ewise_small_2d_<1, Config>(
            std::forward<Op>(op),
            std::forward<Input>(input),
            std::forward<Reduced>(reduced),
            std::forward<Output>(output),
            shape, stream
        );
    }

    // nvcc bug - this could be a lambda, but nvcc <=12.6 is broken...
    template<size_t ALIGNMENT, typename Config,
             typename Input, typename Reduced, typename Output,
             typename Index, typename Op>
    void launch_reduce_ewise_small_4d_(
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Shape<Index, 4> shape,
        bool is_per_batch,
        Stream& stream
    ) {
        // In this config, the input cannot be easily interpreted as a 1d array.
        // As such, the 3 outermost dimensions are batched in a set of rows. Each block reduces at least one row.
        // If the innermost dimension is contiguous, blocks can use vectorize loads to read their row(s).
        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;

        constexpr auto TO_4D = nd::AccessorConfig<0>{.enforce_contiguous = is_vectorized<InputVec>()};
        auto input_4d = nd::reconfig_accessors<TO_4D>(std::forward<Input>(input));

        // Grid/Block shape. One block to reduce n_rows. GridY to reduce batch.
        constexpr u32 N_THREADS_X = Constant::WARP_SIZE;
        const u32 n_threads_y = max(Config::block_size / N_THREADS_X, u32{1});
        const auto n_rows = shape[2] * shape[1] * (is_per_batch ? 1 : shape[0]);
        const auto grid_y = GridY(is_per_batch ? shape[0] : 1, 1);

        using Block = ReduceEwise4dConfig<Config, N_THREADS_X, VEC_SIZE, true>;
        using Interface = Config::interface;
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using OutputDecay = std::decay_t<Output>;
        using Input4D = decltype(input_4d);

        for (u32 y{}; y < grid_y.n_launches(); ++y) {
            nd::offset_accessors(Vec{grid_y.offset_additive(y)}, input_4d, output);
            const auto config = LaunchConfig{
                .n_blocks = dim3(1, grid_y.n_blocks(y)),
                .n_threads = dim3(N_THREADS_X, n_threads_y),
            };
            stream.enqueue(
                reduce_ewise_4d<Block, Interface, OpDecay, Index, Input4D, InputVec, ReducedDecay, OutputDecay>,
                config, op, input_4d, reduced, shape.pop_front(), n_rows, output
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
        Shape<Index, 4> shape,
        bool is_per_batch,
        Stream& stream
    ) {
        if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
            size_t alignment = min_address_alignment(input, shape.pop_back());
            if (alignment == 16) {
                return launch_reduce_ewise_small_4d_<16, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    shape, is_per_batch, stream
                );
            } else if (alignment == 8) {
                return launch_reduce_ewise_small_4d_<8, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    shape, is_per_batch, stream
                );
            } else if (alignment == 4) {
                return launch_reduce_ewise_small_4d_<4, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    shape, is_per_batch, stream
                );
            } else if (alignment == 2) {
                return launch_reduce_ewise_small_4d_<2, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    shape, is_per_batch, stream
                );
            }
        }
        details::launch_reduce_ewise_small_4d_<1, Config>(
            std::forward<Op>(op),
            std::forward<Input>(input),
            std::forward<Reduced>(reduced),
            std::forward<Output>(output),
            shape, is_per_batch, stream
        );
    }

    // Allocate the joined buffers and set the accessors.
    template<typename Joined>
    auto get_joined_buffer(u32 n_blocks_x, u32 n_blocks_y, Joined& joined, u32 max_vector_size, Stream& stream) {
        return joined.map([&]<typename A>(A& accessor) {
            const u32 pitch = next_multiple_of(n_blocks_x, max_vector_size);
            auto buffer = AllocatorDevice::allocate_async<typename A::value_type>(pitch * n_blocks_y, stream);
            accessor = A(buffer.get(), Strides<typename A::index_type, 2>{pitch, 1});
            return buffer;
        });
    }

    // nvcc bug - this could be a lambda, but nvcc <=12.6 is broken...
    template<size_t ALIGNMENT, typename Config,
             typename Op, typename Index,
             typename Input, typename Reduced, typename Joined>
    void launch_reduce_ewise_large_2d_(
        Op& op,
        Input&& input,
        Reduced& reduced,
        Joined joined, // copy, we offset inplace
        Stream& stream,
        const Shape<Index, 2>& shape,
        u32 n_blocks_x,
        const GridY grid_y
    ) {
        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        constexpr auto TO_2D = nd::AccessorConfig<2>{
            .enforce_contiguous = is_vectorized<InputVec>(),
            .enforce_restrict = false,
            .filter = {0, 3},
        };
        auto input_2d = nd::reconfig_accessors<TO_2D>(std::forward<Input>(input));

        using Input2D = decltype(input_2d);
        using Block = ReduceEwise2dConfig<Config, VEC_SIZE, false>;
        using Interface = Config::interface;

        for (u32 y{}; y < grid_y.n_launches(); ++y) {
            nd::offset_accessors(Vec{grid_y.offset_additive(y)}, input_2d, joined);
            const auto config = LaunchConfig{
                .n_blocks = dim3(n_blocks_x, grid_y.n_blocks(y)),
                .n_threads = dim3(Config::block_size, 1),
            };
            stream.enqueue(
                reduce_ewise_2d<Block, Interface, Op, Index, Input2D, InputVec, Reduced, Joined>,
                config, op, input_2d, shape[1], reduced, joined
            );
        }
    }

    template<typename Config,
             typename Op, typename Index,
             typename Input, typename Reduced, typename Output>
    void launch_reduce_ewise_large_2d(
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Shape<Index, 2> shape,
        Stream& stream
    ) {
        // In this config, the inputs can be interpreted as 1d arrays. If the innermost dimension is contiguous,
        // i.e. if all elements to reduce are contiguous, we can vectorize loads for the first kernel.

        // First kernel, limit the number of blocks, otherwise the second kernel would have too much work to do.
        const auto grid_x = Grid<Config::max_grid_size>(shape[1], Config::block_work_size);
        const auto grid_y = GridY(shape[0], 1);
        const auto n_blocks_x = grid_x.n_blocks(0);
        const auto n_blocks_y = safe_cast<u32>(grid_y.n_blocks_total());

        // Allocate the joined buffer.
        using Joined = joined_tuple_t<2, Index, Reduced>; // Tuple<AccessorRestrictContiguous<T, 2, Index>,...>
        constexpr size_t JOINED_VEC_SIZE = maximum_allowed_aligned_buffer_size<16, Joined>();
        using JoinedVec = to_aligned_buffer_t<Joined, 16, JOINED_VEC_SIZE>;
        Joined joined;
        [[maybe_unused]] auto joined_buffer = get_joined_buffer(
            n_blocks_x, n_blocks_y, joined, JOINED_VEC_SIZE, stream);

        if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
            size_t alignment = min_address_alignment(input, Shape<Index, 3>{shape[0], 1, 1});
            if (alignment == 16) {
                launch_reduce_ewise_large_2d_<16, Config>(
                    op, std::forward<Input>(input), reduced, joined, stream, shape, n_blocks_x, grid_y
                );
            } else if (alignment == 8) {
                launch_reduce_ewise_large_2d_<8, Config>(
                    op, std::forward<Input>(input), reduced, joined, stream, shape, n_blocks_x, grid_y
                );
            } else if (alignment == 4) {
                launch_reduce_ewise_large_2d_<4, Config>(
                    op, std::forward<Input>(input), reduced, joined, stream, shape, n_blocks_x, grid_y
                );
            } else if (alignment == 2) {
                launch_reduce_ewise_large_2d_<2, Config>(
                    op, std::forward<Input>(input), reduced, joined, stream, shape, n_blocks_x, grid_y
                );
            } else {
                launch_reduce_ewise_large_2d_<1, Config>(
                    op, std::forward<Input>(input), reduced, joined, stream, shape, n_blocks_x, grid_y
                );
            }
        } else {
            launch_reduce_ewise_large_2d_<1, Config>(
                op, std::forward<Input>(input), reduced, joined, stream, shape, n_blocks_x, grid_y
            );
        }

        // Second kernel.
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using OutputDecay = std::decay_t<Output>;
        using SecondBlock = ReduceEwise2dConfig<Config, JOINED_VEC_SIZE>;
        using Interface = Config::interface;
        stream.enqueue(
            reduce_ewise_second<SecondBlock, Interface, OpDecay, Index, Joined, JoinedVec, ReducedDecay, OutputDecay>,
            LaunchConfig{.n_blocks = n_blocks_y, .n_threads = Config::block_size},
            std::forward<Op>(op), joined, n_blocks_x, std::forward<Reduced>(reduced), std::forward<Output>(output)
        );
    }

    // nvcc bug - this could be a lambda, but nvcc <=12.6 is broken...
    template<size_t BLOCK_SIZE_X, size_t ALIGNMENT, typename Config,
             typename Input, typename Reduced, typename Joined,
             typename Index, typename Op>
    void launch_reduce_ewise_large_4d_(
        Op& op,
        Input&& input,
        Reduced& reduced,
        Joined& joined,
        Shape<Index, 4> shape,
        Stream& stream,
        Index n_rows,
        dim3 n_threads,
        u32 n_blocks_x,
        const GridY& grid_y
    ) {
        // Prepare the input for vectorization.
        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        constexpr auto TO_4D = nd::AccessorConfig<0>{.enforce_contiguous = is_vectorized<InputVec>()};
        auto input_4d = nd::reconfig_accessors<TO_4D>(std::forward<Input>(input));

        // Launch the kernel.
        using Block = ReduceEwise4dConfig<Config, BLOCK_SIZE_X, VEC_SIZE>;
        using Interface = Config::interface;
        using Input4D = decltype(input_4d);

        for (u32 y{}; y < grid_y.n_launches(); y++) {
            nd::offset_accessors(Vec{grid_y.offset_additive(y)}, input_4d, joined);

            const auto config = LaunchConfig{
                .n_blocks = dim3(n_blocks_x, grid_y.n_blocks(y)),
                .n_threads = n_threads,
            };
            stream.enqueue(
                reduce_ewise_4d<Block, Interface, Op, Index, Input4D, InputVec, Reduced, Joined>,
                config, op, input_4d, reduced, shape.pop_front(), n_rows, joined
            );
        }
    }

    template<typename Config,
             typename Input, typename Reduced, typename Output,
             typename Index, typename Op>
    void launch_reduce_ewise_large_4d(
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Shape<Index, 4> shape,
        bool is_per_batch,
        Stream& stream
    ) {
        // In this config, the input cannot be easily interpreted as a 1d array.
        // As such, the 3 outermost dimensions are batched in a set of rows. Each block reduces at least one row.
        // If the innermost dimension is contiguous, blocks can use vectorize loads to read their row(s).

        // Block shape.
        u32 n_threads_x = shape[3] > 512 ? 256 : 64; // TODO better heuristic?
        if (not is_multiple_of(Config::block_size, n_threads_x))
            n_threads_x = Constant::WARP_SIZE;
        const u32 n_threads_y = max(Config::block_size / n_threads_x, u32{1});
        const auto n_threads = dim3(n_threads_x, n_threads_y);

        // Grid shape.
        const auto n_rows = shape[2] * shape[1] * (is_per_batch ? 1 : shape[0]);
        const auto grid_x = Grid<Config::max_grid_size>(n_rows, n_threads_y);
        const auto grid_y = GridY(is_per_batch ? shape[0] : 1, 1);
        const auto n_blocks_x = grid_x.n_blocks(0);
        const auto n_blocks_y = safe_cast<u32>(grid_y.n_blocks_total());

        // Allocate the joined buffer.
        using Joined = joined_tuple_t<2, Index, Reduced>; // Tuple<AccessorRestrictContiguous<T, 2, Index>,...>
        constexpr size_t JOINED_VEC_SIZE = maximum_allowed_aligned_buffer_size<16, Joined>();
        using JoinedVec = to_aligned_buffer_t<Joined, 16, JOINED_VEC_SIZE>;
        Joined joined;
        [[maybe_unused]] auto joined_buffer = get_joined_buffer(
            n_blocks_x, n_blocks_y, joined, JOINED_VEC_SIZE, stream);

        // TODO Reduce the number of instantiations by adding support of a runtime block_size_x.
        //      That would mean using dynamic shared memory in block_reduce, which is fine...
        // Note that these if statements are likely to instantiate the same kernel. For instance, if the smallest
        // type has an alignment of 4, only 6 kernels are created (3 with n_threads_x=256, 3 with n_threads_x=64).
        // The worst case is when the smallest type is aligned to only one byte, in which case 10 kernels are created.
        if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
            const size_t alignment = min_address_alignment(input, shape.pop_back());
            if (n_threads_x == 256) {
                if (alignment == 16) {
                    launch_reduce_ewise_large_4d_<256, 16, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else if (alignment == 8) {
                    launch_reduce_ewise_large_4d_<256, 8, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else if (alignment == 4) {
                    launch_reduce_ewise_large_4d_<256, 4, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else if (alignment == 2) {
                    launch_reduce_ewise_large_4d_<256, 2, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else {
                    launch_reduce_ewise_large_4d_<256, 1, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                }
            } else {
                if (alignment == 16) {
                    launch_reduce_ewise_large_4d_<64, 16, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else if (alignment == 8) {
                    launch_reduce_ewise_large_4d_<64, 8, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else if (alignment == 4) {
                    launch_reduce_ewise_large_4d_<64, 4, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else if (alignment == 2) {
                    launch_reduce_ewise_large_4d_<64, 2, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else {
                    launch_reduce_ewise_large_4d_<64, 1, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                }
            }
        } else {
            if (n_threads_x == 256) {
                launch_reduce_ewise_large_4d_<256, 1, Config>(
                    op, std::forward<Input>(input), reduced, joined, shape, stream,
                    n_rows, n_threads, n_blocks_x, grid_y
                );
            } else {
                launch_reduce_ewise_large_4d_<64, 1, Config>(
                    op, std::forward<Input>(input), reduced, joined, shape, stream,
                    n_rows, n_threads, n_blocks_x, grid_y
                );
            }
        }

        // Second kernel.
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using OutputDecay = std::decay_t<Output>;
        using SecondBlock = ReduceEwise2dConfig<Config, JOINED_VEC_SIZE>;
        using Interface = Config::interface;

        const auto config = LaunchConfig{.n_blocks = n_blocks_y, .n_threads = Config::block_size};
        stream.enqueue(
            reduce_ewise_second<SecondBlock, Interface, OpDecay, Index, Joined, JoinedVec, ReducedDecay, OutputDecay>,
            config, std::forward<Op>(op), joined, n_blocks_x, std::forward<Reduced>(reduced), output
        );
    }
}

namespace noa::cuda {
    template<bool ZipInput = false,
             bool ZipReduced = false,
             bool ZipOutput = false,
             u32 ElementsPerThread = 8,
             u32 BlockSize = 512,
             u32 MaxGridSize = 4096,
             bool EnableVectorization = true>
    struct ReduceEwiseConfig {
        static_assert(is_power_of_2(ElementsPerThread));
        static_assert(is_power_of_2(BlockSize) and BlockSize <= Limits::MAX_THREADS);

        using interface = nd::ReduceEwiseInterface<ZipInput, ZipReduced, ZipOutput>;
        static constexpr u32 max_grid_size = MaxGridSize;
        static constexpr u32 block_size = BlockSize;
        static constexpr u32 n_elements_per_thread = ElementsPerThread;
        static constexpr u32 block_work_size = block_size * n_elements_per_thread;
        static constexpr bool enable_vectorization = EnableVectorization;
    };

    template<typename Config = ReduceEwiseConfig<>,
             typename Input, typename Reduced, typename Output, typename Index, typename Op>
    requires (nt::tuple_of_accessor_nd<std::decay_t<Input>, 4> and
              not nt::tuple_of_accessor_value<std::decay_t<Input>> and // at least one varray
              nt::tuple_of_accessor_pure_nd<std::decay_t<Output>, 1> and
              nt::tuple_of_accessor_value<std::decay_t<Reduced>>)
    NOA_NOINLINE void reduce_ewise(
        const Shape<Index, 4>& shape,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream
    ) {
        const auto n_elements = safe_cast<Index>(shape.template as_safe<isize>().n_elements());
        const Vec<bool, 4> is_contiguous = ni::is_contiguous(input, shape);

        constexpr auto SMALL_THRESHOLD = Config::block_work_size * 4;
        if (is_contiguous.pop_back() == true) {
            if (n_elements <= SMALL_THRESHOLD) {
                details::launch_reduce_ewise_small_2d<Config>(
                    std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), Shape<Index, 2>{1, n_elements}, stream
                );
            } else {
                details::launch_reduce_ewise_large_2d<Config>(
                    std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), Shape<Index, 2>{1, n_elements}, stream
                );
            }
        } else {
            if (n_elements <= SMALL_THRESHOLD) {
                details::launch_reduce_ewise_small_4d<Config>(
                    std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), shape, false, stream
                );
            } else {
                details::launch_reduce_ewise_large_4d<Config>(
                    std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), shape, false, stream
                );
            }
        }
    }
}

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif
