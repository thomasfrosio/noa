#pragma once
#include "catch2/internal/catch_decomposer.hpp"
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/base/Config.hpp"
#include "noa/base/Vec.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Block.cuh"
#include "noa/runtime/cuda/ComputeHandle.cuh"

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
    struct ReduceEwiseNdConfig {
        using interface = Config::interface;
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = Config::block_size;
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = max(block_size / block_size_x, 1u);
        static constexpr u32 n_elements_per_thread_x = max(MaxVectorSize, Config::n_elements_per_thread);
        static constexpr u32 block_work_size_x = block_size_x * n_elements_per_thread_x;
        static constexpr bool is_final = IsFinal;
    };

    // Reduce rows, each row is reduced individually..
    // N=1: 1d accessors, 1 row to reduce, 1d grid.
    // N=2: 2d accessors, n row(s) to reduce, 2d grid (y is n).
    template<typename Block, typename Interface, typename Op, typename Index, usize N,
             typename Input, typename InputAlignedBuffers, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_ewise_rows(
        Op op, Input input, Index width, Reduced reduced, Output output,
        Vec<u32, N - 1> grid_size_y, Vec<u32, N - 1> block_index_offset_y
    ) {
        const Index row = blockIdx.y;
        const Index bid = blockIdx.x;
        const Index tid = threadIdx.x;
        const Index starting_index = Block::block_work_size * bid;
        const Index grid_work_size = Block::block_work_size * gridDim.x;

        auto input_1d = std::move(input).map([row]<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T>) {
                return std::forward<T>(accessor);
            } else if constexpr (nt::is_accessor_nd_v<T, 2>) {
                // Offset the input accessor to the current row,
                // so that it can be used later to reset the pointer.
                accessor.offset_inplace(row);
                return accessor[0]; // 1d AccessorReference
            } else if constexpr (nt::is_accessor_nd_v<T, 1>) {
                return AccessorReference(accessor);
            } else {
                static_assert(nt::always_false<T>);
            }
        });

        const auto ci = ComputeHandle<Index, N, 1, true, false, not Block::is_final>(grid_size_y, block_index_offset_y);
        Interface::init(ci, op, row);

        for (Index cid = starting_index; cid < width; cid += grid_work_size) {
            input_1d.for_each_enumerate([&input, cid]<size_t I>(auto& accessor) {
                accessor.reset_pointer(input[Tag<I>{}].get());
                accessor.offset_inplace(cid);
            });
            block_call_ewise_1d
                <Block::block_size, Block::n_elements_per_thread, InputAlignedBuffers, Interface>
                (ci, op, input_1d, width - cid, reduced, tid);
        }

        Interface::deinit(ci, op, row);

        if constexpr (Block::is_final) {
            // There's one block per row, so compute the reduced value for the block
            // and save it in the output at the row index.
            block_join_and_post<Interface, Block::block_size, true>(op, reduced, output, tid, row);
        } else {
            // The output is the "joined" buffer, which is a buffer with one value per block and per row.
            // These values will then be reduced by the second reduction kernel (see below).
            block_join<Interface, Block::block_size, true>(op, reduced, output, tid, row, bid);
        }
    }

    // Here the input is organized has a series of rows. Given the original DHW shape of the input and the row index,
    // we can derive the BDH indices. Each dimension can have an arbitrary stride, but if the rows themselves are
    // contiguous (if the W stride is 1), then vectorized load/stores can be used to load/store elements from the rows.
    //
    // This kernel explicitly supports per-batch reductions (see reduce_axes_ewise), in which case the grid should be
    // 2d (gridDim.x is the number of blocks to reduce the rows of a given batch and gridDim.y is the number of batches)
    // and joined should be 2d Accessors, where the outer dimension is the batch.
    template<typename Block, typename Interface, typename Op, typename Index, usize N,
             typename Input, typename InputAlignedBuffers, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_ewise_nd(
        Op op,
        Input input,
        Reduced reduced,
        Vec<Index, N> sizes_for_offset2index,
        Index n_rows,
        Index width,
        Output output,
        Vec<u32, 1> grid_size_y,
        Vec<u32, 1> block_index_offset_y
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
                if constexpr (nt::is_accessor_nd_v<T, 3>)
                    return accessor[0][0];
                else if constexpr (nt::is_accessor_nd_v<T, 4>)
                    return accessor[0][0][0];
                else
                    static_assert(nt::always_false<T>);
            }
        });

        const auto ci = ComputeHandle<Index, 2, 2, true, false, not Block::is_final>(grid_size_y, block_index_offset_y);
        Interface::init(ci, op, batch);

        for (Index row = initial_row; row < n_rows; row += n_rows_per_grid) { // for every row (within a batch)
            auto outer_indices = offset2index(row, sizes_for_offset2index);
            for (Index cid = 0; cid < width; cid += Block::block_work_size_x) { // consume the row
                input_1d.for_each_enumerate([&input, &outer_indices, &cid]<size_t I>(auto& accessor_1d) {
                    if constexpr (not nt::is_accessor_value_v<decltype(accessor_1d)>) {
                        auto& accessor = input[Tag<I>{}];
                        auto new_pointer = accessor.offset_pointer(accessor.get(), outer_indices.push_back(cid));
                        accessor_1d.reset_pointer(new_pointer);
                    }
                });
                block_call_ewise_1d
                    <Block::block_size_x, Block::n_elements_per_thread_x, InputAlignedBuffers, Interface>
                    (ci, op, input_1d, width - cid, reduced, static_cast<Index>(threadIdx.x));
            }
        }

        Interface::deinit(ci, op, batch);

        const Index tid = threadIdx.y * Block::block_size_x + threadIdx.x;
        if constexpr (Block::is_final)
            block_join_and_post<Interface, Block::block_size, true>(op, reduced, output, tid, batch);
        else
            block_join<Interface, Block::block_size, true>(op, reduced, output, tid, batch, bid);
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

        if constexpr (not nt::empty_tuple<Reduced>) {
            auto joined_1d = joined.map([&](auto& accessor) { return accessor[batch]; });
            for (Index cid = 0; cid < n_elements; cid += Block::block_work_size) {
                block_join_ewise_1d
                    <Block::block_size, Block::n_elements_per_thread, JoinedAlignedBuffers, Interface>
                    (op, joined_1d, n_elements - cid, reduced, tid);
                joined_1d.for_each([](auto& accessor) {
                    constexpr auto OFFSET = Block::block_work_size;
                    accessor.offset_inplace(OFFSET);
                });
            }
        }

        block_join_and_post<Interface, Block::block_size, true>(op, reduced, output, tid, batch);
    }
}

namespace noa::cuda::details {
    // nvcc bug - this could be a lambda, but nvcc <=12.6 is broken...
    template<usize ALIGNMENT, usize N, typename Config,
             typename Op, typename Index,
             typename Input, typename Reduced, typename Output>
    void launch_reduce_ewise_small_rows_(
        Op&& op,
        Input&& input_collapsed, // HW with other (outer) dimensions empty
        Reduced&& reduced,
        Output&& output,
        u32 n_rows,
        Index width,
        Stream& stream
    ) {
        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        using Block = ReduceEwise2dConfig<Config, VEC_SIZE, true>;

        using Interface = Config::interface;
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using OutputDecay = std::decay_t<Output>;

        if constexpr (N == 1) {
            constexpr auto CONFIG = nd::AccessorConfig<2>{
                .enforce_contiguous = is_vectorized<InputVec>(),
                .enforce_restrict = false,
            };
            auto input_1d = nd::reconfig_accessors<CONFIG>(std::forward<Input>(input_collapsed));
            const auto config = LaunchConfig{
                .n_blocks = dim3(1, 1, 1),
                .n_threads = dim3(Config::block_size, 1, 1)
            };
            stream.enqueue(
                reduce_ewise_rows<Block, Interface, OpDecay, Index, 1,
                decltype(input_1d), InputVec, ReducedDecay, OutputDecay>,
                config, op, input_1d, width, reduced, output, {}, {}
            );
        } else {
            constexpr auto DIMENSIONS = Vec<usize, 2>{0, 1} + N - 2;
            constexpr auto TO_2D = nd::AccessorConfig<2>{
                .enforce_contiguous = is_vectorized<InputVec>(),
                .enforce_restrict = false,
                .filter = DIMENSIONS,
            };
            auto input_2d = nd::reconfig_accessors<TO_2D>(std::forward<Input>(input_collapsed));

            const auto grid_y = GridY(n_rows, 1);
            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                nd::offset_accessors(Vec{grid_y.offset_additive(y)}, input_2d, output);
                const auto config = LaunchConfig{
                    .n_blocks = dim3(1, grid_y.n_blocks(y)),
                    .n_threads = dim3(Config::block_size, 1)
                };
                stream.enqueue(
                    reduce_ewise_rows<Block, Interface, OpDecay, Index, 2,
                    decltype(input_2d), InputVec, ReducedDecay, OutputDecay>,
                    config, op, input_2d, width, reduced, output,
                    Vec{grid_y.n_blocks_total()}.as<u32>(), Vec{grid_y.offset(y)}
                );
            }
        }
    }

    template<typename Config, usize N,
             typename Op, typename Index,
             typename Input, typename Reduced, typename Output>
    void launch_reduce_ewise_small_rows(
        Op&& op,
        Input&& input_collapsed,
        Reduced&& reduced,
        Output&& output,
        u32 n_rows,
        Index width,
        Stream& stream
    ) {
        if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
            auto shape_without_width = Shape<u32, N - 1>::from_value(1);
            if constexpr (N >= 2)
                shape_without_width[N - 2] = n_rows;
            const auto alignment = min_address_alignment(input_collapsed, shape_without_width);

            if (alignment == 16) {
                return launch_reduce_ewise_small_rows_<16, N, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input_collapsed),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    n_rows, width, stream
                );
            } else if (alignment == 8) {
                return launch_reduce_ewise_small_rows_<8, N, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input_collapsed),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    n_rows, width, stream
                );
            } else if (alignment == 4) {
                return launch_reduce_ewise_small_rows_<4, N, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input_collapsed),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    n_rows, width, stream
                );
            } else if (alignment == 2) {
                return launch_reduce_ewise_small_rows_<2, N, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input_collapsed),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    n_rows, width, stream
                );
            }
        }
        details::launch_reduce_ewise_small_rows_<1, N, Config>(
            std::forward<Op>(op),
            std::forward<Input>(input_collapsed),
            std::forward<Reduced>(reduced),
            std::forward<Output>(output),
            n_rows, width, stream
        );
    }

    // nvcc bug - this could be a lambda, but nvcc <=12.6 is broken...
    template<usize ALIGNMENT, typename Config,
             typename Input, typename Reduced, typename Output,
             typename Index, usize N, typename Op> requires (N >= 3)
    void launch_reduce_ewise_small_nd_(
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Shape<Index, N> shape,
        bool keep_outermost,
        Stream& stream
    ) {
        // The input cannot be interpreted as a 1d or 2d array.
        // So the outer (excluding the outermost if keep_outermost==true) dimensions are batched in a set of rows.
        // Each block reduces at least one row.
        // If the innermost dimension is contiguous, blocks can use vectorize loads to read their row(s).
        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;

        constexpr auto CONFIG = nd::AccessorConfig<0>{.enforce_contiguous = is_vectorized<InputVec>()};
        auto input_nd = nd::reconfig_accessors<CONFIG>(std::forward<Input>(input));

        // Grid/Block shape. One block to reduce n_rows.
        // GridY to reduce the outermost (if reduced).
        constexpr u32 N_THREADS_X = Constant::WARP_SIZE;
        const u32 n_threads_y = max(Config::block_size / N_THREADS_X, u32{1});
        const auto n_reductions = keep_outermost ? shape[0] : 1;
        Index n_rows{1};
        for (usize i = keep_outermost; i < N - 1; ++i)
            n_rows *= shape[i];

        using Block = ReduceEwiseNdConfig<Config, N_THREADS_X, VEC_SIZE, true>;
        using Interface = Config::interface;
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using OutputDecay = std::decay_t<Output>;
        using InputND = decltype(input_nd);

        const auto grid_y = GridY(n_reductions, 1);
        for (u32 y{}; y < grid_y.n_launches(); ++y) {
            nd::offset_accessors(Vec{grid_y.offset_additive(y)}, input_nd, output);
            const auto config = LaunchConfig{
                .n_blocks = dim3(1, grid_y.n_blocks(y)),
                .n_threads = dim3(N_THREADS_X, n_threads_y),
            };
            stream.enqueue(
                reduce_ewise_nd<Block, Interface, OpDecay, Index, N - 2, InputND, InputVec, ReducedDecay, OutputDecay>,
                config, op, input_nd, reduced, shape.pop_front().pop_back().vec, n_rows, shape[N - 1], output,
                Vec{grid_y.n_blocks_total()}.as<u32>(), Vec{grid_y.offset(y)}
            );
        }
    }

    template<typename Config,
             typename Input, typename Reduced, typename Output,
             typename Index, usize N, typename Op> requires (N >= 3)
    void launch_reduce_ewise_small_nd(
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Shape<Index, N> shape,
        bool keep_outermost,
        Stream& stream
    ) {
        if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
            const auto alignment = min_address_alignment(input, shape.pop_back());
            if (alignment == 16) {
                return launch_reduce_ewise_small_nd_<16, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    shape, keep_outermost, stream
                );
            } else if (alignment == 8) {
                return launch_reduce_ewise_small_nd_<8, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    shape, keep_outermost, stream
                );
            } else if (alignment == 4) {
                return launch_reduce_ewise_small_nd_<4, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    shape, keep_outermost, stream
                );
            } else if (alignment == 2) {
                return launch_reduce_ewise_small_nd_<2, Config>(
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    shape, keep_outermost, stream
                );
            }
        }
        details::launch_reduce_ewise_small_nd_<1, Config>(
            std::forward<Op>(op),
            std::forward<Input>(input),
            std::forward<Reduced>(reduced),
            std::forward<Output>(output),
            shape, keep_outermost, stream
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
    template<size_t ALIGNMENT, usize N, typename Config,
             typename Op, typename Index,
             typename Input, typename Reduced, typename Joined>
    void launch_reduce_ewise_large_rows_(
        Op& op,
        Input&& input_collapsed,
        Reduced& reduced,
        Joined joined, // copy, we offset inplace
        Stream& stream,
        Index width,
        u32 n_blocks_x,
        const GridY grid_y
    ) {
        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        using Block = ReduceEwise2dConfig<Config, VEC_SIZE, false>;
        using Interface = Config::interface;

        if constexpr (N == 1) {
            constexpr auto CONFIG = nd::AccessorConfig<2>{
                .enforce_contiguous = is_vectorized<InputVec>(),
                .enforce_restrict = false,
            };
            auto input_1d = nd::reconfig_accessors<CONFIG>(std::forward<Input>(input_collapsed));
            const auto config = LaunchConfig{
                .n_blocks = dim3(n_blocks_x, 1),
                .n_threads = dim3(Config::block_size, 1),
            };
            stream.enqueue(
                reduce_ewise_rows<Block, Interface, Op, Index, 1, decltype(input_1d), InputVec, Reduced, Joined>,
                config, op, input_1d, width, reduced, joined, {}, {}
            );
        } else {
            constexpr auto DIMENSIONS = Vec<usize, 2>{0, 1} + N - 2;
            constexpr auto TO_2D = nd::AccessorConfig<2>{
                .enforce_contiguous = is_vectorized<InputVec>(),
                .enforce_restrict = false,
                .filter = DIMENSIONS,
            };
            auto input_2d = nd::reconfig_accessors<TO_2D>(std::forward<Input>(input_collapsed));

            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                nd::offset_accessors(Vec{grid_y.offset_additive(y)}, input_2d, joined);
                const auto config = LaunchConfig{
                    .n_blocks = dim3(n_blocks_x, grid_y.n_blocks(y)),
                    .n_threads = dim3(Config::block_size, 1),
                };
                stream.enqueue(
                    reduce_ewise_rows<Block, Interface, Op, Index, 2, decltype(input_2d), InputVec, Reduced, Joined>,
                    config, op, input_2d, width, reduced, joined,
                    Vec{grid_y.n_blocks_total()}.as<u32>(), Vec{grid_y.offset(y)}
                );
            }
        }
    }

    template<typename Config, usize N,
             typename Op, typename Index,
             typename Input, typename Reduced, typename Output>
    void launch_reduce_ewise_large_rows(
        Op&& op,
        Input&& input_collapsed,
        Reduced&& reduced,
        Output&& output,
        u32 n_rows,
        Index width,
        Stream& stream
    ) {
        check(N > 1 or n_rows == 1);

        // In this config, the inputs can be interpreted as 1d or 2d arrays. If the innermost dimension is contiguous,
        // i.e., if all elements to reduce are contiguous, we can vectorize loads for the first kernel.

        // First kernel, limit the number of blocks, otherwise the second kernel would have too much work to do.
        // This is not true if there's no value to reduce, in which case the grid can be maximized.
        const auto grid_x = Grid<Config::max_grid_size>(width, Config::block_work_size);
        const auto grid_y = GridY(n_rows, 1);
        const auto n_blocks_x = grid_x.n_blocks(0);
        const auto n_blocks_y = safe_cast<u32>(grid_y.n_blocks_total());

        // Allocate the joined buffer.
        // If Reduced is an empty tuple, Joined will be empty too and no allocation will be performed.
        using Joined = joined_tuple_t<2, Index, Reduced>; // Tuple<AccessorRestrictContiguous<T, 2, Index>,...>
        constexpr size_t JOINED_VEC_SIZE = maximum_allowed_aligned_buffer_size<16, Joined>();
        using JoinedVec = to_aligned_buffer_t<Joined, 16, JOINED_VEC_SIZE>;
        Joined joined;
        [[maybe_unused]] auto joined_buffer = get_joined_buffer(
            n_blocks_x, n_blocks_y, joined, JOINED_VEC_SIZE, stream);

        if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
            auto shape_without_width = Shape<u32, N - 1>::from_value(1);
            if constexpr (N >= 2)
                shape_without_width[N - 2] = n_rows;
            const auto alignment = min_address_alignment(input_collapsed, shape_without_width);

            if (alignment == 16) {
                launch_reduce_ewise_large_rows_<16, N, Config>(
                    op, std::forward<Input>(input_collapsed), reduced, joined, stream, width, n_blocks_x, grid_y
                );
            } else if (alignment == 8) {
                launch_reduce_ewise_large_rows_<8, N, Config>(
                    op, std::forward<Input>(input_collapsed), reduced, joined, stream, width, n_blocks_x, grid_y
                );
            } else if (alignment == 4) {
                launch_reduce_ewise_large_rows_<4, N, Config>(
                    op, std::forward<Input>(input_collapsed), reduced, joined, stream, width, n_blocks_x, grid_y
                );
            } else if (alignment == 2) {
                launch_reduce_ewise_large_rows_<2, N, Config>(
                    op, std::forward<Input>(input_collapsed), reduced, joined, stream, width, n_blocks_x, grid_y
                );
            } else {
                launch_reduce_ewise_large_rows_<1, N, Config>(
                    op, std::forward<Input>(input_collapsed), reduced, joined, stream, width, n_blocks_x, grid_y
                );
            }
        } else {
            launch_reduce_ewise_large_rows_<1, N, Config>(
                op, std::forward<Input>(input_collapsed), reduced, joined, stream, width, n_blocks_x, grid_y
            );
        }

        // Second kernel.
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using OutputDecay = std::decay_t<Output>;
        using SecondBlock = ReduceEwise2dConfig<Config, JOINED_VEC_SIZE>;
        using Interface = Config::interface;
        // TODO If Reduced is an empty tuple, check op has a valid post, otherwise don't even run this.
        constexpr bool HAS_REDUCED = not nt::empty_tuple<Reduced>;
        stream.enqueue(
            reduce_ewise_second<SecondBlock, Interface, OpDecay, Index, Joined, JoinedVec, ReducedDecay, OutputDecay>,
            LaunchConfig{.n_blocks = n_blocks_y, .n_threads = HAS_REDUCED ? Config::block_size : 1},
            std::forward<Op>(op), joined, n_blocks_x, std::forward<Reduced>(reduced), std::forward<Output>(output)
        );
    }

    // nvcc bug - this could be a lambda, but nvcc <=12.6 is broken...
    template<size_t BLOCK_SIZE_X, size_t ALIGNMENT, typename Config,
             typename Input, typename Reduced, typename Joined,
             typename Index, usize N, typename Op>
    void launch_reduce_ewise_large_nd_(
        Op& op,
        Input&& input,
        Reduced& reduced,
        Joined& joined,
        Shape<Index, N> shape,
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
        using Block = ReduceEwiseNdConfig<Config, BLOCK_SIZE_X, VEC_SIZE>;
        using Interface = Config::interface;
        using Input4D = decltype(input_4d);

        for (u32 y{}; y < grid_y.n_launches(); y++) {
            nd::offset_accessors(Vec{grid_y.offset_additive(y)}, input_4d, joined);

            const auto config = LaunchConfig{
                .n_blocks = dim3(n_blocks_x, grid_y.n_blocks(y)),
                .n_threads = n_threads,
            };
            stream.enqueue(
                reduce_ewise_nd<Block, Interface, Op, Index, Input4D, InputVec, Reduced, Joined>,
                config, op, input_4d, reduced, shape.pop_front().pop_back(), n_rows, shape[N - 1], joined,
                Vec{grid_y.n_blocks_total()}.as<u32>(), Vec{grid_y.offset(y)}
            );
        }
    }

    template<typename Config,
             typename Input, typename Reduced, typename Output,
             typename Index, usize N, typename Op>
    void launch_reduce_ewise_large_nd(
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Shape<Index, N> shape,
        bool keep_outermost,
        Stream& stream
    ) {
        // In this config, the input cannot be easily interpreted as a 1d array.
        // As such, the 3 outermost dimensions are batched in a set of rows. Each block reduces at least one row.
        // If the innermost dimension is contiguous, blocks can use vectorize loads to read their row(s).

        // Block shape.
        u32 n_threads_x = shape[3] > 512 ? 256 : 64; // TODO better heuristic
        if (not is_multiple_of(Config::block_size, n_threads_x))
            n_threads_x = Constant::WARP_SIZE;
        const u32 n_threads_y = max(Config::block_size / n_threads_x, u32{1});
        const auto n_threads = dim3(n_threads_x, n_threads_y);

        // Grid shape.
        const auto n_reductions = keep_outermost ? shape[0] : 1;
        Index n_rows{1};
        for (usize i = keep_outermost; i < N - 1; ++i)
            n_rows *= shape[i];

        const auto grid_x = Grid<Config::max_grid_size>(n_rows, n_threads_y);
        const auto grid_y = GridY(n_reductions, 1);
        const auto n_blocks_x = grid_x.n_blocks(0);
        const auto n_blocks_y = safe_cast<u32>(grid_y.n_blocks_total());

        // Allocate the joined buffer.
        // If Reduced is an empty tuple, Joined will be empty too and no allocation will be performed.
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
                    launch_reduce_ewise_large_nd_<256, 16, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else if (alignment == 8) {
                    launch_reduce_ewise_large_nd_<256, 8, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else if (alignment == 4) {
                    launch_reduce_ewise_large_nd_<256, 4, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else if (alignment == 2) {
                    launch_reduce_ewise_large_nd_<256, 2, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else {
                    launch_reduce_ewise_large_nd_<256, 1, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                }
            } else {
                if (alignment == 16) {
                    launch_reduce_ewise_large_nd_<64, 16, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else if (alignment == 8) {
                    launch_reduce_ewise_large_nd_<64, 8, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else if (alignment == 4) {
                    launch_reduce_ewise_large_nd_<64, 4, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else if (alignment == 2) {
                    launch_reduce_ewise_large_nd_<64, 2, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                } else {
                    launch_reduce_ewise_large_nd_<64, 1, Config>(
                        op, std::forward<Input>(input), reduced, joined, shape, stream,
                        n_rows, n_threads, n_blocks_x, grid_y
                    );
                }
            }
        } else {
            if (n_threads_x == 256) {
                launch_reduce_ewise_large_nd_<256, 1, Config>(
                    op, std::forward<Input>(input), reduced, joined, shape, stream,
                    n_rows, n_threads, n_blocks_x, grid_y
                );
            } else {
                launch_reduce_ewise_large_nd_<64, 1, Config>(
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

        // TODO If Reduced is an empty tuple, check op has a valid post, otherwise don't even run this.
        constexpr bool HAS_REDUCED = not nt::empty_tuple<Reduced>;
        stream.enqueue(
            reduce_ewise_second<SecondBlock, Interface, OpDecay, Index, Joined, JoinedVec, ReducedDecay, OutputDecay>,
            LaunchConfig{.n_blocks = n_blocks_y, .n_threads = HAS_REDUCED ? Config::block_size : 1},
            std::forward<Op>(op), joined, n_blocks_x, std::forward<Reduced>(reduced), output
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
             typename Input, typename Reduced, typename Output, typename Index, usize N, typename Op>
    requires (nt::tuple_of_accessor_nd<std::decay_t<Input>, N> and
              not nt::tuple_of_accessor_value<std::decay_t<Input>> and // at least one varray
              nt::tuple_of_accessor_pure_nd<std::decay_t<Output>, 1> and
              nt::tuple_of_accessor_value_or_empty<std::decay_t<Reduced>>)
    NOA_NOINLINE void reduce_ewise(
        const Shape<Index, N>& shape,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream
    ) {
        // Collapse contiguous dimensions.
        const auto shape_iz = shape.template as_safe<isize>();
        const auto contiguity = nd::accessors_contiguity(shape, input);
        const auto broadcasting = nd::accessors_broadcasting(shape, input);
        auto collapsed_shape = noa::collapse_contiguous_dimensions(shape_iz, contiguity, broadcasting);
        collapsed_shape = collapsed_shape.permute(squeeze_empty_dimensions_left(collapsed_shape));

        // Reshape the accessors to the new shape.
        auto input_collapsed = std::forward<Input>(input);
        check(nd::reshape_accessors<true>(shape_iz, collapsed_shape, input_collapsed),
              "INTERNAL: reshape failed, please report this issue, shape={}, contiguity={}, broadcasting={}",
              shape_iz, contiguity, broadcasting);

        // Check whether we can use the 2d kernels.
        bool use_rows{true};
        if constexpr (N >= 3) {
            for (usize i{}; i < N - 2; ++i)
                if (collapsed_shape[i] > 1)
                    use_rows = false;
        }

        constexpr auto SMALL_THRESHOLD = Config::block_work_size * 4;
        if (use_rows) {
            const auto n_rows = N == 1 ? u32{1} : safe_cast<u32>(collapsed_shape[N - 2]);
            const auto width = safe_cast<Index>(collapsed_shape[N - 1]);
            if (width <= SMALL_THRESHOLD) {
                details::launch_reduce_ewise_small_rows<Config, N>(
                    std::forward<Op>(op), std::forward<Input>(input_collapsed), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), n_rows, width, stream
                );
            } else {
                details::launch_reduce_ewise_large_rows<Config, N>(
                    std::forward<Op>(op), std::forward<Input>(input_collapsed), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), n_rows, width, stream
                );
            }
            return;
        }

        if constexpr (N >= 3) {
            if (shape_iz.n_elements() <= SMALL_THRESHOLD) {
                details::launch_reduce_ewise_small_nd<Config>(
                    std::forward<Op>(op), std::forward<Input>(input_collapsed), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), collapsed_shape, false, stream
                );
            } else {
                details::launch_reduce_ewise_large_nd<Config>(
                    std::forward<Op>(op), std::forward<Input>(input_collapsed), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), collapsed_shape, false, stream
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
