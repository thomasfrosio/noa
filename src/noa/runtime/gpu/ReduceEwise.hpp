#pragma once
#include "noa/runtime/gpu/IncludeGuard.hpp"

#include "noa/base/Config.hpp"
#include "noa/base/Vec.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/gpu/Allocators.hpp"
#include "noa/runtime/gpu/Block.hpp"
#include "noa/runtime/gpu/ComputeHandle.hpp"

#if defined(NOA_COMPILER_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wduplicated-branches"
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(push, 0)
#endif

// These reduction kernels are adapted from different sources, but the main logic comes from:
//  - https://github.com/NVIDIA/cuda-samples/tree/master/Samples/reduction
//  - https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

namespace noa::gpu::details {
    // 1d grid of 1d blocks.
    // Each block writes one element in "joined", which should thus have as many elements as there are blocks.
    template<typename Config, u32 MaxVectorSize, bool IsFinal = false>
    struct ReduceEwiseRowConfig {
        static_assert(is_power_of_2(Config::n_elements_per_thread));
        static_assert(
            is_power_of_2(Config::block_size) and
            Constant::WARP_SIZE <= Config::block_size and
            Config::block_size <= Limits::MAX_THREADS);

        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = Config::block_size;
        static constexpr u32 block_size_x = block_size;
        static constexpr u32 block_size_y = 1;
        static constexpr u32 block_size_z = 1;
        static constexpr u32 block_ndim = 1;
        static constexpr u32 n_elements_per_thread = max(MaxVectorSize, Config::n_elements_per_thread);
        static constexpr u32 block_work_size = block_size * n_elements_per_thread;
        static constexpr bool is_final = IsFinal;
    };

    template<typename Config, u32 BlockSizeX, u32 MaxVectorSize, bool IsFinal = false>
    struct ReduceEwiseNdConfig {
        static_assert(is_power_of_2(Config::n_elements_per_thread));
        static_assert(
            is_power_of_2(Config::block_size) and
            Constant::WARP_SIZE <= Config::block_size and
            Config::block_size <= Limits::MAX_THREADS);

        using interface = Config::interface;
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = Config::block_size;
        static constexpr u32 block_size_x = is_multiple_of(Config::block_size, BlockSizeX) ? min(BlockSizeX, Config::block_size) : Constant::WARP_SIZE;
        static constexpr u32 block_size_y = block_size / block_size_x;
        static constexpr u32 block_size_z = 1;
        static constexpr u32 block_ndim = block_size_y == 1 ? 1 : 2;
        static constexpr u32 n_elements_per_thread_x = max(MaxVectorSize, Config::n_elements_per_thread);
        static constexpr u32 block_work_size_x = block_size_x * n_elements_per_thread_x;
        static constexpr bool is_final = IsFinal;
    };

    template<u32 GRID_NDIM, bool IS_STATIC_SCRATCH_ENABLED,
             typename Block, typename Interface, typename Op, typename Index,
             typename Input, typename InputAlignedBuffers, typename Reduced, typename Output>
    struct ReduceEwiseRows {
        static_assert(Block::block_ndim == 1 and (GRID_NDIM == 1 or GRID_NDIM == 2));
        static constexpr bool SINGLE_PASS = Block::is_final;
        static constexpr bool DOUBLE_PASS = not SINGLE_PASS;
        using handle_type = ComputeHandle<Index, GRID_NDIM, Block, DOUBLE_PASS, false, DOUBLE_PASS>;
        static constexpr auto BLOCK_WORK_SIZE = static_cast<Index>(Block::block_work_size);

        NOA_NO_UNIQUE_ADDRESS Op op;
        NOA_NO_UNIQUE_ADDRESS Input input;
        NOA_NO_UNIQUE_ADDRESS Reduced reduced;
        NOA_NO_UNIQUE_ADDRESS Output output;
        Index width;

    public:
        // Reduce rows, each row is reduced individually.
        // N=1: 1d accessors, 1 row to reduce, 1d grid.
        // N=2: 2d accessors, n row(s) to reduce, 2d grid (y is n).
        NOA_FD void operator()(const handle_type& handle) {
            const auto row = SINGLE_PASS ? handle.block_index_x() : handle.block_index_y();
            const auto bid = SINGLE_PASS ? 0 :handle.block_index_x();
            const auto tid = handle.template thread_index_x<i32>();
            const auto starting_column_index = BLOCK_WORK_SIZE * bid;
            const auto grid_work_size = BLOCK_WORK_SIZE * (SINGLE_PASS ? 1 : handle.grid_size_x());

            auto input_1d = std::move(input).map([row]<typename T>(T&& accessor) {
                if constexpr (nt::is_accessor_value_v<T>) {
                    return NOA_FWD(accessor);
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

            Interface::init(handle, op, row);

            for (Index cid = starting_column_index; cid < width; cid += grid_work_size) {
                input_1d.for_each_enumerate([this, cid]<usize I>(auto& accessor) {
                    accessor.reset_pointer(input[Tag<I>{}].get());
                    accessor.offset_inplace(cid);
                });
                block_call_ewise_1d<Block::block_size, Block::n_elements_per_thread, InputAlignedBuffers, Interface>(
                    handle, op, input_1d, width - cid, reduced, tid);
            }

            Interface::deinit(handle, op, row);

            if constexpr (SINGLE_PASS) {
                // There's one block per row, so compute the reduced value for the block
                // and save it in the output at the row index.
                block_join_and_post<Interface, Block::block_size, IS_STATIC_SCRATCH_ENABLED>(
                    handle, op, reduced, output, tid, row);
            } else {
                // The output is the "joined" buffer, which is a buffer with one value per block and per row.
                // These values will then be reduced by the second reduction kernel (see below).
                block_join<Interface, Block::block_size, IS_STATIC_SCRATCH_ENABLED>(
                    handle, op, reduced, output, tid, row, bid);
            }
        }
    };

    // Here the input is organized has a series of (H=B..)rows. Given the original (B..)W shape of the input and the row
    // index, we can derive the (B..) indices. Each dimension can have an arbitrary stride, but if the rows themselves
    // are contiguous (W stride is 1), then vectorized load/stores can be used to load/store elements from the rows.
    // This kernel explicitly supports per-batch reductions (see reduce_axes_ewise).
    template<usize N, bool IS_STATIC_SCRATCH_ENABLED,
             typename Block, typename Interface, typename Op, typename Index,
             typename Input, typename InputAlignedBuffers, typename Reduced, typename Output>
    struct ReduceEwiseND {
        // If SINGLE_PASS=true, the grid is always 1D, but the compute handle requires grid_ndim >= block_ndim,
        // so use block_ndim (1 or 2). This has no effect in CUDA (grid_size_y is never retrieved), and in
        // SYCL it simply bumps nd_item<1> to nd_item<2>. Since SINGLE_PASS isn't a multi-grid kernel, the handle
        // ctor has no argument so the launch site doesn't need to be aware of this quirk.
        static constexpr bool SINGLE_PASS = Block::is_final;
        static constexpr bool DOUBLE_PASS = not SINGLE_PASS;
        static constexpr u32 GRID_NDIM = DOUBLE_PASS ? 2 : Block::block_ndim;
        using handle_type = ComputeHandle<Index, GRID_NDIM, Block, DOUBLE_PASS, false, DOUBLE_PASS>;

        NOA_NO_UNIQUE_ADDRESS Op op;
        NOA_NO_UNIQUE_ADDRESS Input input;
        NOA_NO_UNIQUE_ADDRESS Reduced reduced;
        NOA_NO_UNIQUE_ADDRESS Output output;
        Index n_rows;
        Index width;
        Vec<Index, N - 2> outer_shape_no_leftmost;

    public:
        NOA_FD void operator()(const handle_type& handle) {
            // SINGLE_PASS: 1D grid along x, one 1|2D block per batch. There's no "joined" buffer,  the output is
            // passed directly. Otherwise, two passes: 2D grid (grid_size_x is the number of blocks along W to reduce
            // a given batch, grid_size_y is the number of batches) and the output should be the "joined" 2d accessors,
            // where the outer dimension is the batch.
            const auto tid = handle.template thread_indices<i32, 2>();
            const auto batch = SINGLE_PASS ? handle.block_index_x() : handle.block_index_y();
            const auto bid = SINGLE_PASS ? 0 : handle.block_index_x();
            const auto n_blocks_per_batch = SINGLE_PASS ? 1 : handle.grid_size_x();
            const auto n_rows_per_block = static_cast<Index>(Block::block_size_y);
            const auto n_rows_per_grid = n_blocks_per_batch * n_rows_per_block;
            const auto initial_row = bid * n_rows_per_block + static_cast<Index>(tid[0]);

            auto input_1d = std::move(input).map([batch]<typename T>(T&& accessor) {
                if constexpr (nt::is_accessor_value_v<T>) {
                    return NOA_FWD(accessor);
                } else {
                    // Offset the input accessor to the current batch,
                    // so that it can be used later to reset the pointer.
                    accessor.offset_inplace(batch);

                    // Return the row accessors.
                    using value_t = nt::value_type_t<T>;
                    using index_t = nt::index_type_t<T>;
                    constexpr auto R = T::SIZE;
                    constexpr auto S = T::STRIDES_TRAIT;
                    constexpr auto P = T::POINTER_TRAIT;
                    return Accessor<value_t, 1, index_t, S, P>(accessor.get(), Strides{accessor.template stride<R - 1>()});
                }
            });

            Interface::init(handle, op, batch);

            for (Index row = initial_row; row < n_rows; row += n_rows_per_grid) { // for every row (within a batch)
                Vec<Index, N - 1> outer_indices;
                if constexpr (N == 2)
                    outer_indices = row;
                else
                    outer_indices = noa::offset2index(row, outer_shape_no_leftmost);

                for (Index cid = 0; cid < width; cid += Block::block_work_size_x) { // consume the row
                    input_1d.for_each_enumerate([this, &outer_indices, &cid]<size_t I>(auto& accessor_1d) {
                        if constexpr (not nt::is_accessor_value_v<decltype(accessor_1d)>) {
                            auto& accessor = input[Tag<I>{}];
                            auto new_pointer = accessor.offset_pointer(accessor.get(), outer_indices.push_back(cid));
                            accessor_1d.reset_pointer(new_pointer);
                        }
                    });
                    block_call_ewise_1d<Block::block_size_x, Block::n_elements_per_thread_x, InputAlignedBuffers, Interface>(
                        handle, op, input_1d, width - cid, reduced, tid[1]);
                }
            }

            Interface::deinit(handle, op, batch);

            const auto linear_tid = tid[0] * static_cast<i32>(Block::block_size_x) + tid[1];
            if constexpr (SINGLE_PASS)
                block_join_and_post<Interface, Block::block_size, IS_STATIC_SCRATCH_ENABLED>(
                    handle, op, reduced, output, linear_tid, batch);
            else
                block_join<Interface, Block::block_size, IS_STATIC_SCRATCH_ENABLED>(
                    handle, op, reduced, output, linear_tid, batch, bid);
        }
    };

    // One 1d block per batch to finish joining the reduced values and compute the final output.
    template<bool IS_STATIC_SCRATCH_ENABLED,
             typename Block, typename Interface, typename Op, typename Index,
             typename Joined, typename JoinedAlignedBuffers, typename Reduced, typename Output>
    struct ReduceEwiseSecond {
        using handle_type = ComputeHandle<Index, 1, Block, false, false, true>;

        NOA_NO_UNIQUE_ADDRESS Op op;
        NOA_NO_UNIQUE_ADDRESS Joined joined; // Tuple of 2d Accessor(s) corresponding to Reduced
        NOA_NO_UNIQUE_ADDRESS Reduced reduced; // Tuple of AccessorValue(s)
        NOA_NO_UNIQUE_ADDRESS Output output; // Tuple of 1d Accessor(s)
        Index n_elements;

    public:
        NOA_FD void operator()(const handle_type& handle) {
            const Index batch = handle.block_index_x();
            const auto tid = handle.template thread_index_x<i32>();

            if constexpr (not nt::empty_tuple<Reduced>) {
                auto joined_1d = joined.map([&](auto& accessor) { return accessor[batch]; });
                for (Index cid = 0; cid < n_elements; cid += Block::block_work_size) {
                    block_join_ewise_1d<Block::block_size, Block::n_elements_per_thread, JoinedAlignedBuffers, Interface>(
                        op, joined_1d, n_elements - cid, reduced, tid);
                    joined_1d.for_each([](auto& accessor) {
                        constexpr auto OFFSET = Block::block_work_size;
                        accessor.offset_inplace(OFFSET);
                    });
                }
            }

            block_join_and_post<Interface, Block::block_size, IS_STATIC_SCRATCH_ENABLED>(
                handle, op, reduced, output, tid, batch);
        }
    };
}

namespace noa::gpu::details {
    // nvcc bug - this could be a lambda, but nvcc <=12.6 is broken...
    template<usize ALIGNMENT, usize N, typename Config,
             typename Op, typename Index,
             typename Input, typename Reduced, typename Output>
    void launch_reduce_ewise_small_rows_(
        Op&& op,
        Input&& input_collapsed,
        Reduced&& reduced,
        Output&& output,
        u32 n_rows,
        Index width,
        Stream& stream
    ) {
        constexpr usize VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        using Block = ReduceEwiseRowConfig<Config, VEC_SIZE, true>;

        constexpr auto AXES = Vec{N - 2, N - 1}.template pop_front<(N == 1 ? 1 : 0)>();
        constexpr auto CONFIG = nd::AccessorConfig{
            .enforce_contiguous = is_vectorized<InputVec>(),
            .enforce_restrict = false,
        };

        auto input_rows = nd::reconfig_accessors<CONFIG>(NOA_FWD(input_collapsed), AXES);
        const auto grid_x = GridX(n_rows, 1);
        check(grid_x.n_launches() == 1);

        using Reduced_ = std::decay_t<Reduced>;
        static constexpr bool IS_STATIC_SCRATCH_ENABLED = is_static_scratch_enabled();
        constexpr bool HAS_REDUCED = not nt::empty_tuple<Reduced_>;
        const auto config = LaunchConfig{
            .grid_shape = Vec<u32, 3>{1, 1, grid_x.n_blocks(0)},
            .block_shape = block_shape<Block>(),
            .n_bytes_of_scratch =
                IS_STATIC_SCRATCH_ENABLED or not HAS_REDUCED ?
                0 : Block::block_size * sizeof(Reduced_),
        };

        using Interface = Config::interface;
        using InputRows = decltype(input_rows);
        using Kernel = ReduceEwiseRows<
            1, IS_STATIC_SCRATCH_ENABLED, Block, Interface,
            std::decay_t<Op>, Index, InputRows, InputVec,
            Reduced_, std::decay_t<Output>>;
        auto handle = typename Kernel::handle_type{};
        auto kernel = Kernel{op, input_rows, reduced, output, width};
        stream.enqueue(config, handle, kernel);
    }

    template<typename Config, usize N,
             typename Op, typename Index,
             typename Input, typename Reduced, typename Output>
    void launch_reduce_ewise_small_rows(
        Op&& op,
        Input&& input_collapsed, // (H:n_rows, W:width) with other (outer) dimensions empty
        Reduced&& reduced,
        Output&& output, // (H:n_rows)
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
                    NOA_FWD(op), NOA_FWD(input_collapsed), NOA_FWD(reduced), NOA_FWD(output),n_rows, width, stream
                );
            }
            if (alignment == 8) {
                return launch_reduce_ewise_small_rows_<8, N, Config>(
                    NOA_FWD(op), NOA_FWD(input_collapsed), NOA_FWD(reduced), NOA_FWD(output), n_rows, width, stream
                );
            }
            if (alignment == 4) {
                return launch_reduce_ewise_small_rows_<4, N, Config>(
                    NOA_FWD(op), NOA_FWD(input_collapsed), NOA_FWD(reduced), NOA_FWD(output), n_rows, width, stream
                );
            }
            if (alignment == 2) {
                return launch_reduce_ewise_small_rows_<2, N, Config>(
                    NOA_FWD(op), NOA_FWD(input_collapsed), NOA_FWD(reduced), NOA_FWD(output), n_rows, width, stream
                );
            }
        }
        details::launch_reduce_ewise_small_rows_<1, N, Config>(
            NOA_FWD(op), NOA_FWD(input_collapsed), NOA_FWD(reduced), NOA_FWD(output), n_rows, width, stream
        );
    }

    // nvcc bug - this could be a lambda, but nvcc <=12.6 is broken...
    template<usize ALIGNMENT, typename Config,
             typename Input, typename Reduced, typename Output,
             typename Index, usize N, typename Op> requires (N >= 2)
    void launch_reduce_ewise_small_nd_(
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Shape<Index, N> shape,
        bool keep_outermost,
        Stream& stream
    ) {
        // The input cannot be interpreted as a 1d or 2d array. So the outer dimensions (excluding the outermost if
        // keep_outermost==true) are fused in a set of rows. Each block reduces at least one row.
        // If the innermost dimension is contiguous, blocks can use vectorize loads to read their row(s).
        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;

        constexpr auto CONFIG = nd::AccessorConfig{.enforce_contiguous = is_vectorized<InputVec>()};
        auto input_nd = nd::reconfig_accessors<CONFIG>(NOA_FWD(input));

        // One 2D block to reduce n_rows.
        // GridX for one block per reduction (batch).
        const auto n_reductions = keep_outermost ? shape[0] : 1;
        Index n_rows{1};
        for (usize i = keep_outermost; i < N - 1; ++i)
            n_rows *= shape[i];

        constexpr u32 N_THREADS_X = Constant::WARP_SIZE;
        using Block = ReduceEwiseNdConfig<Config, N_THREADS_X, VEC_SIZE, true>;
        using Interface = Config::interface;
        using Reduced_ = std::decay_t<Reduced>;
        static constexpr bool IS_STATIC_SCRATCH_ENABLED = is_static_scratch_enabled();
        constexpr bool HAS_REDUCED = not nt::empty_tuple<Reduced_>;

        const auto grid_x = GridX(n_reductions, 1);
        check(grid_x.n_launches() == 1);
        const auto config = LaunchConfig{
            .grid_shape = Vec<u32, 3>{1, 1, grid_x.n_blocks(0)},
            .block_shape = block_shape<Block>(),
            .n_bytes_of_scratch =
                IS_STATIC_SCRATCH_ENABLED or not HAS_REDUCED ?
                0 : Block::block_size * sizeof(Reduced_),
        };

        using Kernel = ReduceEwiseND<
            N, IS_STATIC_SCRATCH_ENABLED, Block, Interface,
            std::decay_t<Op>, Index, decltype(input_nd), InputVec,
            Reduced_, std::decay_t<Output>>;
        auto handle = typename Kernel::handle_type{};
        auto kernel = Kernel{op, input_nd, reduced, output, n_rows, shape[N - 1], shape.vec.pop_front().pop_back()};
        stream.enqueue(config, handle, kernel);
    }

    template<typename Config,
             typename Input, typename Reduced, typename Output,
             typename Index, usize N, typename Op> requires (N >= 2)
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
                    NOA_FWD(op), NOA_FWD(input), NOA_FWD(reduced), NOA_FWD(output), shape, keep_outermost, stream
                );
            }
            if (alignment == 8) {
                return launch_reduce_ewise_small_nd_<8, Config>(
                    NOA_FWD(op), NOA_FWD(input), NOA_FWD(reduced), NOA_FWD(output), shape, keep_outermost, stream
                );
            }
            if (alignment == 4) {
                return launch_reduce_ewise_small_nd_<4, Config>(
                    NOA_FWD(op), NOA_FWD(input), NOA_FWD(reduced), NOA_FWD(output), shape, keep_outermost, stream
                );
            }
            if (alignment == 2) {
                return launch_reduce_ewise_small_nd_<2, Config>(
                    NOA_FWD(op), NOA_FWD(input), NOA_FWD(reduced), NOA_FWD(output), shape, keep_outermost, stream
                );
            }
        }
        details::launch_reduce_ewise_small_nd_<1, Config>(
            NOA_FWD(op), NOA_FWD(input), NOA_FWD(reduced), NOA_FWD(output), shape, keep_outermost, stream
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
        using Block = ReduceEwiseRowConfig<Config, VEC_SIZE, false>;
        using Interface = Config::interface;

        constexpr usize ACTUAL_N = N == 1 ? 1 : 2;
        constexpr auto AXES = Vec{N - 2, N - 1}.template pop_front<(N == 1 ? 1 : 0)>();
        constexpr auto CONFIG = nd::AccessorConfig{
            .enforce_contiguous = is_vectorized<InputVec>(),
            .enforce_restrict = false,
        };

        auto input_rows = nd::reconfig_accessors<CONFIG>(NOA_FWD(input_collapsed), AXES);
        auto grid_size = Vec<u32, ACTUAL_N - 1>{};
        auto grid_offset = Vec<u32, ACTUAL_N - 1>{};
        if constexpr (N > 1)
            grid_size[0] = static_cast<u32>(grid_y.n_blocks_total());

        static constexpr bool IS_STATIC_SCRATCH_ENABLED = is_static_scratch_enabled();
        constexpr bool HAS_REDUCED = not nt::empty_tuple<Reduced>;

        for (u32 y{}; y < grid_y.n_launches(); ++y) {
            const auto config = LaunchConfig{
                .grid_shape = Vec<u32, 3>{1, grid_y.n_blocks(y), n_blocks_x},
                .block_shape = block_shape<Block>(),
                .n_bytes_of_scratch =
                    IS_STATIC_SCRATCH_ENABLED or not HAS_REDUCED ?
                    0 : Block::block_size * sizeof(Reduced),
            };
            if constexpr (N > 1) {
                nd::offset_accessors(Vec{grid_y.offset_additive(y)}, input_rows, joined);
                grid_offset[0] = grid_y.offset(y);
            }

            using Kernel = ReduceEwiseRows<
                ACTUAL_N, IS_STATIC_SCRATCH_ENABLED, Block, Interface, Op, Index,
                decltype(input_rows), InputVec, Reduced, Joined>;
            auto handle = typename Kernel::handle_type(grid_size, grid_offset);
            auto kernel = Kernel{op, input_rows, reduced, joined, width};
            stream.enqueue(config, handle, kernel);
        }
    }

    template<typename Config, usize N,
             typename Op, typename Index,
             typename Input, typename Reduced, typename Output>
    void launch_reduce_ewise_large_rows(
        Op&& op,
        Input&& input_collapsed, // (H:n_rows, W:width) with other (outer) dimensions empty
        Reduced&& reduced,
        Output&& output, // (H:n_rows)
        u32 n_rows,
        Index width,
        Stream& stream
    ) {
        // In this config, the inputs can be interpreted as 1d or 2d arrays. If the innermost dimension is contiguous,
        // i.e., if all elements to reduce are contiguous, we can vectorize loads for the first kernel.

        // First kernel, limit the number of blocks, otherwise the second kernel would have too much work to do.
        // This is not true if there's no value to reduce, in which case the grid can be maximized.
        check(N > 1 or n_rows == 1);
        const auto grid_x = Grid<Config::max_grid_size>(width, Config::block_work_size);
        const auto grid_y = GridY(n_rows, 1);
        const auto n_blocks_x = grid_x.n_blocks(0);
        const auto n_blocks_y = safe_cast<u32>(grid_y.n_blocks_total());

        // Allocate the joined buffer.
        // If Reduced is an empty tuple, Joined is empty too and no allocation is performed.
        using Joined = joined_tuple_t<2, Index, Reduced>; // Tuple<AccessorRestrictContiguous<T, 2, Index>,...>
        constexpr size_t JOINED_VEC_SIZE = maximum_allowed_aligned_buffer_size<16, Joined>();
        using JoinedVec = to_aligned_buffer_t<Joined, 16, JOINED_VEC_SIZE>;
        Joined joined;
        [[maybe_unused]] auto joined_buffers = get_joined_buffer(
            n_blocks_x, n_blocks_y, joined, JOINED_VEC_SIZE, stream);

        if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
            auto shape_without_width = Shape<u32, N - 1>::from_value(1);
            if constexpr (N >= 2)
                shape_without_width[N - 2] = n_rows;
            const auto alignment = min_address_alignment(input_collapsed, shape_without_width);

            if (alignment == 16) {
                launch_reduce_ewise_large_rows_<16, N, Config>(
                    op, NOA_FWD(input_collapsed), reduced, joined, stream, width, n_blocks_x, grid_y
                );
            } else if (alignment == 8) {
                launch_reduce_ewise_large_rows_<8, N, Config>(
                    op, NOA_FWD(input_collapsed), reduced, joined, stream, width, n_blocks_x, grid_y
                );
            } else if (alignment == 4) {
                launch_reduce_ewise_large_rows_<4, N, Config>(
                    op, NOA_FWD(input_collapsed), reduced, joined, stream, width, n_blocks_x, grid_y
                );
            } else if (alignment == 2) {
                launch_reduce_ewise_large_rows_<2, N, Config>(
                    op, NOA_FWD(input_collapsed), reduced, joined, stream, width, n_blocks_x, grid_y
                );
            } else {
                launch_reduce_ewise_large_rows_<1, N, Config>(
                    op, NOA_FWD(input_collapsed), reduced, joined, stream, width, n_blocks_x, grid_y
                );
            }
        } else {
            launch_reduce_ewise_large_rows_<1, N, Config>(
                op, NOA_FWD(input_collapsed), reduced, joined, stream, width, n_blocks_x, grid_y
            );
        }

        // Second kernel.
        // TODO If Reduced is an empty tuple, check op has a valid post, otherwise don't even run this.
        using SecondBlock = ReduceEwiseRowConfig<Config, JOINED_VEC_SIZE>;
        using Interface = Config::interface;
        using Reduced_ = std::decay_t<Reduced>;
        constexpr bool HAS_REDUCED = not nt::empty_tuple<Reduced_>;
        static constexpr bool IS_STATIC_SCRATCH_ENABLED = is_static_scratch_enabled();
        using Kernel = ReduceEwiseSecond<
            IS_STATIC_SCRATCH_ENABLED, SecondBlock, Interface, std::decay_t<Op>,
            Index, Joined, JoinedVec, Reduced_, std::decay_t<Output>>;
        auto config = LaunchConfig{
            .grid_shape = Vec<u32, 3>{1, 1, n_blocks_y},
            .block_shape = Vec<u32, 3>{1, 1, HAS_REDUCED ? Config::block_size : 1},
            .n_bytes_of_scratch =
                IS_STATIC_SCRATCH_ENABLED or not HAS_REDUCED ?
                0 : SecondBlock::block_size * sizeof(Reduced_),
        };
        auto handle = typename Kernel::handle_type{};
        auto kernel = Kernel{NOA_FWD(op), joined, NOA_FWD(reduced), NOA_FWD(output), n_blocks_x};
        stream.enqueue(config, handle, kernel);
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
        u32 n_blocks_x,
        const GridY& grid_y
    ) {
        // Prepare the input for vectorization.
        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        constexpr auto CONFIG = nd::AccessorConfig{.enforce_contiguous = is_vectorized<InputVec>()};
        auto input_nd = nd::reconfig_accessors<CONFIG>(NOA_FWD(input));

        // Launch the kernel.
        using Block = ReduceEwiseNdConfig<Config, BLOCK_SIZE_X, VEC_SIZE>;
        using Interface = Config::interface;
        using InputND = decltype(input_nd);
        constexpr bool HAS_REDUCED = not nt::empty_tuple<Reduced>;
        static constexpr bool IS_STATIC_SCRATCH_ENABLED = is_static_scratch_enabled();

        for (u32 y{}; y < grid_y.n_launches(); y++) {
            nd::offset_accessors(Vec{grid_y.offset_additive(y)}, input_nd, joined);

            const auto config = LaunchConfig{
                .grid_shape = Vec<u32, 3>{1, grid_y.n_blocks(y), n_blocks_x},
                .block_shape = block_shape<Block>(),
                .n_bytes_of_scratch =
                    IS_STATIC_SCRATCH_ENABLED or not HAS_REDUCED ?
                    0 : Block::block_size * sizeof(Reduced),
            };

            using Kernel = ReduceEwiseND<
                N, IS_STATIC_SCRATCH_ENABLED, Block, Interface, Op,
                Index, InputND, InputVec, Reduced, Joined>;
            auto handle = typename Kernel::handle_type(Vec{grid_y.n_blocks_total()}.as<u32>(), Vec{grid_y.offset(y)});
            auto kernel = Kernel{op, input_nd, reduced, joined, n_rows, shape[N - 1], shape.vec.pop_front().pop_back()};
            stream.enqueue(config, handle, kernel);
        }
    }

    template<typename Config,
             typename Input, typename Reduced, typename Output,
             typename Index, usize N, typename Op> requires (N >= 2)
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
        using large_block_t = ReduceEwiseNdConfig<Config, 256, 1>;
        using small_block_t = ReduceEwiseNdConfig<Config, 64, 1>;
        u32 n_threads_x;
        u32 n_threads_y;
        if (shape[N - 1] > 512) { // TODO better heuristic
            n_threads_x = large_block_t::block_size_x;
            n_threads_y = large_block_t::block_size_y;
        } else {
            n_threads_x = small_block_t::block_size_x;
            n_threads_y = small_block_t::block_size_y;
        }

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
        [[maybe_unused]] auto joined_buffers = get_joined_buffer(
            n_blocks_x, n_blocks_y, joined, JOINED_VEC_SIZE, stream);

        // TODO Reduce the number of instantiations by adding support of a runtime block_size_x.
        //      That would mean using dynamic shared memory in block_reduce, which is fine...
        // Note that these if statements are likely to instantiate the same kernel. For instance, if the smallest
        // type has an alignment of 4, only 6 kernels are created (3 with n_threads_x=256, 3 with n_threads_x=64).
        // The worst case is when the smallest type is aligned to only one byte, in which case 10 kernels are created.
        if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
            const size_t alignment = min_address_alignment(input, shape.pop_back());
            if (n_threads_x == large_block_t::block_size_x) {
                if (alignment == 16) {
                    launch_reduce_ewise_large_nd_<large_block_t::block_size_x, 16, Config>(
                        op, NOA_FWD(input), reduced, joined, shape, stream, n_rows, n_blocks_x, grid_y
                    );
                } else if (alignment == 8) {
                    launch_reduce_ewise_large_nd_<large_block_t::block_size_x, 8, Config>(
                        op, NOA_FWD(input), reduced, joined, shape, stream, n_rows, n_blocks_x, grid_y
                    );
                } else if (alignment == 4) {
                    launch_reduce_ewise_large_nd_<large_block_t::block_size_x, 4, Config>(
                        op, NOA_FWD(input), reduced, joined, shape, stream, n_rows, n_blocks_x, grid_y
                    );
                } else if (alignment == 2) {
                    launch_reduce_ewise_large_nd_<large_block_t::block_size_x, 2, Config>(
                        op, NOA_FWD(input), reduced, joined, shape, stream, n_rows, n_blocks_x, grid_y
                    );
                } else {
                    launch_reduce_ewise_large_nd_<large_block_t::block_size_x, 1, Config>(
                        op, NOA_FWD(input), reduced, joined, shape, stream, n_rows, n_blocks_x, grid_y
                    );
                }
            } else {
                if (alignment == 16) {
                    launch_reduce_ewise_large_nd_<small_block_t::block_size_x, 16, Config>(
                        op, NOA_FWD(input), reduced, joined, shape, stream, n_rows, n_blocks_x, grid_y
                    );
                } else if (alignment == 8) {
                    launch_reduce_ewise_large_nd_<small_block_t::block_size_x, 8, Config>(
                        op, NOA_FWD(input), reduced, joined, shape, stream, n_rows, n_blocks_x, grid_y
                    );
                } else if (alignment == 4) {
                    launch_reduce_ewise_large_nd_<small_block_t::block_size_x, 4, Config>(
                        op, NOA_FWD(input), reduced, joined, shape, stream, n_rows, n_blocks_x, grid_y
                    );
                } else if (alignment == 2) {
                    launch_reduce_ewise_large_nd_<small_block_t::block_size_x, 2, Config>(
                        op, NOA_FWD(input), reduced, joined, shape, stream, n_rows, n_blocks_x, grid_y
                    );
                } else {
                    launch_reduce_ewise_large_nd_<small_block_t::block_size_x, 1, Config>(
                        op, NOA_FWD(input), reduced, joined, shape, stream, n_rows, n_blocks_x, grid_y
                    );
                }
            }
        } else {
            if (n_threads_x == large_block_t::block_size_x) {
                launch_reduce_ewise_large_nd_<large_block_t::block_size_x, 1, Config>(
                    op, NOA_FWD(input), reduced, joined, shape, stream, n_rows, n_blocks_x, grid_y
                );
            } else {
                launch_reduce_ewise_large_nd_<small_block_t::block_size_x, 1, Config>(
                    op, NOA_FWD(input), reduced, joined, shape, stream, n_rows, n_blocks_x, grid_y
                );
            }
        }

        // Second kernel.
        // TODO If Reduced is an empty tuple, check op has a valid post, otherwise don't even run this.
        using SecondBlock = ReduceEwiseRowConfig<Config, JOINED_VEC_SIZE>;
        using Interface = Config::interface;
        constexpr bool HAS_REDUCED = not nt::empty_tuple<Reduced>;
        static constexpr bool IS_STATIC_SCRATCH_ENABLED = is_static_scratch_enabled();
        using Kernel = ReduceEwiseSecond<
            IS_STATIC_SCRATCH_ENABLED, SecondBlock, Interface, std::decay_t<Op>,
            Index, Joined, JoinedVec, std::decay_t<Reduced>, std::decay_t<Output>>;
        auto config = LaunchConfig{
            .grid_shape = Vec<u32, 3>{1, 1, n_blocks_y},
            .block_shape = Vec<u32, 3>{1, 1, HAS_REDUCED ? Config::block_size : 1},
            .n_bytes_of_scratch =
                IS_STATIC_SCRATCH_ENABLED or not HAS_REDUCED ?
                0 : SecondBlock::block_size * sizeof(Reduced),
        };
        auto handle = typename Kernel::handle_type{};
        auto kernel = Kernel{NOA_FWD(op), joined, NOA_FWD(reduced), NOA_FWD(output), n_blocks_x};
        stream.enqueue(config, handle, kernel);
    }
}

namespace noa::gpu {
    template<bool ZipInput = false,
             bool ZipReduced = false,
             bool ZipOutput = false,
             u32 ElementsPerThread = 8,
             u32 BlockSize = 512,
             u32 MaxGridSize = 4096,
             bool EnableVectorization = true>
    struct ReduceEwiseConfig {
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
        // Assume layout is optimized and shape is collapsed.
        bool use_row_kernels{true};
        if constexpr (N > 1) {
            for (usize i{}; i < N - 1; ++i)
                if (shape[i] > 1)
                    use_row_kernels = false;
        }

        constexpr auto SMALL_THRESHOLD = Config::block_work_size * 4;
        if (use_row_kernels) {
            const auto width = safe_cast<Index>(shape[N - 1]);
            if (width <= SMALL_THRESHOLD) {
                details::launch_reduce_ewise_small_rows<Config, N>(
                    NOA_FWD(op), NOA_FWD(input), NOA_FWD(reduced),
                    NOA_FWD(output), 1, width, stream
                );
            } else {
                details::launch_reduce_ewise_large_rows<Config, N>(
                    NOA_FWD(op), NOA_FWD(input), NOA_FWD(reduced),
                    NOA_FWD(output), 1, width, stream
                );
            }
            return;
        }

        if constexpr (N >= 2) {
            if (shape.template as<isize>().n_elements() <= SMALL_THRESHOLD) {
                details::launch_reduce_ewise_small_nd<Config>(
                    NOA_FWD(op), NOA_FWD(input), NOA_FWD(reduced),
                    NOA_FWD(output), shape, false, stream
                );
            } else {
                details::launch_reduce_ewise_large_nd<Config>(
                    NOA_FWD(op), NOA_FWD(input), NOA_FWD(reduced),
                    NOA_FWD(output), shape, false, stream
                );
            }
        }
    }
}

#if defined(NOA_COMPILER_GCC)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif
