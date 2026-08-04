#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/base/Config.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/cuda/Block.cuh"
#include "noa/runtime/cuda/ReduceEwise.cuh"

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wduplicated-branches"
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(push, 0)
#endif

namespace noa::cuda::details {
    template<typename Config, u32 BlockSizeX, u32 MaxVectorSize>
    struct ReduceAxesEwiseWidthConfig {
        static constexpr u32 block_size = max(BlockSizeX, Config::block_size);
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = block_size / block_size_x;
        static constexpr u32 n_elements_per_thread_x = max(MaxVectorSize, Config::n_elements_per_thread);
        static constexpr u32 block_work_size_x = block_size_x * n_elements_per_thread_x;
    };

    template<typename Block, typename Interface, typename Op, typename Index, usize N,
             typename Input, typename InputAlignedBuffer, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_width_ewise(
        Op op,
        Input input,
        Shape<Index, 2> shape_hw,
        Reduced reduced,
        Output output,
        Vec<u32, (N <= 3 ? N - 2 : 2)> grid_outer_shape, // ((z,)y)
        Vec<u32, (N <= 3 ? N - 2 : 2)> grid_outer_offset, // ((z,)y)
        Vec<u32, (N <= 4 ? 0 : N - 4)> grid_fused_shape_in_x_unbatched
    ) {
        // Handle multigrid case by adding the Z and Y grid offset.
        constexpr usize GRID_NDIM = N <= 3 ? N - 1 : 3;
        auto bid = block_indices<u32, GRID_NDIM>();
        if constexpr (N >= 3)
            bid[0] += grid_outer_offset[0];
        if constexpr (N >= 4)
            bid[1] += grid_outer_offset[1];

        // Compute the current global indices.
        // The grid maps all axes except the width.
        // For N >= 5, the rightmost axis of the grid needs to be unfused.
        const auto tid = thread_indices<Index, 2>();
        Vec<Index, N> gid;
        if constexpr (N == 3) {
            gid = Vec<Index, 3>::from_values(
                bid[0],
                bid[1] * Block::block_size_y + tid[0],
                tid[1]);
        } else if constexpr (N == 4) {
            gid = Vec<Index, 4>::from_values(
                bid[0],
                bid[1],
                bid[2] * Block::block_size_y + tid[0],
                tid[1]);
        } else if constexpr (N == 5) {
            Vec<u32, 2> block_indices = noa::offset2index(bid[2], grid_fused_shape_in_x_unbatched);
            gid = Vec<Index, 5>::from_values(
                bid[0],
                bid[1],
                block_indices[0],
                block_indices[1] * Block::block_size_y + tid[0],
                tid[1]);
        } else if constexpr (N == 6) {
            Vec<u32, 3> block_indices = noa::offset2index(bid[2], grid_fused_shape_in_x_unbatched);
            gid = Vec<Index, 6>::from_values(
                bid[0],
                bid[1],
                block_indices[0],
                block_indices[1],
                block_indices[2] * Block::block_size_y + tid[0],
                tid[1]);
        } else {
            static_assert(nt::always_false<Interface>);
        }

        const auto ci = ComputeHandle<Index, GRID_NDIM, 2, true, false, false>(grid_outer_shape, grid_outer_offset);
        if constexpr (N == 2)
            Interface::init(ci, op, gid[0]);
        else if constexpr (N == 3)
            Interface::init(ci, op, gid[0], gid[1]);
        else if constexpr (N == 4)
            Interface::init(ci, op, gid[0], gid[1], gid[2]);
        else if constexpr (N == 5)
            Interface::init(ci, op, gid[0], gid[1], gid[2], gid[3]);
        else if constexpr (N == 6)
            Interface::init(ci, op, gid[0], gid[1], gid[2], gid[3], gid[4]);

        auto input_row = std::move(input).map([&gid]<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T>)
                return std::forward<T>(accessor);
            else
                return accessor[gid.pop_back()]; // 1D accessor
        });

        // Initial reduction. Loop until the end of the row is reached.
        constexpr auto OFFSET = Block::block_work_size_x;
        const bool is_valid_row = gid[N - 2] < shape_hw[0];
        for (Index cid = 0; cid < shape_hw[1] and is_valid_row; cid += OFFSET) {
            block_call_ewise_1d<Block::block_size_x, Block::n_elements_per_thread_x, InputAlignedBuffer, Interface>(
                ci, op, input_row, shape_hw[1] - cid, reduced, tid[1]);

            // Offset along the row to the next work space.
            input_row.for_each([](auto& accessor) { accessor.offset_inplace(OFFSET); });
        }

        if constexpr (N == 2)
            Interface::deinit(ci, op, gid[0]);
        else if constexpr (N == 3)
            Interface::deinit(ci, op, gid[0], gid[1]);
        else if constexpr (N == 4)
            Interface::deinit(ci, op, gid[0], gid[1], gid[2]);
        else if constexpr (N == 5)
            Interface::deinit(ci, op, gid[0], gid[1], gid[2], gid[3]);
        else if constexpr (N == 6)
            Interface::deinit(ci, op, gid[0], gid[1], gid[2], gid[3], gid[4]);

        if constexpr (not nt::empty_tuple<Reduced>) {
            // Share the threads' initial reduction with the rest of the block.
            __shared__ Uninitialized<Reduced> shared_buffer_[Block::block_size];
            auto* shared_buffer = reinterpret_cast<Reduced*>(shared_buffer_);
            Reduced* joined = shared_buffer + tid[0] * Block::block_size_x;
            joined[tid[1]] = reduced;
            block_synchronize();

            // Reduce shared data to one element.
            reduced = block_join_shared<Interface, Block::block_size_x>(op, joined, tid[1]);
        }

        if (gid[N - 1] == 0 and is_valid_row) {
            if constexpr (N == 2)
                Interface::post(op, reduced, output, gid[0]);
            else if constexpr (N == 3)
                Interface::post(op, reduced, output, gid[0], gid[1]);
            else if constexpr (N == 4)
                Interface::post(op, reduced, output, gid[0], gid[1], gid[2]);
            else if constexpr (N == 5)
                Interface::post(op, reduced, output, gid[0], gid[1], gid[2], gid[3]);
            else if constexpr (N == 6)
                Interface::post(op, reduced, output, gid[0], gid[1], gid[2], gid[3], gid[4]);
        }
    }

    template<typename Config, u32 BlockSizeX>
    struct ReduceAxesEwiseHeightConfig {
        static constexpr u32 block_size = max(BlockSizeX, Config::block_size);
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = block_size / block_size_x;
    };

    template<typename Block, typename Interface, typename Op, typename Index, usize N, typename Input, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_height_ewise(
        Op op,
        Input input,
        Shape<Index, 2> shape_hw,
        Reduced reduced,
        Output output,
        Vec<u32, (N <= 3 ? N - 2 : 2)> grid_outer_shape, // ((z,)y)
        Vec<u32, (N <= 3 ? N - 2 : 2)> grid_outer_offset, // ((z,)y)
        Vec<u32, (N <= 4 ? 0 : N - 4)> grid_fused_shape_in_x_unbatched
    ) {
        // Handle multigrid case by adding the Z and Y grid offset.
        constexpr usize GRID_NDIM = N <= 3 ? N - 1 : 3;
        auto bid = block_indices<u32, GRID_NDIM>();
        if constexpr (N >= 3)
            bid[0] += grid_outer_offset[0];
        if constexpr (N >= 4)
            bid[1] += grid_outer_offset[1];

        // Global indices. One block along the height.
        const auto tid = thread_indices<Index, 2>();
        Vec<Index, N> gid;
        if constexpr (N == 2) {
            gid = Vec<Index, 2>::from_values(
                tid[0],
                bid[0] * Block::block_size_x + tid[1]);
        } else if constexpr (N == 3) {
            gid = Vec<Index, 3>::from_values(
                bid[0],
                tid[0],
                bid[1] * Block::block_size_x + tid[1]);
        } else if constexpr (N == 4) {
            gid = Vec<Index, 4>::from_values(
                bid[0],
                bid[1],
                tid[0],
                bid[2] * Block::block_size_x + tid[1]);
        } else if constexpr (N == 5) {
            Vec<u32, 2> block_indices = noa::offset2index(bid[2], grid_fused_shape_in_x_unbatched);
            gid = Vec<Index, 5>::from_values(
                bid[0],
                bid[1],
                block_indices[0],
                tid[0],
                block_indices[1] * Block::block_size_x + tid[1]);
        } else if constexpr (N == 6) {
            Vec<u32, 3> block_indices = noa::offset2index(bid[2], grid_fused_shape_in_x_unbatched);
            gid = Vec<Index, 6>::from_values(
                bid[0],
                bid[1],
                block_indices[0],
                block_indices[1],
                tid[0],
                block_indices[2] * Block::block_size_x + tid[1]);
        } else {
            static_assert(nt::always_false<Interface>);
        }

        const auto ci = ComputeHandle<Index, GRID_NDIM, 2, true, false, false>(grid_outer_shape, grid_outer_offset);
        if constexpr (N == 2)
            Interface::init(ci, op, gid[1]);
        else if constexpr (N == 3)
            Interface::init(ci, op, gid[0], gid[2]);
        else if constexpr (N == 4)
            Interface::init(ci, op, gid[0], gid[1], gid[3]);
        else if constexpr (N == 5)
            Interface::init(ci, op, gid[0], gid[1], gid[2], gid[4]);
        else if constexpr (N == 6)
            Interface::init(ci, op, gid[0], gid[1], gid[2], gid[3], gid[5]);
        else
            static_assert(nt::always_false<Interface>);

        // Process every row.
        auto input_rows = std::move(input).map([&gid]<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T> or nt::accessor_pure_nd<T, 2>)
                return std::forward<T>(accessor);
            else
                return accessor[gid.template pop_back<2>()]; // 2D accessor
        });
        const bool is_valid_column = gid[N - 1] < shape_hw[1];
        for (Index tidy = gid[N - 2]; tidy < shape_hw[0] and is_valid_column; tidy += Block::block_size_y)
            Interface::call(ci, op, input_rows, reduced, tidy, gid[N - 1]);

        if constexpr (N == 2)
            Interface::deinit(ci, op, gid[1]);
        else if constexpr (N == 3)
            Interface::deinit(ci, op, gid[0], gid[2]);
        else if constexpr (N == 4)
            Interface::deinit(ci, op, gid[0], gid[1], gid[3]);
        else if constexpr (N == 5)
            Interface::deinit(ci, op, gid[0], gid[1], gid[2], gid[4]);
        else if constexpr (N == 6)
            Interface::deinit(ci, op, gid[0], gid[1], gid[2], gid[3], gid[5]);
        else
            static_assert(nt::always_false<Interface>);

        if constexpr (not nt::empty_tuple<Reduced>) {
            // Share the threads' initial reduction with the rest of the block.
            __shared__ Uninitialized<Reduced> shared_buffer_[Block::block_size];
            auto* shared_buffer = reinterpret_cast<Reduced*>(shared_buffer_);
            Reduced* joined = shared_buffer + threadIdx.y * Block::block_size_x + threadIdx.x;
            *joined = reduced;
            block_synchronize();

            // Reduce the height of the block.
            #pragma unroll
            for (u32 size = Block::block_size_y; size >= 2; size /= 2) {
                if (threadIdx.y < size / 2)
                    Interface::join(op, joined[Block::block_size_x * size / 2], *joined);
                block_synchronize();
            }
            reduced = *joined;
        }

        if (gid[N - 2] == 0 and is_valid_column) {
            if constexpr (N == 2)
                Interface::post(op, reduced, output, gid[1]);
            else if constexpr (N == 3)
                Interface::post(op, reduced, output, gid[0], gid[2]);
            else if constexpr (N == 4)
                Interface::post(op, reduced, output, gid[0], gid[1], gid[3]);
            else if constexpr (N == 5)
                Interface::post(op, reduced, output, gid[0], gid[1], gid[2], gid[4]);
            else if constexpr (N == 6)
                Interface::post(op, reduced, output, gid[0], gid[1], gid[2], gid[3], gid[5]);
            else
                static_assert(nt::always_false<Interface>);
        }
    }
}

namespace noa::cuda::details {
    template<size_t BLOCK_SIZE_X, size_t ALIGNMENT, typename Config,
             typename Op, typename Input, typename Reduced, typename Output, typename Index, usize N>
    void launch_reduce_ewise_width_(
        const Shape<Index, N>& shape,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream,
        dim3 n_threads
    ) {
        // The width of the output is empty/reduced, remove it.
        auto output_no_width = nd::reconfig_accessors(std::forward<Output>(output), Vec<usize, N - 1>::arange());
        using OutputNoWidth = decltype(output_no_width);

        // Prepare the input for vectorization.
        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        constexpr auto CONFIG = nd::AccessorConfig{.enforce_contiguous = is_vectorized<InputVec>()};
        auto input_nd = nd::reconfig_accessors<CONFIG>(std::forward<Input>(input));
        using InputND = decltype(input_nd);

        // Grid shape. Exclude the width, each block line reduces its row.
        // N==3 -> 2D grid.
        // N==4 -> 3D grid.
        // N>=5 -> 3D grid (with fused axes in X).
        constexpr usize GRID_POP_FRONT = N <= 3 ? 1 : 0;
        const auto grid = GridND(shape.pop_back(), Shape<Index, N - 1>::from_value(1).template set<N - 2>(n_threads.y));
        const auto grid_outer_shape = grid.outer_shape().vec.template pop_front<GRID_POP_FRONT>();
        const auto grid_fused_shape_in_x_unbatched = grid.fused_shape().vec.pop_front();
        check(grid.n_launches_x() == 1);

        // Launch the kernel(s).
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using Block = ReduceAxesEwiseWidthConfig<Config, BLOCK_SIZE_X, VEC_SIZE>;
        using Interface = Config::interface;

        for (u32 z{}; z < grid.n_launches_z(); z++) {
            for (u32 y{}; y < grid.n_launches_y(); y++) {
                const auto config = LaunchConfig{
                    .n_blocks = grid.dim3_shape_for_launch(z, y, 0),
                    .n_threads = n_threads,
                };
                const auto grid_outer_offset = grid.block_offset_for_launch(z, y).template pop_front<GRID_POP_FRONT>();
                stream.enqueue(
                    reduce_width_ewise<Block, Interface, OpDecay, Index, N, InputND, InputVec, ReducedDecay, OutputNoWidth>,
                    config, op, input_nd, shape.filter(N - 2, N - 1), reduced, output_no_width,
                    grid_outer_shape, grid_outer_offset, grid_fused_shape_in_x_unbatched
                );
            }
        }
    }

    template<typename Config, typename Op, typename Input, typename Reduced, typename Output, typename Index, usize N>
        requires (N >= 3)
    void launch_reduce_ewise_width(
        const Shape<Index, N>& shape,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream
    ) {
        // Block shape.
        u32 n_threads_x = shape[N - 1] > 512 ? 256u : 64u;
        if (not noa::is_multiple_of(Config::block_size, n_threads_x))
            n_threads_x = Constant::WARP_SIZE;
        const u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
        const auto n_threads = dim3(n_threads_x, n_threads_y);

        // TODO Reduce the number of instantiations by adding support of a runtime block_size_x.
        //      That would mean using dynamic shared memory in block_reduce, which is fine...
        // Note that these if statements are likely to instantiate the same kernel. For instance, if the smallest
        // type has an alignment of 4, only 6 kernels are created (3 with n_threads_x=256, 3 with n_threads_x=64).
        // The worst case is when the smallest type is aligned to only one byte, in which case 10 kernels are created.
        if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
            const size_t alignment = min_address_alignment(input, shape.pop_back());
            if (n_threads_x == 256) {
                if (alignment == 16) {
                    launch_reduce_ewise_width_<256, 16, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else if (alignment == 8) {
                    launch_reduce_ewise_width_<256, 8, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else if (alignment == 4) {
                    launch_reduce_ewise_width_<256, 4, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else if (alignment == 2) {
                    launch_reduce_ewise_width_<256, 2, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else {
                    launch_reduce_ewise_width_<256, 1, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                }
            } else {
                if (alignment == 16) {
                    launch_reduce_ewise_width_<64, 16, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else if (alignment == 8) {
                    launch_reduce_ewise_width_<64, 8, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else if (alignment == 4) {
                    launch_reduce_ewise_width_<64, 4, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else if (alignment == 2) {
                    launch_reduce_ewise_width_<64, 2, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else {
                    launch_reduce_ewise_width_<64, 1, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                }
            }
        } else {
            if (n_threads_x == 256) {
                launch_reduce_ewise_width_<256, 1, Config>(
                    shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), stream, n_threads
                );
            } else {
                launch_reduce_ewise_width_<64, 1, Config>(
                    shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), stream, n_threads
                );
            }
        }
    }

    template<typename Config, typename Op, typename Input, typename Reduced, typename Output, typename Index, usize N>
        requires (N >= 2)
    void launch_reduce_ewise_height(
        const Shape<Index, N>& shape,
        const Vec<bool, N>& axes_to_reduce,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream
    ) {
        // First copy|move the input and output since they'll need to be reordered.
        auto input_ = std::forward<Input>(input);
        auto output_ = std::forward<Output>(output);

        // The kernel needs the axis to reduce at the "height" position.
        // The width should still be at the rightmost dimension.
        auto order = noa::squeeze_empty_dimensions_left(axes_to_reduce.template as<i32>() + 1);
        std::swap(order[N - 1], order[N - 2]);

        // Reorder to (X, X, axis_to_reduce, width).
        const auto reordered_shape = shape.permute(order);
        input_.for_each([&order](auto& accessor) { accessor = accessor.permute(order); });
        output_.for_each([&order](auto& accessor) { accessor = accessor.permute(order); });

        // Remove the empty/reduced axis from the output.
        auto order_output_dim = Vec<u32, N - 1>::arange().template set<N - 2>(N - 1);
        auto reordered_output_no_height = nd::reconfig_accessors(std::move(output_), order_output_dim);
        auto reordered_shape_no_height = reordered_shape.filter(order_output_dim);
        using ReorderedOutputNoHeight = decltype(reordered_output_no_height);

        using OpDecay = std::decay_t<Op>;
        using InputDecay = std::decay_t<Input>;
        using ReducedDecay = std::decay_t<Reduced>;
        using Interface = Config::interface;

        // Block shape.
        constexpr u32 N_THREADS_X = Constant::WARP_SIZE;
        constexpr u32 N_THREADS_Y = max(Config::block_size, N_THREADS_X) / N_THREADS_X;
        constexpr auto N_THREADS = dim3(N_THREADS_X, N_THREADS_Y);
        using Block = ReduceAxesEwiseHeightConfig<Config, N_THREADS_X>;

        // Grid shape.
        const auto grid = GridND(
            reordered_shape_no_height,
            Shape<isize, N - 1>::from_value(1).template set<N - 2>(N_THREADS_X)
        );
        check(grid.n_launches_x() == 1);

        // Launch kernel.
        constexpr usize GRID_POP_FRONT = N <= 3 ? 4 - N : 0; // N=2:none, N==3:y, N>=4:zy
        const auto grid_outer_shape = grid.outer_shape().vec.template pop_front<GRID_POP_FRONT>();
        const auto grid_fused_shape_in_x_unbatched = grid.fused_shape().vec.pop_front();
        const auto reordered_shape_hw = reordered_shape.filter(N - 2, N - 1);

        for (u32 z{}; z < grid.n_launches_z(); z++) {
            for (u32 y{}; y < grid.n_launches_y(); y++) {
                const auto config = LaunchConfig{
                    .n_blocks = grid.dim3_shape_for_launch(z, y, 0),
                    .n_threads = N_THREADS,
                };
                const auto grid_outer_offset = grid.block_offset_for_launch(z, y).template pop_front<GRID_POP_FRONT>();
                stream.enqueue(
                    reduce_height_ewise<Block, Interface, OpDecay, Index, N, InputDecay, ReducedDecay, ReorderedOutputNoHeight>,
                    config, op, input_, reordered_shape_hw, reduced, reordered_output_no_height,
                    grid_outer_shape, grid_outer_offset, grid_fused_shape_in_x_unbatched
                );
            }
        }
    }
}

namespace noa::cuda {
    template<typename Config = ReduceEwiseConfig<>,
             typename Op, typename Input, typename Reduced, typename Output, typename Index, usize N>
    requires (nt::tuple_of_accessor_nd<std::decay_t<Input>, N> and
              not nt::tuple_of_accessor_value<std::decay_t<Input>> and // at least one varray
              nt::tuple_of_accessor_pure_nd<std::decay_t<Output>, N> and
              nt::tuple_of_accessor_value_or_empty<std::decay_t<Reduced>>)
    NOA_NOINLINE void reduce_axes_ewise(
        const Shape<Index, N>& input_shape,
        const Shape<Index, N>& output_shape,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream
    ) {
        const auto axes_to_reduce = input_shape.cmp_ne(output_shape);
        const auto axes_empty_or_to_reduce = output_shape.cmp_eq(1) or axes_to_reduce;

        // Find the first non-empty axis.
        usize first_non_empty{N - 1};
        if constexpr (N > 1) {
            for (usize i = 0; i < N - 1; ++i) {
                if (input_shape[i] > 1) {
                    first_non_empty = i;
                    break;
                }
            }
        }

        // Batch reduction is when all axes after first non-empty axis are empty or reduced.
        bool batched_reduction{true};
        for (usize i = first_non_empty + 1; i < N; ++i)
            if (not axes_empty_or_to_reduce[i])
                batched_reduction = false;

        if (batched_reduction) {
            constexpr auto SMALL_THRESHOLD = Config::block_work_size * 4;
            auto output_1d = nd::reconfig_accessors(std::forward<Output>(output), first_non_empty);

            // Use the row kernels if the reduction is along a single axis.
            // Given that the shapes are collapsed, this means the first non-empty input axis
            // is N-2 (batched reduction) or N-1 (batch=1, single reduction).
            if (N <= 2 or first_non_empty >= N - 2) {
                const auto batch = N == 1 ? u32{1} : safe_cast<u32>(output_shape[N - 2]);
                const auto width = safe_cast<Index>(input_shape[N - 1]);
                if (width <= SMALL_THRESHOLD) {
                    details::launch_reduce_ewise_small_rows<Config, N>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::move(output_1d), batch, width, stream
                    );
                } else {
                    details::launch_reduce_ewise_large_rows<Config, N>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::move(output_1d), batch, width, stream
                    );
                }
                return;
            }
            if constexpr (N >= 2) {
                // The batch should be the leftmost dimensions. In cases with empty dimensions before the first
                // non-empty axis, the input and input shape should be reordered.
                auto order_to_batch = Vec<usize, N>::arange();
                std::swap(order_to_batch[0], order_to_batch[first_non_empty]);
                auto input_nd = std::forward<Input>(input);
                nd::permute_accessors(order_to_batch, input_nd);
                auto input_shape_nd = input_shape.filter(order_to_batch);

                if (input_shape.template as<isize>().n_elements() <= SMALL_THRESHOLD) {
                    details::launch_reduce_ewise_small_nd<Config>(
                        std::forward<Op>(op), std::move(input_nd), std::forward<Reduced>(reduced),
                        std::move(output_1d), input_shape_nd, true, stream
                    );
                } else {
                    details::launch_reduce_ewise_large_nd<Config>(
                        std::forward<Op>(op), std::move(input_nd), std::forward<Reduced>(reduced),
                        std::move(output_1d), input_shape_nd, true, stream
                    );
                }
            }
            return;
        }

        if constexpr (N >= 3) {
            if (axes_to_reduce[N - 1]) {
                details::launch_reduce_ewise_width<Config>(
                    input_shape,
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    stream
                );
                return;
            }
        }
        if constexpr (N >= 2) {
            details::launch_reduce_ewise_height<Config>(
                input_shape, axes_to_reduce,
                std::forward<Op>(op),
                std::forward<Input>(input),
                std::forward<Reduced>(reduced),
                std::forward<Output>(output),
                stream
            );
        }
    }
}

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif
