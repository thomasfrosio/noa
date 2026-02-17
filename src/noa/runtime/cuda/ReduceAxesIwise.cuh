#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Block.cuh"
#include "noa/runtime/cuda/ReduceIwise.cuh"

namespace noa::cuda::details {
    template<typename Config, u32 WarpCountBlockSizeX>
    struct ReduceIwiseSingleAxisBlock {
        // Get a warp count that makes block_size_x a multiple of block_size.
        static consteval auto best_block_size_x() {
            u32 c = max(1u, WarpCountBlockSizeX);
            for (;;) {
                if (Constant::WARP_SIZE * c <= Config::block_size and
                    is_multiple_of(Config::block_size, Constant::WARP_SIZE * c)) {
                    break;
                }
                --c; // try a block size smaller by one warp
            }
            return Constant::WARP_SIZE * c;
        }

        static constexpr u32 block_size = Config::block_size; // is multiple of WARP_SIZE
        static constexpr u32 block_size_x = Config::allow_block_reshape ? best_block_size_x() : Config::block_size_x;
        static constexpr u32 block_size_y = Config::allow_block_reshape ? block_size / block_size_x : Config::block_size_y;
        static constexpr u32 block_ndim = block_size_y > 1 ? 2 : 1;
    };

    template<size_t N, typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_width_iwise(
        Op op,
        Shape<Index, 2> shape_hw,
        Reduced reduced,
        Output output,
        u32 scratch_size,
        Vec<u32, N - 2> grid_size,
        Vec<u32, N - 2> grid_offset
    ) {
        // Multigrid case.
        auto bid = block_indices<u32, 3>();
        if constexpr (N == 4)
            bid[0] += grid_offset[0];
        if constexpr (N >= 3)
            bid[1] += grid_offset[N - 3];

        // Global indices.
        const auto tid = Vec<Index, 2>::from_values(threadIdx.y, threadIdx.x);
        const auto gid = Vec<Index, 4>::from_values(
            bid[0], // always 0 if N < 4
            bid[1], // always 0 if N < 3
            bid[2] * Block::block_size_y + tid[0],
            tid[1]);
        const bool is_valid_row = gid[2] < shape_hw[0];

        const auto ci = ComputeHandle<Index, N - 1, 2, true, true, false>(scratch_size, grid_size, grid_offset);
        if constexpr (N == 4)
            Interface::init(ci, op, gid[0], gid[1], gid[2]);
        else if constexpr (N == 3)
            Interface::init(ci, op, gid[1], gid[2]);
        else
            static_assert(nt::always_false<Interface>);

        // Loop until the end of the row is reached.
        for (Index cid = gid[3]; cid < shape_hw[1] and is_valid_row; cid += Block::block_size_x) {
            if constexpr (N == 4)
                Interface::call(ci, op, reduced, gid[0], gid[1], gid[2], cid);
            else if constexpr (N == 3)
                Interface::call(ci, op, reduced, gid[1], gid[2], cid);
            else
                static_assert(nt::always_false<Block>);
        }

        if constexpr (N == 4)
            Interface::deinit(ci, op, gid[0], gid[1], gid[2]);
        else if constexpr (N == 3)
            Interface::deinit(ci, op, gid[1], gid[2]);
        else
            static_assert(nt::always_false<Interface>);

        if constexpr (not nt::empty_tuple<Reduced>) {
            // Share the threads' initial reduction with the rest of the block.
            Reduced* shared_buffer = dynamic_shared_memory_pointer<Reduced>(); // (as least) Block::block_size
            Reduced* joined = shared_buffer + tid[0] * Block::block_size_x;
            joined[tid[1]] = reduced;
            block_synchronize();

            // Reduce each "shared row" to one element.
            reduced = block_join_shared<Interface, Block::block_size_x>(op, joined, tid[1]);
        }

        if (gid[3] == 0 and is_valid_row) {
            if constexpr (N == 4)
                Interface::post(op, reduced, output, gid[0], gid[1], gid[2]);
            else if constexpr (N == 3)
                Interface::post(op, reduced, output, gid[1], gid[2]);
            else
                static_assert(nt::always_false<Interface>);
        }
    }

    template<size_t N, size_t R, typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_height_iwise(
        Op op,
        Shape<Index, 2> shape_hw,
        Reduced reduced,
        Output output,
        u32 scratch_size,
        Vec<u32, N - 2> grid_size,
        Vec<u32, N - 2> grid_offset
    ) {
        // Multigrid case.
        auto bid = block_indices<u32, 3>();
        if constexpr (N == 4)
            bid[0] += grid_offset[0];
        if constexpr (N >= 3)
            bid[1] += grid_offset[N - 3];

        // Global indices.
        const auto gid = Vec<Index, 4>::from_values(
            bid[0], // always 0 if N < 4
            bid[1], // always 0 if N < 3
            threadIdx.y, // one block along the height
            bid[2] * Block::block_size_x + threadIdx.x);
        const bool is_valid_column = gid[3] < shape_hw[1];

        const auto ci = ComputeHandle<Index, N - 1, 2, true, true, false>(scratch_size, grid_size, grid_offset);
        if constexpr (N == 4)
            Interface::init(ci, op, gid[0], gid[1], gid[3]);
        else if constexpr (N == 3)
            Interface::init(ci, op, gid[1], gid[3]);
        else if constexpr (N == 2)
            Interface::init(ci, op, gid[3]);
        else
            static_assert(nt::always_false<Interface>);

        // Process every row.
        for (Index tidy = gid[2]; tidy < shape_hw[0] and is_valid_column; tidy += Block::block_size_y) {
            if constexpr (N == 4) {
                if constexpr (R == 0) {
                    Interface::call(ci, op, reduced, tidy, gid[0], gid[1], gid[3]);
                } else if constexpr (R == 1) {
                    Interface::call(ci, op, reduced, gid[0], tidy, gid[1], gid[3]);
                } else if constexpr (R == 2) {
                    Interface::call(ci, op, reduced, gid[0], gid[1], tidy, gid[3]);
                } else {
                    static_assert(nt::always_false<Interface>);
                }
            } else if constexpr (N == 3) {
                if constexpr (R == 0) {
                    Interface::call(ci, op, reduced, tidy, gid[1], gid[3]);
                } else if constexpr (R == 1) {
                    Interface::call(ci, op, reduced, gid[1], tidy, gid[3]);
                } else {
                    static_assert(nt::always_false<Interface>);
                }
            } else if constexpr (N == 2 and R == 0) {
                Interface::call(ci, op, reduced, tidy, gid[3]);
            } else {
                static_assert(nt::always_false<Interface>);
            }
        }

        if constexpr (N == 4)
            Interface::deinit(ci, op, gid[0], gid[1], gid[3]);
        else if constexpr (N == 3)
            Interface::deinit(ci, op, gid[1], gid[3]);
        else if constexpr (N == 2)
            Interface::deinit(ci, op, gid[3]);
        else
            static_assert(nt::always_false<Interface>);

        if constexpr (not nt::empty_tuple<Reduced>) {
            // Share the threads' initial reduction with the rest of the block.
            Reduced* shared_buffer = dynamic_shared_memory_pointer<Reduced>(); // (as least) Block::block_size
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

        if (threadIdx.y == 0 and is_valid_column) {
            if constexpr (N == 4)
                Interface::post(op, reduced, output, gid[0], gid[1], gid[3]);
            else if constexpr (N == 3)
                Interface::post(op, reduced, output, gid[1], gid[3]);
            else if constexpr (N == 2)
                Interface::post(op, reduced, output, gid[3]);
            else
                static_assert(nt::always_false<Interface>);
        }
    }
}

namespace noa::cuda::details {
    template<typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Joined>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_4d_first(
        Op op,
        Reduced reduced,
        Joined joined, // 2d Accessor of Reduced
        Vec<Index, 3> shape_dhw,
        Vec<u32, 2> n_blocks_hw,
        u32 scratch_size,
        u32 grid_size_z,
        u32 block_index_offset_z
    ) {
        // Get the position within the 4d span.
        const Index cb = blockIdx.z + block_index_offset_z;
        const Vec<u32, 2> index = offset2index(blockIdx.x, n_blocks_hw[1]);
        const auto gid = Vec<Index, 3>::from_values(
            blockIdx.y,
            Block::block_size_y * index[0] + threadIdx.y,
            Block::block_size_x * index[1] + threadIdx.x
        );

        const auto ci = ComputeHandle<Index, 3, 2, true, true, true>(
            scratch_size, Vec<u32, 2>{grid_size_z, 1}, Vec<u32, 2>{block_index_offset_z, 0});
        Interface::init(ci, op, cb);

        // Traverse the entire 3d span of this batch.
        for (Index cd = gid[0]; cd < shape_dhw[0]; cd += gridDim.y)
            for (Index ch = gid[1]; ch < shape_dhw[1]; ch += Block::block_size_y * n_blocks_hw[0])
                for (Index cw = gid[2]; cw < shape_dhw[2]; cw += Block::block_size_x * n_blocks_hw[1])
                    Interface::call(ci, op, reduced, cb, cd, ch, cw);

        Interface::deinit(ci, op, cb);

        // Reduce to one value per block.
        const Index tid = threadIdx.y * Block::block_size_x + threadIdx.x;
        const Index bid = blockIdx.y * gridDim.x + blockIdx.x;
        block_join<Interface, Block::block_size, false>(op, reduced, joined, tid, cb, bid);
    }

    template<typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Joined>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_3d_first(
        Op op,
        Reduced reduced,
        Joined joined, // 2d Accessor of Reduced
        Vec<Index, 2> shape_hw,
        u32 scratch_size,
        u32 grid_size_z,
        u32 block_index_offset_z
    ) {
        const Index cd = blockIdx.z + block_index_offset_z;
        const auto gid = Vec<Index, 2>::from_values(
            Block::block_size_y * blockIdx.y + threadIdx.y,
            Block::block_size_x * blockIdx.x + threadIdx.x
        );

        const auto ci = ComputeHandle<Index, 3, 2, true, true, true>(
            scratch_size, Vec<u32, 2>{grid_size_z, 1}, Vec<u32, 2>{block_index_offset_z, 0});
        Interface::init(ci, op, cd);

        for (Index ch = gid[0]; ch < shape_hw[0]; ch += Block::block_size_y * gridDim.y)
            for (Index cw = gid[1]; cw < shape_hw[1]; cw += Block::block_size_x * gridDim.x)
                Interface::call(ci, op, reduced, cd, ch, cw);

        Interface::deinit(ci, op, cd);

        const Index tid = threadIdx.y * Block::block_size_x + threadIdx.x;
        const Index bid = blockIdx.y * gridDim.x + blockIdx.x;
        block_join<Interface, Block::block_size, false>(op, reduced, joined, tid, cd, bid);
    }

    template<typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Joined>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_2d_first(
        Op op,
        Reduced reduced,
        Joined joined,
        Vec<Index, 1> shape,
        u32 scratch_size,
        u32 grid_size_z,
        u32 block_index_offset_z
    ) {
        static_assert(Block::block_size_y == 1);
        const Index ch = blockIdx.z + block_index_offset_z;
        const Index gid = Block::block_size_x * blockIdx.x + threadIdx.x;

        const auto ci = ComputeHandle<Index, 3, 1, true, true, true>(
            scratch_size, Vec<u32, 2>{grid_size_z, 1}, Vec<u32, 2>{block_index_offset_z, 0});
        Interface::init(ci, op, ch);

        for (Index cw = gid; cw < shape[0]; cw += Block::block_size_x * gridDim.x)
            Interface::call(ci, op, reduced, ch, cw);

        Interface::deinit(ci, op, ch);

        const Index tid = threadIdx.x;
        const Index bid = blockIdx.x;
        block_join<Interface, Block::block_size, false>(op, reduced, joined, tid, ch, bid);
    }

    template<u32 BlockSize, typename Interface, typename Op, typename Index, typename Joined, typename Reduced, typename Output>
    __global__ __launch_bounds__(BlockSize)
    void reduce_axes_iwise_second(
        Op op,
        Joined to_join,
        Index n_to_join,
        Reduced reduced,
        Output output
    ) {
        const Index batch = blockIdx.x;
        const Index tid = threadIdx.x;
        if constexpr (not nt::empty_tuple<Reduced>) {
            for (Index cid = tid; cid < n_to_join; cid += BlockSize)
                Interface::join(op, to_join(batch, cid), reduced);
        }
        block_join_and_post<Interface, BlockSize, true>(op, reduced, output, tid, batch);
    }

    template<typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_4d_small(Op op, Reduced reduced, Output output, Vec<Index, 3> shape_dhw, u32 scratch_size) {
        const Index cb = blockIdx.x;

        const auto ci = ComputeHandle<Index, 1, 2, false, true, false>(scratch_size);
        Interface::init(ci, op, cb);

        const auto gid = Vec<Index, 3>::from_values(0, threadIdx.y, threadIdx.x);
        for (Index cd = gid[0]; cd < shape_dhw[0]; ++cd)
            for (Index ch = gid[1]; ch < shape_dhw[1]; ch += Block::block_size_y)
                for (Index cw = gid[2]; cw < shape_dhw[2]; cw += Block::block_size_x)
                    Interface::call(ci, op, reduced, cb, cd, ch, cw);

        Interface::deinit(ci, op, cb);

        const Index tid = threadIdx.y * Block::block_size_x + threadIdx.x;
        block_join_and_post<Interface, Block::block_size, false>(op, reduced, output, tid, cb);
    }

    template<typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_3d_small(Op op, Reduced reduced, Output output, Vec<Index, 2> shape_hw, u32 scratch_size) {
        const Index cd = blockIdx.x;

        const auto ci = ComputeHandle<Index, 1, 2, false, true, false>(scratch_size);
        Interface::init(ci, op, cd);

        const auto gid = Vec<Index, 2>::from_values(threadIdx.y, threadIdx.x);
        for (Index ch = gid[0]; ch < shape_hw[0]; ch += Block::block_size_y)
            for (Index cw = gid[1]; cw < shape_hw[1]; cw += Block::block_size_x)
                Interface::call(ci, op, reduced, cd, ch, cw);

        Interface::deinit(ci, op, cd);

        const Index tid = threadIdx.y * Block::block_size_x + threadIdx.x;
        block_join_and_post<Interface, Block::block_size, false>(op, reduced, output, tid, cd);
    }

    template<typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_2d_small(Op op, Reduced reduced, Output output, Vec<Index, 1> shape, u32 scratch_size) {
        static_assert(Block::block_size_y == 1);
        const Index ch = blockIdx.x;

        const auto ci = ComputeHandle<Index, 1, 2, false, true, false>(scratch_size);
        Interface::init(ci, op, ch);

        const Index tid = threadIdx.x;
        for (Index cw = tid; cw < shape[0]; cw += Block::block_size_x)
            Interface::call(ci, op, reduced, ch, cw);

        Interface::deinit(ci, op, ch);

        block_join_and_post<Interface, Block::block_size, false>(op, reduced, output, tid, ch);
    }
}

namespace noa::cuda::details {
    template<typename Config, typename Op, typename Reduced, typename Output, typename Index>
    void launch_reduce_axes_iwise_4d(
        const Shape<Index, 4>& input_shape,
        Vec<bool, 4> axes_to_reduce,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        Stream& stream,
        usize scratch_size
    ) {
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using Interface = Config::interface;

        const auto n_bytes_of_shared_memory =
            n_bytes_of_shared_memory_to_allocate_for_reduction<Reduced>(
                Config::block_size, scratch_size);

        if (axes_to_reduce[3]) {
            const auto shape_u32 = input_shape.template as_safe<u32>();
            const auto shape_hw = input_shape.filter(2, 3);

            // The width of the output is empty/reduced, remove it.
            constexpr auto TO_3D = nd::AccessorConfig<3>{.filter={0, 1, 2}};
            auto output_3d = nd::reconfig_accessors<TO_3D>(output);
            using Output3D = decltype(output_3d);

            // Block shape.
            using block_big_x = ReduceIwiseSingleAxisBlock<Config, 8>;
            using block_small_x = ReduceIwiseSingleAxisBlock<Config, 2>;
            const bool is_big_x = shape_u32[3] > 512;
            const auto n_threads_x = is_big_x ? block_big_x::block_size_x : block_small_x::block_size_x;
            const auto n_threads_y = is_big_x ? block_big_x::block_size_y : block_small_x::block_size_y;
            const auto n_threads = dim3(n_threads_x, n_threads_y);

            // Grid shape.
            const auto grid_x = GridX(shape_u32[2], n_threads_y);
            const auto grid_y = GridY(shape_u32[1], 1);
            const auto grid_z = GridZ(shape_u32[0], 1);
            check(grid_x.n_launches() == 1);

            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    const auto config = LaunchConfig{
                        .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                        .n_threads = n_threads,
                        .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                    };
                    const auto grid_size = Vec{grid_z.n_blocks_total(), grid_y.n_blocks_total()}.template as<u32>();
                    const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y)};
                    if (is_big_x) {
                        stream.enqueue(
                            reduce_width_iwise<4, block_big_x, Interface, OpDecay, Index, ReducedDecay, Output3D>,
                            config, op, shape_hw, reduced, output_3d,
                            static_cast<u32>(scratch_size), grid_size, grid_offset
                        );
                    } else {
                        stream.enqueue(
                            reduce_width_iwise<4, block_small_x, Interface, OpDecay, Index, ReducedDecay, Output3D>,
                            config, op, shape_hw, reduced, output_3d,
                            static_cast<u32>(scratch_size), grid_size, grid_offset
                        );
                    }
                }
            }
        } else {
            // The kernel needs the axis to reduce at the "height" position.
            // The width should still be at the rightmost dimension.
            auto order = squeeze_left(axes_to_reduce.as<i32>() + 1);
            order = order.filter(0, 1, 3, 2); // move the width back to rightmost

            // Reorder to (X, X, axis_to_reduce, width).
            auto reordered_shape = input_shape.permute(order);
            auto reordered_output = output.map([&order](const auto& accessor) {
                return accessor.permute(order);
            });

            // Remove the empty/reduced axis from the output.
            constexpr auto TO_3D = nd::AccessorConfig<3>{.filter = {0, 1, 3}};
            auto reordered_output_3d = nd::reconfig_accessors<TO_3D>(reordered_output);
            using ReorderedOutput3D = decltype(reordered_output_3d);

            // Block shape.
            using Block = ReduceIwise2dBlock<Config>;
            const auto n_threads = dim3(Block::block_size_x, Block::block_size_y);

            // Grid shape.
            const auto grid_x = GridX(reordered_shape[3], Block::block_size_x);
            const auto grid_y = GridY(reordered_shape[1], 1);
            const auto grid_z = GridZ(reordered_shape[0], 1);
            check(grid_x.n_launches() == 1);

            const auto shape_2d = reordered_shape.template pop_front<2>();

            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    const auto config = LaunchConfig{
                        .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                        .n_threads = n_threads,
                        .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                    };
                    const auto grid_size = Vec{grid_z.n_blocks_total(), grid_y.n_blocks_total()}.template as<u32>();
                    const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y)};
                    if (axes_to_reduce[2]) {
                        stream.enqueue(
                            reduce_height_iwise<4, 2, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutput3D>,
                            config, op, shape_2d, reduced, reordered_output_3d,
                             static_cast<u32>(scratch_size), grid_size, grid_offset
                        );
                    } else if (axes_to_reduce[1]) {
                        stream.enqueue(
                            reduce_height_iwise<4, 1, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutput3D>,
                            config, op, shape_2d, reduced, reordered_output_3d,
                             static_cast<u32>(scratch_size), grid_size, grid_offset
                        );
                    } else {
                        stream.enqueue(
                            reduce_height_iwise<4, 0, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutput3D>,
                            config, op, shape_2d, reduced, reordered_output_3d,
                             static_cast<u32>(scratch_size), grid_size, grid_offset
                        );
                    }
                }
            }
        }
    }

    template<typename Config, typename Op, typename Reduced, typename Output, typename Index>
    void launch_reduce_axes_iwise_3d(
        const Shape<Index, 3>& input_shape,
        Vec<bool, 3> axes_to_reduce,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        Stream& stream,
        usize scratch_size
    ) {
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using Interface = Config::interface;

        const auto n_bytes_of_shared_memory =
            n_bytes_of_shared_memory_to_allocate_for_reduction<Reduced>(
                Config::block_size, scratch_size);

        if (axes_to_reduce[2]) {
            const auto shape_u32 = input_shape.template as_safe<u32>();
            const auto shape_hw = input_shape.filter(1, 2);

            // The width of the output is empty/reduced, remove it.
            constexpr auto to_2d = nd::AccessorConfig<2>{.filter={0, 1}};
            auto output_2d = nd::reconfig_accessors<to_2d>(output);
            using Output2D = decltype(output_2d);

            // Block shape.
            using block_big_x = ReduceIwiseSingleAxisBlock<Config, 8>;
            using block_small_x = ReduceIwiseSingleAxisBlock<Config, 2>;
            const bool is_big_x = shape_u32[2] > 512;
            const auto n_threads_x = is_big_x ? block_big_x::block_size_x : block_small_x::block_size_x;
            const auto n_threads_y = is_big_x ? block_big_x::block_size_y : block_small_x::block_size_y;
            const auto n_threads = dim3(n_threads_x, n_threads_y);

            // Grid shape.
            const auto grid_x = GridX(shape_u32[1], n_threads_y);
            const auto grid_y = GridY(shape_u32[0], 1);
            check(grid_x.n_launches() == 1);

            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                const auto config = LaunchConfig{
                    .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y)),
                    .n_threads = n_threads,
                    .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                };
                const auto grid_size = Vec{grid_y.n_blocks_total()}.template as<u32>();;
                const auto grid_offset = Vec{grid_y.offset(y)};
                if (is_big_x) {
                    stream.enqueue(
                        reduce_width_iwise<3, block_big_x, Interface, OpDecay, Index, ReducedDecay, Output2D>,
                        config, op, shape_hw, reduced, output_2d,
                        static_cast<u32>(scratch_size), grid_size, grid_offset
                    );
                } else {
                    stream.enqueue(
                        reduce_width_iwise<3, block_small_x, Interface, OpDecay, Index, ReducedDecay, Output2D>,
                        config, op, shape_hw, reduced, output_2d,
                        static_cast<u32>(scratch_size), grid_size, grid_offset
                    );
                }
            }
        } else {
            // The kernel needs the axis to reduce at the "height" position.
            // The width should still be at the rightmost dimension.
            auto order = squeeze_left(axes_to_reduce.as<i32>() + 1);
            order = order.filter(0, 2, 1); // move the width back to rightmost

            // Reorder to (X, axis_to_reduce, width).
            auto reordered_shape = input_shape.permute(order);
            auto reordered_output = output.map([&order](const auto& accessor) {
                return accessor.permute(order);
            });

            // Remove the empty/reduced axis from the output.
            constexpr auto TO_2D = nd::AccessorConfig<2>{.filter = {0, 2}};
            auto reordered_output_2d = nd::reconfig_accessors<TO_2D>(reordered_output);
            using ReorderedOutput2D = decltype(reordered_output_2d);

            // Block shape.
            using Block = ReduceIwise2dBlock<Config>;
            const auto n_threads = dim3(Block::block_size_x, Block::block_size_y);

            // Grid shape.
            const auto grid_x = GridX(reordered_shape[2], Block::block_size_x);
            const auto grid_y = GridY(reordered_shape[0], 1);
            check(grid_x.n_launches() == 1);

            const auto shape_2d = reordered_shape.template pop_front<1>();

            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                const auto config = LaunchConfig{
                    .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), 1),
                    .n_threads = n_threads,
                    .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                };
                const auto grid_size = Vec{grid_y.n_blocks_total()}.template as<u32>();
                const auto grid_offset = Vec{grid_y.offset(y)};
                if (axes_to_reduce[1]) {
                    stream.enqueue(
                        reduce_height_iwise<3, 1, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutput2D>,
                        config, op, shape_2d, reduced, reordered_output_2d,
                        static_cast<u32>(scratch_size), grid_size, grid_offset
                    );
                } else {
                    stream.enqueue(
                        reduce_height_iwise<3, 0, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutput2D>,
                        config, op, shape_2d, reduced, reordered_output_2d,
                        static_cast<u32>(scratch_size), grid_size, grid_offset
                    );
                }
            }
        }
    }

    template<typename Config, typename Op, typename Reduced, typename Output, typename Index>
    void launch_reduce_axes_iwise_2d(
        const Shape<Index, 2>& input_shape,
        Vec<bool, 2> axes_to_reduce,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        Stream& stream,
        usize scratch_size
    ) {
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;

        const auto n_bytes_of_shared_memory =
            n_bytes_of_shared_memory_to_allocate_for_reduction<Reduced>(
                Config::block_size, scratch_size);

        if (axes_to_reduce[0]) {
            const auto input_shape_u32 = input_shape.template as<u32>();

            using Block = ReduceIwise2dBlock<Config>;
            const u32 n_blocks_x = divide_up(input_shape_u32[1], Block::block_size_x);
            const auto config = LaunchConfig{
                .n_blocks = dim3(n_blocks_x, 1, 1),
                .n_threads = dim3(Block::block_size_x, Block::block_size_y),
                .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
            };

            constexpr auto TO_1D = nd::AccessorConfig<1>{.filter = {1}};
            auto output_1d = nd::reconfig_accessors<TO_1D>(output);
            using Output1D = decltype(output_1d);

            using Interface = Config::interface;
            stream.enqueue(
                reduce_height_iwise<2, 0, Block, Interface, OpDecay, Index, ReducedDecay, Output1D>,
                config, std::forward<Op>(op), input_shape, std::forward<Reduced>(reduced), output_1d,
                static_cast<u32>(scratch_size), Vec<u32, 0>{}, Vec<u32, 0>{}
            );
        } else {
            panic("unreachable");
        }
    }
}

namespace noa::cuda {
    template<typename Config = ReduceIwiseConfig<>,
             typename Op, typename Reduced, typename Output, typename Index, size_t N>
    requires (nt::tuple_of_accessor_value_or_empty<std::decay_t<Reduced>> and
              nt::tuple_of_accessor_pure_nd_or_empty<Output, N>)
    NOA_NOINLINE void reduce_axes_iwise(
        const Shape<Index, N>& input_shape,
        const Shape<Index, N>& output_shape,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        Stream& stream,
        usize scratch_size = 0
    ) {
        const auto axes_to_reduce = input_shape.cmp_ne(output_shape);
        const auto axes_empty_or_to_reduce = output_shape.cmp_eq(1) or axes_to_reduce;

        check((axes_to_reduce and output_shape.cmp_ne(1)) == false,
              "Dimensions should match the input shape, or be 1, "
              "indicating the dimension should be reduced to one element. "
              "Got shape input={}, output={}", input_shape, output_shape);
        check(axes_to_reduce.any_eq(true),
              "No reduction to compute. Got shape input={}, output={}. Please use iwise instead.",
              input_shape, output_shape);

        // Reduce to one value.
        if (axes_empty_or_to_reduce == true) {
            constexpr auto TO_1D = nd::AccessorConfig<1>{.enforce_contiguous = true, .filter = {0}};
            const auto output_1d = nd::reconfig_accessors<TO_1D>(output);
            return reduce_iwise<Config>(
                input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced),
                output_1d, stream, scratch_size
            );
        }

        const auto n_bytes_of_shared_memory =
            n_bytes_of_shared_memory_to_allocate_for_reduction<Reduced>(
                Config::block_size, scratch_size);

        if constexpr (N > 1) {
            const auto shape_to_reduce = input_shape.pop_front();
            const auto shape_to_reduce_iz = shape_to_reduce.template as_safe<isize>();
            const auto batch = safe_cast<u32>(input_shape[0]);

            using OpDecay = std::decay_t<Op>;
            using ReducedDecay = std::decay_t<Reduced>;

            // Reduce to one value per leftmost.
            if (axes_empty_or_to_reduce.pop_front() == true) {
                using block_1d = details::ReduceIwise1dBlock<Config>;
                using block_2d = details::ReduceIwise2dBlock<Config>;
                using Block = std::conditional_t<N == 2, block_1d, block_2d>;
                using Interface = Config::interface;

                const auto n_elements = shape_to_reduce_iz.n_elements();
                constexpr auto SMALL_THRESHOLD = static_cast<isize>(Config::block_size * 32);
                auto output_1d = nd::reconfig_accessors<nd::AccessorConfig<1>{.filter={0}}>(output);
                using Output1d = decltype(output_1d);

                if (not Config::allow_two_kernels or n_elements <= SMALL_THRESHOLD) {
                    const auto n_threads = dim3(Block::block_size_x, Block::block_size_y);
                    const auto config = LaunchConfig{
                        .n_blocks = batch,
                        .n_threads = n_threads,
                        .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                    };

                    if constexpr (N == 4) {
                        stream.enqueue(
                            details::reduce_axes_iwise_4d_small<Block, Interface, OpDecay, Index, ReducedDecay, Output1d>,
                            config, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, shape_to_reduce.vec,
                            static_cast<u32>(scratch_size)
                        );
                    } else if constexpr (N == 3) {
                        stream.enqueue(
                            details::reduce_axes_iwise_3d_small<Block, Interface, OpDecay, Index, ReducedDecay, Output1d>,
                            config, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, shape_to_reduce.vec,
                            static_cast<u32>(scratch_size)
                        );
                    } else {
                        stream.enqueue(
                            details::reduce_axes_iwise_2d_small<Block, Interface, OpDecay, Index, ReducedDecay, Output1d>,
                            config, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, shape_to_reduce.vec,
                            static_cast<u32>(scratch_size)
                        );
                    }
                } else {
                    // Compute the grid shape necessary to loop through the reduced elements.
                    // We add one to the shape_to_reduce so that the returned config has config.n_blocks.z == 1.
                    // This is because the leftmost dimension is on a separate dimension of the grid (z).
                    auto [config, n_blocks_per_batch, n_blocks_hw] =
                        details::reduce_iwise_nd_first_config<Block>(
                            shape_to_reduce_iz.push_front(1));

                    // Allocate the 2d buffer.
                    constexpr bool HAS_REDUCED = not nt::empty_tuple<Reduced>;
                    const auto joined_size = HAS_REDUCED ? n_blocks_per_batch * batch : 0;
                    using Joined = AccessorRestrictContiguous<ReducedDecay, 2, Index>;
                    auto buffer = AllocatorDevice::allocate_async<ReducedDecay>(joined_size, stream);
                    auto joined = Joined(buffer.get(), Strides<Index, 2>::from_values(n_blocks_per_batch, 1));

                    config.n_bytes_of_shared_memory = n_bytes_of_shared_memory;

                    auto grid_z = GridZ(batch, 1);
                    for (u32 z{}; z < grid_z.n_launches(); ++z) {
                        config.n_blocks.z = grid_z.n_blocks(z);

                        if constexpr (N == 4) {
                            stream.enqueue(
                                details::reduce_axes_iwise_4d_first<Block, Interface, OpDecay, Index, ReducedDecay, Joined>,
                                config, op, reduced, joined, shape_to_reduce.vec, n_blocks_hw,
                                static_cast<u32>(scratch_size), static_cast<u32>(grid_z.n_blocks_total()), grid_z.offset(z)
                            );
                        } else if constexpr (N == 3) {
                            stream.enqueue(
                                details::reduce_axes_iwise_3d_first<Block, Interface, OpDecay, Index, ReducedDecay, Joined>,
                                config, op, reduced, joined, shape_to_reduce.vec,
                                static_cast<u32>(scratch_size), static_cast<u32>(grid_z.n_blocks_total()), grid_z.offset(z)
                            );
                        } else {
                            stream.enqueue(
                                details::reduce_axes_iwise_2d_first<Block, Interface, OpDecay, Index, ReducedDecay, Joined>,
                                config, op, reduced, joined, shape_to_reduce.vec,
                                static_cast<u32>(scratch_size), static_cast<u32>(grid_z.n_blocks_total()), grid_z.offset(z)
                            );
                        }
                    }

                    // Second kernel.
                    // TODO If there's no value to reduce, check that op has a valid post and if not skip the launch.
                    stream.enqueue(
                        details::reduce_axes_iwise_second<Config::block_size, Interface, OpDecay, Index, Joined, ReducedDecay, Output1d>,
                        LaunchConfig{.n_blocks = batch, .n_threads = HAS_REDUCED ? Config::block_size : 1},
                        std::forward<Op>(op), joined, static_cast<Index>(n_blocks_per_batch),
                        std::forward<Reduced>(reduced), output_1d
                    );
                }
                return;
            }
        }

        // Reduce one axis.
        const i32 nb_axes_to_reduce = sum(axes_to_reduce.template as<i32>());
        check(nb_axes_to_reduce == 1,
              "Reducing more than one axis at a time is currently limited to a reduction that would "
              "result in one value per batch, i.e. the DHW dimensions should empty after reduction. "
              "Got input_shape={}, output_shape={}, axes_to_reduce={}",
              input_shape, output_shape, axes_to_reduce);

        if constexpr (N == 4) {
            details::launch_reduce_axes_iwise_4d<Config>(
                input_shape, axes_to_reduce, std::forward<Op>(op),
                std::forward<Reduced>(reduced), output, stream,
                scratch_size
            );
        } else if constexpr (N == 3) {
            details::launch_reduce_axes_iwise_3d<Config>(
                input_shape, axes_to_reduce, std::forward<Op>(op),
                std::forward<Reduced>(reduced), output, stream,
                scratch_size
            );
        } else if constexpr (N == 2) {
            details::launch_reduce_axes_iwise_2d<Config>(
                input_shape, axes_to_reduce, std::forward<Op>(op),
                std::forward<Reduced>(reduced), output, stream,
                scratch_size
            );
        } else {
            panic("unreachable");
        }
    }
}
