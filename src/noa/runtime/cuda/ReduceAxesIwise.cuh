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

        const bool is_valid_row = gid[N - 2] < shape_hw[0];

        const auto ci = ComputeHandle<Index, GRID_NDIM, 2, true, true, false>(scratch_size, grid_outer_shape, grid_outer_offset);
        if constexpr (N == 3)
            Interface::init(ci, op, gid[0], gid[1]);
        else if constexpr (N == 4)
            Interface::init(ci, op, gid[0], gid[1], gid[2]);
        else if constexpr (N == 5)
            Interface::init(ci, op, gid[0], gid[1], gid[2], gid[3]);
        else if constexpr (N == 6)
            Interface::init(ci, op, gid[0], gid[1], gid[2], gid[3], gid[4]);
        else
            static_assert(nt::always_false<Interface>);

        // Loop until the end of the row is reached.
        for (Index cid = gid[N - 1]; cid < shape_hw[1] and is_valid_row; cid += Block::block_size_x) {
            if constexpr (N == 3)
                Interface::call(ci, op, reduced, gid[0], gid[1], cid);
            else if constexpr (N == 4)
                Interface::call(ci, op, reduced, gid[0], gid[1], gid[2], cid);
            else if constexpr (N == 5)
                Interface::call(ci, op, reduced, gid[0], gid[1], gid[2], gid[3], cid);
            else if constexpr (N == 6)
                Interface::call(ci, op, reduced, gid[0], gid[1], gid[2], gid[3], gid[4], cid);
            else
                static_assert(nt::always_false<Block>);
        }

        if constexpr (N == 3)
            Interface::deinit(ci, op, gid[0], gid[1]);
        else if constexpr (N == 4)
            Interface::deinit(ci, op, gid[0], gid[1], gid[2]);
        else if constexpr (N == 5)
            Interface::deinit(ci, op, gid[0], gid[1], gid[2], gid[3]);
        else if constexpr (N == 6)
            Interface::deinit(ci, op, gid[0], gid[1], gid[2], gid[3], gid[4]);
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

        if (gid[N - 1] == 0 and is_valid_row) {
            if constexpr (N == 3)
                Interface::post(op, reduced, output, gid[0], gid[1]);
            else if constexpr (N == 4)
                Interface::post(op, reduced, output, gid[0], gid[1], gid[2]);
            else if constexpr (N == 5)
                Interface::post(op, reduced, output, gid[0], gid[1], gid[2], gid[3]);
            else if constexpr (N == 6)
                Interface::post(op, reduced, output, gid[0], gid[1], gid[2], gid[3], gid[4]);
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

        const auto ci = ComputeHandle<Index, GRID_NDIM, 2, true, true, false>(scratch_size, grid_outer_shape, grid_outer_offset);
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
        static_assert(R < N - 1);
        const bool is_valid_column = gid[N - 1] < shape_hw[1];
        for (Index tidy = gid[N - 2]; tidy < shape_hw[0] and is_valid_column; tidy += Block::block_size_y) {
            if constexpr (N == 2 and R == 0) {
                Interface::call(ci, op, reduced, tidy, gid[1]);
            } else if constexpr (N == 3) {
                if constexpr (R == 0) {
                    Interface::call(ci, op, reduced, tidy, gid[0], gid[2]);
                } else if constexpr (R == 1) {
                    Interface::call(ci, op, reduced, gid[0], tidy, gid[2]);
                }
            } else if constexpr (N == 4) {
                if constexpr (R == 0) {
                    Interface::call(ci, op, reduced, tidy, gid[0], gid[1], gid[3]);
                } else if constexpr (R == 1) {
                    Interface::call(ci, op, reduced, gid[0], tidy, gid[1], gid[3]);
                } else if constexpr (R == 2) {
                    Interface::call(ci, op, reduced, gid[0], gid[1], tidy, gid[3]);
                }
            } else if constexpr (N == 5) {
                if constexpr (R == 0) {
                    Interface::call(ci, op, reduced, tidy, gid[0], gid[1], gid[2], gid[4]);
                } else if constexpr (R == 1) {
                    Interface::call(ci, op, reduced, gid[0], tidy, gid[1], gid[2], gid[4]);
                } else if constexpr (R == 2) {
                    Interface::call(ci, op, reduced, gid[0], gid[1], tidy, gid[2], gid[4]);
                } else if constexpr (R == 3) {
                    Interface::call(ci, op, reduced, gid[0], gid[1], gid[2], tidy, gid[4]);
                }
            } else if constexpr (N == 6) {
                if constexpr (R == 0) {
                    Interface::call(ci, op, reduced, tidy, gid[0], gid[1], gid[2], gid[3], gid[5]);
                } else if constexpr (R == 1) {
                    Interface::call(ci, op, reduced, gid[0], tidy, gid[1], gid[2], gid[3], gid[5]);
                } else if constexpr (R == 2) {
                    Interface::call(ci, op, reduced, gid[0], gid[1], tidy, gid[2], gid[3], gid[5]);
                } else if constexpr (R == 3) {
                    Interface::call(ci, op, reduced, gid[0], gid[1], gid[2], tidy, gid[3], gid[5]);
                } else if constexpr (R == 4) {
                    Interface::call(ci, op, reduced, gid[0], gid[1], gid[2], gid[3], tidy, gid[5]);
                }
            } else {
                static_assert(nt::always_false<Interface>);
            }
        }

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
    template<typename Block, typename Interface, typename Op, typename Index, usize N, typename Reduced, typename Joined>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_nd_first(
        Op op,
        Reduced reduced,
        Joined joined, // 2d Accessor of Reduced
        Shape<Index, N - 1> reduced_shape,
        Shape<u32, N - 2> fused_grid_shape_in_x,
        u32 scratch_size,
        u32 grid_size_z,
        u32 block_index_offset_z
    ) {
        static_assert(Block::block_ndim == 2);
        const auto ci = ComputeHandle<Index, 3, Block::block_ndim, true, true, true>(
            scratch_size, Vec<u32, 2>{grid_size_z, 1}, Vec<u32, 2>{block_index_offset_z, 0});

        // Get the position within the span.
        const Index batch = blockIdx.z + block_index_offset_z;
        const Vec<u32, N - 2> block_indices = noa::offset2index(blockIdx.x, fused_grid_shape_in_x);
        Vec<Index, N - 1> gid;
        if constexpr (N == 4) {
            gid = Vec<Index, 3>::from_values(
                blockIdx.y,
                Block::block_size_y * block_indices[0] + threadIdx.y,
                Block::block_size_x * block_indices[1] + threadIdx.x
            );
        } else if constexpr (N == 5) {
            gid = Vec<Index, 4>::from_values(
                blockIdx.y,
                block_indices[0],
                Block::block_size_y * block_indices[1] + threadIdx.y,
                Block::block_size_x * block_indices[2] + threadIdx.x
            );
        } else if constexpr (N == 6) {
            gid = Vec<Index, 5>::from_values(
                blockIdx.y,
                block_indices[0],
                block_indices[1],
                Block::block_size_y * block_indices[2] + threadIdx.y,
                Block::block_size_x * block_indices[3] + threadIdx.x
            );
        }
        Interface::init(ci, op, batch);

        // Traverse the entire span of this batch.
        if constexpr (N == 4) {
            for (Index cd = gid[0]; cd < reduced_shape[0]; cd += gridDim.y)
                for (Index ch = gid[1]; ch < reduced_shape[1]; ch += Block::block_size_y * fused_grid_shape_in_x[0])
                    for (Index cw = gid[2]; cw < reduced_shape[2]; cw += Block::block_size_x * fused_grid_shape_in_x[1])
                        Interface::call(ci, op, reduced, batch, cd, ch, cw);
        } else if constexpr (N == 5) {
            for (Index i = gid[0]; i < reduced_shape[0]; i += gridDim.y)
                for (Index j = gid[1]; j < reduced_shape[1]; j += fused_grid_shape_in_x[0])
                    for (Index k = gid[2]; k < reduced_shape[2]; k += Block::block_size_y * fused_grid_shape_in_x[1])
                        for (Index l = gid[3]; l < reduced_shape[3]; l += Block::block_size_x * fused_grid_shape_in_x[2])
                                Interface::call(ci, op, reduced, batch, i, j, k, l);
        } else if constexpr (N == 6) {
            for (Index i = gid[0]; i < reduced_shape[0]; i += gridDim.y)
                for (Index j = gid[1]; j < reduced_shape[1]; j += fused_grid_shape_in_x[0])
                    for (Index k = gid[2]; k < reduced_shape[2]; k += fused_grid_shape_in_x[1])
                        for (Index l = gid[3]; l < reduced_shape[3]; l += Block::block_size_y * fused_grid_shape_in_x[2])
                            for (Index m = gid[4]; m < reduced_shape[4]; m += Block::block_size_x * fused_grid_shape_in_x[3])
                                Interface::call(ci, op, reduced, batch, i, j, k, l, m);
        }

        Interface::deinit(ci, op, batch);

        // Reduce to one value per block.
        const Index tid = threadIdx.y * Block::block_size_x + threadIdx.x;
        const Index bid = blockIdx.y * gridDim.x + blockIdx.x;
        block_join<Interface, Block::block_size, false>(op, reduced, joined, tid, batch, bid);
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

    template<typename Block, typename Interface, typename Op, typename Index, usize N, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_nd_small(Op op, Reduced reduced, Output output, Vec<Index, N - 1> shape_unbatched, u32 scratch_size) {
        static_assert(N > 2 or Block::block_size_y == 1);
        const Index batch = blockIdx.x;

        const auto ci = ComputeHandle<Index, 1, Block::block_ndim, false, true, false>(scratch_size);
        Interface::init(ci, op, batch);

        if constexpr (N == 2) {
            const Index gid = threadIdx.x;
            for (Index i = gid; i < shape_unbatched[0]; i += Block::block_size_x)
                Interface::call(ci, op, reduced, batch, i);
        } else if constexpr (N == 3) {
            const auto gid = Vec<Index, 2>::from_values(threadIdx.y, threadIdx.x);
            for (Index i = gid[0]; i < shape_unbatched[0]; i += Block::block_size_y)
                for (Index j = gid[1]; j < shape_unbatched[1]; j += Block::block_size_x)
                    Interface::call(ci, op, reduced, batch, i, j);
        } else if constexpr (N == 4) {
            const auto gid = Vec<Index, 3>::from_values(0, threadIdx.y, threadIdx.x);
            for (Index i = gid[0]; i < shape_unbatched[0]; ++i)
                for (Index j = gid[1]; j < shape_unbatched[1]; j += Block::block_size_y)
                    for (Index k = gid[2]; k < shape_unbatched[2]; k += Block::block_size_x)
                        Interface::call(ci, op, reduced, batch, i, j, k);
        } else if constexpr (N == 5) {
            const auto gid = Vec<Index, 4>::from_values(0, 0, threadIdx.y, threadIdx.x);
            for (Index i = gid[0]; i < shape_unbatched[0]; ++i)
                for (Index j = gid[1]; j < shape_unbatched[1]; ++j)
                    for (Index k = gid[2]; k < shape_unbatched[2]; k += Block::block_size_y)
                        for (Index l = gid[3]; l < shape_unbatched[3]; l += Block::block_size_x)
                            Interface::call(ci, op, reduced, batch, i, j, k, l);
        } else if constexpr (N == 6) {
            const auto gid = Vec<Index, 5>::from_values(0, 0, 0, threadIdx.y, threadIdx.x);
            for (Index i = gid[0]; i < shape_unbatched[0]; ++i)
                for (Index j = gid[1]; j < shape_unbatched[1]; ++j)
                    for (Index k = gid[2]; k < shape_unbatched[2]; ++k)
                        for (Index l = gid[3]; l < shape_unbatched[3]; l += Block::block_size_y)
                            for (Index m = gid[4]; m < shape_unbatched[4]; m += Block::block_size_x)
                                Interface::call(ci, op, reduced, batch, i, j, k, l, m);
        } else {
            static_assert(nt::always_false<Op>);
        }

        Interface::deinit(ci, op, batch);

        Index tid;
        if constexpr (N == 2)
            tid = threadIdx.x;
        else
            tid = threadIdx.y * Block::block_size_x + threadIdx.x;
        block_join_and_post<Interface, Block::block_size, false>(op, reduced, output, tid, batch);
    }
}

namespace noa::cuda::details {
    template<typename Config, typename Op, typename Reduced, typename Output, typename Index, usize N> requires (N >= 2)
    void launch_reduce_axes_iwise_single_axis(
        const Shape<Index, N>& input_shape,
        Vec<bool, N> axes_to_reduce,
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

        constexpr usize POP_COUNT_FOR_OUTER_GRID = N <= 3 ? 4 - N : 0; // N=2->2, N=3->1, N>3->0
        if constexpr (N >= 3) {
            if (axes_to_reduce[N - 1]) {
                auto output_no_width = nd::reconfig_accessors(output, Vec<u32, N - 1>::arange());
                using OutputNoWidth = decltype(output_no_width);

                // Block shape.
                using block_big_x = ReduceIwiseSingleAxisBlock<Config, 8>;
                using block_small_x = ReduceIwiseSingleAxisBlock<Config, 2>;
                const bool is_big_x = safe_cast<u32>(input_shape[N - 1]) > 512;
                const auto n_threads_x = is_big_x ? block_big_x::block_size_x : block_small_x::block_size_x;
                const auto n_threads_y = is_big_x ? block_big_x::block_size_y : block_small_x::block_size_y;
                const auto shape_hw = input_shape.filter(N - 2, N - 1);

                // Grid shape. Exclude the width, each block line reduces its row.
                // N==2 -> 1D grid.
                // N==3 -> 2D grid.
                // N==4 -> 3D grid.
                // N>=5 -> 3D grid (with fused axes in X).
                const auto grid = GridND(
                    input_shape.pop_back(),
                    Shape<isize, N - 1>::from_value(1).template set<N - 2>(n_threads_y)
                );
                check(grid.n_launches_x() == 1);
                const auto grid_outer_shape = grid.outer_shape().vec.template pop_front<POP_COUNT_FOR_OUTER_GRID>();
                const auto grid_fused_shape_in_x_unbatched = grid.fused_shape().vec.pop_front();

                for (u32 z{}; z < grid.n_launches_z(); ++z) {
                    for (u32 y{}; y < grid.n_launches_y(); ++y) {
                        const auto config = LaunchConfig{
                            .n_blocks = grid.dim3_shape_for_launch(z, y, 0),
                            .n_threads = dim3(n_threads_x, n_threads_y, 1),
                            .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                        };

                        const auto grid_outer_offset = grid.block_offset_for_launch(z, y).template pop_front<POP_COUNT_FOR_OUTER_GRID>();
                        if (is_big_x) {
                            stream.enqueue(
                                reduce_width_iwise<N, block_big_x, Interface, OpDecay, Index, ReducedDecay, OutputNoWidth>,
                                config, op, shape_hw, reduced, output_no_width,
                                static_cast<u32>(scratch_size), grid_outer_shape, grid_outer_offset, grid_fused_shape_in_x_unbatched
                            );
                        } else {
                            stream.enqueue(
                                reduce_width_iwise<N, block_small_x, Interface, OpDecay, Index, ReducedDecay, OutputNoWidth>,
                                config, op, shape_hw, reduced, output_no_width,
                                static_cast<u32>(scratch_size), grid_outer_shape, grid_outer_offset, grid_fused_shape_in_x_unbatched
                            );
                        }
                    }
                }
                return;
            }
        } else {
            check(axes_to_reduce[N - 1] == false); // N == 2
        }

        // Reduce "height".
        // The kernel needs the axis to reduce at the "height" position.
        // The width should still be at the rightmost dimension.
        auto order = squeeze_empty_dimensions_left(axes_to_reduce.template as<i32>() + 1);
        std::swap(order[N - 2], order[N - 1]); // move the width back to rightmost

        // Reorder to (X..., axis_to_reduce, width).
        auto reordered_shape = input_shape.permute(order);
        auto reordered_output = output.map([&order](const auto& accessor) {
            return accessor.permute(order);
        });

        // Remove the empty/reduced axis from the output.
        auto order_output_dim = Vec<u32, N - 1>::arange().template set<N - 2>(N - 1);
        auto reordered_output_no_height = nd::reconfig_accessors(reordered_output, order_output_dim);
        using ReorderedOutputNoHeight = decltype(reordered_output_no_height);

        // Block shape.
        using Block = ReduceIwise2dBlock<Config>;
        const auto n_threads = dim3(Block::block_size_x, Block::block_size_y);

        // Grid shape.
        // N==2 -> 1D grid.
        // N==3 -> 2D grid.
        // N==4 -> 3D grid.
        // N>=5 -> 3D grid (with fused axes in X).
        const auto grid = GridND(
            reordered_shape.filter(order_output_dim),
            Shape<isize, N - 1>::from_value(1).template set<N - 2>(Block::block_size_x)
        );
        check(grid.n_launches_x() == 1);

        const auto grid_outer_shape = grid.outer_shape().vec.template pop_front<POP_COUNT_FOR_OUTER_GRID>();
        const auto grid_fused_shape_in_x_unbatched = grid.fused_shape().vec.pop_front();
        const auto shape_hw = reordered_shape.filter(N - 2, N - 1);

        for (u32 z{}; z < grid.n_launches_z(); ++z) {
            for (u32 y{}; y < grid.n_launches_y(); ++y) {
                const auto config = LaunchConfig{
                    .n_blocks = grid.dim3_shape_for_launch(z, y, 0),
                    .n_threads = n_threads,
                    .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                };
                const auto grid_outer_offset = grid.block_offset_for_launch(z, y).template pop_front<POP_COUNT_FOR_OUTER_GRID>();

                if constexpr (N >= 2) {
                    if (axes_to_reduce[0]) {
                        stream.enqueue(
                            reduce_height_iwise<N, 0, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutputNoHeight>,
                            config, op, shape_hw, reduced, reordered_output_no_height,
                            static_cast<u32>(scratch_size), grid_outer_shape, grid_outer_offset, grid_fused_shape_in_x_unbatched
                        );
                        continue;
                    }
                }
                if constexpr (N >= 3) {
                    if (axes_to_reduce[1]) {
                        stream.enqueue(
                            reduce_height_iwise<N, 1, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutputNoHeight>,
                            config, op, shape_hw, reduced, reordered_output_no_height,
                            static_cast<u32>(scratch_size), grid_outer_shape, grid_outer_offset, grid_fused_shape_in_x_unbatched
                        );
                        continue;
                    }
                }
                if constexpr (N >= 4) {
                    if (axes_to_reduce[2]) {
                        stream.enqueue(
                            reduce_height_iwise<N, 2, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutputNoHeight>,
                            config, op, shape_hw, reduced, reordered_output_no_height,
                            static_cast<u32>(scratch_size), grid_outer_shape, grid_outer_offset, grid_fused_shape_in_x_unbatched
                        );
                        continue;
                    }
                }
                if constexpr (N >= 5) {
                    if (axes_to_reduce[3]) {
                        stream.enqueue(
                            reduce_height_iwise<N, 3, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutputNoHeight>,
                            config, op, shape_hw, reduced, reordered_output_no_height,
                            static_cast<u32>(scratch_size), grid_outer_shape, grid_outer_offset, grid_fused_shape_in_x_unbatched
                        );
                        continue;
                    }
                }
                if constexpr (N >= 6) {
                    if (axes_to_reduce[4]) {
                        stream.enqueue(
                            reduce_height_iwise<N, 4, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutputNoHeight>,
                            config, op, shape_hw, reduced, reordered_output_no_height,
                            static_cast<u32>(scratch_size), grid_outer_shape, grid_outer_offset, grid_fused_shape_in_x_unbatched
                        );
                    }
                }
            }
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

        // Reduce to one value.
        if (axes_empty_or_to_reduce == true) {
            constexpr auto CONTIGUOUS = nd::AccessorConfig{.enforce_contiguous = true};
            const auto output_1d = nd::reconfig_accessors<CONTIGUOUS>(output, 0);
            return reduce_iwise<Config>(
                input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced),
                output_1d, stream, scratch_size
            );
        }

        if constexpr (N > 1) {
            // Batch reduction.
            if (axes_empty_or_to_reduce.pop_front() == true) {
                const auto shape_to_reduce = input_shape.pop_front();
                const auto shape_to_reduce_iz = shape_to_reduce.template as_safe<isize>();
                const auto batch = safe_cast<u32>(input_shape[0]);

                using OpDecay = std::decay_t<Op>;
                using ReducedDecay = std::decay_t<Reduced>;

                using block_1d = details::ReduceIwise1dBlock<Config>;
                using block_2d = details::ReduceIwise2dBlock<Config>;
                using Block = std::conditional_t<N == 2, block_1d, block_2d>;
                using Interface = Config::interface;

                const auto n_elements = shape_to_reduce_iz.n_elements();
                constexpr auto SMALL_THRESHOLD = static_cast<isize>(Config::block_size * 32);
                auto output_1d = nd::reconfig_accessors(output, 0);
                using Output1d = decltype(output_1d);

                const auto n_bytes_of_shared_memory =
                    n_bytes_of_shared_memory_to_allocate_for_reduction<Reduced>(
                        Config::block_size, scratch_size);

                if (not Config::allow_two_kernels or n_elements <= SMALL_THRESHOLD) {
                    const auto config = LaunchConfig{
                        .n_blocks = batch,
                        .n_threads = dim3(Block::block_size_x, Block::block_size_y, 1),
                        .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                    };
                    stream.enqueue(
                        details::reduce_axes_iwise_nd_small<Block, Interface, OpDecay, Index, N, ReducedDecay, Output1d>,
                        config, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, shape_to_reduce.vec,
                        static_cast<u32>(scratch_size)
                    );
                } else {
                    // Compute the grid shape necessary to loop through the reduced elements.
                    // We add one to the shape_to_reduce so that the returned config has config.n_blocks.z == 1.
                    // This is because the leftmost dimension is on a separate dimension of the grid (z).
                    auto [config, n_blocks_per_batch, fused_grid_shape_in_x] =
                        details::reduce_iwise_nd_first_config<Block>(
                            shape_to_reduce_iz.push_front(1));
                    config.n_bytes_of_shared_memory = n_bytes_of_shared_memory;

                    // Allocate the 2d buffer.
                    constexpr bool HAS_REDUCED = not nt::empty_tuple<Reduced>;
                    const auto joined_size = HAS_REDUCED ? n_blocks_per_batch * batch : 0;
                    using Joined = AccessorRestrictContiguous<ReducedDecay, 2, Index>;
                    auto buffer = AllocatorDevice::allocate_async<ReducedDecay>(joined_size, stream);
                    auto joined = Joined(buffer.get(), Strides<Index, 2>::from_values(n_blocks_per_batch, 1));

                    auto grid_z = GridZ(batch, 1);
                    for (u32 z{}; z < grid_z.n_launches(); ++z) {
                        config.n_blocks.z = grid_z.n_blocks(z);
                        if constexpr (N >= 4) {
                            stream.enqueue(
                                details::reduce_axes_iwise_nd_first<Block, Interface, OpDecay, Index, N, ReducedDecay, Joined>,
                                config, op, reduced, joined, shape_to_reduce, fused_grid_shape_in_x,
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

            // Reduce one axis.
            details::launch_reduce_axes_iwise_single_axis<Config>(
                input_shape, axes_to_reduce, std::forward<Op>(op),
                std::forward<Reduced>(reduced), output, stream,
                scratch_size
            );
        }
    }
}
