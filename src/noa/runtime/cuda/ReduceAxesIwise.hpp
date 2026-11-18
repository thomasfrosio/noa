#pragma once
#include "noa/runtime/cuda/IncludeGuard.hpp"

#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Block.hpp"
#include "noa/runtime/cuda/ReduceIwise.hpp"

namespace noa::gpu::details {
    template<typename Config, u32 WarpCountBlockSizeX>
    struct ReduceIwiseSingleAxisBlock {
        static_assert(
            is_multiple_of(Config::block_size, Constant::WARP_SIZE) and
            Constant::WARP_SIZE <= Config::block_size and
            Config::block_size <= Limits::MAX_THREADS);

        static constexpr u32 block_size = Config::block_size; // is multiple of WARP_SIZE
        static constexpr u32 block_size_x = Config::allow_block_reshape ? best_block_size_x<Config, WarpCountBlockSizeX>() : Config::block_size_x;
        static constexpr u32 block_size_y = Config::allow_block_reshape ? block_size / block_size_x : Config::block_size_y;
        static constexpr u32 block_size_z = 1;
        static constexpr u32 block_ndim = block_size_y > 1 ? 2 : 1;
    };

    template<
        usize N, usize R,
        typename Block, typename Interface, typename Index,
        typename Op, typename Reduced, typename Output>
    struct ReduceIwiseWidthOrHeighSinglePass {
        static constexpr bool REDUCE_WIDTH = R == N - 1;
        static_assert(R <= N - 1 and N >= 2 + REDUCE_WIDTH);
        static constexpr Index BLOCK_SIZE_X = static_cast<Index>(Block::block_size_x);
        static constexpr Index BLOCK_SIZE_Y = static_cast<Index>(Block::block_size_y);
        static constexpr u32 GRID_NDIM = N <= 3 ? N - 1 : 3;
        static constexpr u32 HANDLE_GRID_NDIM = max(GRID_NDIM, Block::block_ndim);
        using handle_type = ComputeHandle<Index, HANDLE_GRID_NDIM, Block, true, true, false>;
        using grid_fused_shape_type = Vec<u32, (N >= 5 ? N - 4 : 0)>;

        NOA_NO_UNIQUE_ADDRESS Op op;
        NOA_NO_UNIQUE_ADDRESS Reduced reduced;
        NOA_NO_UNIQUE_ADDRESS Output output;
        Shape<Index, 2> shape_hw;
        NOA_NO_UNIQUE_ADDRESS grid_fused_shape_type grid_fused_shape_in_x_unbatched{};

    public:
        NOA_FD void operator()(const handle_type& handle) requires (REDUCE_WIDTH) {
            // Compute the current global indices.
            // The grid maps all axes except the width.
            // For N >= 5, the rightmost axis of the grid needs to be unfused.
            const auto bid = handle.template block_indices<Index, GRID_NDIM>();
            const auto tid = handle.template thread_indices<Index, 2>();
            Vec<Index, N> gid;
            if constexpr (N == 3) {
                gid = Vec<Index, 3>::from_values(
                    bid[0],
                    bid[1] * BLOCK_SIZE_Y + tid[0],
                    tid[1]);
            } else if constexpr (N == 4) {
                gid = Vec<Index, 4>::from_values(
                    bid[0],
                    bid[1],
                    bid[2] * BLOCK_SIZE_Y + tid[0],
                    tid[1]);
            } else if constexpr (N == 5) {
                Vec<u32, 2> block_indices = noa::offset2index(
                    static_cast<u32>(bid[2]), grid_fused_shape_in_x_unbatched);
                gid = Vec<Index, 5>::from_values(
                    bid[0],
                    bid[1],
                    block_indices[0],
                    static_cast<Index>(block_indices[1]) * BLOCK_SIZE_Y + tid[0],
                    tid[1]);
            } else if constexpr (N == 6) {
                Vec<u32, 3> block_indices = noa::offset2index(
                    static_cast<u32>(bid[2]), grid_fused_shape_in_x_unbatched);
                gid = Vec<Index, 6>::from_values(
                    bid[0],
                    bid[1],
                    block_indices[0],
                    block_indices[1],
                    static_cast<Index>(block_indices[2]) * BLOCK_SIZE_Y + tid[0],
                    tid[1]);
            } else {
                static_assert(nt::always_false<Interface>);
            }

            if constexpr (N == 3)
                Interface::init(handle, op, gid[0], gid[1]);
            else if constexpr (N == 4)
                Interface::init(handle, op, gid[0], gid[1], gid[2]);
            else if constexpr (N == 5)
                Interface::init(handle, op, gid[0], gid[1], gid[2], gid[3]);
            else if constexpr (N == 6)
                Interface::init(handle, op, gid[0], gid[1], gid[2], gid[3], gid[4]);
            else
                static_assert(nt::always_false<Interface>);

            // Loop until the end of the row is reached.
            const bool is_valid_row = gid[N - 2] < shape_hw[0];
            for (Index cid = gid[N - 1]; cid < shape_hw[1] and is_valid_row; cid += BLOCK_SIZE_X) {
                if constexpr (N == 3)
                    Interface::call(handle, op, reduced, gid[0], gid[1], cid);
                else if constexpr (N == 4)
                    Interface::call(handle, op, reduced, gid[0], gid[1], gid[2], cid);
                else if constexpr (N == 5)
                    Interface::call(handle, op, reduced, gid[0], gid[1], gid[2], gid[3], cid);
                else if constexpr (N == 6)
                    Interface::call(handle, op, reduced, gid[0], gid[1], gid[2], gid[3], gid[4], cid);
                else
                    static_assert(nt::always_false<Block>);
            }

            if constexpr (N == 3)
                Interface::deinit(handle, op, gid[0], gid[1]);
            else if constexpr (N == 4)
                Interface::deinit(handle, op, gid[0], gid[1], gid[2]);
            else if constexpr (N == 5)
                Interface::deinit(handle, op, gid[0], gid[1], gid[2], gid[3]);
            else if constexpr (N == 6)
                Interface::deinit(handle, op, gid[0], gid[1], gid[2], gid[3], gid[4]);
            else
                static_assert(nt::always_false<Interface>);

            if constexpr (not nt::empty_tuple<Reduced>) {
                // Share the threads' initial reduction with the rest of the block.
                Reduced* shared_buffer = handle.template scratch_pointer<Reduced>(); // (as least) Block::block_size
                Reduced* joined = shared_buffer + tid[0] * BLOCK_SIZE_X;
                joined[tid[1]] = reduced;
                handle.block_synchronize();

                // Reduce each "shared row" to one element.
                reduced = block_join_shared<Interface, Block::block_size_x>(
                    handle, op, joined, static_cast<i32>(tid[1]));
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

        NOA_FD void operator()(const handle_type& handle) requires (not REDUCE_WIDTH) {
            // Global indices. One block along the height.
            const auto bid = handle.template block_indices<Index, GRID_NDIM>();
            const auto tid = handle.template thread_indices<Index, 2>();
            Vec<Index, N> gid;
            if constexpr (N == 2) {
                gid = Vec<Index, 2>::from_values(
                    tid[0],
                    bid[0] * BLOCK_SIZE_X + tid[1]);
            } else if constexpr (N == 3) {
                gid = Vec<Index, 3>::from_values(
                    bid[0],
                    tid[0],
                    bid[1] * BLOCK_SIZE_X + tid[1]);
            } else if constexpr (N == 4) {
                gid = Vec<Index, 4>::from_values(
                    bid[0],
                    bid[1],
                    tid[0],
                    bid[2] * BLOCK_SIZE_X + tid[1]);
            } else if constexpr (N == 5) {
                Vec<u32, 2> block_indices = noa::offset2index(
                    static_cast<u32>(bid[2]), grid_fused_shape_in_x_unbatched);
                gid = Vec<Index, 5>::from_values(
                    bid[0],
                    bid[1],
                    block_indices[0],
                    tid[0],
                    block_indices[1] * BLOCK_SIZE_X + tid[1]);
            } else if constexpr (N == 6) {
                Vec<u32, 3> block_indices = noa::offset2index(
                    static_cast<u32>(bid[2]), grid_fused_shape_in_x_unbatched);
                gid = Vec<Index, 6>::from_values(
                    bid[0],
                    bid[1],
                    block_indices[0],
                    block_indices[1],
                    tid[0],
                    block_indices[2] * BLOCK_SIZE_X + tid[1]);
            } else {
                static_assert(nt::always_false<Interface>);
            }

            if constexpr (N == 2)
                Interface::init(handle, op, gid[1]);
            else if constexpr (N == 3)
                Interface::init(handle, op, gid[0], gid[2]);
            else if constexpr (N == 4)
                Interface::init(handle, op, gid[0], gid[1], gid[3]);
            else if constexpr (N == 5)
                Interface::init(handle, op, gid[0], gid[1], gid[2], gid[4]);
            else if constexpr (N == 6)
                Interface::init(handle, op, gid[0], gid[1], gid[2], gid[3], gid[5]);
            else
                static_assert(nt::always_false<Interface>);

            // Process every row.
            const bool is_valid_column = gid[N - 1] < shape_hw[1];
            for (Index tidy = gid[N - 2]; tidy < shape_hw[0] and is_valid_column; tidy += BLOCK_SIZE_Y) {
                if constexpr (N == 2 and R == 0) {
                    Interface::call(handle, op, reduced, tidy, gid[1]);
                } else if constexpr (N == 3) {
                    if constexpr (R == 0) {
                        Interface::call(handle, op, reduced, tidy, gid[0], gid[2]);
                    } else if constexpr (R == 1) {
                        Interface::call(handle, op, reduced, gid[0], tidy, gid[2]);
                    }
                } else if constexpr (N == 4) {
                    if constexpr (R == 0) {
                        Interface::call(handle, op, reduced, tidy, gid[0], gid[1], gid[3]);
                    } else if constexpr (R == 1) {
                        Interface::call(handle, op, reduced, gid[0], tidy, gid[1], gid[3]);
                    } else if constexpr (R == 2) {
                        Interface::call(handle, op, reduced, gid[0], gid[1], tidy, gid[3]);
                    }
                } else if constexpr (N == 5) {
                    if constexpr (R == 0) {
                        Interface::call(handle, op, reduced, tidy, gid[0], gid[1], gid[2], gid[4]);
                    } else if constexpr (R == 1) {
                        Interface::call(handle, op, reduced, gid[0], tidy, gid[1], gid[2], gid[4]);
                    } else if constexpr (R == 2) {
                        Interface::call(handle, op, reduced, gid[0], gid[1], tidy, gid[2], gid[4]);
                    } else if constexpr (R == 3) {
                        Interface::call(handle, op, reduced, gid[0], gid[1], gid[2], tidy, gid[4]);
                    }
                } else if constexpr (N == 6) {
                    if constexpr (R == 0) {
                        Interface::call(handle, op, reduced, tidy, gid[0], gid[1], gid[2], gid[3], gid[5]);
                    } else if constexpr (R == 1) {
                        Interface::call(handle, op, reduced, gid[0], tidy, gid[1], gid[2], gid[3], gid[5]);
                    } else if constexpr (R == 2) {
                        Interface::call(handle, op, reduced, gid[0], gid[1], tidy, gid[2], gid[3], gid[5]);
                    } else if constexpr (R == 3) {
                        Interface::call(handle, op, reduced, gid[0], gid[1], gid[2], tidy, gid[3], gid[5]);
                    } else if constexpr (R == 4) {
                        Interface::call(handle, op, reduced, gid[0], gid[1], gid[2], gid[3], tidy, gid[5]);
                    }
                } else {
                    static_assert(nt::always_false<Interface>);
                }
            }

            if constexpr (N == 2)
                Interface::deinit(handle, op, gid[1]);
            else if constexpr (N == 3)
                Interface::deinit(handle, op, gid[0], gid[2]);
            else if constexpr (N == 4)
                Interface::deinit(handle, op, gid[0], gid[1], gid[3]);
            else if constexpr (N == 5)
                Interface::deinit(handle, op, gid[0], gid[1], gid[2], gid[4]);
            else if constexpr (N == 6)
                Interface::deinit(handle, op, gid[0], gid[1], gid[2], gid[3], gid[5]);
            else
                static_assert(nt::always_false<Interface>);

            if constexpr (not nt::empty_tuple<Reduced>) {
                // Share the threads' initial reduction with the rest of the block.
                Reduced* shared_buffer = handle.template scratch_pointer<Reduced>(); // (as least) Block::block_size
                Reduced* joined = shared_buffer + tid[0] * BLOCK_SIZE_X + tid[1];
                *joined = reduced;
                handle.block_synchronize();

                // Reduce the height of the block.
                #pragma unroll
                for (u32 size = Block::block_size_y; size >= 2; size /= 2) {
                    if (static_cast<u32>(tid[0]) < size / 2)
                        Interface::join(op, joined[Block::block_size_x * size / 2], *joined);
                    handle.block_synchronize();
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
    };

    template<
        usize N,
        typename Block, typename Interface, typename Index,
        typename Op, typename Reduced, typename Joined>
    struct ReduceIwisePerBatchFirstPass {
        static constexpr auto BLOCK_SIZE_Y = static_cast<Index>(Block::block_size_y);
        static constexpr auto BLOCK_SIZE_X = static_cast<Index>(Block::block_size_x);
        static constexpr u32 GRID_NDIM = N >= 3 ? 3 : 2;
        using grid_fused_shape_type = std::conditional_t<(N > 3), Shape<u32, N - 2>, Empty>;
        using handle_type = ComputeHandle<Index, GRID_NDIM, Block, true, true, true>;
        static_assert(Block::block_ndim == (N >= 3 ? 2 : 1));

        NOA_NO_UNIQUE_ADDRESS Op op;
        NOA_NO_UNIQUE_ADDRESS Reduced reduced;
        NOA_NO_UNIQUE_ADDRESS Joined joined;
        Shape<Index, N - 1> shape_no_batch;
        NOA_NO_UNIQUE_ADDRESS grid_fused_shape_type grid_fused_shape_in_x{};

        NOA_FD void operator()(const handle_type& handle) requires (N >= 4 and N <= 6) {
            // Multigrid is only along Z, so instead of handle.block_indices(),
            // call the individual indices to save one addition.
            const auto bid = Vec<Index, 3>{
                handle.block_index_z(),
                handle.raw_block_index_y(),
                handle.raw_block_index_x(),
            };

            // Get the position within the span.
            const auto tid = handle.template thread_indices<Index, 2>();
            const Vec<u32, N - 2> block_indices = noa::offset2index(static_cast<u32>(bid[2]), grid_fused_shape_in_x);
            Vec<Index, N - 1> gid;
            if constexpr (N == 4) {
                gid = Vec<Index, 3>::from_values(
                    bid[1],
                    static_cast<Index>(Block::block_size_y * block_indices[0]) + tid[0],
                    static_cast<Index>(Block::block_size_x * block_indices[1]) + tid[1]
                );
            } else if constexpr (N == 5) {
                gid = Vec<Index, 4>::from_values(
                    bid[1],
                    block_indices[0],
                    static_cast<Index>(Block::block_size_y * block_indices[1]) + tid[0],
                    static_cast<Index>(Block::block_size_x * block_indices[2]) + tid[1]
                );
            } else if constexpr (N == 6) {
                gid = Vec<Index, 5>::from_values(
                    bid[1],
                    block_indices[0],
                    block_indices[1],
                    static_cast<Index>(Block::block_size_y * block_indices[2]) + tid[0],
                    static_cast<Index>(Block::block_size_x * block_indices[3]) + tid[1]
                );
            }
            Interface::init(handle, op, bid[0]);

            // Traverse the entire span of this batch.
            const auto grid_shape_2d = handle.template grid_shape<Index, 2>();
            const auto offset_h = static_cast<Index>(Block::block_size_y * grid_fused_shape_in_x[N - 4]);
            const auto offset_w = static_cast<Index>(Block::block_size_x * grid_fused_shape_in_x[N - 3]);
            if constexpr (N == 4) {
                for (Index cd = gid[0]; cd < shape_no_batch[0]; cd += grid_shape_2d[0])
                    for (Index ch = gid[1]; ch < shape_no_batch[1]; ch += offset_h)
                        for (Index cw = gid[2]; cw < shape_no_batch[2]; cw += offset_w)
                            Interface::call(handle, op, reduced, bid[0], cd, ch, cw);
            } else if constexpr (N == 5) {
                for (Index i = gid[0]; i < shape_no_batch[0]; i += grid_shape_2d[0])
                    for (Index j = gid[1]; j < shape_no_batch[1]; j += grid_fused_shape_in_x[0])
                        for (Index k = gid[2]; k < shape_no_batch[2]; k += offset_h)
                            for (Index l = gid[3]; l < shape_no_batch[3]; l += offset_w)
                                    Interface::call(handle, op, reduced, bid[0], i, j, k, l);
            } else if constexpr (N == 6) {
                for (Index i = gid[0]; i < shape_no_batch[0]; i += grid_shape_2d[0])
                    for (Index j = gid[1]; j < shape_no_batch[1]; j += grid_fused_shape_in_x[0])
                        for (Index k = gid[2]; k < shape_no_batch[2]; k += grid_fused_shape_in_x[1])
                            for (Index l = gid[3]; l < shape_no_batch[3]; l += offset_h)
                                for (Index m = gid[4]; m < shape_no_batch[4]; m += offset_w)
                                    Interface::call(handle, op, reduced, bid[0], i, j, k, l, m);
            }

            Interface::deinit(handle, op, bid[0]);

            // Reduce to one value per block.
            const auto linear_tid = static_cast<i32>(tid[0] * BLOCK_SIZE_X + tid[1]);
            const auto linear_bid = bid[1] * grid_shape_2d[1] + bid[2];
            block_join<Interface, Block::block_size, false>(
                handle, op, reduced, joined, linear_tid, bid[0], linear_bid);
        }

        NOA_FD void operator()(const handle_type& handle) requires (N == 3) {
            // Multigrid is only along Z, so instead of handle.block_indices(),
            // call the individual indices to save one addition.
            const auto bid = Vec<Index, 3>{
                handle.block_index_z(),
                handle.raw_block_index_y(),
                handle.raw_block_index_x(),
            };
            const auto tid = handle.template thread_indices<Index, 2>();
            const auto gid = Vec<Index, 2>::from_values(
                Block::block_size_y * bid[1] + tid[0],
                Block::block_size_x * bid[2] + tid[1]
            );

            Interface::init(handle, op, bid[0]);

            const auto grid_shape_2d = handle.template grid_shape<Index, 2>();
            const auto offset_h = BLOCK_SIZE_Y * grid_shape_2d[0];
            const auto offset_w = BLOCK_SIZE_X * grid_shape_2d[1];
            for (Index ch = gid[0]; ch < shape_no_batch[0]; ch += offset_h)
                for (Index cw = gid[1]; cw < shape_no_batch[1]; cw += offset_w)
                    Interface::call(handle, op, reduced, bid[0], ch, cw);

            Interface::deinit(handle, op, bid[0]);

            const auto linear_tid = static_cast<i32>(tid[0] * BLOCK_SIZE_X + tid[1]);
            const auto linear_bid = bid[1] * grid_shape_2d[1] + bid[2];
            block_join<Interface, Block::block_size, false>(
                handle, op, reduced, joined, linear_tid, bid[0], linear_bid);
        }

        NOA_FD void operator()(const handle_type& handle) requires (N == 2) {
            const auto bid = handle.template block_indices<Index, 2>();
            const auto tid = handle.template thread_index_x<Index>();
            const Index gid = BLOCK_SIZE_X * bid[1] + tid;

            Interface::init(handle, op, bid[0]);

            for (Index cw = gid; cw < shape_no_batch[0]; cw += BLOCK_SIZE_X * handle.grid_size_x())
                Interface::call(handle, op, reduced, bid[0], cw);

            Interface::deinit(handle, op, bid[0]);

            block_join<Interface, Block::block_size, false>(
                handle, op, reduced, joined, static_cast<i32>(tid), bid[0], bid[1]);
        }
    };

    template<
        bool IS_STATIC_SCRATCH_ENABLED,
        typename Block, typename Interface, typename Index,
        typename Op, typename Reduced, typename Joined, typename Output>
    struct ReduceIwisePerBatchSecondPass {
        static_assert(Block::block_ndim == 1);
        using handle_type = ComputeHandle<Index, 1, Block, false, false, true>;

        NOA_NO_UNIQUE_ADDRESS Op op;
        NOA_NO_UNIQUE_ADDRESS Joined to_join;
        NOA_NO_UNIQUE_ADDRESS Reduced reduced;
        NOA_NO_UNIQUE_ADDRESS Output output;
        Index n_to_join;

        NOA_FD void operator()(const handle_type& handle) {
            const Index batch = handle.block_index_x();
            const Index tid = handle.thread_index_x();
            if constexpr (not nt::empty_tuple<Reduced>) {
                for (Index cid = tid; cid < n_to_join; cid += static_cast<Index>(Block::block_size_x))
                    Interface::join(op, to_join(batch, cid), reduced);
            }
            block_join_and_post<Interface, Block::block_size_x, IS_STATIC_SCRATCH_ENABLED>(
                handle, op, reduced, output, static_cast<i32>(tid), batch);
        }
    };

    template<usize N, typename Block, typename Interface, typename Index, typename Op,  typename Reduced, typename Output>
    struct ReduceIwisePerBatchSmall {
        static constexpr u32 BLOCK_NDIM = Block::block_ndim;
        static constexpr auto BLOCK_SIZE_Y = static_cast<Index>(Block::block_size_y);
        static constexpr auto BLOCK_SIZE_X = static_cast<Index>(Block::block_size_x);
        static_assert(N > 1 and BLOCK_NDIM == (N == 2 ? 1 : 2));
        using handle_type = ComputeHandle<Index, BLOCK_NDIM, Block, false, true, false>; // actual grid is 1d

        NOA_NO_UNIQUE_ADDRESS Op op;
        NOA_NO_UNIQUE_ADDRESS Reduced reduced;
        NOA_NO_UNIQUE_ADDRESS Output output;
        Shape<Index, N - 1> shape_no_batch;

        NOA_FD void operator()(const handle_type& handle) {
            const auto tid = handle.template thread_indices<Index, BLOCK_NDIM>();
            const Index batch = handle.block_index_x();

            Interface::init(handle, op, batch);

            if constexpr (N == 2) {
                for (Index i = tid[0]; i < shape_no_batch[0]; i += BLOCK_SIZE_X)
                    Interface::call(handle, op, reduced, batch, i);
            } else if constexpr (N == 3) {
                for (Index i = tid[0]; i < shape_no_batch[0]; i += BLOCK_SIZE_Y)
                    for (Index j = tid[1]; j < shape_no_batch[1]; j += BLOCK_SIZE_X)
                        Interface::call(handle, op, reduced, batch, i, j);
            } else if constexpr (N == 4) {
                for (Index i = 0; i < shape_no_batch[0]; ++i)
                    for (Index j = tid[0]; j < shape_no_batch[1]; j += BLOCK_SIZE_Y)
                        for (Index k = tid[1]; k < shape_no_batch[2]; k += BLOCK_SIZE_X)
                            Interface::call(handle, op, reduced, batch, i, j, k);
            } else if constexpr (N == 5) {
                for (Index i = 0; i < shape_no_batch[0]; ++i)
                    for (Index j = 0; j < shape_no_batch[1]; ++j)
                        for (Index k = tid[0]; k < shape_no_batch[2]; k += BLOCK_SIZE_Y)
                            for (Index l = tid[1]; l < shape_no_batch[3]; l += BLOCK_SIZE_X)
                                Interface::call(handle, op, reduced, batch, i, j, k, l);
            } else if constexpr (N == 6) {
                for (Index i = 0; i < shape_no_batch[0]; ++i)
                    for (Index j = 0; j < shape_no_batch[1]; ++j)
                        for (Index k = 0; k < shape_no_batch[2]; ++k)
                            for (Index l = tid[0]; l < shape_no_batch[3]; l += BLOCK_SIZE_Y)
                                for (Index m = tid[1]; m < shape_no_batch[4]; m += BLOCK_SIZE_X)
                                    Interface::call(handle, op, reduced, batch, i, j, k, l, m);
            } else {
                static_assert(nt::always_false<Op>);
            }

            Interface::deinit(handle, op, batch);

            const auto linear_tid = handle.template thread_linear_index<i32>();
            block_join_and_post<Interface, Block::block_size, false>(handle, op, reduced, output, linear_tid, batch);
        }
    };
}

namespace noa::gpu::details {
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
        using Op_ = std::decay_t<Op>;
        using Reduced_ = std::decay_t<Reduced>;
        using Interface = Config::interface;

        const auto n_bytes_of_scratch =
            n_bytes_of_scratch_to_allocate_for_reduction<Reduced>(
                Config::block_size, scratch_size);
        const auto scratch_size_ = static_cast<u32>(scratch_size);

        if constexpr (N >= 3) {
            if (axes_to_reduce[N - 1]) {
                auto output_no_width = nd::reconfig_accessors(output, Vec<u32, N - 1>::arange());
                using OutputNoWidth = decltype(output_no_width);

                // Block shape.
                using block_big_x = ReduceIwiseSingleAxisBlock<Config, 8>;
                using block_small_x = ReduceIwiseSingleAxisBlock<Config, 2>;
                const bool is_big_x = safe_cast<u32>(input_shape[N - 1]) > 512;
                const auto n_threads_y = is_big_x ? block_big_x::block_size_y : block_small_x::block_size_y;
                const auto shape_hw = input_shape.filter(N - 2, N - 1);

                // Grid shape. Exclude the width, each block line reduces its row.
                // N==2 -> 1D grid.
                // N==3 -> 2D grid.
                // N==4 -> 3D grid.
                // N>=5 -> 3D grid (with fused axes in X).
                const auto grid = GridND(
                    input_shape.pop_back(),
                    Shape<u32, N - 1>::from_value(1).template set<N - 2>(n_threads_y)
                );
                check(grid.n_launches_x() == 1);
                const auto grid_fused_shape_in_x_unbatched = grid.fused_shape().vec.pop_front();

                for (u32 z{}; z < grid.n_launches_z(); ++z) {
                    for (u32 y{}; y < grid.n_launches_y(); ++y) {
                        auto config = LaunchConfig{
                            .grid_shape = grid.shape_for_launch(z, y, 0),
                            .n_bytes_of_scratch = n_bytes_of_scratch,
                        };

                        const auto grid_outer_shape = grid.outer_shape().vec;
                        const auto grid_outer_offset = grid.block_offset_for_launch(z, y);

                        if (is_big_x) {
                            using Kernel = ReduceIwiseWidthOrHeighSinglePass<
                                N, N - 1, block_big_x, Interface, Index, Op_, Reduced_, OutputNoWidth>;
                            config.block_shape = block_shape<block_big_x>();
                            auto handle = typename Kernel::handle_type(scratch_size_, grid_outer_shape, grid_outer_offset);
                            auto kernel = Kernel(op, reduced, output_no_width, shape_hw, grid_fused_shape_in_x_unbatched);
                            stream.enqueue(config, handle, kernel);
                        } else {
                            config.block_shape = block_shape<block_small_x>();
                            using Kernel = ReduceIwiseWidthOrHeighSinglePass<
                                N, N - 1, block_small_x, Interface, Index, Op_, Reduced_, OutputNoWidth>;
                            auto handle = typename Kernel::handle_type(scratch_size_, grid_outer_shape, grid_outer_offset);
                            auto kernel = Kernel(op, reduced, output_no_width, shape_hw, grid_fused_shape_in_x_unbatched);
                            stream.enqueue(config, handle, kernel);
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

        // Block.
        using Block = ReduceIwise2dBlock<Config>;

        // Grid shape.
        // N==2 -> 1D grid.
        // N==3 -> 2D grid.
        // N==4 -> 3D grid.
        // N>=5 -> 3D grid (with fused axes in X).
        const auto grid = GridND(
            reordered_shape.filter(order_output_dim),
            Shape<u32, N - 1>::from_value(1).template set<N - 2>(Block::block_size_x)
        );
        check(grid.n_launches_x() == 1);

        const auto grid_outer_shape = grid.outer_shape().vec;
        const auto grid_fused_shape_in_x_unbatched = grid.fused_shape().vec.pop_front();
        const auto shape_hw = reordered_shape.filter(N - 2, N - 1);

        for (u32 z{}; z < grid.n_launches_z(); ++z) {
            for (u32 y{}; y < grid.n_launches_y(); ++y) {
                const auto config = LaunchConfig{
                    .grid_shape = grid.shape_for_launch(z, y, 0),
                    .block_shape = block_shape<Block>(),
                    .n_bytes_of_scratch = n_bytes_of_scratch,
                };
                const auto grid_outer_offset = grid.block_offset_for_launch(z, y);

                if constexpr (N >= 2) {
                    if (axes_to_reduce[0]) {
                        using Kernel = ReduceIwiseWidthOrHeighSinglePass<N, 0, Block, Interface, Index, Op_, Reduced_, ReorderedOutputNoHeight>;
                        auto handle = typename Kernel::handle_type(scratch_size_, grid_outer_shape, grid_outer_offset);
                        stream.enqueue(config, handle, Kernel(op, reduced, reordered_output_no_height, shape_hw, grid_fused_shape_in_x_unbatched));
                        continue;
                    }
                }
                if constexpr (N >= 3) {
                    if (axes_to_reduce[1]) {
                        using Kernel = ReduceIwiseWidthOrHeighSinglePass<N, 1, Block, Interface, Index, Op_, Reduced_, ReorderedOutputNoHeight>;
                        auto handle = typename Kernel::handle_type(scratch_size_, grid_outer_shape, grid_outer_offset);
                        stream.enqueue(config, handle, Kernel(op, reduced, reordered_output_no_height, shape_hw, grid_fused_shape_in_x_unbatched));
                        continue;
                    }
                }
                if constexpr (N >= 4) {
                    if (axes_to_reduce[2]) {
                        using Kernel = ReduceIwiseWidthOrHeighSinglePass<N, 2, Block, Interface, Index, Op_, Reduced_, ReorderedOutputNoHeight>;
                        auto handle = typename Kernel::handle_type(scratch_size_, grid_outer_shape, grid_outer_offset);
                        stream.enqueue(config, handle, Kernel(op, reduced, reordered_output_no_height, shape_hw, grid_fused_shape_in_x_unbatched));
                        continue;
                    }
                }
                if constexpr (N >= 5) {
                    if (axes_to_reduce[3]) {
                        using Kernel = ReduceIwiseWidthOrHeighSinglePass<N, 3, Block, Interface, Index, Op_, Reduced_, ReorderedOutputNoHeight>;
                        auto handle = typename Kernel::handle_type(scratch_size_, grid_outer_shape, grid_outer_offset);
                        stream.enqueue(config, handle, Kernel(op, reduced, reordered_output_no_height, shape_hw, grid_fused_shape_in_x_unbatched));
                        continue;
                    }
                }
                if constexpr (N >= 6) {
                    if (axes_to_reduce[4]) {
                        using Kernel = ReduceIwiseWidthOrHeighSinglePass<N, 4, Block, Interface, Index, Op_, Reduced_, ReorderedOutputNoHeight>;
                        auto handle = typename Kernel::handle_type(scratch_size_, grid_outer_shape, grid_outer_offset);
                        stream.enqueue(config, handle, Kernel(op, reduced, reordered_output_no_height, shape_hw, grid_fused_shape_in_x_unbatched));
                    }
                }
            }
        }
    }
}

namespace noa::gpu {
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
                input_shape, NOA_FWD(op), NOA_FWD(reduced),
                output_1d, stream, scratch_size
            );
        }

        if constexpr (N > 1) {
            // Batch reduction.
            if (axes_empty_or_to_reduce.pop_front() == true) {
                const auto shape_to_reduce = input_shape.pop_front();
                const auto shape_to_reduce_iz = shape_to_reduce.template as_safe<isize>();
                const auto batch = safe_cast<u32>(input_shape[0]);

                using Op_ = std::decay_t<Op>;
                using Reduced_ = std::decay_t<Reduced>;

                using block_1d = details::ReduceIwise1dBlock<Config>;
                using block_2d = details::ReduceIwise2dBlock<Config>;
                using Block = std::conditional_t<N == 2, block_1d, block_2d>;
                using Interface = Config::interface;

                const auto n_elements = shape_to_reduce_iz.n_elements();
                constexpr auto SMALL_THRESHOLD = static_cast<isize>(Config::block_size * 32);
                auto output_1d = nd::reconfig_accessors(output, 0);
                using Output1d = decltype(output_1d);

                const auto n_bytes_of_scratch =
                    n_bytes_of_scratch_to_allocate_for_reduction<Reduced>(
                        Config::block_size, scratch_size);

                if (not Config::allow_two_kernels or n_elements <= SMALL_THRESHOLD) {
                    const auto config = LaunchConfig{
                        .grid_shape = Vec<u32, 3>{1, 1, batch},
                        .block_shape = details::block_shape<Block>(),
                        .n_bytes_of_scratch = n_bytes_of_scratch,
                    };
                    using Kernel = details::ReduceIwisePerBatchSmall<N, Block, Interface, Index, Op_, Reduced_, Output1d>;
                    auto handle = typename Kernel::handle_type(static_cast<u32>(scratch_size));
                    auto kernel = Kernel{NOA_FWD(op), NOA_FWD(reduced), output_1d, shape_to_reduce};
                    stream.enqueue(config, handle, std::move(kernel));
                } else {
                    // Compute the grid shape necessary to loop through the reduced elements.
                    // We add one to the shape_to_reduce so that the returned config has config.n_blocks.z == 1.
                    // This is because the leftmost dimension is on a separate dimension of the grid (z).
                    auto [config, n_blocks_per_batch, fused_grid_shape_in_x] =
                        details::reduce_iwise_nd_first_config<Block>(
                            shape_to_reduce_iz.push_front(1));
                    config.n_bytes_of_scratch = n_bytes_of_scratch;

                    // Allocate the 2d buffer.
                    constexpr bool HAS_REDUCED = not nt::empty_tuple<Reduced>;
                    const auto joined_size = HAS_REDUCED ? n_blocks_per_batch * batch : 0;
                    using Joined = AccessorRestrictContiguous<Reduced_, 2, Index>;
                    auto buffer = AllocatorDevice::allocate_async<Reduced_>(joined_size, stream);
                    auto joined = Joined(buffer.get(), Strides<Index, 2>::from_values(n_blocks_per_batch, 1));

                    using KernelFirst = details::ReduceIwisePerBatchFirstPass<N, Block, Interface, Index, Op_, Reduced_, Joined>;
                    auto kernel = KernelFirst{op, reduced, joined, shape_to_reduce};
                    if constexpr (N >= 4)
                        kernel.grid_fused_shape_in_x = fused_grid_shape_in_x;

                    // Launch.
                    // For N > 2, the grid is 3D and the batch is along the grid Z.
                    // For N == 2, the grid is 2D and the batch is along the grid Y.
                    // TODO Multigrid is only along Z for N >= 3, so the compute handle stores
                    //      8 additional bytes for nothing. Add option to ComputeHandle to have
                    //      per axis multigrid, instead of assuming ZY.
                    constexpr auto GRID_NDIM = KernelFirst::GRID_NDIM;
                    using GridBatch = std::conditional_t<GRID_NDIM == 2, GridY, GridZ>; // most likely GridZ == GridY
                    auto grid = GridBatch(batch, 1);
                    NOA_ASSERT(grid.n_blocks_total() == batch);
                    auto grid_shape = Vec<u32, 2>::from_values(batch, config.grid_shape[1]);
                    auto grid_offset = Vec<u32, 2>::from_value(0);

                    constexpr usize B = 0 + N == 2; // Z, or Y if N == 2
                    for (u32 b{}; b < grid.n_launches(); ++b) {
                        config.grid_shape[B] = grid.n_blocks(b);
                        grid_offset[B] = grid.offset(b);
                        auto handle = typename KernelFirst::handle_type(
                            static_cast<u32>(scratch_size), grid_shape, grid_offset);
                        stream.enqueue(config, handle, kernel);
                    }

                    // Second kernel.
                    // TODO If there's no value to reduce, check that op has a valid post and if not skip the launch.
                    constexpr bool IS_STATIC_SCRATCH_ENABLED = is_static_scratch_enabled();
                    using BlockSecond = StaticBlock<HAS_REDUCED ? Config::block_size : 1, 1, 1>;
                    using KernelSecond = details::ReduceIwisePerBatchSecondPass<
                        IS_STATIC_SCRATCH_ENABLED, BlockSecond, Interface, Index, Op_, Reduced_, Joined, Output1d>;
                    using HandleSecond = KernelSecond::handle_type;
                    const auto config_second = LaunchConfig{
                        .grid_shape = Vec<u32, 3>{1, 1, batch},
                        .block_shape = details::block_shape<BlockSecond>(),
                        .n_bytes_of_scratch =
                            (HAS_REDUCED and not IS_STATIC_SCRATCH_ENABLED) ?
                            BlockSecond::block_size * sizeof(Reduced_) : 0,
                    };
                    stream.enqueue(config_second, HandleSecond{}, KernelSecond{
                        NOA_FWD(op), joined, NOA_FWD(reduced), output_1d, static_cast<Index>(n_blocks_per_batch)
                    });
                }
                return;
            }

            // Reduce one axis.
            details::launch_reduce_axes_iwise_single_axis<Config>(
                input_shape, axes_to_reduce, NOA_FWD(op),
                NOA_FWD(reduced), output, stream, scratch_size
            );
        }
    }
}
