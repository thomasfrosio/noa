#pragma once
#include "noa/runtime/gpu/IncludeGuard.hpp"

#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/gpu/Allocators.hpp"
#include "noa/runtime/gpu/Block.hpp"
#include "noa/runtime/gpu/ComputeHandle.hpp"

namespace noa::gpu::details {
    template<typename Config>
    struct ReduceIwise2dBlock {
        static_assert(
            is_multiple_of(Config::block_size, Constant::WARP_SIZE) and
            Constant::WARP_SIZE <= Config::block_size and
            Config::block_size <= Limits::MAX_THREADS);

        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = Config::block_size; // is multiple of WARP_SIZE
        static constexpr u32 block_size_x = Config::allow_block_reshape ? Constant::WARP_SIZE : Config::block_size_x;
        static constexpr u32 block_size_y = Config::allow_block_reshape ? block_size / block_size_x : Config::block_size_y;
        static constexpr u32 block_size_z = 1;
        static constexpr u32 block_work_size_x = block_size_x * Config::indices_per_thread_x;
        static constexpr u32 block_work_size_y = block_size_y * Config::indices_per_thread_y;
        static constexpr u32 block_ndim = block_size_y > 1 ? 2 : 1;
    };

    template<typename Config>
    struct ReduceIwise1dBlock {
        // Enforce a 1d block. If we are not allowed to reshape the 2d block to 1d, throw an error.
        static_assert(Config::allow_block_reshape or (Config::block_size_y == 1 and Config::indices_per_thread_y == 1));
        static_assert(
            is_multiple_of(Config::block_size, Constant::WARP_SIZE) and
            Constant::WARP_SIZE <= Config::block_size and
            Config::block_size <= Limits::MAX_THREADS);

        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = Config::block_size; // is multiple of WARP_SIZE
        static constexpr u32 block_size_x = block_size;
        static constexpr u32 block_size_y = 1;
        static constexpr u32 block_size_z = 1;
        static constexpr u32 block_work_size_x = block_size_x * Config::indices_per_thread_x;
        static constexpr u32 block_work_size_y = 1;
        static constexpr u32 block_ndim = 1;
    };
}

namespace noa::gpu::details {
    template<usize N, typename Block, typename Interface, typename Op, typename Index, typename Reduced>
    struct ReduceIwiseFirstPassLarge {
        static constexpr usize GRID_NDIM = N >= 3 ? 3 : N;
        using handle_type = ComputeHandle<Index, GRID_NDIM, Block, false, true, true>;
        using grid_fused_shape_type = std::conditional_t<(N > 3), Shape<u32, N - 2>, Empty>;

        NOA_NO_UNIQUE_ADDRESS Op op;
        NOA_NO_UNIQUE_ADDRESS Reduced reduced;
        Reduced* joined;
        Shape<Index, N> shape;
        NOA_NO_UNIQUE_ADDRESS grid_fused_shape_type grid_fused_shape_in_x{};

    public:
        NOA_FD void operator()(const handle_type& handle) requires (N == 1) {
            Interface::init(handle, op, Index{});

            const Index tid = handle.thread_index_x();
            const Index bid = handle.block_index_x();
            const auto gid = Vec<Index, 1>::from_values(static_cast<Index>(Block::block_size) * bid + tid);

            const auto offset = static_cast<Index>(Block::block_size * handle.template grid_size_x<u32>());
            for (Index cw = gid[0]; cw < shape[0]; cw += offset)
                Interface::call(handle, op, reduced, cw);

            Interface::deinit(handle, op, Index{});

            block_join<Interface, Block::block_size, false>(handle, op, reduced, joined, static_cast<i32>(tid), bid);
        }

        NOA_FD void operator()(const handle_type& handle) requires (N == 2) {
            Interface::init(handle, op, Index{});

            const auto tid = handle.template thread_indices<u32, 2>();
            const auto bid = handle.template block_indices<u32, 2>();
            const auto grid_shape = handle.template grid_shape<u32, 2>();

            const auto gid = Vec<Index, 2>::from_values(
                Block::block_size_y * bid[0] + tid[0],
                Block::block_size_x * bid[1] + tid[1]
            );

            const auto offset_y = static_cast<Index>(Block::block_size_y * grid_shape[0]);
            const auto offset_x = static_cast<Index>(Block::block_size_x * grid_shape[1]);
            for (Index ch = gid[0]; ch < shape[0]; ch += offset_y)
                for (Index cw = gid[1]; cw < shape[1]; cw += offset_x)
                    Interface::call(handle, op, reduced, ch, cw);

            Interface::deinit(handle, op, Index{});

            const auto linear_tid = static_cast<i32>(tid[0] * Block::block_size_x + tid[1]);
            const auto linear_bid = static_cast<Index>(bid[0] * grid_shape[1] + bid[1]);
            block_join<Interface, Block::block_size, false>(handle, op, reduced, joined, linear_tid, linear_bid);
        }

        NOA_FD void operator()(const handle_type& handle) requires (N == 3) {
            Interface::init(handle, op, Index{});

            const auto tid = handle.template thread_indices<u32, 2>();
            const auto bid = handle.template block_indices<u32, 3>();
            const auto grid_shape = handle.template grid_shape<u32, 3>();

            const auto gid = Vec<Index, 3>::from_values(
                bid[0],
                Block::block_size_y * bid[1] + tid[0],
                Block::block_size_x * bid[2] + tid[1]
            );

            for (Index cd = gid[0]; cd < shape[0]; cd += static_cast<Index>(grid_shape[0]))
                for (Index ch = gid[1]; ch < shape[1]; ch += static_cast<Index>(Block::block_size_y * grid_shape[1]))
                    for (Index cw = gid[2]; cw < shape[2]; cw += static_cast<Index>(Block::block_size_x * grid_shape[2]))
                        Interface::call(handle, op, reduced, cd, ch, cw);

            Interface::deinit(handle, op, Index{});

            const auto linear_tid = static_cast<i32>(tid[0] * Block::block_size_x + tid[1]);
            const auto linear_bid = static_cast<Index>((bid[0] * grid_shape[1] + bid[1]) * grid_shape[2] + bid[2]);
            block_join<Interface, Block::block_size, false>(handle, op, reduced, joined, linear_tid, linear_bid);
        }

        NOA_FD void operator()(const handle_type& handle) requires (N > 3) {
            Interface::init(handle, op, Index{});

            const auto tid = handle.template thread_indices<u32, 2>();
            const auto bid = handle.template block_indices<u32, 3>();
            const auto grid_shape = handle.template grid_shape<u32, 3>();

            const Vec<u32, N - 2> block_indices = noa::offset2index(bid[2], grid_fused_shape_in_x);
            const auto offset_y = static_cast<Index>(Block::block_size_y * grid_fused_shape_in_x[N - 4]);
            const auto offset_x = static_cast<Index>(Block::block_size_x * grid_fused_shape_in_x[N - 3]);

            if constexpr (N == 4) {
                const auto gid = Vec<Index, 4>::from_values(
                    bid[0],
                    bid[1],
                    Block::block_size_y * block_indices[0] + tid[0],
                    Block::block_size_x * block_indices[1] + tid[1]
                );
                for (Index i = gid[0]; i < shape[0]; i += static_cast<Index>(grid_shape[0]))
                    for (Index j = gid[1]; j < shape[1]; j += static_cast<Index>(grid_shape[1]))
                        for (Index k = gid[2]; k < shape[2]; k += offset_y)
                            for (Index l = gid[3]; l < shape[3]; l += offset_x)
                                Interface::call(handle, op, reduced, i, j, k, l);
            } else if constexpr (N == 5) {
                const auto gid = Vec<Index, 5>::from_values(
                    bid[0],
                    bid[1],
                    block_indices[0],
                    Block::block_size_y * block_indices[1] + tid[0],
                    Block::block_size_x * block_indices[2] + tid[1]
                );
                for (Index i = gid[0]; i < shape[0]; i += static_cast<Index>(grid_shape[0]))
                    for (Index j = gid[1]; j < shape[1]; j += static_cast<Index>(grid_shape[1]))
                        for (Index k = gid[2]; k < shape[2]; k += static_cast<Index>(grid_fused_shape_in_x[0]))
                            for (Index l = gid[3]; l < shape[3]; l += offset_y)
                                for (Index m = gid[4]; m < shape[4]; m += offset_x)
                                    Interface::call(handle, op, reduced, i, j, k, l, m);
            } else if constexpr (N == 6) {
                const auto gid = Vec<Index, 6>::from_values(
                    bid[0],
                    bid[1],
                    block_indices[0],
                    block_indices[1],
                    Block::block_size_y * block_indices[2] + tid[0],
                    Block::block_size_x * block_indices[3] + tid[1]
                );
                for (Index i = gid[0]; i < shape[0]; i += static_cast<Index>(grid_shape[0]))
                    for (Index j = gid[1]; j < shape[1]; j += static_cast<Index>(grid_shape[1]))
                        for (Index k = gid[2]; k < shape[2]; k += static_cast<Index>(grid_fused_shape_in_x[0]))
                            for (Index l = gid[3]; l < shape[3]; l += static_cast<Index>(grid_fused_shape_in_x[1]))
                                for (Index m = gid[4]; m < shape[4]; m += offset_y)
                                    for (Index n = gid[5]; n < shape[5]; n += offset_x)
                                        Interface::call(handle, op, reduced, i, j, k, l, m, n);
            } else {
                static_assert(nt::always_false<Op>);
            }

            Interface::deinit(handle, op, Index{});

            // Reduce to one value per block.
            const auto linear_tid = static_cast<i32>(tid[0] * Block::block_size_x + tid[1]);
            const auto linear_bid = static_cast<Index>((bid[0] * grid_shape[1] + bid[1]) * grid_shape[2] + bid[2]);
            block_join<Interface, Block::block_size, false>(handle, op, reduced, joined, linear_tid, linear_bid);
        }
    };

    template<
        bool IS_STATIC_SCRATCH_ENABLED,
        typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    struct ReduceIwiseSecondPassLarge {
        using handle_type = ComputeHandle<Index, 1, Block, false, false, true>;

        NOA_NO_UNIQUE_ADDRESS Op op;
        Reduced* to_join;
        Index n_to_join;
        NOA_NO_UNIQUE_ADDRESS Reduced reduced;
        NOA_NO_UNIQUE_ADDRESS Output output;

    public:
        NOA_FD void operator()(const handle_type& handle) {
            const Index tid = handle.thread_index_x();
            if constexpr (not nt::empty_tuple<Reduced>) {
                for (Index cid = tid; cid < n_to_join; cid += static_cast<Index>(Block::block_size))
                    Interface::join(op, to_join[cid], reduced);
            }
            block_join_and_post<Interface, Block::block_size, IS_STATIC_SCRATCH_ENABLED>(
                handle, op, reduced, output, static_cast<i32>(tid), Index{});
        }
    };

    template<usize N, typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    struct ReduceIwiseSmall {
        static constexpr u32 BLOCK_NDIM = N == 1 ? 1 : 2;
        static constexpr u32 GRID_NDIM = BLOCK_NDIM;
        using handle_type = ComputeHandle<Index, GRID_NDIM, Block, false, true, false>;

        NOA_NO_UNIQUE_ADDRESS Op op;
        NOA_NO_UNIQUE_ADDRESS Reduced reduced;
        NOA_NO_UNIQUE_ADDRESS Output output;
        Shape<Index, N> shape;

    public:
        NOA_FD void operator()(const handle_type& handle) {
            Interface::init(handle, op, Index{});

            const auto tid = handle.template thread_indices<Index, BLOCK_NDIM>();
            if constexpr (N == 1) {
                for (Index i = tid[0]; i < shape[0]; i += static_cast<Index>(Block::block_size))
                    Interface::call(handle, op, reduced, i);
            } else if constexpr (N == 2) {
                for (Index i = tid[0]; i < shape[0]; i += static_cast<Index>(Block::block_size_y))
                    for (Index j = tid[1]; j < shape[1]; j += static_cast<Index>(Block::block_size_x))
                        Interface::call(handle, op, reduced, i, j);
            } else if constexpr (N == 3) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = tid[0]; j < shape[1]; j += static_cast<Index>(Block::block_size_y))
                        for (Index k = tid[1]; k < shape[2]; k += static_cast<Index>(Block::block_size_x))
                            Interface::call(handle, op, reduced, i, j, k);
            } else if constexpr (N == 4) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = tid[0]; k < shape[2]; k += static_cast<Index>(Block::block_size_y))
                            for (Index l = tid[1]; l < shape[3]; l += static_cast<Index>(Block::block_size_x))
                                Interface::call(handle, op, reduced, i, j, k, l);
            } else if constexpr (N == 5) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            for (Index l = tid[0]; l < shape[3]; l += static_cast<Index>(Block::block_size_y))
                                for (Index m = tid[1]; m < shape[4]; m += static_cast<Index>(Block::block_size_x))
                                        Interface::call(handle, op, reduced, i, j, k, l, m);
            } else if constexpr (N == 6) {
                for (Index i = 0; i < shape[0]; ++i)
                    for (Index j = 0; j < shape[1]; ++j)
                        for (Index k = 0; k < shape[2]; ++k)
                            for (Index l = 0; l < shape[3]; ++l)
                                for (Index m = tid[0]; m < shape[4]; m += static_cast<Index>(Block::block_size_y))
                                    for (Index n = tid[1]; n < shape[5]; n += static_cast<Index>(Block::block_size_x))
                                        Interface::call(handle, op, reduced, i, j, k, l, m, n);
            }

            Interface::deinit(handle, op, Index{});

            i32 linear_tid = static_cast<i32>(tid.back());
            if constexpr (N > 1)
                linear_tid += static_cast<i32>(tid[0]) * static_cast<i32>(Block::block_size_x);
            block_join_and_post<Interface, Block::block_size, false>(handle, op, reduced, output, linear_tid, Index{});
        }
    };

    template<typename Block, usize N>
    auto reduce_iwise_nd_first_config(const Shape<isize, N>& shape_to_reduce) {
        // Note that while we could increase the grid size when the reduced tuple is empty, we keep it as specified
        // by the caller. The default value is large enough, and it's only for large reductions that increasing
        // would help. In this rare case, it's up to the user, for now at least.
        constexpr auto max_grid_size = static_cast<isize>(Block::max_grid_size);
        constexpr auto block_work_size = [] {
            auto o = Vec<isize, N>::from_value(1);
            if constexpr (N > 1)
                o[N - 2] = static_cast<isize>(Block::block_work_size_y);
            o[N - 1] = static_cast<isize>(Block::block_work_size_x);
            return o;
        }();

        // Set the number of blocks while keeping it under the maximum allowed.
        // Use the block work-size to account for the number of iterations per thread.
        auto grid_shape_isize = Shape<isize, N>::from_value(1);
        for (isize i = static_cast<isize>(N) - 1; i >= 0; --i) {
            const auto n_blocks_necessary = divide_up(shape_to_reduce[i], block_work_size[i]);
            const auto n_blocks_allowed = max_grid_size / grid_shape_isize.n_elements();
            grid_shape_isize[i] = min(n_blocks_necessary, n_blocks_allowed);
        }

        // Build the grid.
        // Fuse the rightmost dimensions together to fit the 3D grid and
        // save the fused block shape to decompose the offset to indices inside the kernel.
        const auto grid_shape = grid_shape_isize.template as<u32>();
        const auto [grid_shape_3d, grid_shape_fused_in_x] = [&] {
            if constexpr (N >= 3) {
                auto blocks_to_fused = grid_shape.template pop_front<2>();
                auto n_blocks = product(blocks_to_fused);
                if constexpr (N == 3)
                    return Pair{Vec<u32, 3>{grid_shape[0], grid_shape[1], n_blocks}, Empty{}};
                else
                    return Pair{Vec<u32, 3>{grid_shape[0], grid_shape[1], n_blocks}, blocks_to_fused};
            } else if constexpr (N == 2)
                return Pair{Vec<u32, 3>{1, grid_shape[0], grid_shape[1]}, Empty{}};
            else // N == 1
                return Pair{Vec<u32, 3>{1, 1, grid_shape[0]}, Empty{}};
        }();

        auto config = LaunchConfig{
            .grid_shape = grid_shape_3d,
            .block_shape = block_shape<Block>(),
        };
        return noa::make_tuple(config, grid_shape.n_elements(), grid_shape_fused_in_x);
    }
}

namespace noa::gpu {
    template<bool ZipReduced = false,
             bool ZipOutput = false,
             u32 BlockSizeX = 512,
             u32 BlockSizeY = 1,
             u32 IndicesPerThreadX = 1,
             u32 IndicesPerThreadY = 1,
             u32 MaxGridSize = 4096,
             bool AllowBlockReshape = true,
             bool AllowTwoKernels = true>
    struct ReduceIwiseConfig {
        using interface = nd::ReduceIwiseInterface<ZipReduced, ZipOutput>;

        static constexpr u32 block_size = BlockSizeX * BlockSizeY;
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = BlockSizeY;
        static constexpr u32 indices_per_thread_x = IndicesPerThreadX;
        static constexpr u32 indices_per_thread_y = IndicesPerThreadY;
        static constexpr u32 max_grid_size = MaxGridSize;
        static constexpr bool allow_block_reshape = AllowBlockReshape;
        static constexpr bool allow_two_kernels = AllowTwoKernels;
    };

    template<typename Config = ReduceIwiseConfig<>,
             typename Op, typename Reduced, typename Output, typename Index, size_t N>
    requires (nt::tuple_of_accessor_value_or_empty<std::decay_t<Reduced>> and
              nt::tuple_of_accessor_pure_nd_or_empty<Output, 1>)
    NOA_NOINLINE void reduce_iwise(
        const Shape<Index, N>& shape,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        Stream& stream,
        usize scratch_size = 0
    ) {
        using Op_ = std::decay_t<Op>;
        using Reduced_ = std::decay_t<Reduced>;
        using Output_ = std::decay_t<Output>;
        using Interface = Config::interface;

        constexpr auto SMALL_THRESHOLD = static_cast<isize>(Config::block_size * 32);
        constexpr bool HAS_REDUCED = not nt::empty_tuple<Reduced>;
        const auto n_elements = shape.template as_safe<isize>().n_elements();
        const auto n_bytes_of_scratch =
            n_bytes_of_scratch_to_allocate_for_reduction<Reduced>(
                Config::block_size, scratch_size);

        if (not Config::allow_two_kernels or n_elements <= SMALL_THRESHOLD) {
            using Block = std::conditional_t<N == 1,
                details::ReduceIwise1dBlock<Config>,
                details::ReduceIwise2dBlock<Config>>;
            auto config = LaunchConfig{
                .grid_shape = Vec<u32, 3>{1, 1, 1},
                .block_shape = details::block_shape<Block>(),
                .n_bytes_of_scratch = n_bytes_of_scratch
            };
            using ReduceIwise = details::ReduceIwiseSmall<N, Block, Interface, Op_, Index, Reduced_, Output_>;
            auto kernel = ReduceIwise{NOA_FWD(op), NOA_FWD(reduced), output, shape};
            auto handle = typename ReduceIwise::handle_type(static_cast<u32>(scratch_size));
            stream.enqueue(config, handle, kernel);
        } else {
            using BlockFirst = std::conditional_t<N == 1,
                details::ReduceIwise1dBlock<Config>,
                details::ReduceIwise2dBlock<Config>>;
            const auto shape_iz = shape.template as_safe<isize>();
            auto [config_first, n_blocks, fused_grid_shape_in_x] = details::reduce_iwise_nd_first_config<BlockFirst>(shape_iz);
            config_first.n_bytes_of_scratch = n_bytes_of_scratch;

            using Buffer = AllocatorDevice::allocate_type<Reduced_>;
            Buffer joined{};
            Index n_joined{};
            if constexpr (HAS_REDUCED) {
                n_joined = static_cast<Index>(n_blocks);
                joined = AllocatorDevice::allocate_async<Reduced_>(static_cast<isize>(n_joined), stream);
            }

            // First kernel.
            using KernelFirst = details::ReduceIwiseFirstPassLarge<N, BlockFirst, Interface, Op_, Index, Reduced_>;
            auto kernel_first = KernelFirst{op, reduced, joined.get(), shape, fused_grid_shape_in_x};
            auto handle_first = typename KernelFirst::handle_type(static_cast<u32>(scratch_size));
            stream.enqueue(config_first, handle_first, kernel_first);

            // Second kernel.
            // TODO If there's no value to reduce, check that op has a valid post and if not skip the launch.
            constexpr bool IS_STATIC_SCRATCH_ENABLED = is_static_scratch_enabled();
            using BlockSecond = StaticBlock<HAS_REDUCED ? Config::block_size : 1, 1, 1>;
            using KernelSecond = details::ReduceIwiseSecondPassLarge<
                IS_STATIC_SCRATCH_ENABLED, BlockSecond, Interface, Op_, Index, Reduced_, Output_>;
            auto config_second = LaunchConfig{
                .grid_shape = Vec<u32, 3>{1, 1, 1},
                .block_shape = details::block_shape<BlockSecond>(),
                .n_bytes_of_scratch =
                    (HAS_REDUCED and not IS_STATIC_SCRATCH_ENABLED) ?
                    BlockSecond::block_size * sizeof(Reduced_) : 0,
            };
            auto handle_second = typename KernelSecond::handle_type{};
            auto kernel_second = KernelSecond{NOA_FWD(op), joined.get(), n_joined, NOA_FWD(reduced), output};
            stream.enqueue(config_second, handle_second, kernel_second);
        }
    }
}
