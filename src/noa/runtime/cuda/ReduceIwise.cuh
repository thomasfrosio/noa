#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Block.cuh"
#include "noa/runtime/cuda/ComputeHandle.cuh"

namespace noa::cuda::details {
    template<typename Config>
    struct ReduceIwise2dBlock {
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = max(Constant::WARP_SIZE, Config::block_size);
        static constexpr u32 block_size_x = Constant::WARP_SIZE;
        static constexpr u32 block_size_y = block_size / block_size_x;
        static constexpr u32 block_ndim = 2;
    };

    template<typename Config>
    struct ReduceIwise1dBlock {
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = Config::block_size;
        static constexpr u32 block_ndim = 1;
    };
}

namespace noa::cuda::details {
    template<typename Config, typename Interface, typename Op, typename Index, typename Reduced>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_4d_first(
        Op op,
        Reduced reduced,
        Reduced* joined,
        Vec<Index, 4> shape,
        Vec<u32, 2> n_blocks_hw,
        u32 scratch_size
    ) {
        const auto ci = ComputeHandle<Index, 3, Config::block_ndim, true, true>(scratch_size);
        Interface::init(ci, op, Index{});

        // Get the position within the 4d span.
        const auto index = offset2index(blockIdx.x, n_blocks_hw[1]);
        const auto gid = Vec<Index, 4>::from_values(
            blockIdx.z,
            blockIdx.y,
            Config::block_size_y * index[0] + threadIdx.y,
            Config::block_size_x * index[1] + threadIdx.x
        );

        // Traverse the entire 4d span.
        for (Index cb = gid[0]; cb < shape[0]; cb += gridDim.z)
            for (Index cd = gid[1]; cd < shape[1]; cd += gridDim.y)
                for (Index ch = gid[2]; ch < shape[2]; ch += Config::block_size_y * n_blocks_hw[0])
                    for (Index cw = gid[3]; cw < shape[3]; cw += Config::block_size_x * n_blocks_hw[1])
                        Interface::call(ci, op, reduced, cb, cd, ch, cw);

        Interface::deinit(ci, op, Index{});

        // Reduce to one value per block.
        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        const Index bid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        block_join<Interface, Config::block_size, false>(op, reduced, joined, tid, bid);
    }

    template<typename Config, typename Interface, typename Op, typename Index, typename Reduced>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_3d_first(
        Op op,
        Reduced reduced,
        Reduced* joined,
        Vec<Index, 3> shape,
        u32 scratch_size
    ) {
        const auto ci = ComputeHandle<Index, 3, Config::block_ndim, true, true>(scratch_size);
        Interface::init(ci, op, Index{});

        const auto gid = Vec<Index, 3>::from_values(
            blockIdx.z,
            Config::block_size_y * blockIdx.y + threadIdx.y,
            Config::block_size_x * blockIdx.x + threadIdx.x
        );

        for (Index cd = gid[0]; cd < shape[0]; cd += gridDim.z)
            for (Index ch = gid[1]; ch < shape[1]; ch += Config::block_size_y * gridDim.y)
                for (Index cw = gid[2]; cw < shape[2]; cw += Config::block_size_x * gridDim.x)
                    Interface::call(ci, op, reduced, cd, ch, cw);

        Interface::deinit(ci, op, Index{});

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        const Index bid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        block_join<Interface, Config::block_size, false>(op, reduced, joined, tid, bid);
    }

    template<typename Config, typename Interface, typename Op, typename Index, typename Reduced>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_2d_first(
        Op op,
        Reduced reduced,
        Reduced* joined,
        Vec<Index, 2> shape,
        u32 scratch_size
    ) {
        const auto ci = ComputeHandle<Index, 2, Config::block_ndim, true, true>(scratch_size);
        Interface::init(ci, op, Index{});

        const auto gid = Vec<Index, 2>::from_values(
            Config::block_size_y * blockIdx.y + threadIdx.y,
            Config::block_size_x * blockIdx.x + threadIdx.x
        );

        for (Index ch = gid[0]; ch < shape[0]; ch += Config::block_size_y * gridDim.y)
            for (Index cw = gid[1]; cw < shape[1]; cw += Config::block_size_x * gridDim.x)
                Interface::call(ci, op, reduced, ch, cw);

        Interface::deinit(ci, op, Index{});

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        const Index bid = blockIdx.y * gridDim.x + blockIdx.x;
        block_join<Interface, Config::block_size, false>(op, reduced, joined, tid, bid);
    }

    template<typename Config, typename Interface, typename Op, typename Index, typename Reduced>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_1d_first(
        Op op,
        Reduced reduced,
        Reduced* joined,
        Vec<Index, 1> shape,
        u32 scratch_size
    ) {
        const auto ci = ComputeHandle<Index, 1, 1, true, true>(scratch_size);
        Interface::init(ci, op, Index{});

        const Index tid = threadIdx.x;
        const Index bid = blockIdx.x;
        const auto gid = Vec<Index, 1>::from_values(Config::block_size * bid + tid);

        for (Index cw = gid[0]; cw < shape[0]; cw += Config::block_size * gridDim.x)
            Interface::call(ci, op, reduced, cw);

        Interface::deinit(ci, op, Index{});

        block_join<Interface, Config::block_size, false>(op, reduced, joined, tid, bid);
    }

    template<typename Config, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_second(
        Op op,
        Reduced* to_join,
        Index n_to_join,
        Reduced reduced,
        Output output
    ) {
        const Index tid = threadIdx.x;
        if constexpr (not nt::empty_tuple<Reduced>) {
            for (Index cid = tid; cid < n_to_join; cid += Config::block_size)
                Interface::join(op, to_join[cid], reduced);
        }
        block_join_and_post<Interface, Config::block_size, true>(op, reduced, output, tid, Index{});
    }

    template<typename Config, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_4d_small(Op op, Reduced reduced, Output output, Vec<Index, 4> shape, u32 scratch_size) {
        const auto ci = ComputeHandle<Index, 1, Config::block_ndim, true, false>(scratch_size);
        Interface::init(ci, op, Index{});

        const auto gid = Vec<Index, 4>::from_values(0, 0, threadIdx.y, threadIdx.x);
        for (Index i = gid[0]; i < shape[0]; ++i)
            for (Index j = gid[1]; j < shape[1]; ++j)
                for (Index k = gid[2]; k < shape[2]; k += Config::block_size_y)
                    for (Index l = gid[3]; l < shape[3]; l += Config::block_size_x)
                        Interface::call(ci, op, reduced, i, j, k, l);

        Interface::deinit(ci, op, Index{});

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        block_join_and_post<Interface, Config::block_size, false>(op, reduced, output, tid, Index{});
    }

    template<typename Config, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_3d_small(Op op, Reduced reduced, Output output, Vec<Index, 3> shape, u32 scratch_size) {
        const auto ci = ComputeHandle<Index, 1, Config::block_ndim, true, false>(scratch_size);
        Interface::init(ci, op, Index{});

        const auto gid = Vec<Index, 3>::from_values(0, threadIdx.y, threadIdx.x);
        for (Index i = gid[0]; i < shape[0]; ++i)
            for (Index j = gid[1]; j < shape[1]; j += Config::block_size_y)
                for (Index k = gid[2]; k < shape[2]; k += Config::block_size_x)
                    Interface::call(ci, op, reduced, i, j, k);

        Interface::deinit(ci, op, Index{});

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        block_join_and_post<Interface, Config::block_size, false>(op, reduced, output, tid, Index{});
    }

    template<typename Config, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_2d_small(Op op, Reduced reduced, Output output, Vec<Index, 2> shape, u32 scratch_size) {
        const auto ci = ComputeHandle<Index, 1, Config::block_ndim, true, false>(scratch_size);
        Interface::init(ci, op, Index{});

        const auto gid = Vec<Index, 2>::from_values(threadIdx.y, threadIdx.x);
        for (Index i = gid[0]; i < shape[0]; i += Config::block_size_y)
            for (Index j = gid[1]; j < shape[1]; j += Config::block_size_x)
                Interface::call(ci, op, reduced, i, j);

        Interface::deinit(ci, op, Index{});

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        block_join_and_post<Interface, Config::block_size, false>(op, reduced, output, tid, Index{});
    }

    template<typename Config, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_1d_small(Op op, Reduced reduced, Output output, Vec<Index, 1> shape, u32 scratch_size) {
        const auto ci = ComputeHandle<Index, 1, Config::block_ndim, true, false>(scratch_size);
        Interface::init(ci, op, Index{});

        const auto gid = Vec<Index, 1>::from_values(threadIdx.x);
        for (Index i = gid[0]; i < shape[0]; i += Config::block_size)
            Interface::call(ci, op, reduced, i);

        Interface::deinit(ci, op, Index{});

        block_join_and_post<Interface, Config::block_size, false>(op, reduced, output, gid[0], Index{});
    }
}

namespace noa::cuda::details {
    template<typename Block, usize N>
    auto reduce_iwise_nd_first_config(const Shape<isize, N>& shape) {
        // Note that while we could increase the grid size when the reduced tuple is empty, we keep it as specified
        // by the caller. The default value is large enough, and it's only for large reductions that increasing
        // would help. In this rare case, it's up to the user, for now at least.
        constexpr auto max_grid_size = static_cast<isize>(Block::max_grid_size);
        constexpr auto block_size = [] {
            if constexpr (N > 1) // 2d blocks
                return Vec<isize, 4>::from_values(1, 1, Block::block_size_y, Block::block_size_x);
            else // 1d blocks
                return Vec<isize, 4>::from_values(1, 1, 1, Block::block_size);
        }();

        // Set the number of blocks, while keep the total number of blocks within the maximum allowed.
        auto n_blocks = Vec<isize, 4>::from_value(1);
        const auto shape_whdb = shape.flip();
        for (isize i = 0; i <  static_cast<isize>(N); ++i) {
            const auto n_blocks_allowed = max_grid_size / product(n_blocks);
            n_blocks[3 - i] = min(divide_up(shape_whdb[i], block_size[3 - i]), n_blocks_allowed);
        }

        const auto n_blocks_u32 = n_blocks.as<u32>();
        const auto n_threads = dim3(block_size[3], block_size[2], 1);
        const auto n_blocks_dim3 = [&] {
            if constexpr (N == 4)
                return dim3(n_blocks_u32[3] * n_blocks_u32[2], n_blocks_u32[1], n_blocks_u32[0]);
            else
                return dim3(n_blocks_u32[3], n_blocks_u32[2], n_blocks_u32[1]);
        }();

        auto config = LaunchConfig{.n_blocks=n_blocks_dim3, .n_threads=n_threads};
        auto n_blocks_yx = Vec{n_blocks_u32[2], n_blocks_u32[3]};
        return make_tuple(config, product(n_blocks_u32), n_blocks_yx);
    }
}

namespace noa::cuda {
    template<bool ZipReduced = false,
             bool ZipOutput = false,
             u32 BlockSize = 512,
             u32 MaxGridSize = 4096,
             bool AllowTwoKernels = true>
    struct ReduceIwiseConfig {
        static_assert(is_multiple_of(BlockSize, Constant::WARP_SIZE) and BlockSize <= Limits::MAX_THREADS);
        using interface = nd::ReduceIwiseInterface<ZipReduced, ZipOutput>;
        static constexpr u32 block_size = BlockSize;
        static constexpr u32 max_grid_size = MaxGridSize;
        static constexpr bool allow_two_kernels = AllowTwoKernels;
    };

    template<typename Config = ReduceIwiseConfig<>,
             typename Op, typename Reduced, typename Output, typename Index, size_t N>
    requires (nt::tuple_of_accessor_value<std::decay_t<Reduced>> and
              nt::tuple_of_accessor_pure_nd_or_empty<Output, 1>)
    NOA_NOINLINE void reduce_iwise(
        const Shape<Index, N>& shape,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        Stream& stream,
        usize scratch_size = 0
    ) {
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using OutputDecay = std::decay_t<Output>;
        using Interface = Config::interface;

        constexpr auto SMALL_THRESHOLD = static_cast<isize>(Config::block_size * 32);
        constexpr bool HAS_REDUCED = not nt::empty_tuple<Reduced>;
        const auto n_elements = shape.template as_safe<isize>().n_elements();
        const auto n_bytes_of_shared_memory =
            n_bytes_of_shared_memory_to_allocate_for_reduction<Reduced>(
                Config::block_size, scratch_size);

        if (not Config::allow_two_kernels or n_elements <= SMALL_THRESHOLD) {
            auto config = LaunchConfig{.n_blocks = 1, .n_bytes_of_shared_memory = n_bytes_of_shared_memory};
            if constexpr (N == 1) {
                using Block = details::ReduceIwise1dBlock<Config>;
                config.n_threads = dim3(Block::block_size);
                stream.enqueue(
                    details::reduce_iwise_1d_small<Block, Interface, OpDecay, Index, ReducedDecay, OutputDecay>,
                    config, std::forward<Op>(op), std::forward<Reduced>(reduced), output, shape.vec,
                    static_cast<u32>(scratch_size)
                );
            } else if constexpr (N == 2) {
                using Block = details::ReduceIwise2dBlock<Config>;
                config.n_threads = dim3(Block::block_size_x, Block::block_size_y);
                stream.enqueue(
                    details::reduce_iwise_2d_small<Block, Interface, OpDecay, Index, ReducedDecay, OutputDecay>,
                    config, std::forward<Op>(op), std::forward<Reduced>(reduced), output, shape.vec,
                    static_cast<u32>(scratch_size)
                );
            } else if constexpr (N == 3) {
                using Block = details::ReduceIwise2dBlock<Config>;
                config.n_threads = dim3(Block::block_size_x, Block::block_size_y);
                stream.enqueue(
                    details::reduce_iwise_3d_small<Block, Interface, OpDecay, Index, ReducedDecay, OutputDecay>,
                    config, std::forward<Op>(op), std::forward<Reduced>(reduced), output, shape.vec,
                    static_cast<u32>(scratch_size)
                );
            } else if constexpr (N == 4) {
                using Block = details::ReduceIwise2dBlock<Config>;
                config.n_threads = dim3(Block::block_size_x, Block::block_size_y);
                stream.enqueue(
                    details::reduce_iwise_4d_small<Block, Interface, OpDecay, Index, ReducedDecay, OutputDecay>,
                    config, std::forward<Op>(op), std::forward<Reduced>(reduced), output, shape.vec,
                    static_cast<u32>(scratch_size)
                );
            } else {
                static_assert(nt::always_false<Op>);
            }
        } else {
            const auto shape_iz = shape.template as_safe<isize>();
            using Buffer = AllocatorDevice::allocate_type<ReducedDecay>;
            Buffer joined{};
            Index n_joined{};
            auto allocate_joined = [&](u32 n) {
                if constexpr (HAS_REDUCED) {
                    n_joined = static_cast<Index>(n);
                    joined = AllocatorDevice::allocate_async<ReducedDecay>(static_cast<isize>(n_joined), stream);
                }
            };

            // First kernel.
            if constexpr (N == 1) {
                using Block = details::ReduceIwise1dBlock<Config>;
                auto [config, n_blocks, _] = details::reduce_iwise_nd_first_config<Block>(shape_iz);
                config.n_bytes_of_shared_memory = n_bytes_of_shared_memory;
                allocate_joined(n_blocks);
                stream.enqueue(
                    details::reduce_iwise_1d_first<Block, Interface, OpDecay, Index, ReducedDecay>,
                    config, op, reduced, joined.get(), shape.vec, static_cast<u32>(scratch_size)
                );
            } else if constexpr (N == 2) {
                using Block = details::ReduceIwise2dBlock<Config>;
                auto [config, n_blocks, _] = details::reduce_iwise_nd_first_config<Block>(shape_iz);
                config.n_bytes_of_shared_memory = n_bytes_of_shared_memory;
                allocate_joined(n_blocks);
                stream.enqueue(
                    details::reduce_iwise_2d_first<Block, Interface, OpDecay, Index, ReducedDecay>,
                    config, op, reduced, joined.get(), shape.vec, static_cast<u32>(scratch_size)
                );
            } else if constexpr (N == 3) {
                using Block = details::ReduceIwise2dBlock<Config>;
                auto [config, n_blocks, _] = details::reduce_iwise_nd_first_config<Block>(shape_iz);
                config.n_bytes_of_shared_memory = n_bytes_of_shared_memory;
                allocate_joined(n_blocks);
                stream.enqueue(
                    details::reduce_iwise_3d_first<Block, Interface, OpDecay, Index, ReducedDecay>,
                    config, op, reduced, joined.get(), shape.vec, static_cast<u32>(scratch_size)
                );
            } else if constexpr (N == 4) {
                using Block = details::ReduceIwise2dBlock<Config>;
                auto [config, n_blocks, n_blocks_hw] = details::reduce_iwise_nd_first_config<Block>(shape_iz);
                config.n_bytes_of_shared_memory = n_bytes_of_shared_memory;
                allocate_joined(n_blocks);
                stream.enqueue(
                    details::reduce_iwise_4d_first<Block, Interface, OpDecay, Index, ReducedDecay>,
                    config, op, reduced, joined.get(), shape.vec, n_blocks_hw, static_cast<u32>(scratch_size)
                );
            } else {
                static_assert(nt::always_false<Op>);
            }

            // Second kernel.
            // TODO If there's no value to reduce, check that op has a valid post and if not skip the launch.
            using Block = Config;
            stream.enqueue(
                details::reduce_iwise_second<Block, Interface, OpDecay, Index, ReducedDecay, OutputDecay>,
                LaunchConfig{.n_blocks = 1, .n_threads = HAS_REDUCED ? Config::block_size : 1},
                std::forward<Op>(op), joined.get(), n_joined, std::forward<Reduced>(reduced), output
            );
        }
    }
}
