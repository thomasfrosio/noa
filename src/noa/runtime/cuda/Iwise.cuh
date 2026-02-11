#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/cuda/Block.cuh"
#include "noa/runtime/cuda/Constants.hpp"
#include "noa/runtime/cuda/Stream.hpp"
#include "noa/runtime/cuda/ComputeHandle.cuh"

namespace noa::cuda::details {
    template<typename Block, typename Interface, typename Op, typename Index>
    __global__ __launch_bounds__(Block::block_size)
    void iwise_4d_static(Op op, Vec<Index, 3> shape, Vec<u32, 2> block_offset_zy, u32 n_blocks_x) {
        const auto ci = ComputeHandle<Index, 3, Block::block_ndim>{};
        Interface::init(ci, op);

        const auto gid = global_indices_4d<Index, Block>(n_blocks_x, block_offset_zy);
        for (Index d = 0; d < Block::n_elements_per_thread_z; ++d) {
            for (Index h = 0; h < Block::n_elements_per_thread_y; ++h) {
                for (Index w = 0; w < Block::n_elements_per_thread_x; ++w) {
                    const Index id = gid[1] + Block::block_size_z * d;
                    const Index ih = gid[2] + Block::block_size_y * h;
                    const Index iw = gid[3] + Block::block_size_x * w;
                    if (id < shape[0] and ih < shape[1] and iw < shape[2])
                        Interface::call(ci, op, gid[0], id, ih, iw);
                }
            }
        }
        Interface::deinit(ci, op);
    }

    template<typename Block, typename Interface, typename Op, typename Index>
    __global__ __launch_bounds__(Block::block_size)
    void iwise_3d_static(Op op, Vec<Index, 3> shape, Vec<u32, 2> block_offset_y) {
        const auto ci = ComputeHandle<Index, 3, Block::block_ndim>{};
        Interface::init(ci, op);

        const auto gid = global_indices_3d<Index, Block>(block_offset_y);
        for (Index d = 0; d < Block::n_elements_per_thread_z; ++d) {
            for (Index h = 0; h < Block::n_elements_per_thread_y; ++h) {
                for (Index w = 0; w < Block::n_elements_per_thread_x; ++w) {
                    const Index id = gid[0] + Block::block_size_z * d;
                    const Index ih = gid[1] + Block::block_size_y * h;
                    const Index iw = gid[2] + Block::block_size_x * w;
                    if (id < shape[0] and ih < shape[1] and iw < shape[2])
                        Interface::call(ci, op, id, ih, iw);
                }
            }
        }
        Interface::deinit(ci, op);
    }

    template<typename Block, typename Interface, typename Op, typename Index>
    __global__ __launch_bounds__(Block::block_size)
    void iwise_2d_static(Op op, Vec<Index, 2> shape, Vec<u32, 1> block_offset_y) {
        const auto ci = ComputeHandle<Index, 2, Block::block_ndim>{};
        Interface::init(ci, op);

        const auto gid = global_indices_2d<Index, Block>(block_offset_y);
        for (Index h = 0; h < Block::n_elements_per_thread_y; ++h) {
            for (Index w = 0; w < Block::n_elements_per_thread_x; ++w) {
                const Index ih = gid[0] + Block::block_size_y * h;
                const Index iw = gid[1] + Block::block_size_x * w;
                if (ih < shape[0] and iw < shape[1])
                    Interface::call(ci, op, ih, iw);
            }
        }
        Interface::deinit(ci, op);
    }

    template<typename Block, typename Interface, typename Op, typename Index>
    __global__ __launch_bounds__(Block::block_size)
    void iwise_1d_static(Op op, Vec<Index, 1> shape) {
        const auto ci = ComputeHandle<Index, 1, Block::block_ndim>{};
        Interface::init(ci, op);

        const auto gid = global_indices_1d<Index, Block>();
        for (Index w = 0; w < Block::n_elements_per_thread_x; ++w) {
            const Index iw = gid[0] + Block::block_size_x * w;
            if (iw < shape[0])
                Interface::call(ci, op, iw);
        }
        Interface::deinit(ci, op);
    }
}

namespace noa::cuda {
    template<usize N>
    using IwiseConfig = std::conditional_t<
        N == 1,
        StaticBlock<Constant::WARP_SIZE * 8, 1, 1>,
        StaticBlock<Constant::WARP_SIZE, 256 / Constant::WARP_SIZE, 1>>;

    template<usize N, typename Config = IwiseConfig<N>, typename Index, typename Op>
    NOA_NOINLINE void iwise(
        const Shape<Index, N>& shape,
        Op&& op,
        Stream& stream,
        usize scratch_size = 0
    ) {
        static_assert(N >= Config::block_ndim);
        using Interface = nd::IwiseInterface;

        if constexpr (N == 4) {
            auto grid_x = GridXY(shape[3], shape[2], Config::block_work_size_x, Config::block_work_size_y);
            auto grid_y = GridY(shape[1], Config::block_work_size_z);
            auto grid_z = GridZ(shape[0], 1);
            check(grid_x.n_launches() == 1);
            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    const auto config = LaunchConfig{
                        .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                        .n_threads = dim3(Config::block_size_x, Config::block_size_y, Config::block_size_z),
                        .n_bytes_of_shared_memory = scratch_size,
                    };
                    const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y)};
                    stream.enqueue(
                        details::iwise_4d_static<Config, Interface, std::decay_t<Op>, Index>,
                        config, op, shape.vec.pop_front(), grid_offset, grid_x.n_blocks_x()
                    );
                }
            }
        } else if constexpr (N == 3) {
            auto grid_x = GridX(shape[2], Config::block_work_size_x);
            auto grid_y = GridY(shape[1], Config::block_work_size_y);
            auto grid_z = GridZ(shape[0], Config::block_work_size_z);
            check(grid_x.n_launches() == 1);
            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    const auto config = LaunchConfig{
                        .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                        .n_threads = dim3(Config::block_size_x, Config::block_size_y, Config::block_size_z),
                        .n_bytes_of_shared_memory = scratch_size,
                    };
                    const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y)};
                    stream.enqueue(
                        details::iwise_3d_static<Config, Interface, std::decay_t<Op>, Index>,
                        config, op, shape.vec, grid_offset
                    );
                }
            }
        } else if constexpr (N == 2) {
            auto grid_x = GridX(shape[1], Config::block_work_size_x);
            auto grid_y = GridY(shape[0], Config::block_work_size_y);
            check(grid_x.n_launches() == 1);
            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                const auto config = LaunchConfig{
                    .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y)),
                    .n_threads = dim3(Config::block_size_x, Config::block_size_y),
                    .n_bytes_of_shared_memory = scratch_size,
                };
                const auto grid_offset = Vec{grid_y.offset(y)};
                stream.enqueue(
                    details::iwise_2d_static<Config, Interface, std::decay_t<Op>, Index>,
                    config, op, shape.vec, grid_offset
                );
            }
        } else if constexpr (N == 1) {
            auto grid_x = GridX(shape[0], Config::block_work_size_x);
            check(grid_x.n_launches() == 1);
            const auto config = LaunchConfig{
                .n_blocks = dim3(grid_x.n_blocks(0)),
                .n_threads = dim3(Config::block_size_x),
                .n_bytes_of_shared_memory = scratch_size,
            };
            stream.enqueue(
                details::iwise_1d_static<Config, Interface, std::decay_t<Op>, Index>,
                config, op, shape.vec
            );
        }
    }
}
