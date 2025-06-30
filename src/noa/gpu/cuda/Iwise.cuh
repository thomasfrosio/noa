#pragma once
#include "noa/gpu/cuda/IncludeGuard.cuh"

#include "noa/core/Interfaces.hpp"
#include "noa/gpu/cuda/Block.cuh"
#include "noa/gpu/cuda/Constants.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::guts {
    template<typename Block, typename Interface, typename Op, typename Index>
    __global__ __launch_bounds__(Block::block_size)
    void iwise_4d_static(Op op, Vec<Index, 3> shape, Vec<u32, 2> block_offset_zy, u32 n_blocks_x) {
        auto gid = global_indices_4d<Index, Block>(n_blocks_x, block_offset_zy);

        Interface::init(op, thread_uid<3>());

        for (Index d = 0; d < Block::n_elements_per_thread_z; ++d) {
            for (Index h = 0; h < Block::n_elements_per_thread_y; ++h) {
                for (Index w = 0; w < Block::n_elements_per_thread_x; ++w) {
                    const Index id = gid[1] + Block::block_size_z * d;
                    const Index ih = gid[2] + Block::block_size_y * h;
                    const Index iw = gid[3] + Block::block_size_x * w;
                    if (id < shape[0] and ih < shape[1] and iw < shape[2])
                        Interface::call(op, gid[0], id, ih, iw);
                }
            }
        }
        Interface::final(op, thread_uid<3>());
    }

    template<typename Block, typename Interface, typename Op, typename Index>
    __global__ __launch_bounds__(Block::block_size)
    void iwise_3d_static(Op op, Vec<Index, 3> shape, Vec<u32, 2> block_offset_y) {
        auto gid = global_indices_3d<Index, Block>(block_offset_y);

        Interface::init(op, thread_uid<3>());

        for (Index d = 0; d < Block::n_elements_per_thread_z; ++d) {
            for (Index h = 0; h < Block::n_elements_per_thread_y; ++h) {
                for (Index w = 0; w < Block::n_elements_per_thread_x; ++w) {
                    const Index id = gid[0] + Block::block_size_z * d;
                    const Index ih = gid[1] + Block::block_size_y * h;
                    const Index iw = gid[2] + Block::block_size_x * w;
                    if (id < shape[0] and ih < shape[1] and iw < shape[2])
                        Interface::call(op, id, ih, iw);
                }
            }
        }
        Interface::final(op, thread_uid<3>());
    }

    template<typename Block, typename Interface, typename Op, typename Index>
    __global__ __launch_bounds__(Block::block_size)
    void iwise_2d_static(Op op, Vec<Index, 2> shape, Vec<u32, 1> block_offset_y) {
        auto gid = global_indices_2d<Index, Block>(block_offset_y);

        Interface::init(op, thread_uid<2>());

        for (Index h = 0; h < Block::n_elements_per_thread_y; ++h) {
            for (Index w = 0; w < Block::n_elements_per_thread_x; ++w) {
                const Index ih = gid[0] + Block::block_size_y * h;
                const Index iw = gid[1] + Block::block_size_x * w;
                if (ih < shape[0] and iw < shape[1])
                    Interface::call(op, ih, iw);
            }
        }
        Interface::final(op, thread_uid<2>());
    }

    template<typename Block, typename Interface, typename Op, typename Index>
    __global__ __launch_bounds__(Block::block_size)
    void iwise_1d_static(Op op, Vec<Index, 1> shape) {
        auto gid = global_indices_1d<Index, Block>();

        Interface::init(op, thread_uid<1>());

        for (Index w = 0; w < Block::n_elements_per_thread_x; ++w) {
            const Index iw = gid[0] + Block::block_size_x * w;
            if (iw < shape[0])
                Interface::call(op, iw);
        }
        Interface::final(op, thread_uid<1>());
    }
}

namespace noa::cuda {
    template<size_t N>
    using IwiseConfig = std::conditional_t<
        N == 1,
        StaticBlock<Constant::WARP_SIZE * 8, 1, 1>,
        StaticBlock<Constant::WARP_SIZE, 256 / Constant::WARP_SIZE, 1>>;

    template<size_t N, typename Config = IwiseConfig<N>, typename Index, typename Op>
    NOA_NOINLINE void iwise(
        const Shape<Index, N>& shape,
        Op&& op,
        Stream& stream,
        size_t n_bytes_of_shared_memory = 0
    ) {
        static_assert(N >= Config::ndim);
        using Block = Config;
        using Interface = ng::IwiseInterface;

        if constexpr (N == 4) {
            auto grid_x = GridXY(shape[3], shape[2], Block::block_work_size_x, Block::block_work_size_y);
            auto grid_y = GridY(shape[1], Block::block_work_size_z);
            auto grid_z = GridZ(shape[0], 1);
            check(grid_x.n_launches() == 1);
            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    const auto config = LaunchConfig{
                        .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                        .n_threads = dim3(Block::block_size_x, Block::block_size_y, Block::block_size_z),
                        .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                    };
                    const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y)};
                    stream.enqueue(
                        guts::iwise_4d_static<Block, Interface, std::decay_t<Op>, Index>,
                        config, op, shape.vec.pop_front(), grid_offset, grid_x.n_blocks_x()
                    );
                }
            }
        } else if constexpr (N == 3) {
            auto grid_x = GridX(shape[2], Block::block_work_size_x);
            auto grid_y = GridY(shape[1], Block::block_work_size_y);
            auto grid_z = GridZ(shape[0], Block::block_work_size_z);
            check(grid_x.n_launches() == 1);
            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    const auto config = LaunchConfig{
                        .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                        .n_threads = dim3(Block::block_size_x, Block::block_size_y, Block::block_size_z),
                        .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                    };
                    const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y)};
                    stream.enqueue(
                        guts::iwise_3d_static<Block, Interface, std::decay_t<Op>, Index>,
                        config, op, shape.vec, grid_offset
                    );
                }
            }
        } else if constexpr (N == 2) {
            auto grid_x = GridX(shape[1], Block::block_work_size_x);
            auto grid_y = GridY(shape[0], Block::block_work_size_y);
            check(grid_x.n_launches() == 1);
            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                const auto config = LaunchConfig{
                    .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y)),
                    .n_threads = dim3(Block::block_size_x, Block::block_size_y),
                    .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                };
                const auto grid_offset = Vec{grid_y.offset(y)};
                stream.enqueue(
                    guts::iwise_2d_static<Block, Interface, std::decay_t<Op>, Index>,
                    config, op, shape.vec, grid_offset
                );
            }
        } else if constexpr (N == 1) {
            auto grid_x = GridX(shape[0], Block::block_work_size_x);
            check(grid_x.n_launches() == 1);
            const auto config = LaunchConfig{
                .n_blocks = dim3(grid_x.n_blocks(0)),
                .n_threads = dim3(Block::block_size_x),
                .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
            };
            stream.enqueue(
                guts::iwise_1d_static<Block, Interface, std::decay_t<Op>, Index>,
                config, op, shape.vec
            );
        }
    }
}
