#pragma once
#include "noa/gpu/cuda/IncludeGuard.cuh"

#include "noa/core/Config.hpp"
#include "noa/core/Interfaces.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/gpu/cuda/Block.cuh"
#include "noa/gpu/cuda/Constants.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::guts {
    template<nt::integer T, typename Config>
    NOA_FHD auto global_indices_4d(const Vec<u32, 3>& grid_offset, u32 grid_size_x) {
        const auto bid = grid_offset.as<T>() + block_indices<T, 3>();
        const Vec<T, 2> bid_yx = ni::offset2index(bid[2], static_cast<T>(grid_size_x));
        auto gid = Vec{
            bid[0],
            static_cast<T>(Config::block_work_size_z) * bid[1],
            static_cast<T>(Config::block_work_size_y) * bid_yx[0],
            static_cast<T>(Config::block_work_size_x) * bid_yx[1],
        };
        if constexpr (Config::ndim == 3)
            gid[1] += static_cast<T>(threadIdx.z);
        if constexpr (Config::ndim >= 2)
            gid[2] += static_cast<T>(threadIdx.y);
        if constexpr (Config::ndim >= 1)
            gid[3] += static_cast<T>(threadIdx.x);
        return gid;
    }

    template<nt::integer T, typename Config>
    NOA_FHD auto global_indices_3d(const Vec<u32, 3>& grid_offset) {
        const auto bid = grid_offset.as<T>() + block_indices<T, 3>();
        auto gid = Vec{
            static_cast<T>(Config::block_work_size_z) * static_cast<T>(bid[0]),
            static_cast<T>(Config::block_work_size_y) * static_cast<T>(bid[1]),
            static_cast<T>(Config::block_work_size_x) * static_cast<T>(bid[2]),
        };
        if constexpr (Config::ndim == 3)
            gid[0] += static_cast<T>(threadIdx.z);
        if constexpr (Config::ndim >= 2)
            gid[1] += static_cast<T>(threadIdx.y);
        if constexpr (Config::ndim >= 1)
            gid[2] += static_cast<T>(threadIdx.x);
        return gid;
    }

    template<nt::integer T, typename Config>
    NOA_FHD auto global_indices_2d(const Vec<u32, 2>& grid_offset) {
        const auto bid = grid_offset.as<T>() + block_indices<T, 2>();
        auto gid = Vec{
            static_cast<T>(Config::block_work_size_y) * bid[0],
            static_cast<T>(Config::block_work_size_x) * bid[1],
        };
        if constexpr (Config::ndim >= 2)
            gid[0] += static_cast<T>(threadIdx.y);
        if constexpr (Config::ndim >= 1)
            gid[1] += static_cast<T>(threadIdx.x);
        return gid;
    }

    template<nt::integer T, typename Config>
    NOA_FHD auto global_indices_1d(const Vec<u32, 1>& grid_offset) {
        const auto bid = grid_offset.as<T>() + block_indices<T, 1>();
        return Vec{
            static_cast<T>(Config::block_work_size_x) * bid[0] + static_cast<T>(threadIdx.x),
        };
    }

    template<typename Config, typename Op, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void iwise_4d_static(Op op, Vec<Index, 3> shape, Vec<u32, 3> grid_offset, u32 grid_size_x) {
        auto bdhw = global_indices_4d<Index, Config>(grid_offset, grid_size_x);

        using interface = ng::IwiseInterface;
        interface::init(op, thread_uid<3>());

        for (Index d = 0; d < Config::n_elements_per_thread_z; ++d) {
            for (Index h = 0; h < Config::n_elements_per_thread_y; ++h) {
                for (Index w = 0; w < Config::n_elements_per_thread_x; ++w) {
                    const Index id = bdhw[1] + Config::block_size_z * d;
                    const Index ih = bdhw[2] + Config::block_size_y * h;
                    const Index iw = bdhw[3] + Config::block_size_x * w;
                    if (id < shape[0] and ih < shape[1] and iw < shape[2])
                        interface::call(op, bdhw[0], id, ih, iw);
                }
            }
        }
        interface::final(op, thread_uid<3>());
    }

    template<typename Config, typename Op, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void iwise_3d_static(Op op, Vec<Index, 3> shape, Vec<u32, 3> grid_offset) {
        auto dhw = global_indices_3d<Index, Config>(grid_offset);

        using interface = ng::IwiseInterface;
        interface::init(op, thread_uid<3>());

        for (Index d = 0; d < Config::n_elements_per_thread_z; ++d) {
            for (Index h = 0; h < Config::n_elements_per_thread_y; ++h) {
                for (Index w = 0; w < Config::n_elements_per_thread_x; ++w) {
                    const Index id = dhw[0] + Config::block_size_z * d;
                    const Index ih = dhw[1] + Config::block_size_y * h;
                    const Index iw = dhw[2] + Config::block_size_x * w;
                    if (id < shape[0] and ih < shape[1] and iw < shape[2])
                        interface::call(op, id, ih, iw);
                }
            }
        }
        interface::final(op, thread_uid<3>());
    }

    template<typename Config, typename Op, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void iwise_2d_static(Op op, Vec<Index, 2> shape, Vec<u32, 2> grid_offset) {
        auto hw = global_indices_2d<Index, Config>(grid_offset);

        using interface = ng::IwiseInterface;
        interface::init(op, thread_uid<2>());

        for (Index h = 0; h < Config::n_elements_per_thread_y; ++h) {
            for (Index w = 0; w < Config::n_elements_per_thread_x; ++w) {
                const Index ih = hw[0] + Config::block_size_y * h;
                const Index iw = hw[1] + Config::block_size_x * w;
                if (ih < shape[0] and iw < shape[1])
                    interface::call(op, ih, iw);
            }
        }
        interface::final(op, thread_uid<2>());
    }

    template<typename Config, typename Op, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void iwise_1d_static(Op op, Vec<Index, 1> shape, Vec<u32, 1> grid_offset) {
        auto index = global_indices_1d<Index, Config>(grid_offset);

        using interface = ng::IwiseInterface;
        interface::init(op, thread_uid<1>());

        for (Index w = 0; w < Config::n_elements_per_thread_x; ++w) {
            const Index iw = index[0] + Config::block_size_x * w;
            if (iw < shape[0])
                interface::call(op, iw);
        }
        interface::final(op, thread_uid<1>());
    }
}

namespace noa::cuda {
    template<u32 BlockSizeX, u32 BlockSizeY, u32 BlockSizeZ,
             u32 ElementsPerThreadX = 1, u32 ElementsPerThreadY = 1, u32 ElementsPerThreadZ = 1>
    struct IwiseConfig {
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = BlockSizeY;
        static constexpr u32 block_size_z = BlockSizeZ;
        static constexpr u32 block_size = block_size_x * block_size_y * block_size_z;
        static constexpr u32 ndim = block_size_z > 1 ? 3 : block_size_y > 1 ? 2 : 1;

        static_assert(block_size > 0 and block_size < Limits::MAX_THREADS);

        static constexpr u32 n_elements_per_thread_x = ElementsPerThreadX;
        static constexpr u32 n_elements_per_thread_y = ElementsPerThreadY;
        static constexpr u32 n_elements_per_thread_z = ElementsPerThreadZ;

        static constexpr u32 block_work_size_x = block_size_x * n_elements_per_thread_x;
        static constexpr u32 block_work_size_y = block_size_y * n_elements_per_thread_y;
        static constexpr u32 block_work_size_z = block_size_z * n_elements_per_thread_z;
    };

    template<size_t N>
    using IwiseConfigDefault = std::conditional_t<
        N == 1,
        IwiseConfig<Constant::WARP_SIZE * 8, 1, 1>,
        IwiseConfig<Constant::WARP_SIZE, 256 / Constant::WARP_SIZE, 1>>;

    template<size_t N, typename Config = IwiseConfigDefault<N>, typename Index, typename Op>
    FMT_NOINLINE void iwise(
        const Shape<Index, N>& shape,
        Op&& op,
        Stream& stream,
        size_t n_bytes_of_shared_memory = 0
    ) {
        static_assert(N >= Config::ndim);
        if constexpr (N == 4) {
            auto grid_x = GridXY(shape[3], shape[2], Config::block_work_size_x, Config::block_work_size_y);
            auto grid_y = GridY(shape[1], Config::block_work_size_z);
            auto grid_z = GridZ(shape[0], 1); // we also assume it's 1 in the kernel by not checking the batch size
            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    for (u32 x{}; x < grid_x.n_launches(); ++x) {
                        const auto launch_config = LaunchConfig{
                            .n_blocks = dim3(grid_x.n_blocks(x), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                            .n_threads = dim3(Config::block_size_x, Config::block_size_y, Config::block_size_z),
                            .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                        };
                        const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y), grid_x.offset(x)};
                        stream.enqueue(
                            guts::iwise_4d_static<Config, std::decay_t<Op>, Index>,
                            launch_config, op, shape.vec.pop_front(), grid_offset, grid_x.n_blocks_x()
                        );
                    }
                }
            }
        } else if constexpr (N == 3) {
            auto grid_x = GridX(shape[2], Config::block_work_size_x);
            auto grid_y = GridY(shape[1], Config::block_work_size_y);
            auto grid_z = GridZ(shape[0], Config::block_work_size_z);
            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    for (u32 x{}; x < grid_x.n_launches(); ++x) {
                        const auto launch_config = LaunchConfig{
                            .n_blocks = dim3(grid_x.n_blocks(x), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                            .n_threads = dim3(Config::block_size_x, Config::block_size_y, Config::block_size_z),
                            .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                        };
                        const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y), grid_x.offset(x)};
                        stream.enqueue(
                            guts::iwise_3d_static<Config, std::decay_t<Op>, Index>,
                            launch_config, op, shape.vec, grid_offset
                        );
                    }
                }
            }
        } else if constexpr (N == 2) {
            auto grid_x = GridX(shape[1], Config::block_work_size_x);
            auto grid_y = GridY(shape[0], Config::block_work_size_y);
            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                for (u32 x{}; x < grid_x.n_launches(); ++x) {
                    const auto launch_config = LaunchConfig{
                        .n_blocks = dim3(grid_x.n_blocks(x), grid_y.n_blocks(y)),
                        .n_threads = dim3(Config::block_size_x, Config::block_size_y),
                        .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                    };
                    const auto grid_offset = Vec{grid_y.offset(y), grid_x.offset(x)};
                    stream.enqueue(
                        guts::iwise_2d_static<Config, std::decay_t<Op>, Index>,
                        launch_config, op, shape.vec, grid_offset
                    );
                }
            }
        } else if constexpr (N == 1) {
            auto grid_x = GridX(shape[0], Config::block_work_size_x);
            for (u32 x{}; x < grid_x.n_launches(); ++x) {
                const auto launch_config = LaunchConfig{
                    .n_blocks = dim3(grid_x.n_blocks(x)),
                    .n_threads = dim3(Config::block_size_x),
                    .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                };
                const auto grid_offset = Vec{grid_x.offset(x)};
                stream.enqueue(
                    guts::iwise_1d_static<Config, std::decay_t<Op>, Index>,
                    launch_config, op, shape.vec, grid_offset
                );
            }
        }
    }
}
