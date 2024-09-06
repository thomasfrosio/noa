#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Interfaces.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/gpu/cuda/Constants.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/kernels/Block.cuh"

namespace noa::cuda::guts {
    template<typename Config, typename Op, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void iwise_4d_static(Op op, Vec2<Index> end_hw, u32 n_blocks_x) {
        const Vec2<u32> index = ni::offset2index(blockIdx.x, n_blocks_x);
        auto bdhw = Vec4<Index>::from_values(
            blockIdx.z,
            blockIdx.y,
            Config::block_work_size_y * index[0] + threadIdx.y,
            Config::block_work_size_x * index[1] + threadIdx.x
        );

        using interface = ng::IwiseInterface;
        interface::init(op, thread_uid<3>());

        for (Index h = 0; h < Config::n_elements_per_thread_y; ++h) {
            for (Index w = 0; w < Config::n_elements_per_thread_x; ++w) {
                const Index ih = bdhw[2] + Config::block_size_y * h;
                const Index iw = bdhw[3] + Config::block_size_x * w;
                if (ih < end_hw[0] and iw < end_hw[1])
                    interface::call(op, bdhw[0], bdhw[1], ih, iw);
            }
        }
        interface::final(op, thread_uid<3>());
    }

    template<typename Config, typename Op, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void iwise_3d_static(Op op, Vec2<Index> end_hw) {
        auto dhw = Vec3<Index>::from_values(
            blockIdx.z,
            Config::block_work_size_y * blockIdx.y + threadIdx.y,
            Config::block_work_size_x * blockIdx.x + threadIdx.x
        );

        using interface = ng::IwiseInterface;
        interface::init(op, thread_uid<3>());

        for (Index h = 0; h < Config::n_elements_per_thread_y; ++h) {
            for (Index w = 0; w < Config::n_elements_per_thread_x; ++w) {
                const Index ih = dhw[1] + Config::block_size_y * h;
                const Index iw = dhw[2] + Config::block_size_x * w;
                if (ih < end_hw[0] and iw < end_hw[1])
                    interface::call(op, dhw[0], ih, iw);
            }
        }
        interface::final(op, thread_uid<3>());
    }

    template<typename Config, typename Op, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void iwise_2d_static(Op op, Vec2<Index> end_hw) {
        auto hw = Vec2<Index>::from_values(
            Config::block_work_size_y * blockIdx.y + threadIdx.y,
            Config::block_work_size_x * blockIdx.x + threadIdx.x
        );

        using interface = ng::IwiseInterface;
        interface::init(op, thread_uid<2>());

        for (Index h = 0; h < Config::n_elements_per_thread_y; ++h) {
            for (Index w = 0; w < Config::n_elements_per_thread_x; ++w) {
                const Index ih = hw[0] + Config::block_size_y * h;
                const Index iw = hw[1] + Config::block_size_x * w;
                if (ih < end_hw[0] and iw < end_hw[1])
                    interface::call(op, ih, iw);
            }
        }
        interface::final(op, thread_uid<2>());
    }

    template<typename Config, typename Op, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void iwise_1d_static(Op op, Vec1<Index> end) {
        auto index = Vec1<Index>::from_values(Config::block_work_size_x * blockIdx.x + threadIdx.x);

        using interface = ng::IwiseInterface;
        interface::init(op, thread_uid<1>());

        for (Index w = 0; w < Config::n_elements_per_thread_x; ++w) {
            const Index iw = index[0] + Config::block_size_x * w;
            if (iw < end[0])
                interface::call(op, iw);
        }
        interface::final(op, thread_uid<1>());
    }
}

#ifdef NOA_IS_OFFLINE
namespace noa::cuda::guts {
    template<typename Config, typename Index>
    auto iwise_4d_static_config(
        const Shape4<Index>& shape,
        size_t n_bytes_of_shared_memory
    ) -> Pair<LaunchConfig, u32> {
        const auto iwise_shape = shape.template as_safe<u32>();
        const u32 n_blocks_x = divide_up(iwise_shape[3], Config::block_work_size_x);
        const u32 n_blocks_y = divide_up(iwise_shape[2], Config::block_work_size_y);
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x * n_blocks_y, iwise_shape[1], iwise_shape[0]),
                .n_threads=dim3(Config::block_size_x, Config::block_size_y),
                .n_bytes_of_shared_memory=n_bytes_of_shared_memory};
        return {launch_config, n_blocks_x};
    }

    template<typename Config, typename Index>
    auto iwise_3d_static_config(
        const Shape3<Index>& shape,
        size_t n_bytes_of_shared_memory
    ) -> LaunchConfig {
        const auto iwise_shape = shape.template as_safe<u32>();
        const u32 n_blocks_x = divide_up(iwise_shape[2], Config::block_work_size_x);
        const u32 n_blocks_y = divide_up(iwise_shape[1], Config::block_work_size_y);
        return {
            .n_blocks = dim3(n_blocks_x, n_blocks_y, iwise_shape[0]),
            .n_threads = dim3(Config::block_size_x, Config::block_size_y),
            .n_bytes_of_shared_memory = n_bytes_of_shared_memory
        };
    }

    template<typename Config, typename Index>
    auto iwise_2d_static_config(
        const Shape2<Index>& shape,
        size_t n_bytes_of_shared_memory
    ) -> LaunchConfig {
        const auto iwise_shape = shape.template as_safe<u32>();
        const u32 n_blocks_x = divide_up(iwise_shape[1], Config::block_work_size_x);
        const u32 n_blocks_y = divide_up(iwise_shape[0], Config::block_work_size_y);
        return LaunchConfig{
            .n_blocks = dim3(n_blocks_x, n_blocks_y),
            .n_threads = dim3(Config::block_size_x, Config::block_size_y),
            .n_bytes_of_shared_memory = n_bytes_of_shared_memory
        };
    }

    template<typename Config, typename Index>
    auto iwise_1d_static_config(
        const Shape1<Index>& shape,
        size_t n_bytes_of_shared_memory
    ) -> LaunchConfig {
        static_assert(Config::block_size_y == 1, "1d index-wise doesn't support 2d blocks");
        const auto iwise_shape = shape.template as_safe<u32>();
        const u32 n_blocks_x = divide_up(iwise_shape[0], Config::block_work_size_x);
        return LaunchConfig{
            .n_blocks = dim3(n_blocks_x),
            .n_threads = dim3(Config::block_size_x),
            .n_bytes_of_shared_memory = n_bytes_of_shared_memory
        };
    }
}

namespace noa::cuda {
    template<u32 BlockSizeX, u32 BlockSizeY,
             u32 ElementsPerThreadX = 1,
             u32 ElementsPerThreadY = 1>
    struct IwiseConfig {
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = BlockSizeY;
        static constexpr u32 block_size = block_size_x * block_size_y;
        static constexpr u32 n_elements_per_thread_x = ElementsPerThreadX;
        static constexpr u32 n_elements_per_thread_y = ElementsPerThreadY;
        static constexpr u32 block_work_size_y = block_size_y * n_elements_per_thread_y;
        static constexpr u32 block_work_size_x = block_size_x * n_elements_per_thread_x;
        static_assert(block_size_x * block_size_y <= Limits::MAX_THREADS);
    };

    template<size_t N>
    using IwiseConfigDefault = std::conditional_t<
        N == 1,
        IwiseConfig<Constant::WARP_SIZE * 8, 1, 1, 1>,
        IwiseConfig<Constant::WARP_SIZE, 256 / Constant::WARP_SIZE, 1, 1>>;

    template<size_t N, typename Config = IwiseConfigDefault<N>, typename Index, typename Op>
    void iwise(
        const Shape<Index, N>& shape,
        Op&& op,
        Stream& stream,
        size_t n_bytes_of_shared_memory = 0
    ) {
        if constexpr (N == 4) {
            const auto [launch_config, n_blocks_x] = guts::iwise_4d_static_config<Config>(shape, n_bytes_of_shared_memory);
            const auto end_2d = shape.filter(2, 3).vec;
            stream.enqueue(guts::iwise_4d_static<Config, std::decay_t<Op>, Index>,
                           launch_config, std::forward<Op>(op), end_2d, n_blocks_x);
        } else if constexpr (N == 3) {
            const auto launch_config = guts::iwise_3d_static_config<Config>(shape, n_bytes_of_shared_memory);
            const auto end_2d = shape.pop_front().vec;
            stream.enqueue(guts::iwise_3d_static<Config, std::decay_t<Op>, Index>,
                           launch_config, std::forward<Op>(op), end_2d);
        } else if constexpr (N == 2) {
            const auto launch_config = guts::iwise_2d_static_config<Config>(shape, n_bytes_of_shared_memory);
            stream.enqueue(guts::iwise_2d_static<Config, std::decay_t<Op>, Index>,
                           launch_config, std::forward<Op>(op), shape.vec);
        } else if constexpr (N == 1) {
            const auto launch_config = guts::iwise_1d_static_config<Config>(shape, n_bytes_of_shared_memory);
            stream.enqueue(guts::iwise_1d_static<Config, std::decay_t<Op>, Index>,
                           launch_config, std::forward<Op>(op), shape.vec);
        }
    }
}
#endif
