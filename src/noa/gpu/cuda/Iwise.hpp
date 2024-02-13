#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/Exception.hpp"
#include "noa/gpu/cuda/kernels/Iwise.cuh"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

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
                .n_blocks=dim3(n_blocks_x, n_blocks_y, iwise_shape[0]),
                .n_threads=dim3(Config::block_size_x, Config::block_size_y),
                .n_bytes_of_shared_memory=n_bytes_of_shared_memory
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
                .n_blocks=dim3(n_blocks_x, n_blocks_y),
                .n_threads=dim3(Config::block_size_x, Config::block_size_y),
                .n_bytes_of_shared_memory=n_bytes_of_shared_memory
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
                .n_blocks=dim3(n_blocks_x),
                .n_threads=dim3(Config::block_size_x),
                .n_bytes_of_shared_memory=n_bytes_of_shared_memory
        };
    }
}

namespace noa::cuda {
    template<size_t N, typename Config = IwiseConfigDefault<N>, typename Index, typename Op>
    void iwise(
            const Vec<Index, N>& start,
            const Vec<Index, N>& end,
            Op&& op,
            Stream& stream,
            size_t n_bytes_of_shared_memory = 0
    ) {
        NOA_ASSERT(all(end >= 0) && all(end > start));
        const auto shape = Shape{end - start};
        if constexpr (N == 4) {
            const auto [launch_config, n_blocks_x] = guts::iwise_4d_static_config<Config>(shape, n_bytes_of_shared_memory);
            const auto end_2d = end.filter(2, 3);
            stream.enqueue(guts::iwise_4d_static<Config, std::decay_t<Op>, Index, Vec4<Index>>,
                           launch_config, std::forward<Op>(op), start, end_2d, n_blocks_x);
        } else if constexpr (N == 3) {
            const auto launch_config = guts::iwise_3d_static_config<Config>(shape, n_bytes_of_shared_memory);
            const auto end_2d = end.pop_front();
            stream.enqueue(guts::iwise_3d_static<Config, std::decay_t<Op>, Vec3<Index>>,
                           launch_config, std::forward<Op>(op), start, end_2d);
        } else if constexpr (N == 2) {
            const auto launch_config = guts::iwise_2d_static_config<Config>(shape, n_bytes_of_shared_memory);
            stream.enqueue(guts::iwise_2d_static<Config, std::decay_t<Op>, Index, Vec2<Index>>,
                           launch_config, std::forward<Op>(op), start, end);
        } else if constexpr (N == 1) {
            const auto launch_config = guts::iwise_1d_static_config<Config>(shape, n_bytes_of_shared_memory);
            stream.enqueue(guts::iwise_1d_static<Config, std::decay_t<Op>, Index, Index>,
                           launch_config, std::forward<Op>(op), start, end);
        }
    }

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
            stream.enqueue(guts::iwise_4d_static<Config, std::decay_t<Op>, Index, Empty>,
                           launch_config, std::forward<Op>(op), Empty{}, end_2d, n_blocks_x);
        } else if constexpr (N == 3) {
            const auto launch_config = guts::iwise_3d_static_config<Config>(shape, n_bytes_of_shared_memory);
            const auto end_2d = shape.pop_front().vec;
            stream.enqueue(guts::iwise_3d_static<Config, std::decay_t<Op>, Index, Empty>,
                           launch_config, std::forward<Op>(op), Empty{}, end_2d);
        } else if constexpr (N == 2) {
            const auto launch_config = guts::iwise_2d_static_config<Config>(shape, n_bytes_of_shared_memory);
            stream.enqueue(guts::iwise_2d_static<Config, std::decay_t<Op>, Index, Empty>,
                           launch_config, std::forward<Op>(op), Empty{}, shape.vec);
        } else if constexpr (N == 1) {
            const auto launch_config = guts::iwise_1d_static_config<Config>(shape, n_bytes_of_shared_memory);
            stream.enqueue(guts::iwise_1d_static<Config, std::decay_t<Op>, Index, Empty>,
                           launch_config, std::forward<Op>(op), Empty{}, shape.vec);
        }
    }
}
#endif
