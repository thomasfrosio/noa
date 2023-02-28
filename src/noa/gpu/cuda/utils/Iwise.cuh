#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/utils/Indexing.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::utils {
    template<u32 THREADS_X, u32 THREADS_Y,
             u32 ITERATIONS_X = 1, u32 ITERATIONS_Y = 1>
    struct IwiseStaticConfig {
        static constexpr u32 BLOCK_SIZE_X = THREADS_X;
        static constexpr u32 BLOCK_SIZE_Y = THREADS_Y;
        static constexpr u32 ELEMENTS_PER_THREAD_X = ITERATIONS_X;
        static constexpr u32 ELEMENTS_PER_THREAD_Y = ITERATIONS_Y;
        static constexpr u32 BLOCK_WORK_SIZE_Y = BLOCK_SIZE_Y * ELEMENTS_PER_THREAD_Y;
        static constexpr u32 BLOCK_WORK_SIZE_X = BLOCK_SIZE_X * ELEMENTS_PER_THREAD_X;

        static_assert(BLOCK_SIZE_X * BLOCK_SIZE_Y <= Limits::MAX_THREADS);
    };

    using IwiseStaticConfigDefault2D = IwiseStaticConfig<Constant::WARP_SIZE, 256 / Constant::WARP_SIZE, 1, 1>;
    using IwiseStaticConfigDefault1D = IwiseStaticConfig<Constant::WARP_SIZE * 8, 1, 1, 1>;

    // Given a shape, find a good block shape, with one element per thread.
    // TODO Use CUDA tools to find block/grid that maximizes occupancy.
    struct IwiseDynamicConfig {
        static constexpr u32 MAX_BLOCK_SIZE = 512;
    };
}

namespace noa::cuda::utils::details {
    template<typename IwiseOp, typename Index, typename Vec4OrEmpty, typename Config>
    __global__ void __launch_bounds__(Config::BLOCK_SIZE_X * Config::BLOCK_SIZE_Y)
    iwise_4d_static(IwiseOp iwise_op,
                    Vec4OrEmpty start,
                    Vec2<Index> end_yx,
                    u32 blocks_x) {
        const Vec2<u32> index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        Vec4<Index> gid{blockIdx.z,
                        blockIdx.y,
                        Config::BLOCK_WORK_SIZE_Y * index[0] + threadIdx.y,
                        Config::BLOCK_WORK_SIZE_X * index[1] + threadIdx.x};
        if constexpr (!std::is_empty_v<Vec4OrEmpty>)
            gid += start;

        #pragma unroll
        for (Index k = 0; k < Config::ELEMENTS_PER_THREAD_Y; ++k) {
            #pragma unroll
            for (Index l = 0; l < Config::ELEMENTS_PER_THREAD_X; ++l) {
                const Index ik = gid[2] + Config::BLOCK_SIZE_Y * k;
                const Index il = gid[3] + Config::BLOCK_SIZE_X * l;
                if (ik < end_yx[0] && il < end_yx[1])
                    iwise_op(gid[0], gid[1], ik, il);
            }
        }
    }

    template<typename Config, typename Index>
    std::pair<LaunchConfig, u32>
    iwise_4d_static_config(const Shape4<Index>& shape, size_t bytes_shared_memory) {
        const auto iwise_shape = shape.filter(2, 3).template as_safe<u32>();
        const u32 blocks_x = noa::math::divide_up(iwise_shape[1], Config::BLOCK_WORK_SIZE_X);
        const u32 blocks_y = noa::math::divide_up(iwise_shape[0], Config::BLOCK_WORK_SIZE_Y);
        const dim3 blocks(blocks_x * blocks_y,
                          iwise_shape[1],
                          iwise_shape[0]);
        const LaunchConfig config{blocks, dim3(Config::BLOCK_SIZE_X, Config::BLOCK_SIZE_Y), bytes_shared_memory};
        return {config, blocks_x};
    }
}

namespace noa::cuda::utils::details {
    template<typename IwiseOp, typename Index, typename Vec3OrEmpty, typename Config>
    __global__ void __launch_bounds__(Config::BLOCK_SIZE_X * Config::BLOCK_SIZE_Y)
    iwise_3d_static(IwiseOp iwise_op, Vec3OrEmpty start, Vec2<Index> end_yx) {
        Vec3<Index> gid{blockIdx.z,
                        Config::BLOCK_WORK_SIZE_Y * blockIdx.y + threadIdx.y,
                        Config::BLOCK_WORK_SIZE_X * blockIdx.x + threadIdx.x};
        if constexpr (!std::is_empty_v<Vec3OrEmpty>)
            gid += start;

        #pragma unroll
        for (Index k = 0; k < Config::ELEMENTS_PER_THREAD_Y; ++k) {
            #pragma unroll
            for (Index l = 0; l < Config::ELEMENTS_PER_THREAD_X; ++l) {
                const Index ik = gid[1] + Config::BLOCK_SIZE_Y * k;
                const Index il = gid[2] + Config::BLOCK_SIZE_X * l;
                if (ik < end_yx[0] && il < end_yx[1])
                    iwise_op(gid[0], ik, il);
            }
        }
    }

    template<typename Config, typename Index>
    LaunchConfig iwise_3d_static_config(const Shape3<Index>& shape, size_t bytes_shared_memory) {
        const auto iwise_shape = shape.template as_safe<u32>();
        const dim3 blocks(noa::math::divide_up(iwise_shape[2], Config::BLOCK_WORK_SIZE_X),
                          noa::math::divide_up(iwise_shape[1], Config::BLOCK_WORK_SIZE_Y),
                          iwise_shape[0]);
        return {blocks, dim3(Config::BLOCK_SIZE_X, Config::BLOCK_SIZE_Y), bytes_shared_memory};
    }
}

namespace noa::cuda::utils::details {
    template<typename IwiseOp, typename Index, typename Vec2OrEmpty, typename Config>
    __global__ void __launch_bounds__(Config::BLOCK_SIZE_X * Config::BLOCK_SIZE_Y)
    iwise_2d_static(IwiseOp iwise_op, Vec2OrEmpty start, Vec2<Index> end) {
        Vec2<Index> gid{Config::BLOCK_WORK_SIZE_Y * blockIdx.y + threadIdx.y,
                        Config::BLOCK_WORK_SIZE_X * blockIdx.x + threadIdx.x};
        if constexpr (!std::is_empty_v<Vec2OrEmpty>)
            gid += start;

        #pragma unroll
        for (Index k = 0; k < Config::ELEMENTS_PER_THREAD_Y; ++k) {
            #pragma unroll
            for (Index l = 0; l < Config::ELEMENTS_PER_THREAD_X; ++l) {
                const Index ik = gid[0] + Config::BLOCK_SIZE_Y * k;
                const Index il = gid[1] + Config::BLOCK_SIZE_X * l;
                if (ik < end[0] && il < end[1])
                    iwise_op(ik, il);
            }
        }
    }

    template<typename Config, typename Index>
    LaunchConfig iwise_2d_static_config(const Shape2<Index>& shape, size_t bytes_shared_memory) {
        const auto iwise_shape = shape.template as_safe<u32>();
        const dim3 blocks(noa::math::divide_up(iwise_shape[1], Config::BLOCK_WORK_SIZE_X),
                          noa::math::divide_up(iwise_shape[0], Config::BLOCK_WORK_SIZE_Y));
        return {blocks, dim3(Config::BLOCK_SIZE_X, Config::BLOCK_SIZE_Y), bytes_shared_memory};
    }
}

namespace noa::cuda::utils::details {
    template<typename IwiseOp, typename Index, typename IndexOrEmpty, typename Config>
    __global__ void __launch_bounds__(Config::BLOCK_SIZE_X)
    iwise_1d_static(IwiseOp iwise_op, IndexOrEmpty start, Index end) {
        auto gid = static_cast<Index>(Config::BLOCK_WORK_SIZE_X * blockIdx.x + threadIdx.x);
        if constexpr (!std::is_empty_v<IndexOrEmpty>)
            gid += start;

        #pragma unroll
        for (Index l = 0; l < Config::ELEMENTS_PER_THREAD_X; ++l) {
            const Index il = gid + Config::BLOCK_SIZE_X * l;
            if (il < end)
                iwise_op(il);
        }
    }

    template<typename Config, typename Index>
    LaunchConfig iwise_1d_static_config(Index size, size_t bytes_shared_memory) {
        static_assert(Config::BLOCK_SIZE_Y == 1, "1D index-wise doesn't support 2D blocks");
        static_assert(Config::ELEMENTS_PER_THREAD_Y == 1, "1D index-wise doesn't support 2D blocks");
        const auto iwise_size = safe_cast<u32>(size);
        const dim3 blocks(noa::math::divide_up(iwise_size, Config::BLOCK_WORK_SIZE_X));
        return {blocks, dim3(Config::BLOCK_SIZE_X), bytes_shared_memory};
    }
}

namespace noa::cuda::utils {
    // Index-wise kernels, looping in the rightmost-order. The iwise_op (of course) copied to the device,
    // so it shouldn't have any host references. By default, the launch configuration is static and can
    // be changed by the caller. The iwise_op can of course use any valid CUDA C++, including static
    // shared-memory. These functions also support dynamic shared-memory.

    template<typename Config = IwiseStaticConfigDefault2D, typename Index, typename IwiseOp>
    void iwise_4d(const char* name,
                  const Vec4<Index>& start,
                  const Vec4<Index>& end,
                  IwiseOp&& iwise_op,
                  Stream& stream,
                  size_t bytes_shared_memory = 0) {
        NOA_ASSERT(noa::all(end >= 0) && noa::all(end > start));
        using iwise_op_value_t = noa::traits::remove_ref_cv_t<IwiseOp>;
        const auto shape = Shape4<Index>(end - start);
        if constexpr (std::is_same_v<Config, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<Index>, "TODO");
        } else {
            const auto [config, blocks_x] = details::iwise_4d_static_config<Config>(shape, bytes_shared_memory);
            const auto end_2d = end.filter(2, 3);
            stream.enqueue(name,
                           details::iwise_4d_static<iwise_op_value_t, Index, Vec4<Index>, Config>,
                           config, std::forward<IwiseOp>(iwise_op), start, end_2d, blocks_x);
        }
    }

    template<typename Config = IwiseStaticConfigDefault2D, typename Index, typename IwiseOp>
    void iwise_4d(const char* name,
                  const Shape4<Index>& shape,
                  IwiseOp&& iwise_op,
                  Stream& stream,
                  size_t bytes_shared_memory = 0) {
        using iwise_op_value_t = noa::traits::remove_ref_cv_t<IwiseOp>;
        if constexpr (std::is_same_v<Config, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<Index>, "TODO");
        } else {
            const auto[config, blocks_x] = details::iwise_4d_static_config<Config>(shape, bytes_shared_memory);
            const auto end_2d = shape.filter(2, 3).vec();
            stream.enqueue(name,
                           details::iwise_4d_static<iwise_op_value_t, Index, noa::traits::Empty, Config>,
                           config, std::forward<IwiseOp>(iwise_op), noa::traits::Empty{}, end_2d, blocks_x);
        }
    }

    template<typename Config = IwiseStaticConfigDefault2D, typename Index, typename IwiseOp>
    void iwise_3d(const char* name,
                  const Vec3<Index>& start,
                  const Vec3<Index>& end,
                  IwiseOp&& iwise_op,
                  Stream& stream,
                  size_t bytes_shared_memory = 0) {
        NOA_ASSERT(noa::all(end >= 0) && noa::all(end > start));
        using iwise_op_value_t = noa::traits::remove_ref_cv_t<IwiseOp>;
        const auto shape = Shape3<Index>(end - start);
        if constexpr (std::is_same_v<Config, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<Index>, "TODO");
        } else {
            const auto config = details::iwise_3d_static_config<Config>(shape, bytes_shared_memory);
            const auto end_2d = end.pop_front();
            stream.enqueue(name,
                           details::iwise_3d_static<iwise_op_value_t, Index, Vec3<Index>, Config>,
                           config, std::forward<IwiseOp>(iwise_op), start, end_2d);
        }
    }

    template<typename Config = IwiseStaticConfigDefault2D, typename Index, typename IwiseOp>
    void iwise_3d(const char* name,
                  const Shape3<Index>& shape,
                  IwiseOp&& iwise_op,
                  Stream& stream,
                  size_t bytes_shared_memory = 0) {
        using iwise_op_value_t = noa::traits::remove_ref_cv_t<IwiseOp>;
        if constexpr (std::is_same_v<Config, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<Index>, "TODO");
        } else {
            const auto config = details::iwise_3d_static_config<Config>(shape, bytes_shared_memory);
            const auto end_2d = shape.pop_front().vec();
            stream.enqueue(name,
                           details::iwise_3d_static<iwise_op_value_t, Index, noa::traits::Empty, Config>,
                           config, std::forward<IwiseOp>(iwise_op), noa::traits::Empty{}, end_2d);
        }
    }

    template<typename Config = IwiseStaticConfigDefault2D, typename Index, typename IwiseOp>
    void iwise_2d(const char* name,
                  const Vec2<Index>& start,
                  const Vec2<Index>& end,
                  IwiseOp&& iwise_op,
                  Stream& stream,
                  size_t bytes_shared_memory = 0) {
        NOA_ASSERT(noa::all(end >= 0) && noa::all(end > start));
        using iwise_op_value_t = noa::traits::remove_ref_cv_t<IwiseOp>;
        const auto shape = Shape2<Index>(end - start);
        if constexpr (std::is_same_v<Config, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<Index>, "TODO");
        } else {
            const auto config = details::iwise_2d_static_config<Config>(shape, bytes_shared_memory);
            stream.enqueue(name,
                           details::iwise_2d_static<iwise_op_value_t, Index, Vec2<Index>, Config>,
                           config, std::forward<IwiseOp>(iwise_op), start, end);
        }
    }

    template<typename Config = IwiseStaticConfigDefault2D, typename Index, typename IwiseOp>
    void iwise_2d(const char* name,
                  const Shape2<Index>& shape,
                  IwiseOp&& iwise_op,
                  Stream& stream,
                  size_t bytes_shared_memory = 0) {
        using iwise_op_value_t = noa::traits::remove_ref_cv_t<IwiseOp>;
        if constexpr (std::is_same_v<Config, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<Index>, "TODO");
        } else {
            const auto config = details::iwise_2d_static_config<Config>(shape, bytes_shared_memory);
            stream.enqueue(name,
                           details::iwise_2d_static<iwise_op_value_t, Index, noa::traits::Empty, Config>,
                           config, std::forward<IwiseOp>(iwise_op), noa::traits::Empty{}, shape);
        }
    }

    template<typename Config = IwiseStaticConfigDefault1D, typename Index, typename IwiseOp>
    void iwise_1d(const char* name,
                  Index start,
                  Index end,
                  IwiseOp&& iwise_op,
                  Stream& stream,
                  size_t bytes_shared_memory = 0) {
        NOA_ASSERT(noa::all(end >= 0) && noa::all(end > start));
        using iwise_op_value_t = noa::traits::remove_ref_cv_t<IwiseOp>;
        const auto size = end - start;
        if constexpr (std::is_same_v<Config, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<Index>, "TODO");
        } else {
            const auto config = details::iwise_1d_static_config<Config>(size, bytes_shared_memory);
            stream.enqueue(name,
                           details::iwise_1d_static<iwise_op_value_t, Index, Index, Config>,
                           config, std::forward<IwiseOp>(iwise_op), start, end);
        }
    }

    template<typename Config = IwiseStaticConfigDefault1D, typename Index, typename IwiseOp>
    void iwise_1d(const char* name,
                  Index size,
                  IwiseOp&& iwise_op,
                  Stream& stream,
                  size_t bytes_shared_memory = 0) {
        using iwise_op_value_t = noa::traits::remove_ref_cv_t<IwiseOp>;
        if constexpr (std::is_same_v<Config, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<Index>, "TODO");
        } else {
            const auto config = details::iwise_1d_static_config<Config>(size, bytes_shared_memory);
            stream.enqueue(name,
                           details::iwise_1d_static<iwise_op_value_t, Index, noa::traits::Empty, Config>,
                           config, std::forward<IwiseOp>(iwise_op), noa::traits::Empty{}, size);
        }
    }
}
