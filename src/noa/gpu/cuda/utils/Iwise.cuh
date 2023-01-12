#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Indexing.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::utils {
    template<uint32_t THREADS_X, uint32_t THREADS_Y,
             uint32_t ITERATIONS_X = 1, uint32_t ITERATIONS_Y = 1>
    struct IwiseStaticConfig {
        static constexpr uint32_t BLOCK_SIZE_X = THREADS_X;
        static constexpr uint32_t BLOCK_SIZE_Y = THREADS_Y;
        static constexpr uint32_t ELEMENTS_PER_THREAD_X = ITERATIONS_X;
        static constexpr uint32_t ELEMENTS_PER_THREAD_Y = ITERATIONS_Y;
        static constexpr uint32_t BLOCK_WORK_SIZE_Y = BLOCK_SIZE_Y * ELEMENTS_PER_THREAD_Y;
        static constexpr uint32_t BLOCK_WORK_SIZE_X = BLOCK_SIZE_X * ELEMENTS_PER_THREAD_X;

        static_assert(BLOCK_SIZE_X * BLOCK_SIZE_Y <= Limits::MAX_THREADS);
    };

    using IwiseStaticConfigDefault = IwiseStaticConfig<Limits::WARP_SIZE, 256 / Limits::WARP_SIZE, 1, 1>;

    // Given a shape, find a good block shape, with one element per thread.
    // TODO Use CUDA tools to find block/grid that maximizes occupancy.
    struct IwiseDynamicConfig {
        static constexpr uint32_t MAX_BLOCK_SIZE = 512;
    };
}

namespace noa::cuda::utils::details {
    template<typename functor_t, typename index_t, typename index4_or_empty_t, typename config_t>
    __global__ void __launch_bounds__(config_t::BLOCK_SIZE_X * config_t::BLOCK_SIZE_Y)
    iwise4DStatic(functor_t functor,
                  index4_or_empty_t start,
                  Int2<index_t> end_yx,
                  uint32_t blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        Int4<index_t> gid{blockIdx.z,
                          blockIdx.y,
                          config_t::BLOCK_WORK_SIZE_Y * index[0] + threadIdx.y,
                          config_t::BLOCK_WORK_SIZE_X * index[1] + threadIdx.x};
        if constexpr (!std::is_empty_v<index4_or_empty_t>)
            gid += start;

        #pragma unroll
        for (index_t k = 0; k < config_t::ELEMENTS_PER_THREAD_Y; ++k) {
            #pragma unroll
            for (index_t l = 0; l < config_t::ELEMENTS_PER_THREAD_X; ++l) {
                const index_t ik = gid[2] + config_t::BLOCK_SIZE_Y * k;
                const index_t il = gid[3] + config_t::BLOCK_SIZE_X * l;
                if (ik < end_yx[0] && il < end_yx[1])
                    functor(gid[0], gid[1], ik, il);
            }
        }
    }

    template<typename config_t, typename index_t>
    std::pair<LaunchConfig, uint32_t>
    iwise4DStaticLaunchConfig(Int4<index_t> shape, size_t bytes_shared_memory) {
        const auto iwise_shape = safe_cast<uint4_t>(shape);
        const uint32_t blocks_x = math::divideUp(iwise_shape[3], config_t::BLOCK_WORK_SIZE_X);
        const uint32_t blocks_y = math::divideUp(iwise_shape[2], config_t::BLOCK_WORK_SIZE_Y);
        const dim3 blocks(blocks_x * blocks_y,
                          iwise_shape[1],
                          iwise_shape[0]);
        const LaunchConfig config{blocks, dim3(config_t::BLOCK_SIZE_X, config_t::BLOCK_SIZE_Y), bytes_shared_memory};
        return {config, blocks_x};
    }
}

namespace noa::cuda::utils::details {
    template<typename functor_t, typename index_t, typename index3_or_empty_t, typename config_t>
    __global__ void __launch_bounds__(config_t::BLOCK_SIZE_X * config_t::BLOCK_SIZE_Y)
    iwise3DStatic(functor_t functor, index3_or_empty_t start, Int2<index_t> end_yx) {
        Int3<index_t> gid{blockIdx.z,
                          config_t::BLOCK_WORK_SIZE_Y * blockIdx.y + threadIdx.y,
                          config_t::BLOCK_WORK_SIZE_X * blockIdx.x + threadIdx.x};
        if constexpr (!std::is_empty_v<index3_or_empty_t>)
            gid += start;

        #pragma unroll
        for (index_t k = 0; k < config_t::ELEMENTS_PER_THREAD_Y; ++k) {
            #pragma unroll
            for (index_t l = 0; l < config_t::ELEMENTS_PER_THREAD_X; ++l) {
                const index_t ik = gid[1] + config_t::BLOCK_SIZE_Y * k;
                const index_t il = gid[2] + config_t::BLOCK_SIZE_X * l;
                if (ik < end_yx[0] && il < end_yx[1])
                    functor(gid[0], ik, il);
            }
        }
    }

    template<typename config_t, typename index_t>
    LaunchConfig iwise3DStaticLaunchConfig(Int3<index_t> shape, size_t bytes_shared_memory) {
        const auto iwise_shape = safe_cast<uint3_t>(shape);
        const dim3 blocks(math::divideUp(iwise_shape[2], config_t::BLOCK_WORK_SIZE_X),
                          math::divideUp(iwise_shape[1], config_t::BLOCK_WORK_SIZE_Y),
                          iwise_shape[0]);
        return {blocks, dim3(config_t::BLOCK_SIZE_X, config_t::BLOCK_SIZE_Y), bytes_shared_memory};
    }
}

namespace noa::cuda::utils::details {
    template<typename functor_t, typename index_t, typename index2_or_empty_t, typename config_t>
    __global__ void __launch_bounds__(config_t::BLOCK_SIZE_X * config_t::BLOCK_SIZE_Y)
    iwise2DStatic(functor_t functor, index2_or_empty_t start, Int2<index_t> end) {
        Int2<index_t> gid{config_t::BLOCK_WORK_SIZE_Y * blockIdx.y + threadIdx.y,
                          config_t::BLOCK_WORK_SIZE_X * blockIdx.x + threadIdx.x};
        if constexpr (!std::is_empty_v<index2_or_empty_t>)
            gid += start;

        #pragma unroll
        for (index_t k = 0; k < config_t::ELEMENTS_PER_THREAD_Y; ++k) {
            #pragma unroll
            for (index_t l = 0; l < config_t::ELEMENTS_PER_THREAD_X; ++l) {
                const index_t ik = gid[0] + config_t::BLOCK_SIZE_Y * k;
                const index_t il = gid[1] + config_t::BLOCK_SIZE_X * l;
                if (ik < end[0] && il < end[1])
                    functor(ik, il);
            }
        }
    }

    template<typename config_t, typename index_t>
    LaunchConfig iwise2DStaticLaunchConfig(Int2<index_t> shape, size_t bytes_shared_memory) {
        const auto iwise_shape = safe_cast<uint2_t>(shape);
        const dim3 blocks(math::divideUp(iwise_shape[1], config_t::BLOCK_WORK_SIZE_X),
                          math::divideUp(iwise_shape[0], config_t::BLOCK_WORK_SIZE_Y));
        return {blocks, dim3(config_t::BLOCK_SIZE_X, config_t::BLOCK_SIZE_Y), bytes_shared_memory};
    }
}

namespace noa::cuda::utils::details {
    template<typename functor_t, typename index_t, typename index_or_empty_t, typename config_t>
    __global__ void __launch_bounds__(config_t::BLOCK_SIZE_X)
    iwise1DStatic(functor_t functor, index_or_empty_t start, index_t end) {
        auto gid = static_cast<index_t>(config_t::BLOCK_WORK_SIZE_X * blockIdx.x + threadIdx.x);
        if constexpr (!std::is_empty_v<index_or_empty_t>)
            gid += start;

        #pragma unroll
        for (index_t l = 0; l < config_t::ELEMENTS_PER_THREAD_X; ++l) {
            const index_t il = gid + config_t::BLOCK_SIZE_X * l;
            if (il < end)
                functor(il);
        }
    }

    template<typename config_t, typename index_t>
    LaunchConfig iwise1DStaticLaunchConfig(index_t size, size_t bytes_shared_memory) {
        static_assert(config_t::BLOCK_SIZE_Y == 1, "1D index-wise doesn't support 2D blocks");
        static_assert(config_t::ELEMENTS_PER_THREAD_Y == 1, "1D index-wise doesn't support 2D blocks");
        const auto iwise_size = safe_cast<uint32_t>(size);
        const dim3 blocks(math::divideUp(iwise_size, config_t::BLOCK_WORK_SIZE_X));
        return {blocks, dim3(config_t::BLOCK_SIZE_X), bytes_shared_memory};
    }
}

namespace noa::cuda::utils {
    // Index-wise kernels, looping in the rightmost-order. The functor (of course) copied to the device,
    // so it shouldn't have any host references. By default, the launch configuration is static and can
    // be changed by the caller. The functor can of course use any valid CUDA C++, including static
    // shared-memory. These functions also support dynamic shared-memory.

    template<typename config_t = IwiseStaticConfigDefault, typename index_t, typename functor_t>
    void iwise4D(const char* name,
                 const Int4<index_t>& start,
                 const Int4<index_t>& end,
                 functor_t&& functor,
                 Stream& stream,
                 size_t bytes_shared_memory = 0) {
        NOA_ASSERT(all(end >= 0) && all(end > start));
        using functor_value_t = noa::traits::remove_ref_cv_t<functor_t>;
        const auto shape = end - start;
        if constexpr (std::is_same_v<config_t, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<index_t>, "TODO");
        } else {
            const auto [config, blocks_x] = details::iwise4DStaticLaunchConfig<config_t>(shape, bytes_shared_memory);
            const Int2<index_t> end_2d(end.get(2));
            stream.enqueue(name,
                           details::iwise4DStatic<functor_value_t, index_t, Int4<index_t>, config_t>,
                           config, std::forward<functor_t>(functor), start, end_2d, blocks_x);
        }
    }

    template<typename config_t = IwiseStaticConfigDefault, typename index_t, typename functor_t>
    void iwise4D(const char* name,
                 const Int4<index_t>& shape,
                 functor_t&& functor,
                 Stream& stream,
                 size_t bytes_shared_memory = 0) {
        using functor_value_t = noa::traits::remove_ref_cv_t<functor_t>;
        if constexpr (std::is_same_v<config_t, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<index_t>, "TODO");
        } else {
            const auto[config, blocks_x] = details::iwise4DStaticLaunchConfig<config_t>(shape, bytes_shared_memory);
            const Int2<index_t> end_2d(shape.get(2));
            stream.enqueue(name,
                           details::iwise4DStatic<functor_value_t, index_t, empty_t, config_t>,
                           config, std::forward<functor_t>(functor), empty_t{}, end_2d, blocks_x);
        }
    }

    template<typename config_t = IwiseStaticConfigDefault, typename index_t, typename functor_t>
    void iwise3D(const char* name,
                 const Int3<index_t>& start,
                 const Int3<index_t>& end,
                 functor_t&& functor,
                 Stream& stream,
                 size_t bytes_shared_memory = 0) {
        NOA_ASSERT(all(end >= 0) && all(end > start));
        using functor_value_t = noa::traits::remove_ref_cv_t<functor_t>;
        const auto shape = end - start;
        if constexpr (std::is_same_v<config_t, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<index_t>, "TODO");
        } else {
            const auto config = details::iwise3DStaticLaunchConfig<config_t>(shape, bytes_shared_memory);
            const Int2<index_t> end_2d(end.get(1));
            stream.enqueue(name,
                           details::iwise3DStatic<functor_value_t, index_t, Int3<index_t>, config_t>,
                           config, std::forward<functor_t>(functor), start, end_2d);
        }
    }

    template<typename config_t = IwiseStaticConfigDefault, typename index_t, typename functor_t>
    void iwise3D(const char* name,
                 const Int3<index_t>& shape,
                 functor_t&& functor,
                 Stream& stream,
                 size_t bytes_shared_memory = 0) {
        using functor_value_t = noa::traits::remove_ref_cv_t<functor_t>;
        if constexpr (std::is_same_v<config_t, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<index_t>, "TODO");
        } else {
            const auto config = details::iwise3DStaticLaunchConfig<config_t>(shape, bytes_shared_memory);
            const Int2<index_t> end_2d(shape.get(1));
            stream.enqueue(name,
                           details::iwise3DStatic<functor_value_t, index_t, empty_t, config_t>,
                           config, std::forward<functor_t>(functor), empty_t{}, end_2d);
        }
    }

    template<typename config_t = IwiseStaticConfigDefault, typename index_t, typename functor_t>
    void iwise2D(const char* name,
                 const Int2<index_t>& start,
                 const Int2<index_t>& end,
                 functor_t&& functor,
                 Stream& stream,
                 size_t bytes_shared_memory = 0) {
        NOA_ASSERT(all(end >= 0) && all(end > start));
        using functor_value_t = noa::traits::remove_ref_cv_t<functor_t>;
        const auto shape = end - start;
        if constexpr (std::is_same_v<config_t, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<index_t>, "TODO");
        } else {
            const auto config = details::iwise2DStaticLaunchConfig<config_t>(shape, bytes_shared_memory);
            stream.enqueue(name,
                           details::iwise2DStatic<functor_value_t, index_t, Int2<index_t>, config_t>,
                           config, std::forward<functor_t>(functor), start, end);
        }
    }

    template<typename config_t = IwiseStaticConfigDefault, typename index_t, typename functor_t>
    void iwise2D(const char* name,
                 const Int2<index_t>& shape,
                 functor_t&& functor,
                 Stream& stream,
                 size_t bytes_shared_memory = 0) {
        using functor_value_t = noa::traits::remove_ref_cv_t<functor_t>;
        if constexpr (std::is_same_v<config_t, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<index_t>, "TODO");
        } else {
            const auto config = details::iwise2DStaticLaunchConfig<config_t>(shape, bytes_shared_memory);
            stream.enqueue(name,
                           details::iwise2DStatic<functor_value_t, index_t, empty_t, config_t>,
                           config, std::forward<functor_t>(functor), empty_t{}, shape);
        }
    }

    template<typename config_t = IwiseStaticConfigDefault, typename index_t, typename functor_t>
    void iwise1D(const char* name,
                 index_t start,
                 index_t end,
                 functor_t&& functor,
                 Stream& stream,
                 size_t bytes_shared_memory = 0) {
        NOA_ASSERT(all(end >= 0) && all(end > start));
        using functor_value_t = noa::traits::remove_ref_cv_t<functor_t>;
        const auto size = end - start;
        if constexpr (std::is_same_v<config_t, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<index_t>, "TODO");
        } else {
            const auto config = details::iwise1DStaticLaunchConfig<config_t>(size, bytes_shared_memory);
            stream.enqueue(name,
                           details::iwise1DStatic<functor_value_t, index_t, index_t, config_t>,
                           config, std::forward<functor_t>(functor), start, end);
        }
    }

    template<typename config_t = IwiseStaticConfigDefault, typename index_t, typename functor_t>
    void iwise1D(const char* name,
                 index_t size,
                 functor_t&& functor,
                 Stream& stream,
                 size_t bytes_shared_memory = 0) {
        using functor_value_t = noa::traits::remove_ref_cv_t<functor_t>;
        if constexpr (std::is_same_v<config_t, IwiseDynamicConfig>) {
            static_assert(noa::traits::always_false_v<index_t>, "TODO");
        } else {
            const auto config = details::iwise1DStaticLaunchConfig<config_t>(size, bytes_shared_memory);
            stream.enqueue(name,
                           details::iwise1DStatic<functor_value_t, index_t, empty_t, config_t>,
                           config, std::forward<functor_t>(functor), empty_t{}, size);
        }
    }
}
