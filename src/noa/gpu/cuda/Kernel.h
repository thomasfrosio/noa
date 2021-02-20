#pragma once

#include "noa/gpu/cuda/CudaRuntime.h"
#include "noa/gpu/cuda/Stream.h"

namespace Noa::CUDA {
    /** A launch configuration. Mostly used with the launch function. */
    struct LaunchConfig {
        using dim_t = dim3; // For now, use dim3 since it is enough for LaunchConfig.
        dim_t blocks_per_grid{0};
        dim_t threads_per_block{0};
        size_t shared_memory{0};

        /** Empty config. Member variables should be accessed individually. */
        NOA_IH constexpr LaunchConfig() = default;

        /** Initializes a 1D config. */
        NOA_IH constexpr LaunchConfig(uint a_blocks_per_grid, uint a_threads_per_block, size_t a_shared_memory = 0U)
                : blocks_per_grid(a_blocks_per_grid),
                  threads_per_block(a_threads_per_block),
                  shared_memory(a_shared_memory) {}

        /** Initializes a 1D, 2D, or 3D config. */
        NOA_IH constexpr LaunchConfig(dim_t a_blocks_per_grid, dim_t a_threads_per_block, size_t a_shared_memory = 0U)
                : blocks_per_grid(a_blocks_per_grid),
                  threads_per_block(a_threads_per_block),
                  shared_memory(a_shared_memory) {}
    };

    /**
     * Launch a @a kernel (asynchronously).
     * @tparam Kernel               Function or function pointer.
     * @tparam KernelParameters     Parameter types of the @a kernel.
     * @param kernel                Kernel to launch, on a @a stream, with a given @a config and @a parameters.
     * @param[in] config            A launch configuration.
     * @param[in] stream            Stream to enqueue.
     * @param[in] parameters        Parameters of @a kernel. (optional)
     * @return                      Last error. This function does not check error before launching the kernel.
     *
     * @note    templated functions can be used, e.g. `launch(kernel<T1, T2>, ...)`.
     * @note    It is meant to be used with @c NOA_THROW_IF, to get a meaningful file/function/line.
     * @warning This function should only be used in CUDA translation units (compilation steered by nvcc).
     *
     * @todo I wonder how difficult it is to make a wrapper that can be called from a .cpp source file.
     */
    template<typename Kernel, typename... KernelParameters>
    NOA_IHD cudaError_t launch(Kernel kernel, const LaunchConfig& config,
                       const Stream& stream, KernelParameters&& ... parameters) {
#ifndef __CUDACC__
        NOA_THROW("DEV: Trying to launch a CUDA kernel from a C++ source file..."); // static_assert is not good here.
#else
        static_assert(std::is_function_v<Kernel> || Traits::is_function_ptr_v<Kernel>);
        kernel<<<config.blocks_per_grid,
                 config.threads_per_block,
                 config.shared_memory,
                 stream.get()>>>(std::forward<KernelParameters>(parameters)...);
        return cudaPeekAtLastError();
#endif
    }
}
