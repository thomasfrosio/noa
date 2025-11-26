#ifdef NOA_ENABLE_OPENMP
#include <omp.h>
#endif

#include "noa/cpu/fft/Plan.hpp"
#ifdef NOA_ENABLE_CUDA
#include <cstdlib>
#include <cuda.h>
#include "noa/gpu/cuda/fft/Plan.hpp"
#include "noa/gpu/cuda/Blas.hpp"
#endif

#include "noa/unified/Session.hpp"
#include "noa/core/utils/Strings.hpp"

noa::i64 noa::Session::m_thread_limit = 0;

namespace noa::inline types {
    void Session::set_thread_limit(i64 n_threads) {
        if (n_threads > 0) {
            m_thread_limit = n_threads;
        } else {
            i64 max_threads;
            const char* str = std::getenv("NOA_THREADS");
            if (str) {
                max_threads = ns::parse<i64>(str).value_or(1);
            } else {
                #ifdef NOA_ENABLE_OPENMP
                str = std::getenv("OMP_NUM_THREADS");
                if (str)
                    max_threads = ns::parse<i64>(str).value_or(1);
                else
                    max_threads = static_cast<i64>(omp_get_max_threads());
                #else
                max_threads = std::thread::hardware_concurrency();
                #endif
            }
            m_thread_limit = std::max(max_threads, i64{1});
        }
    }

    bool Session::set_gpu_lazy_loading() {
        #if defined(NOA_ENABLE_CUDA) && CUDART_VERSION >= 11070
        // It seems that CUDA_MODULE_LOADING is only read once during cuInit().
        // Try lazy loading if not explicitly set already.
        [[maybe_unused]] const auto err = setenv("CUDA_MODULE_LOADING", "LAZY", /*replace=*/false);

        // Check whether lazy mode is already enabled.
        CUmoduleLoadingMode mode;
        if (cuInit(0) != CUDA_SUCCESS or cuModuleGetLoadingMode(&mode) != CUDA_SUCCESS)
            panic("Failed to initialize and query the CUDA driver");
        return mode == CU_MODULE_LAZY_LOADING;
        #else
        return false;
        #endif
    }

    auto Session::clear_fft_cache(Device device) -> i64 {
        if (device.is_cpu())
            return noa::cpu::fft::clear_cache();
        #ifdef NOA_ENABLE_CUDA
        auto cuda_device = noa::cuda::Device(device.id(), Unchecked{});
        return noa::cuda::fft::clear_cache(cuda_device);
        #else
        return 0;
        #endif
    }

    auto Session::set_fft_cache_limit(i64 count, Device device) -> i64 {
        #ifdef NOA_ENABLE_CUDA
        if (device.is_gpu()) {
            auto cuda_device = noa::cuda::Device(device.id(), Unchecked{});
            return noa::cuda::fft::set_cache_limit(cuda_device, safe_cast<i32>(count));
        }
        #else
        (void) count;
        (void) device;
        #endif
        return -1;
    }

    auto Session::fft_cache_limit(Device device) -> i64 {
        #ifdef NOA_ENABLE_CUDA
        if (device.is_gpu()) {
            auto cuda_device = noa::cuda::Device(device.id(), Unchecked{});
            return noa::cuda::fft::cache_limit(cuda_device);
        }
        #else
        (void) device;
        #endif
        return -1;
    }

    auto Session::fft_cache_size(Device device) -> i64 {
        if (device.is_cpu())
            return noa::cpu::fft::cache_size();
        #ifdef NOA_ENABLE_CUDA
        auto cuda_device = noa::cuda::Device(device.id(), Unchecked{});
        return noa::cuda::fft::cache_size(cuda_device);
        #else
        return -1;
        #endif
    }

    auto Session::fft_workspace_left_to_allocate(Device device) -> size_t {
        #ifdef NOA_ENABLE_CUDA
        if (device.is_gpu()) {
            auto cuda_device = noa::cuda::Device(device.id(), Unchecked{});
            return noa::cuda::fft::workspace_left_to_allocate(cuda_device);
        }
        #else
        (void) device;
        #endif
        return 0;
    }

    void Session::clear_blas_cache(Device device) {
        if (device.is_cpu())
            return;
        #ifdef NOA_ENABLE_CUDA
        auto cuda_device = noa::cuda::Device(device.id(), Unchecked{});
        noa::cuda::cublas_clear_cache(cuda_device);
        #else
        (void) device;
        #endif
    }
}
