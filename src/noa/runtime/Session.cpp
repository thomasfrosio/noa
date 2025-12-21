#ifdef NOA_ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef NOA_ENABLE_CUDA
#include <cstdlib>
#include <cuda.h>
#include "noa/runtime/cuda/Blas.hpp"
#endif

#include "noa/base/Strings.hpp"
#include "noa/runtime/Session.hpp"

noa::i32 noa::Session::m_thread_limit = 0;

namespace noa::inline types {
    void Session::set_thread_limit(i32 n_threads) {
        if (n_threads > 0) {
            m_thread_limit = n_threads;
        } else {
            i32 max_threads;
            const char* str = std::getenv("NOA_THREADS");
            if (str) {
                max_threads = nd::parse<i32>(str).value_or(1);
            } else {
                #ifdef NOA_ENABLE_OPENMP
                str = std::getenv("OMP_NUM_THREADS");
                if (str)
                    max_threads = nd::parse<i32>(str).value_or(1);
                else
                    max_threads = static_cast<i32>(omp_get_max_threads());
                #else
                max_threads = std::thread::hardware_concurrency();
                #endif
            }
            m_thread_limit = std::max(max_threads, i32{1});
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
