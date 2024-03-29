#ifdef NOA_ENABLE_OPENMP
    #include <omp.h>
#endif

#include "noa/cpu/fft/Plan.hpp"
#ifdef NOA_ENABLE_CUDA
    #include <cstdlib>
    #include "cuda.h"
    #include "noa/gpu/cuda/fft/Plan.hpp"
    #include "noa/gpu/cuda/math/Blas.hpp"
#endif

#include "noa/Session.hpp"
#include "noa/core/string/Parse.hpp"

noa::i64 noa::Session::m_thread_limit = 0;

namespace noa {
    void Session::set_thread_limit(i64 n_threads) {
        if (n_threads > 0) {
            m_thread_limit = n_threads;
        } else {
            i64 max_threads{};
            const char* str{};
            try {
                str = std::getenv("NOA_THREADS");
                if (str) {
                    max_threads = noa::string::parse<i64>(str);
                } else {
                    #ifdef NOA_ENABLE_OPENMP
                    str = std::getenv("OMP_NUM_THREADS");
                    if (str)
                        max_threads = noa::string::parse<i64>(str);
                    else
                        max_threads = static_cast<i64>(omp_get_max_threads());
                    #else
                    max_threads = std::thread::hardware_concurrency();
                    #endif
                }
            } catch (...) {
                max_threads = 1;
            }
            m_thread_limit = std::max(max_threads, i64{1});
        }
    }

    bool Session::set_cuda_lazy_loading() {
        #if defined(NOA_ENABLE_CUDA) && CUDART_VERSION >= 11070
        // It seems that CUDA_MODULE_LOADING is only read once during cuInit().
        // Try lazy loading if not explicitly set already.
        [[maybe_unused]] const auto err = setenv("CUDA_MODULE_LOADING", "LAZY", /*replace=*/false);

        // Check whether lazy mode is already enabled.
        CUmoduleLoadingMode mode;
        NOA_CHECK(cuInit(0) == CUDA_SUCCESS && cuModuleGetLoadingMode(&mode) == CUDA_SUCCESS,
                  "Failed to initialize and query the CUDA driver");
        return mode == CU_MODULE_LAZY_LOADING;
        #else
        return false;
        #endif
    }

    i64 Session::clear_fft_cache(Device device) {
        if (device.is_cpu())
            return noa::cpu::fft::fftw_clear_cache();
        #ifdef NOA_ENABLE_CUDA
        return noa::cuda::fft::cufft_clear_cache(device.id());
        #else
        return 0;
        #endif
    }

    void Session::set_fft_cache_limit(i64 count, Device device) {
        if (device.is_cpu())
            return; // TODO we could have a more flexible caching mechanism for FFTW
        #ifdef NOA_ENABLE_CUDA
        noa::cuda::fft::cufft_cache_limit(clamp_cast<i32>(count), device.id());
        #endif
    }

    void Session::clear_cublas_cache(Device device) {
        #ifdef NOA_ENABLE_CUDA
        if (device.is_cpu())
            return;
        noa::cuda::math::cublas_clear_cache(device.id());
        #endif
    }
}

