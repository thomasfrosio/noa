#pragma once

#include "noa/Version.hpp"

// Platform detection.
#if defined(_WIN64) || defined(__APPLE__) || defined(__MACH__) || defined(__ANDROID__)
#   error "Platform is not supported"
#elif defined(__linux__) || defined(__CUDACC_RTC__)
#   define NOA_PLATFORM_LINUX
#else
#   error "Unknown platform!"
#endif

// Whether the code is compiled for device/GPU code.
#ifdef __CUDA_ARCH__
#   define NOA_IS_GPU_CODE
#else
#   define NOA_IS_CPU_CODE
#endif

// Device/Host declarations.
// If the compilation is not steered by nvcc/nvrtc, these attributes should not be used
// since the CUDA runtime might not be included in the translation unit.
#ifdef __CUDACC__
#   ifndef NOA_FD
#       define NOA_FD __forceinline__ __device__
#   endif
#   ifndef NOA_FH
#       define NOA_FH __forceinline__ __host__
#   endif
#   ifndef NOA_FHD
#       define NOA_FHD __forceinline__ __host__ __device__
#   endif
#   ifndef NOA_ID
#       define NOA_ID inline __device__
#   endif
#   ifndef NOA_IH
#       define NOA_IH inline __host__
#   endif
#   ifndef NOA_IHD
#       define NOA_IHD inline __host__ __device__
#   endif
#   ifndef NOA_HD
#       define NOA_HD __host__ __device__
#   endif
#   ifndef NOA_DEVICE
#       define NOA_DEVICE __device__
#   endif
#   ifndef NOA_HOST
#       define NOA_HOST __host__
#   endif
#else // __CUDACC__
#   ifndef NOA_FD
#       define NOA_FD inline
#   endif
#   ifndef NOA_FH
#       define NOA_FH inline
#   endif
#   ifndef NOA_FHD
#       define NOA_FHD inline
#   endif
#   ifndef NOA_ID
#       define NOA_ID inline
#   endif
#   ifndef NOA_IH
#       define NOA_IH inline
#   endif
#   ifndef NOA_IHD
#       define NOA_IHD inline
#   endif
#   ifndef NOA_HD
#       define NOA_HD
#   endif
#   ifndef NOA_DEVICE
#       define NOA_DEVICE
#   endif
#   ifndef NOA_HOST
#       define NOA_HOST
#   endif
#endif // __CUDACC__

// Detect host compiler
#if defined(__clang__)
#   define NOA_COMPILER_CLANG
#elif defined(__GNUG__)
#   define NOA_COMPILER_GCC
#elif defined(_MSC_VER)
#   define NOA_COMPILER_MSVC
#endif

// nvcc says it's support [[no_unique_address]], but about half the time it freaks out
// with a internal compiler error. In order to keep the layout consistent between host
// and device (which is necessary to pass objects to CUDA), turn of the attribute
// entirely for now.
#ifdef NOA_ENABLE_CUDA
#   define NOA_ENABLE_GPU
#   define NOA_NO_UNIQUE_ADDRESS
#else
#   define NOA_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

// nvcc warnings
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#   define NOA_PRAGMA_(value) _Pragma(#value)
#   define NOA_NV_DIAG_SUPPRESS(nb) NOA_PRAGMA_(nv_diag_suppress=nb)
#   define NOA_NV_DIAG_DEFAULT(nb) NOA_PRAGMA_(nv_diag_default=nb)
#else
#   define NOA_NV_DIAG_SUPPRESS(nb)
#   define NOA_NV_DIAG_DEFAULT(nb)
#endif

#ifdef __CUDACC__
#   define NOA_RESTRICT_ATTRIBUTE __restrict__
#else
#   define NOA_RESTRICT_ATTRIBUTE __restrict
#endif

// Assertions.
// CUDA device code supports the assert macro, but the code bloat is really not worth it.
#ifdef NOA_DEBUG
#   ifdef NOA_IS_GPU_CODE
#       define NOA_ASSERT(cond) (static_cast<bool>(cond) ? void(0) : __trap())
#   else
#       include <cassert>
#       define NOA_ASSERT(cond) assert(cond)
#   endif
#else
#   define NOA_ASSERT(cond)
#endif

// Error policy.
namespace noa::config {
    enum class error_policy_type { ABORT, TERMINATE, THROW };
    inline constexpr auto error_policy = static_cast<error_policy_type>(NOA_ERROR_POLICY);
}
#ifdef __has_builtin
#  if __has_builtin(__builtin_trap)
#    define NOA_HAS_BUILTIN_TRAP 1
#  endif
#endif

// GCC/Clang extension for static bounds checking.
#if defined(__has_cpp_attribute) && defined(__has_builtin)
#   if __has_builtin(__builtin_constant_p) && __has_cpp_attribute(gnu::error)
#       define NOA_HAS_GCC_STATIC_BOUNDS_CHECKING
#   endif
#endif
