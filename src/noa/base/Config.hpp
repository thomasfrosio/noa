#pragma once

#include "noa/Version.hpp"

// Platform detection.
#if defined(_WIN64) || defined(__ANDROID__)
#   error "Platform is not supported"
#elif defined(__linux__) || defined(__CUDACC_RTC__)
#   define NOA_PLATFORM_LINUX
#elif defined(__APPLE__) || defined(__MACH__)
#   define NOA_PLATFORM_APPLE
#else
#   error "Unknown platform!"
#endif

// Device/Host declarations.
#if (defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)) && !defined(NOA_ENABLE_SYCL)
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
#else
#   ifndef NOA_FD
#       define NOA_FD inline __attribute__((always_inline))
#   endif
#   ifndef NOA_FH
#       define NOA_FH inline __attribute__((always_inline))
#   endif
#   ifndef NOA_FHD
#       define NOA_FHD inline __attribute__((always_inline))
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
#endif

// Detect host compiler
#if defined(__clang__)
#   define NOA_COMPILER_CLANG
#elif defined(__GNUG__)
#   define NOA_COMPILER_GCC
#endif

// nvcc says it support [[no_unique_address]], but about half the time it freaks out
// with a internal compiler error. In order to keep the layout consistent between host
// and device (which is necessary to pass objects to CUDA), turn of the attribute
// entirely for now.
#if defined(NOA_ENABLE_CUDA) && defined(__NVCC__)
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

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#   define NOA_RESTRICT_ATTRIBUTE __restrict__
#else
#   define NOA_RESTRICT_ATTRIBUTE __restrict
#endif

// Assertions.
#ifdef NOA_DEBUG
#   include <cassert>
#   if defined(NOA_ENABLE_CUDA)
#       if defined(__NVCOMPILER) // nvc++
#           include <nv/target>
#           include <type_traits>
#           define NOA_ASSERT(cond) \
                do { \
                    if (std::is_constant_evaluated()) { assert(cond); } \
                    else { NV_IF_TARGET(NV_IS_DEVICE, (if (!static_cast<bool>(cond)) __trap();), (assert(cond);)) } \
                } while (false)
#       elif defined(__CUDA_ARCH__) // nvcc and clang-cuda, __CUDA_ARCH__ is device only.
#           define NOA_ASSERT(cond) (static_cast<bool>(cond) ? void(0) : __trap())
#       else
#           define NOA_ASSERT(cond) assert(cond)
#       endif
#   elif defined(NOA_ENABLE_HIP)
#       ifdef __HIP_DEVICE_COMPILE__
#           define NOA_ASSERT(cond) (static_cast<bool>(cond) ? void(0) : __builtin_trap())
#       else
#           define NOA_ASSERT(cond) assert(cond)
#       endif
#   elif defined(NOA_ENABLE_SYCL)
#       include <sycl/sycl.hpp>
#       if defined(__ACPP__) || defined(__HIPSYCL__)
#           include <type_traits>
#           define NOA_ASSERT(cond) \
                do { \
                    if (std::is_constant_evaluated()) { assert(cond); } \
                    else { __acpp_if_target_host(assert(cond);) __acpp_if_target_device((void)0;) } \
                } while (false)
#       else // DPC++
#           ifdef __SYCL_DEVICE_ONLY__
#               if defined(__NVPTX__) || defined(__AMDGCN__)
#                   define NOA_ASSERT(cond) (static_cast<bool>(cond) ? void(0) : __builtin_trap())
#               else
#                   define NOA_ASSERT(cond) ((void)0)
#               endif
#           else
#               define NOA_ASSERT(cond) assert(cond)
#           endif
#       endif
#   else // CPU-only
#       define NOA_ASSERT(cond) assert(cond)
#   endif
#else
#   define NOA_ASSERT(cond)
#endif

// Emit arguments only when compiling for a device or host target.
#if defined(NOA_ENABLE_SYCL)
#   if defined(__ACPP__) || defined(__HIPSYCL__)
#       include <sycl/sycl.hpp>
#       include <type_traits>
#       define NOA_IF_TARGET_DEVICE(...) if (!std::is_constant_evaluated()) { __acpp_if_target_device(__VA_ARGS__) }
#       define NOA_IF_TARGET_HOST(...) if (std::is_constant_evaluated()) { __VA_ARGS__ } else { __acpp_if_target_host(__VA_ARGS__) }
#   else // DPC++ (separate device pass)
#       ifdef __SYCL_DEVICE_ONLY__
#           define NOA_IF_TARGET_DEVICE(...) __VA_ARGS__
#           define NOA_IF_TARGET_HOST(...)
#       else
#           define NOA_IF_TARGET_DEVICE(...)
#           define NOA_IF_TARGET_HOST(...) __VA_ARGS__
#       endif
#   endif

#elif defined(NOA_ENABLE_CUDA)
#   if defined(__NVCOMPILER) // nvc++ cuda (single pass)
#       include <nv/target>
#       include <type_traits>
#       define NOA_IF_TARGET_DEVICE(...) if (!std::is_constant_evaluated()) { NV_IF_TARGET(NV_IS_DEVICE, (__VA_ARGS__)) }
#       define NOA_IF_TARGET_HOST(...) if (std::is_constant_evaluated()) { __VA_ARGS__ } else { NV_IF_TARGET(NV_IS_HOST, (__VA_ARGS__)) }
#   elif defined(__CUDA_ARCH__) // nvcc and clang-cuda (separate passes, __CUDA_ARCH__ is device only)
#       define NOA_IF_TARGET_DEVICE(...) __VA_ARGS__
#       define NOA_IF_TARGET_HOST(...)
#   else
#       define NOA_IF_TARGET_DEVICE(...)
#       define NOA_IF_TARGET_HOST(...) __VA_ARGS__
#   endif

#elif defined(NOA_ENABLE_HIP)
#   ifdef __HIP_DEVICE_COMPILE__ // hipcc (clang based, separate passes, __HIP_DEVICE_COMPILE__ is device only)
#       define NOA_IF_TARGET_DEVICE(...) __VA_ARGS__
#       define NOA_IF_TARGET_HOST(...)
#   else
#       define NOA_IF_TARGET_DEVICE(...)
#       define NOA_IF_TARGET_HOST(...) __VA_ARGS__
#   endif

#else // CPU-only
#   define NOA_IF_TARGET_DEVICE(...)
#   define NOA_IF_TARGET_HOST(...) __VA_ARGS__
#endif

#if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
#   if defined(__NVCOMPILER)
#       include <nv/target>
#       define NOA_IF_TARGET_SM53(hw, sw) NV_IF_TARGET(NV_PROVIDES_SM_53, hw, sw)
#       define NOA_IF_TARGET_SM60(hw, sw) NV_IF_TARGET(NV_PROVIDES_SM_60, hw, sw)
#       define NOA_IF_TARGET_SM70(hw, sw) NV_IF_TARGET(NV_PROVIDES_SM_70, hw, sw)
#       define NOA_IF_TARGET_SM80(hw, sw) NV_IF_TARGET(NV_PROVIDES_SM_80, hw, sw)
#   else // nvcc, clang-cuda, hip
#       define NOA_F16_UNPAREN(...) __VA_ARGS__
#       if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
#           define NOA_IF_TARGET_SM53(hw, sw) NOA_F16_UNPAREN hw
#       else
#           define NOA_IF_TARGET_SM53(hw, sw) NOA_F16_UNPAREN sw
#       endif
#       if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
#           define NOA_IF_TARGET_SM60(hw, sw) NOA_F16_UNPAREN hw
#       else
#           define NOA_IF_TARGET_SM60(hw, sw) NOA_F16_UNPAREN sw
#       endif
#       if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#           define NOA_IF_TARGET_SM70(hw, sw) NOA_F16_UNPAREN hw
#       else
#           define NOA_IF_TARGET_SM70(hw, sw) NOA_F16_UNPAREN sw
#       endif
#       if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#           define NOA_IF_TARGET_SM80(hw, sw) NOA_F16_UNPAREN hw
#       else
#           define NOA_IF_TARGET_SM80(hw, sw) NOA_F16_UNPAREN sw
#       endif
#   endif
#else // SYCL, CPU-only
#   define NOA_F16_UNPAREN(...) __VA_ARGS__
#   define NOA_IF_TARGET_SM53(hw, sw) NOA_F16_UNPAREN sw
#   define NOA_IF_TARGET_SM60(hw, sw) NOA_F16_UNPAREN sw
#   define NOA_IF_TARGET_SM70(hw, sw) NOA_F16_UNPAREN sw
#   define NOA_IF_TARGET_SM80(hw, sw) NOA_F16_UNPAREN sw
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

#define NOA_NOINLINE [[gnu::noinline]]
