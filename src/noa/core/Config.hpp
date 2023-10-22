#pragma once

// --- Platform detection ---
#if defined(_WIN32)
    #if defined(_WIN64)
        #define NOA_PLATFORM_WINDOWS
        #error "Windows x64 is not supported yet!"
    #else // _WIN32
        #error "Windows x86 is not supported!"
    #endif
#elif defined(__APPLE__) || defined(__MACH__)
    #include <TargetConditionals.h>
    // TARGET_OS_MAC exists on all the platforms so we must check all of them (in this order)
    // to ensure that we're running on MAC and not some other Apple platform.
    #if TARGET_IPHONE_SIMULATOR == 1
        #error "IOS simulator is not supported!"
    #elif TARGET_OS_IPHONE == 1
        #error "IOS is not supported!"
    #elif TARGET_OS_MAC == 1
        #error "MacOS is not supported!"
    #else
        #error "Unknown Apple platform!"
    #endif
#elif defined(__ANDROID__)
    // We also have to check __ANDROID__ before __linux__ since android is based on the linux kernel
    // it has __linux__ defined
    #error "Android is not supported!"
#elif defined(__linux__)
    #define NOA_PLATFORM_LINUX
#else
    #error "Unknown platform!"
#endif

#if defined(__CUDA_ARCH__)
    // Code is compiled for the GPU (by nvcc, nvc++ or nvrtc)
    #define NOA_IS_GPU_CODE
#else
    // Code is compiled for the CPU
    #define NOA_IS_CPU_CODE
#endif

#if defined(__CUDACC_RTC__)
    // Code is JIT-compiled by nvrtc
    #define NOA_IS_JIT
#else
    // Code is compiled offline by a C++ compiler (which includes nvcc and nvc++)
    #define NOA_IS_OFFLINE
#endif

// --- Assertions ---
// CUDA device code supports the assert macro, but the code bloat is really not worth it.
#if defined(NOA_ENABLE_ASSERTS) && !defined(NOA_IS_GPU_CODE)
    #include <cassert>
    #define NOA_ASSERT(check) assert(check)
#else
    #define NOA_ASSERT(check)
#endif

// --- Device/Host declarations ---
// If the compilation is not steered by nvcc/nvrtc, these attributes should not be used
// since the CUDA runtime might not be included in the translation unit.
#if defined(__CUDACC__)
    #if !defined(NOA_FD)
        #define NOA_FD __forceinline__ __device__
    #endif
    #if !defined(NOA_FH)
        #define NOA_FH __forceinline__ __host__
    #endif
    #if !defined(NOA_FHD)
        #define NOA_FHD __forceinline__ __host__ __device__
    #endif
    #if !defined(NOA_ID)
        #define NOA_ID inline __device__
    #endif
    #if !defined(NOA_IH)
        #define NOA_IH inline __host__
    #endif
    #if !defined(NOA_IHD)
        #define NOA_IHD inline __host__ __device__
    #endif
    #if !defined(NOA_HD)
        #define NOA_HD __host__ __device__
    #endif
    #if !defined(NOA_DEVICE)
        #define NOA_DEVICE __device__
    #endif
    #if !defined(NOA_HOST)
        #define NOA_HOST __host__
    #endif
#else // __CUDACC__
    #if !defined(NOA_FD)
        #define NOA_FD inline
    #endif
    #if !defined(NOA_FH)
        #define NOA_FH inline
    #endif
    #if !defined(NOA_FHD)
        #define NOA_FHD inline
    #endif
    #if !defined(NOA_ID)
        #define NOA_ID inline
    #endif
    #if !defined(NOA_IH)
        #define NOA_IH inline
    #endif
    #if !defined(NOA_IHD)
        #define NOA_IHD inline
    #endif
    #if !defined(NOA_HD)
        #define NOA_HD
    #endif
    #if !defined(NOA_DEVICE)
        #define NOA_DEVICE
    #endif
    #if !defined(NOA_HOST)
        #define NOA_HOST
    #endif
#endif // __CUDACC__

// --- Debug break ---
#if defined(NOA_DEBUG)
    #if defined(NOA_PLATFORM_WINDOWS)
        #define NOA_DEBUG_BREAK() __debugbreak()
    #elif defined(NOA_PLATFORM_LINUX)
        #if !defined(__CUDACC__)
            #include <csignal>
            #define NOA_DEBUG_BREAK() raise(SIGTRAP)
        #else
            #define NOA_DEBUG_BREAK()
        #endif
    #else
        #error "Platform doesn't support debugbreak yet!"
    #endif
    #define NOA_ENABLE_ASSERTS
#else
    #define NOA_DEBUGBREAK()
#endif

// Detect host compiler
#if defined(__clang__)
#define NOA_COMPILER_CLANG
#elif defined(__GNUG__)
#define NOA_COMPILER_GCC
#elif defined(_MSC_VER)
#define NOA_COMPILER_MSVC
#endif

// Even in C++ 17, [[no_unique_address]] works for G++/Clang 9 or later.
// However, nvrtc doesn't support it and sometimes nvcc outputs internal errors because of it.
// So to keep the layouts the same between CPU and GPU, keep it to C++20.
#if __cplusplus >= 202002L
    #define NOA_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
    #define NOA_NO_UNIQUE_ADDRESS
#endif


