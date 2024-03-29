#pragma once

// --- Platform detection ----------------------------------------------------------------------------------------------

#ifdef _WIN32
    #ifdef _WIN64
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

// --- Device/Host declarations ----------------------------------------------------------------------------------------

// If the compilation is not steered by nvcc, these attributes should not be used
// since the CUDA runtime might not be included in the translation unit.
#ifdef __CUDACC__
    #ifndef NOA_FD
        #define NOA_FD __forceinline__ __device__
    #endif
    #ifndef NOA_FH
        #define NOA_FH __forceinline__ __host__
    #endif
    #ifndef NOA_FHD
        #define NOA_FHD __forceinline__ __host__ __device__
    #endif
    #ifndef NOA_ID
        #define NOA_ID inline __device__
    #endif
    #ifndef NOA_IH
        #define NOA_IH inline __host__
    #endif
    #ifndef NOA_IHD
        #define NOA_IHD inline __host__ __device__
    #endif
    #ifndef NOA_HD
        #define NOA_HD __host__ __device__
    #endif
    #ifndef NOA_DEVICE
        #define NOA_DEVICE __device__
    #endif
    #ifndef NOA_HOST
        #define NOA_HOST __host__
    #endif
#else // __CUDACC__
    #ifndef NOA_FD
        #define NOA_FD inline
    #endif
    #ifndef NOA_FH
        #define NOA_FH inline
    #endif
    #ifndef NOA_FHD
        #define NOA_FHD inline
    #endif
    #ifndef NOA_ID
        #define NOA_ID inline
    #endif
    #ifndef NOA_IH
        #define NOA_IH inline
    #endif
    #ifndef NOA_IHD
        #define NOA_IHD inline
    #endif
    #ifndef NOA_HD
        #define NOA_HD
    #endif
    #ifndef NOA_DEVICE
        #define NOA_DEVICE
    #endif
    #ifndef NOA_HOST
        #define NOA_HOST
    #endif
#endif // __CUDACC__

// --- Debug break -----------------------------------------------------------------------------------------------------

#if defined(NOA_DEBUG)
    #ifdef NOA_PLATFORM_WINDOWS
        #define NOA_DEBUG_BREAK() __debugbreak()
    #elif defined(NOA_PLATFORM_LINUX)
        #include <csignal>
        #define NOA_DEBUG_BREAK() raise(SIGTRAP)
    #else
        #error "Platform doesn't support debugbreak yet!"
    #endif
    #define NOA_ENABLE_ASSERTS
#else
    #define NOA_DEBUGBREAK()
#endif

// --- Detect host compiler --------------------------------------------------------------------------------------------
// Even in C++ 17, [[no_unique_address]] works for G++/Clang 9 or later.

#if defined(__clang__)
    #define NOA_COMPILER_CLANG
    #if __clang_major__ > 9
        #define NOA_NO_UNIQUE_ADDRESS [[no_unique_address]]
    #else
        #define NOA_NO_UNIQUE_ADDRESS
    #endif
#elif defined(__GNUG__)
    #define NOA_COMPILER_GCC
    #if __GNUC__ > 9
    #define NOA_NO_UNIQUE_ADDRESS [[no_unique_address]]
    #else
    #define NOA_NO_UNIQUE_ADDRESS
    #endif
#elif defined(_MSC_VER)
    #define NOA_COMPILER_MSVC
    #define NOA_NO_UNIQUE_ADDRESS
#else
    #error "C++ compiler isn't recognized"
#endif
