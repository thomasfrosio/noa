/**
 * @file noa/Define.h
 * @brief Some useful macros.
 * @author Thomas - ffyr2w
 * @date 20 Jul 2020
 */
#pragma once

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
