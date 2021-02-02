/**
 * @file Define.h
 * @brief Some useful macros.
 * @author Thomas - ffyr2w
 * @date 20 Jul 2020
 */
#pragma once

#ifdef NOA_DH
#undef NOA_DH
#endif

// If the compilation is not steered by nvcc, these attributes should not be used
// since the CUDA runtime might not be included in the translation unit.
#ifdef __CUDACC__
    #define NOA_DH __host__ __device__
#else
    #define NOA_DH
#endif
