#pragma once

#ifdef NOA_DH
#undef NOA_DH
#endif

#ifdef __CUDACC__
    #define NOA_DH __host__ __device__
#else
    #define NOA_DH
#endif
