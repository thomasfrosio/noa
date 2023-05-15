#include "noa/gpu/cuda/Ewise.hpp"
#include "noa/gpu/cuda/utils/EwiseTrinary.cuh"

NOA_CUDA_EWISE_TRINARY_GENERATE_API

#define NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(T, U, V, W)                \
    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,U,V,W,::noa::plus_t)           \
    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,U,V,W,::noa::plus_multiply_t)  \
    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,U,V,W,::noa::plus_minus_t)     \
    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,U,V,W,::noa::plus_divide_t)

#define NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC_ALL(R, C)  \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(C, R, R, C)    \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(R, C, R, C)    \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(R, R, C, C)    \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(R, C, C, C)    \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(C, R, C, C)    \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(C, C, R, C)    \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(C, C, C, C)

NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC_ALL(f16, c16)
NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC_ALL(f32, c32)
NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC_ALL(f64, c64)
