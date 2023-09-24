#include "noa/gpu/cuda/Ewise.hpp"
#include "noa/gpu/cuda/utils/EwiseTrinary.cuh"

NOA_CUDA_EWISE_TRINARY_GENERATE_API

#define NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(T) \
    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,T,T,T,::noa::divide_t)
//    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,T,T,T,::noa::multiply_t)           \
//    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,T,T,T,::noa::multiply_plus_t)      \
//    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,T,T,T,::noa::multiply_minus_t)     \
//    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,T,T,T,::noa::multiply_divide_t)    \
//    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,T,T,T,::noa::divide_plus_t)        \
//    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,T,T,T,::noa::divide_minus_t)       \
//    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,T,T,T,::noa::divide_multiply_t)    \
//    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,T,T,T,::noa::divide_epsilon_t)

//NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(f16)
NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(f32)
//NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(f64)
//NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(i8)
//NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(i16)
//NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(i32)
//NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(i64)
//NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(u8)
//NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(u16)
//NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(u32)
//NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(u64)
