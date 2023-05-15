#include "noa/gpu/cuda/Ewise.hpp"
#include "noa/gpu/cuda/utils/EwiseTrinary.cuh"

NOA_CUDA_EWISE_TRINARY_GENERATE_API

#define NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(T, W)                      \
    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,T,T,W,::noa::within_t)     \
    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,T,T,W,::noa::within_equal_t)

NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(i8, i8)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(i16, i16)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(i32, i32)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(i64, i64)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(u8, u8)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(u16, u16)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(u32, u32)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(u64, u64)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(f16, f16)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(f32, f32)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(f64, f64)

NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(i8, bool)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(i16, bool)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(i32, bool)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(i64, bool)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(u8, bool)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(u16, bool)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(u32, bool)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(u64, bool)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(f16, bool)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(f32, bool)
NOA_INSTANTIATE_EWISE_TRINARY_WITHIN(f64, bool)

#define NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(T) \
    NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(T,T,T,T,::noa::clamp_t)

NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(i8)
NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(i16)
NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(i32)
NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(i64)
NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(u8)
NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(u16)
NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(u32)
NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(u64)
NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(f16)
NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(f32)
NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(f64)
