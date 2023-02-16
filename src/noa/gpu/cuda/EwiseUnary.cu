#include "noa/gpu/cuda/Ewise.hpp"
#include "noa/gpu/cuda/utils/EwiseUnary.cuh"

namespace noa::cuda {
    template<typename In, typename Out, typename UnaryOp, typename>
    void ewise_unary(const In* input, const Strides4<i64>& input_strides,
                     Out* output, const Strides4<i64>& output_strides,
                     const Shape4<i64>& shape, UnaryOp unary_op, Stream& stream) {
        cuda::utils::ewise_unary(
                "math::ewise",
                input, input_strides,
                output, output_strides,
                shape, stream, unary_op);
    }

    #define NOA_INSTANTIATE_EWISE_UNARY(T,U,UNARY)  \
    template void ewise_unary<T,U,UNARY,void>(      \
        const T*, const Strides4<i64>&,             \
        U*, const Strides4<i64>&,                   \
        const Shape4<i64>&, UNARY, Stream&)

    #define NOA_INSTANTIATE_EWISE_UNARY_INT(T,U)        \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::copy_t);     \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::square_t);   \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::abs_t);      \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::negate_t);   \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::one_minus_t)

    NOA_INSTANTIATE_EWISE_UNARY_INT(i16, i16);
    NOA_INSTANTIATE_EWISE_UNARY_INT(i32, i32);
    NOA_INSTANTIATE_EWISE_UNARY_INT(i64, i64);

    #define NOA_INSTANTIATE_EWISE_UNARY_UINT(T,U)   \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::copy_t); \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::square_t)

    NOA_INSTANTIATE_EWISE_UNARY_UINT(u16, u16);
    NOA_INSTANTIATE_EWISE_UNARY_UINT(u32, u32);
    NOA_INSTANTIATE_EWISE_UNARY_UINT(u64, u64);

    #define NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(T,U)    \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::nonzero_t);  \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::logical_not_t)

    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(i16, i16);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(i32, i32);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(i64, i64);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(u16, u16);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(u32, u32);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(u64, u64);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(i16, bool);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(i32, bool);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(i64, bool);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(u16, bool);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(u32, bool);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(u64, bool);

    #define NOA_INSTANTIATE_EWISE_UNARY_FLOAT(T)            \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::copy_t);         \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::square_t);       \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::abs_t);          \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::negate_t);       \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::one_minus_t);    \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::inverse_t);      \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::sqrt_t);         \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::rsqrt_t);        \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::exp_t);          \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::log_t);          \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::cos_t);          \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::sin_t);          \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::round_t);        \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::rint_t);         \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::ceil_t);         \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::floor_t);        \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::trunc_t);        \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::one_log_t);      \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::abs_one_log_t)

    NOA_INSTANTIATE_EWISE_UNARY_FLOAT(f16);
    NOA_INSTANTIATE_EWISE_UNARY_FLOAT(f32);
    NOA_INSTANTIATE_EWISE_UNARY_FLOAT(f64);

    #define NOA_INSTANTIATE_EWISE_UNARY_COMPLEX(C,R)        \
    NOA_INSTANTIATE_EWISE_UNARY(C,C,::noa::one_minus_t);    \
    NOA_INSTANTIATE_EWISE_UNARY(C,C,::noa::square_t);       \
    NOA_INSTANTIATE_EWISE_UNARY(C,C,::noa::inverse_t);      \
    NOA_INSTANTIATE_EWISE_UNARY(C,C,::noa::normalize_t);    \
    NOA_INSTANTIATE_EWISE_UNARY(C,C,::noa::conj_t);         \
    NOA_INSTANTIATE_EWISE_UNARY(C,R,::noa::abs_t);          \
    NOA_INSTANTIATE_EWISE_UNARY(C,R,::noa::real_t);         \
    NOA_INSTANTIATE_EWISE_UNARY(C,R,::noa::imag_t);         \
    NOA_INSTANTIATE_EWISE_UNARY(C,R,::noa::abs_squared_t);  \
    NOA_INSTANTIATE_EWISE_UNARY(C,R,::noa::abs_one_log_t)

    NOA_INSTANTIATE_EWISE_UNARY_COMPLEX(c16, f16);
    NOA_INSTANTIATE_EWISE_UNARY_COMPLEX(c32, f32);
    NOA_INSTANTIATE_EWISE_UNARY_COMPLEX(c64, f64);
}
