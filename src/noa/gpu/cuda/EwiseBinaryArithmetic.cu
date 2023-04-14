#include "noa/gpu/cuda/Ewise.hpp"
#include "noa/gpu/cuda/utils/EwiseBinary.cuh"

namespace noa::cuda {
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp, typename>
    void ewise_binary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                      const Rhs* rhs, const Strides4<i64>& rhs_strides,
                      Out* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp binary_op, Stream& stream) {
        cuda::utils::ewise_binary(
                "ewise_binary",
                lhs, lhs_strides,
                rhs, rhs_strides,
                output, output_strides,
                shape, stream, binary_op);
    }

    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp, typename>
    void ewise_binary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                      Rhs rhs,
                      Out* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp binary_op, Stream& stream) {
        cuda::utils::ewise_binary(
                "ewise_binary",
                lhs, lhs_strides,
                rhs,
                output, output_strides,
                shape, stream, binary_op);
    }

    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp, typename>
    void ewise_binary(Lhs lhs,
                      const Rhs* rhs, const Strides4<i64>& rhs_strides,
                      Out* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp binary_op, Stream& stream) {
        cuda::utils::ewise_binary(
                "ewise_binary",
                lhs,
                rhs, rhs_strides,
                output, output_strides,
                shape, stream, binary_op);
    }

    #define NOA_INSTANTIATE_EWISE_BINARY(T,U,V,BINARY)  \
    template void ewise_binary<T,U,V,BINARY,void>(      \
        const T*, const Strides4<i64>&,                 \
        const U*, const Strides4<i64>&,                 \
        V*, const Strides4<i64>&,                       \
        const Shape4<i64>&, BINARY, Stream&);           \
    template void ewise_binary<T,U,V,BINARY,void>(      \
        const T*, const Strides4<i64>&,                 \
        U,                                              \
        V*, const Strides4<i64>&,                       \
        const Shape4<i64>&, BINARY, Stream&);           \
    template void ewise_binary<T,U,V,BINARY,void>(      \
        T,                                              \
        const U*, const Strides4<i64>&,                 \
        V*, const Strides4<i64>&,                       \
        const Shape4<i64>&, BINARY, Stream&)

    #define NOA_INSTANTIATE_EWISE_BINARY_SCALAR(T)              \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::plus_t);          \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::minus_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::multiply_t);      \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::divide_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::divide_safe_t);   \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::dist2_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::min_t);           \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::max_t);           \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::modulo_t)

    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(i8);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(i16);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(i32);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(i64);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(u8);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(u16);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(u32);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(u64);

    #define NOA_INSTANTIATE_EWISE_BINARY_FLOAT(T)               \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::plus_t);          \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::minus_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::multiply_t);      \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::divide_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::divide_safe_t);   \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::dist2_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::min_t);           \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::max_t);           \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::pow_t)

    NOA_INSTANTIATE_EWISE_BINARY_FLOAT(f16);
    NOA_INSTANTIATE_EWISE_BINARY_FLOAT(f32);
    NOA_INSTANTIATE_EWISE_BINARY_FLOAT(f64);

    #define NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(T,U,V)         \
    NOA_INSTANTIATE_EWISE_BINARY(T,U,V,::noa::plus_t);          \
    NOA_INSTANTIATE_EWISE_BINARY(T,U,V,::noa::minus_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,U,V,::noa::multiply_t);      \
    NOA_INSTANTIATE_EWISE_BINARY(T,U,V,::noa::divide_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,U,V,::noa::divide_safe_t);   \
    NOA_INSTANTIATE_EWISE_BINARY(T,U,V,::noa::dist2_t)

    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(c16, c16, c16);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(c32, c32, c32);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(c64, c64, c64);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(c16, f16, c16);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(c32, f32, c32);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(c64, f64, c64);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(f16, c16, c16);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(f32, c32, c32);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(f64, c64, c64);

    NOA_INSTANTIATE_EWISE_BINARY(c16, c16, c16, ::noa::multiply_conj_t);
    NOA_INSTANTIATE_EWISE_BINARY(c32, c32, c32, ::noa::multiply_conj_t);
    NOA_INSTANTIATE_EWISE_BINARY(c64, c64, c64, ::noa::multiply_conj_t);

    #define NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(T,V)           \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::logical_and_t);   \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::logical_or_t)

    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(i8, i8);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(i16, i16);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(i32, i32);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(i64, i64);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(u8, u8);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(u16, u16);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(u32, u32);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(u64, u64);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(i8, bool);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(i16, bool);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(i32, bool);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(i64, bool);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(u8, bool);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(u16, bool);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(u32, bool);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(u64, bool);
}
