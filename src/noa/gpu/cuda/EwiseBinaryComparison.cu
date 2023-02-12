#include "noa/gpu/cuda/Ewise.h"
#include "noa/gpu/cuda/utils/EwiseBinary.cuh"

namespace noa::cuda {
    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp, typename>
    void ewise_binary(const Shared<Lhs[]>& lhs, const Strides4<i64>& lhs_strides,
                      const Shared<Rhs[]>& rhs, const Strides4<i64>& rhs_strides,
                      const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp binary_op, Stream& stream) {
        cuda::utils::ewise_binary(
                "ewise_binary",
                lhs.get(), lhs_strides,
                rhs.get(), rhs_strides,
                output.get(), output_strides,
                shape, stream, binary_op);
        stream.attach(lhs, rhs, output);
    }

    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp, typename>
    void ewise_binary(const Shared<Lhs[]>& lhs, const Strides4<i64>& lhs_strides,
                      Rhs rhs,
                      const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp binary_op, Stream& stream) {
        cuda::utils::ewise_binary(
                "ewise_binary",
                lhs.get(), lhs_strides,
                rhs,
                output.get(), output_strides,
                shape, stream, binary_op);
        stream.attach(lhs, output);
    }

    template<typename Lhs, typename Rhs, typename Out, typename BinaryOp, typename>
    void ewise_binary(Lhs lhs,
                      const Shared<Rhs[]>& rhs, const Strides4<i64>& rhs_strides,
                      const Shared<Out[]>& output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, BinaryOp binary_op, Stream& stream) {
        cuda::utils::ewise_binary(
                "ewise_binary",
                lhs,
                rhs.get(), rhs_strides,
                output.get(), output_strides,
                shape, stream, binary_op);
        stream.attach(rhs, output);
    }

    #define NOA_INSTANTIATE_EWISE_BINARY(T,U,V,BINARY)  \
    template void ewise_binary<T,U,V,BINARY,void>(      \
        const Shared<T[]>&, const Strides4<i64>&,       \
        const Shared<U[]>&, const Strides4<i64>&,       \
        const Shared<V[]>&, const Strides4<i64>&,       \
        const Shape4<i64>&, BINARY, Stream&);           \
    template void ewise_binary<T,U,V,BINARY,void>(      \
        const Shared<T[]>&, const Strides4<i64>&,       \
        U,                                              \
        const Shared<V[]>&, const Strides4<i64>&,       \
        const Shape4<i64>&, BINARY, Stream&);           \
    template void ewise_binary<T,U,V,BINARY,void>(      \
        T,                                              \
        const Shared<U[]>&, const Strides4<i64>&,       \
        const Shared<V[]>&, const Strides4<i64>&,       \
        const Shape4<i64>&, BINARY, Stream&);           \

    #define NOA_INSTANTIATE_EWISE_BINARY_SCALAR(T)              \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::equal_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::not_equal_t);     \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::less_t);          \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::less_equal_t);    \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::greater_t);       \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::greater_equal_t)

    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(i16);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(i32);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(i64);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(u16);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(u32);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(u64);

    #define NOA_INSTANTIATE_EWISE_BINARY_FLOAT(T)               \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::equal_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::not_equal_t);     \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::less_t);          \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::less_equal_t);    \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::greater_t);       \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::greater_equal_t)

    NOA_INSTANTIATE_EWISE_BINARY_FLOAT(f16);
    NOA_INSTANTIATE_EWISE_BINARY_FLOAT(f32);
    NOA_INSTANTIATE_EWISE_BINARY_FLOAT(f64);

    #define NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(T,V)   \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::equal_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::not_equal_t);     \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::less_t);          \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::less_equal_t);    \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::greater_t);       \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::greater_equal_t)

    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(f16, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(f32, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(f64, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(i16, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(i32, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(i64, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(u16, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(u32, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(u64, bool);
}
