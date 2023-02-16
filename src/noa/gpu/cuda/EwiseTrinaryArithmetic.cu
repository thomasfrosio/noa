#include "noa/gpu/cuda/Ewise.hpp"
#include "noa/gpu/cuda/utils/EwiseTrinary.cuh"

namespace noa::cuda {
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::utils::ewise_trinary(
                "ewise_trinary",
                lhs, lhs_strides,
                mhs, mhs_strides,
                rhs, rhs_strides,
                output, output_strides,
                shape, stream, trinary_op);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise_trinary(Lhs lhs,
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::utils::ewise_trinary(
                "ewise_trinary",
                lhs,
                mhs, mhs_strides,
                rhs, rhs_strides,
                output, output_strides,
                shape, stream, trinary_op);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                       Mhs mhs,
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::utils::ewise_trinary(
                "ewise_trinary",
                lhs, lhs_strides,
                mhs,
                rhs, rhs_strides,
                output, output_strides,
                shape, stream, trinary_op);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,
                       Rhs rhs,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::utils::ewise_trinary(
                "ewise_trinary",
                lhs, lhs_strides,
                mhs, mhs_strides,
                rhs,
                output, output_strides,
                shape, stream, trinary_op);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,
                       Mhs mhs,
                       Rhs rhs,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::utils::ewise_trinary(
                "ewise_trinary",
                lhs, lhs_strides,
                mhs,
                rhs,
                output, output_strides,
                shape, stream, trinary_op);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise_trinary(Lhs lhs,
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,
                       Rhs rhs,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::utils::ewise_trinary(
                "ewise_trinary",
                lhs,
                mhs, mhs_strides,
                rhs,
                output, output_strides,
                shape, stream, trinary_op);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise_trinary(Lhs lhs,
                       Mhs mhs,
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,
                       Out* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::utils::ewise_trinary(
                "ewise_trinary",
                lhs,
                mhs,
                rhs, rhs_strides,
                output, output_strides,
                shape, stream, trinary_op);
    }

    #define NOA_INSTANTIATE_EWISE_TRINARY(T,U,V,W,O)    \
    template void ewise_trinary<T,U,V,W,O,void>(        \
        const T*, const Strides4<i64>&,                 \
        const U*, const Strides4<i64>&,                 \
        const V*, const Strides4<i64>&,                 \
        W*, const Strides4<i64>&,                       \
        const Shape4<i64>&, O, Stream&);                \
    template void ewise_trinary<T,U,V,W,O,void>(        \
        T,                                              \
        const U*, const Strides4<i64>&,                 \
        const V*, const Strides4<i64>&,                 \
        W*, const Strides4<i64>&,                       \
        const Shape4<i64>&, O, Stream&);                \
    template void ewise_trinary<T,U,V,W,O,void>(        \
        const T*, const Strides4<i64>&,                 \
        U,                                              \
        const V*, const Strides4<i64>&,                 \
        W*, const Strides4<i64>&,                       \
        const Shape4<i64>&, O, Stream&);                \
    template void ewise_trinary<T,U,V,W,O,void>(        \
        const T*, const Strides4<i64>&,                 \
        const U*, const Strides4<i64>&,                 \
        V,                                              \
        W*, const Strides4<i64>&,                       \
        const Shape4<i64>&, O, Stream&);                \
    template void ewise_trinary<T,U,V,W,O,void>(        \
        const T*, const Strides4<i64>&,                 \
        U,                                              \
        V,                                              \
        W*, const Strides4<i64>&,                       \
        const Shape4<i64>&, O, Stream&);                \
    template void ewise_trinary<T,U,V,W,O,void>(        \
        T,                                              \
        const U*, const Strides4<i64>&,                 \
        V,                                              \
        W*, const Strides4<i64>&,                       \
        const Shape4<i64>&, O, Stream&);                \
    template void ewise_trinary<T,U,V,W,O,void>(        \
        T,                                              \
        U,                                              \
        const V*, const Strides4<i64>&,                 \
        W*, const Strides4<i64>&,                       \
        const Shape4<i64>&, O, Stream&)

    #define NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(T)               \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::plus_t);             \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::plus_minus_t);       \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::plus_multiply_t);    \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::plus_divide_t);      \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::minus_t);            \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::minus_plus_t);       \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::minus_multiply_t);   \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::minus_divide_t);     \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::multiply_t);         \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::multiply_plus_t);    \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::multiply_minus_t);   \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::multiply_divide_t);  \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::divide_t);           \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::divide_plus_t);      \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::divide_minus_t);     \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::divide_multiply_t);  \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,T,::noa::divide_epsilon_t)

    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(f16);
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(f32);
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(f64);
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(i16);
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(i32);
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(i64);
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(u16);
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(u32);
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(u64);
}
