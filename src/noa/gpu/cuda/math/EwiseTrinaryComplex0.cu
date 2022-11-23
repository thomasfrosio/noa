#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/utils/EwiseTrinary.cuh"

namespace noa::cuda::math {
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides, Mhs mhs, Rhs rhs,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::utils::ewise::trinary(
                "math::ewise",
                lhs.get(), lhs_strides, mhs, rhs,
                output.get(), output_strides,
                shape, true, stream, trinary_op);
        stream.attach(lhs, output);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               const shared_t<Mhs[]>& mhs, dim4_t mhs_strides,
               Rhs rhs,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::utils::ewise::trinary(
                "math::ewise",
                lhs.get(), lhs_strides,
                mhs.get(), mhs_strides,
                rhs,
                output.get(), output_strides,
                shape, true, stream, trinary_op);
        stream.attach(lhs, mhs, output);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               Mhs mhs,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::utils::ewise::trinary(
                "math::ewise",
                lhs.get(), lhs_strides,
                mhs,
                rhs.get(), rhs_strides,
                output.get(), output_strides,
                shape, true, stream, trinary_op);
        stream.attach(lhs, rhs, output);
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>
    void ewise(const shared_t<Lhs[]>& lhs, dim4_t lhs_strides,
               const shared_t<Mhs[]>& mhs, dim4_t mhs_strides,
               const shared_t<Rhs[]>& rhs, dim4_t rhs_strides,
               const shared_t<Out[]>& output, dim4_t output_strides,
               dim4_t shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::utils::ewise::trinary(
                "math::ewise",
                lhs.get(), lhs_strides,
                mhs.get(), mhs_strides,
                rhs.get(), rhs_strides,
                output.get(), output_strides,
                shape, true, stream, trinary_op);
        stream.attach(lhs, mhs, rhs, output);
    }

    #define NOA_INSTANTIATE_EWISE_TRINARY(T,U,V,W,O)        \
    template void ewise<T,U,V,W,O,void>(                    \
        const shared_t<T[]>&, dim4_t, U, V,                 \
        const shared_t<W[]>&, dim4_t, dim4_t, O, Stream&);  \
    template void ewise<T,U,V,W,O,void>(                    \
        const shared_t<T[]>&, dim4_t,                       \
        const shared_t<U[]>&, dim4_t, V,                    \
        const shared_t<W[]>&, dim4_t, dim4_t, O, Stream&);  \
    template void ewise<T,U,V,W,O,void>(                    \
        const shared_t<T[]>&, dim4_t, U,                    \
        const shared_t<V[]>&, dim4_t,                       \
        const shared_t<W[]>&, dim4_t, dim4_t, O, Stream&);  \
    template void ewise<T,U,V,W,O,void>(                    \
        const shared_t<T[]>&, dim4_t,                       \
        const shared_t<U[]>&, dim4_t,                       \
        const shared_t<V[]>&, dim4_t,                       \
        const shared_t<W[]>&, dim4_t, dim4_t, O, Stream&)

    #define NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(T,U,V,W)               \
    NOA_INSTANTIATE_EWISE_TRINARY(T,U,V,W,::noa::math::plus_t);             \
    NOA_INSTANTIATE_EWISE_TRINARY(T,U,V,W,::noa::math::plus_minus_t);       \
    NOA_INSTANTIATE_EWISE_TRINARY(T,U,V,W,::noa::math::plus_multiply_t);    \
    NOA_INSTANTIATE_EWISE_TRINARY(T,U,V,W,::noa::math::plus_divide_t);      \
    NOA_INSTANTIATE_EWISE_TRINARY(T,U,V,W,::noa::math::minus_t);            \
    NOA_INSTANTIATE_EWISE_TRINARY(T,U,V,W,::noa::math::minus_plus_t);       \
    NOA_INSTANTIATE_EWISE_TRINARY(T,U,V,W,::noa::math::minus_multiply_t);   \
    NOA_INSTANTIATE_EWISE_TRINARY(T,U,V,W,::noa::math::minus_divide_t)

    #define NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC_ALL(R, C)  \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(C, R, R, C);       \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(R, C, R, C);       \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(R, R, C, C);       \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(R, C, C, C);       \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(C, R, C, C);       \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(C, C, R, C);       \
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC(C, C, C, C);

    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC_ALL(half_t, chalf_t);
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC_ALL(float, cfloat_t);
    NOA_INSTANTIATE_EWISE_TRINARY_ARITHMETIC_ALL(double, cdouble_t);
}
