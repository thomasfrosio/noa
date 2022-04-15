#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/util/EwiseBinary.cuh"

namespace noa::cuda::math {
    template<typename T, typename U, typename V, typename BinaryOp, typename>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_stride, U rhs,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream) {
        cuda::util::ewise::binary("math::ewise",
                                  lhs.get(), lhs_stride, rhs,
                                  output.get(), output_stride,
                                  shape, stream, binary_op);
        stream.attach(lhs, output);
    }

    template<typename T, typename U, typename V, typename BinaryOp, typename>
    void ewise(T lhs, const shared_t<U[]>& rhs, size4_t rhs_stride,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream) {
        cuda::util::ewise::binary("math::ewise",
                                  lhs, rhs.get(), rhs_stride,
                                  output.get(), output_stride,
                                  shape, stream, binary_op);
        stream.attach(rhs, output);
    }

    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_stride,
               const shared_t<U[]>& rhs, size4_t rhs_stride,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream) {
        cuda::util::ewise::binary("math::ewise",
                                  lhs.get(), lhs_stride,
                                  rhs.get(), rhs_stride,
                                  output.get(), output_stride,
                                  shape, stream, binary_op);
        stream.attach(lhs, rhs, output);
    }

    #define NOA_INSTANTIATE_EWISE_BINARY(T,U,V,BINARY)                                                                                  \
    template void ewise<T,U,V,BINARY,void>(const shared_t<T[]>&, size4_t, U, const shared_t<V[]>&, size4_t, size4_t, BINARY, Stream&);  \
    template void ewise<T,U,V,BINARY,void>(T, const shared_t<U[]>&, size4_t, const shared_t<V[]>&, size4_t, size4_t, BINARY, Stream&);  \
    template void ewise<T,U,V,BINARY>(const shared_t<T[]>&, size4_t, const shared_t<U[]>&, size4_t, const shared_t<V[]>&, size4_t, size4_t, BINARY, Stream&)

    #define NOA_INSTANTIATE_EWISE_BINARY_SCALAR(T)                   \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::equal_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::not_equal_t);    \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::less_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::less_equal_t);   \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::greater_t);      \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::greater_equal_t)

    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(int16_t);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(int32_t);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(int64_t);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(uint16_t);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(uint32_t);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(uint64_t);

    #define NOA_INSTANTIATE_EWISE_BINARY_FLOAT(T)                    \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::equal_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::not_equal_t);    \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::less_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::less_equal_t);   \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::greater_t);      \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::greater_equal_t)

    NOA_INSTANTIATE_EWISE_BINARY_FLOAT(half_t);
    NOA_INSTANTIATE_EWISE_BINARY_FLOAT(float);
    NOA_INSTANTIATE_EWISE_BINARY_FLOAT(double);

    #define NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(T,V)       \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::math::equal_t);       \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::math::not_equal_t);   \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::math::less_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::math::less_equal_t);  \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::math::greater_t);     \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::math::greater_equal_t)

    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(half_t, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(float, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(double, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(int16_t, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(int32_t, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(int64_t, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(uint16_t, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(uint32_t, bool);
    NOA_INSTANTIATE_EWISE_BINARY_COMPARISON_CAST(uint64_t, bool);
}
