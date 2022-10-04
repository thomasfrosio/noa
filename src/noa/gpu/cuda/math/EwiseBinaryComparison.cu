#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/util/EwiseBinary.cuh"

namespace noa::cuda::math {
    template<typename T, typename U, typename V, typename BinaryOp,
             std::enable_if_t<details::is_valid_ewise_binary_v<T, U, V, BinaryOp>, bool>>
    void ewise(const shared_t<T[]>& lhs, dim4_t lhs_strides, U rhs,
               const shared_t<V[]>& output, dim4_t output_strides,
               dim4_t shape, BinaryOp binary_op, Stream& stream) {
        cuda::util::ewise::binary("math::ewise",
                                  lhs.get(), lhs_strides, rhs,
                                  output.get(), output_strides,
                                  shape, true, stream, binary_op);
        stream.attach(lhs, output);
    }

    template<typename T, typename U, typename V, typename BinaryOp,
             std::enable_if_t<details::is_valid_ewise_binary_v<T, U, V, BinaryOp>, bool>>
    void ewise(T lhs, const shared_t<U[]>& rhs, dim4_t rhs_strides,
               const shared_t<V[]>& output, dim4_t output_strides,
               dim4_t shape, BinaryOp binary_op, Stream& stream) {
        cuda::util::ewise::binary("math::ewise",
                                  lhs, rhs.get(), rhs_strides,
                                  output.get(), output_strides,
                                  shape, true, stream, binary_op);
        stream.attach(rhs, output);
    }

    template<typename T, typename U, typename V, typename BinaryOp, typename>
    void ewise(const shared_t<T[]>& lhs, dim4_t lhs_strides,
               const shared_t<U[]>& rhs, dim4_t rhs_strides,
               const shared_t<V[]>& output, dim4_t output_strides,
               dim4_t shape, BinaryOp binary_op, Stream& stream) {
        cuda::util::ewise::binary("math::ewise",
                                  lhs.get(), lhs_strides,
                                  rhs.get(), rhs_strides,
                                  output.get(), output_strides,
                                  shape, true, stream, binary_op);
        stream.attach(lhs, rhs, output);
    }

    #define NOA_INSTANTIATE_EWISE_BINARY(T,U,V,BINARY)                                                                              \
    template void ewise<T,U,V,BINARY,true>(const shared_t<T[]>&, dim4_t, U, const shared_t<V[]>&, dim4_t, dim4_t, BINARY, Stream&); \
    template void ewise<T,U,V,BINARY,true>(T, const shared_t<U[]>&, dim4_t, const shared_t<V[]>&, dim4_t, dim4_t, BINARY, Stream&); \
    template void ewise<T,U,V,BINARY,void>(const shared_t<T[]>&, dim4_t, const shared_t<U[]>&, dim4_t, const shared_t<V[]>&, dim4_t, dim4_t, BINARY, Stream&)

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
