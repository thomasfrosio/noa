#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/util/Pointers.h"
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

    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_stride,
               const shared_t<U[]>& rhs,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream) {
        const shared_t<U[]> d_rhs = cuda::util::ensureDeviceAccess(rhs, stream, shape[0]);
        cuda::util::ewise::binary("math::ewise",
                                  lhs.get(), lhs_stride, d_rhs.get(),
                                  output.get(), output_stride,
                                  shape, stream, binary_op);
        stream.attach(lhs, d_rhs, output);
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

    #define NOA_INSTANTIATE_EWISE_BINARY(T,U,V,BINARY)                                                                                                  \
    template void ewise<T,U,V,BINARY,void>(const shared_t<T[]>&, size4_t, U, const shared_t<V[]>&, size4_t, size4_t, BINARY, Stream&);                  \
    template void ewise<T,U,V,BINARY>(const shared_t<T[]>&, size4_t, const shared_t<U[]>&, const shared_t<V[]>&, size4_t, size4_t, BINARY, Stream&);    \
    template void ewise<T,U,V,BINARY>(const shared_t<T[]>&, size4_t, const shared_t<U[]>&, size4_t, const shared_t<V[]>&, size4_t, size4_t, BINARY, Stream&)

    #define NOA_INSTANTIATE_EWISE_BINARY_SCALAR(T)                   \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::plus_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::minus_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::multiply_t);     \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::divide_t);       \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::divide_safe_t);  \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::dist2_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::min_t);          \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::max_t);          \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::equal_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::not_equal_t);    \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::less_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::less_equal_t);   \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::greater_t);      \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::greater_equal_t);\
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::modulo_t)

    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(int16_t);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(int32_t);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(int64_t);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(uint16_t);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(uint32_t);
    NOA_INSTANTIATE_EWISE_BINARY_SCALAR(uint64_t);

    #define NOA_INSTANTIATE_EWISE_BINARY_FLOAT(T)                    \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::plus_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::minus_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::multiply_t);     \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::divide_t);       \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::divide_safe_t);  \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::dist2_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::min_t);          \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::max_t);          \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::equal_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::not_equal_t);    \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::less_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::less_equal_t);   \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::greater_t);      \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::greater_equal_t);\
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::pow_t)

    NOA_INSTANTIATE_EWISE_BINARY_FLOAT(half_t);
    NOA_INSTANTIATE_EWISE_BINARY_FLOAT(float);
    NOA_INSTANTIATE_EWISE_BINARY_FLOAT(double);

    #define NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(T,U,V)              \
    NOA_INSTANTIATE_EWISE_BINARY(T,U,V,::noa::math::plus_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,U,V,::noa::math::minus_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,U,V,::noa::math::multiply_t);     \
    NOA_INSTANTIATE_EWISE_BINARY(T,U,V,::noa::math::divide_t);       \
    NOA_INSTANTIATE_EWISE_BINARY(T,U,V,::noa::math::dist2_t)

    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(chalf_t, chalf_t, chalf_t);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(cfloat_t, cfloat_t, cfloat_t);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(cdouble_t, cdouble_t, cdouble_t);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(chalf_t, half_t, chalf_t);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(cfloat_t, float, cfloat_t);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(cdouble_t, double, cdouble_t);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(half_t, chalf_t, chalf_t);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(float, cfloat_t, cfloat_t);
    NOA_INSTANTIATE_EWISE_BINARY_COMPLEX(double, cdouble_t, cdouble_t);

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

    #define NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(T,V)               \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::math::logical_and_t); \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,V,::noa::math::logical_or_t)

    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(int16_t, int16_t);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(int32_t, int32_t);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(int64_t, int64_t);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(uint16_t, uint16_t);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(uint32_t, uint32_t);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(uint64_t, uint64_t);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(int16_t, bool);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(int32_t, bool);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(int64_t, bool);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(uint16_t, bool);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(uint32_t, bool);
    NOA_INSTANTIATE_EWISE_BINARY_LOGICAL(uint64_t, bool);
}
