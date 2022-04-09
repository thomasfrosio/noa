#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/util/EwiseUnary.cuh"

namespace noa::cuda::math {
    template<typename T, typename U, typename UnaryOp>
    void ewise(const shared_t<T[]>& input, size4_t input_stride,
               const shared_t<U[]>& output, size4_t output_stride,
               size4_t shape, UnaryOp unary_op, Stream& stream) {
        cuda::util::ewise::unary( "math::ewise", input.get(), input_stride,
                                  output.get(), output_stride,
                                  shape, stream, unary_op);
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_EWISE_UNARY(T,U,UNARY) \
    template void ewise<T,U,UNARY>(const shared_t<T[]>&, size4_t, const shared_t<U[]>&, size4_t, size4_t, UNARY, Stream&)

    #define NOA_INSTANTIATE_EWISE_UNARY_INT(T,U)                \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::math::copy_t);       \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::math::square_t);     \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::math::abs_t);        \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::math::negate_t);     \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::math::one_minus_t)

    NOA_INSTANTIATE_EWISE_UNARY_INT(int16_t, int16_t);
    NOA_INSTANTIATE_EWISE_UNARY_INT(int32_t, int32_t);
    NOA_INSTANTIATE_EWISE_UNARY_INT(int64_t, int64_t);

    #define NOA_INSTANTIATE_EWISE_UNARY_UINT(T,U)               \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::math::copy_t);       \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::math::square_t)

    NOA_INSTANTIATE_EWISE_UNARY_UINT(uint16_t, uint16_t);
    NOA_INSTANTIATE_EWISE_UNARY_UINT(uint32_t, uint32_t);
    NOA_INSTANTIATE_EWISE_UNARY_UINT(uint64_t, uint64_t);

    #define NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(T,U)            \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::math::nonzero_t);    \
    NOA_INSTANTIATE_EWISE_UNARY(T,U,::noa::math::logical_not_t)

    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(int16_t, int16_t);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(int32_t, int32_t);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(int64_t, int64_t);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(uint16_t, uint16_t);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(uint32_t, uint32_t);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(uint64_t, uint64_t);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(int16_t, bool);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(int32_t, bool);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(int64_t, bool);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(uint16_t, bool);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(uint32_t, bool);
    NOA_INSTANTIATE_EWISE_UNARY_TO_BOOL(uint64_t, bool);

    #define NOA_INSTANTIATE_EWISE_UNARY_FLOAT(T)                \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::math::copy_t);       \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::math::square_t);     \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::math::abs_t);        \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::math::negate_t);     \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::math::one_minus_t);  \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::math::inverse_t);    \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::math::sqrt_t);       \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::math::rsqrt_t);      \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::math::exp_t);        \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::math::log_t);        \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::math::cos_t);        \
    NOA_INSTANTIATE_EWISE_UNARY(T,T,::noa::math::sin_t)

    NOA_INSTANTIATE_EWISE_UNARY_FLOAT(half_t);
    NOA_INSTANTIATE_EWISE_UNARY_FLOAT(float);
    NOA_INSTANTIATE_EWISE_UNARY_FLOAT(double);

    #define NOA_INSTANTIATE_EWISE_UNARY_COMPLEX(C,R)            \
    NOA_INSTANTIATE_EWISE_UNARY(C,C,::noa::math::one_minus_t);  \
    NOA_INSTANTIATE_EWISE_UNARY(C,C,::noa::math::square_t);     \
    NOA_INSTANTIATE_EWISE_UNARY(C,C,::noa::math::inverse_t);    \
    NOA_INSTANTIATE_EWISE_UNARY(C,C,::noa::math::normalize_t);  \
    NOA_INSTANTIATE_EWISE_UNARY(C,R,::noa::math::abs_t);        \
    NOA_INSTANTIATE_EWISE_UNARY(C,R,::noa::math::real_t);       \
    NOA_INSTANTIATE_EWISE_UNARY(C,R,::noa::math::imag_t)

    NOA_INSTANTIATE_EWISE_UNARY_COMPLEX(chalf_t, half_t);
    NOA_INSTANTIATE_EWISE_UNARY_COMPLEX(cfloat_t, float);
    NOA_INSTANTIATE_EWISE_UNARY_COMPLEX(cdouble_t, double);
}
