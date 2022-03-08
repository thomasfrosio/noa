#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/util/EwiseTrinary.cuh"

namespace noa::cuda::math {
    template<typename T, typename U, typename V, typename TrinaryOp, typename>
    void ewise(const T* lhs, size4_t lhs_stride, U mhs, U rhs,
               V* output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::util::ewise::trinary("math::ewise", lhs, lhs_stride, mhs, rhs,
                                   output, output_stride, shape, stream, trinary_op);
    }

    template<typename T, typename U, typename V, typename TrinaryOp>
    void ewise(const T* lhs, size4_t lhs_stride, const U* mhs, const U* rhs,
               V* output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        memory::PtrDevice<U> buffer1, buffer2;
        mhs = cuda::util::ensureDeviceAccess(mhs, stream, buffer1, shape[0]);
        rhs = cuda::util::ensureDeviceAccess(rhs, stream, buffer2, shape[0]);
        cuda::util::ewise::trinary("math::ewise", lhs, lhs_stride, mhs, rhs,
                                   output, output_stride, shape, stream, trinary_op);
    }

    template<typename T, typename U, typename V, typename W, typename TrinaryOp>
    void ewise(const T* lhs, size4_t lhs_stride,
               const U* mhs, size4_t mhs_stride,
               const V* rhs, size4_t rhs_stride,
               W* output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::util::ewise::trinary("math::ewise", lhs, lhs_stride, mhs, mhs_stride, rhs, rhs_stride,
                                   output, output_stride, shape, stream, trinary_op);
    }

    #define NOA_INSTANTIATE_EWISE_TRINARY(T,U,V,TRINARY)                                                                \
    template void ewise<T,U,V,TRINARY,void>(const T*, size4_t, U, U, V*, size4_t, size4_t, TRINARY, Stream&);           \
    template void ewise<T,U,V,TRINARY>(const T*, size4_t, const U*, const U*, V*, size4_t, size4_t, TRINARY, Stream&);  \
    template void ewise<T,U,U,V,TRINARY>(const T*, size4_t, const U*, size4_t, const U*, size4_t, V*, size4_t, size4_t, TRINARY, Stream&)

    #define NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(T,V)                   \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,V,::noa::math::within_t);         \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,V,::noa::math::within_equal_t)

    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(int16_t, int16_t);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(int32_t, int32_t);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(int64_t, int64_t);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(uint16_t, uint16_t);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(uint32_t, uint32_t);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(uint64_t, uint64_t);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(half_t, half_t);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(float, float);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(double, double);

    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(int16_t, bool);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(int32_t, bool);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(int64_t, bool);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(uint16_t, bool);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(uint32_t, bool);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(uint64_t, bool);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(half_t, bool);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(float, bool);
    NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(double, bool);

    #define NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(T) \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,T,::noa::math::clamp_t)

    NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(int16_t);
    NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(int32_t);
    NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(int64_t);
    NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(uint16_t);
    NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(uint32_t);
    NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(uint64_t);
    NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(half_t);
    NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(float);
    NOA_INSTANTIATE_EWISE_TRINARY_CLAMP(double);

    #define NOA_INSTANTIATE_EWISE_TRINARY_FMA(T,U)  \
    NOA_INSTANTIATE_EWISE_TRINARY(T,U,T,::noa::math::fma_t)

    NOA_INSTANTIATE_EWISE_TRINARY_FMA(chalf_t, chalf_t);
    NOA_INSTANTIATE_EWISE_TRINARY_FMA(cfloat_t, cfloat_t);
    NOA_INSTANTIATE_EWISE_TRINARY_FMA(cdouble_t, cdouble_t);
    NOA_INSTANTIATE_EWISE_TRINARY_FMA(chalf_t, half_t);
    NOA_INSTANTIATE_EWISE_TRINARY_FMA(cfloat_t, float);
    NOA_INSTANTIATE_EWISE_TRINARY_FMA(cdouble_t, double);

    NOA_INSTANTIATE_EWISE_TRINARY_FMA(half_t, half_t);
    NOA_INSTANTIATE_EWISE_TRINARY_FMA(float, float);
    NOA_INSTANTIATE_EWISE_TRINARY_FMA(double, double);
    NOA_INSTANTIATE_EWISE_TRINARY_FMA(int16_t, int16_t);
    NOA_INSTANTIATE_EWISE_TRINARY_FMA(int32_t, int32_t);
    NOA_INSTANTIATE_EWISE_TRINARY_FMA(int64_t, int64_t);
    NOA_INSTANTIATE_EWISE_TRINARY_FMA(uint16_t, uint16_t);
    NOA_INSTANTIATE_EWISE_TRINARY_FMA(uint32_t, uint32_t);
    NOA_INSTANTIATE_EWISE_TRINARY_FMA(uint64_t, uint64_t);
}
