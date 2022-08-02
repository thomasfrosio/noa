#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/util/EwiseTrinary.cuh"

namespace noa::cuda::math {
    template<typename T, typename U, typename V, typename TrinaryOp, typename>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides, U mhs, U rhs,
               const shared_t<V[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::util::ewise::trinary("math::ewise",
                                   lhs.get(), lhs_strides, mhs, rhs,
                                   output.get(), output_strides,
                                   shape, true, stream, trinary_op);
        stream.attach(lhs, output);
    }

    template<typename T, typename U, typename V, typename W, typename TrinaryOp, typename>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides,
               const shared_t<U[]>& mhs, size4_t mhs_strides,
               V rhs,
               const shared_t<W[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::util::ewise::trinary("math::ewise",
                                   lhs.get(), lhs_strides,
                                   mhs.get(), mhs_strides,
                                   rhs,
                                   output.get(), output_strides,
                                   shape, true, stream, trinary_op);
        stream.attach(lhs, mhs, output);
    }

    template<typename T, typename U, typename V, typename W, typename TrinaryOp, typename>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides,
               V mhs,
               const shared_t<U[]>& rhs, size4_t rhs_strides,
               const shared_t<W[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::util::ewise::trinary("math::ewise",
                                   lhs.get(), lhs_strides,
                                   mhs,
                                   rhs.get(), rhs_strides,
                                   output.get(), output_strides,
                                   shape, true, stream, trinary_op);
        stream.attach(lhs, rhs, output);
    }

    template<typename T, typename U, typename V, typename W, typename TrinaryOp, typename>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_strides,
               const shared_t<U[]>& mhs, size4_t mhs_strides,
               const shared_t<V[]>& rhs, size4_t rhs_strides,
               const shared_t<W[]>& output, size4_t output_strides,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        cuda::util::ewise::trinary("math::ewise",
                                   lhs.get(), lhs_strides,
                                   mhs.get(), mhs_strides,
                                   rhs.get(), rhs_strides,
                                   output.get(), output_strides,
                                   shape, true, stream, trinary_op);
        stream.attach(lhs, mhs, rhs, output);
    }

    #define NOA_INSTANTIATE_EWISE_TRINARY(T,U,V,TRINARY)                                                \
    template void ewise<T,U,V,TRINARY,void>(const shared_t<T[]>&, size4_t, U, U,                        \
                                            const shared_t<V[]>&, size4_t, size4_t, TRINARY, Stream&);  \
    template void ewise<T,U,U,V,TRINARY,void>(const shared_t<T[]>&, size4_t,                            \
                                              const shared_t<U[]>&, size4_t, U,                         \
                                              const shared_t<V[]>&, size4_t, size4_t, TRINARY, Stream&);\
    template void ewise<T,U,U,V,TRINARY,void>(const shared_t<T[]>&, size4_t, U,                         \
                                              const shared_t<U[]>&, size4_t,                            \
                                              const shared_t<V[]>&, size4_t, size4_t, TRINARY, Stream&);\
    template void ewise<T,U,U,V,TRINARY,void>(const shared_t<T[]>&, size4_t,                            \
                                              const shared_t<U[]>&, size4_t,                            \
                                              const shared_t<U[]>&, size4_t,                            \
                                              const shared_t<V[]>&, size4_t, size4_t, TRINARY, Stream&)

    #define NOA_INSTANTIATE_EWISE_TRINARY_SCALAR(T,V)           \
    NOA_INSTANTIATE_EWISE_TRINARY(T,T,V,::noa::math::within_t); \
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

    #define NOA_INSTANTIATE_EWISE_TRINARY_FMA(T,U)          \
    NOA_INSTANTIATE_EWISE_TRINARY(T,U,T,::noa::math::fma_t);\
    NOA_INSTANTIATE_EWISE_TRINARY(T,U,T,::noa::math::plus_divide_t)

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
