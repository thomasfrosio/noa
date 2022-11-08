#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/utils/EwiseBinary.cuh"

namespace noa::cuda::math {
    template<typename T, typename U, typename V, typename BinaryOp,
             std::enable_if_t<details::is_valid_ewise_binary_v<T, U, V, BinaryOp>, bool>>
    void ewise(const shared_t<T[]>& lhs, dim4_t lhs_strides, U rhs,
               const shared_t<V[]>& output, dim4_t output_strides,
               dim4_t shape, BinaryOp binary_op, Stream& stream) {
        cuda::utils::ewise::binary(
                "math::ewise",
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
        cuda::utils::ewise::binary(
                "math::ewise",
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
        cuda::utils::ewise::binary(
                "math::ewise",
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
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::plus_t);         \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::minus_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::multiply_t);     \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::divide_t);       \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::divide_safe_t);  \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::dist2_t);        \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::min_t);          \
    NOA_INSTANTIATE_EWISE_BINARY(T,T,T,::noa::math::max_t);          \
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

    NOA_INSTANTIATE_EWISE_BINARY(chalf_t, chalf_t, chalf_t, ::noa::math::multiply_conj_t);
    NOA_INSTANTIATE_EWISE_BINARY(cfloat_t, cfloat_t, cfloat_t, ::noa::math::multiply_conj_t);
    NOA_INSTANTIATE_EWISE_BINARY(cdouble_t, cdouble_t, cdouble_t, ::noa::math::multiply_conj_t);

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
