#include "noa/common/Functors.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/utils/EwiseUnary.cuh"

namespace noa::cuda::memory::details {
    template<typename T, typename>
    void copy(const T* src, dim4_t src_strides,
              T* dst, dim4_t dst_strides,
              dim4_t shape, Stream& stream) {
        // This function is called from noa::cuda::memory::copy(), which already has reordered to C-major.
        utils::ewise::unary<true>(
                "memory::copy",
                src, src_strides,
                dst, dst_strides,
                shape, false, stream, noa::math::copy_t{});
    }

    #define NOA_INSTANTIATE_COPY_(T) \
    template void copy<T,void>(const T*, dim4_t, T*, dim4_t, dim4_t, Stream&)

    NOA_INSTANTIATE_COPY_(bool);
    NOA_INSTANTIATE_COPY_(int8_t);
    NOA_INSTANTIATE_COPY_(int16_t);
    NOA_INSTANTIATE_COPY_(int32_t);
    NOA_INSTANTIATE_COPY_(int64_t);
    NOA_INSTANTIATE_COPY_(uint8_t);
    NOA_INSTANTIATE_COPY_(uint16_t);
    NOA_INSTANTIATE_COPY_(uint32_t);
    NOA_INSTANTIATE_COPY_(uint64_t);
    NOA_INSTANTIATE_COPY_(half_t);
    NOA_INSTANTIATE_COPY_(float);
    NOA_INSTANTIATE_COPY_(double);
    NOA_INSTANTIATE_COPY_(chalf_t);
    NOA_INSTANTIATE_COPY_(cfloat_t);
    NOA_INSTANTIATE_COPY_(cdouble_t);
}
