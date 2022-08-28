#include "noa/common/Functors.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/util/EwiseUnary.cuh"

namespace noa::cuda::memory::details {
    template<typename T, typename>
    void copy(const shared_t<T[]>& src, size4_t src_strides,
              const shared_t<T[]>& dst, size4_t dst_strides,
              size4_t shape, Stream& stream) {
        // This function is called from noa::cuda::memory::copy(), which already has reordered to C-major.
        util::ewise::unary<true>("memory::copy",
                                 src.get(), src_strides,
                                 dst.get(), dst_strides,
                                 shape, false, stream, noa::math::copy_t{});
        stream.attach(src, dst);
    }

    #define NOA_INSTANTIATE_COPY_(T) \
    template void copy<T,void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&)

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
