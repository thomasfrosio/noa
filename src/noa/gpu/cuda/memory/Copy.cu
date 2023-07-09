#include "noa/core/types/Functors.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/utils/EwiseUnary.cuh"

namespace noa::cuda::memory::details {
    template<typename T, typename>
    void copy(const T* src, const Strides4<i64>& src_strides,
              T* dst, const Strides4<i64>& dst_strides,
              const Shape4<i64>& shape, Stream& stream) {
        noa::cuda::utils::ewise_unary<PointerTraits::RESTRICT>(
                src, src_strides,
                dst, dst_strides,
                shape, stream, noa::copy_t{});
    }

    #define NOA_INSTANTIATE_COPY_(T) \
    template void copy<T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, Stream&)

    NOA_INSTANTIATE_COPY_(bool);
    NOA_INSTANTIATE_COPY_(i8);
    NOA_INSTANTIATE_COPY_(i16);
    NOA_INSTANTIATE_COPY_(i32);
    NOA_INSTANTIATE_COPY_(i64);
    NOA_INSTANTIATE_COPY_(u8);
    NOA_INSTANTIATE_COPY_(u16);
    NOA_INSTANTIATE_COPY_(u32);
    NOA_INSTANTIATE_COPY_(u64);
    NOA_INSTANTIATE_COPY_(f16);
    NOA_INSTANTIATE_COPY_(f32);
    NOA_INSTANTIATE_COPY_(f64);
    NOA_INSTANTIATE_COPY_(c16);
    NOA_INSTANTIATE_COPY_(c32);
    NOA_INSTANTIATE_COPY_(c64);
}
