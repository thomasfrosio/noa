#include "noa/gpu/cuda/memory/Arange.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/algorithms/memory/Arange.hpp"

namespace noa::cuda::memory {
    template<typename T, typename>
    void arange(T* src, i64 elements, T start, T step, Stream& stream) {
        if (!elements)
            return;

        NOA_ASSERT_DEVICE_PTR(src, stream.device());
        const auto kernel = noa::algorithm::memory::arange_1d<i64, i64>(src, start, step);
        noa::cuda::utils::iwise_1d(elements, kernel, stream);
    }

    template<typename T, typename>
    void arange(T* src, const Strides4<i64>& strides, const Shape4<i64>& shape, T start, T step, Stream& stream) {
        if (!shape.elements())
            return;

        NOA_ASSERT_DEVICE_PTR(src, stream.device());
        const auto kernel = noa::algorithm::memory::arange_4d<i64, i64>(src, strides, shape, start, step);
        noa::cuda::utils::iwise_4d(shape, kernel, stream);
    }

    #define NOA_INSTANTIATE_ARANGE_(T)                      \
    template void arange<T, void>(T*, i64, T, T, Stream&);  \
    template void arange<T, void>(T*, const Strides4<i64>&, const Shape4<i64>&, T, T, Stream&)

    NOA_INSTANTIATE_ARANGE_(i8);
    NOA_INSTANTIATE_ARANGE_(i16);
    NOA_INSTANTIATE_ARANGE_(i32);
    NOA_INSTANTIATE_ARANGE_(i64);
    NOA_INSTANTIATE_ARANGE_(u8);
    NOA_INSTANTIATE_ARANGE_(u16);
    NOA_INSTANTIATE_ARANGE_(u32);
    NOA_INSTANTIATE_ARANGE_(u64);
    NOA_INSTANTIATE_ARANGE_(f16);
    NOA_INSTANTIATE_ARANGE_(f32);
    NOA_INSTANTIATE_ARANGE_(f64);
    NOA_INSTANTIATE_ARANGE_(c16);
    NOA_INSTANTIATE_ARANGE_(c32);
    NOA_INSTANTIATE_ARANGE_(c64);
}
