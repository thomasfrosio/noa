#include "noa/gpu/cuda/memory/Arange.hpp"
#include "noa/gpu/cuda/memory/Iota.hpp"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/algorithms/memory/Iota.hpp"

namespace noa::cuda::memory {
    template<typename T, typename>
    void iota(T* src, i64 elements, i64 tile, Stream& stream) {
        if (tile == elements)
            return arange(src, elements, T{0}, T{1}, stream);

        NOA_ASSERT_DEVICE_PTR(src, stream.device());
        const auto kernel = noa::algorithm::memory::iota_1d<i64, i64>(src, tile);
        noa::cuda::utils::iwise_1d(elements, kernel, stream);
    }

    template<typename T, typename>
    void iota(T* src, const Strides4<i64>& strides, const Shape4<i64>& shape, const Vec4<i64>& tile, Stream& stream) {
        if (noa::all(tile == shape.vec()))
            return arange(src, strides, shape, T{0}, T{1}, stream);

        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(src, stream.device());
        const auto kernel = noa::algorithm::memory::iota_4d<i64, i64>(src, strides, shape, tile);
        noa::cuda::utils::iwise_4d(shape, kernel, stream);
    }

    #define NOA_INSTANTIATE_IOTA_(T)                    \
    template void iota<T, void>(T*, i64, i64, Stream&); \
    template void iota<T, void>(T*, const Strides4<i64>&, const Shape4<i64>&, const Vec4<i64>&, Stream&)

    NOA_INSTANTIATE_IOTA_(i8);
    NOA_INSTANTIATE_IOTA_(i16);
    NOA_INSTANTIATE_IOTA_(i32);
    NOA_INSTANTIATE_IOTA_(i64);
    NOA_INSTANTIATE_IOTA_(u8);
    NOA_INSTANTIATE_IOTA_(u16);
    NOA_INSTANTIATE_IOTA_(u32);
    NOA_INSTANTIATE_IOTA_(u64);
    NOA_INSTANTIATE_IOTA_(f16);
    NOA_INSTANTIATE_IOTA_(f32);
    NOA_INSTANTIATE_IOTA_(f64);
}
