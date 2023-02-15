#include "noa/gpu/cuda/memory/Linspace.h"
#include "noa/gpu/cuda/memory/Set.h"
#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/algorithms/memory/Linspace.hpp"

namespace noa::cuda::memory {
    template<typename T, typename>
    T linspace(T* src, i64 elements, T start, T stop, bool endpoint, Stream& stream) {
        if (elements <= 1) {
            set(src, elements, start, stream);
            return T{0};
        }

        NOA_ASSERT_DEVICE_PTR(src, stream.device());
        const auto [kernel, step] = noa::algorithm::memory::linspace_1d<i64, i64>(
                src, elements, start, stop, endpoint);
        noa::cuda::utils::iwise_1d("linspace", elements, kernel, stream);
        return step;
    }

    template<typename T, typename>
    T linspace(T* src, const Strides4<i64>& strides, const Shape4<i64>& shape,
               T start, T stop, bool endpoint, Stream& stream) {
        const auto elements = shape.elements();
        if (elements <= 1) {
            set(src, elements, start, stream);
            return T{0};
        }

        NOA_ASSERT_DEVICE_PTR(src, stream.device());
        const auto [kernel, step] = noa::algorithm::memory::linspace_4d<i64, i64>(
                src, strides, shape, start, stop, endpoint);
        noa::cuda::utils::iwise_4d("linspace", shape, kernel, stream);
        return step;
    }

    #define NOA_INSTANTIATE_LINSPACE_(T)                        \
    template T linspace<T, void>(T*, i64, T, T, bool, Stream&); \
    template T linspace<T, void>(T*, const Strides4<i64>&, const Shape4<i64>&, T, T, bool, Stream&)

    NOA_INSTANTIATE_LINSPACE_(i8);
    NOA_INSTANTIATE_LINSPACE_(i16);
    NOA_INSTANTIATE_LINSPACE_(i32);
    NOA_INSTANTIATE_LINSPACE_(i64);
    NOA_INSTANTIATE_LINSPACE_(u8);
    NOA_INSTANTIATE_LINSPACE_(u16);
    NOA_INSTANTIATE_LINSPACE_(u32);
    NOA_INSTANTIATE_LINSPACE_(u64);
    NOA_INSTANTIATE_LINSPACE_(f16);
    NOA_INSTANTIATE_LINSPACE_(f32);
    NOA_INSTANTIATE_LINSPACE_(f64);
    NOA_INSTANTIATE_LINSPACE_(c16);
    NOA_INSTANTIATE_LINSPACE_(c32);
    NOA_INSTANTIATE_LINSPACE_(c64);
}
