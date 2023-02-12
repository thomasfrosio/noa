#include "noa/gpu/cuda/memory/Set.h"
#include "noa/gpu/cuda/utils/EwiseUnary.cuh"
#include "noa/gpu/cuda/utils/Pointers.h"

namespace noa::cuda::memory::details {
    template<typename T, typename _>
    void set(T* src, i64 elements, T value, Stream& stream) {
        if (!elements)
            return;

        NOA_ASSERT_DEVICE_PTR(src, stream.device());
        auto set_op = [value]__device__(T) { return value; };
        const auto shape = Shape4<i64>{1, 1, 1, elements};
        noa::cuda::utils::ewise_unary<StridesTraits::CONTIGUOUS>(
                "memory::copy", src, shape.strides(), shape, stream, set_op);
    }

    template<typename T, typename _>
    void set(const Shared<T[]>& src, const Strides4<i64>& strides, const Shape4<i64>& shape, T value, Stream& stream) {
        if (!shape.elements())
            return;

        NOA_ASSERT_DEVICE_PTR(src.get(), stream.device());
        auto set_op = [value]__device__(T) { return value; };
        noa::cuda::utils::ewise_unary<StridesTraits::CONTIGUOUS>(
                "memory::copy", src.get(), strides, shape, stream, set_op);
        stream.attach(src);
    }

    #define NOA_INSTANTIATE_SET_(T)                     \
    template void set<T, void>(T*, i64, T, Stream&);    \
    template void set<T, void>(const Shared<T[]>&, const Strides4<i64>&, const Shape4<i64>&, T, Stream&)

    NOA_INSTANTIATE_SET_(bool);
    NOA_INSTANTIATE_SET_(i8);
    NOA_INSTANTIATE_SET_(i16);
    NOA_INSTANTIATE_SET_(i32);
    NOA_INSTANTIATE_SET_(i64);
    NOA_INSTANTIATE_SET_(u8);
    NOA_INSTANTIATE_SET_(u16);
    NOA_INSTANTIATE_SET_(u32);
    NOA_INSTANTIATE_SET_(u64);
    NOA_INSTANTIATE_SET_(f16);
    NOA_INSTANTIATE_SET_(f32);
    NOA_INSTANTIATE_SET_(f64);
    NOA_INSTANTIATE_SET_(c16);
    NOA_INSTANTIATE_SET_(c32);
    NOA_INSTANTIATE_SET_(c64);
}
