#include "noa/gpu/cuda/memory/Cast.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/utils/EwiseUnary.cuh"

namespace noa::cuda::memory {
    template<typename T, typename U, typename _>
    void cast(const Shared<T[]>& input, const Shared<U[]>& output,
              i64 elements, bool clamp, Stream& stream) {
        if constexpr (std::is_same_v<T, U>) {
            copy(input, output, elements, stream);
        } else {
            NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
            NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
            const auto shape = Shape4<i64>{1, 1, 1, elements};
            const auto strides = shape.strides();
            ::noa::cuda::utils::ewise_unary(
                    "memory::cast", input.get(), strides, output.get(), strides, shape, stream,
                    [clamp] __device__(T a) { return clamp ? clamp_cast<U>(a) : static_cast<U>(a); });
            stream.attach(input, output);
        }
    }

    template<typename T, typename U, typename _>
    void cast(const Shared<T[]>& input, const Strides4<i64>& input_strides,
              const Shared<U[]>& output, const Strides4<i64>& output_strides,
              const Shape4<i64>& shape, bool clamp, Stream& stream) {
        if constexpr (std::is_same_v<T, U>) {
            copy(input, input_strides, output, output_strides, shape, stream);
        } else {
            NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
            NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
            ::noa::cuda::utils::ewise_unary(
                    "memory::cast", input.get(), input_strides, output.get(), output_strides, shape, stream,
                    [clamp] __device__(T a) { return clamp ? clamp_cast<U>(a) : static_cast<U>(a); });
            stream.attach(input, output);
        }
    }

    #define NOA_INSTANTIATE_CAST_(T, U)                             \
    template void cast<T, U, void>(                                 \
    const Shared<T[]>&, const Shared<U[]>&, i64, bool, Stream&);    \
    template void cast<T, U, void>(                                 \
        const Shared<T[]>&, const Strides4<i64>&,                   \
        const Shared<U[]>&, const Strides4<i64>&,                   \
        const Shape4<i64>&, bool, Stream&)

    #define NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(T) \
    NOA_INSTANTIATE_CAST_(T, bool); \
    NOA_INSTANTIATE_CAST_(T, i8);   \
    NOA_INSTANTIATE_CAST_(T, u8);   \
    NOA_INSTANTIATE_CAST_(T, i16);  \
    NOA_INSTANTIATE_CAST_(T, u16);  \
    NOA_INSTANTIATE_CAST_(T, i32);  \
    NOA_INSTANTIATE_CAST_(T, u32);  \
    NOA_INSTANTIATE_CAST_(T, i64);  \
    NOA_INSTANTIATE_CAST_(T, u64);  \
    NOA_INSTANTIATE_CAST_(T, f16);  \
    NOA_INSTANTIATE_CAST_(T, f32);  \
    NOA_INSTANTIATE_CAST_(T, f64)

    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(bool);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(i8);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(u8);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(i16);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(u16);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(i32);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(u32);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(i64);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(u64);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(f16);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(f32);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(f64);

    NOA_INSTANTIATE_CAST_(c16, c16);
    NOA_INSTANTIATE_CAST_(c16, c32);
    NOA_INSTANTIATE_CAST_(c16, c64);
    NOA_INSTANTIATE_CAST_(c32, c16);
    NOA_INSTANTIATE_CAST_(c32, c32);
    NOA_INSTANTIATE_CAST_(c32, c64);
    NOA_INSTANTIATE_CAST_(c64, c16);
    NOA_INSTANTIATE_CAST_(c64, c32);
    NOA_INSTANTIATE_CAST_(c64, c64);
}
