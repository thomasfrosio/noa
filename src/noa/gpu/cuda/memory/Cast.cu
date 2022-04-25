#include "noa/gpu/cuda/memory/Cast.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/util/EwiseUnary.cuh"

namespace noa::cuda::memory {
    template<typename T, typename U>
    void cast(const shared_t<T[]>& input, const shared_t<U[]>& output,
              size_t elements, bool clamp, Stream& stream) {
        if constexpr (std::is_same_v<T, U>) {
            copy(input, output, elements, stream);
        } else {
            const size4_t shape{1, 1, 1, elements};
            const size4_t stride = shape.stride();
            cuda::util::ewise::unary<true>(
                    "memory::cast", input.get(), stride, output.get(), stride, shape, stream,
            [clamp] __device__(T a) { return clamp ? clamp_cast<U>(a) : static_cast<U>(a); });
            stream.attach(input, output);
        }
    }

    template<typename T, typename U>
    void cast(const shared_t<T[]>& input, size4_t input_stride,
              const shared_t<U[]>& output, size4_t output_stride,
              size4_t shape, bool clamp, Stream& stream) {
        if constexpr (std::is_same_v<T, U>) {
            copy(input, input_stride, output, output_stride, shape, stream);
        } else {
            cuda::util::ewise::unary<true>(
                    "memory::cast", input.get(), input_stride, output.get(), output_stride, shape, stream,
                    [clamp] __device__(T a) { return clamp ? clamp_cast<U>(a) : static_cast<U>(a); });
            stream.attach(input, output);
        }
    }

    #define NOA_INSTANTIATE_CAST_(T, U)                                                             \
    template void cast<T, U>(const shared_t<T[]>&, const shared_t<U[]>&, size_t, bool, Stream&);    \
    template void cast<T, U>(const shared_t<T[]>&, size4_t, const shared_t<U[]>&, size4_t, size4_t, bool, Stream&)

    #define NOA_INSTANTIATE_CAST_TO_ALL_SCALAR(T) \
    NOA_INSTANTIATE_CAST_(T, bool);     \
    NOA_INSTANTIATE_CAST_(T, int8_t);   \
    NOA_INSTANTIATE_CAST_(T, uint8_t);  \
    NOA_INSTANTIATE_CAST_(T, int16_t);  \
    NOA_INSTANTIATE_CAST_(T, uint16_t); \
    NOA_INSTANTIATE_CAST_(T, int32_t);  \
    NOA_INSTANTIATE_CAST_(T, uint32_t); \
    NOA_INSTANTIATE_CAST_(T, int64_t);  \
    NOA_INSTANTIATE_CAST_(T, uint64_t); \
    NOA_INSTANTIATE_CAST_(T, half_t);   \
    NOA_INSTANTIATE_CAST_(T, float);    \
    NOA_INSTANTIATE_CAST_(T, double)

    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR(bool);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR(int8_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR(uint8_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR(int16_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR(uint16_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR(int32_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR(uint32_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR(int64_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR(uint64_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR(half_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR(float);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR(double);

    NOA_INSTANTIATE_CAST_(chalf_t, cfloat_t);
    NOA_INSTANTIATE_CAST_(chalf_t, cdouble_t);
    NOA_INSTANTIATE_CAST_(cfloat_t, chalf_t);
    NOA_INSTANTIATE_CAST_(cfloat_t, cdouble_t);
    NOA_INSTANTIATE_CAST_(cdouble_t, chalf_t);
    NOA_INSTANTIATE_CAST_(cdouble_t, cfloat_t);
}
