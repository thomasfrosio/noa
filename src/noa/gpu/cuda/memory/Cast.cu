#include "noa/gpu/cuda/memory/Cast.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/utils/EwiseUnary.cuh"

namespace noa::cuda::memory {
    template<typename T, typename U, typename V>
    void cast(const shared_t<T[]>& input, const shared_t<U[]>& output,
              dim_t elements, bool clamp, Stream& stream) {
        if constexpr (std::is_same_v<T, U>) {
            copy(input, output, elements, stream);
        } else {
            const dim4_t shape{1, 1, 1, elements};
            const dim4_t strides = shape.strides();
            ::noa::cuda::utils::ewise::unary<true>(
                    "memory::cast", input.get(), strides, output.get(), strides, shape, true, stream,
                    [clamp] __device__(T a) { return clamp ? clamp_cast<U>(a) : static_cast<U>(a); });
            stream.attach(input, output);
        }
    }

    template<typename T, typename U, typename V>
    void cast(const shared_t<T[]>& input, dim4_t input_strides,
              const shared_t<U[]>& output, dim4_t output_strides,
              dim4_t shape, bool clamp, Stream& stream) {
        if constexpr (std::is_same_v<T, U>) {
            copy(input, input_strides, output, output_strides, shape, stream);
        } else {
            ::noa::cuda::utils::ewise::unary<true>(
                    "memory::cast", input.get(), input_strides, output.get(), output_strides, shape, true, stream,
                    [clamp] __device__(T a) { return clamp ? clamp_cast<U>(a) : static_cast<U>(a); });
            stream.attach(input, output);
        }
    }

    #define NOA_INSTANTIATE_CAST_(T, U)                                                                 \
    template void cast<T, U, void>(const shared_t<T[]>&, const shared_t<U[]>&, dim_t, bool, Stream&);   \
    template void cast<T, U, void>(const shared_t<T[]>&, dim4_t, const shared_t<U[]>&, dim4_t, dim4_t, bool, Stream&)

    #define NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(T) \
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

    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(bool);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(int8_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(uint8_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(int16_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(uint16_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(int32_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(uint32_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(int64_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(uint64_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(half_t);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(float);
    NOA_INSTANTIATE_CAST_TO_ALL_SCALAR_(double);

    NOA_INSTANTIATE_CAST_(chalf_t, chalf_t);
    NOA_INSTANTIATE_CAST_(chalf_t, cfloat_t);
    NOA_INSTANTIATE_CAST_(chalf_t, cdouble_t);
    NOA_INSTANTIATE_CAST_(cfloat_t, chalf_t);
    NOA_INSTANTIATE_CAST_(cfloat_t, cfloat_t);
    NOA_INSTANTIATE_CAST_(cfloat_t, cdouble_t);
    NOA_INSTANTIATE_CAST_(cdouble_t, chalf_t);
    NOA_INSTANTIATE_CAST_(cdouble_t, cfloat_t);
    NOA_INSTANTIATE_CAST_(cdouble_t, cdouble_t);
}
