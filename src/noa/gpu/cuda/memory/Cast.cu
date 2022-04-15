#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/memory/Cast.h"
#include "noa/gpu/cuda/util/EwiseUnary.cuh"

namespace noa::cuda::memory {
    template<typename T, typename U>
    void cast(const shared_t<T[]>& input, const shared_t<U[]>& output,
              size_t elements, bool clamp, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const size4_t shape{1, 1, 1, elements};
        const size4_t stride = shape.stride();
        cuda::util::ewise::unary<true>(
                "memory::cast", input.get(), stride, output.get(), stride, shape, stream,
                [clamp] __device__(T a) { return clamp ? clamp_cast<U>(a) : static_cast<U>(a); });
        stream.attach(input, output);
    }

    template<typename T, typename U>
    void cast(const shared_t<T[]>& input, size4_t input_stride,
              const shared_t<U[]>& output, size4_t output_stride,
              size4_t shape, bool clamp, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        cuda::util::ewise::unary<true>(
                "memory::cast", input.get(), input_stride, output.get(), output_stride, shape, stream,
                [clamp] __device__(T a) { return clamp ? clamp_cast<U>(a) : static_cast<U>(a); });
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_CAST_(T, U)                                                             \
    template void cast<T, U>(const shared_t<T[]>&, const shared_t<U[]>&, size_t, bool, Stream&);    \
    template void cast<T, U>(const shared_t<T[]>&, size4_t, const shared_t<U[]>&, size4_t, size4_t, bool, Stream&)

    NOA_INSTANTIATE_CAST_(int8_t, uint8_t);
    NOA_INSTANTIATE_CAST_(int8_t, int16_t);
    NOA_INSTANTIATE_CAST_(int8_t, uint16_t);
    NOA_INSTANTIATE_CAST_(int8_t, uint32_t);
    NOA_INSTANTIATE_CAST_(int8_t, int64_t);
    NOA_INSTANTIATE_CAST_(int8_t, uint64_t);
    NOA_INSTANTIATE_CAST_(int8_t, half_t);
    NOA_INSTANTIATE_CAST_(int8_t, float);
    NOA_INSTANTIATE_CAST_(int8_t, double);

    NOA_INSTANTIATE_CAST_(uint8_t, int8_t);
    NOA_INSTANTIATE_CAST_(uint8_t, int16_t);
    NOA_INSTANTIATE_CAST_(uint8_t, uint16_t);
    NOA_INSTANTIATE_CAST_(uint8_t, uint32_t);
    NOA_INSTANTIATE_CAST_(uint8_t, int64_t);
    NOA_INSTANTIATE_CAST_(uint8_t, uint64_t);
    NOA_INSTANTIATE_CAST_(uint8_t, half_t);
    NOA_INSTANTIATE_CAST_(uint8_t, float);
    NOA_INSTANTIATE_CAST_(uint8_t, double);

    NOA_INSTANTIATE_CAST_(int16_t, int8_t);
    NOA_INSTANTIATE_CAST_(int16_t, uint8_t);
    NOA_INSTANTIATE_CAST_(int16_t, uint16_t);
    NOA_INSTANTIATE_CAST_(int16_t, uint32_t);
    NOA_INSTANTIATE_CAST_(int16_t, int64_t);
    NOA_INSTANTIATE_CAST_(int16_t, uint64_t);
    NOA_INSTANTIATE_CAST_(int16_t, half_t);
    NOA_INSTANTIATE_CAST_(int16_t, float);
    NOA_INSTANTIATE_CAST_(int16_t, double);

    NOA_INSTANTIATE_CAST_(uint16_t, int8_t);
    NOA_INSTANTIATE_CAST_(uint16_t, uint8_t);
    NOA_INSTANTIATE_CAST_(uint16_t, int16_t);
    NOA_INSTANTIATE_CAST_(uint16_t, uint32_t);
    NOA_INSTANTIATE_CAST_(uint16_t, int64_t);
    NOA_INSTANTIATE_CAST_(uint16_t, uint64_t);
    NOA_INSTANTIATE_CAST_(uint16_t, half_t);
    NOA_INSTANTIATE_CAST_(uint16_t, float);
    NOA_INSTANTIATE_CAST_(uint16_t, double);

    NOA_INSTANTIATE_CAST_(int32_t, int8_t);
    NOA_INSTANTIATE_CAST_(int32_t, uint8_t);
    NOA_INSTANTIATE_CAST_(int32_t, int16_t);
    NOA_INSTANTIATE_CAST_(int32_t, uint16_t);
    NOA_INSTANTIATE_CAST_(int32_t, uint32_t);
    NOA_INSTANTIATE_CAST_(int32_t, int64_t);
    NOA_INSTANTIATE_CAST_(int32_t, uint64_t);
    NOA_INSTANTIATE_CAST_(int32_t, half_t);
    NOA_INSTANTIATE_CAST_(int32_t, float);
    NOA_INSTANTIATE_CAST_(int32_t, double);

    NOA_INSTANTIATE_CAST_(uint32_t, int8_t);
    NOA_INSTANTIATE_CAST_(uint32_t, uint8_t);
    NOA_INSTANTIATE_CAST_(uint32_t, int16_t);
    NOA_INSTANTIATE_CAST_(uint32_t, uint16_t);
    NOA_INSTANTIATE_CAST_(uint32_t, int32_t);
    NOA_INSTANTIATE_CAST_(uint32_t, int64_t);
    NOA_INSTANTIATE_CAST_(uint32_t, uint64_t);
    NOA_INSTANTIATE_CAST_(uint32_t, half_t);
    NOA_INSTANTIATE_CAST_(uint32_t, float);
    NOA_INSTANTIATE_CAST_(uint32_t, double);

    NOA_INSTANTIATE_CAST_(int64_t, int8_t);
    NOA_INSTANTIATE_CAST_(int64_t, uint8_t);
    NOA_INSTANTIATE_CAST_(int64_t, int16_t);
    NOA_INSTANTIATE_CAST_(int64_t, uint16_t);
    NOA_INSTANTIATE_CAST_(int64_t, int32_t);
    NOA_INSTANTIATE_CAST_(int64_t, uint32_t);
    NOA_INSTANTIATE_CAST_(int64_t, uint64_t);
    NOA_INSTANTIATE_CAST_(int64_t, half_t);
    NOA_INSTANTIATE_CAST_(int64_t, float);
    NOA_INSTANTIATE_CAST_(int64_t, double);

    NOA_INSTANTIATE_CAST_(uint64_t, int8_t);
    NOA_INSTANTIATE_CAST_(uint64_t, uint8_t);
    NOA_INSTANTIATE_CAST_(uint64_t, int16_t);
    NOA_INSTANTIATE_CAST_(uint64_t, uint16_t);
    NOA_INSTANTIATE_CAST_(uint64_t, int32_t);
    NOA_INSTANTIATE_CAST_(uint64_t, uint32_t);
    NOA_INSTANTIATE_CAST_(uint64_t, int64_t);
    NOA_INSTANTIATE_CAST_(uint64_t, half_t);
    NOA_INSTANTIATE_CAST_(uint64_t, float);
    NOA_INSTANTIATE_CAST_(uint64_t, double);

    NOA_INSTANTIATE_CAST_(half_t, int8_t);
    NOA_INSTANTIATE_CAST_(half_t, uint8_t);
    NOA_INSTANTIATE_CAST_(half_t, int16_t);
    NOA_INSTANTIATE_CAST_(half_t, uint16_t);
    NOA_INSTANTIATE_CAST_(half_t, int32_t);
    NOA_INSTANTIATE_CAST_(half_t, uint32_t);
    NOA_INSTANTIATE_CAST_(half_t, int64_t);
    NOA_INSTANTIATE_CAST_(half_t, uint64_t);
    NOA_INSTANTIATE_CAST_(half_t, float);
    NOA_INSTANTIATE_CAST_(half_t, double);

    NOA_INSTANTIATE_CAST_(float, int8_t);
    NOA_INSTANTIATE_CAST_(float, uint8_t);
    NOA_INSTANTIATE_CAST_(float, int16_t);
    NOA_INSTANTIATE_CAST_(float, uint16_t);
    NOA_INSTANTIATE_CAST_(float, int32_t);
    NOA_INSTANTIATE_CAST_(float, uint32_t);
    NOA_INSTANTIATE_CAST_(float, int64_t);
    NOA_INSTANTIATE_CAST_(float, uint64_t);
    NOA_INSTANTIATE_CAST_(float, half_t);
    NOA_INSTANTIATE_CAST_(float, double);

    NOA_INSTANTIATE_CAST_(double, int8_t);
    NOA_INSTANTIATE_CAST_(double, uint8_t);
    NOA_INSTANTIATE_CAST_(double, int16_t);
    NOA_INSTANTIATE_CAST_(double, uint16_t);
    NOA_INSTANTIATE_CAST_(double, int32_t);
    NOA_INSTANTIATE_CAST_(double, uint32_t);
    NOA_INSTANTIATE_CAST_(double, int64_t);
    NOA_INSTANTIATE_CAST_(double, uint64_t);
    NOA_INSTANTIATE_CAST_(double, half_t);
    NOA_INSTANTIATE_CAST_(double, float);

    NOA_INSTANTIATE_CAST_(chalf_t, cfloat_t);
    NOA_INSTANTIATE_CAST_(chalf_t, cdouble_t);
    NOA_INSTANTIATE_CAST_(cfloat_t, chalf_t);
    NOA_INSTANTIATE_CAST_(cfloat_t, cdouble_t);
    NOA_INSTANTIATE_CAST_(cdouble_t, chalf_t);
    NOA_INSTANTIATE_CAST_(cdouble_t, cfloat_t);
}
