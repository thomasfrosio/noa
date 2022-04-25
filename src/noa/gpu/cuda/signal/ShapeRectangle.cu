#include "noa/common/Profiler.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/signal/Shape.h"


namespace {
    using namespace noa;
    constexpr dim3 BLOCK_SIZE(32, 8);

    template<bool INVERT>
    __forceinline__ __device__ float getSoftMask_(float3_t distance, float3_t radius,
                                                  float3_t radius_with_taper, float taper_size) {
        constexpr float PI = math::Constants<float>::PI;
        float mask_value;
        if constexpr (INVERT) {
            if (any(radius_with_taper < distance)) {
                mask_value = 1.f;
            } else if (all(distance <= radius)) {
                mask_value = 0.f;
            } else {
                mask_value = 1.f;
                if (radius[0] < distance[0] && distance[0] <= radius_with_taper[0])
                    mask_value *= (1.f + math::cos(PI * (distance[0] - radius[0]) / taper_size)) * 0.5f;
                if (radius[1] < distance[1] && distance[1] <= radius_with_taper[1])
                    mask_value *= (1.f + math::cos(PI * (distance[1] - radius[1]) / taper_size)) * 0.5f;
                if (radius[2] < distance[2] && distance[2] <= radius_with_taper[2])
                    mask_value *= (1.f + math::cos(PI * (distance[2] - radius[2]) / taper_size)) * 0.5f;
                mask_value = 1.f - mask_value;
            }
        } else {
            if (any(radius_with_taper < distance)) {
                mask_value = 0.f;
            } else if (all(distance <= radius)) {
                mask_value = 1.f;
            } else {
                mask_value = 1.f;
                if (radius[0] < distance[0] && distance[0] <= radius_with_taper[0])
                    mask_value *= (1.f + math::cos(PI * (distance[0] - radius[0]) / taper_size)) * 0.5f;
                if (radius[1] < distance[1] && distance[1] <= radius_with_taper[1])
                    mask_value *= (1.f + math::cos(PI * (distance[1] - radius[1]) / taper_size)) * 0.5f;
                if (radius[2] < distance[2] && distance[2] <= radius_with_taper[2])
                    mask_value *= (1.f + math::cos(PI * (distance[2] - radius[2]) / taper_size)) * 0.5f;
            }
        }
        return mask_value;
    }

    template<bool INVERT>
    __forceinline__ __device__ float getHardMask_(float3_t distance, float3_t radius) {
        float mask_value;
        if constexpr (INVERT) {
            if (all(distance <= radius))
                mask_value = 0;
            else
                mask_value = 1;
        } else {
            if (all(distance <= radius))
                mask_value = 1;
            else
                mask_value = 0;
        }
        return mask_value;
    }

    template<bool TAPER, bool INVERT, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void rectangle_(const T* input, uint4_t input_stride, T* output, uint4_t output_stride,
                    uint2_t shape, uint batches,
                    float3_t center, float3_t radius, float taper_size) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * BLOCK_SIZE.y + threadIdx.y,
                          blockIdx.x * BLOCK_SIZE.x + threadIdx.x};
        if (gid[1] >= shape[0] || gid[2] >= shape[1])
            return;

        float3_t distance{gid};
        distance -= center;
        distance = math::abs(distance);

        float mask;
        if constexpr (TAPER) {
            const float3_t radius_taper = radius + taper_size;
            mask = getSoftMask_<INVERT>(distance, radius, radius_taper, taper_size);
        } else {
            mask = getHardMask_<INVERT>(distance, radius);
            (void) taper_size;
        }

        using real_t = traits::value_type_t<T>;
        const uint offset = gid[0] * input_stride[1] + gid[1] * input_stride[2] + gid[2] * input_stride[3];
        output += gid[0] * output_stride[1] + gid[1] * output_stride[2] + gid[2] * output_stride[3];
        for (uint batch = 0; batch < batches; ++batch) {
            output[batch * output_stride[0]] =
                    input ?
                    input[batch * input_stride[0] + offset] * static_cast<real_t>(mask) :
                    static_cast<real_t>(mask);
        }
    }
}

namespace noa::cuda::signal {
    template<bool INVERT, typename T>
    void rectangle(const shared_t<T[]>& input, size4_t input_stride,
                   const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                   float3_t center, float3_t radius, float taper_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const uint2_t u_shape{shape.get() + 2};
        const bool taper = taper_size > 1e-5f;
        const dim3 blocks(math::divideUp(u_shape[1], BLOCK_SIZE.x),
                          math::divideUp(u_shape[0], BLOCK_SIZE.y),
                          shape[1]);
        const LaunchConfig config{blocks, BLOCK_SIZE};
        stream.enqueue("filter::rectangle", taper ? rectangle_<true, INVERT, T> : rectangle_<false, INVERT, T>, config,
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride}, u_shape, shape[0],
                       center, radius, taper_size);
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_RECTANGLE_(T)                                                                                                           \
    template void rectangle<true, T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float3_t, float, Stream&);    \
    template void rectangle<false, T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_RECTANGLE_(half_t);
    NOA_INSTANTIATE_RECTANGLE_(float);
    NOA_INSTANTIATE_RECTANGLE_(double);
    NOA_INSTANTIATE_RECTANGLE_(chalf_t);
    NOA_INSTANTIATE_RECTANGLE_(cfloat_t);
    NOA_INSTANTIATE_RECTANGLE_(cdouble_t);
}
