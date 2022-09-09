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
    void rectangle_(const T* input, uint4_t input_strides, T* output, uint4_t output_strides,
                    uint3_t start, uint2_t end, uint batches,
                    float3_t center, float3_t radius, float taper_size) {
        const uint3_t gid{blockIdx.z + start[0],
                          blockIdx.y * BLOCK_SIZE.y + threadIdx.y + start[1],
                          blockIdx.x * BLOCK_SIZE.x + threadIdx.x + start[2]};
        if (gid[1] >= end[0] || gid[2] >= end[1])
            return;

        float3_t distance(gid);
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
        const uint offset = gid[0] * input_strides[1] + gid[1] * input_strides[2] + gid[2] * input_strides[3];
        output += gid[0] * output_strides[1] + gid[1] * output_strides[2] + gid[2] * output_strides[3];
        for (uint batch = 0; batch < batches; ++batch) {
            output[batch * output_strides[0]] =
                    input ?
                    input[batch * input_strides[0] + offset] * static_cast<real_t>(mask) :
                    static_cast<real_t>(mask);
        }
    }
}

namespace noa::cuda::signal {
    template<bool INVERT, typename T, typename>
    void rectangle(const shared_t<T[]>& input, size4_t input_strides,
                   const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                   float3_t center, float3_t radius, float taper_size, Stream& stream) {
        const size3_t order_3d = indexing::order(size3_t(output_strides.get(1)), size3_t(shape.get(1)));
        if (any(order_3d != size3_t{0, 1, 2})) {
            const size4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
        }

        uint3_t start{0}, end(shape.get(1));
        if (INVERT && input.get() == output.get()) {
            start = uint3_t(noa::math::clamp(int3_t(center - (radius + taper_size)), int3_t{}, int3_t(end)));
            end = uint3_t(noa::math::clamp(int3_t(center + (radius + taper_size) + 1), int3_t{}, int3_t(end)));
            if (any(end <= start))
                return;
        }
        const uint3_t shape_(end - start);
        const dim3 blocks(math::divideUp(shape_[2], BLOCK_SIZE.x),
                          math::divideUp(shape_[1], BLOCK_SIZE.y),
                          shape_[0]);
        const LaunchConfig config{blocks, BLOCK_SIZE};

        const bool taper = taper_size > 1e-5f;
        stream.enqueue("signal::rectangle", taper ? rectangle_<true, INVERT, T> : rectangle_<false, INVERT, T>, config,
                       input.get(), uint4_t(input_strides), output.get(), uint4_t(output_strides),
                       start, uint2_t(end.get(1)), shape[0],
                       center, radius, taper_size);
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_RECTANGLE_(T)                                                                                                               \
    template void rectangle<true, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float3_t, float, Stream&);  \
    template void rectangle<false, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_RECTANGLE_(half_t);
    NOA_INSTANTIATE_RECTANGLE_(float);
    NOA_INSTANTIATE_RECTANGLE_(double);
    NOA_INSTANTIATE_RECTANGLE_(chalf_t);
    NOA_INSTANTIATE_RECTANGLE_(cfloat_t);
    NOA_INSTANTIATE_RECTANGLE_(cdouble_t);
}
