#include "noa/common/Math.h"
#include "noa/gpu/cuda/signal/Shape.h"

// TODO Add vectorized loads/stores?
namespace {
    using namespace noa;
    constexpr dim3 BLOCK_SIZE(32, 8);

    template<bool INVERT>
    __forceinline__ __device__ float getSoftMask_(float dst_sqd, float radius, float radius_sqd,
                                                  float radius_taper_sqd, float taper_size) {
        float mask_value;
        constexpr float PI = math::Constants<float>::PI;
        if constexpr (INVERT) {
            if (dst_sqd > radius_taper_sqd) {
                mask_value = 1.f;
            } else if (dst_sqd <= radius_sqd) {
                mask_value = 0.f;
            } else {
                float dst = math::sqrt(dst_sqd);
                mask_value = (1.f - math::cos(PI * (dst - radius) / taper_size)) * 0.5f;
            }
        } else {
            if (dst_sqd > radius_taper_sqd) {
                mask_value = 0.f;
            } else if (dst_sqd <= radius_sqd) {
                mask_value = 1.f;
            } else {
                dst_sqd = math::sqrt(dst_sqd);
                mask_value = (1.f + math::cos(PI * (dst_sqd - radius) / taper_size)) * 0.5f;
            }
        }
        return mask_value;
    }

    template<bool INVERT>
    __forceinline__ __device__ float getHardMask_(float dst_sqd, float radius_sqd) {
        float mask_value;
        if constexpr (INVERT) {
            if (dst_sqd > radius_sqd)
                mask_value = 1;
            else
                mask_value = 0;
        } else {
            if (dst_sqd > radius_sqd)
                mask_value = 0;
            else
                mask_value = 1;
        }
        return mask_value;
    }

    template<bool TAPER, bool INVERT, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void sphere_(const T* input, uint4_t input_stride, T* output, uint4_t output_stride,
                 uint2_t start, uint2_t end, uint batches,
                 float3_t center, float radius, float taper_size) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * BLOCK_SIZE.y + threadIdx.y + start[0],
                          blockIdx.x * BLOCK_SIZE.x + threadIdx.x + start[1]};
        if (gid[1] >= end[0] || gid[2] >= end[1])
            return;

        const float radius_sqd = radius * radius;
        const float3_t tmp{float3_t(gid) - center};
        const float dst_sqd = math::dot(tmp, tmp);

        float mask;
        if constexpr (TAPER) {
            float radius_taper_sqd = radius + taper_size;
            radius_taper_sqd *= radius_taper_sqd;
            mask = getSoftMask_<INVERT>(dst_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
        } else {
            mask = getHardMask_<INVERT>(dst_sqd, radius_sqd);
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
    template<bool INVERT, typename T, typename>
    void sphere(const shared_t<T[]>& input, size4_t input_stride,
                const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                float3_t center, float radius, float taper_size, Stream& stream) {
        uint3_t start{0}, end{shape.get(1)};
        if (INVERT && input.get() == output.get()) {
            start = uint3_t{noa::math::clamp(int3_t{center - (radius + taper_size)}, int3_t{}, int3_t{end})};
            end = uint3_t{noa::math::clamp(int3_t{center + (radius + taper_size) + 1}, int3_t{}, int3_t{end})};
            if (any(end <= start))
                return;
        }
        const uint3_t shape_{end - start};

        const dim3 blocks(math::divideUp(shape_[2], BLOCK_SIZE.x),
                          math::divideUp(shape_[1], BLOCK_SIZE.y),
                          shape_[0]);
        const LaunchConfig config{blocks, BLOCK_SIZE};
        const bool taper = taper_size > 1e-5f;
        stream.enqueue("signal::sphere", taper ? sphere_<true, INVERT, T> : sphere_<false, INVERT, T>, config,
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride},
                       uint2_t{start.get(1)}, uint2_t{end.get(1)}, shape[0],
                       center, radius, taper_size);
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_SPHERE_(T)                                                                                                              \
    template void sphere<true, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float, float, Stream&);    \
    template void sphere<false, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float, float, Stream&)

    NOA_INSTANTIATE_SPHERE_(half_t);
    NOA_INSTANTIATE_SPHERE_(float);
    NOA_INSTANTIATE_SPHERE_(double);
    NOA_INSTANTIATE_SPHERE_(chalf_t);
    NOA_INSTANTIATE_SPHERE_(cfloat_t);
    NOA_INSTANTIATE_SPHERE_(cdouble_t);
}
