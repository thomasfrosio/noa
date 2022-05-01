#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/signal/Shape.h"

namespace {
    using namespace noa;
    constexpr dim3 BLOCK_SIZE(32, 8);

    template<bool INVERT>
    inline __device__ float getSoftMask_(float dst_yx_sqd, float radius_sqd, float radius,
                                         float radius_sqd_with_taper, float dst_z, float length,
                                         float length_with_taper, float taper_size) {
        constexpr float PI = math::Constants<float>::PI;
        float mask;
        if constexpr (INVERT) {
            if (dst_z > length_with_taper || dst_yx_sqd > radius_sqd_with_taper) {
                mask = 1.f;
            } else {
                if (dst_yx_sqd <= radius_sqd) {
                    mask = 1.f;
                } else {
                    float dst_yx = math::sqrt(dst_yx_sqd);
                    mask = (1.f + math::cos(PI * (dst_yx - radius) / taper_size)) * 0.5f;
                }
                if (dst_z > length)
                    mask *= (1.f + math::cos(PI * (dst_z - length) / taper_size)) * 0.5f;
                mask = 1 - mask;
            }
        } else {
            if (dst_z > length_with_taper || dst_yx_sqd > radius_sqd_with_taper) {
                mask = 0.f;
            } else {
                if (dst_yx_sqd <= radius_sqd) {
                    mask = 1.f;
                } else {
                    float dst_yx = math::sqrt(dst_yx_sqd);
                    mask = (1.f + math::cos(PI * (dst_yx - radius) / taper_size)) * 0.5f;
                }
                if (dst_z > length)
                    mask *= (1.f + math::cos(PI * (dst_z - length) / taper_size)) * 0.5f;
            }
        }
        return mask;
    }

    template<bool INVERT>
    inline __device__ float getHardMask_(float distance_yx_sqd, float radius_sqd, float distance_z, float length) {
        float mask;
        if constexpr (INVERT) {
            if (distance_z > length || distance_yx_sqd > radius_sqd)
                mask = 1.f;
            else
                mask = 0.f;
        } else {
            if (distance_z > length || distance_yx_sqd > radius_sqd)
                mask = 0.f;
            else
                mask = 1.f;
        }
        return mask;
    }

    template<bool TAPER, bool INVERT, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void cylinder_(const T* input, uint4_t input_stride, T* output, uint4_t output_stride,
                   uint2_t start, uint2_t end, uint batches,
                   float3_t center, float radius, float length, float taper_size) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * BLOCK_SIZE.y + threadIdx.y + start[0],
                          blockIdx.x * BLOCK_SIZE.x + threadIdx.x + start[1]};
        if (gid[1] >= end[0] || gid[2] >= end[1])
            return;

        const float radius_sqd = radius * radius;
        const float dst_z = math::abs(static_cast<float>(gid[0]) - center[0]);
        float2_t tmp{gid[1], gid[2]};
        tmp -= float2_t{center[1], center[2]};
        const float dst_yx_sqd = math::dot(tmp, tmp);

        float mask;
        if constexpr (TAPER) {
            const float length_taper = length + taper_size;
            float radius_taper_sqd = radius + taper_size;
            radius_taper_sqd *= radius_taper_sqd;
            mask = getSoftMask_<INVERT>(dst_yx_sqd, radius_sqd, radius, radius_taper_sqd,
                                        dst_z, length, length_taper, taper_size);
        } else {
            mask = getHardMask_<INVERT>(dst_yx_sqd, radius_sqd, dst_z, length);
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
    void cylinder(const shared_t<T[]>& input, size4_t input_stride,
                  const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                  float3_t center, float radius, float length, float taper_size, Stream& stream) {
        uint3_t start{0}, end{shape.get() + 1};
        if (INVERT && input.get() == output.get()) {
            float3_t radius_{length, radius, radius};
            radius_ += taper_size;
            start = uint3_t{noa::math::clamp(int3_t{center - radius_}, int3_t{}, int3_t{end})};
            end = uint3_t{noa::math::clamp(int3_t{center + radius_ + 1}, int3_t{}, int3_t{end})};
            if (any(end <= start))
                return;
        }
        const uint3_t shape_{end - start};
        const dim3 blocks(math::divideUp(shape_[2], BLOCK_SIZE.x),
                          math::divideUp(shape_[1], BLOCK_SIZE.y),
                          shape_[0]);
        const LaunchConfig config{blocks, BLOCK_SIZE};

        const bool taper = taper_size > 1e-5f;
        stream.enqueue("signal::cylinder", taper ? cylinder_<true, INVERT, T> : cylinder_<false, INVERT, T>, config,
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride},
                       uint2_t{start.get() + 1}, uint2_t{end.get() + 1}, shape[0],
                       center, radius, length, taper_size);
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_CYLINDER_(T)                                                                                                                    \
    template void cylinder<true, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float, float, float, Stream&);   \
    template void cylinder<false, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float, float, float, Stream&)

    NOA_INSTANTIATE_CYLINDER_(half_t);
    NOA_INSTANTIATE_CYLINDER_(float);
    NOA_INSTANTIATE_CYLINDER_(double);
    NOA_INSTANTIATE_CYLINDER_(chalf_t);
    NOA_INSTANTIATE_CYLINDER_(cfloat_t);
    NOA_INSTANTIATE_CYLINDER_(cdouble_t);
}
