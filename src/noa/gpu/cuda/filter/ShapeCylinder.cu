#include "noa/common/Profiler.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/filter/Shape.h"

// Soft edges:
namespace {
    using namespace noa;
    constexpr dim3 THREADS(32, 8);

    template<bool INVERT>
    inline __device__ float getSoftMask_(float distance_xy_sqd, float radius_xy_sqd,
                                         float radius_xy, float radius_xy_sqd_with_taper,
                                         float distance_z, float radius_z, float radius_z_with_taper,
                                         float taper_size) {
        constexpr float PI = math::Constants<float>::PI;
        float mask_value;
        if constexpr (INVERT) {
            if (distance_z > radius_z_with_taper || distance_xy_sqd > radius_xy_sqd_with_taper) {
                mask_value = 1.f;
            } else {
                if (distance_xy_sqd <= radius_xy_sqd) {
                    mask_value = 1.f;
                } else {
                    float distance_xy = math::sqrt(distance_xy_sqd);
                    mask_value = (1.f + math::cos(PI * (distance_xy - radius_xy) / taper_size)) * 0.5f;
                }
                if (distance_z > radius_z)
                    mask_value *= (1.f + math::cos(PI * (distance_z - radius_z) / taper_size)) * 0.5f;
                mask_value = 1 - mask_value;
            }
        } else {
            if (distance_z > radius_z_with_taper || distance_xy_sqd > radius_xy_sqd_with_taper) {
                mask_value = 0.f;
            } else {
                if (distance_xy_sqd <= radius_xy_sqd) {
                    mask_value = 1.f;
                } else {
                    float distance_xy = math::sqrt(distance_xy_sqd);
                    mask_value = (1.f + math::cos(PI * (distance_xy - radius_xy) / taper_size)) * 0.5f;
                }
                if (distance_z > radius_z)
                    mask_value *= (1.f + math::cos(PI * (distance_z - radius_z) / taper_size)) * 0.5f;
            }
        }
        return mask_value;
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void cylinderSoft_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch,
                       uint3_t shape, uint batches,
                       float3_t center, float radius_xy, float radius_z, float taper_size) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        const float radius_z_taper = radius_z + taper_size;
        const float radius_xy_sqd = radius_xy * radius_xy;
        float radius_xy_taper_sqd = radius_xy + taper_size;
        radius_xy_taper_sqd *= radius_xy_taper_sqd;

        const float distance_z = math::abs(static_cast<float>(gid.z) - center.z);
        float2_t tmp(gid.x, gid.y);
        tmp -= float2_t(center.x, center.y);
        const float distance_xy_sqd = math::dot(tmp, tmp);
        const float mask_value = getSoftMask_<INVERT>(distance_xy_sqd, radius_xy_sqd, radius_xy, radius_xy_taper_sqd,
                                                      distance_z, radius_z, radius_z_taper, taper_size);

        using real_t = traits::value_type_t<T>;
        inputs += (gid.z * shape.y + gid.y) * input_pitch + gid.x;
        outputs += (gid.z * shape.y + gid.y) * output_pitch + gid.x;

        for (uint batch = 0; batch < batches; ++batch)
            outputs[batch * output_pitch * rows(shape)] =
                    inputs[batch * input_pitch * rows(shape)] * static_cast<real_t>(mask_value);
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void cylinderSoft_(T* output_mask, uint output_mask_pitch,
                       uint3_t shape, uint batches,
                       float3_t center, float radius_xy, float radius_z, float taper_size) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        const float radius_z_taper = radius_z + taper_size;
        const float radius_xy_sqd = radius_xy * radius_xy;
        float radius_xy_taper_sqd = radius_xy + taper_size;
        radius_xy_taper_sqd *= radius_xy_taper_sqd;

        const float distance_z = math::abs(static_cast<float>(gid.z) - center.z);
        float2_t tmp(gid.x, gid.y);
        tmp -= float2_t(center.x, center.y);
        const float distance_xy_sqd = math::dot(tmp, tmp);
        const float mask_value = getSoftMask_<INVERT>(distance_xy_sqd, radius_xy_sqd, radius_xy, radius_xy_taper_sqd,
                                                      distance_z, radius_z, radius_z_taper, taper_size);

        using real_t = traits::value_type_t<T>;
        output_mask += (gid.z * shape.y + gid.y) * output_mask_pitch + gid.x;

        for (uint batch = 0; batch < batches; ++batch)
            output_mask[batch * output_mask_pitch * rows(shape)] = static_cast<real_t>(mask_value);
    }
}

// Hard edges:
namespace {
    using namespace noa;

    template<bool INVERT, typename T>
    __forceinline__ __device__ T getHardMask_(float distance_xy_sqd, float radius_xy_sqd,
                                              float distance_z, float radius_z) {
        T mask_value;
        if constexpr (INVERT) {
            if (distance_z > radius_z || distance_xy_sqd > radius_xy_sqd)
                mask_value = 1;
            else
                mask_value = 0;
        } else {
            if (distance_z > radius_z || distance_xy_sqd > radius_xy_sqd)
                mask_value = 0;
            else
                mask_value = 1;
        }
        return mask_value;
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void cylinderHard_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch,
                       uint3_t shape, uint batches, float3_t center,
                       float radius_xy, float radius_z) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        using real_t = traits::value_type_t<T>;
        const float radius_xy_sqd = radius_xy * radius_xy;
        const float distance_z = math::abs(static_cast<float>(gid.z) - center.z);
        float2_t tmp(gid.x, gid.y);
        tmp -= float2_t(center.x, center.y);
        const float distance_xy_sqd = math::dot(tmp, tmp);
        const auto mask_value = getHardMask_<INVERT, real_t>(distance_xy_sqd, radius_xy_sqd, distance_z, radius_z);

        inputs += (gid.z * shape.y + gid.y) * input_pitch + gid.x;
        outputs += (gid.z * shape.y + gid.y) * output_pitch + gid.x;

        for (uint batch = 0; batch < batches; ++batch)
            outputs[batch * output_pitch * rows(shape)] =
                    inputs[batch * input_pitch * rows(shape)] * mask_value;
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void cylinderHard_(T* output_mask, uint output_mask_pitch,
                       uint3_t shape, uint batches,
                       float3_t center, float radius_xy, float radius_z) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        using real_t = traits::value_type_t<T>;
        const float radius_xy_sqd = radius_xy * radius_xy;
        const float distance_z = math::abs(static_cast<float>(gid.z) - center.z);
        float2_t tmp(gid.x, gid.y);
        tmp -= float2_t(center.x, center.y);
        const float distance_xy_sqd = math::dot(tmp, tmp);
        const auto mask_value = getHardMask_<INVERT, real_t>(distance_xy_sqd, radius_xy_sqd, distance_z, radius_z);

        output_mask += (gid.z * shape.y + gid.y) * output_mask_pitch + gid.x;

        for (uint batch = 0; batch < batches; ++batch)
            output_mask[batch * output_mask_pitch * rows(shape)] = mask_value;
    }
}

namespace noa::cuda::filter {
    template<bool INVERT, typename T>
    void cylinder3D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                    size3_t shape, size_t batches,
                    float3_t center, float radius_xy, float radius_z, float taper_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const uint3_t u_shape(shape);
        const dim3 blocks(math::divideUp(u_shape.x, THREADS.x),
                          math::divideUp(u_shape.y, THREADS.y),
                          u_shape.z);
        if (inputs) {
            if (taper_size > 1e-5f) {
                cylinderSoft_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, input_pitch, outputs, output_pitch,
                        u_shape, batches, center, radius_xy, radius_z, taper_size);
            } else {
                cylinderHard_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, input_pitch, outputs, output_pitch,
                        u_shape, batches, center, radius_xy, radius_z);
            }
        } else {
            if (taper_size > 1e-5f) {
                cylinderSoft_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        outputs, output_pitch, u_shape, batches, center, radius_xy, radius_z, taper_size);

            } else {
                cylinderHard_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        outputs, output_pitch, u_shape, batches, center, radius_xy, radius_z);
            }
        }
        NOA_THROW_IF(cudaGetLastError());
    }

    #define NOA_INSTANTIATE_CYLINDER_(T)                                                                                        \
    template void cylinder3D<true, T>(const T*, size_t, T*, size_t, size3_t, size_t, float3_t, float, float, float, Stream&);   \
    template void cylinder3D<false, T>(const T*, size_t, T*, size_t, size3_t, size_t, float3_t, float, float, float, Stream&)

    NOA_INSTANTIATE_CYLINDER_(float);
    NOA_INSTANTIATE_CYLINDER_(double);
    NOA_INSTANTIATE_CYLINDER_(cfloat_t);
    NOA_INSTANTIATE_CYLINDER_(cdouble_t);
}
