#include "noa/common/Profiler.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/filter/Shape.h"

// Soft edges:
namespace {
    using namespace noa;
    constexpr dim3 THREADS(32, 8);

    template<bool INVERT>
    __forceinline__ __device__ float getSoftMask_(float distance_sqd, float radius, float radius_sqd,
                                                  float radius_taper_sqd, float taper_size) {
        float mask_value;
        constexpr float PI = math::Constants<float>::PI;
        if constexpr (INVERT) {
            if (distance_sqd > radius_taper_sqd) {
                mask_value = 1.f;
            } else if (distance_sqd <= radius_sqd) {
                mask_value = 0.f;
            } else {
                float distance = math::sqrt(distance_sqd);
                mask_value = (1.f - math::cos(PI * (distance - radius) / taper_size)) * 0.5f;
            }
        } else {
            if (distance_sqd > radius_taper_sqd) {
                mask_value = 0.f;
            } else if (distance_sqd <= radius_sqd) {
                mask_value = 1.f;
            } else {
                distance_sqd = math::sqrt(distance_sqd);
                mask_value = (1.f + math::cos(PI * (distance_sqd - radius) / taper_size)) * 0.5f;
            }
        }
        return mask_value;
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void sphereSoft3D_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch,
                       uint3_t shape, uint batches, float3_t center, float radius, float taper_size) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float radius_sqd = radius * radius;
        float radius_taper_sqd = radius + taper_size;
        radius_taper_sqd *= radius_taper_sqd;

        float3_t tmp(float3_t(gid) - center);
        float distance_sqd = math::dot(tmp, tmp);
        float mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);

        using real_t = traits::value_type_t<T>;
        inputs += (gid.z * shape.y + gid.y) * input_pitch + gid.x;
        outputs += (gid.z * shape.y + gid.y) * output_pitch + gid.x;
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            outputs[batch * output_pitch * rows(shape)] =
                    inputs[batch * input_pitch * rows(shape)] * static_cast<real_t>(mask_value);
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void sphereSoft2D_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch,
                       uint2_t shape, uint batches, float2_t center, float radius, float taper_size) {
        const uint2_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float radius_sqd = radius * radius;
        float radius_taper_sqd = radius + taper_size;
        radius_taper_sqd *= radius_taper_sqd;

        float2_t tmp(float2_t(gid) - center);
        float distance_sqd = math::dot(tmp, tmp);
        float mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);

        using real_t = traits::value_type_t<T>;
        inputs += gid.y * input_pitch + gid.x;
        outputs += gid.y * output_pitch + gid.x;
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            outputs[batch * output_pitch * shape.y] =
                    inputs[batch * input_pitch * shape.y] * static_cast<real_t>(mask_value);
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void sphereSoft3D_(T* output_mask, uint output_mask_pitch,
                       uint3_t shape, uint batches, float3_t center, float radius, float taper_size) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float radius_sqd = radius * radius;
        float radius_taper_sqd = radius + taper_size;
        radius_taper_sqd *= radius_taper_sqd;

        float3_t tmp(float3_t(gid) - center);
        float distance_sqd = math::dot(tmp, tmp);
        float mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);

        using real_t = traits::value_type_t<T>;
        output_mask += (gid.z * shape.y + gid.y) * output_mask_pitch + gid.x;
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            output_mask[batch * output_mask_pitch * rows(shape)] = static_cast<real_t>(mask_value);
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void sphereSoft2D_(T* output_mask, uint output_mask_pitch,
                       uint2_t shape, uint batches, float2_t center, float radius, float taper_size) {
        const uint2_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float radius_sqd = radius * radius;
        float radius_taper_sqd = radius + taper_size;
        radius_taper_sqd *= radius_taper_sqd;

        float2_t tmp(float2_t(gid) - center);
        float distance_sqd = math::dot(tmp, tmp);
        float mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);

        using real_t = traits::value_type_t<T>;
        output_mask += gid.y * output_mask_pitch + gid.x;
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            output_mask[batch * output_mask_pitch * shape.y] = static_cast<real_t>(mask_value);
    }
}

// Hard edges:
namespace {
    using namespace noa;

    template<bool INVERT>
    __forceinline__ __device__ float getHardMask_(float distance_sqd, float radius_sqd) {
        float mask_value;
        if constexpr (INVERT) {
            if (distance_sqd > radius_sqd)
                mask_value = 1;
            else
                mask_value = 0;
        } else {
            if (distance_sqd > radius_sqd)
                mask_value = 0;
            else
                mask_value = 1;
        }
        return mask_value;
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void sphereHard3D_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch,
                       uint3_t shape, uint batches, float3_t center, float radius) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float3_t tmp(float3_t(gid) - center);
        float distance_sqd = math::dot(tmp, tmp);
        float mask_value = getHardMask_<INVERT>(distance_sqd, radius * radius);

        using real_t = traits::value_type_t<T>;
        inputs += (gid.z * shape.y + gid.y) * input_pitch + gid.x;
        outputs += (gid.z * shape.y + gid.y) * output_pitch + gid.x;
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            outputs[batch * output_pitch * rows(shape)] =
                    inputs[batch * input_pitch * rows(shape)] * static_cast<real_t>(mask_value);
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void sphereHard2D_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch,
                       uint2_t shape, uint batches, float2_t center, float radius) {
        const uint2_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float2_t tmp(float2_t(gid) - center);
        float distance_sqd = math::dot(tmp, tmp);
        float mask_value = getHardMask_<INVERT>(distance_sqd, radius * radius);

        using real_t = traits::value_type_t<T>;
        inputs += gid.y * input_pitch + gid.x;
        outputs += gid.y * output_pitch + gid.x;
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            outputs[batch * output_pitch * shape.y] =
                    inputs[batch * input_pitch * shape.y] * static_cast<real_t>(mask_value);
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void sphereHard3D_(T* output_mask, uint output_mask_pitch,
                       uint3_t shape, uint batches, float3_t center, float radius) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float3_t tmp(float3_t(gid) - center);
        float distance_sqd = math::dot(tmp, tmp);
        float mask_value = getHardMask_<INVERT>(distance_sqd, radius * radius);

        using real_t = traits::value_type_t<T>;
        output_mask += (gid.z * shape.y + gid.y) * output_mask_pitch + gid.x;
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            output_mask[batch * output_mask_pitch * rows(shape)] = static_cast<real_t>(mask_value);
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void sphereHard2D_(T* output_mask, uint output_mask_pitch, uint2_t shape, uint batches,
                       float2_t center, float radius) {
        const uint2_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float2_t tmp(float2_t(gid) - center);
        float distance_sqd = math::dot(tmp, tmp);
        float mask_value = getHardMask_<INVERT>(distance_sqd, radius * radius);

        using real_t = traits::value_type_t<T>;
        output_mask += gid.y * output_mask_pitch + gid.x;
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            output_mask[batch * output_mask_pitch * shape.y] = static_cast<real_t>(mask_value);
    }
}

namespace noa::cuda::filter {
    template<bool INVERT, typename T>
    void sphere2D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                  size2_t shape, size_t batches,
                  float2_t center, float radius, float taper_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint2_t u_shape(shape);
        dim3 blocks(math::divideUp(u_shape.x, THREADS.x),
                    math::divideUp(u_shape.y, THREADS.y));
        if (inputs) {
            if (taper_size > 1e-5f) {
                sphereSoft2D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, input_pitch, outputs, output_pitch,
                        u_shape, batches, center, radius, taper_size);
            } else {
                sphereHard2D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, input_pitch, outputs, output_pitch,
                        u_shape, batches, center, radius);
            }
        } else {
            if (taper_size > 1e-5f) {
                sphereSoft2D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        outputs, output_pitch, u_shape, batches, center, radius, taper_size);

            } else {
                sphereHard2D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        outputs, output_pitch, u_shape, batches, center, radius);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<bool INVERT, typename T>
    void sphere3D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                  size3_t shape, size_t batches,
                  float3_t center, float radius, float taper_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint3_t u_shape(shape);
        dim3 blocks(math::divideUp(u_shape.x, THREADS.x),
                    math::divideUp(u_shape.y, THREADS.y),
                    u_shape.z);
        if (inputs) {
            if (taper_size > 1e-5f) {
                sphereSoft3D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, input_pitch, outputs, output_pitch,
                        u_shape, batches, center, radius, taper_size);
            } else {
                sphereHard3D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, input_pitch, outputs, output_pitch,
                        u_shape, batches, center, radius);
            }
        } else {
            if (taper_size > 1e-5f) {
                sphereSoft3D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        outputs, output_pitch, u_shape, batches, center, radius, taper_size);

            } else {
                sphereHard3D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        outputs, output_pitch, u_shape, batches, center, radius);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define NOA_INSTANTIATE_SPHERE_(T)                                                                                  \
    template void sphere2D<true, T>(const T*, size_t, T*, size_t, size2_t, size_t, float2_t, float, float, Stream&);    \
    template void sphere2D<false, T>(const T*, size_t, T*, size_t, size2_t, size_t, float2_t, float, float, Stream&);   \
    template void sphere3D<true, T>(const T*, size_t, T*, size_t, size3_t, size_t, float3_t, float, float, Stream&);    \
    template void sphere3D<false, T>(const T*, size_t, T*, size_t, size3_t, size_t, float3_t, float, float, Stream&)

    NOA_INSTANTIATE_SPHERE_(float);
    NOA_INSTANTIATE_SPHERE_(double);
    NOA_INSTANTIATE_SPHERE_(cfloat_t);
    NOA_INSTANTIATE_SPHERE_(cdouble_t);
}
