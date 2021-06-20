#include "noa/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/mask/Sphere.h"

// Soft edges:
namespace {
    using namespace noa;

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
    __global__ void sphereSoft3D_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                  uint3_t shape, float3_t center, float radius, float taper_size, uint batches) {
        uint y = blockIdx.x, z = blockIdx.y;

        uint offset_inputs = (z * shape.y + y) * inputs_pitch;
        uint offset_outputs = (z * shape.y + y) * outputs_pitch;

        uint rows = getRows(shape);
        uint elements_inputs = inputs_pitch * rows;
        uint elements_outputs = outputs_pitch * rows;

        float radius_sqd = radius * radius;
        float tmp_distance_sqd = static_cast<float>(z) - center.z;
        tmp_distance_sqd *= tmp_distance_sqd;
        float tmp_y = static_cast<float>(y) - center.y;
        tmp_distance_sqd += tmp_y * tmp_y;

        float radius_taper_sqd = radius + taper_size;
        radius_taper_sqd *= radius_taper_sqd;

        float distance_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_sqd = static_cast<float>(x) - center.x;
            distance_sqd *= distance_sqd;
            distance_sqd += tmp_distance_sqd;

            mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);

            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + offset_outputs + x] =
                        inputs[batch * elements_inputs + offset_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void sphereSoft2D_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                  uint2_t shape, float2_t center, float radius, float taper_size, uint batches) {
        uint y = blockIdx.x;

        uint offset_inputs = y * inputs_pitch;
        uint offset_outputs = y * outputs_pitch;
        uint elements_inputs = inputs_pitch * shape.y;
        uint elements_outputs = outputs_pitch * shape.y;

        float radius_sqd = radius * radius;
        float tmp_distance_sqd = static_cast<float>(y) - center.y;
        tmp_distance_sqd *= tmp_distance_sqd;

        float radius_taper_sqd = radius + taper_size;
        radius_taper_sqd *= radius_taper_sqd;

        float distance_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_sqd = static_cast<float>(x) - center.x;
            distance_sqd *= distance_sqd;
            distance_sqd += tmp_distance_sqd;

            mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);

            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + offset_outputs + x] =
                        inputs[batch * elements_inputs + offset_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void sphereSoft3D_(T* output_mask, uint output_mask_pitch,
                                  uint3_t shape, float3_t center, float radius, float taper_size) {
        uint y = blockIdx.x, z = blockIdx.y;
        uint offset = (z * shape.y + y) * output_mask_pitch;

        float radius_sqd = radius * radius;
        float tmp_distance_sqd = static_cast<float>(z) - center.z;
        tmp_distance_sqd *= tmp_distance_sqd;
        float tmp_y = static_cast<float>(y) - center.y;
        tmp_distance_sqd += tmp_y * tmp_y;

        float radius_taper_sqd = radius + taper_size;
        radius_taper_sqd *= radius_taper_sqd;

        float distance_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_sqd = static_cast<float>(x) - center.x;
            distance_sqd *= distance_sqd;
            distance_sqd += tmp_distance_sqd;
            mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
            output_mask[offset + x] = static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void sphereSoft2D_(T* output_mask, uint output_mask_pitch,
                                  uint shape_x, float2_t center, float radius, float taper_size) {
        uint y = blockIdx.x;
        uint offset = y * output_mask_pitch;

        float radius_sqd = radius * radius;
        float tmp_distance_sqd = static_cast<float>(y) - center.y;
        tmp_distance_sqd *= tmp_distance_sqd;

        float radius_taper_sqd = radius + taper_size;
        radius_taper_sqd *= radius_taper_sqd;

        float distance_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape_x; x += blockDim.x) {
            distance_sqd = static_cast<float>(x) - center.x;
            distance_sqd *= distance_sqd;
            distance_sqd += tmp_distance_sqd;
            mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
            output_mask[offset + x] = static_cast<T>(mask_value);
        }
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
    __global__ void sphereHard3D_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                  uint3_t shape, float3_t center, float radius, uint batches) {
        uint y = blockIdx.x, z = blockIdx.y;

        uint offset_inputs = (z * shape.y + y) * inputs_pitch;
        uint offset_outputs = (z * shape.y + y) * outputs_pitch;

        uint rows = getRows(shape);
        uint elements_inputs = inputs_pitch * rows;
        uint elements_outputs = outputs_pitch * rows;

        float radius_sqd = radius * radius;
        float tmp_distance_sqd = static_cast<float>(z) - center.z;
        tmp_distance_sqd *= tmp_distance_sqd;
        float tmp_y = static_cast<float>(y) - center.y;
        tmp_distance_sqd += tmp_y * tmp_y;

        float distance_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_sqd = static_cast<float>(x) - center.x;
            distance_sqd *= distance_sqd;
            distance_sqd += tmp_distance_sqd;

            mask_value = getHardMask_<INVERT>(distance_sqd, radius_sqd);

            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + offset_outputs + x] =
                        inputs[batch * elements_inputs + offset_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void sphereHard2D_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                  uint2_t shape, float2_t center, float radius, uint batches) {
        uint y = blockIdx.x;

        uint offset_inputs = y * inputs_pitch;
        uint offset_outputs = y * outputs_pitch;

        uint elements_inputs = inputs_pitch * shape.y;
        uint elements_outputs = outputs_pitch * shape.y;

        float radius_sqd = radius * radius;
        float tmp_distance_sqd = static_cast<float>(y) - center.y;
        tmp_distance_sqd *= tmp_distance_sqd;

        float distance_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_sqd = static_cast<float>(x) - center.x;
            distance_sqd *= distance_sqd;
            distance_sqd += tmp_distance_sqd;

            mask_value = getHardMask_<INVERT>(distance_sqd, radius_sqd);

            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + offset_outputs + x] =
                        inputs[batch * elements_inputs + offset_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void sphereHard3D_(T* output_mask, uint output_mask_pitch,
                                  uint3_t shape, float3_t center, float radius) {
        uint y = blockIdx.x, z = blockIdx.y;
        uint offset = (z * shape.y + y) * output_mask_pitch;

        float radius_sqd = radius * radius;
        float tmp_distance_sqd = static_cast<float>(z) - center.z;
        tmp_distance_sqd *= tmp_distance_sqd;
        float tmp_y = static_cast<float>(y) - center.y;
        tmp_distance_sqd += tmp_y * tmp_y;

        float distance_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_sqd = static_cast<float>(x) - center.x;
            distance_sqd *= distance_sqd;
            distance_sqd += tmp_distance_sqd;
            mask_value = getHardMask_<INVERT>(distance_sqd, radius_sqd);
            output_mask[offset + x] = static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void sphereHard2D_(T* output_mask, uint output_mask_pitch, uint shape_x, float2_t center, float radius) {
        uint y = blockIdx.x;
        uint offset = y * output_mask_pitch;

        float radius_sqd = radius * radius;
        float tmp_distance_sqd = static_cast<float>(y) - center.y;
        tmp_distance_sqd *= tmp_distance_sqd;

        float distance_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape_x; x += blockDim.x) {
            distance_sqd = static_cast<float>(x) - center.x;
            distance_sqd *= distance_sqd;
            distance_sqd += tmp_distance_sqd;
            mask_value = getHardMask_<INVERT>(distance_sqd, radius_sqd);
            output_mask[offset + x] = static_cast<T>(mask_value);
        }
    }
}

namespace noa::cuda::mask {
    template<bool INVERT, typename T>
    void sphere(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch, size3_t shape, float3_t shifts,
                float radius, float taper_size, uint batches, Stream& stream) {
        uint3_t tmp_shape(shape);
        float3_t center(tmp_shape / 2U);
        center += shifts;

        uint threads = math::min(128U, math::nextMultipleOf(tmp_shape.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        uint ndim = getNDim(shape);
        if (ndim == 3) {
            if (taper_size > 1e-5f) {
                sphereSoft3D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        inputs, inputs_pitch, outputs, outputs_pitch,
                        tmp_shape, center, radius, taper_size, batches);
            } else {
                sphereHard3D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        inputs, inputs_pitch, outputs, outputs_pitch,
                        tmp_shape, center, radius, batches);
            }
        } else if (ndim == 2) {
            uint2_t shape_2D(shape.x, shape.y);
            float2_t center_2D(center.x, center.y);
            if (taper_size > 1e-5f) {
                sphereSoft2D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        inputs, inputs_pitch, outputs, outputs_pitch,
                        shape_2D, center_2D, radius, taper_size, batches);
            } else {
                sphereHard2D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        inputs, inputs_pitch, outputs, outputs_pitch,
                        shape_2D, center_2D, radius, batches);
            }
        } else {
            NOA_THROW("Cannot compute a sphere with shape:{}", shape);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<bool INVERT, typename T>
    void sphere(T* output_mask, size_t output_mask_pitch, size3_t shape, float3_t shifts,
                float radius, float taper_size, Stream& stream) {
        uint3_t tmp_shape(shape);
        float3_t center(tmp_shape / 2U);
        center += shifts;

        uint threads = math::min(128U, math::nextMultipleOf(tmp_shape.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        uint ndim = getNDim(shape);
        if (ndim == 3) {
            if (taper_size > 1e-5f) {
                sphereSoft3D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        output_mask, output_mask_pitch, tmp_shape, center, radius, taper_size);
            } else {
                sphereHard3D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        output_mask, output_mask_pitch, tmp_shape, center, radius);
            }
        } else if (ndim == 2) {
            if (taper_size > 1e-5f) {
                sphereSoft2D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        output_mask, output_mask_pitch, shape.x, {center.x, center.y}, radius, taper_size);

            } else {
                sphereHard2D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        output_mask, output_mask_pitch, shape.x, {center.x, center.y}, radius);
            }
        } else {
            NOA_THROW("Cannot compute a sphere with shape:{}", shape);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define INSTANTIATE_SPHERE(T)                                                                                   \
    template void sphere<true, T>(const T*, size_t, T*, size_t, size3_t, float3_t, float, float, uint, Stream&);    \
    template void sphere<false, T>(const T*, size_t, T*, size_t, size3_t, float3_t, float, float, uint, Stream&);   \
    template void sphere<true, T>(T*, size_t, size3_t, float3_t, float, float, Stream&);                            \
    template void sphere<false, T>(T*, size_t, size3_t, float3_t, float, float, Stream&)

    INSTANTIATE_SPHERE(float);
    INSTANTIATE_SPHERE(double);
}
