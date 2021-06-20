#include "noa/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/mask/Rectangle.h"

// Soft edges:
namespace {
    using namespace noa;

    template<bool INVERT>
    __forceinline__ __device__ float getSoftMask3D_(float3_t distance, float3_t radius,
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
                if (radius.x < distance.x && distance.x <= radius_with_taper.x)
                    mask_value *= (1.f + math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                if (radius.y < distance.y && distance.y <= radius_with_taper.y)
                    mask_value *= (1.f + math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
                if (radius.z < distance.z && distance.z <= radius_with_taper.z)
                    mask_value *= (1.f + math::cos(PI * (distance.z - radius.z) / taper_size)) * 0.5f;
                mask_value = 1.f - mask_value;
            }
        } else {
            if (any(radius_with_taper < distance)) {
                mask_value = 0.f;
            } else if (all(distance <= radius)) {
                mask_value = 1.f;
            } else {
                mask_value = 1.f;
                if (radius.x < distance.x && distance.x <= radius_with_taper.x)
                    mask_value *= (1.f + math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                if (radius.y < distance.y && distance.y <= radius_with_taper.y)
                    mask_value *= (1.f + math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
                if (radius.z < distance.z && distance.z <= radius_with_taper.z)
                    mask_value *= (1.f + math::cos(PI * (distance.z - radius.z) / taper_size)) * 0.5f;
            }
        }
        return mask_value;
    }

    template<bool INVERT>
    __forceinline__ __device__ float getSoftMask2D_(float2_t distance, float2_t radius,
                                                    float2_t radius_with_taper, float taper_size) {
        constexpr float PI = math::Constants<float>::PI;
        float mask_value;
        if constexpr (INVERT) {
            if (any(radius_with_taper < distance)) {
                mask_value = 1.f;
            } else if (all(distance <= radius)) {
                mask_value = 0.f;
            } else {
                mask_value = 1.f;
                if (radius.x < distance.x && distance.x <= radius_with_taper.x)
                    mask_value *= (1.f + math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                if (radius.y < distance.y && distance.y <= radius_with_taper.y)
                    mask_value *= (1.f + math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
                mask_value = 1.f - mask_value;
            }
        } else {
            if (any(radius_with_taper < distance)) {
                mask_value = 0.f;
            } else if (all(distance <= radius)) {
                mask_value = 1.f;
            } else {
                mask_value = 1.f;
                if (radius.x < distance.x && distance.x <= radius_with_taper.x)
                    mask_value *= (1.f + math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                if (radius.y < distance.y && distance.y <= radius_with_taper.y)
                    mask_value *= (1.f + math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
            }
        }
        return mask_value;
    }

    template<bool INVERT, typename T>
    __global__ void rectangleSoft3D_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                     uint3_t shape, float3_t center, float3_t radius,
                                     float taper_size, uint batches) {
        uint y = blockIdx.x, z = blockIdx.y;
        inputs += (z * shape.y + y) * inputs_pitch;
        outputs += (z * shape.y + y) * outputs_pitch;
        uint rows = getRows(shape);
        uint elements_inputs = inputs_pitch * rows;
        uint elements_outputs = outputs_pitch * rows;

        float3_t radius_with_taper = radius + taper_size;
        float3_t distance;
        distance.z = math::abs(static_cast<float>(z) - center.z);
        distance.y = math::abs(static_cast<float>(y) - center.y);

        float mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance.x = math::abs(static_cast<float>(x) - center.x);
            mask_value = getSoftMask3D_<INVERT>(distance, radius, radius_with_taper, taper_size);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] =
                        inputs[batch * elements_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void rectangleSoft2D_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                     uint2_t shape, float2_t center, float2_t radius,
                                     float taper_size, uint batches) {
        uint y = blockIdx.x;
        inputs += y * inputs_pitch;
        outputs += y * outputs_pitch;
        uint elements_inputs = inputs_pitch * shape.y;
        uint elements_outputs = outputs_pitch * shape.y;

        float2_t radius_with_taper = radius + taper_size;
        float2_t distance;
        distance.y = math::abs(static_cast<float>(y) - center.y);

        float mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance.x = math::abs(static_cast<float>(x) - center.x);
            mask_value = getSoftMask2D_<INVERT>(distance, radius, radius_with_taper, taper_size);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] =
                        inputs[batch * elements_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void rectangleSoft3D_(T* output_mask, uint output_mask_pitch,
                                     uint3_t shape, float3_t center, float3_t radius, float taper_size) {
        uint y = blockIdx.x, z = blockIdx.y;
        output_mask += (z * shape.y + y) * output_mask_pitch;

        float3_t radius_with_taper = radius + taper_size;
        float3_t distance;
        distance.z = math::abs(static_cast<float>(z) - center.z);
        distance.y = math::abs(static_cast<float>(y) - center.y);

        float mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance.x = math::abs(static_cast<float>(x) - center.x);
            mask_value = getSoftMask3D_<INVERT>(distance, radius, radius_with_taper, taper_size);
            output_mask[x] = static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void rectangleSoft2D_(T* output_mask, uint output_mask_pitch,
                                     uint shape_x, float2_t center, float2_t radius, float taper_size) {
        uint y = blockIdx.x;
        output_mask += y * output_mask_pitch;

        float2_t radius_with_taper = radius + taper_size;
        float2_t distance;
        distance.y = math::abs(static_cast<float>(y) - center.y);

        float mask_value;
        for (uint x = threadIdx.x; x < shape_x; x += blockDim.x) {
            distance.x = math::abs(static_cast<float>(x) - center.x);
            mask_value = getSoftMask2D_<INVERT>(distance, radius, radius_with_taper, taper_size);
            output_mask[x] = static_cast<T>(mask_value);
        }
    }
}

// Hard edges:
namespace {
    using namespace noa;

    template<bool INVERT, typename T>
    __forceinline__ __device__ T getHardMask3D_(float3_t distance, float3_t radius) {
        T mask_value;
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

    template<bool INVERT, typename T>
    __forceinline__ __device__ T getHardMask2D_(float2_t distance, float2_t radius) {
        T mask_value;
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

    template<bool INVERT, typename T>
    __global__ void rectangleHard3D_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                     uint3_t shape, float3_t center, float3_t radius, uint batches) {
        uint y = blockIdx.x, z = blockIdx.y;
        inputs += (z * shape.y + y) * inputs_pitch;
        outputs += (z * shape.y + y) * outputs_pitch;
        uint rows = getRows(shape);
        uint elements_inputs = inputs_pitch * rows;
        uint elements_outputs = outputs_pitch * rows;

        float3_t distance;
        distance.z = math::abs(static_cast<float>(z) - center.z);
        distance.y = math::abs(static_cast<float>(y) - center.y);

        T mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance.x = math::abs(static_cast<float>(x) - center.x);
            mask_value = getHardMask3D_<INVERT, T>(distance, radius);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] = inputs[batch * elements_inputs + x] * mask_value;
        }
    }

    template<bool INVERT, typename T>
    __global__ void rectangleHard2D_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                     uint2_t shape, float2_t center, float2_t radius, uint batches) {
        uint y = blockIdx.x;
        inputs += y * inputs_pitch;
        outputs += y * outputs_pitch;
        uint elements_inputs = inputs_pitch * shape.y;
        uint elements_outputs = outputs_pitch * shape.y;

        float2_t distance;
        distance.y = math::abs(static_cast<float>(y) - center.y);

        T mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance.x = math::abs(static_cast<float>(x) - center.x);
            mask_value = getHardMask2D_<INVERT, T>(distance, radius);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] = inputs[batch * elements_inputs + x] * mask_value;
        }
    }

    template<bool INVERT, typename T>
    __global__ void rectangleHard3D_(T* output_mask, uint output_mask_pitch,
                                     uint3_t shape, float3_t center, float3_t radius) {
        uint y = blockIdx.x, z = blockIdx.y;
        output_mask += (z * shape.y + y) * output_mask_pitch;

        float3_t distance;
        distance.z = math::abs(static_cast<float>(z) - center.z);
        distance.y = math::abs(static_cast<float>(y) - center.y);
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance.x = math::abs(static_cast<float>(x) - center.x);
            output_mask[x] = getHardMask3D_<INVERT, T>(distance, radius);
        }
    }

    template<bool INVERT, typename T>
    __global__ void rectangleHard2D_(T* output_mask, uint output_mask_pitch,
                                     uint shape_x, float2_t center, float2_t radius) {
        uint y = blockIdx.x;
        output_mask += y * output_mask_pitch;

        float2_t distance;
        distance.y = math::abs(static_cast<float>(y) - center.y);
        for (uint x = threadIdx.x; x < shape_x; x += blockDim.x) {
            distance.x = math::abs(static_cast<float>(x) - center.x);
            output_mask[x] = getHardMask2D_<INVERT, T>(distance, radius);
        }
    }
}

namespace noa::cuda::mask {
    template<bool INVERT, typename T>
    void rectangle(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                   size3_t shape, float3_t shifts, float3_t radius,
                   float taper_size, uint batches, Stream& stream) {
        uint3_t tmp_shape(shape);
        float3_t center(tmp_shape / 2U);
        center += shifts;

        uint threads = math::min(128U, math::nextMultipleOf(tmp_shape.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        uint ndim = getNDim(shape);
        if (ndim == 3) {
            if (taper_size > 1e-5f) {
                rectangleSoft3D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        inputs, inputs_pitch, outputs, outputs_pitch,
                        tmp_shape, center, radius, taper_size, batches);
            } else {
                rectangleHard3D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        inputs, inputs_pitch, outputs, outputs_pitch,
                        tmp_shape, center, radius, batches);
            }
        } else if (ndim == 2) {
            uint2_t shape_2D(shape.x, shape.y);
            float2_t center_2D(center.x, center.y);
            float2_t radius_2D(radius.x, radius.y);
            if (taper_size > 1e-5f) {
                rectangleSoft2D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        inputs, inputs_pitch, outputs, outputs_pitch,
                        shape_2D, center_2D, radius_2D, taper_size, batches);
            } else {
                rectangleHard2D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        inputs, inputs_pitch, outputs, outputs_pitch,
                        shape_2D, center_2D, radius_2D, batches);
            }
        } else {
            NOA_THROW("Cannot compute a rectangle with shape:{}", shape);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<bool INVERT, typename T>
    void rectangle(T* output_mask, size_t output_mask_pitch,
                   size3_t shape, float3_t shifts, float3_t radius, float taper_size, Stream& stream) {
        uint3_t tmp_shape(shape);
        float3_t center(tmp_shape / 2U);
        center += shifts;

        uint threads = math::min(128U, math::nextMultipleOf(tmp_shape.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        uint ndim = getNDim(shape);
        if (ndim == 3) {
            if (taper_size > 1e-5f) {
                rectangleSoft3D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        output_mask, output_mask_pitch, tmp_shape, center, radius, taper_size);
            } else {
                rectangleHard3D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        output_mask, output_mask_pitch, tmp_shape, center, radius);
            }
        } else if (ndim == 2) {
            float2_t center_2D(center.x, center.y);
            float2_t radius_2D(radius.x, radius.y);
            if (taper_size > 1e-5f) {
                rectangleSoft2D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        output_mask, output_mask_pitch, shape.x, center_2D, radius_2D, taper_size);

            } else {
                rectangleHard2D_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                        output_mask, output_mask_pitch, shape.x, center_2D, radius_2D);
            }
        } else {
            NOA_THROW("Cannot compute a sphere with shape:{}", shape);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define INSTANTIATE_RECTANGLE(T)                                                                                    \
    template void rectangle<true, T>(const T*, size_t, T*, size_t, size3_t, float3_t, float3_t, float, uint, Stream&);  \
    template void rectangle<false, T>(const T*, size_t, T*, size_t, size3_t, float3_t, float3_t, float, uint, Stream&); \
    template void rectangle<true, T>(T*, size_t, size3_t, float3_t, float3_t, float, Stream&);                          \
    template void rectangle<false, T>(T*, size_t, size3_t, float3_t, float3_t, float, Stream&)

    INSTANTIATE_RECTANGLE(float);
    INSTANTIATE_RECTANGLE(double);
}
