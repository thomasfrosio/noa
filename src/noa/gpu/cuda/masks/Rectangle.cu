#include "noa/gpu/cuda/masks/Rectangle.h"
#include "noa/Exception.h"
#include "noa/Math.h"

// Soft edges:
namespace {
    using namespace Noa;

    template<bool INVERT>
    __forceinline__ __device__ float getSoftMask3D_(float3_t distance, float3_t radius,
                                                    float3_t radius_with_taper, float taper_size) {
        constexpr float PI = Math::Constants<float>::PI;
        float mask_value;
        if constexpr (INVERT) {
            if (any(radius_with_taper < distance)) {
                mask_value = 1.f;
            } else if (all(distance <= radius)) {
                mask_value = 0.f;
            } else {
                mask_value = 1.f;
                if (radius.x < distance.x && distance.x <= radius_with_taper.x)
                    mask_value *= (1.f + Math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                if (radius.y < distance.y && distance.y <= radius_with_taper.y)
                    mask_value *= (1.f + Math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
                if (radius.z < distance.z && distance.z <= radius_with_taper.z)
                    mask_value *= (1.f + Math::cos(PI * (distance.z - radius.z) / taper_size)) * 0.5f;
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
                    mask_value *= (1.f + Math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                if (radius.y < distance.y && distance.y <= radius_with_taper.y)
                    mask_value *= (1.f + Math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
                if (radius.z < distance.z && distance.z <= radius_with_taper.z)
                    mask_value *= (1.f + Math::cos(PI * (distance.z - radius.z) / taper_size)) * 0.5f;
            }
        }
        return mask_value;
    }

    template<bool INVERT>
    __forceinline__ __device__ float getSoftMask2D_(float2_t distance, float2_t radius,
                                                    float2_t radius_with_taper, float taper_size) {
        constexpr float PI = Math::Constants<float>::PI;
        float mask_value;
        if constexpr (INVERT) {
            if (any(radius_with_taper < distance)) {
                mask_value = 1.f;
            } else if (all(distance <= radius)) {
                mask_value = 0.f;
            } else {
                mask_value = 1.f;
                if (radius.x < distance.x && distance.x <= radius_with_taper.x)
                    mask_value *= (1.f + Math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                if (radius.y < distance.y && distance.y <= radius_with_taper.y)
                    mask_value *= (1.f + Math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
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
                    mask_value *= (1.f + Math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                if (radius.y < distance.y && distance.y <= radius_with_taper.y)
                    mask_value *= (1.f + Math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
            }
        }
        return mask_value;
    }

    template<bool INVERT, typename T>
    __global__ void rectangleSoft3D_(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                     uint3_t shape, float3_t center, float3_t radius,
                                     float taper_size, uint batches) {
        uint y = blockIdx.x, z = blockIdx.y;
        inputs += (z * shape.y + y) * pitch_inputs;
        outputs += (z * shape.y + y) * pitch_outputs;
        uint rows = getRows(shape);
        uint elements_inputs = pitch_inputs * rows;
        uint elements_outputs = pitch_outputs * rows;

        float3_t radius_with_taper = radius + taper_size;
        float3_t distance;
        distance.z = Math::abs(static_cast<float>(z) - center.z);
        distance.y = Math::abs(static_cast<float>(y) - center.y);

        float mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance.x = Math::abs(static_cast<float>(x) - center.x);
            mask_value = getSoftMask3D_<INVERT>(distance, radius, radius_with_taper, taper_size);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] =
                        inputs[batch * elements_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void rectangleSoft2D_(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                     uint2_t shape, float2_t center, float2_t radius,
                                     float taper_size, uint batches) {
        uint y = blockIdx.x;
        inputs += y * pitch_inputs;
        outputs += y * pitch_outputs;
        uint elements_inputs = pitch_inputs * shape.y;
        uint elements_outputs = pitch_outputs * shape.y;

        float2_t radius_with_taper = radius + taper_size;
        float2_t distance;
        distance.y = Math::abs(static_cast<float>(y) - center.y);

        float mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance.x = Math::abs(static_cast<float>(x) - center.x);
            mask_value = getSoftMask2D_<INVERT>(distance, radius, radius_with_taper, taper_size);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] =
                        inputs[batch * elements_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void rectangleSoft3D_(T* output_mask, uint pitch_output_mask,
                                     uint3_t shape, float3_t center, float3_t radius, float taper_size) {
        uint y = blockIdx.x, z = blockIdx.y;
        output_mask += (z * shape.y + y) * pitch_output_mask;

        float3_t radius_with_taper = radius + taper_size;
        float3_t distance;
        distance.z = Math::abs(static_cast<float>(z) - center.z);
        distance.y = Math::abs(static_cast<float>(y) - center.y);

        float mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance.x = Math::abs(static_cast<float>(x) - center.x);
            mask_value = getSoftMask3D_<INVERT>(distance, radius, radius_with_taper, taper_size);
            output_mask[x] = static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void rectangleSoft2D_(T* output_mask, uint pitch_output_mask,
                                     uint shape_x, float2_t center, float2_t radius, float taper_size) {
        uint y = blockIdx.x;
        output_mask += y * pitch_output_mask;

        float2_t radius_with_taper = radius + taper_size;
        float2_t distance;
        distance.y = Math::abs(static_cast<float>(y) - center.y);

        float mask_value;
        for (uint x = threadIdx.x; x < shape_x; x += blockDim.x) {
            distance.x = Math::abs(static_cast<float>(x) - center.x);
            mask_value = getSoftMask2D_<INVERT>(distance, radius, radius_with_taper, taper_size);
            output_mask[x] = static_cast<T>(mask_value);
        }
    }
}

// Hard edges:
namespace {
    using namespace Noa;

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
    __global__ void rectangleHard3D_(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                     uint3_t shape, float3_t center, float3_t radius, uint batches) {
        uint y = blockIdx.x, z = blockIdx.y;
        inputs += (z * shape.y + y) * pitch_inputs;
        outputs += (z * shape.y + y) * pitch_outputs;
        uint rows = getRows(shape);
        uint elements_inputs = pitch_inputs * rows;
        uint elements_outputs = pitch_outputs * rows;

        float3_t distance;
        distance.z = Math::abs(static_cast<float>(z) - center.z);
        distance.y = Math::abs(static_cast<float>(y) - center.y);

        T mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance.x = Math::abs(static_cast<float>(x) - center.x);
            mask_value = getHardMask3D_<INVERT, T>(distance, radius);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] = inputs[batch * elements_inputs + x] * mask_value;
        }
    }

    template<bool INVERT, typename T>
    __global__ void rectangleHard2D_(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                     uint2_t shape, float2_t center, float2_t radius, uint batches) {
        uint y = blockIdx.x;
        inputs += y * pitch_inputs;
        outputs += y * pitch_outputs;
        uint elements_inputs = pitch_inputs * shape.y;
        uint elements_outputs = pitch_outputs * shape.y;

        float2_t distance;
        distance.y = Math::abs(static_cast<float>(y) - center.y);

        T mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance.x = Math::abs(static_cast<float>(x) - center.x);
            mask_value = getHardMask2D_<INVERT, T>(distance, radius);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] = inputs[batch * elements_inputs + x] * mask_value;
        }
    }

    template<bool INVERT, typename T>
    __global__ void rectangleHard3D_(T* output_mask, uint pitch_output_mask,
                                     uint3_t shape, float3_t center, float3_t radius) {
        uint y = blockIdx.x, z = blockIdx.y;
        output_mask += (z * shape.y + y) * pitch_output_mask;

        float3_t distance;
        distance.z = Math::abs(static_cast<float>(z) - center.z);
        distance.y = Math::abs(static_cast<float>(y) - center.y);
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance.x = Math::abs(static_cast<float>(x) - center.x);
            output_mask[x] = getHardMask3D_<INVERT, T>(distance, radius);
        }
    }

    template<bool INVERT, typename T>
    __global__ void rectangleHard2D_(T* output_mask, uint pitch_output_mask,
                                     uint shape_x, float2_t center, float2_t radius) {
        uint y = blockIdx.x;
        output_mask += y * pitch_output_mask;

        float2_t distance;
        distance.y = Math::abs(static_cast<float>(y) - center.y);
        for (uint x = threadIdx.x; x < shape_x; x += blockDim.x) {
            distance.x = Math::abs(static_cast<float>(x) - center.x);
            output_mask[x] = getHardMask2D_<INVERT, T>(distance, radius);
        }
    }
}

namespace Noa::CUDA::Mask {
    template<bool INVERT, typename T>
    void rectangle(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs,
                   size3_t shape, float3_t shifts, float3_t radius,
                   float taper_size, uint batches, Stream& stream) {
        uint3_t tmp_shape(shape);
        float3_t center(tmp_shape / 2U);
        center += shifts;

        uint threads = Math::min(128U, Math::nextMultipleOf(tmp_shape.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        uint ndim = getNDim(shape);
        if (ndim == 3) {
            if (taper_size > 1e-5f) {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                rectangleSoft3D_<INVERT>,
                                inputs, pitch_inputs, outputs, pitch_outputs,
                                tmp_shape, center, radius, taper_size, batches);
            } else {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                rectangleHard3D_<INVERT>,
                                inputs, pitch_inputs, outputs, pitch_outputs,
                                tmp_shape, center, radius, batches);
            }
        } else if (ndim == 2) {
            uint2_t shape_2D(shape.x, shape.y);
            float2_t center_2D(center.x, center.y);
            float2_t radius_2D(radius.x, radius.y);
            if (taper_size > 1e-5f) {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                rectangleSoft2D_<INVERT>,
                                inputs, pitch_inputs, outputs, pitch_outputs,
                                shape_2D, center_2D, radius_2D, taper_size, batches);
            } else {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                rectangleHard2D_<INVERT>,
                                inputs, pitch_inputs, outputs, pitch_outputs,
                                shape_2D, center_2D, radius_2D, batches);
            }
        } else {
            NOA_THROW("Cannot compute a rectangle with shape:{}", shape);
        }
    }

    template<bool INVERT, typename T>
    void rectangle(T* output_mask, size_t pitch_output_mask,
                   size3_t shape, float3_t shifts, float3_t radius, float taper_size, Stream& stream) {
        uint3_t tmp_shape(shape);
        float3_t center(tmp_shape / 2U);
        center += shifts;

        uint threads = Math::min(128U, Math::nextMultipleOf(tmp_shape.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        uint ndim = getNDim(shape);
        if (ndim == 3) {
            if (taper_size > 1e-5f) {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                rectangleSoft3D_<INVERT>,
                                output_mask, pitch_output_mask, tmp_shape, center, radius, taper_size);
            } else {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                rectangleHard3D_<INVERT>,
                                output_mask, pitch_output_mask, tmp_shape, center, radius);
            }
        } else if (ndim == 2) {
            float2_t center_2D(center.x, center.y);
            float2_t radius_2D(radius.x, radius.y);
            if (taper_size > 1e-5f) {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                rectangleSoft2D_<INVERT>,
                                output_mask, pitch_output_mask, shape.x, center_2D, radius_2D, taper_size);

            } else {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                rectangleHard2D_<INVERT>,
                                output_mask, pitch_output_mask, shape.x, center_2D, radius_2D);
            }
        } else {
            NOA_THROW("Cannot compute a sphere with shape:{}", shape);
        }
    }

    #define INSTANTIATE_RECTANGLE(T)                                                        \
    template void rectangle<true, T>(T*, size_t, T*, size_t, size3_t, float3_t, float3_t, float, uint, Stream&);     \
    template void rectangle<false, T>(T*, size_t, T*, size_t, size3_t, float3_t, float3_t, float, uint, Stream&);    \
    template void rectangle<true, T>(T*, size_t, size3_t, float3_t, float3_t, float, Stream&);               \
    template void rectangle<false, T>(T*, size_t, size3_t, float3_t, float3_t, float, Stream&)

    INSTANTIATE_RECTANGLE(float);
    INSTANTIATE_RECTANGLE(double);
}
