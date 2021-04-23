#include "noa/gpu/cuda/masks/Sphere.h"
#include "noa/Exception.h"
#include "noa/Math.h"

// Soft edges:
namespace {
    using namespace Noa;

    template<bool INVERT>
    NOA_FD float getSoftMask(float distance_sqd, float radius, float radius_sqd,
                             float radius_taper_sqd, float taper_size) {
        float mask_value;
        constexpr float PI = Math::Constants<float>::PI;
        if constexpr (INVERT) {
            if (distance_sqd > radius_taper_sqd) {
                mask_value = 1.f;
            } else if (distance_sqd <= radius_sqd) {
                mask_value = 0.f;
            } else {
                float distance = Math::sqrt(distance_sqd);
                mask_value = (1.f - Math::cos(PI * (distance - radius) / taper_size)) * 0.5f;
            }
        } else {
            if (distance_sqd > radius_taper_sqd) {
                mask_value = 0.f;
            } else if (distance_sqd <= radius_sqd) {
                mask_value = 1.f;
            } else {
                distance_sqd = Math::sqrt(distance_sqd);
                mask_value = (1.f + Math::cos(PI * (distance_sqd - radius) / taper_size)) * 0.5f;
            }
        }
        return mask_value;
    }

    template<bool INVERT, typename T>
    __global__ void sphereSoft3D(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                 uint3_t shape, float3_t center, float radius, float taper_size, uint batches) {
        uint y = blockIdx.x, z = blockIdx.y;

        uint offset_inputs = (z * shape.y + y) * pitch_inputs;
        uint offset_outputs = (z * shape.y + y) * pitch_outputs;

        uint rows = getRows(shape);
        uint elements_inputs = pitch_inputs * rows;
        uint elements_outputs = pitch_outputs * rows;

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

            mask_value = getSoftMask<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);

            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + offset_outputs + x] =
                        inputs[batch * elements_inputs + offset_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void sphereSoft2D(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                 uint2_t shape, float2_t center, float radius, float taper_size, uint batches) {
        uint y = blockIdx.x;

        uint offset_inputs = y * pitch_inputs;
        uint offset_outputs = y * pitch_outputs;
        uint elements_inputs = pitch_inputs * shape.y;
        uint elements_outputs = pitch_outputs * shape.y;

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

            mask_value = getSoftMask<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);

            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + offset_outputs + x] =
                        inputs[batch * elements_inputs + offset_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void sphereSoft3D(T* output_mask, uint pitch_output_mask,
                                 uint3_t shape, float3_t center, float radius, float taper_size) {
        uint y = blockIdx.x, z = blockIdx.y;
        uint offset = (z * shape.y + y) * pitch_output_mask;

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
            mask_value = getSoftMask<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
            output_mask[offset + x] = static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void sphereSoft2D(T* output_mask, uint pitch_output_mask,
                                 uint shape_x, float2_t center, float radius, float taper_size) {
        uint y = blockIdx.x;
        uint offset = y * pitch_output_mask;

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
            mask_value = getSoftMask<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
            output_mask[offset + x] = static_cast<T>(mask_value);
        }
    }
}

// Hard edges:
namespace {
    using namespace Noa;

    template<bool INVERT>
    NOA_FD float getHardMask(float distance_sqd, float radius_sqd) {
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
    __global__ void sphereHard3D(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                 uint3_t shape, float3_t center, float radius, uint batches) {
        uint y = blockIdx.x, z = blockIdx.y;

        uint offset_inputs = (z * shape.y + y) * pitch_inputs;
        uint offset_outputs = (z * shape.y + y) * pitch_outputs;

        uint rows = getRows(shape);
        uint elements_inputs = pitch_inputs * rows;
        uint elements_outputs = pitch_outputs * rows;

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

            mask_value = getHardMask<INVERT>(distance_sqd, radius_sqd);

            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + offset_outputs + x] =
                        inputs[batch * elements_inputs + offset_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void sphereHard2D(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                 uint2_t shape, float2_t center, float radius, uint batches) {
        uint y = blockIdx.x;

        uint offset_inputs = y * pitch_inputs;
        uint offset_outputs = y * pitch_outputs;

        uint elements_inputs = pitch_inputs * shape.y;
        uint elements_outputs = pitch_outputs * shape.y;

        float radius_sqd = radius * radius;
        float tmp_distance_sqd = static_cast<float>(y) - center.y;
        tmp_distance_sqd *= tmp_distance_sqd;

        float distance_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_sqd = static_cast<float>(x) - center.x;
            distance_sqd *= distance_sqd;
            distance_sqd += tmp_distance_sqd;

            mask_value = getHardMask<INVERT>(distance_sqd, radius_sqd);

            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + offset_outputs + x] =
                        inputs[batch * elements_inputs + offset_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void sphereHard3D(T* output_mask, uint pitch_output_mask, uint3_t shape, float3_t center, float radius) {
        uint y = blockIdx.x, z = blockIdx.y;
        uint offset = (z * shape.y + y) * pitch_output_mask;

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
            mask_value = getHardMask<INVERT>(distance_sqd, radius_sqd);
            output_mask[offset + x] = static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void sphereHard2D(T* output_mask, uint pitch_output_mask, uint shape_x, float2_t center, float radius) {
        uint y = blockIdx.x;
        uint offset = y * pitch_output_mask;

        float radius_sqd = radius * radius;
        float tmp_distance_sqd = static_cast<float>(y) - center.y;
        tmp_distance_sqd *= tmp_distance_sqd;

        float distance_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape_x; x += blockDim.x) {
            distance_sqd = static_cast<float>(x) - center.x;
            distance_sqd *= distance_sqd;
            distance_sqd += tmp_distance_sqd;
            mask_value = getHardMask<INVERT>(distance_sqd, radius_sqd);
            output_mask[offset + x] = static_cast<T>(mask_value);
        }
    }
}

// Definitions & Instantiations:
namespace Noa::CUDA::Mask {
    template<bool INVERT, typename T>
    void sphere(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs, size3_t shape, float3_t shifts,
                float radius, float taper_size, uint batches, Stream& stream) {
        uint3_t tmp_shape(shape);
        float3_t center(tmp_shape / 2U);
        center += shifts;

        uint threads = Math::min(128U, Math::nextMultipleOf(tmp_shape.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        uint ndim = getNDim(shape);
        if (ndim == 3) {
            if (taper_size > 1e-5f) {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                sphereSoft3D<INVERT>,
                                inputs, pitch_inputs, outputs, pitch_outputs,
                                tmp_shape, center, radius, taper_size, batches);
            } else {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                sphereHard3D<INVERT>,
                                inputs, pitch_inputs, outputs, pitch_outputs,
                                tmp_shape, center, radius, batches);
            }
        } else if (ndim == 2) {
            uint2_t shape_2D(shape.x, shape.y);
            float2_t center_2D(center.x, center.y);
            if (taper_size > 1e-5f) {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                sphereSoft2D<INVERT>,
                                inputs, pitch_inputs, outputs, pitch_outputs,
                                shape_2D, center_2D, radius, taper_size, batches);
            } else {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                sphereHard2D<INVERT>,
                                inputs, pitch_inputs, outputs, pitch_outputs,
                                shape_2D, center_2D, radius, batches);
            }
        } else {
            NOA_THROW("Cannot compute a sphere with shape:{}", shape);
        }
    }

    template<bool INVERT, typename T>
    void sphere(T* output_mask, size_t pitch_output_mask, size3_t shape, float3_t shifts,
                float radius, float taper_size, Stream& stream) {
        uint3_t tmp_shape(shape);
        float3_t center(tmp_shape / 2U);
        center += shifts;

        uint threads = Math::min(128U, Math::nextMultipleOf(tmp_shape.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        uint ndim = getNDim(shape);
        if (ndim == 3) {
            if (taper_size > 1e-5f) {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                sphereSoft3D<INVERT>,
                                output_mask, pitch_output_mask, tmp_shape, center, radius, taper_size);
            } else {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                sphereHard3D<INVERT>,
                                output_mask, pitch_output_mask, tmp_shape, center, radius);
            }
        } else if (ndim == 2) {
            if (taper_size > 1e-5f) {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                sphereSoft2D<INVERT>,
                                output_mask, pitch_output_mask, shape.x, { center.x, center.y }, radius, taper_size);

            } else {
                NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                                sphereHard2D<INVERT>,
                                output_mask, pitch_output_mask, shape.x, { center.x, center.y }, radius);
            }
        } else {
            NOA_THROW("Cannot compute a sphere with shape:{}", shape);
        }
    }

    #define INSTANTIATE_SPHERE(T)                                                                       \
    template void sphere<true, T>(T*, size_t, T*, size_t, size3_t, float3_t, float, float, uint, Stream&);  \
    template void sphere<false, T>(T*, size_t, T*, size_t, size3_t, float3_t, float, float, uint, Stream&); \
    template void sphere<true, T>(T*, size_t, size3_t, float3_t, float, float, Stream&);                     \
    template void sphere<false, T>(T*, size_t, size3_t, float3_t, float, float, Stream&)

    INSTANTIATE_SPHERE(float);
    INSTANTIATE_SPHERE(double);
}
