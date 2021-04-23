#include "noa/gpu/cuda/masks/Rectangle.h"
#include "noa/Exception.h"
#include "noa/Math.h"

// Soft edges:
namespace {
    using namespace Noa;

    template<bool INVERT>
    NOA_ID float getSoftMask(float distance_xy_sqd, float radius_xy_sqd,
                             float radius_xy, float radius_xy_sqd_with_taper,
                             float distance_z, float radius_z, float radius_z_with_taper,
                             float taper_size) {
        constexpr float PI = Math::Constants<float>::PI;
        float mask_value;
        if constexpr (INVERT) {
            if (distance_z > radius_z_with_taper || distance_xy_sqd > radius_xy_sqd_with_taper) {
                mask_value = 1.f;
            } else {
                if (distance_xy_sqd <= radius_xy_sqd) {
                    mask_value = 1.f;
                } else {
                    float distance_xy = Math::sqrt(distance_xy_sqd);
                    mask_value = (1.f + Math::cos(PI * (distance_xy - radius_xy) / taper_size)) * 0.5f;
                }
                if (distance_z > radius_z)
                    mask_value *= (1.f + Math::cos(PI * (distance_z - radius_z) / taper_size)) * 0.5f;
                mask_value = 1 - mask_value;
            }
        } else {
            if (distance_z > radius_z_with_taper || distance_xy_sqd > radius_xy_sqd_with_taper) {
                mask_value = 0.f;
            } else {
                if (distance_xy_sqd <= radius_xy_sqd) {
                    mask_value = 1.f;
                } else {
                    float distance_xy = Math::sqrt(distance_xy_sqd);
                    mask_value = (1.f + Math::cos(PI * (distance_xy - radius_xy) / taper_size)) * 0.5f;
                }
                if (distance_z > radius_z)
                    mask_value *= (1.f + Math::cos(PI * (distance_z - radius_z) / taper_size)) * 0.5f;
            }
        }
        return mask_value;
    }

    template<bool INVERT, typename T>
    __global__ void cylinderSoft(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                 uint3_t shape, float3_t center, float radius_xy, float radius_z,
                                 float taper_size, uint batches) {
        uint y = blockIdx.x, z = blockIdx.y;
        inputs += (z * shape.y + y) * pitch_inputs;
        outputs += (z * shape.y + y) * pitch_outputs;

        uint rows = getRows(shape);
        uint elements_inputs = pitch_inputs * rows;
        uint elements_outputs = pitch_outputs * rows;

        float radius_z_taper = radius_z + taper_size;
        float radius_xy_sqd = radius_xy * radius_xy;
        float radius_xy_taper_sqd = radius_xy + taper_size;
        radius_xy_taper_sqd *= radius_xy_taper_sqd;

        float distance_z = Math::abs(static_cast<float>(z) - center.z);
        float distance_y_sqd = static_cast<float>(y) - center.y;
        distance_y_sqd *= distance_y_sqd;

        float distance_xy_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_xy_sqd = static_cast<float>(x) - center.x;
            distance_xy_sqd *= distance_xy_sqd;
            distance_xy_sqd += distance_y_sqd;
            mask_value = getSoftMask<INVERT>(distance_xy_sqd, radius_xy_sqd, radius_xy, radius_xy_taper_sqd,
                                             distance_z, radius_z, radius_z_taper, taper_size);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] =
                        inputs[batch * elements_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void cylinderSoft(T* output_mask, uint pitch_output_mask,
                                 uint3_t shape, float3_t center, float radius_xy, float radius_z,
                                 float taper_size) {
        uint y = blockIdx.x, z = blockIdx.y;
        output_mask += (z * shape.y + y) * pitch_output_mask;

        float radius_z_taper = radius_z + taper_size;
        float radius_xy_sqd = radius_xy * radius_xy;
        float radius_xy_taper_sqd = radius_xy + taper_size;
        radius_xy_taper_sqd *= radius_xy_taper_sqd;

        float distance_z = Math::abs(static_cast<float>(z) - center.z);
        float distance_y_sqd = static_cast<float>(y) - center.y;
        distance_y_sqd *= distance_y_sqd;

        float distance_xy_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_xy_sqd = static_cast<float>(x) - center.x;
            distance_xy_sqd *= distance_xy_sqd;
            distance_xy_sqd += distance_y_sqd;
            mask_value = getSoftMask<INVERT>(distance_xy_sqd, radius_xy_sqd, radius_xy, radius_xy_taper_sqd,
                                             distance_z, radius_z, radius_z_taper, taper_size);
            output_mask[x] = static_cast<T>(mask_value);
        }
    }
}

// Hard edges:
namespace {
    using namespace Noa;

    template<bool INVERT>
    NOA_FD float getHardMask(float distance_xy_sqd, float radius_xy_sqd, float distance_z, float radius_z) {
        float mask_value;
        if constexpr (INVERT) {
            if (distance_z > radius_z || distance_xy_sqd > radius_xy_sqd)
                mask_value = 1.f;
            else
                mask_value = 0.f;
        } else {
            if (distance_z > radius_z || distance_xy_sqd > radius_xy_sqd)
                mask_value = 0.f;
            else
                mask_value = 1.f;
        }
        return mask_value;
    }

    template<bool INVERT, typename T>
    __global__ void cylinderHard(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                 uint3_t shape, float3_t center, float radius_xy, float radius_z, uint batches) {
        uint y = blockIdx.x, z = blockIdx.y;
        inputs += (z * shape.y + y) * pitch_inputs;
        outputs += (z * shape.y + y) * pitch_outputs;

        uint rows = getRows(shape);
        uint elements_inputs = pitch_inputs * rows;
        uint elements_outputs = pitch_outputs * rows;

        float radius_xy_sqd = radius_xy * radius_xy;
        float distance_z = Math::abs(static_cast<float>(z) - center.z);
        float distance_y_sqd = static_cast<float>(y) - center.y;
        distance_y_sqd *= distance_y_sqd;

        float distance_xy_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_xy_sqd = static_cast<float>(x) - center.x;
            distance_xy_sqd *= distance_xy_sqd;
            distance_xy_sqd += distance_y_sqd;
            mask_value = getHardMask<INVERT>(distance_xy_sqd, radius_xy_sqd, distance_z, radius_z);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] =
                        inputs[batch * elements_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void cylinderHard(T* output_mask, uint pitch_output_mask,
                                 uint3_t shape, float3_t center, float radius_xy, float radius_z) {
        uint y = blockIdx.x, z = blockIdx.y;
        output_mask += (z * shape.y + y) * pitch_output_mask;

        float radius_xy_sqd = radius_xy * radius_xy;
        float distance_z = Math::abs(static_cast<float>(z) - center.z);
        float distance_y_sqd = static_cast<float>(y) - center.y;
        distance_y_sqd *= distance_y_sqd;

        float distance_xy_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_xy_sqd = static_cast<float>(x) - center.x;
            distance_xy_sqd *= distance_xy_sqd;
            distance_xy_sqd += distance_y_sqd;
            mask_value = getHardMask<INVERT>(distance_xy_sqd, radius_xy_sqd, distance_z, radius_z);
            output_mask[x] = static_cast<T>(mask_value);
        }
    }
}

// Definitions & Instantiations:
namespace Noa::CUDA::Mask {
    template<bool INVERT, typename T>
    NOA_HOST void cylinder(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs,
                           size3_t shape, float3_t shifts, float radius_xy, float radius_z,
                           float taper_size, uint batches, Stream& stream) {
        uint3_t tmp_shape(shape);
        float3_t center(tmp_shape / 2U);
        center += shifts;

        uint threads = Math::min(128U, Math::nextMultipleOf(tmp_shape.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        if (taper_size > 1e-5f) {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            cylinderSoft<INVERT>,
                            inputs, pitch_inputs, outputs, pitch_outputs,
                            tmp_shape, center, radius_xy, radius_z, taper_size, batches);
        } else {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            cylinderHard<INVERT>,
                            inputs, pitch_inputs, outputs, pitch_outputs,
                            tmp_shape, center, radius_xy, radius_z, batches);
        }
    }

    template<bool INVERT, typename T>
    NOA_HOST void cylinder(T* output_mask, size_t pitch_output_mask, size3_t shape, float3_t shifts,
                           float radius_xy, float radius_z, float taper_size, Stream& stream) {
        uint3_t tmp_shape(shape);
        float3_t center(tmp_shape / 2U);
        center += shifts;

        uint threads = Math::min(128U, Math::nextMultipleOf(tmp_shape.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        if (taper_size > 1e-5f) {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            cylinderSoft<INVERT>,
                            output_mask, pitch_output_mask, tmp_shape, center, radius_xy, radius_z, taper_size);
        } else {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            cylinderHard<INVERT>,
                            output_mask, pitch_output_mask, tmp_shape, center, radius_xy, radius_z);
        }
    }

    #define INSTANTIATE_CYLINDER(T)                                                                                     \
    template void cylinder<true, T>(T*, size_t, T*, size_t, size3_t, float3_t, float, float, float, uint, Stream&);     \
    template void cylinder<false, T>(T*, size_t, T*, size_t, size3_t, float3_t, float, float, float, uint, Stream&);    \
    template void cylinder<true, T>(T*, size_t, size3_t, float3_t, float, float, float, Stream&);                       \
    template void cylinder<false, T>(T*, size_t, size3_t, float3_t, float, float, float, Stream&)

    INSTANTIATE_CYLINDER(float);
    INSTANTIATE_CYLINDER(double);
}
