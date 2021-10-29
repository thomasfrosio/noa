#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/filter/Rectangle.h"

// Soft edges:
namespace {
    using namespace noa;

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
    __global__ void cylinderSoft_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch,
                                  uint3_t shape, float3_t center, float radius_xy, float radius_z,
                                  float taper_size, uint batches) {
        uint y = blockIdx.x, z = blockIdx.y;
        inputs += (z * shape.y + y) * input_pitch;
        outputs += (z * shape.y + y) * output_pitch;

        uint elements_inputs = input_pitch * rows(shape);
        uint elements_outputs = output_pitch * rows(shape);

        float radius_z_taper = radius_z + taper_size;
        float radius_xy_sqd = radius_xy * radius_xy;
        float radius_xy_taper_sqd = radius_xy + taper_size;
        radius_xy_taper_sqd *= radius_xy_taper_sqd;

        float distance_z = math::abs(static_cast<float>(z) - center.z);
        float distance_y_sqd = static_cast<float>(y) - center.y;
        distance_y_sqd *= distance_y_sqd;

        float distance_xy_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_xy_sqd = static_cast<float>(x) - center.x;
            distance_xy_sqd *= distance_xy_sqd;
            distance_xy_sqd += distance_y_sqd;
            mask_value = getSoftMask_<INVERT>(distance_xy_sqd, radius_xy_sqd, radius_xy, radius_xy_taper_sqd,
                                              distance_z, radius_z, radius_z_taper, taper_size);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] =
                        inputs[batch * elements_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void cylinderSoft_(T* output_mask, uint output_mask_pitch,
                                  uint3_t shape, float3_t center, float radius_xy, float radius_z,
                                  float taper_size) {
        uint y = blockIdx.x, z = blockIdx.y;
        output_mask += (z * shape.y + y) * output_mask_pitch;

        float radius_z_taper = radius_z + taper_size;
        float radius_xy_sqd = radius_xy * radius_xy;
        float radius_xy_taper_sqd = radius_xy + taper_size;
        radius_xy_taper_sqd *= radius_xy_taper_sqd;

        float distance_z = math::abs(static_cast<float>(z) - center.z);
        float distance_y_sqd = static_cast<float>(y) - center.y;
        distance_y_sqd *= distance_y_sqd;

        float distance_xy_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_xy_sqd = static_cast<float>(x) - center.x;
            distance_xy_sqd *= distance_xy_sqd;
            distance_xy_sqd += distance_y_sqd;
            mask_value = getSoftMask_<INVERT>(distance_xy_sqd, radius_xy_sqd, radius_xy, radius_xy_taper_sqd,
                                              distance_z, radius_z, radius_z_taper, taper_size);
            output_mask[x] = static_cast<T>(mask_value);
        }
    }
}

// Hard edges:
namespace {
    using namespace noa;

    template<bool INVERT>
    __forceinline__ __device__ float getHardMask_(float distance_xy_sqd, float radius_xy_sqd,
                                                  float distance_z, float radius_z) {
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
    __global__ void cylinderHard_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch,
                                  uint3_t shape, float3_t center, float radius_xy, float radius_z, uint batches) {
        uint y = blockIdx.x, z = blockIdx.y;
        inputs += (z * shape.y + y) * input_pitch;
        outputs += (z * shape.y + y) * output_pitch;

        uint elements_inputs = input_pitch * rows(shape);
        uint elements_outputs = output_pitch * rows(shape);

        float radius_xy_sqd = radius_xy * radius_xy;
        float distance_z = math::abs(static_cast<float>(z) - center.z);
        float distance_y_sqd = static_cast<float>(y) - center.y;
        distance_y_sqd *= distance_y_sqd;

        float distance_xy_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_xy_sqd = static_cast<float>(x) - center.x;
            distance_xy_sqd *= distance_xy_sqd;
            distance_xy_sqd += distance_y_sqd;
            mask_value = getHardMask_<INVERT>(distance_xy_sqd, radius_xy_sqd, distance_z, radius_z);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] =
                        inputs[batch * elements_inputs + x] * static_cast<T>(mask_value);
        }
    }

    template<bool INVERT, typename T>
    __global__ void cylinderHard_(T* output_mask, uint output_mask_pitch,
                                  uint3_t shape, float3_t center, float radius_xy, float radius_z) {
        uint y = blockIdx.x, z = blockIdx.y;
        output_mask += (z * shape.y + y) * output_mask_pitch;

        float radius_xy_sqd = radius_xy * radius_xy;
        float distance_z = math::abs(static_cast<float>(z) - center.z);
        float distance_y_sqd = static_cast<float>(y) - center.y;
        distance_y_sqd *= distance_y_sqd;

        float distance_xy_sqd, mask_value;
        for (uint x = threadIdx.x; x < shape.x; x += blockDim.x) {
            distance_xy_sqd = static_cast<float>(x) - center.x;
            distance_xy_sqd *= distance_xy_sqd;
            distance_xy_sqd += distance_y_sqd;
            mask_value = getHardMask_<INVERT>(distance_xy_sqd, radius_xy_sqd, distance_z, radius_z);
            output_mask[x] = static_cast<T>(mask_value);
        }
    }
}

namespace noa::cuda::filter {
    template<bool INVERT, typename T>
    void cylinder(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                  size3_t shape, float3_t shifts, float radius_xy, float radius_z,
                  float taper_size, uint batches, Stream& stream) {
        uint3_t tmp_shape(shape);
        float3_t center(tmp_shape / 2U);
        center += shifts;

        uint threads = math::min(128U, math::nextMultipleOf(tmp_shape.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        if (taper_size > 1e-5f) {
            cylinderSoft_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                    inputs, input_pitch, outputs, output_pitch,
                    tmp_shape, center, radius_xy, radius_z, taper_size, batches);
        } else {
            cylinderHard_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                    inputs, input_pitch, outputs, output_pitch,
                    tmp_shape, center, radius_xy, radius_z, batches);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<bool INVERT, typename T>
    void cylinder(T* output_mask, size_t output_mask_pitch, size3_t shape, float3_t shifts,
                  float radius_xy, float radius_z, float taper_size, Stream& stream) {
        uint3_t tmp_shape(shape);
        float3_t center(tmp_shape / 2U);
        center += shifts;

        uint threads = math::min(128U, math::nextMultipleOf(tmp_shape.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        if (taper_size > 1e-5f) {
            cylinderSoft_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                    output_mask, output_mask_pitch, tmp_shape, center, radius_xy, radius_z, taper_size);
        } else {
            cylinderHard_<INVERT><<<blocks, threads, 0, stream.id()>>>(
                    output_mask, output_mask_pitch, tmp_shape, center, radius_xy, radius_z);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define NOA_INSTANTIATE_CYLINDER_(T)                                                                                    \
    template void cylinder<true, T>(const T*, size_t, T*, size_t, size3_t, float3_t, float, float, float, uint, Stream&);   \
    template void cylinder<false, T>(const T*, size_t, T*, size_t, size3_t, float3_t, float, float, float, uint, Stream&);  \
    template void cylinder<true, T>(T*, size_t, size3_t, float3_t, float, float, float, Stream&);                           \
    template void cylinder<false, T>(T*, size_t, size3_t, float3_t, float, float, float, Stream&)

    NOA_INSTANTIATE_CYLINDER_(float);
    NOA_INSTANTIATE_CYLINDER_(double);
}
