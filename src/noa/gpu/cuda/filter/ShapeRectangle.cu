#include "noa/common/Profiler.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/filter/Shape.h"

// Soft edges:
namespace {
    using namespace noa;
    constexpr dim3 THREADS(32, 8);

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
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void rectangleSoft3D_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch,
                          uint3_t shape, uint batches,
                          float3_t center, float3_t radius, float taper_size) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float3_t radius_with_taper = radius + taper_size;
        float3_t distance(math::abs(float3_t(gid) - center));
        float mask_value = getSoftMask3D_<INVERT>(distance, radius, radius_with_taper, taper_size);

        using real_t = traits::value_type_t<T>;
        inputs += (gid.z * shape.y + gid.y) * input_pitch + gid.x;
        outputs += (gid.z * shape.y + gid.y) * output_pitch + gid.x;
        uint elements_inputs = input_pitch * rows(shape);
        uint elements_outputs = output_pitch * rows(shape);
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            outputs[batch * elements_outputs] = inputs[batch * elements_inputs] * static_cast<real_t>(mask_value);
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void rectangleSoft2D_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch,
                          uint2_t shape, uint batches,
                          float2_t center, float2_t radius, float taper_size) {
        const uint2_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float2_t radius_with_taper = radius + taper_size;
        float2_t distance(math::abs(float2_t(gid) - center));
        float mask_value = getSoftMask2D_<INVERT>(distance, radius, radius_with_taper, taper_size);

        using real_t = traits::value_type_t<T>;
        inputs += gid.y * input_pitch + gid.x;
        outputs += gid.y * output_pitch + gid.x;
        uint elements_inputs = input_pitch * shape.y;
        uint elements_outputs = output_pitch * shape.y;
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            outputs[batch * elements_outputs] = inputs[batch * elements_inputs] * static_cast<real_t>(mask_value);
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void rectangleSoft3D_(T* output_mask, uint output_mask_pitch,
                          uint3_t shape, uint batches,
                          float3_t center, float3_t radius, float taper_size) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float3_t radius_with_taper = radius + taper_size;
        float3_t distance(math::abs(float3_t(gid) - center));
        float mask_value = getSoftMask3D_<INVERT>(distance, radius, radius_with_taper, taper_size);

        using real_t = traits::value_type_t<T>;
        output_mask += (gid.z * shape.y + gid.y) * output_mask_pitch + gid.x;
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            output_mask[batch * output_mask_pitch * rows(shape)] = static_cast<real_t>(mask_value);
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void rectangleSoft2D_(T* output_mask, uint output_mask_pitch,
                          uint2_t shape, uint batches,
                          float2_t center, float2_t radius, float taper_size) {
        const uint2_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float2_t radius_with_taper = radius + taper_size;
        float2_t distance(math::abs(float2_t(gid) - center));
        float mask_value = getSoftMask2D_<INVERT>(distance, radius, radius_with_taper, taper_size);

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
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void rectangleHard3D_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch,
                          uint3_t shape, uint batches, float3_t center, float3_t radius) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        using real_t = traits::value_type_t<T>;
        float3_t distance(math::abs(float3_t(gid) - center));
        auto mask_value = getHardMask3D_<INVERT, real_t>(distance, radius);

        inputs += (gid.z * shape.y + gid.y) * input_pitch + gid.x;
        outputs += (gid.z * shape.y + gid.y) * output_pitch + gid.x;
        uint elements_inputs = input_pitch * rows(shape);
        uint elements_outputs = output_pitch * rows(shape);
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            outputs[batch * elements_outputs] = inputs[batch * elements_inputs] * mask_value;
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void rectangleHard2D_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch,
                          uint2_t shape, uint batches, float2_t center, float2_t radius) {
        const uint2_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        using real_t = traits::value_type_t<T>;
        float2_t distance(math::abs(float2_t(gid) - center));
        auto mask_value = getHardMask2D_<INVERT, real_t>(distance, radius);

        inputs += gid.y * input_pitch + gid.x;
        outputs += gid.y * output_pitch + gid.x;
        uint elements_inputs = input_pitch * shape.y;
        uint elements_outputs = output_pitch * shape.y;
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            outputs[batch * elements_outputs] = inputs[batch * elements_inputs] * mask_value;
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void rectangleHard3D_(T* output_mask, uint output_mask_pitch,
                          uint3_t shape, uint batches, float3_t center, float3_t radius) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        using real_t = traits::value_type_t<T>;
        float3_t distance(math::abs(float3_t(gid) - center));
        auto mask_value = getHardMask3D_<INVERT, real_t>(distance, radius);

        output_mask += (gid.z * shape.y + gid.y) * output_mask_pitch + gid.x;
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            output_mask[batch * output_mask_pitch * rows(shape)] = mask_value;
    }

    template<bool INVERT, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void rectangleHard2D_(T* output_mask, uint output_mask_pitch,
                          uint2_t shape, uint batches, float2_t center, float2_t radius) {
        const uint2_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        using real_t = traits::value_type_t<T>;
        float2_t distance(math::abs(float2_t(gid) - center));
        auto mask_value = getHardMask2D_<INVERT, real_t>(distance, radius);

        output_mask += gid.y * output_mask_pitch + gid.x;
        #pragma unroll 2
        for (uint batch = 0; batch < batches; ++batch)
            output_mask[batch * output_mask_pitch * shape.y] = mask_value;
    }
}

namespace noa::cuda::filter {
    template<bool INVERT, typename T>
    void rectangle2D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                     size2_t shape, size_t batches,
                     float2_t center, float2_t radius, float taper_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint2_t u_shape(shape);
        dim3 blocks(math::divideUp(u_shape.x, THREADS.x),
                    math::divideUp(u_shape.y, THREADS.y));
        if (inputs) {
            if (taper_size > 1e-5f) {
                rectangleSoft2D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, input_pitch, outputs, output_pitch,
                        u_shape, batches, center, radius, taper_size);
            } else {
                rectangleHard2D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, input_pitch, outputs, output_pitch,
                        u_shape, batches, center, radius);
            }
        } else {
            if (taper_size > 1e-5f) {
                rectangleSoft2D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        outputs, output_pitch, u_shape, batches, center, radius, taper_size);

            } else {
                rectangleHard2D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        outputs, output_pitch, u_shape, batches, center, radius);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<bool INVERT, typename T>
    void rectangle3D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                     size3_t shape, size_t batches,
                     float3_t center, float3_t radius, float taper_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint3_t u_shape(shape);
        dim3 blocks(math::divideUp(u_shape.x, THREADS.x),
                    math::divideUp(u_shape.y, THREADS.y),
                    u_shape.z);
        if (inputs) {
            if (taper_size > 1e-5f) {
                rectangleSoft3D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, input_pitch, outputs, output_pitch,
                        u_shape, batches, center, radius, taper_size);
            } else {
                rectangleHard3D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, input_pitch, outputs, output_pitch,
                        u_shape, batches, center, radius);
            }
        } else {
            if (taper_size > 1e-5f) {
                rectangleSoft3D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        outputs, output_pitch, u_shape, batches, center, radius, taper_size);

            } else {
                rectangleHard3D_<INVERT><<<blocks, THREADS, 0, stream.id()>>>(
                        outputs, output_pitch, u_shape, batches, center, radius);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define NOA_INSTANTIATE_RECTANGLE_(T)                                                                                   \
    template void rectangle2D<true, T>(const T*, size_t, T*, size_t, size2_t, size_t, float2_t, float2_t, float, Stream&);  \
    template void rectangle2D<false, T>(const T*, size_t, T*, size_t, size2_t, size_t, float2_t, float2_t, float, Stream&); \
    template void rectangle3D<true, T>(const T*, size_t, T*, size_t, size3_t, size_t, float3_t, float3_t, float, Stream&);  \
    template void rectangle3D<false, T>(const T*, size_t, T*, size_t, size3_t, size_t, float3_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_RECTANGLE_(float);
    NOA_INSTANTIATE_RECTANGLE_(double);
    NOA_INSTANTIATE_RECTANGLE_(cfloat_t);
    NOA_INSTANTIATE_RECTANGLE_(cdouble_t);
}
