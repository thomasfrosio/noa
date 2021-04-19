#include "noa/cpu/masks/Rectangle.h"

#include "noa/Math.h"
#include "noa/Profiler.h"

namespace Noa::Mask::Details {
    template<bool INVERT, bool ON_THE_FLY, typename T>
    void rectangle3D(T* inputs, T* outputs, size3_t shape, float3_t shifts, float3_t radius,
                     float taper_size, uint batches) {
        constexpr float PI = Math::Constants<float>::PI;
        NOA_PROFILE_FUNCTION();

        size_t elements;
        if constexpr (ON_THE_FLY)
            elements = getElements(shape);

        float3_t center(shape / size_t{2});
        center += shifts;

        float3_t radius_with_taper = radius + taper_size;

        // Compute the mask using single precision, even if T is double.
        float mask_value;
        float3_t distance;
        for (uint z = 0; z < shape.z; ++z) {
            distance.z = Math::abs(static_cast<float>(z) - center.z);

            for (uint y = 0; y < shape.y; ++y) {
                distance.y = Math::abs(static_cast<float>(y) - center.y);
                size_t offset = (z * shape.y + y) * shape.x;

                for (uint x = 0; x < shape.x; ++x) {
                    distance.x = Math::abs(static_cast<float>(x) - center.x);

                    if constexpr (INVERT) {
                        if (radius_with_taper.z < distance.z ||
                            radius_with_taper.y < distance.y ||
                            radius_with_taper.x < distance.x) {
                            mask_value = 1.f;
                        } else if (distance <= radius) {
                            mask_value = 0.f;
                        } else {
                            mask_value = 1.f;
                            if (radius.x < distance.x && distance.x < radius_with_taper.x)
                                mask_value *= (1.f + Math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                            if (radius.y < distance.y && distance.y < radius_with_taper.y)
                                mask_value *= (1.f + Math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
                            if (radius.z < distance.z && distance.z < radius_with_taper.z)
                                mask_value *= (1.f + Math::cos(PI * (distance.z - radius.z) / taper_size)) * 0.5f;
                            mask_value = 1.f - mask_value;
                        }
                    } else {
                        if (radius_with_taper.z < distance.z ||
                            radius_with_taper.y < distance.y ||
                            radius_with_taper.x < distance.x) {
                            mask_value = 0.f;
                        } else if (distance <= radius) {
                            mask_value = 1.f;
                        } else {
                            mask_value = 1.f;
                            if (radius.x < distance.x && distance.x < radius_with_taper.x)
                                mask_value *= (1.f + Math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                            if (radius.y < distance.y && distance.y < radius_with_taper.y)
                                mask_value *= (1.f + Math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
                            if (radius.z < distance.z && distance.z < radius_with_taper.z)
                                mask_value *= (1.f + Math::cos(PI * (distance.z - radius.z) / taper_size)) * 0.5f;
                        }
                    }

                    if constexpr (ON_THE_FLY) {
                        for (uint batch = 0; batch < batches; ++batch)
                            outputs[batch * elements + offset + x] = inputs[batch * elements + offset + x] *
                                                                     static_cast<T>(mask_value);
                    } else {
                        outputs[offset + x] = static_cast<T>(mask_value);
                    }
                }
            }
        }
    }

    template<bool INVERT, bool ON_THE_FLY, typename T>
    void rectangle2D(T* inputs, T* outputs, size2_t shape, float2_t shifts, float2_t radius,
                     float taper_size, uint batches) {
        constexpr float PI = Math::Constants<float>::PI;
        NOA_PROFILE_FUNCTION();

        size_t elements;
        if constexpr (ON_THE_FLY)
            elements = getElements(shape);

        float2_t center(shape / size_t{2});
        center += shifts;

        float2_t radius_with_taper = radius + taper_size;

        // Compute the mask using single precision, even if T is double.
        float mask_value;
        float2_t distance;

        for (uint y = 0; y < shape.y; ++y) {
            distance.y = Math::abs(static_cast<float>(y) - center.y);
            size_t offset = y * shape.x;

            for (uint x = 0; x < shape.x; ++x) {
                distance.x = Math::abs(static_cast<float>(x) - center.x);

                if constexpr (INVERT) {
                    if (radius_with_taper.y < distance.y ||
                        radius_with_taper.x < distance.x) {
                        mask_value = 1.f;
                    } else if (distance <= radius) {
                        mask_value = 0.f;
                    } else {
                        mask_value = 1.f;
                        if (radius.x < distance.x && distance.x < radius_with_taper.x)
                            mask_value *= (1.f + Math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                        if (radius.y < distance.y && distance.y < radius_with_taper.y)
                            mask_value *= (1.f + Math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
                        mask_value = 1.f - mask_value;
                    }
                } else {
                    if (radius_with_taper.y < distance.y ||
                        radius_with_taper.x < distance.x) {
                        mask_value = 0.f;
                    } else if (distance <= radius) {
                        mask_value = 1.f;
                    } else {
                        mask_value = 1.f;
                        if (radius.x < distance.x && distance.x < radius_with_taper.x)
                            mask_value *= (1.f + Math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                        if (radius.y < distance.y && distance.y < radius_with_taper.y)
                            mask_value *= (1.f + Math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
                    }
                }

                if constexpr (ON_THE_FLY) {
                    for (uint batch = 0; batch < batches; ++batch)
                        outputs[batch * elements + offset + x] = inputs[batch * elements + offset + x] *
                                                                 static_cast<T>(mask_value);
                } else {
                    outputs[offset + x] = static_cast<T>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, bool ON_THE_FLY, typename T>
    void rectangle(T* inputs, T* outputs, size3_t shape, float3_t shifts, float3_t radius, uint batches) {
        NOA_PROFILE_FUNCTION();

        size_t elements;
        if constexpr (ON_THE_FLY)
            elements = getElements(shape);

        float3_t center(shape / size_t{2});
        center += shifts;

        // Compute the mask using single precision, even if T is double.
        T mask_value;
        float3_t distance;
        for (uint z = 0; z < shape.z; ++z) {
            distance.z = Math::abs(static_cast<float>(z) - center.z);

            for (uint y = 0; y < shape.y; ++y) {
                distance.y = Math::abs(static_cast<float>(y) - center.y);
                size_t offset = (z * shape.y + y) * shape.x;

                for (uint x = 0; x < shape.x; ++x) {
                    distance.x = Math::abs(static_cast<float>(x) - center.x);

                    if constexpr (INVERT) {
                        if (distance <= radius)
                            mask_value = 0;
                        else
                            mask_value = 1;
                    } else {
                        if (distance <= radius)
                            mask_value = 1;
                        else
                            mask_value = 0;
                    }

                    if constexpr (ON_THE_FLY) {
                        for (uint batch = 0; batch < batches; ++batch)
                            outputs[batch * elements + offset + x] = inputs[batch * offset + x] * mask_value;
                    } else {
                        outputs[offset + x] = mask_value;
                    }
                }
            }
        }
    }

    #define INSTANTIATE_RECTANGLE(T)                                                                \
    template void rectangle3D<true, true, T>(T*, T*, size3_t, float3_t, float3_t, float, uint);     \
    template void rectangle3D<true, false, T>(T*, T*, size3_t, float3_t, float3_t, float, uint);    \
    template void rectangle3D<false, true, T>(T*, T*, size3_t, float3_t, float3_t, float, uint);    \
    template void rectangle3D<false, false, T>(T*, T*, size3_t, float3_t, float3_t, float, uint);   \
    template void rectangle2D<true, true, T>(T*, T*, size2_t, float2_t, float2_t, float, uint);     \
    template void rectangle2D<true, false, T>(T*, T*, size2_t, float2_t, float2_t, float, uint);    \
    template void rectangle2D<false, true, T>(T*, T*, size2_t, float2_t, float2_t, float, uint);    \
    template void rectangle2D<false, false, T>(T*, T*, size2_t, float2_t, float2_t, float, uint);   \
    template void rectangle<true, true, T>(T*, T*, size3_t, float3_t, float3_t, uint);              \
    template void rectangle<true, false, T>(T*, T*, size3_t, float3_t, float3_t, uint);             \
    template void rectangle<false, true, T>(T*, T*, size3_t, float3_t, float3_t, uint);             \
    template void rectangle<false, false, T>(T*, T*, size3_t, float3_t, float3_t, uint)

    INSTANTIATE_RECTANGLE(float);
    INSTANTIATE_RECTANGLE(double);
}
