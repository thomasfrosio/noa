#include "noa/cpu/masks/Sphere.h"

#include "noa/Math.h"
#include "noa/Profiler.h"

namespace Noa::Mask::Details {
    template<bool INVERT, bool ON_THE_FLY, typename T>
    void sphere(T* inputs, T* outputs, size3_t shape, float3_t shifts, float radius, float taper_size, uint batches) {
        constexpr float PI = Math::Constants<float>::PI;
        NOA_PROFILE_FUNCTION();

        size_t elements;
        if constexpr (ON_THE_FLY)
            elements = getElements(shape);

        float3_t center(shape / size_t{2});
        center += shifts;

        float radius_squared = radius * radius;
        float radius_squared_taper = Math::pow(radius + taper_size, 2.f);

        // Compute the mask using single precision, even if T is double.
        float tmp_z, tmp_y, distance, mask_value;
        for (uint z = 0; z < shape.z; ++z) {
            tmp_z = Math::pow(static_cast<float>(z) - center.z, 2.f);
            for (uint y = 0; y < shape.y; ++y) {
                tmp_y = Math::pow(static_cast<float>(y) - center.y, 2.f);
                size_t offset = (z * shape.y + y) * shape.x;

                for (uint x = 0; x < shape.x; ++x) {
                    distance = (static_cast<float>(x) - center.x) * (static_cast<float>(x) - center.x) + tmp_y + tmp_z;

                    if constexpr (INVERT) {
                        if (distance > radius_squared_taper) {
                            mask_value = 1.f;
                        } else if (distance <= radius_squared) {
                            mask_value = 0.f;
                        } else {
                            distance = Math::sqrt(distance);
                            mask_value = (1.f - Math::cos(PI * (distance - radius) / taper_size)) * 0.5f;
                        }
                    } else {
                        if (distance > radius_squared_taper) { // excluded
                            mask_value = 0.f;
                        } else if (distance <= radius_squared) { // included
                            mask_value = 1.f;
                        } else { // within the taper
                            distance = Math::sqrt(distance);
                            mask_value = (1.f + Math::cos(PI * (distance - radius) / taper_size)) * 0.5f;
                            // If cos returns value < -1 (because of reasons?), then mask_value is negative...
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
    void sphere(T* inputs, T* outputs, size3_t shape, float3_t shifts, float radius, uint batches) {
        NOA_PROFILE_FUNCTION();

        size_t elements;
        if constexpr (ON_THE_FLY)
            elements = getElements(shape);

        float3_t center(shape / size_t{2});
        center += shifts;

        // Compute the mask using single precision, even if T is double.
        float radius_squared = radius * radius;
        float tmp_z, tmp_y, distance;
        T mask_value;
        for (uint z = 0; z < shape.z; ++z) {
            tmp_z = Math::pow(static_cast<float>(z) - center.z, 2.f);
            for (uint y = 0; y < shape.y; ++y) {
                tmp_y = Math::pow(static_cast<float>(y) - center.y, 2.f);
                size_t offset = (z * shape.y + y) * shape.x;

                for (uint x = 0; x < shape.x; ++x) {
                    distance = (static_cast<float>(x) - center.x) * (static_cast<float>(x) - center.x) + tmp_y + tmp_z;

                    if constexpr (INVERT) {
                        if (distance > radius_squared)
                            mask_value = 1;
                        else
                            mask_value = 0;
                    } else {
                        if (distance > radius_squared)
                            mask_value = 0;
                        else
                            mask_value = 1;
                    }
                    if constexpr (ON_THE_FLY) {
                        for (uint batch = 0; batch < batches; ++batch)
                            outputs[batch * elements + offset + x] = inputs[batch * elements + offset + x] * mask_value;
                    } else {
                        outputs[offset + x] = mask_value;
                    }
                }
            }
        }
    }

    #define INSTANTIATE_SPHERE(T)                                                           \
    template void sphere<true, true, T>(T*, T*, size3_t, float3_t, float, float, uint);     \
    template void sphere<true, false, T>(T*, T*, size3_t, float3_t, float, float, uint);    \
    template void sphere<false, true, T>(T*, T*, size3_t, float3_t, float, float, uint);    \
    template void sphere<false, false, T>(T*, T*, size3_t, float3_t, float, float, uint);   \
    template void sphere<true, true, T>(T*, T*, size3_t, float3_t, float, uint);            \
    template void sphere<true, false, T>(T*, T*, size3_t, float3_t, float, uint);           \
    template void sphere<false, true, T>(T*, T*, size3_t, float3_t, float, uint);           \
    template void sphere<false, false, T>(T*, T*, size3_t, float3_t, float, uint)

    INSTANTIATE_SPHERE(float);
    INSTANTIATE_SPHERE(double);
}
