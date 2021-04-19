#include "noa/cpu/masks/Cylinder.h"

#include "noa/Math.h"
#include "noa/Profiler.h"

namespace Noa::Mask::Details {
    template<bool INVERT, bool ON_THE_FLY, typename T>
    void cylinder(T* inputs, T* outputs, size3_t shape, float3_t shifts, float radius_xy, float radius_z,
                  float taper_size, uint batches) {
        constexpr float PI = Math::Constants<float>::PI;
        NOA_PROFILE_FUNCTION();

        size_t elements;
        if constexpr (ON_THE_FLY)
            elements = getElements(shape);

        float3_t center(shape / size_t{2});
        center += shifts;

        float radius_squared = radius_xy * radius_xy;
        float radius_squared_taper = Math::pow(radius_xy + taper_size, 2.f);
        float radius_z_taper = radius_z + taper_size;

        // Compute the mask using single precision, even if T is double.
        float tmp_y, distance_xy, distance_z, mask_value;
        for (uint z = 0; z < shape.z; ++z) {
            distance_z = Math::abs(static_cast<float>(z) - center.z);

            for (uint y = 0; y < shape.y; ++y) {
                tmp_y = Math::pow(static_cast<float>(y) - center.y, 2.f);
                size_t offset = (z * shape.y + y) * shape.x;

                for (uint x = 0; x < shape.x; ++x) {
                    distance_xy = (static_cast<float>(x) - center.x) * (static_cast<float>(x) - center.x) + tmp_y;

                    if constexpr (INVERT) {
                        if (distance_z > radius_z_taper || distance_xy > radius_squared_taper) {
                            mask_value = 1.f;
                        } else {
                            if (distance_xy <= radius_squared) {
                                mask_value = 1.f;
                            } else {
                                distance_xy = Math::sqrt(distance_xy);
                                mask_value = (1.f + Math::cos(PI * (distance_xy - radius_xy) / taper_size)) * 0.5f;
                            }
                            if (distance_z > radius_z) { // <= ?
                                mask_value *= (1.f + Math::cos(PI * (distance_z - radius_z) / taper_size)) * 0.5f;
                            }
                            mask_value = 1 - mask_value;
                        }
                    } else {
                        if (distance_z > radius_z_taper || distance_xy > radius_squared_taper) {
                            mask_value = 0.f;
                        } else {
                            if (distance_xy <= radius_squared) {
                                mask_value = 1.f;
                            } else {
                                distance_xy = Math::sqrt(distance_xy);
                                mask_value = (1.f + Math::cos(PI * (distance_xy - radius_xy) / taper_size)) * 0.5f;
                            }
                            if (distance_z > radius_z) { // <= ?
                                mask_value *= (1.f + Math::cos(PI * (distance_z - radius_z) / taper_size)) * 0.5f;
                            }
                        }
                    }
                    if constexpr (ON_THE_FLY) {
                        for (uint batch = 0; batch < batches; ++batch)
                            outputs[batch * elements + offset + x] = inputs[batch * offset + x] *
                                                                     static_cast<T>(mask_value);
                    } else {
                        outputs[offset + x] = static_cast<T>(mask_value);
                    }
                }
            }
        }
    }

    template<bool INVERT, bool ON_THE_FLY, typename T>
    void cylinder(T* inputs, T* outputs, size3_t shape, float3_t shifts,
                  float radius_xy, float radius_z, uint batches) {
        NOA_PROFILE_FUNCTION();

        size_t elements;
        if constexpr (ON_THE_FLY)
            elements = getElements(shape);

        float3_t center(shape / size_t{2});
        center += shifts;

        float radius_squared = radius_xy * radius_xy;

        // Compute the mask using single precision, even if T is double.
        float tmp_y, distance_xy, distance_z, mask_value;
        for (uint z = 0; z < shape.z; ++z) {
            distance_z = Math::abs(static_cast<float>(z) - center.z);

            for (uint y = 0; y < shape.y; ++y) {
                tmp_y = Math::pow(static_cast<float>(y) - center.y, 2.f);
                size_t offset = (z * shape.y + y) * shape.x;

                for (uint x = 0; x < shape.x; ++x) {
                    distance_xy = (static_cast<float>(x) - center.x) * (static_cast<float>(x) - center.x) + tmp_y;

                    if constexpr (INVERT) {
                        if (distance_z > radius_z || distance_xy > radius_squared) {
                            mask_value = 1.f;
                        } else {
                            mask_value = 0.f;
                        }
                    } else {
                        if (distance_z > radius_z || distance_xy > radius_squared) {
                            mask_value = 0.f;
                        } else {
                            mask_value = 1.f;
                        }
                    }
                    if constexpr (ON_THE_FLY) {
                        for (uint batch = 0; batch < batches; ++batch)
                            outputs[batch * elements + offset + x] = inputs[batch * offset + x] *
                                                                     static_cast<T>(mask_value);
                    } else {
                        outputs[offset + x] = static_cast<T>(mask_value);
                    }
                }
            }
        }
    }

    #define INSTANTIATE_SPHERE(T)                                                                   \
    template void cylinder<true, true, T>(T*, T*, size3_t, float3_t, float, float, float, uint);    \
    template void cylinder<true, false, T>(T*, T*, size3_t, float3_t, float, float, float, uint);   \
    template void cylinder<false, true, T>(T*, T*, size3_t, float3_t, float, float, float, uint);   \
    template void cylinder<false, false, T>(T*, T*, size3_t, float3_t, float, float, float, uint);  \
    template void cylinder<true, true, T>(T*, T*, size3_t, float3_t, float, float, uint);           \
    template void cylinder<true, false, T>(T*, T*, size3_t, float3_t, float, float, uint);          \
    template void cylinder<false, true, T>(T*, T*, size3_t, float3_t, float, float, uint);          \
    template void cylinder<false, false, T>(T*, T*, size3_t, float3_t, float, float, uint)

    INSTANTIATE_SPHERE(float);
    INSTANTIATE_SPHERE(double);
}
