#include "noa/common/Exception.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/filter/Rectangle.h"

// Soft edges:
namespace {
    using namespace noa;

    template<bool INVERT>
    float getSoftMask3D_(float3_t distance, float3_t radius, float3_t radius_with_taper, float taper_size) {
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
    float getSoftMask2D_(float2_t distance, float2_t radius, float2_t radius_with_taper, float taper_size) {
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
    void rectangleSoft3D_(const T* inputs, T* outputs, size3_t shape, float3_t shifts, float3_t radius,
                          float taper_size, uint batches) {
        size_t elements = noa::elements(shape);
        float3_t center(shape / size_t{2});
        center += shifts;

        float3_t radius_with_taper = radius + taper_size;
        float mask_value;
        float3_t distance;
        for (uint z = 0; z < shape.z; ++z) {
            distance.z = math::abs(static_cast<float>(z) - center.z);
            for (uint y = 0; y < shape.y; ++y) {
                distance.y = math::abs(static_cast<float>(y) - center.y);
                size_t offset = (z * shape.y + y) * shape.x;
                for (uint x = 0; x < shape.x; ++x) {
                    distance.x = math::abs(static_cast<float>(x) - center.x);
                    mask_value = getSoftMask3D_<INVERT>(distance, radius, radius_with_taper, taper_size);
                    for (uint batch = 0; batch < batches; ++batch)
                        outputs[batch * elements + offset + x] = inputs[batch * elements + offset + x] *
                                                                 static_cast<T>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void rectangleSoft2D_(const T* inputs, T* outputs, size2_t shape, float2_t shifts, float2_t radius,
                          float taper_size, uint batches) {
        size_t elements = noa::elements(shape);
        float2_t center(shape / size_t{2});
        center += shifts;

        float2_t radius_with_taper = radius + taper_size;
        float mask_value;
        float2_t distance;
        for (uint y = 0; y < shape.y; ++y) {
            distance.y = math::abs(static_cast<float>(y) - center.y);
            size_t offset = y * shape.x;
            for (uint x = 0; x < shape.x; ++x) {
                distance.x = math::abs(static_cast<float>(x) - center.x);
                mask_value = getSoftMask2D_<INVERT>(distance, radius, radius_with_taper, taper_size);
                for (uint batch = 0; batch < batches; ++batch)
                    outputs[batch * elements + offset + x] = inputs[batch * elements + offset + x] *
                                                             static_cast<T>(mask_value);
            }
        }
    }

    template<bool INVERT, typename T>
    void rectangleSoft3D_(T* output_mask, size3_t shape, float3_t shifts, float3_t radius, float taper_size) {
        float3_t center(shape / size_t{2});
        center += shifts;

        float3_t radius_with_taper = radius + taper_size;
        float mask_value;
        float3_t distance;
        for (uint z = 0; z < shape.z; ++z) {
            distance.z = math::abs(static_cast<float>(z) - center.z);
            for (uint y = 0; y < shape.y; ++y) {
                distance.y = math::abs(static_cast<float>(y) - center.y);
                size_t offset = (z * shape.y + y) * shape.x;
                for (uint x = 0; x < shape.x; ++x) {
                    distance.x = math::abs(static_cast<float>(x) - center.x);
                    mask_value = getSoftMask3D_<INVERT>(distance, radius, radius_with_taper, taper_size);
                    output_mask[offset + x] = static_cast<T>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void rectangleSoft2D_(T* output_mask, size2_t shape, float2_t shifts, float2_t radius, float taper_size) {
        float2_t center(shape / size_t{2});
        center += shifts;

        float2_t distance, radius_with_taper = radius + taper_size;
        float mask_value;
        for (uint y = 0; y < shape.y; ++y) {
            distance.y = math::abs(static_cast<float>(y) - center.y);
            size_t offset = y * shape.x;
            for (uint x = 0; x < shape.x; ++x) {
                distance.x = math::abs(static_cast<float>(x) - center.x);
                mask_value = getSoftMask2D_<INVERT>(distance, radius, radius_with_taper, taper_size);
                output_mask[offset + x] = static_cast<T>(mask_value);
            }
        }
    }
}

// Hard edges:
namespace {
    using namespace noa;

    template<bool INVERT, typename T>
    T getHardMask3D_(float3_t distance, float3_t radius) {
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
    T getHardMask2D_(float2_t distance, float2_t radius) {
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
    void rectangleHard3D_(const T* inputs, T* outputs, size3_t shape, float3_t shifts, float3_t radius, uint batches) {
        size_t elements = noa::elements(shape);
        float3_t center(shape / size_t{2});
        center += shifts;

        // Compute the mask using single precision, even if T is double.
        T mask_value;
        float3_t distance;
        for (uint z = 0; z < shape.z; ++z) {
            distance.z = math::abs(static_cast<float>(z) - center.z);
            for (uint y = 0; y < shape.y; ++y) {
                distance.y = math::abs(static_cast<float>(y) - center.y);
                size_t offset = (z * shape.y + y) * shape.x;
                for (uint x = 0; x < shape.x; ++x) {
                    distance.x = math::abs(static_cast<float>(x) - center.x);
                    mask_value = getHardMask3D_<INVERT, T>(distance, radius);
                    for (uint batch = 0; batch < batches; ++batch)
                        outputs[batch * elements + offset + x] = inputs[batch * elements + offset + x] * mask_value;
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void rectangleHard2D_(const T* inputs, T* outputs, size2_t shape, float2_t shifts, float2_t radius, uint batches) {
        size_t elements = noa::elements(shape);
        float2_t center(shape / size_t{2});
        center += shifts;

        // Compute the mask using single precision, even if T is double.
        T mask_value;
        float2_t distance;
        for (uint y = 0; y < shape.y; ++y) {
            distance.y = math::abs(static_cast<float>(y) - center.y);
            size_t offset = y * shape.x;
            for (uint x = 0; x < shape.x; ++x) {
                distance.x = math::abs(static_cast<float>(x) - center.x);
                mask_value = getHardMask2D_<INVERT, T>(distance, radius);
                for (uint batch = 0; batch < batches; ++batch)
                    outputs[batch * elements + offset + x] = inputs[batch * elements + offset + x] * mask_value;
            }
        }
    }

    template<bool INVERT, typename T>
    void rectangleHard3D_(T* output_mask, size3_t shape, float3_t shifts, float3_t radius) {
        float3_t center(shape / size_t{2});
        center += shifts;

        // Compute the mask using single precision, even if T is double.
        float3_t distance;
        for (uint z = 0; z < shape.z; ++z) {
            distance.z = math::abs(static_cast<float>(z) - center.z);
            for (uint y = 0; y < shape.y; ++y) {
                distance.y = math::abs(static_cast<float>(y) - center.y);
                size_t offset = (z * shape.y + y) * shape.x;
                for (uint x = 0; x < shape.x; ++x) {
                    distance.x = math::abs(static_cast<float>(x) - center.x);
                    output_mask[offset + x] = getHardMask3D_<INVERT, T>(distance, radius);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void rectangleHard2D_(T* output_mask, size2_t shape, float2_t shifts, float2_t radius) {
        float2_t center(shape / size_t{2});
        center += shifts;

        // Compute the mask using single precision, even if T is double.
        float2_t distance;
        for (uint y = 0; y < shape.y; ++y) {
            distance.y = math::abs(static_cast<float>(y) - center.y);
            size_t offset = y * shape.x;
            for (uint x = 0; x < shape.x; ++x) {
                distance.x = math::abs(static_cast<float>(x) - center.x);
                output_mask[offset + x] = getHardMask2D_<INVERT, T>(distance, radius);
            }
        }
    }
}

// Definitions & Instantiations:
namespace noa::cpu::filter {
    template<bool INVERT, typename T>
    void rectangle(const T* inputs, T* outputs, size3_t shape, float3_t shifts, float3_t radius,
                   float taper_size, uint batches) {
        NOA_PROFILE_FUNCTION();
        size_t dim = ndim(shape);
        if (dim == 3) {
            if (taper_size > 1e-5f)
                rectangleSoft3D_<INVERT, T>(inputs, outputs, shape, shifts, radius, taper_size, batches);
            else
                rectangleHard3D_<INVERT, T>(inputs, outputs, shape, shifts, radius, batches);
        } else if (dim == 2) {
            size2_t shape2D(shape.x, shape.y);
            float2_t shifts2D(shifts.x, shifts.y);
            float2_t radius2D(radius.x, radius.y);
            if (taper_size > 1e-5f)
                rectangleSoft2D_<INVERT>(inputs, outputs, shape2D, shifts2D, radius2D, taper_size, batches);
            else
                rectangleHard2D_<INVERT>(inputs, outputs, shape2D, shifts2D, radius2D, batches);
        } else {
            NOA_THROW("Cannot compute a sphere with shape:{}", shape);
        }
    }

    template<bool INVERT, typename T>
    void rectangle(T* output_mask, size3_t shape, float3_t shifts, float3_t radius, float taper_size) {
        NOA_PROFILE_FUNCTION();
        size_t dim = ndim(shape);
        if (dim == 3) {
            if (taper_size > 1e-5f)
                rectangleSoft3D_<INVERT>(output_mask, shape, shifts, radius, taper_size);
            else
                rectangleHard3D_<INVERT>(output_mask, shape, shifts, radius);
        } else if (dim == 2) {
            size2_t shape2D(shape.x, shape.y);
            float2_t shifts2D(shifts.x, shifts.y);
            float2_t radius2D(radius.x, radius.y);
            if (taper_size > 1e-5f)
                rectangleSoft2D_<INVERT>(output_mask, shape2D, shifts2D, radius2D, taper_size);
            else
                rectangleHard2D_<INVERT>(output_mask, shape2D, shifts2D, radius2D);
        } else {
            NOA_THROW("Cannot compute a sphere with shape:{}", shape);
        }
    }

    #define INSTANTIATE_RECTANGLE(T)                                                            \
    template void rectangle<true, T>(const T*, T*, size3_t, float3_t, float3_t, float, uint);   \
    template void rectangle<false, T>(const T*, T*, size3_t, float3_t, float3_t, float, uint);  \
    template void rectangle<true, T>(T*, size3_t, float3_t, float3_t, float);                   \
    template void rectangle<false, T>(T*, size3_t, float3_t, float3_t, float)

    INSTANTIATE_RECTANGLE(float);
    INSTANTIATE_RECTANGLE(double);
}
