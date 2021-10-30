#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/filter/Shape.h"

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
    void rectangleSoft3D_(const T* inputs, T* outputs, size3_t shape, float3_t center,
                          float3_t radius, float taper_size) {
        using real_t = traits::value_type_t<T>;
        float3_t radius_with_taper = radius + taper_size;
        float3_t distance;
        for (size_t z = 0; z < shape.z; ++z) {
            distance.z = math::abs(static_cast<float>(z) - center.z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance.y = math::abs(static_cast<float>(y) - center.y);
                size_t offset = (z * shape.y + y) * shape.x;
                for (size_t x = 0; x < shape.x; ++x) {
                    distance.x = math::abs(static_cast<float>(x) - center.x);
                    float mask_value = getSoftMask3D_<INVERT>(distance, radius, radius_with_taper, taper_size);
                    outputs[offset + x] = inputs[offset + x] * static_cast<real_t>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void rectangleSoft2D_(const T* inputs, T* outputs, size2_t shape, float2_t center, float2_t radius,
                          float taper_size) {
        using real_t = traits::value_type_t<T>;
        float2_t radius_with_taper = radius + taper_size;
        float2_t distance;
        for (size_t y = 0; y < shape.y; ++y) {
            distance.y = math::abs(static_cast<float>(y) - center.y);
            size_t offset = y * shape.x;
            for (size_t x = 0; x < shape.x; ++x) {
                distance.x = math::abs(static_cast<float>(x) - center.x);
                float mask_value = getSoftMask2D_<INVERT>(distance, radius, radius_with_taper, taper_size);
                outputs[offset + x] = inputs[offset + x] * static_cast<real_t>(mask_value);
            }
        }
    }

    template<bool INVERT, typename T>
    void rectangleSoft3D_(T* output_mask, size3_t shape, float3_t center, float3_t radius, float taper_size) {
        using real_t = traits::value_type_t<T>;
        float3_t radius_with_taper = radius + taper_size;
        float3_t distance;
        for (uint z = 0; z < shape.z; ++z) {
            distance.z = math::abs(static_cast<float>(z) - center.z);
            for (uint y = 0; y < shape.y; ++y) {
                distance.y = math::abs(static_cast<float>(y) - center.y);
                size_t offset = (z * shape.y + y) * shape.x;
                for (uint x = 0; x < shape.x; ++x) {
                    distance.x = math::abs(static_cast<float>(x) - center.x);
                    float mask_value = getSoftMask3D_<INVERT>(distance, radius, radius_with_taper, taper_size);
                    output_mask[offset + x] = static_cast<real_t>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void rectangleSoft2D_(T* output_mask, size2_t shape, float2_t center, float2_t radius, float taper_size) {
        using real_t = traits::value_type_t<T>;
        float2_t distance, radius_with_taper = radius + taper_size;
        for (size_t y = 0; y < shape.y; ++y) {
            distance.y = math::abs(static_cast<float>(y) - center.y);
            size_t offset = y * shape.x;
            for (size_t x = 0; x < shape.x; ++x) {
                distance.x = math::abs(static_cast<float>(x) - center.x);
                float mask_value = getSoftMask2D_<INVERT>(distance, radius, radius_with_taper, taper_size);
                output_mask[offset + x] = static_cast<real_t>(mask_value);
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
    void rectangleHard3D_(const T* inputs, T* outputs, size3_t shape, float3_t center, float3_t radius) {
        using real_t = traits::value_type_t<T>;
        float3_t distance;
        for (size_t z = 0; z < shape.z; ++z) {
            distance.z = math::abs(static_cast<float>(z) - center.z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance.y = math::abs(static_cast<float>(y) - center.y);
                size_t offset = (z * shape.y + y) * shape.x;
                for (size_t x = 0; x < shape.x; ++x) {
                    distance.x = math::abs(static_cast<float>(x) - center.x);
                    auto mask_value = getHardMask3D_<INVERT, real_t>(distance, radius);
                    outputs[offset + x] = inputs[offset + x] * mask_value;
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void rectangleHard2D_(const T* inputs, T* outputs, size2_t shape, float2_t center, float2_t radius) {
        using real_t = traits::value_type_t<T>;
        float2_t distance;
        for (size_t y = 0; y < shape.y; ++y) {
            distance.y = math::abs(static_cast<float>(y) - center.y);
            size_t offset = y * shape.x;
            for (size_t x = 0; x < shape.x; ++x) {
                distance.x = math::abs(static_cast<float>(x) - center.x);
                auto mask_value = getHardMask2D_<INVERT, real_t>(distance, radius);
                outputs[offset + x] = inputs[offset + x] * mask_value;
            }
        }
    }

    template<bool INVERT, typename T>
    void rectangleHard3D_(T* output_mask, size3_t shape, float3_t center, float3_t radius) {
        using real_t = traits::value_type_t<T>;
        float3_t distance;
        for (size_t z = 0; z < shape.z; ++z) {
            distance.z = math::abs(static_cast<float>(z) - center.z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance.y = math::abs(static_cast<float>(y) - center.y);
                size_t offset = (z * shape.y + y) * shape.x;
                for (size_t x = 0; x < shape.x; ++x) {
                    distance.x = math::abs(static_cast<float>(x) - center.x);
                    output_mask[offset + x] = getHardMask3D_<INVERT, real_t>(distance, radius);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void rectangleHard2D_(T* output_mask, size2_t shape, float2_t center, float2_t radius) {
        using real_t = traits::value_type_t<T>;
        float2_t distance;
        for (size_t y = 0; y < shape.y; ++y) {
            distance.y = math::abs(static_cast<float>(y) - center.y);
            size_t offset = y * shape.x;
            for (size_t x = 0; x < shape.x; ++x) {
                distance.x = math::abs(static_cast<float>(x) - center.x);
                output_mask[offset + x] = getHardMask2D_<INVERT, real_t>(distance, radius);
            }
        }
    }
}

// Definitions & Instantiations:
namespace noa::cpu::filter {
    template<bool INVERT, typename T>
    void rectangle2D(const T* inputs, T* outputs, size2_t shape, size_t batches,
                     float2_t center, float2_t radius, float taper_size) {
        NOA_PROFILE_FUNCTION();
        size_t elements = noa::elements(shape);
        if (inputs) {
            for (size_t batch = 1; batch < batches; ++batch) {
                if (taper_size > 1e-5f)
                    rectangleSoft2D_<INVERT, T>(inputs + batch * elements, outputs + batch * elements,
                                                shape, center, radius, taper_size);
                else
                    rectangleHard2D_<INVERT, T>(inputs + batch * elements, outputs + batch * elements,
                                                shape, center, radius);
            }
        } else {
            if (taper_size > 1e-5f)
                rectangleSoft2D_<INVERT, T>(outputs, shape, center, radius, taper_size);
            else
                rectangleHard2D_<INVERT, T>(outputs, shape, center, radius);
            for (size_t batch = 1; batch < batches; ++batch)
                memory::copy(outputs, outputs + elements * batch, elements);
        }
    }

    template<bool INVERT, typename T>
    void rectangle3D(const T* inputs, T* outputs, size3_t shape, size_t batches,
                     float3_t center, float3_t radius, float taper_size) {
        NOA_PROFILE_FUNCTION();
        size_t elements = noa::elements(shape);
        if (inputs) {
            for (size_t batch = 1; batch < batches; ++batch) {
                if (taper_size > 1e-5f)
                    rectangleSoft3D_<INVERT, T>(inputs + batch * elements, outputs + batch * elements,
                                                shape, center, radius, taper_size);
                else
                    rectangleHard3D_<INVERT, T>(inputs + batch * elements, outputs + batch * elements,
                                                shape, center, radius);
            }
        } else {
            if (taper_size > 1e-5f)
                rectangleSoft3D_<INVERT, T>(outputs, shape, center, radius, taper_size);
            else
                rectangleHard3D_<INVERT, T>(outputs, shape, center, radius);
            for (size_t batch = 1; batch < batches; ++batch)
                memory::copy(outputs, outputs + elements * batch, elements);
        }
    }

    #define NOA_INSTANTIATE_RECTANGLE_(T)                                                           \
    template void rectangle2D<true, T>(const T*, T*, size2_t, size_t, float2_t, float2_t, float);   \
    template void rectangle2D<false, T>(const T*, T*, size2_t, size_t, float2_t, float2_t, float);  \
    template void rectangle3D<true, T>(const T*, T*, size3_t, size_t, float3_t, float3_t, float);   \
    template void rectangle3D<false, T>(const T*, T*, size3_t, size_t, float3_t, float3_t, float)

    NOA_INSTANTIATE_RECTANGLE_(float);
    NOA_INSTANTIATE_RECTANGLE_(double);
    NOA_INSTANTIATE_RECTANGLE_(cfloat_t);
    NOA_INSTANTIATE_RECTANGLE_(cdouble_t);
}
