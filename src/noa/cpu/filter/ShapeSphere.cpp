#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/filter/Shape.h"

// Soft edges:
namespace {
    using namespace noa;

    template<bool INVERT>
    static float getSoftMask_(float distance_sqd, float radius, float radius_sqd,
                              float radius_taper_sqd, float taper_size) {
        float mask_value;
        constexpr float PI = math::Constants<float>::PI;
        if constexpr (INVERT) {
            if (distance_sqd > radius_taper_sqd) {
                mask_value = 1.f;
            } else if (distance_sqd <= radius_sqd) {
                mask_value = 0.f;
            } else {
                float distance = math::sqrt(distance_sqd);
                mask_value = (1.f - math::cos(PI * (distance - radius) / taper_size)) * 0.5f;
            }
        } else {
            if (distance_sqd > radius_taper_sqd) {
                mask_value = 0.f;
            } else if (distance_sqd <= radius_sqd) {
                mask_value = 1.f;
            } else {
                distance_sqd = math::sqrt(distance_sqd);
                mask_value = (1.f + math::cos(PI * (distance_sqd - radius) / taper_size)) * 0.5f;
            }
        }
        return mask_value;
    }

    template<bool INVERT, typename T>
    void sphereSoft3D_(const T* inputs, T* outputs, size3_t shape,
                       float3_t center, float radius, float taper_size) {
        using real_t = traits::value_type_t<T>;
        float radius_sqd = radius * radius;
        float radius_taper_sqd = math::pow(radius + taper_size, 2.f);

        // Compute the mask using single precision, even if T is double.
        float distance_sqd_z, distance_sqd_y, distance_sqd, mask_value;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd_z = math::pow(static_cast<float>(z) - center.z, 2.f);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
                size_t offset = (z * shape.y + y) * shape.x;
                for (size_t x = 0; x < shape.x; ++x) {
                    distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                    distance_sqd += distance_sqd_y + distance_sqd_z;
                    mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
                    outputs[offset + x] = inputs[offset + x] * static_cast<real_t>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void sphereSoft3D_(T* output_mask, size3_t shape, float3_t center, float radius, float taper_size) {
        using real_t = traits::value_type_t<T>;
        float radius_sqd = radius * radius;
        float radius_taper_sqd = math::pow(radius + taper_size, 2.f);
        float distance_sqd_z, distance_sqd_y, distance_sqd, mask_value;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd_z = math::pow(static_cast<float>(z) - center.z, 2.f);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
                size_t offset = (z * shape.y + y) * shape.x;
                for (size_t x = 0; x < shape.x; ++x) {
                    distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                    distance_sqd += distance_sqd_y + distance_sqd_z;
                    mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
                    output_mask[offset + x] = static_cast<real_t>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void sphereSoft2D_(const T* inputs, T* outputs, size2_t shape,
                       float2_t center, float radius, float taper_size) {
        using real_t = traits::value_type_t<T>;
        float radius_sqd = radius * radius;
        float radius_taper_sqd = math::pow(radius + taper_size, 2.f);
        float distance_sqd_y, distance_sqd, mask_value;
        for (size_t y = 0; y < shape.y; ++y) {
            distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
            size_t offset = y * shape.x;
            for (size_t x = 0; x < shape.x; ++x) {
                distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                distance_sqd += distance_sqd_y;
                mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
                outputs[offset + x] = inputs[offset + x] * static_cast<real_t>(mask_value);
            }
        }
    }

    template<bool INVERT, typename T>
    void sphereSoft2D_(T* output_mask, size2_t shape, float2_t center, float radius, float taper_size) {
        using real_t = traits::value_type_t<T>;
        float radius_sqd = radius * radius;
        float radius_taper_sqd = math::pow(radius + taper_size, 2.f);
        float distance_sqd_y, distance_sqd, mask_value;
        for (size_t y = 0; y < shape.y; ++y) {
            distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
            size_t offset = y * shape.x;
            for (size_t x = 0; x < shape.x; ++x) {
                distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                distance_sqd += distance_sqd_y;
                mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
                output_mask[offset + x] = static_cast<real_t>(mask_value);
            }
        }
    }
}

// Hard edges:
namespace {
    using namespace noa;

    template<bool INVERT>
    static float getHardMask_(float distance_sqd, float radius_sqd) {
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
    void sphereHard3D_(const T* inputs, T* outputs, size3_t shape, float3_t center, float radius) {
        using real_t = traits::value_type_t<T>;
        float radius_sqd = radius * radius;
        float distance_sqd_z, distance_sqd_y, distance_sqd, mask_value;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd_z = math::pow(static_cast<float>(z) - center.z, 2.f);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
                size_t offset = (z * shape.y + y) * shape.x;
                for (size_t x = 0; x < shape.x; ++x) {
                    distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                    distance_sqd += distance_sqd_y + distance_sqd_z;
                    mask_value = getHardMask_<INVERT>(distance_sqd, radius_sqd);
                    outputs[offset + x] = inputs[offset + x] * static_cast<real_t>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void sphereHard3D_(T* output_mask, size3_t shape, float3_t center, float radius) {
        using real_t = traits::value_type_t<T>;
        float radius_sqd = radius * radius;
        float distance_sqd_z, distance_sqd_y, distance_sqd, mask_value;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd_z = math::pow(static_cast<float>(z) - center.z, 2.f);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
                size_t offset = (z * shape.y + y) * shape.x;
                for (size_t x = 0; x < shape.x; ++x) {
                    distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                    distance_sqd += distance_sqd_y + distance_sqd_z;
                    mask_value = getHardMask_<INVERT>(distance_sqd, radius_sqd);
                    output_mask[offset + x] = static_cast<real_t>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void sphereHard2D_(const T* inputs, T* outputs, size2_t shape, float2_t center, float radius) {
        using real_t = traits::value_type_t<T>;
        float radius_sqd = radius * radius;
        float distance_sqd_y, distance_sqd, mask_value;
        for (size_t y = 0; y < shape.y; ++y) {
            distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
            size_t offset = y * shape.x;
            for (size_t x = 0; x < shape.x; ++x) {
                distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                distance_sqd += distance_sqd_y;
                mask_value = getHardMask_<INVERT>(distance_sqd, radius_sqd);
                outputs[offset + x] = inputs[offset + x] * static_cast<real_t>(mask_value);
            }
        }
    }

    template<bool INVERT, typename T>
    void sphereHard2D_(T* output_mask, size2_t shape, float2_t center, float radius) {
        using real_t = traits::value_type_t<T>;
        float radius_sqd = radius * radius;
        float distance_sqd_y, distance_sqd, mask_value;
        for (size_t y = 0; y < shape.y; ++y) {
            distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
            size_t offset = y * shape.x;
            for (size_t x = 0; x < shape.x; ++x) {
                distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                distance_sqd += distance_sqd_y;
                mask_value = getHardMask_<INVERT>(distance_sqd, radius_sqd);
                output_mask[offset + x] = static_cast<real_t>(mask_value);
            }
        }
    }
}

namespace noa::cpu::filter {
    template<bool INVERT, typename T>
    void sphere2D(const T* inputs, T* outputs, size2_t shape, size_t batches,
                  float2_t center, float radius, float taper_size) {
        NOA_PROFILE_FUNCTION();
        size_t elements = noa::elements(shape);
        if (inputs) {
            for (size_t batch = 0; batch < batches; ++batch) {
                if (taper_size > 1e-5f)
                    sphereSoft2D_<INVERT, T>(inputs + batch * elements, outputs + batch * elements,
                                             shape, center, radius, taper_size);
                else
                    sphereHard2D_<INVERT, T>(inputs + batch * elements, outputs + batch * elements,
                                             shape, center, radius);
            }
        } else {
            if (taper_size > 1e-5f)
                sphereSoft2D_<INVERT, T>(outputs, shape, center, radius, taper_size);
            else
                sphereHard2D_<INVERT, T>(outputs, shape, center, radius);
            for (size_t batch = 1; batch < batches; ++batch)
                memory::copy(outputs, outputs + elements * batch, elements);
        }
    }

    template<bool INVERT, typename T>
    void sphere3D(const T* inputs, T* outputs, size3_t shape, size_t batches,
                  float3_t center, float radius, float taper_size) {
        NOA_PROFILE_FUNCTION();
        size_t elements = noa::elements(shape);
        if (inputs) {
            for (size_t batch = 0; batch < batches; ++batch) {
                if (taper_size > 1e-5f)
                    sphereSoft3D_<INVERT, T>(inputs + batch * elements, outputs + batch * elements,
                                             shape, center, radius, taper_size);
                else
                    sphereHard3D_<INVERT, T>(inputs + batch * elements, outputs + batch * elements,
                                             shape, center, radius);
            }
        } else {
            if (taper_size > 1e-5f)
                sphereSoft3D_<INVERT, T>(outputs, shape, center, radius, taper_size);
            else
                sphereHard3D_<INVERT, T>(outputs, shape, center, radius);
            for (size_t batch = 1; batch < batches; ++batch)
                memory::copy(outputs, outputs + elements * batch, elements);
        }
    }

    #define NOA_INSTANTIATE_SPHERE_(T)                                                          \
    template void sphere2D<true, T>(const T*, T*, size2_t, size_t, float2_t, float, float);     \
    template void sphere2D<false, T>(const T*, T*, size2_t, size_t, float2_t, float, float);    \
    template void sphere3D<true, T>(const T*, T*, size3_t, size_t, float3_t, float, float);     \
    template void sphere3D<false, T>(const T*, T*, size3_t, size_t, float3_t, float, float)

    NOA_INSTANTIATE_SPHERE_(float);
    NOA_INSTANTIATE_SPHERE_(double);
    NOA_INSTANTIATE_SPHERE_(cfloat_t);
    NOA_INSTANTIATE_SPHERE_(cdouble_t);
}
