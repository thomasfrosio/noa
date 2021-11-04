#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/filter/Shape.h"

// Soft edges:
namespace {
    using namespace noa;

    template<bool INVERT>
    float getSoftMask_(float distance_xy_sqd, float radius_xy_sqd, float radius_xy, float radius_xy_sqd_with_taper,
                       float distance_z, float radius_z, float radius_z_with_taper, float taper_size) {
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
    void cylinderSoft_(const T* input, T* output, size3_t shape, float3_t center,
                       float radius_xy, float radius_z, float taper_size) {
        using real_t = traits::value_type_t<T>;
        const float radius_xy_sqd = radius_xy * radius_xy;
        const float radius_xy_taper_sqd = math::pow(radius_xy + taper_size, 2.f);
        const float radius_z_taper = radius_z + taper_size;

        for (size_t z = 0; z < shape.z; ++z) {
            const float distance_z = math::abs(static_cast<float>(z) - center.z);
            for (size_t y = 0; y < shape.y; ++y) {
                const float distance_y_sqd = math::pow(static_cast<float>(y) - center.y, 2.f);
                const size_t offset = (z * shape.y + y) * shape.x;
                for (size_t x = 0; x < shape.x; ++x) {
                    float distance_xy_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                    distance_xy_sqd += distance_y_sqd;
                    const float mask_value = getSoftMask_<INVERT>(
                            distance_xy_sqd, radius_xy_sqd, radius_xy, radius_xy_taper_sqd,
                            distance_z, radius_z, radius_z_taper, taper_size);
                    output[offset + x] = input[offset + x] * static_cast<real_t>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void cylinderSoft_(T* output_mask, size3_t shape, float3_t center,
                       float radius_xy, float radius_z, float taper_size) {
        using real_t = traits::value_type_t<T>;
        const float radius_xy_sqd = radius_xy * radius_xy;
        const float radius_xy_taper_sqd = math::pow(radius_xy + taper_size, 2.f);
        const float radius_z_taper = radius_z + taper_size;

        for (size_t z = 0; z < shape.z; ++z) {
            const float distance_z = math::abs(static_cast<float>(z) - center.z);
            for (size_t y = 0; y < shape.y; ++y) {
                const float distance_y_sqd = math::pow(static_cast<float>(y) - center.y, 2.f);
                const size_t offset = (z * shape.y + y) * shape.x;
                for (size_t x = 0; x < shape.x; ++x) {
                    float distance_xy_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                    distance_xy_sqd += distance_y_sqd;
                    const float mask_value = getSoftMask_<INVERT>(
                            distance_xy_sqd, radius_xy_sqd, radius_xy, radius_xy_taper_sqd,
                            distance_z, radius_z, radius_z_taper, taper_size);
                    output_mask[offset + x] = static_cast<real_t>(mask_value);
                }
            }
        }
    }
}

// Hard edges:
namespace {
    using namespace noa;

    template<bool INVERT>
    float getHardMask_(float distance_xy_sqd, float radius_xy_sqd, float distance_z, float radius_z) {
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
    void cylinderHard_(const T* input, T* output, size3_t shape, float3_t center,
                       float radius_xy, float radius_z) {
        using real_t = traits::value_type_t<T>;
        const float radius_xy_sqd = radius_xy * radius_xy;
        for (size_t z = 0; z < shape.z; ++z) {
            const float distance_z = math::abs(static_cast<float>(z) - center.z);
            for (size_t y = 0; y < shape.y; ++y) {
                const float distance_y_sqd = math::pow(static_cast<float>(y) - center.y, 2.f);
                const size_t offset = (z * shape.y + y) * shape.x;
                for (size_t x = 0; x < shape.x; ++x) {
                    float distance_xy_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                    distance_xy_sqd += distance_y_sqd;
                    const float mask_value = getHardMask_<INVERT>(distance_xy_sqd, radius_xy_sqd, distance_z, radius_z);
                    output[offset + x] = input[offset + x] * static_cast<real_t>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void cylinderHard_(T* output_mask, size3_t shape, float3_t center, float radius_xy, float radius_z) {
        using real_t = traits::value_type_t<T>;
        const float radius_xy_sqd = radius_xy * radius_xy;
        for (size_t z = 0; z < shape.z; ++z) {
            const float distance_z = math::abs(static_cast<float>(z) - center.z);
            for (size_t y = 0; y < shape.y; ++y) {
                const float distance_y_sqd = math::pow(static_cast<float>(y) - center.y, 2.f);
                const size_t offset = (z * shape.y + y) * shape.x;
                for (size_t x = 0; x < shape.x; ++x) {
                    float distance_xy_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                    distance_xy_sqd += distance_y_sqd;
                    const float mask_value = getHardMask_<INVERT>(distance_xy_sqd, radius_xy_sqd, distance_z, radius_z);
                    output_mask[offset + x] = static_cast<real_t>(mask_value);
                }
            }
        }
    }
}

namespace noa::cpu::filter {
    template<bool INVERT, typename T>
    void cylinder3D(const T* inputs, T* outputs, size3_t shape, size_t batches,
                    float3_t center, float radius_xy, float radius_z, float taper_size) {
        NOA_PROFILE_FUNCTION();
        size_t elements = noa::elements(shape);
        if (inputs) {
            for (size_t batch = 0; batch < batches; ++batch) {
                if (taper_size > 1e-5f)
                    cylinderSoft_<INVERT, T>(inputs + batch * elements, outputs + batch * elements,
                                             shape, center, radius_xy, radius_z, taper_size);
                else
                    cylinderHard_<INVERT, T>(inputs + batch * elements, outputs + batch * elements,
                                             shape, center, radius_xy, radius_z);
            }
        } else {
            if (taper_size > 1e-5f)
                cylinderSoft_<INVERT, T>(outputs, shape, center, radius_xy, radius_z, taper_size);
            else
                cylinderHard_<INVERT, T>(outputs, shape, center, radius_xy, radius_z);
            for (size_t batch = 1; batch < batches; ++batch)
                memory::copy(outputs, outputs + elements * batch, elements);
        }
    }


    #define NOA_INSTANTIATE_CYLINDER_(T)                                                                \
    template void cylinder3D<true, T>(const T*, T*, size3_t, size_t, float3_t, float, float, float);    \
    template void cylinder3D<false, T>(const T*, T*, size3_t, size_t, float3_t, float, float, float)

    NOA_INSTANTIATE_CYLINDER_(float);
    NOA_INSTANTIATE_CYLINDER_(double);
    NOA_INSTANTIATE_CYLINDER_(cfloat_t);
    NOA_INSTANTIATE_CYLINDER_(cdouble_t);
}
