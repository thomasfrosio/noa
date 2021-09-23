#include "noa/common/Exception.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/filter/Sphere.h"

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
    void sphereSoft3D_(const T* inputs, T* outputs, size3_t shape, float3_t shifts,
                       float radius, float taper_size, uint batches) {
        size_t elements = getElements(shape);
        float3_t center(shape / size_t{2});
        center += shifts;

        float radius_sqd = radius * radius;
        float radius_taper_sqd = math::pow(radius + taper_size, 2.f);

        // Compute the mask using single precision, even if T is double.
        float distance_sqd_z, distance_sqd_y, distance_sqd, mask_value;
        for (uint z = 0; z < shape.z; ++z) {
            distance_sqd_z = math::pow(static_cast<float>(z) - center.z, 2.f);
            for (uint y = 0; y < shape.y; ++y) {
                distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
                size_t offset = (z * shape.y + y) * shape.x;

                for (uint x = 0; x < shape.x; ++x) {
                    distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                    distance_sqd += distance_sqd_y + distance_sqd_z;
                    mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);

                    for (uint batch = 0; batch < batches; ++batch)
                        outputs[batch * elements + offset + x] = inputs[batch * elements + offset + x] *
                                                                 static_cast<T>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void sphereSoft3D_(T* output_mask, size3_t shape, float3_t shifts, float radius, float taper_size) {
        float3_t center(shape / size_t{2});
        center += shifts;

        float radius_sqd = radius * radius;
        float radius_taper_sqd = math::pow(radius + taper_size, 2.f);

        // Compute the mask using single precision, even if T is double.
        float distance_sqd_z, distance_sqd_y, distance_sqd, mask_value;
        for (uint z = 0; z < shape.z; ++z) {
            distance_sqd_z = math::pow(static_cast<float>(z) - center.z, 2.f);
            for (uint y = 0; y < shape.y; ++y) {
                distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
                size_t offset = (z * shape.y + y) * shape.x;

                for (uint x = 0; x < shape.x; ++x) {
                    distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                    distance_sqd += distance_sqd_y + distance_sqd_z;
                    mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
                    output_mask[offset + x] = static_cast<T>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void sphereSoft2D_(const T* inputs, T* outputs, size2_t shape, float2_t shifts,
                       float radius, float taper_size, uint batches) {
        size_t elements = getElements(shape);
        float2_t center(shape / size_t{2});
        center += shifts;

        float radius_sqd = radius * radius;
        float radius_taper_sqd = math::pow(radius + taper_size, 2.f);

        // Compute the mask using single precision, even if T is double.
        float distance_sqd_y, distance_sqd, mask_value;
        for (uint y = 0; y < shape.y; ++y) {
            distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
            size_t offset = y * shape.x;

            for (uint x = 0; x < shape.x; ++x) {
                distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                distance_sqd += distance_sqd_y;
                mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);

                for (uint batch = 0; batch < batches; ++batch)
                    outputs[batch * elements + offset + x] = inputs[batch * elements + offset + x] *
                                                             static_cast<T>(mask_value);
            }
        }
    }

    template<bool INVERT, typename T>
    void sphereSoft2D_(T* output_mask, size2_t shape, float2_t shifts, float radius, float taper_size) {
        float2_t center(shape / size_t{2});
        center += shifts;

        float radius_sqd = radius * radius;
        float radius_taper_sqd = math::pow(radius + taper_size, 2.f);

        // Compute the mask using single precision, even if T is double.
        float distance_sqd_y, distance_sqd, mask_value;
        for (uint y = 0; y < shape.y; ++y) {
            distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
            size_t offset = y * shape.x;

            for (uint x = 0; x < shape.x; ++x) {
                distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                distance_sqd += distance_sqd_y;
                mask_value = getSoftMask_<INVERT>(distance_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
                output_mask[offset + x] = static_cast<T>(mask_value);
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
    void sphereHard3D_(const T* inputs, T* outputs, size3_t shape, float3_t shifts, float radius, uint batches) {
        size_t elements = getElements(shape);
        float3_t center(shape / size_t{2});
        center += shifts;

        float radius_sqd = radius * radius;

        // Compute the mask using single precision, even if T is double.
        float distance_sqd_z, distance_sqd_y, distance_sqd, mask_value;
        for (uint z = 0; z < shape.z; ++z) {
            distance_sqd_z = math::pow(static_cast<float>(z) - center.z, 2.f);
            for (uint y = 0; y < shape.y; ++y) {
                distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
                size_t offset = (z * shape.y + y) * shape.x;

                for (uint x = 0; x < shape.x; ++x) {
                    distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                    distance_sqd += distance_sqd_y + distance_sqd_z;
                    mask_value = getHardMask_<INVERT>(distance_sqd, radius_sqd);

                    for (uint batch = 0; batch < batches; ++batch)
                        outputs[batch * elements + offset + x] = inputs[batch * elements + offset + x] *
                                                                 static_cast<T>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void sphereHard3D_(T* output_mask, size3_t shape, float3_t shifts, float radius) {
        float3_t center(shape / size_t{2});
        center += shifts;

        float radius_sqd = radius * radius;

        // Compute the mask using single precision, even if T is double.
        float distance_sqd_z, distance_sqd_y, distance_sqd, mask_value;
        for (uint z = 0; z < shape.z; ++z) {
            distance_sqd_z = math::pow(static_cast<float>(z) - center.z, 2.f);
            for (uint y = 0; y < shape.y; ++y) {
                distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
                size_t offset = (z * shape.y + y) * shape.x;

                for (uint x = 0; x < shape.x; ++x) {
                    distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                    distance_sqd += distance_sqd_y + distance_sqd_z;
                    mask_value = getHardMask_<INVERT>(distance_sqd, radius_sqd);
                    output_mask[offset + x] = static_cast<T>(mask_value);
                }
            }
        }
    }

    template<bool INVERT, typename T>
    void sphereHard2D_(const T* inputs, T* outputs, size2_t shape, float2_t shifts, float radius, uint batches) {
        size_t elements = getElements(shape);
        float2_t center(shape / size_t{2});
        center += shifts;

        float radius_sqd = radius * radius;

        // Compute the mask using single precision, even if T is double.
        float distance_sqd_y, distance_sqd, mask_value;
        for (uint y = 0; y < shape.y; ++y) {
            distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
            size_t offset = y * shape.x;

            for (uint x = 0; x < shape.x; ++x) {
                distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                distance_sqd += distance_sqd_y;
                mask_value = getHardMask_<INVERT>(distance_sqd, radius_sqd);

                for (uint batch = 0; batch < batches; ++batch)
                    outputs[batch * elements + offset + x] = inputs[batch * elements + offset + x] *
                                                             static_cast<T>(mask_value);
            }
        }
    }

    template<bool INVERT, typename T>
    void sphereHard2D_(T* output_mask, size2_t shape, float2_t shifts, float radius) {
        float2_t center(shape / size_t{2});
        center += shifts;

        float radius_sqd = radius * radius;

        // Compute the mask using single precision, even if T is double.
        float distance_sqd_y, distance_sqd, mask_value;
        for (uint y = 0; y < shape.y; ++y) {
            distance_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
            size_t offset = y * shape.x;

            for (uint x = 0; x < shape.x; ++x) {
                distance_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                distance_sqd += distance_sqd_y;
                mask_value = getHardMask_<INVERT>(distance_sqd, radius_sqd);
                output_mask[offset + x] = static_cast<T>(mask_value);
            }
        }
    }
}

// Definitions & Instantiations:
namespace noa::cpu::filter {
    template<bool INVERT, typename T>
    void sphere(const T* inputs, T* outputs, size3_t shape, float3_t shifts,
                float radius, float taper_size, uint batches) {
        NOA_PROFILE_FUNCTION();
        uint ndim = getNDim(shape);
        if (ndim == 3) {
            if (taper_size > 1e-5f)
                sphereSoft3D_<INVERT, T>(inputs, outputs, shape, shifts, radius, taper_size, batches);
            else
                sphereHard3D_<INVERT, T>(inputs, outputs, shape, shifts, radius, batches);
        } else if (ndim == 2) {
            size2_t shape_2D(shape.x, shape.y);
            float2_t shifts_2D(shifts.x, shifts.y);
            if (taper_size > 1e-5f)
                sphereSoft2D_<INVERT, T>(inputs, outputs, shape_2D, shifts_2D, radius, taper_size, batches);
            else
                sphereHard2D_<INVERT, T>(inputs, outputs, shape_2D, shifts_2D, radius, batches);
        } else {
            NOA_THROW("Cannot compute a sphere with shape:{}", shape);
        }
    }

    template<bool INVERT, typename T>
    void sphere(T* output_mask, size3_t shape, float3_t shifts, float radius, float taper_size) {
        NOA_PROFILE_FUNCTION();
        uint ndim = getNDim(shape);
        if (ndim == 3) {
            if (taper_size > 1e-5f)
                sphereSoft3D_<INVERT, T>(output_mask, shape, shifts, radius, taper_size);
            else
                sphereHard3D_<INVERT, T>(output_mask, shape, shifts, radius);
        } else if (ndim == 2) {
            size2_t shape_2D(shape.x, shape.y);
            float2_t shifts_2D(shifts.x, shifts.y);
            if (taper_size > 1e-5f)
                sphereSoft2D_<INVERT, T>(output_mask, shape_2D, shifts_2D, radius, taper_size);
            else
                sphereHard2D_<INVERT, T>(output_mask, shape_2D, shifts_2D, radius);
        } else {
            NOA_THROW("Cannot compute a sphere with shape:{}", shape);
        }
    }

    #define INSTANTIATE_SPHERE(T)                                                           \
    template void sphere<true, T>(const T*, T*, size3_t, float3_t, float, float, uint);     \
    template void sphere<false, T>(const T*, T*, size3_t, float3_t, float, float, uint);    \
    template void sphere<true, T>(T*, size3_t, float3_t, float, float);                     \
    template void sphere<false, T>(T*, size3_t, float3_t, float, float)

    INSTANTIATE_SPHERE(float);
    INSTANTIATE_SPHERE(double);
}