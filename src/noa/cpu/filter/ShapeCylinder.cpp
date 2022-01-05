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
        float mask;
        if constexpr (INVERT) {
            if (distance_z > radius_z_with_taper || distance_xy_sqd > radius_xy_sqd_with_taper) {
                mask = 1.f;
            } else {
                if (distance_xy_sqd <= radius_xy_sqd) {
                    mask = 1.f;
                } else {
                    float distance_xy = math::sqrt(distance_xy_sqd);
                    mask = (1.f + math::cos(PI * (distance_xy - radius_xy) / taper_size)) * 0.5f;
                }
                if (distance_z > radius_z)
                    mask *= (1.f + math::cos(PI * (distance_z - radius_z) / taper_size)) * 0.5f;
                mask = 1 - mask;
            }
        } else {
            if (distance_z > radius_z_with_taper || distance_xy_sqd > radius_xy_sqd_with_taper) {
                mask = 0.f;
            } else {
                if (distance_xy_sqd <= radius_xy_sqd) {
                    mask = 1.f;
                } else {
                    float distance_xy = math::sqrt(distance_xy_sqd);
                    mask = (1.f + math::cos(PI * (distance_xy - radius_xy) / taper_size)) * 0.5f;
                }
                if (distance_z > radius_z)
                    mask *= (1.f + math::cos(PI * (distance_z - radius_z) / taper_size)) * 0.5f;
            }
        }
        return mask;
    }

    template<bool INVERT>
    float getHardMask_(float distance_xy_sqd, float radius_xy_sqd, float distance_z, float radius_z) {
        float mask;
        if constexpr (INVERT) {
            if (distance_z > radius_z || distance_xy_sqd > radius_xy_sqd)
                mask = 1.f;
            else
                mask = 0.f;
        } else {
            if (distance_z > radius_z || distance_xy_sqd > radius_xy_sqd)
                mask = 0.f;
            else
                mask = 1.f;
        }
        return mask;
    }

    template<bool TAPER, bool INVERT, typename T>
    void cylinderOMP_(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                      size_t batches, float3_t center, float radius_xy, float radius_z, float taper_size,
                      size_t threads) {
        using real_t = traits::value_type_t<T>;
        const float radius_xy_sqd = radius_xy * radius_xy;
        [[maybe_unused]] const float radius_xy_taper_sqd = math::pow(radius_xy + taper_size, 2.f);
        [[maybe_unused]] const float radius_z_taper = radius_z + taper_size;

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(inputs, input_pitch, outputs, output_pitch, shape, batches, center, radius_xy, radius_z, taper_size, \
               radius_xy_sqd, radius_xy_taper_sqd, radius_z_taper)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = 0; x < shape.x; ++x) {

                        const T* input = inputs + batch * elements(input_pitch);
                        T* output = outputs + batch * elements(output_pitch);
                        output += index(x, y, z, output_pitch);

                        float2_t dst(x, y);
                        dst -= {center.x, center.y};
                        dst *= dst;
                        const float dst_sd_xy = math::sum(dst);
                        const float dst_z = math::abs(static_cast<float>(z) - center.z);

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(dst_sd_xy, radius_xy_sqd, radius_xy, radius_xy_taper_sqd,
                                                        dst_z, radius_z, radius_z_taper, taper_size);
                        else
                            mask = getHardMask_<INVERT>(dst_sd_xy, radius_xy_sqd, dst_z, radius_z);

                        if (inputs)
                            *output = input[index(x, y, z, input_pitch)] * static_cast<real_t>(mask);
                        else
                            *output = static_cast<real_t>(mask);
                    }
                }
            }
        }
    }

    template<bool TAPER, bool INVERT, typename T>
    void cylinder_(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                   size_t batches, float3_t center, float radius_xy, float radius_z, float taper_size) {
        using real_t = traits::value_type_t<T>;
        const float radius_xy_sqd = radius_xy * radius_xy;
        [[maybe_unused]] const float radius_xy_taper_sqd = math::pow(radius_xy + taper_size, 2.f);
        [[maybe_unused]] const float radius_z_taper = radius_z + taper_size;

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + batch * elements(input_pitch);
            T* output = outputs + batch * elements(output_pitch);

            for (size_t z = 0; z < shape.z; ++z) {
                const float distance_z = math::abs(static_cast<float>(z) - center.z);
                for (size_t y = 0; y < shape.y; ++y) {
                    const float distance_y_sqd = math::pow(static_cast<float>(y) - center.y, 2.f);
                    const size_t iffset = index(y, z, input_pitch);
                    const size_t offset = index(y, z, output_pitch);
                    for (size_t x = 0; x < shape.x; ++x) {
                        float distance_xy_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                        distance_xy_sqd += distance_y_sqd;

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(distance_xy_sqd, radius_xy_sqd, radius_xy, radius_xy_taper_sqd,
                                                        distance_z, radius_z, radius_z_taper, taper_size);
                        else
                            mask = getHardMask_<INVERT>(distance_xy_sqd, radius_xy_sqd, distance_z, radius_z);

                        if (inputs)
                            output[offset + x] = input[iffset + x] * static_cast<real_t>(mask);
                        else
                            output[offset + x] = static_cast<real_t>(mask);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::filter {
    template<bool INVERT, typename T>
    void cylinder(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                  size_t batches, float3_t center, float radius_xy, float radius_z, float taper_size,
                  Stream& stream) {
        NOA_PROFILE_FUNCTION();

        // Instead of computing the mask for each batch, compute once and then copy to the other batches.
        if (batches > 1 && !inputs) {
            cylinder<INVERT>(inputs, input_pitch, outputs, output_pitch, shape, 1,
                             center, radius_xy, radius_z, taper_size, stream);
            for (size_t batch = 1; batch < batches; ++batch)
                memory::copy(outputs, output_pitch,
                             outputs + batch * elements(output_pitch), output_pitch,
                             shape, 1, stream);
            return;
        }

        const size_t threads = stream.threads();
        const bool taper = taper_size > 1e-5f;
        if (threads > 1)
            stream.enqueue(taper ? cylinderOMP_<true, INVERT, T> : cylinderOMP_<false, INVERT, T>,
                           inputs, input_pitch, outputs, output_pitch, shape, batches,
                           center, radius_xy, radius_z, taper_size, threads);
        else
            stream.enqueue(taper ? cylinder_<true, INVERT, T> : cylinder_<false, INVERT, T>,
                           inputs, input_pitch, outputs, output_pitch, shape, batches,
                           center, radius_xy, radius_z, taper_size);
    }

    #define NOA_INSTANTIATE_CYLINDER_(T)                                                                                        \
    template void cylinder<true, T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float3_t, float, float, float, Stream&);   \
    template void cylinder<false, T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float3_t, float, float, float, Stream&)

    NOA_INSTANTIATE_CYLINDER_(half_t);
    NOA_INSTANTIATE_CYLINDER_(float);
    NOA_INSTANTIATE_CYLINDER_(double);
    NOA_INSTANTIATE_CYLINDER_(chalf_t);
    NOA_INSTANTIATE_CYLINDER_(cfloat_t);
    NOA_INSTANTIATE_CYLINDER_(cdouble_t);
}
