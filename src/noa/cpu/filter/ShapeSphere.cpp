#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/filter/Shape.h"

namespace {
    using namespace noa;

    template<bool INVERT>
    static float getSoftMask_(float distance_sqd, float radius, float radius_sqd,
                              float radius_taper_sqd, float taper_size) {
        float mask;
        constexpr float PI = math::Constants<float>::PI;
        if constexpr (INVERT) {
            if (distance_sqd > radius_taper_sqd) {
                mask = 1.f;
            } else if (distance_sqd <= radius_sqd) {
                mask = 0.f;
            } else {
                float distance = math::sqrt(distance_sqd);
                mask = (1.f - math::cos(PI * (distance - radius) / taper_size)) * 0.5f;
            }
        } else {
            if (distance_sqd > radius_taper_sqd) {
                mask = 0.f;
            } else if (distance_sqd <= radius_sqd) {
                mask = 1.f;
            } else {
                distance_sqd = math::sqrt(distance_sqd);
                mask = (1.f + math::cos(PI * (distance_sqd - radius) / taper_size)) * 0.5f;
            }
        }
        return mask;
    }

    template<bool INVERT>
    static float getHardMask_(float distance_sqd, float radius_sqd) {
        float mask;
        if constexpr (INVERT) {
            if (distance_sqd > radius_sqd)
                mask = 1;
            else
                mask = 0;
        } else {
            if (distance_sqd > radius_sqd)
                mask = 0;
            else
                mask = 1;
        }
        return mask;
    }

    template<bool TAPER, bool INVERT, typename T>
    void sphereOMP_(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                    size3_t shape, size_t batches, float3_t center, float radius, float taper_size,
                    size_t threads) {
        using real_t = traits::value_type_t<T>;
        const float radius_sqd = radius * radius;
        [[maybe_unused]] float radius_taper_sqd = math::pow(radius + taper_size, 2.f);

        #pragma omp parallel for default(none) collapse(4) num_threads(threads) \
        shared(inputs, input_pitch, outputs, output_pitch, shape, batches, center, radius, taper_size, \
               radius_sqd, radius_taper_sqd)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = 0; x < shape.x; ++x) {

                        const T* input = inputs + batch * elements(input_pitch);
                        T* output = outputs + batch * elements(output_pitch);
                        output += index(x, y, z, output_pitch);

                        float3_t pos_sqd(x, y, z);
                        pos_sqd -= center;
                        pos_sqd *= pos_sqd;
                        const float dst_sqd = math::sum(pos_sqd);

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(dst_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
                        else
                            mask = getHardMask_<INVERT>(dst_sqd, radius_sqd);

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
    void sphere_(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                 size3_t shape, size_t batches, float3_t center, float radius, float taper_size) {
        using real_t = traits::value_type_t<T>;
        const float radius_sqd = radius * radius;
        [[maybe_unused]] float radius_taper_sqd = math::pow(radius + taper_size, 2.f);

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + batch * elements(input_pitch);
            T* output = outputs + batch * elements(output_pitch);

            for (size_t z = 0; z < shape.z; ++z) {
                const float dst_sqd_z = math::pow(static_cast<float>(z) - center.z, 2.f);

                for (size_t y = 0; y < shape.y; ++y) {
                    const float dst_sqd_y = math::pow(static_cast<float>(y) - center.y, 2.f);
                    const size_t iffset = index(y, z, input_pitch);
                    const size_t offset = index(y, z, output_pitch);

                    for (size_t x = 0; x < shape.x; ++x) {
                        float dst_sqd = math::pow(static_cast<float>(x) - center.x, 2.f);
                        dst_sqd += dst_sqd_y + dst_sqd_z;

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(dst_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
                        else
                            mask = getHardMask_<INVERT>(dst_sqd, radius_sqd);

                        if (inputs) // should be nicely predicted
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
    void sphere(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                size_t batches, float3_t center, float radius, float taper_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();

        // Instead of computing the mask for each batch, compute once and then copy to the other batches.
        if (batches > 1 && !inputs) {
            sphere<INVERT>(inputs, input_pitch, outputs, output_pitch, shape, 1, center, radius, taper_size, stream);
            for (size_t batch = 1; batch < batches; ++batch)
                memory::copy(outputs, output_pitch,
                             outputs + batch * elements(output_pitch), output_pitch,
                             shape, 1, stream);
            return;
        }

        const size_t threads = stream.threads();
        const bool taper = taper_size > 1e-5f;
        if (threads > 1)
            stream.enqueue(taper ? sphereOMP_<true, INVERT, T> : sphereOMP_<false, INVERT, T>,
                           inputs, input_pitch, outputs, output_pitch, shape, batches,
                           center, radius, taper_size, threads);
        else
            stream.enqueue(taper ? sphere_<true, INVERT, T> : sphere_<false, INVERT, T>,
                           inputs, input_pitch, outputs, output_pitch, shape, batches,
                           center, radius, taper_size);
    }

    #define NOA_INSTANTIATE_SPHERE_(T)                                                                                  \
    template void sphere<true, T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float3_t, float, float, Stream&);    \
    template void sphere<false, T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float3_t, float, float, Stream&)

    NOA_INSTANTIATE_SPHERE_(half_t);
    NOA_INSTANTIATE_SPHERE_(float);
    NOA_INSTANTIATE_SPHERE_(double);
    NOA_INSTANTIATE_SPHERE_(chalf_t);
    NOA_INSTANTIATE_SPHERE_(cfloat_t);
    NOA_INSTANTIATE_SPHERE_(cdouble_t);
}
