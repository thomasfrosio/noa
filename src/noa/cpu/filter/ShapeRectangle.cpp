#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/filter/Shape.h"

// Soft edges:
namespace {
    using namespace noa;

    template<bool INVERT>
    float getSoftMask_(float3_t distance, float3_t radius, float3_t radius_with_taper, float taper_size) {
        constexpr float PI = math::Constants<float>::PI;
        float mask;
        if constexpr (INVERT) {
            if (any(radius_with_taper < distance)) {
                mask = 1.f;
            } else if (all(distance <= radius)) {
                mask = 0.f;
            } else {
                mask = 1.f;
                if (radius.x < distance.x && distance.x <= radius_with_taper.x)
                    mask *= (1.f + math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                if (radius.y < distance.y && distance.y <= radius_with_taper.y)
                    mask *= (1.f + math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
                if (radius.z < distance.z && distance.z <= radius_with_taper.z)
                    mask *= (1.f + math::cos(PI * (distance.z - radius.z) / taper_size)) * 0.5f;
                mask = 1.f - mask;
            }
        } else {
            if (any(radius_with_taper < distance)) {
                mask = 0.f;
            } else if (all(distance <= radius)) {
                mask = 1.f;
            } else {
                mask = 1.f;
                if (radius.x < distance.x && distance.x <= radius_with_taper.x)
                    mask *= (1.f + math::cos(PI * (distance.x - radius.x) / taper_size)) * 0.5f;
                if (radius.y < distance.y && distance.y <= radius_with_taper.y)
                    mask *= (1.f + math::cos(PI * (distance.y - radius.y) / taper_size)) * 0.5f;
                if (radius.z < distance.z && distance.z <= radius_with_taper.z)
                    mask *= (1.f + math::cos(PI * (distance.z - radius.z) / taper_size)) * 0.5f;
            }
        }
        return mask;
    }

    template<bool INVERT>
    float getHardMask_(float3_t distance, float3_t radius) {
        float mask;
        if constexpr (INVERT) {
            if (all(distance <= radius))
                mask = 0;
            else
                mask = 1;
        } else {
            if (all(distance <= radius))
                mask = 1;
            else
                mask = 0;
        }
        return mask;
    }

    template<bool TAPER, bool INVERT, typename T>
    void rectangleOMP_(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                         size_t batches, float3_t center, float3_t radius, float taper_size, size_t threads) {
        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] const float3_t radius_with_taper = radius + taper_size;

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(inputs, input_pitch, outputs, output_pitch, shape, batches, center, radius, taper_size, radius_with_taper)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = 0; x < shape.x; ++x) {

                        const T* input = inputs + batch * elements(input_pitch);
                        T* output = outputs + batch * elements(output_pitch);
                        output += index(x, y, z, output_pitch);

                        float3_t distance(x, y, z);
                        distance -= center;
                        distance = math::abs(distance);

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(distance, radius, radius_with_taper, taper_size);
                        else
                            mask = getHardMask_<INVERT>(distance, radius);

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
    void rectangle_(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                      size_t batches, float3_t center, float3_t radius, float taper_size) {
        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] const float3_t radius_with_taper = radius + taper_size;

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + batch * elements(input_pitch);
            T* output = outputs + batch * elements(output_pitch);

            float3_t distance;
            for (size_t z = 0; z < shape.z; ++z) {
                distance.z = math::abs(static_cast<float>(z) - center.z);
                for (size_t y = 0; y < shape.y; ++y) {
                    distance.y = math::abs(static_cast<float>(y) - center.y);
                    const size_t iffset = index(y, z, input_pitch);
                    const size_t offset = index(y, z, output_pitch);
                    for (size_t x = 0; x < shape.x; ++x) {
                        distance.x = math::abs(static_cast<float>(x) - center.x);

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(distance, radius, radius_with_taper, taper_size);
                        else
                            mask = getHardMask_<INVERT>(distance, radius);

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
    void rectangle(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                   size_t batches, float3_t center, float3_t radius, float taper_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();

        // Instead of computing the mask for each batch, compute once and then copy to the other batches.
        if (batches > 1 && !inputs) {
            rectangle<INVERT>(inputs, input_pitch, outputs, output_pitch, shape, 1, center, radius, taper_size, stream);
            for (size_t batch = 1; batch < batches; ++batch)
                memory::copy(outputs, output_pitch,
                             outputs + batch * elements(output_pitch), output_pitch,
                             shape, 1, stream);
            return;
        }

        const size_t threads = stream.threads();
        const bool taper = taper_size > 1e-5f;
        if (threads > 1)
            stream.enqueue(taper ? rectangleOMP_<true, INVERT, T> : rectangleOMP_<false, INVERT, T>,
                           inputs, input_pitch, outputs, output_pitch, shape, batches,
                           center, radius, taper_size, threads);
        else
            stream.enqueue(taper ? rectangle_<true, INVERT, T> : rectangle_<false, INVERT, T>,
                           inputs, input_pitch, outputs, output_pitch, shape, batches,
                           center, radius, taper_size);
    }

    #define NOA_INSTANTIATE_RECTANGLE_(T)                                                                                   \
    template void rectangle<true, T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float3_t, float3_t, float, Stream&);  \
    template void rectangle<false, T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float3_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_RECTANGLE_(half_t);
    NOA_INSTANTIATE_RECTANGLE_(float);
    NOA_INSTANTIATE_RECTANGLE_(double);
    NOA_INSTANTIATE_RECTANGLE_(chalf_t);
    NOA_INSTANTIATE_RECTANGLE_(cfloat_t);
    NOA_INSTANTIATE_RECTANGLE_(cdouble_t);
}
