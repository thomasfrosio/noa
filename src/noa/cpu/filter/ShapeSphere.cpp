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
    void sphereOMP_(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                    size4_t shape, float3_t center, float radius, float taper_size, size_t threads) {
        using real_t = traits::value_type_t<T>;
        const float radius_sqd = radius * radius;
        [[maybe_unused]] float radius_taper_sqd = math::pow(radius + taper_size, 2.f);

        #pragma omp parallel for default(none) collapse(4) num_threads(threads) \
        shared(input, input_stride, output, output_stride, shape, center, radius, taper_size, \
               radius_sqd, radius_taper_sqd)

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {

                        float3_t pos_sqd{j, k, l};
                        pos_sqd -= center;
                        pos_sqd *= pos_sqd;
                        const float dst_sqd = math::sum(pos_sqd);

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(dst_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
                        else
                            mask = getHardMask_<INVERT>(dst_sqd, radius_sqd);

                        output[at(i, j, k, l, output_stride)] =
                                input ?
                                input[at(i, j, k, l, input_stride)] * static_cast<real_t>(mask) :
                                static_cast<real_t>(mask);
                    }
                }
            }
        }
    }

    template<bool TAPER, bool INVERT, typename T>
    void sphere_(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                 size4_t shape, float3_t center, float radius, float taper_size) {
        using real_t = traits::value_type_t<T>;
        const float radius_sqd = radius * radius;
        [[maybe_unused]] float radius_taper_sqd = math::pow(radius + taper_size, 2.f);

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {

                        const float dst_sqd_j = math::pow(static_cast<float>(j) - center[0], 2.f);
                        const float dst_sqd_k = math::pow(static_cast<float>(k) - center[1], 2.f);
                        const float dst_sqd_l = math::pow(static_cast<float>(l) - center[2], 2.f);
                        const float dst_sqd = dst_sqd_j + dst_sqd_k + dst_sqd_l;

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(dst_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
                        else
                            mask = getHardMask_<INVERT>(dst_sqd, radius_sqd);

                        output[at(i, j, k, l, output_stride)] =
                                input ?
                                input[at(i, j, k, l, input_stride)] * static_cast<real_t>(mask) :
                                static_cast<real_t>(mask);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::filter {
    template<bool INVERT, typename T>
    void sphere(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                float3_t center, float radius, float taper_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();

        // Instead of computing the mask for each batch, compute once and then copy to the other batches.
        if (shape[0] > 1 && !input) {
            const size4_t shape_3d{1, shape[1], shape[2], shape[3]};
            sphere<INVERT>(input, input_stride, output, output_stride, shape_3d, center, radius, taper_size, stream);
            memory::copy(output, {0, output_stride[1], output_stride[2], output_stride[3]},
                         output + output_stride[0], output_stride,
                         {shape[0] - 1, shape[1], shape[2], shape[3]}, stream);
            return;
        }

        const size_t threads = stream.threads();
        const bool taper = taper_size > 1e-5f;
        if (threads > 1)
            stream.enqueue(taper ? sphereOMP_<true, INVERT, T> : sphereOMP_<false, INVERT, T>,
                           input, input_stride, output, output_stride, shape,
                           center, radius, taper_size, threads);
        else
            stream.enqueue(taper ? sphere_<true, INVERT, T> : sphere_<false, INVERT, T>,
                           input, input_stride, output, output_stride, shape,
                           center, radius, taper_size);
    }

    #define NOA_INSTANTIATE_SPHERE_(T)                                                                      \
    template void sphere<true, T>(const T*, size4_t, T*, size4_t, size4_t, float3_t, float, float, Stream&);\
    template void sphere<false, T>(const T*, size4_t, T*, size4_t, size4_t, float3_t, float, float, Stream&)

    NOA_INSTANTIATE_SPHERE_(half_t);
    NOA_INSTANTIATE_SPHERE_(float);
    NOA_INSTANTIATE_SPHERE_(double);
    NOA_INSTANTIATE_SPHERE_(chalf_t);
    NOA_INSTANTIATE_SPHERE_(cfloat_t);
    NOA_INSTANTIATE_SPHERE_(cdouble_t);
}
