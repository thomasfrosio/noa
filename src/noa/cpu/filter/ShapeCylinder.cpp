#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/filter/Shape.h"

// Soft edges:
namespace {
    using namespace noa;

    template<bool INVERT>
    float getSoftMask_(float distance_kl_sqd, float radius_sqd, float radius, float radius_sqd_with_taper,
                       float distance_j, float length, float length_with_taper, float taper_size) {
        constexpr float PI = math::Constants<float>::PI;
        float mask;
        if constexpr (INVERT) {
            if (distance_j > length_with_taper || distance_kl_sqd > radius_sqd_with_taper) {
                mask = 1.f;
            } else {
                if (distance_kl_sqd <= radius_sqd) {
                    mask = 1.f;
                } else {
                    float distance_kl = math::sqrt(distance_kl_sqd);
                    mask = (1.f + math::cos(PI * (distance_kl - radius) / taper_size)) * 0.5f;
                }
                if (distance_j > length)
                    mask *= (1.f + math::cos(PI * (distance_j - length) / taper_size)) * 0.5f;
                mask = 1 - mask;
            }
        } else {
            if (distance_j > length_with_taper || distance_kl_sqd > radius_sqd_with_taper) {
                mask = 0.f;
            } else {
                if (distance_kl_sqd <= radius_sqd) {
                    mask = 1.f;
                } else {
                    float distance_kl = math::sqrt(distance_kl_sqd);
                    mask = (1.f + math::cos(PI * (distance_kl - radius) / taper_size)) * 0.5f;
                }
                if (distance_j > length)
                    mask *= (1.f + math::cos(PI * (distance_j - length) / taper_size)) * 0.5f;
            }
        }
        return mask;
    }

    template<bool INVERT>
    float getHardMask_(float distance_kl_sqd, float radius_sqd, float distance_j, float length) {
        float mask;
        if constexpr (INVERT) {
            if (distance_j > length || distance_kl_sqd > radius_sqd)
                mask = 1.f;
            else
                mask = 0.f;
        } else {
            if (distance_j > length || distance_kl_sqd > radius_sqd)
                mask = 0.f;
            else
                mask = 1.f;
        }
        return mask;
    }

    template<bool TAPER, bool INVERT, typename T>
    void cylinderOMP_(const shared_t<const T[]> input, size4_t input_stride,
                      const shared_t<T[]> output, size4_t output_stride, size4_t shape,
                      float3_t center, float radius, float length, float taper_size, size_t threads) {
        NOA_PROFILE_FUNCTION();
        const T* iptr = input.get();
        T* optr = output.get();
        using real_t = traits::value_type_t<T>;
        const float radius_sqd = radius * radius;
        [[maybe_unused]] const float radius_taper_sqd = math::pow(radius + taper_size, 2.f);
        [[maybe_unused]] const float length_plus_taper = length + taper_size;

        #pragma omp parallel for collapse(4) default(none) num_threads(threads)                     \
        shared(iptr, input_stride, optr, output_stride, shape, center, length, radius, taper_size,  \
               length_plus_taper, radius_sqd, radius_taper_sqd)

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {

                        float2_t dst{k, l};
                        dst -= {center[1], center[2]};
                        dst *= dst;
                        const float dst_j = math::abs(static_cast<float>(j) - center[0]);
                        const float dst_kl_sqd = math::sum(dst);

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(dst_kl_sqd, radius_sqd, radius, radius_taper_sqd,
                                                        dst_j, length, length_plus_taper, taper_size);
                        else
                            mask = getHardMask_<INVERT>(dst_kl_sqd, radius_sqd, dst_j, length);

                        optr[indexing::at(i, j, k, l, output_stride)] =
                                iptr ?
                                iptr[indexing::at(i, j, k, l, input_stride)] * static_cast<real_t>(mask) :
                                static_cast<real_t>(mask);
                    }
                }
            }
        }
    }

    template<bool TAPER, bool INVERT, typename T>
    void cylinder_(const shared_t<const T[]> input, size4_t input_stride,
                   const shared_t<T[]> output, size4_t output_stride, size4_t shape,
                   float3_t center, float radius, float length, float taper_size) {
        NOA_PROFILE_FUNCTION();
        const T* iptr = input.get();
        T* optr = output.get();
        using real_t = traits::value_type_t<T>;
        const float radius_sqd = radius * radius;
        [[maybe_unused]] const float radius_taper_sqd = math::pow(radius + taper_size, 2.f);
        [[maybe_unused]] const float length_taper = length + taper_size;

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {

                        const float dst_j = math::abs(static_cast<float>(j) - center[0]);
                        const float dst_k_sqd = math::pow(static_cast<float>(k) - center[1], 2.f);
                        const float dst_l_sqd = math::pow(static_cast<float>(l) - center[2], 2.f);
                        const float dst_kl_sqd = dst_k_sqd + dst_l_sqd;

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(dst_kl_sqd, radius_sqd, radius, radius_taper_sqd,
                                                        dst_j, length, length_taper, taper_size);
                        else
                            mask = getHardMask_<INVERT>(dst_kl_sqd, radius_sqd, dst_j, length);

                        optr[indexing::at(i, j, k, l, output_stride)] =
                                iptr ?
                                iptr[indexing::at(i, j, k, l, input_stride)] * static_cast<real_t>(mask) :
                                static_cast<real_t>(mask);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::filter {
    template<bool INVERT, typename T>
    void cylinder(const shared_t<const T[]>& input, size4_t input_stride,
                  const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                  float3_t center, float radius, float length, float taper_size, Stream& stream) {
        const size_t threads = stream.threads();
        const bool taper = taper_size > 1e-5f;
        if (threads > 1)
            stream.enqueue(taper ? cylinderOMP_<true, INVERT, T> : cylinderOMP_<false, INVERT, T>,
                           input, input_stride, output, output_stride, shape,
                           center, radius, length, taper_size, threads);
        else
            stream.enqueue(taper ? cylinder_<true, INVERT, T> : cylinder_<false, INVERT, T>,
                           input, input_stride, output, output_stride, shape,
                           center, radius, length, taper_size);
    }

    #define NOA_INSTANTIATE_CYLINDER_(T)                                                                                                                    \
    template void cylinder<true, T>(const shared_t<const T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float, float, float, Stream&);   \
    template void cylinder<false, T>(const shared_t<const T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float, float, float, Stream&)

    NOA_INSTANTIATE_CYLINDER_(half_t);
    NOA_INSTANTIATE_CYLINDER_(float);
    NOA_INSTANTIATE_CYLINDER_(double);
    NOA_INSTANTIATE_CYLINDER_(chalf_t);
    NOA_INSTANTIATE_CYLINDER_(cfloat_t);
    NOA_INSTANTIATE_CYLINDER_(cdouble_t);
}
