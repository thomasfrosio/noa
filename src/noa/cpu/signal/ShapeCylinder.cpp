#include "noa/common/Math.h"
#include "noa/cpu/signal/Shape.h"

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
    void cylinderOMP_(shared_t<T[]> input, dim4_t input_strides,
                      shared_t<T[]> output, dim4_t output_strides,
                      dim3_t start, dim3_t end, dim_t batches,
                      float3_t center, float radius, float length, float taper_size, dim_t threads) {

        const Accessor<const T, 4, dim_t> iptr(input.get(), input_strides);
        const Accessor<T, 4, dim_t> optr(output.get(), output_strides);

        using real_t = traits::value_type_t<T>;
        const float radius_sqd = radius * radius;
        [[maybe_unused]] const float radius_taper_sqd = math::pow(radius + taper_size, 2.f);
        [[maybe_unused]] const float length_plus_taper = length + taper_size;

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(start, end, batches, center, length, radius, taper_size,         \
               iptr, optr, length_plus_taper, radius_sqd, radius_taper_sqd)

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t j = start[0]; j < end[0]; ++j) {
                for (dim_t k = start[1]; k < end[1]; ++k) {
                    for (dim_t l = start[2]; l < end[2]; ++l) {

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

                        const auto mask_ = static_cast<real_t>(mask);
                        optr(i, j, k, l) = iptr ? iptr(i, j, k, l) * mask_ : mask_;
                    }
                }
            }
        }
    }

    template<bool TAPER, bool INVERT, typename T>
    void cylinder_(shared_t<T[]> input, dim4_t input_strides,
                   shared_t<T[]> output, dim4_t output_strides,
                   dim3_t start, dim3_t end, dim_t batches,
                   float3_t center, float radius, float length, float taper_size) {

        const Accessor<const T, 4, dim_t> iptr(input.get(), input_strides);
        const Accessor<T, 4, dim_t> optr(output.get(), output_strides);

        using real_t = traits::value_type_t<T>;
        const float radius_sqd = radius * radius;
        [[maybe_unused]] const float radius_taper_sqd = math::pow(radius + taper_size, 2.f);
        [[maybe_unused]] const float length_taper = length + taper_size;

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t j = start[0]; j < end[0]; ++j) {
                for (dim_t k = start[1]; k < end[1]; ++k) {
                    for (dim_t l = start[2]; l < end[2]; ++l) {

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

                        const auto mask_ = static_cast<real_t>(mask);
                        optr(i, j, k, l) = iptr ? iptr(i, j, k, l) * mask_ : mask_;
                    }
                }
            }
        }
    }
}

namespace noa::cpu::signal {
    template<bool INVERT, typename T, typename>
    void cylinder(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  float3_t center, float radius, float length, float taper_size, Stream& stream) {
        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
            std::swap(center[1], center[2]);
        }

        dim3_t start{0}, end(shape.get(1));
        if (INVERT && input.get() == output.get()) {
            float3_t radius_{length, radius, radius};
            radius_ += taper_size;
            start = dim3_t(noa::math::clamp(int3_t(center - radius_), int3_t{}, int3_t(end)));
            end = dim3_t(noa::math::clamp(int3_t(center + radius_ + 1), int3_t{}, int3_t(end)));
            if (any(end <= start))
                return;
        }

        const dim_t threads = stream.threads();
        const bool taper = taper_size > 1e-5f;
        if (threads > 1)
            stream.enqueue(taper ? cylinderOMP_<true, INVERT, T> : cylinderOMP_<false, INVERT, T>,
                           input, input_strides, output, output_strides,
                           start, end, shape[0], center, radius, length, taper_size, threads);
        else
            stream.enqueue(taper ? cylinder_<true, INVERT, T> : cylinder_<false, INVERT, T>,
                           input, input_strides, output, output_strides,
                           start, end, shape[0], center, radius, length, taper_size);
    }

    #define NOA_INSTANTIATE_CYLINDER_(T)                                                                                                                \
    template void cylinder<true, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, float, Stream&);  \
    template void cylinder<false, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, float, Stream&)

    NOA_INSTANTIATE_CYLINDER_(half_t);
    NOA_INSTANTIATE_CYLINDER_(float);
    NOA_INSTANTIATE_CYLINDER_(double);
    NOA_INSTANTIATE_CYLINDER_(chalf_t);
    NOA_INSTANTIATE_CYLINDER_(cfloat_t);
    NOA_INSTANTIATE_CYLINDER_(cdouble_t);
}
