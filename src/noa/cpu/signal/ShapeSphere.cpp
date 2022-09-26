#include "noa/common/Math.h"
#include "noa/cpu/signal/Shape.h"

namespace {
    using namespace noa;

    template<bool INVERT>
    float getSoftMask_(float distance_sqd, float radius, float radius_sqd,
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
    float getHardMask_(float distance_sqd, float radius_sqd) {
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
    void sphereOMP_(const shared_t<T[]> input, dim4_t input_strides,
                    const shared_t<T[]> output, dim4_t output_strides,
                    dim3_t start, dim3_t end, dim_t batches,
                    float3_t center, float radius, float taper_size, dim_t threads) {

        const Accessor<const T, 4, dim_t> iptr(input.get(), input_strides);
        const Accessor<T, 4, dim_t> optr(output.get(), output_strides);

        using real_t = traits::value_type_t<T>;
        const float radius_sqd = radius * radius;
        [[maybe_unused]] float radius_taper_sqd = math::pow(radius + taper_size, 2.f);

        #pragma omp parallel for default(none) collapse(4) num_threads(threads) \
        shared(iptr, optr, start, end, batches, center, radius, taper_size, radius_sqd, radius_taper_sqd)

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t j = start[0]; j < end[0]; ++j) {
                for (dim_t k = start[1]; k < end[1]; ++k) {
                    for (dim_t l = start[2]; l < end[2]; ++l) {

                        float3_t pos_sqd{j, k, l};
                        pos_sqd -= center;
                        pos_sqd *= pos_sqd;
                        const float dst_sqd = math::sum(pos_sqd);

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(dst_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
                        else
                            mask = getHardMask_<INVERT>(dst_sqd, radius_sqd);

                        const auto mask_ = static_cast<real_t>(mask);
                        optr(i, j, k, l) = iptr ? iptr(i, j, k, l) * mask_ : mask_;
                    }
                }
            }
        }
    }

    template<bool TAPER, bool INVERT, typename T>
    void sphere_(const shared_t<T[]> input, dim4_t input_strides,
                 const shared_t<T[]> output, dim4_t output_strides,
                 dim3_t start, dim3_t end, dim_t batches,
                 float3_t center, float radius, float taper_size) {

        const Accessor<const T, 4, dim_t> iptr(input.get(), input_strides);
        const Accessor<T, 4, dim_t> optr(output.get(), output_strides);

        using real_t = traits::value_type_t<T>;
        const float radius_sqd = radius * radius;
        [[maybe_unused]] float radius_taper_sqd = math::pow(radius + taper_size, 2.f);

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t j = start[0]; j < end[0]; ++j) {
                for (dim_t k = start[1]; k < end[1]; ++k) {
                    for (dim_t l = start[2]; l < end[2]; ++l) {

                        const float dst_sqd_j = math::pow(static_cast<float>(j) - center[0], 2.f);
                        const float dst_sqd_k = math::pow(static_cast<float>(k) - center[1], 2.f);
                        const float dst_sqd_l = math::pow(static_cast<float>(l) - center[2], 2.f);
                        const float dst_sqd = dst_sqd_j + dst_sqd_k + dst_sqd_l;

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(dst_sqd, radius, radius_sqd, radius_taper_sqd, taper_size);
                        else
                            mask = getHardMask_<INVERT>(dst_sqd, radius_sqd);

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
    void sphere(const shared_t<T[]>& input, dim4_t input_strides,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                float3_t center, float radius, float taper_size, Stream& stream) {
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
        }

        dim3_t start{0}, end(shape.get(1));
        if (INVERT && input.get() == output.get()) {
            start = dim3_t(noa::math::clamp(int3_t(center - (radius + taper_size)), int3_t{}, int3_t(end)));
            end = dim3_t(noa::math::clamp(int3_t(center + (radius + taper_size) + 1), int3_t{}, int3_t(end)));
            if (any(end <= start))
                return;
        }

        const dim_t threads = stream.threads();
        const bool taper = taper_size > 1e-5f;
        if (threads > 1)
            stream.enqueue(taper ? sphereOMP_<true, INVERT, T> : sphereOMP_<false, INVERT, T>,
                           input, input_strides, output, output_strides,
                           start, end, shape[0], center, radius, taper_size, threads);
        else
            stream.enqueue(taper ? sphere_<true, INVERT, T> : sphere_<false, INVERT, T>,
                           input, input_strides, output, output_strides,
                           start, end, shape[0], center, radius, taper_size);
    }

    #define NOA_INSTANTIATE_SPHERE_(T)                                                                                                          \
    template void sphere<true, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, Stream&);   \
    template void sphere<false, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, float, Stream&)

    NOA_INSTANTIATE_SPHERE_(half_t);
    NOA_INSTANTIATE_SPHERE_(float);
    NOA_INSTANTIATE_SPHERE_(double);
    NOA_INSTANTIATE_SPHERE_(chalf_t);
    NOA_INSTANTIATE_SPHERE_(cfloat_t);
    NOA_INSTANTIATE_SPHERE_(cdouble_t);
}
