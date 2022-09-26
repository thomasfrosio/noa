#include "noa/common/Math.h"
#include "noa/cpu/signal/Shape.h"

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
                if (radius[0] < distance[0] && distance[0] <= radius_with_taper[0])
                    mask *= (1.f + math::cos(PI * (distance[0] - radius[0]) / taper_size)) * 0.5f;
                if (radius[1] < distance[1] && distance[1] <= radius_with_taper[1])
                    mask *= (1.f + math::cos(PI * (distance[1] - radius[1]) / taper_size)) * 0.5f;
                if (radius[2] < distance[2] && distance[2] <= radius_with_taper[2])
                    mask *= (1.f + math::cos(PI * (distance[2] - radius[2]) / taper_size)) * 0.5f;
                mask = 1.f - mask;
            }
        } else {
            if (any(radius_with_taper < distance)) {
                mask = 0.f;
            } else if (all(distance <= radius)) {
                mask = 1.f;
            } else {
                mask = 1.f;
                if (radius[0] < distance[0] && distance[0] <= radius_with_taper[0])
                    mask *= (1.f + math::cos(PI * (distance[0] - radius[0]) / taper_size)) * 0.5f;
                if (radius[1] < distance[1] && distance[1] <= radius_with_taper[1])
                    mask *= (1.f + math::cos(PI * (distance[1] - radius[1]) / taper_size)) * 0.5f;
                if (radius[2] < distance[2] && distance[2] <= radius_with_taper[2])
                    mask *= (1.f + math::cos(PI * (distance[2] - radius[2]) / taper_size)) * 0.5f;
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
    void rectangleOMP_(shared_t<T[]> input, dim4_t input_strides,
                       shared_t<T[]> output, dim4_t output_strides,
                       dim3_t start, dim3_t end, dim_t batches,
                       float3_t center, float3_t radius, float taper_size, dim_t threads) {

        const Accessor<const T, 4, dim_t> iptr(input.get(), input_strides);
        const Accessor<T, 4, dim_t> optr(output.get(), output_strides);

        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] const float3_t radius_with_taper = radius + taper_size;

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(iptr, optr, start, end, batches, center, radius, taper_size, radius_with_taper)

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t j = start[0]; j < end[0]; ++j) {
                for (dim_t k = start[1]; k < end[1]; ++k) {
                    for (dim_t l = start[2]; l < end[2]; ++l) {

                        float3_t distance{j, k, l};
                        distance -= center;
                        distance = math::abs(distance);

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(distance, radius, radius_with_taper, taper_size);
                        else
                            mask = getHardMask_<INVERT>(distance, radius);

                        const auto mask_ = static_cast<real_t>(mask);
                        optr(i, j, k, l) = iptr ? iptr(i, j, k, l) * mask_ : mask_;
                    }
                }
            }
        }
    }

    template<bool TAPER, bool INVERT, typename T>
    void rectangle_(shared_t<T[]> input, dim4_t input_strides,
                    shared_t<T[]> output, dim4_t output_strides,
                    dim3_t start, dim3_t end, dim_t batches,
                    float3_t center, float3_t radius, float taper_size) {

        const Accessor<const T, 4, dim_t> iptr(input.get(), input_strides);
        const Accessor<T, 4, dim_t> optr(output.get(), output_strides);

        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] const float3_t radius_with_taper = radius + taper_size;

        for (dim_t i = 0; i < batches; ++i) {
            for (dim_t j = start[0]; j < end[0]; ++j) {
                for (dim_t k = start[1]; k < end[1]; ++k) {
                    for (dim_t l = start[2]; l < end[2]; ++l) {

                        const float3_t distance{math::abs(static_cast<float>(j) - center[0]),
                                                math::abs(static_cast<float>(k) - center[1]),
                                                math::abs(static_cast<float>(l) - center[2])};

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(distance, radius, radius_with_taper, taper_size);
                        else
                            mask = getHardMask_<INVERT>(distance, radius);

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
    void rectangle(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   float3_t center, float3_t radius, float taper_size, Stream& stream) {
        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1)));
        if (any(order_3d != dim3_t{0, 1, 2})) {
            const dim4_t order{0, order_3d[0] + 1, order_3d[1] + 1, order_3d[2] + 1};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
            center = indexing::reorder(center, order_3d);
            radius = indexing::reorder(radius, order_3d);
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
            stream.enqueue(taper ? rectangleOMP_<true, INVERT, T> : rectangleOMP_<false, INVERT, T>,
                           input, input_strides, output, output_strides,
                           start, end, shape[0], center, radius, taper_size, threads);
        else
            stream.enqueue(taper ? rectangle_<true, INVERT, T> : rectangle_<false, INVERT, T>,
                           input, input_strides, output, output_strides,
                           start, end, shape[0], center, radius, taper_size);
    }

    #define NOA_INSTANTIATE_RECTANGLE_(T)                                                                                                           \
    template void rectangle<true, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, Stream&); \
    template void rectangle<false, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_RECTANGLE_(half_t);
    NOA_INSTANTIATE_RECTANGLE_(float);
    NOA_INSTANTIATE_RECTANGLE_(double);
    NOA_INSTANTIATE_RECTANGLE_(chalf_t);
    NOA_INSTANTIATE_RECTANGLE_(cfloat_t);
    NOA_INSTANTIATE_RECTANGLE_(cdouble_t);
}
