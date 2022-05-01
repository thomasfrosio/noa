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
    void rectangleOMP_(const shared_t<T[]> input, size4_t input_stride,
                       const shared_t<T[]> output, size4_t output_stride,
                       size3_t start, size3_t end, size_t batches,
                       float3_t center, float3_t radius, float taper_size, size_t threads) {
        const T* iptr = input.get();
        T* optr = output.get();
        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] const float3_t radius_with_taper = radius + taper_size;

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(iptr, input_stride, optr, output_stride, start, end, batches,    \
               center, radius, taper_size, radius_with_taper)

        for (size_t i = 0; i < batches; ++i) {
            for (size_t j = start[0]; j < end[0]; ++j) {
                for (size_t k = start[1]; k < end[1]; ++k) {
                    for (size_t l = start[2]; l < end[2]; ++l) {

                        float3_t distance(j, k, l);
                        distance -= center;
                        distance = math::abs(distance);

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(distance, radius, radius_with_taper, taper_size);
                        else
                            mask = getHardMask_<INVERT>(distance, radius);

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
    void rectangle_(const shared_t<T[]> input, size4_t input_stride,
                    const shared_t<T[]> output, size4_t output_stride,
                    size3_t start, size3_t end, size_t batches,
                    float3_t center, float3_t radius, float taper_size) {
        const T* iptr = input.get();
        T* optr = output.get();
        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] const float3_t radius_with_taper = radius + taper_size;

        for (size_t i = 0; i < batches; ++i) {
            for (size_t j = start[0]; j < end[0]; ++j) {
                for (size_t k = start[1]; k < end[1]; ++k) {
                    for (size_t l = start[2]; l < end[2]; ++l) {

                        const float3_t distance{math::abs(static_cast<float>(j) - center[0]),
                                                math::abs(static_cast<float>(k) - center[1]),
                                                math::abs(static_cast<float>(l) - center[2])};

                        float mask;
                        if constexpr (TAPER)
                            mask = getSoftMask_<INVERT>(distance, radius, radius_with_taper, taper_size);
                        else
                            mask = getHardMask_<INVERT>(distance, radius);

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

namespace noa::cpu::signal {
    template<bool INVERT, typename T, typename>
    void rectangle(const shared_t<T[]>& input, size4_t input_stride,
                   const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                   float3_t center, float3_t radius, float taper_size, Stream& stream) {
        size3_t start{0}, end{shape.get() + 1};
        if (INVERT && input.get() == output.get()) {
            start = size3_t{noa::math::clamp(int3_t{center - (radius + taper_size)}, int3_t{}, int3_t{end})};
            end = size3_t{noa::math::clamp(int3_t{center + (radius + taper_size) + 1}, int3_t{}, int3_t{end})};
            if (any(end <= start))
                return;
        }

        const size_t threads = stream.threads();
        const bool taper = taper_size > 1e-5f;
        if (threads > 1)
            stream.enqueue(taper ? rectangleOMP_<true, INVERT, T> : rectangleOMP_<false, INVERT, T>,
                           input, input_stride, output, output_stride,
                           start, end, shape[0], center, radius, taper_size, threads);
        else
            stream.enqueue(taper ? rectangle_<true, INVERT, T> : rectangle_<false, INVERT, T>,
                           input, input_stride, output, output_stride,
                           start, end, shape[0], center, radius, taper_size);
    }

    #define NOA_INSTANTIATE_RECTANGLE_(T)                                                                                                               \
    template void rectangle<true, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float3_t, float, Stream&);  \
    template void rectangle<false, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_RECTANGLE_(half_t);
    NOA_INSTANTIATE_RECTANGLE_(float);
    NOA_INSTANTIATE_RECTANGLE_(double);
    NOA_INSTANTIATE_RECTANGLE_(chalf_t);
    NOA_INSTANTIATE_RECTANGLE_(cfloat_t);
    NOA_INSTANTIATE_RECTANGLE_(cdouble_t);
}
