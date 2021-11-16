#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/math/Arithmetics.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/transform/fft/Apply.h"
#include "noa/cpu/transform/fft/Shift.h"
#include "noa/cpu/transform/fft/Symmetry.h"

namespace {
    using namespace ::noa;

    template<bool IS_DST_CENTERED>
    inline int64_t getFrequency_(int64_t idx, int64_t dim, int64_t dim_half) {
        if constexpr(IS_DST_CENTERED)
            return idx - dim_half;
        else
            return idx < (dim + 1) / 2 ? idx : idx - dim;
    }

    template<typename T>
    inline cfloat_t getPhaseShift_(T shift, T freq) {
        static_assert(traits::is_float2_v<T> || traits::is_float3_v<T>);
        const float factor = -math::dot(shift, freq);
        cfloat_t phase_shift;
        math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    // 2D, centered input.
    template<bool IS_DST_CENTERED, bool APPLY_SHIFT, InterpMode INTERP, typename T>
    void applyCenteredNormalized_(const T* input, T* output, size2_t shape,
                                  float22_t transform,
                                  [[maybe_unused]] float2_t shift,
                                  float max_frequency) {
        long2_t l_shape(shape);
        float2_t f_shape(l_shape.x > 1 ? l_shape.x / 2 * 2 : 1,
                         l_shape.y > 1 ? l_shape.y / 2 * 2 : 1); // if odd, n-1

        if constexpr (APPLY_SHIFT)
            shift *= math::Constants<float>::PI2 / float2_t(l_shape);

        max_frequency = noa::math::clamp(max_frequency, 0.f, 0.5f);
        max_frequency *= max_frequency;

        const size_t half_x = shape.x / 2 + 1;
        const auto l_half_x = static_cast<int64_t>(half_x);
        cpu::transform::Interpolator2D<T> interp(input, {half_x, shape.y}, half_x, 0);

        for (int64_t y = 0; y < l_shape.y; ++y) {
            int64_t v = getFrequency_<IS_DST_CENTERED>(y, l_shape.y, l_shape.y / 2);

            for (int64_t x = 0; x < l_half_x; ++x, ++output) {
                float2_t coordinates = float2_t(x, v) / f_shape; // [-0.5, 0.5]
                if (math::dot(coordinates, coordinates) > max_frequency)
                    continue;

                coordinates = transform * coordinates;
                using real_t = traits::value_type_t<T>;
                [[maybe_unused]] real_t conj = 1;
                if (coordinates.x < 0.f) {
                    coordinates = -coordinates;
                    if constexpr (traits::is_complex_v<T>)
                        conj = -1;
                }
                coordinates.y += 0.5f; // [0, 1]
                coordinates *= f_shape; // [0, N-1]
                T value = interp.template get<INTERP, BORDER_ZERO>(coordinates);
                if constexpr (traits::is_complex_v<T>)
                    value.imag *= conj;
                if constexpr (traits::is_complex_v<T> && APPLY_SHIFT)
                    value *= getPhaseShift_(shift, float2_t(x, v));
                *output = value;
            }
        }
    }

    // 3D, centered input.
    template<bool IS_DST_CENTERED, bool APPLY_SHIFT, InterpMode INTERP, typename T>
    void applyCenteredNormalized_(const T* input, T* output, size3_t shape,
                                  float33_t transform,
                                  [[maybe_unused]] float3_t shift,
                                  float max_frequency) {
        long3_t l_shape(shape);
        float3_t f_shape(l_shape.x / 2 * 2,
                         l_shape.y / 2 * 2,
                         l_shape.z / 2 * 2);

        if constexpr (APPLY_SHIFT)
            shift *= math::Constants<float>::PI2 / float3_t(l_shape);

        max_frequency = noa::math::clamp(max_frequency, 0.f, 0.5f);
        max_frequency *= max_frequency;

        const size_t half_x = shape.x / 2 + 1;
        const auto l_half_x = static_cast<int64_t>(half_x);
        cpu::transform::Interpolator3D<T> interp(input, {half_x, shape.y, shape.z}, half_x, 0);

        for (int64_t z = 0; z < l_shape.z; ++z) {
            int64_t w = getFrequency_<IS_DST_CENTERED>(z, l_shape.z, l_shape.z / 2);

            for (int64_t y = 0; y < l_shape.y; ++y) {
                int64_t v = getFrequency_<IS_DST_CENTERED>(y, l_shape.y, l_shape.y / 2);

                for (int64_t x = 0; x < l_half_x; ++x, ++output) {
                    float3_t coordinates = float3_t(x, v, w) / f_shape;
                    if (math::dot(coordinates, coordinates) > max_frequency)
                        continue;

                    coordinates = transform * coordinates;
                    using real_t = traits::value_type_t<T>;
                    [[maybe_unused]] real_t conj = 1;
                    if (coordinates.x < 0.f) {
                        coordinates = -coordinates;
                        if constexpr (traits::is_complex_v<T>)
                            conj = -1;
                    }
                    coordinates.y += 0.5f;
                    coordinates.z += 0.5f;
                    coordinates *= f_shape;
                    T value = interp.template get<INTERP, BORDER_ZERO>(coordinates);
                    if constexpr (traits::is_complex_v<T>)
                        value.imag *= conj;
                    if constexpr (traits::is_complex_v<T> && APPLY_SHIFT)
                        value *= getPhaseShift_(shift, float3_t(x, v, w));
                    *output = value;
                }
            }
        }
    }

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT,
            typename T, typename SHAPE, typename SHIFT, typename ROT>
    inline void launchCentered_(const T* input, T* output, SHAPE shape,
                                ROT transform, SHIFT shift,
                                float max_frequency, InterpMode interp_mode) {
        switch (interp_mode) {
            case InterpMode::INTERP_NEAREST:
                applyCenteredNormalized_<IS_DST_CENTERED, APPLY_SHIFT, InterpMode::INTERP_NEAREST>(
                        input, output, shape, transform, shift, max_frequency);
                break;
            case InterpMode::INTERP_LINEAR:
                applyCenteredNormalized_<IS_DST_CENTERED, APPLY_SHIFT, InterpMode::INTERP_LINEAR>(
                        input, output, shape, transform, shift, max_frequency);
                break;
            case InterpMode::INTERP_COSINE:
                applyCenteredNormalized_<IS_DST_CENTERED, APPLY_SHIFT, InterpMode::INTERP_COSINE>(
                        input, output, shape, transform, shift, max_frequency);
                break;
            default:
                NOA_THROW_FUNC("apply(2|3)D", "{} is not supported", interp_mode);
        }
    }

    // Atm, input FFTs should be centered. The only flexibility is whether the output should be centered or not.
    template<fft::Remap REMAP, typename T = void>
    constexpr bool parseRemap_() noexcept {
        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (!IS_SRC_CENTERED || REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        return IS_DST_CENTERED;
    }
}

namespace noa::cpu::transform::fft::details {
    template<Remap REMAP, typename T, typename SHAPE, typename SHIFT, typename ROT>
    void applyND(const T* input, T* outputs, SHAPE shape,
                 const ROT* transforms, const SHIFT* shifts, size_t nb_transforms,
                 float max_frequency, InterpMode interp_mode) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != outputs);
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        for (size_t i = 0; i < nb_transforms; ++i) {
            T* output = outputs + elementsFFT(shape) * i;
            if (shifts && any(shifts[i] != 0.f))
                launchCentered_<IS_DST_CENTERED, true>(
                        input, output, shape, transforms[i], shifts[i], max_frequency, interp_mode);
            else
                launchCentered_<IS_DST_CENTERED, false>(
                        input, output, shape, transforms[i], SHIFT{}, max_frequency, interp_mode);
        }
    }

    template<Remap REMAP, typename T, typename SHAPE, typename SHIFT, typename ROT>
    void applyND(const T* input, T* outputs, SHAPE shape,
                 const ROT* transforms, SHIFT shift, size_t nb_transforms,
                 float max_frequency, InterpMode interp_mode) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != outputs);
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        for (size_t i = 0; i < nb_transforms; ++i) {
            T* output = outputs + elementsFFT(shape) * i;
            if (all(shift == 0.f))
                launchCentered_<IS_DST_CENTERED, true>(
                        input, output, shape, transforms[i], shift, max_frequency, interp_mode);
            else
                launchCentered_<IS_DST_CENTERED, false>(
                        input, output, shape, transforms[i], SHIFT{}, max_frequency, interp_mode);
        }
    }

    #define NOA_INSTANTIATE_APPLYND_(T)                                                                                                                        \
    template void applyND<Remap::HC2H, T, size2_t, float2_t, float22_t>(const T*, T*, size2_t, const float22_t*, const float2_t*, size_t, float, InterpMode);  \
    template void applyND<Remap::HC2HC, T, size2_t, float2_t, float22_t>(const T*, T*, size2_t, const float22_t*, const float2_t*, size_t, float, InterpMode); \
    template void applyND<Remap::HC2H, T, size2_t, float2_t, float22_t>(const T*, T*, size2_t, const float22_t*, float2_t, size_t, float, InterpMode);         \
    template void applyND<Remap::HC2HC, T, size2_t, float2_t, float22_t>(const T*, T*, size2_t, const float22_t*, float2_t, size_t, float, InterpMode);        \
    template void applyND<Remap::HC2H, T, size3_t, float3_t, float33_t>(const T*, T*, size3_t, const float33_t*, const float3_t*, size_t, float, InterpMode);  \
    template void applyND<Remap::HC2HC, T, size3_t, float3_t, float33_t>(const T*, T*, size3_t, const float33_t*, const float3_t*, size_t, float, InterpMode); \
    template void applyND<Remap::HC2H, T, size3_t, float3_t, float33_t>(const T*, T*, size3_t, const float33_t*, float3_t, size_t, float, InterpMode);         \
    template void applyND<Remap::HC2HC, T, size3_t, float3_t, float33_t>(const T*, T*, size3_t, const float33_t*, float3_t, size_t, float, InterpMode)

    NOA_INSTANTIATE_APPLYND_(float);
    NOA_INSTANTIATE_APPLYND_(double);
    NOA_INSTANTIATE_APPLYND_(cfloat_t);
    NOA_INSTANTIATE_APPLYND_(cdouble_t);
}
