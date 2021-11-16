#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/transform/fft/Apply.h"
#include "noa/cpu/transform/fft/Symmetry.h"

// TODO(TF) I'm not happy with the implementation. Add benchmark and try to improve it.

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
    // Coordinates are normalized to account for dimensions with different sizes.
    template<bool IS_DST_CENTERED, InterpMode INTERP, typename T>
    void applyCenteredNormalized_(const T* input, T* output, size2_t shape,
                                  float22_t rotm, const float33_t* symmetry_matrices, size_t symmetry_count,
                                  [[maybe_unused]] float2_t shift,
                                  float max_frequency, bool normalize) {
        long2_t l_shape(shape);
        float2_t f_shape(l_shape.x > 1 ? l_shape.x / 2 * 2 : 1,
                         l_shape.y > 1 ? l_shape.y / 2 * 2 : 1);

        [[maybe_unused]] const bool apply_shift = any(shift != 0.f);
        shift *= math::Constants<float>::PI2 / float2_t(l_shape);

        max_frequency = noa::math::clamp(max_frequency, 0.f, 0.5f);
        max_frequency *= max_frequency;

        using real_t = traits::value_type_t<T>;
        real_t scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;

        const size_t half_x = shape.x / 2 + 1;
        const auto l_half_x = static_cast<int64_t>(half_x);
        cpu::transform::Interpolator2D<T> interp(input, {half_x, shape.y}, half_x, 0);

        for (size_t idx = 0; idx < symmetry_count + 1; ++idx) {
            float22_t matrix = rotm;
            if (idx)
                matrix *= float22_t(symmetry_matrices[idx - 1]);
            bool is_final = idx == symmetry_count;

            for (int64_t y = 0; y < l_shape.y; ++y) {
                int64_t v = getFrequency_<IS_DST_CENTERED>(y, l_shape.y, l_shape.y / 2);

                for (int64_t x = 0; x < l_half_x; ++x, ++output) {
                    float2_t freq = float2_t(x, v) / f_shape; // [-0.5, 0.5]
                    if (math::dot(freq, freq) > max_frequency)
                        continue;

                    freq = matrix * freq;
                    [[maybe_unused]] real_t conj = 1;
                    if (freq.x < 0.f) {
                        freq = -freq;
                        if constexpr (traits::is_complex_v<T>)
                            conj = -1;
                    }
                    freq.y += 0.5f; // [0, 1]
                    freq *= f_shape; // [0, N-1]
                    T value = interp.template get<INTERP, BORDER_ZERO>(freq);
                    if constexpr (traits::is_complex_v<T>)
                        value.imag *= conj;

                    if (idx) // add symmetry
                        *output += value;
                    else // first pass, set the output
                        *output = value;

                    // FIXME These ifs are not great but branch prediction should take care of it, right?
                    if (is_final) {
                        if constexpr (traits::is_complex_v<T>) {
                            if (apply_shift) {
                                T phase_shift = static_cast<T>(getPhaseShift_(shift, float2_t(x, v)));
                                *output *= scaling * phase_shift;
                            } else {
                                *output *= scaling;
                            }
                        } else {
                            *output *= scaling;
                        }
                    }
                }
            }
        }
    }

    // 3D, centered input.
    // Coordinates are normalized to account for dimensions with different sizes.
    template<bool IS_DST_CENTERED, InterpMode INTERP, typename T>
    void applyCenteredNormalized_(const T* input, T* output, size3_t shape,
                                  float33_t rotm, const float33_t* symmetry_matrices, size_t symmetry_count,
                                  [[maybe_unused]] float3_t shift,
                                  float max_frequency, bool normalize) {
        using real_t = traits::value_type_t<T>;

        long3_t l_shape(shape);
        float3_t f_shape(l_shape.x > 1 ? l_shape.x / 2 * 2 : 1,
                         l_shape.y > 1 ? l_shape.y / 2 * 2 : 1,
                         l_shape.z > 1 ? l_shape.z / 2 * 2 : 1);

        [[maybe_unused]] const bool apply_shift = any(shift != 0.f);
        shift *= math::Constants<float>::PI2 / float3_t(l_shape);

        max_frequency = noa::math::clamp(max_frequency, 0.f, 0.5f);
        max_frequency *= max_frequency;

        real_t scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;

        const size_t half_x = shape.x / 2 + 1;
        const auto l_half_x = static_cast<int64_t>(half_x);
        cpu::transform::Interpolator3D<T> interp(input, {half_x, shape.y, shape.z}, half_x, 0);

        for (size_t idx = 0; idx < symmetry_count + 1; ++idx) {
            float33_t matrix = rotm;
            if (idx)
                matrix *= symmetry_matrices[idx - 1];
            bool is_final = idx == symmetry_count;

            for (int64_t z = 0; z < l_shape.z; ++z) {
                int64_t w = getFrequency_<IS_DST_CENTERED>(z, l_shape.z, l_shape.z / 2);

                for (int64_t y = 0; y < l_shape.y; ++y) {
                    int64_t v = getFrequency_<IS_DST_CENTERED>(y, l_shape.y, l_shape.y / 2);

                    for (int64_t x = 0; x < l_half_x; ++x, ++output) {
                        float3_t coordinates = float3_t(x, v, w) / f_shape; // [-0.5, 0.5]
                        if (math::dot(coordinates, coordinates) > max_frequency)
                            continue;

                        coordinates = matrix * coordinates;
                        [[maybe_unused]] real_t conj = 1;
                        if (coordinates.x < 0.f) {
                            coordinates = -coordinates;
                            if constexpr (traits::is_complex_v<T>)
                                conj = -1;
                        }
                        coordinates.y += 0.5f; // [0, 1]
                        coordinates.z += 0.5f; // [0, 1]
                        coordinates *= f_shape; // [0, N-1]
                        T value = interp.template get<INTERP, BORDER_ZERO>(coordinates);
                        if constexpr (traits::is_complex_v<T>)
                            value.imag *= conj;

                        if (idx)
                            *output += value;
                        else
                            *output = value;

                        if (is_final) {
                            if constexpr (traits::is_complex_v<T>) {
                                if (apply_shift) {
                                    T phase_shift = static_cast<T>(getPhaseShift_(shift, float3_t(x, v, w)));
                                    *output *= scaling * phase_shift;
                                } else {
                                    *output *= scaling;
                                }
                            } else {
                                *output *= scaling;
                            }
                        }
                    }
                }
            }
        }
    }

    template<bool IS_DST_CENTERED, typename T, typename SHAPE, typename SHIFT, typename ROT>
    inline void launchCentered_(const T* input, T* output, SHAPE shape,
                                ROT rotm, const float33_t* sym, size_t count, SHIFT shift,
                                float max_frequency, InterpMode interp_mode, bool normalize) {
        switch (interp_mode) {
            case InterpMode::INTERP_NEAREST:
                applyCenteredNormalized_<IS_DST_CENTERED, InterpMode::INTERP_NEAREST>(
                        input, output, shape, rotm, sym, count, shift, max_frequency, normalize);
                break;
            case InterpMode::INTERP_LINEAR:
                applyCenteredNormalized_<IS_DST_CENTERED, InterpMode::INTERP_LINEAR>(
                        input, output, shape, rotm, sym, count, shift, max_frequency, normalize);
                break;
            case InterpMode::INTERP_COSINE:
                applyCenteredNormalized_<IS_DST_CENTERED, InterpMode::INTERP_COSINE>(
                        input, output, shape, rotm, sym, count, shift, max_frequency, normalize);
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

namespace noa::cpu::transform::fft {
    template<Remap REMAP, typename T>
    void apply2D(const T* input, T* output, size2_t shape,
                 float22_t transform, const Symmetry& symmetry, float2_t shift,
                 float max_frequency, InterpMode interp_mode, bool normalize) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        const size_t count = symmetry.count();
        if (!count)
            return apply2D<REMAP>(input, output, shape, transform, shift, max_frequency, interp_mode);
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        launchCentered_<IS_DST_CENTERED>(input, output, shape, transform, symmetry.matrices(), count, shift,
                                         max_frequency, interp_mode, normalize);
    }

    template<Remap REMAP, typename T>
    void apply3D(const T* input, T* output, size3_t shape,
                 float33_t transform, const Symmetry& symmetry, float3_t shift,
                 float max_frequency, InterpMode interp_mode, bool normalize) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        const size_t count = symmetry.count();
        if (!count)
            return apply3D<REMAP>(input, output, shape, transform, shift, max_frequency, interp_mode);
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        launchCentered_<IS_DST_CENTERED>(input, output, shape, transform, symmetry.matrices(), count, shift,
                                         max_frequency, interp_mode, normalize);
    }

    #define NOA_INSTANTIATE_APPLY_(T)                                                                                   \
    template void apply2D<Remap::HC2HC, T>(const T*, T*, size2_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool); \
    template void apply2D<Remap::HC2H, T>(const T*, T*, size2_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool);  \
    template void apply3D<Remap::HC2HC, T>(const T*, T*, size3_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool); \
    template void apply3D<Remap::HC2H, T>(const T*, T*, size3_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool)

    NOA_INSTANTIATE_APPLY_(float);
    NOA_INSTANTIATE_APPLY_(double);
    NOA_INSTANTIATE_APPLY_(cfloat_t);
    NOA_INSTANTIATE_APPLY_(cdouble_t);
}

namespace noa::cpu::transform::fft {
    template<Remap REMAP, typename T>
    void symmetrize2D(const T* input, T* output, size2_t shape,
                      const Symmetry& symmetry, float2_t shift,
                      float max_frequency, InterpMode interp_mode, bool normalize) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        launchCentered_<IS_DST_CENTERED>(input, output, shape, float22_t(), symmetry.matrices(), symmetry.count(),
                                         shift, max_frequency, interp_mode, normalize);
    }

    template<Remap REMAP, typename T>
    void symmetrize3D(const T* input, T* output, size3_t shape,
                      const Symmetry& symmetry, float3_t shift,
                      float max_frequency, InterpMode interp_mode, bool normalize) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        launchCentered_<IS_DST_CENTERED>(input, output, shape, float33_t(), symmetry.matrices(), symmetry.count(),
                                         shift, max_frequency, interp_mode, normalize);
    }

    #define NOA_INSTANTIATE_SYMMETRIZE_(T)                                                                                  \
    template void symmetrize2D<Remap::HC2HC, T>(const T*, T*, size2_t, const Symmetry&, float2_t, float, InterpMode, bool); \
    template void symmetrize2D<Remap::HC2H, T>(const T*, T*, size2_t, const Symmetry&, float2_t, float, InterpMode, bool);  \
    template void symmetrize3D<Remap::HC2HC, T>(const T*, T*, size3_t, const Symmetry&, float3_t, float, InterpMode, bool); \
    template void symmetrize3D<Remap::HC2H, T>(const T*, T*, size3_t, const Symmetry&, float3_t, float, InterpMode, bool)

    NOA_INSTANTIATE_SYMMETRIZE_(float);
    NOA_INSTANTIATE_SYMMETRIZE_(double);
    NOA_INSTANTIATE_SYMMETRIZE_(cfloat_t);
    NOA_INSTANTIATE_SYMMETRIZE_(cdouble_t);
}
