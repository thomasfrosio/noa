#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/transform/fft/Apply.h"

namespace {
    using namespace ::noa;

    template<bool IS_DST_CENTERED>
    inline int64_t getFrequency_(int64_t idx, int64_t dim) {
        if constexpr(IS_DST_CENTERED)
            return idx - dim / 2;
        else
            return idx < (dim + 1) / 2 ? idx : idx - dim;
    }

    template<typename C, typename T>
    inline C getPhaseShift_(T shift, T freq) {
        static_assert(traits::is_float2_v<T> || traits::is_float3_v<T>);
        using real_t = traits::value_type_t<C>;
        const auto factor = static_cast<real_t>(-math::dot(shift, freq));
        C phase_shift;
        math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    template<bool IS_DST_CENTERED>
    inline int64_t getInputIndex_(int64_t idx, [[maybe_unused]] int64_t dim) {
        if constexpr (IS_DST_CENTERED)
            return idx;
        else
            return noa::math::FFTShift(idx, dim);
    }

    template<InterpMode INTERP, typename T>
    inline T interpolateFFT(float2_t frequency, float2_t f_shape, const cpu::transform::Interpolator2D<T>& interp) {
        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] real_t conj = 1;
        if (frequency.x < 0.f) {
            frequency = -frequency;
            if constexpr (traits::is_complex_v<T>)
                conj = -1;
        }
        frequency.y += 0.5f; // [0, 1]
        frequency *= f_shape; // [0, N-1]
        T value = interp.template get<INTERP, BORDER_ZERO>(frequency);
        if constexpr (traits::is_complex_v<T>)
            value.imag *= conj;
        return value;
    }

    template<InterpMode INTERP, typename T>
    inline T interpolateFFT(float3_t frequency, float3_t f_shape, const cpu::transform::Interpolator3D<T>& interp) {
        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] real_t conj = 1;
        if (frequency.x < 0.f) {
            frequency = -frequency;
            if constexpr (traits::is_complex_v<T>)
                conj = -1;
        }
        frequency.y += 0.5f; // [0, 1]
        frequency.z += 0.5f; // [0, 1]
        frequency *= f_shape; // [0, N-1]
        T value = interp.template get<INTERP, BORDER_ZERO>(frequency);
        if constexpr (traits::is_complex_v<T>)
            value.imag *= conj;
        return value;
    }

    // 2D, centered input.
    // Coordinates are normalized to account for dimensions with different sizes.
    template<bool IS_DST_CENTERED, bool IS_IDENTITY, InterpMode INTERP, typename T>
    void applyCenteredNormalized_(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                                  [[maybe_unused]] float22_t transform, const transform::Symmetry& symmetry,
                                  [[maybe_unused]] float2_t shift, float cutoff, bool normalize,
                                  size_t threads) {
        const long2_t l_shape(shape);
        const float2_t f_shape(l_shape / 2 * 2 + long2_t{l_shape == 1});

        [[maybe_unused]] const bool apply_shift = any(shift != 0.f);
        shift *= math::Constants<float>::PI2 / float2_t(l_shape);

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        const size_t count = symmetry.count();
        const float33_t* sym_matrices = symmetry.matrices();
        cpu::memory::PtrHost<float22_t> buffer(count);
        float22_t* matrices = buffer.get();
        for (size_t i = 0; i < count; ++i) {
            matrices[i] = float22_t(sym_matrices[i]);
            if constexpr (!IS_IDENTITY)
                matrices[i] *= transform;
        }

        using real_t = traits::value_type_t<T>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(count + 1) : 1;
        cpu::transform::Interpolator2D<T> interp(input, {input_pitch, 0}, {shape.x / 2 + 1, shape.y}, 0);

        #pragma omp parallel for default(none) num_threads(threads) collapse(2) \
        shared(input, input_pitch, output, output_pitch, transform, shift, cutoff, \
               l_shape, f_shape, interp, apply_shift, count, matrices, scaling)

        for (int64_t y = 0; y < l_shape.y; ++y) {
            for (int64_t x = 0; x < l_shape.x / 2 + 1; ++x) {

                const size_t offset = index(x, y, output_pitch);
                const float2_t coordinates(x, getFrequency_<IS_DST_CENTERED>(y, l_shape.y));

                const float2_t freq = coordinates / f_shape; // [-0.5, 0.5]
                if (math::dot(freq, freq) > cutoff) {
                    output[offset] = 0;
                    continue;
                }

                T value;
                if constexpr (IS_IDENTITY)
                    value = input[index(x, getInputIndex_<IS_DST_CENTERED>(y, l_shape.y), input_pitch)];
                else
                    value = interpolateFFT<INTERP>(transform * freq, f_shape, interp);
                for (size_t i = 0; i < count; ++i)
                    value += interpolateFFT<INTERP>(matrices[i] * freq, f_shape, interp);

                if constexpr (traits::is_complex_v<T>) {
                    if (apply_shift)
                        value *= getPhaseShift_<T>(shift, coordinates);
                }
                output[offset] = value * scaling;
            }
        }
    }

    // 3D, centered input.
    // Coordinates are normalized to account for dimensions with different sizes.
    template<bool IS_DST_CENTERED, bool IS_IDENTITY, InterpMode INTERP, typename T>
    void applyCenteredNormalized_(const T* input, size2_t input_pitch, T* output, size2_t output_pitch, size3_t shape,
                                  [[maybe_unused]] float33_t transform, const transform::Symmetry& symmetry,
                                  [[maybe_unused]] float3_t shift, float cutoff, bool normalize,
                                  size_t threads) {
        const long3_t l_shape(shape);
        const float3_t f_shape(l_shape / 2 * 2 + long3_t{l_shape == 1});

        [[maybe_unused]] const bool apply_shift = any(shift != 0.f);
        shift *= math::Constants<float>::PI2 / float3_t(l_shape);

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        const size_t count = symmetry.count();
        const float33_t* sym_matrices = symmetry.matrices();
        cpu::memory::PtrHost<float33_t> buffer;
        const float33_t* matrices;
        if constexpr (IS_IDENTITY) {
            matrices = sym_matrices;
        } else {
            buffer.reset(count);
            for (size_t i = 0; i < count; ++i)
                buffer[i] = sym_matrices[i] * transform;
            matrices = buffer.get();
        }

        using real_t = traits::value_type_t<T>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(count + 1) : 1;
        cpu::transform::Interpolator3D<T> interp(input, {input_pitch, 0}, shapeFFT(shape), 0);

        #pragma omp parallel for default(none) num_threads(threads) collapse(2) \
        shared(input, input_pitch, output, output_pitch, transform, shift, cutoff, l_shape, f_shape, interp, apply_shift, count, matrices, scaling)

        for (int64_t z = 0; z < l_shape.z; ++z) {
            for (int64_t y = 0; y < l_shape.y; ++y) {
                for (int64_t x = 0; x < l_shape.x / 2 + 1; ++x) {

                    const size_t offset = index(x, y, z, output_pitch.x, output_pitch.y);
                    const float3_t coordinates(x,
                                               getFrequency_<IS_DST_CENTERED>(y, l_shape.y),
                                               getFrequency_<IS_DST_CENTERED>(z, l_shape.z));

                    const float3_t freq = coordinates / f_shape; // [-0.5, 0.5]
                    if (math::dot(freq, freq) > cutoff) {
                        output[offset] = 0;
                        continue;
                    }

                    T value;
                    if constexpr (IS_IDENTITY) {
                        const int64_t iy = getInputIndex_<IS_DST_CENTERED>(y, l_shape.y);
                        const int64_t iz = getInputIndex_<IS_DST_CENTERED>(z, l_shape.z);
                        value = input[index(x, iy, iz, input_pitch.x, input_pitch.y)];
                    } else {
                        value = interpolateFFT<INTERP>(transform * freq, f_shape, interp);
                    }
                    for (size_t i = 0; i < count; ++i)
                        value += interpolateFFT<INTERP>(matrices[i] * freq, f_shape, interp);

                    if constexpr (traits::is_complex_v<T>) {
                        if (apply_shift)
                            value *= getPhaseShift_<T>(shift, coordinates);
                    }
                    output[offset] = value * scaling;
                }
            }
        }
    }

    template<bool IS_DST_CENTERED, bool IS_IDENTITY, typename T, typename U, typename V, typename W, typename Z>
    inline void launchCentered_(const T* input, U input_pitch, T* output, U output_pitch, V shape,
                                W transform, const transform::Symmetry& symmetry,
                                Z shift, float cutoff, InterpMode interp_mode, bool normalize, size_t threads) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                return applyCenteredNormalized_<IS_DST_CENTERED, IS_IDENTITY, INTERP_NEAREST>(
                        input, input_pitch, output, output_pitch, shape, transform, symmetry,
                        shift, cutoff, normalize, threads);
            case INTERP_LINEAR:
                return applyCenteredNormalized_<IS_DST_CENTERED, IS_IDENTITY, INTERP_LINEAR>(
                        input, input_pitch, output, output_pitch, shape, transform, symmetry,
                        shift, cutoff, normalize, threads);
            case INTERP_COSINE:
                return applyCenteredNormalized_<IS_DST_CENTERED, IS_IDENTITY, INTERP_COSINE>(
                        input, input_pitch, output, output_pitch, shape, transform, symmetry,
                        shift, cutoff, normalize, threads);
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
    void apply2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                 float22_t transform, const Symmetry& symmetry, float2_t shift,
                 float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        if (!symmetry.count())
            return apply2D<REMAP>(input, input_pitch, output, output_pitch, shape, transform, shift,
                                  cutoff, interp_mode, stream);

        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const size_t threads = stream.threads();
        if (transform == float22_t()) {
            stream.enqueue(launchCentered_<IS_DST_CENTERED, true, T, size_t, size2_t, float22_t, float2_t>,
                           input, input_pitch, output, output_pitch, shape, transform, symmetry, shift,
                           cutoff, interp_mode, normalize, threads);
        } else {
            stream.enqueue(launchCentered_<IS_DST_CENTERED, false, T, size_t, size2_t, float22_t, float2_t>,
                           input, input_pitch, output, output_pitch, shape, transform, symmetry, shift,
                           cutoff, interp_mode, normalize, threads);
        }
    }

    template<Remap REMAP, typename T>
    void apply3D(const T* input, size2_t input_pitch, T* output, size2_t output_pitch, size3_t shape,
                 float33_t transform, const Symmetry& symmetry, float3_t shift,
                 float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        if (!symmetry.count())
            return apply3D<REMAP>(input, input_pitch, output, output_pitch, shape, transform, shift,
                                  cutoff, interp_mode, stream);

        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const size_t threads = stream.threads();
        if (transform == float33_t()) {
            stream.enqueue(launchCentered_<IS_DST_CENTERED, true, T, size2_t, size3_t, float33_t, float3_t>,
                           input, input_pitch, output, output_pitch, shape, transform, symmetry, shift,
                           cutoff, interp_mode, normalize, threads);
        } else {
            stream.enqueue(launchCentered_<IS_DST_CENTERED, false, T, size2_t, size3_t, float33_t, float3_t>,
                           input, input_pitch, output, output_pitch, shape, transform, symmetry, shift,
                           cutoff, interp_mode, normalize, threads);
        }
    }

    #define NOA_INSTANTIATE_APPLY_(T)                                                                                                                       \
    template void apply2D<Remap::HC2HC, T>(const T*, size_t, T*, size_t, size2_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&l); \
    template void apply2D<Remap::HC2H, T>(const T*, size_t, T*, size_t, size2_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&);   \
    template void apply3D<Remap::HC2HC, T>(const T*, size2_t, T*, size2_t, size3_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&);\
    template void apply3D<Remap::HC2H, T>(const T*, size2_t, T*, size2_t, size3_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&)

    NOA_INSTANTIATE_APPLY_(float);
    NOA_INSTANTIATE_APPLY_(double);
    NOA_INSTANTIATE_APPLY_(cfloat_t);
    NOA_INSTANTIATE_APPLY_(cdouble_t);
}
