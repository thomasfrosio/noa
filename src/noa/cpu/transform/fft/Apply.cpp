#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/transform/fft/Apply.h"


// Note: To support rectangular shapes, the kernels compute the transformation using normalized frequencies.
//       One other solution could have been to use an affine transform encoding the appropriate scaling to effectively
//       normalize the frequencies. Both options are fine and probably equivalent performance-wise.

// FIXME    For even sizes, there's an asymmetry due to the fact that there's only one Nyquist. After fftshift,
//          this extra frequency is on the left side of the axis. However, since we work with non-redundant
//          transforms, there's an slight error that can be introduced. For the elements close (+-1 element) to x=0
//          and close (+-1 element) to y=z=0.5, if during the rotation x becomes negative and we have to flip it, the
//          interpolator will incorrectly weight towards 0 the output value. This is simply due to the fact that on
//          the right side of the axis, there's no Nyquist (the axis stops and n+1 element is OOB, i.e. =0).
//          For cryoEM images, this should be fine since values at that point are often 0.

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

    // 2D, centered input.
    template<bool IS_DST_CENTERED, InterpMode INTERP, typename T>
    void applyCenteredNormalized_(const T* inputs, size2_t input_pitch, T* outputs, size2_t output_pitch, size2_t shape,
                                  const float22_t* transforms, const float2_t* shifts, size_t batches,
                                  float max_frequency, size_t threads) {
        long2_t l_shape(shape);
        float2_t f_shape(l_shape / 2 * 2 + long2_t{l_shape == 1}); // if odd, n-1
        cpu::transform::Interpolator2D<T> interp(inputs, input_pitch, {shape.x / 2 + 1, shape.y}, 0);
        const size_t base = elements(output_pitch);

        [[maybe_unused]] const float2_t pre_shift = math::Constants<float>::PI2 / float2_t(l_shape);

        max_frequency = noa::math::clamp(max_frequency, 0.f, 0.5f);
        max_frequency *= max_frequency;

        #pragma omp parallel for default(none) num_threads(threads) collapse(3) \
        shared(outputs, output_pitch, transforms, shifts, batches, max_frequency, l_shape, f_shape, interp, base, pre_shift)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (int64_t y = 0; y < l_shape.y; ++y) {
                for (int64_t x = 0; x < l_shape.x / 2 + 1; ++x) {

                    const size_t offset = batch * base + index(x, y, output_pitch.x);
                    const float2_t freq(x, getFrequency_<IS_DST_CENTERED>(y, l_shape.y));

                    float2_t coordinates = freq / f_shape; // [-0.5, 0.5]
                    if (math::dot(coordinates, coordinates) > max_frequency) {
                        outputs[offset] = 0;
                        continue;
                    }

                    coordinates = transforms[batch] * coordinates;
                    using real_t = traits::value_type_t<T>;
                    [[maybe_unused]] real_t conj = 1;
                    if (coordinates.x < 0.f) {
                        coordinates = -coordinates;
                        if constexpr (traits::is_complex_v<T>)
                            conj = -1;
                    }

                    coordinates.y += 0.5f; // [0, 1]
                    coordinates *= f_shape; // [0, N-1]
                    T value = interp.template get<INTERP, BORDER_ZERO>(coordinates, batch);
                    if constexpr (traits::is_complex_v<T>) {
                        value.imag *= conj;
                        if (shifts)
                            value *= getPhaseShift_<T>(shifts[batch] * pre_shift, freq);
                    }
                    outputs[offset] = value;
                }
            }
        }
    }

    // 3D, centered input.
    template<bool IS_DST_CENTERED, InterpMode INTERP, typename T>
    void applyCenteredNormalized_(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                                  const float33_t* transforms, const float3_t* shifts, size_t batches,
                                  float max_frequency, size_t threads) {
        long3_t l_shape(shape);
        float3_t f_shape(l_shape / 2 * 2 + long3_t{l_shape == 1}); // if odd, n-1

        cpu::transform::Interpolator3D<T> interp(inputs, input_pitch, {shape.x / 2 + 1, shape.y, shape.z}, 0);
        const size_t base = elements(output_pitch);

        [[maybe_unused]] const float3_t pre_shift = math::Constants<float>::PI2 / float3_t(l_shape);

        max_frequency = noa::math::clamp(max_frequency, 0.f, 0.5f);
        max_frequency *= max_frequency;

        #pragma omp parallel for default(none) num_threads(threads) collapse(4) \
        shared(outputs, output_pitch, transforms, shifts, batches, max_frequency, l_shape, f_shape, interp, base, pre_shift)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (int64_t z = 0; z < l_shape.z; ++z) {
                for (int64_t y = 0; y < l_shape.y; ++y) {
                    for (int64_t x = 0; x < l_shape.x / 2 + 1; ++x) {

                        const size_t offset = batch * base + index(x, y, z, output_pitch);
                        const float3_t freq(x,
                                            getFrequency_<IS_DST_CENTERED>(y, l_shape.y),
                                            getFrequency_<IS_DST_CENTERED>(z, l_shape.z));

                        float3_t coordinates = freq / f_shape;
                        if (math::dot(coordinates, coordinates) > max_frequency) {
                            outputs[offset] = 0;
                            continue;
                        }

                        coordinates = transforms[batch] * coordinates;
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
                        T value = interp.template get<INTERP, BORDER_ZERO>(coordinates, batch);
                        if constexpr (traits::is_complex_v<T>) {
                            value.imag *= conj;
                            if (shifts)
                                value *= getPhaseShift_<T>(shifts[batch] * pre_shift, freq);
                        }
                        outputs[offset] = value;
                    }
                }
            }
        }
    }

    template<bool IS_DST_CENTERED, typename T, typename U, typename V, typename W>
    inline void launchCentered_(const T* inputs, U input_pitch, T* outputs, U output_pitch, U shape,
                                const V* transforms, const W* shifts, size_t batches, float max_frequency,
                                InterpMode interp_mode, size_t threads) {
        switch (interp_mode) {
            case InterpMode::INTERP_NEAREST:
                return applyCenteredNormalized_<IS_DST_CENTERED, InterpMode::INTERP_NEAREST>(
                        inputs, input_pitch, outputs, output_pitch, shape,
                        transforms, shifts, batches, max_frequency, threads);
            case InterpMode::INTERP_LINEAR:
                return applyCenteredNormalized_<IS_DST_CENTERED, InterpMode::INTERP_LINEAR>(
                        inputs, input_pitch, outputs, output_pitch, shape,
                        transforms, shifts, batches, max_frequency, threads);
            case InterpMode::INTERP_COSINE:
                return applyCenteredNormalized_<IS_DST_CENTERED, InterpMode::INTERP_COSINE>(
                        inputs, input_pitch, outputs, output_pitch, shape,
                        transforms, shifts, batches, max_frequency, threads);
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
    template<Remap REMAP, typename T, typename U, typename V, typename W>
    void applyND(const T* inputs, U input_pitch, T* outputs, U output_pitch, U shape,
                 const V* transforms, const W* shifts, size_t nb_transforms,
                 float max_frequency, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const size_t threads = stream.threads();

        stream.enqueue(launchCentered_<IS_DST_CENTERED, T, U, V, W>,
                       inputs, input_pitch, outputs, output_pitch, shape,
                       transforms, shifts, nb_transforms, max_frequency,
                       interp_mode, threads);
    }

    #define NOA_INSTANTIATE_APPLYND_(T)                                                                                     \
    template void applyND<Remap::HC2H, T, size2_t, float22_t, float2_t>(                                                    \
        const T*, size2_t, T*, size2_t, size2_t, const float22_t*, const float2_t*, size_t, float, InterpMode, Stream&);    \
    template void applyND<Remap::HC2HC, T, size2_t, float22_t, float2_t>(                                                   \
        const T*, size2_t, T*, size2_t, size2_t, const float22_t*, const float2_t*, size_t, float, InterpMode, Stream&);    \
    template void applyND<Remap::HC2H, T, size3_t, float33_t, float3_t>(                                                    \
        const T*, size3_t, T*, size3_t, size3_t, const float33_t*, const float3_t*, size_t, float, InterpMode, Stream&);    \
    template void applyND<Remap::HC2HC, T, size3_t, float33_t, float3_t>(                                                   \
        const T*, size3_t, T*, size3_t, size3_t, const float33_t*, const float3_t*, size_t, float, InterpMode, Stream&)

    NOA_INSTANTIATE_APPLYND_(float);
    NOA_INSTANTIATE_APPLYND_(double);
    NOA_INSTANTIATE_APPLYND_(cfloat_t);
    NOA_INSTANTIATE_APPLYND_(cdouble_t);
}
