#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/geometry/Interpolator.h"
#include "noa/cpu/geometry/fft/Transform.h"

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
    inline T interpolateFFT_(float2_t frequency, float2_t f_shape,
                            const cpu::geometry::Interpolator2D<T>& interp, size_t offset) {
        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] real_t conj = 1;
        if (frequency[1] < 0.f) {
            frequency = -frequency;
            if constexpr (traits::is_complex_v<T>)
                conj = -1;
        }
        frequency[0] += 0.5f; // [0, 1]
        frequency *= f_shape; // [0, N-1]
        T value = interp.template get<INTERP, BORDER_ZERO>(frequency, offset);
        if constexpr (traits::is_complex_v<T>)
            value.imag *= conj;
        return value;
    }

    template<InterpMode INTERP, typename T>
    inline T interpolateFFT_(float3_t frequency, float3_t f_shape,
                            const cpu::geometry::Interpolator3D<T>& interp, size_t offset) {
        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] real_t conj = 1;
        if (frequency[2] < 0.f) {
            frequency = -frequency;
            if constexpr (traits::is_complex_v<T>)
                conj = -1;
        }
        frequency[0] += 0.5f; // [0, 1]
        frequency[1] += 0.5f; // [0, 1]
        frequency *= f_shape; // [0, N-1]
        T value = interp.template get<INTERP, BORDER_ZERO>(frequency, offset);
        if constexpr (traits::is_complex_v<T>)
            value.imag *= conj;
        return value;
    }

    // 2D, centered input.
    template<bool IS_DST_CENTERED, bool IS_IDENTITY, InterpMode INTERP, typename T>
    void applyCenteredNormalized2D_(const T* input, size3_t input_stride, T* output, size3_t output_stride,
                                    size3_t shape, [[maybe_unused]] float22_t matrix,
                                    const geometry::Symmetry& symmetry, float2_t shift, float cutoff,
                                    bool normalize, size_t threads) {
        const size_t batches = shape[0];
        const long2_t l_shape{shape.get() + 1};
        const float2_t f_shape{l_shape / 2 * 2 + long2_t{l_shape == 1}};

        [[maybe_unused]] const bool apply_shift = any(shift != 0.f);
        shift *= math::Constants<float>::PI2 / float2_t{l_shape};

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        const size_t count = symmetry.count();
        const float33_t* sym_matrices = symmetry.matrices();

        using real_t = traits::value_type_t<T>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(count + 1) : 1;

        const size2_t stride{input_stride.get() + 1};
        const cpu::geometry::Interpolator2D<T> interp(input, stride, size2_t{shape.get() + 1}.fft(), T(0));

        #pragma omp parallel for default(none) num_threads(threads) collapse(3)   \
        shared(input, input_stride, output, output_stride, matrix, shift, cutoff, \
               batches, l_shape, f_shape, interp, apply_shift, count, sym_matrices, scaling)

        for (size_t i = 0; i < batches; ++i) {
            for (int64_t y = 0; y < l_shape[0]; ++y) {
                for (int64_t x = 0; x < l_shape[1] / 2 + 1; ++x) {

                    const float2_t coordinates{getFrequency_<IS_DST_CENTERED>(y, l_shape[0]), x};

                    float2_t freq = coordinates / f_shape; // [-0.5, 0.5]
                    if (math::dot(freq, freq) > cutoff) {
                        output[at(i, y, x, output_stride)] = 0;
                        continue;
                    }

                    T value;
                    if constexpr (IS_IDENTITY) {
                        const int64_t iy = getInputIndex_<IS_DST_CENTERED>(y, l_shape[0]);
                        value = input[at(i, iy, x, input_stride)];
                    } else {
                        freq = matrix * freq;
                        value = interpolateFFT_<INTERP>(freq, f_shape, interp, i * input_stride[0]);
                    }
                    for (size_t s = 0; s < count; ++s) {
                        const float33_t& m = sym_matrices[s];
                        const float22_t sym_matrix{m[1][1], m[1][2],
                                                   m[2][1], m[2][2]};
                        value += interpolateFFT_<INTERP>(sym_matrix * freq, f_shape, interp, i * input_stride[0]);
                    }

                    if constexpr (traits::is_complex_v<T>)
                        if (apply_shift)
                            value *= getPhaseShift_<T>(shift, coordinates);

                    output[at(i, y, x, output_stride)] = value * scaling;
                }
            }
        }
    }

    // 3D, centered input.
    template<bool IS_DST_CENTERED, bool IS_IDENTITY, InterpMode INTERP, typename T>
    void applyCenteredNormalized3D_(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                                  size4_t shape, [[maybe_unused]] float33_t matrix,
                                  const geometry::Symmetry& symmetry, [[maybe_unused]] float3_t shift,
                                  float cutoff, bool normalize, size_t threads) {
        const size_t batches = shape[0];
        const long3_t l_shape{shape.get() + 1};
        const float3_t f_shape{l_shape / 2 * 2 + long3_t{l_shape == 1}};

        [[maybe_unused]] const bool apply_shift = any(shift != 0.f);
        shift *= math::Constants<float>::PI2 / float3_t{l_shape};

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        const size_t count = symmetry.count();
        const float33_t* matrices = symmetry.matrices();

        using real_t = traits::value_type_t<T>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(count + 1) : 1;

        const size3_t stride{input_stride.get() + 1};
        const cpu::geometry::Interpolator3D<T> interp(input, stride, size3_t{shape.get() + 1}.fft(), T(0));

        #pragma omp parallel for default(none) num_threads(threads) collapse(4)   \
        shared(input, input_stride, output, output_stride, matrix, shift, cutoff, \
               batches, l_shape, f_shape, interp, apply_shift, count, matrices, scaling)

        for (size_t i = 0; i < batches; ++i) {
            for (int64_t z = 0; z < l_shape[0]; ++z) {
                for (int64_t y = 0; y < l_shape[1]; ++y) {
                    for (int64_t x = 0; x < l_shape[2] / 2 + 1; ++x) {

                        const float3_t coordinates{getFrequency_<IS_DST_CENTERED>(z, l_shape[0]),
                                                   getFrequency_<IS_DST_CENTERED>(y, l_shape[1]),
                                                   x,};

                        float3_t freq = coordinates / f_shape; // [-0.5, 0.5]
                        if (math::dot(freq, freq) > cutoff) {
                            output[at(i, z, y, x, output_stride)] = 0;
                            continue;
                        }

                        T value;
                        if constexpr (IS_IDENTITY) {
                            const int64_t iz = getInputIndex_<IS_DST_CENTERED>(z, l_shape[0]);
                            const int64_t iy = getInputIndex_<IS_DST_CENTERED>(y, l_shape[1]);
                            value = input[at(i, iz, iy, x, input_stride)];
                        } else {
                            freq = matrix * freq;
                            value = interpolateFFT_<INTERP>(freq, f_shape, interp, i * input_stride[0]);
                        }
                        for (size_t s = 0; s < count; ++s)
                            value += interpolateFFT_<INTERP>(matrices[s] * freq, f_shape, interp, i * input_stride[0]);

                        if constexpr (traits::is_complex_v<T>)
                            if (apply_shift)
                                value *= getPhaseShift_<T>(shift, coordinates);

                        output[at(i, z, y, x, output_stride)] = value * scaling;
                    }
                }
            }
        }
    }

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

namespace noa::cpu::geometry::fft {
    template<Remap REMAP, typename T>
    NOA_IH void transform2D(const T* input, size4_t input_stride,
                            T* output, size4_t output_stride, size4_t shape,
                            float22_t matrix, const Symmetry& symmetry, float2_t shift,
                            float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        if (!symmetry.count())
            return transform2D<REMAP>(input, input_stride, output, output_stride, shape, matrix, shift,
                                      cutoff, interp_mode, stream);

        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const size_t threads = stream.threads();
        const bool is_identity = matrix == float22_t{};

        NOA_ASSERT(shape[1] == 1);
        const size3_t i_stride{input_stride[0], input_stride[2], input_stride[3]};
        const size3_t o_stride{output_stride[0], output_stride[2], output_stride[3]};
        const size3_t shape_2d{shape[0], shape[2], shape[3]};

        switch (interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue(is_identity ?
                                      applyCenteredNormalized2D_<IS_DST_CENTERED, true, INTERP_NEAREST, T> :
                                      applyCenteredNormalized2D_<IS_DST_CENTERED, false, INTERP_NEAREST, T>,
                                      input, i_stride, output, o_stride, shape_2d, matrix, symmetry,
                                      shift, cutoff, normalize, threads);
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return stream.enqueue(is_identity ?
                                      applyCenteredNormalized2D_<IS_DST_CENTERED, true, INTERP_LINEAR, T> :
                                      applyCenteredNormalized2D_<IS_DST_CENTERED, false, INTERP_LINEAR, T>,
                                      input, i_stride, output, o_stride, shape_2d, matrix, symmetry,
                                      shift, cutoff, normalize, threads);
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return stream.enqueue(is_identity ?
                                      applyCenteredNormalized2D_<IS_DST_CENTERED, true, INTERP_COSINE, T> :
                                      applyCenteredNormalized2D_<IS_DST_CENTERED, false, INTERP_COSINE, T>,
                                      input, i_stride, output, o_stride, shape_2d, matrix, symmetry,
                                      shift, cutoff, normalize, threads);
            default:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    template<Remap REMAP, typename T>
    void transform3D(const T* input, size4_t input_stride,
                     T* output, size4_t output_stride, size4_t shape,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        if (!symmetry.count())
            return transform3D<REMAP>(input, input_stride, output, output_stride, shape, matrix, shift,
                                      cutoff, interp_mode, stream);

        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const size_t threads = stream.threads();
        const bool is_identity = matrix == float33_t{};

        switch (interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue(is_identity ?
                                      applyCenteredNormalized3D_<IS_DST_CENTERED, true, INTERP_NEAREST, T> :
                                      applyCenteredNormalized3D_<IS_DST_CENTERED, false, INTERP_NEAREST, T>,
                                      input, input_stride, output, output_stride, shape, matrix, symmetry,
                                      shift, cutoff, normalize, threads);
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return stream.enqueue(is_identity ?
                                      applyCenteredNormalized3D_<IS_DST_CENTERED, true, INTERP_LINEAR, T> :
                                      applyCenteredNormalized3D_<IS_DST_CENTERED, false, INTERP_LINEAR, T>,
                                      input, input_stride, output, output_stride, shape, matrix, symmetry,
                                      shift, cutoff, normalize, threads);
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return stream.enqueue(is_identity ?
                                      applyCenteredNormalized3D_<IS_DST_CENTERED, true, INTERP_COSINE, T> :
                                      applyCenteredNormalized3D_<IS_DST_CENTERED, false, INTERP_COSINE, T>,
                                      input, input_stride, output, output_stride, shape, matrix, symmetry,
                                      shift, cutoff, normalize, threads);
            default:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    #define NOA_INSTANTIATE_TRANSFORM_(T)                                                                                                                           \
    template void transform2D<Remap::HC2HC, T>(const T*, size4_t, T*, size4_t, size4_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&l);   \
    template void transform2D<Remap::HC2H, T>(const T*, size4_t, T*, size4_t, size4_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&);     \
    template void transform3D<Remap::HC2HC, T>(const T*, size4_t, T*, size4_t, size4_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&);    \
    template void transform3D<Remap::HC2H, T>(const T*, size4_t, T*, size4_t, size4_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&)

    NOA_INSTANTIATE_TRANSFORM_(float);
    NOA_INSTANTIATE_TRANSFORM_(double);
    NOA_INSTANTIATE_TRANSFORM_(cfloat_t);
    NOA_INSTANTIATE_TRANSFORM_(cdouble_t);
}
