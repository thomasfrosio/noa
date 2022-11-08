#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/geometry/Interpolator.h"
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

    template<typename interpolator_t>
    inline auto interpolateFFT_(float2_t frequency, float2_t f_shape,
                                const interpolator_t& interpolator, int64_t batch) {
        using data_t = typename interpolator_t::data_type;
        using real_t = traits::value_type_t<data_t>;

        [[maybe_unused]] real_t conj = 1;
        if (frequency[1] < 0.f) {
            frequency = -frequency;
            if constexpr (traits::is_complex_v<data_t>)
                conj = -1;
        }
        frequency[0] += 0.5f; // [0, 1]
        frequency *= f_shape; // [0, N-1]
        data_t value = interpolator(frequency, batch);
        if constexpr (traits::is_complex_v<data_t>)
            value.imag *= conj;
        return value;
    }

    template<typename interpolator_t>
    inline auto interpolateFFT_(float3_t frequency, float3_t f_shape,
                                const interpolator_t& interpolator, int64_t batch) {
        using data_t = typename interpolator_t::data_type;
        using real_t = traits::value_type_t<data_t>;

        [[maybe_unused]] real_t conj = 1;
        if (frequency[2] < 0.f) {
            frequency = -frequency;
            if constexpr (traits::is_complex_v<data_t>)
                conj = -1;
        }
        frequency[0] += 0.5f; // [0, 1]
        frequency[1] += 0.5f; // [0, 1]
        frequency *= f_shape; // [0, N-1]
        data_t value = interpolator(frequency, batch);
        if constexpr (traits::is_complex_v<data_t>)
            value.imag *= conj;
        return value;
    }

    // 2D, centered input.
    template<bool IS_DST_CENTERED, bool IS_IDENTITY, InterpMode INTERP, typename T>
    void applyCenteredNormalized2D_(AccessorRestrict<const T, 3, int64_t> input,
                                    AccessorRestrict<T, 3, dim_t> output,
                                    dim3_t shape, [[maybe_unused]] float22_t matrix,
                                    const geometry::Symmetry& symmetry, float2_t shift, float cutoff,
                                    bool normalize, dim_t threads) {
        const auto batches = static_cast<int64_t>(shape[0]);
        const long2_t l_shape(shape.get(1));
        const float2_t f_shape(l_shape / 2 * 2 + long2_t(l_shape == 1));

        [[maybe_unused]] const bool apply_shift = any(shift != 0.f);
        shift *= math::Constants<float>::PI2 / float2_t(l_shape);

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        const dim_t count = symmetry.count();
        const float33_t* sym_matrices = symmetry.get();

        using real_t = traits::value_type_t<T>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(count + 1) : 1;

        auto interpolator = geometry::interpolator2D<BORDER_ZERO, INTERP>(input, long2_t(shape.get(1)).fft(), T{0});

        #pragma omp parallel for default(none) num_threads(threads) collapse(3) \
        shared(input, output, matrix, shift, cutoff, batches, l_shape, f_shape, \
               interpolator, apply_shift, count, sym_matrices, scaling)

        for (int64_t i = 0; i < batches; ++i) {
            for (int64_t y = 0; y < l_shape[0]; ++y) {
                for (int64_t x = 0; x < l_shape[1] / 2 + 1; ++x) {

                    const float2_t coordinates{getFrequency_<IS_DST_CENTERED>(y, l_shape[0]), x};

                    float2_t freq = coordinates / f_shape; // [-0.5, 0.5]
                    if (math::dot(freq, freq) > cutoff) {
                        output(i, y, x) = 0;
                        continue;
                    }

                    T value;
                    if constexpr (IS_IDENTITY) {
                        const int64_t iy = getInputIndex_<IS_DST_CENTERED>(y, l_shape[0]);
                        value = input(i, iy, x);
                    } else {
                        freq = matrix * freq;
                        value = interpolateFFT_(freq, f_shape, interpolator, i);
                    }
                    for (dim_t s = 0; s < count; ++s) {
                        const float33_t& m = sym_matrices[s];
                        const float22_t sym_matrix{m[1][1], m[1][2],
                                                   m[2][1], m[2][2]};
                        value += interpolateFFT_(sym_matrix * freq, f_shape, interpolator, i);
                    }

                    if constexpr (traits::is_complex_v<T>)
                        if (apply_shift)
                            value *= getPhaseShift_<T>(shift, coordinates);

                    output(i, y, x) = value * scaling;
                }
            }
        }
    }

    // 3D, centered input.
    template<bool IS_DST_CENTERED, bool IS_IDENTITY, InterpMode INTERP, typename T>
    void applyCenteredNormalized3D_(AccessorRestrict<const T, 4, int64_t> input,
                                    AccessorRestrict<T, 4, dim_t> output,
                                    dim4_t shape, [[maybe_unused]] float33_t matrix,
                                    const geometry::Symmetry& symmetry, [[maybe_unused]] float3_t shift,
                                    float cutoff, bool normalize, dim_t threads) {
        const auto batches = static_cast<int64_t>(shape[0]);
        const long3_t l_shape(shape.get(1));
        const float3_t f_shape(l_shape / 2 * 2 + long3_t(l_shape == 1));

        [[maybe_unused]] const bool apply_shift = any(shift != 0.f);
        shift *= math::Constants<float>::PI2 / float3_t(l_shape);

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        const dim_t count = symmetry.count();
        const float33_t* matrices = symmetry.get();

        using real_t = traits::value_type_t<T>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(count + 1) : 1;

        auto interpolator = geometry::interpolator3D<BORDER_ZERO, INTERP>(input, long3_t(shape.get(1)).fft(), T{0});

        #pragma omp parallel for default(none) num_threads(threads) collapse(4)     \
        shared(input, output, matrix, shift, cutoff, batches, l_shape, f_shape,     \
               interpolator, apply_shift, count, matrices, scaling)

        for (int64_t i = 0; i < batches; ++i) {
            for (int64_t z = 0; z < l_shape[0]; ++z) {
                for (int64_t y = 0; y < l_shape[1]; ++y) {
                    for (int64_t x = 0; x < l_shape[2] / 2 + 1; ++x) {

                        const float3_t coordinates{getFrequency_<IS_DST_CENTERED>(z, l_shape[0]),
                                                   getFrequency_<IS_DST_CENTERED>(y, l_shape[1]),
                                                   x,};

                        float3_t freq = coordinates / f_shape; // [-0.5, 0.5]
                        if (math::dot(freq, freq) > cutoff) {
                            output(i, z, y, x) = 0;
                            continue;
                        }

                        T value;
                        if constexpr (IS_IDENTITY) {
                            const int64_t iz = getInputIndex_<IS_DST_CENTERED>(z, l_shape[0]);
                            const int64_t iy = getInputIndex_<IS_DST_CENTERED>(y, l_shape[1]);
                            value = input(i, iz, iy, x);
                        } else {
                            freq = matrix * freq;
                            value = interpolateFFT_(freq, f_shape, interpolator, i);
                        }
                        for (dim_t s = 0; s < count; ++s)
                            value += interpolateFFT_(matrices[s] * freq, f_shape, interpolator, i);

                        if constexpr (traits::is_complex_v<T>)
                            if (apply_shift)
                                value *= getPhaseShift_<T>(shift, coordinates);

                        output(i, z, y, x) = value * scaling;
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
        if constexpr (!IS_SRC_CENTERED || REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);

        return IS_DST_CENTERED;
    }
}

namespace noa::cpu::geometry::fft {
    template<Remap REMAP, typename T, typename>
    void transform2D(const shared_t<T[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {

        if (!symmetry.count())
            return transform2D<REMAP>(input, input_strides, output, output_strides, shape, matrix, shift,
                                      cutoff, interp_mode, stream);

        NOA_ASSERT(input && output && input.get() != output.get() && all(shape > 0));
        NOA_ASSERT(shape[1] == 1);

        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const dim_t threads = stream.threads();
        const bool is_identity = matrix == float22_t{};

        const long3_t i_strides{input_strides[0], input_strides[2], input_strides[3]};
        const dim3_t o_strides{output_strides[0], output_strides[2], output_strides[3]};
        const dim3_t shape_2d{shape[0], shape[2], shape[3]};

        switch (interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue([=](){
                    if (is_identity)
                        applyCenteredNormalized2D_<IS_DST_CENTERED, true, INTERP_NEAREST, T>(
                                {input.get(), i_strides}, {output.get(), o_strides}, shape_2d,
                                matrix, symmetry, shift, cutoff, normalize, threads);
                    else
                        applyCenteredNormalized2D_<IS_DST_CENTERED, false, INTERP_NEAREST, T>(
                                {input.get(), i_strides}, {output.get(), o_strides}, shape_2d,
                                matrix, symmetry, shift, cutoff, normalize, threads);
                });
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return stream.enqueue([=](){
                    if (is_identity)
                        applyCenteredNormalized2D_<IS_DST_CENTERED, true, INTERP_LINEAR, T>(
                                {input.get(), i_strides}, {output.get(), o_strides}, shape_2d,
                                matrix, symmetry, shift, cutoff, normalize, threads);
                    else
                        applyCenteredNormalized2D_<IS_DST_CENTERED, false, INTERP_LINEAR, T>(
                                {input.get(), i_strides}, {output.get(), o_strides}, shape_2d,
                                matrix, symmetry, shift, cutoff, normalize, threads);
                });
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return stream.enqueue([=](){
                    if (is_identity)
                        applyCenteredNormalized2D_<IS_DST_CENTERED, true, INTERP_COSINE, T>(
                                {input.get(), i_strides}, {output.get(), o_strides}, shape_2d,
                                matrix, symmetry, shift, cutoff, normalize, threads);
                    else
                        applyCenteredNormalized2D_<IS_DST_CENTERED, false, INTERP_COSINE, T>(
                                {input.get(), i_strides}, {output.get(), o_strides}, shape_2d,
                                matrix, symmetry, shift, cutoff, normalize, threads);
                });
            default:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    template<Remap REMAP, typename T, typename>
    void transform3D(const shared_t<T[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        if (!symmetry.count())
            return transform3D<REMAP>(input, input_strides, output, output_strides, shape, matrix, shift,
                                      cutoff, interp_mode, stream);

        NOA_ASSERT(!indexing::isOverlap(input.get(), input_strides, output.get(), output_strides, shape.fft()));
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const dim_t threads = stream.threads();
        const bool is_identity = matrix == float33_t{};

        switch (interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue([=](){
                    if (is_identity)
                        applyCenteredNormalized3D_<IS_DST_CENTERED, true, INTERP_NEAREST, T>(
                                {input.get(), input_strides}, {output.get(), output_strides},
                                shape, matrix, symmetry, shift, cutoff, normalize, threads);
                    else
                        applyCenteredNormalized3D_<IS_DST_CENTERED, false, INTERP_NEAREST, T>(
                                {input.get(), input_strides}, {output.get(), output_strides},
                                shape, matrix, symmetry, shift, cutoff, normalize, threads);
                });
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return stream.enqueue([=](){
                    if (is_identity)
                        applyCenteredNormalized3D_<IS_DST_CENTERED, true, INTERP_LINEAR, T>(
                                {input.get(), input_strides}, {output.get(), output_strides},
                                shape, matrix, symmetry, shift, cutoff, normalize, threads);
                    else
                        applyCenteredNormalized3D_<IS_DST_CENTERED, false, INTERP_LINEAR, T>(
                                {input.get(), input_strides}, {output.get(), output_strides},
                                shape, matrix, symmetry, shift, cutoff, normalize, threads);
                });
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return stream.enqueue([=](){
                    if (is_identity)
                        applyCenteredNormalized3D_<IS_DST_CENTERED, true, INTERP_COSINE, T>(
                                {input.get(), input_strides}, {output.get(), output_strides},
                                shape, matrix, symmetry, shift, cutoff, normalize, threads);
                    else
                        applyCenteredNormalized3D_<IS_DST_CENTERED, false, INTERP_COSINE, T>(
                                {input.get(), input_strides}, {output.get(), output_strides},
                                shape, matrix, symmetry, shift, cutoff, normalize, threads);
                });
            default:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    #define NOA_INSTANTIATE_TRANSFORM_(T)                                                                                                                                                           \
    template void transform2D<Remap::HC2HC, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&l);  \
    template void transform2D<Remap::HC2H, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&);    \
    template void transform3D<Remap::HC2HC, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&);   \
    template void transform3D<Remap::HC2H, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&)

    NOA_INSTANTIATE_TRANSFORM_(float);
    NOA_INSTANTIATE_TRANSFORM_(double);
    NOA_INSTANTIATE_TRANSFORM_(cfloat_t);
    NOA_INSTANTIATE_TRANSFORM_(cdouble_t);
}
