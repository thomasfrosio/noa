#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/geometry/Interpolator.h"
#include "noa/cpu/geometry/fft/Apply.h"

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
    template<bool IS_DST_CENTERED, bool MULTIPLE_TRANSFORMS, InterpMode INTERP, typename T>
    void transformCenteredNormalized2D_(const T* input, size3_t input_stride,
                                        T* output, size3_t output_stride, size3_t shape,
                                        const float22_t* matrices, const float2_t* shifts,
                                        float cutoff, size_t threads) {
        const size_t offset = input_stride[0];
        const size2_t stride{input_stride.get() + 1};
        const cpu::geometry::Interpolator2D<T> interp(input, stride, size2_t{shape.get() + 1}.fft(), T(0));

        const size_t batches = shape[0];
        const long2_t l_shape{shape.get() + 1};
        const float2_t f_shape{l_shape / 2 * 2 + long2_t{l_shape == 1}}; // if odd, n-1

        [[maybe_unused]] const float2_t pre_shift = math::Constants<float>::PI2 / float2_t{l_shape};

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        #pragma omp parallel for default(none) num_threads(threads) collapse(3) \
        shared(output, output_stride, matrices, shifts, cutoff, batches, l_shape, f_shape, interp, offset, pre_shift)

        for (size_t i = 0; i < batches; ++i) {
            for (int64_t y = 0; y < l_shape[0]; ++y) {
                for (int64_t x = 0; x < l_shape[1] / 2 + 1; ++x) {

                    const float2_t freq{getFrequency_<IS_DST_CENTERED>(y, l_shape[0]), x};
                    float2_t coordinates = freq / f_shape; // [-0.5, 0.5]
                    if (math::dot(coordinates, coordinates) > cutoff) {
                        output[at(i, y, x, output_stride)] = 0;
                        continue;
                    }

                    coordinates = matrices[MULTIPLE_TRANSFORMS * i] * coordinates;
                    using real_t = traits::value_type_t<T>;
                    [[maybe_unused]] real_t conj = 1;
                    if (coordinates[1] < 0.f) {
                        coordinates = -coordinates;
                        if constexpr (traits::is_complex_v<T>)
                            conj = -1;
                    }

                    coordinates[0] += 0.5f; // [0, 1]
                    coordinates *= f_shape; // [0, N-1]
                    T value = interp.template get<INTERP, BORDER_ZERO>(coordinates, i * offset);
                    if constexpr (traits::is_complex_v<T>) {
                        value.imag *= conj;
                        if (shifts)
                            value *= getPhaseShift_<T>(shifts[MULTIPLE_TRANSFORMS * i] * pre_shift, freq);
                    }
                    output[at(i, y, x, output_stride)] = value;
                }
            }
        }
    }

    // 3D, centered input.
    template<bool IS_DST_CENTERED, bool MULTIPLE_TRANSFORMS, InterpMode INTERP, typename T>
    void transformCenteredNormalized3D_(const T* input, size4_t input_stride,
                                        T* output, size4_t output_stride, size4_t shape,
                                        const float33_t* matrices, const float3_t* shifts,
                                        float cutoff, size_t threads) {
        const size_t offset = input_stride[0];
        const size3_t stride{input_stride.get() + 1};
        const cpu::geometry::Interpolator3D<T> interp(input, stride, size3_t{shape.get() + 1}.fft(), T(0));

        const size_t batches = shape[0];
        long3_t l_shape{shape.get() + 1};
        float3_t f_shape{l_shape / 2 * 2 + long3_t{l_shape == 1}}; // if odd, n-1

        [[maybe_unused]] const float3_t pre_shift = math::Constants<float>::PI2 / float3_t{l_shape};

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        #pragma omp parallel for default(none) num_threads(threads) collapse(4) \
        shared(output, output_stride, matrices, shifts, cutoff, batches, l_shape, f_shape, interp, offset, pre_shift)

        for (size_t i = 0; i < batches; ++i) {
            for (int64_t z = 0; z < l_shape[0]; ++z) {
                for (int64_t y = 0; y < l_shape[1]; ++y) {
                    for (int64_t x = 0; x < l_shape[2] / 2 + 1; ++x) {

                        const float3_t freq{getFrequency_<IS_DST_CENTERED>(z, l_shape[0]),
                                            getFrequency_<IS_DST_CENTERED>(y, l_shape[1]),
                                            x};
                        float3_t coordinates = freq / f_shape;
                        if (math::dot(coordinates, coordinates) > cutoff) {
                            output[at(i, z, y, x, output_stride)] = 0;
                            continue;
                        }

                        coordinates = matrices[MULTIPLE_TRANSFORMS * i] * coordinates;
                        using real_t = traits::value_type_t<T>;
                        [[maybe_unused]] real_t conj = 1;
                        if (coordinates[2] < 0.f) {
                            coordinates = -coordinates;
                            if constexpr (traits::is_complex_v<T>)
                                conj = -1;
                        }

                        coordinates[0] += 0.5f;
                        coordinates[1] += 0.5f;
                        coordinates *= f_shape;
                        T value = interp.template get<INTERP, BORDER_ZERO>(coordinates, i * offset);
                        if constexpr (traits::is_complex_v<T>) {
                            value.imag *= conj;
                            if (shifts)
                                value *= getPhaseShift_<T>(shifts[MULTIPLE_TRANSFORMS * i] * pre_shift, freq);
                        }
                        output[at(i, z, y, x, output_stride)] = value;
                    }
                }
            }
        }
    }

    // input FFTs should be centered. The only flexibility is whether the output is centered or not.
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
    void transform2D(const T* input, size4_t input_stride,
                     T* output, size4_t output_stride, size4_t shape,
                     const float22_t* matrices, const float2_t* shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const size_t threads = stream.threads();

        const size3_t i_stride{input_stride[0], input_stride[2], input_stride[3]};
        const size3_t o_stride{output_stride[0], output_stride[2], output_stride[3]};
        const size3_t shape_2d{shape[0], shape[2], shape[3]};

        switch (interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue(transformCenteredNormalized2D_<IS_DST_CENTERED, true, INTERP_NEAREST, T>,
                                      input, i_stride, output, o_stride, shape_2d,
                                      matrices, shifts, cutoff, threads);
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return stream.enqueue(transformCenteredNormalized2D_<IS_DST_CENTERED, true, INTERP_LINEAR, T>,
                                      input, i_stride, output, o_stride, shape_2d,
                                      matrices, shifts, cutoff, threads);
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return stream.enqueue(transformCenteredNormalized2D_<IS_DST_CENTERED, true, INTERP_COSINE, T>,
                                      input, i_stride, output, o_stride, shape_2d,
                                      matrices, shifts, cutoff, threads);
            default:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    template<Remap REMAP, typename T>
    void transform2D(const T* input, size4_t input_stride,
                     T* output, size4_t output_stride, size4_t shape,
                     float22_t matrix, float2_t shift,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const size_t threads = stream.threads();

        const size3_t i_stride{input_stride[0], input_stride[2], input_stride[3]};
        const size3_t o_stride{output_stride[0], output_stride[2], output_stride[3]};
        const size3_t shape_2d{shape[0], shape[2], shape[3]};

        switch (interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue([=]() {
                    transformCenteredNormalized2D_<IS_DST_CENTERED, false, INTERP_NEAREST, T>(
                            input, i_stride, output, o_stride, shape_2d,
                            &matrix, all(shift == 0) ? nullptr : &shift, cutoff, threads);
                });
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return stream.enqueue([=]() {
                    transformCenteredNormalized2D_<IS_DST_CENTERED, false, INTERP_LINEAR, T>(
                            input, i_stride, output, o_stride, shape_2d,
                            &matrix, all(shift == 0) ? nullptr : &shift, cutoff, threads);
                });
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return stream.enqueue([=]() {
                    transformCenteredNormalized2D_<IS_DST_CENTERED, false, INTERP_COSINE, T>(
                            input, i_stride, output, o_stride, shape_2d,
                            &matrix, all(shift == 0) ? nullptr : &shift, cutoff, threads);
                });
            default:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    template<Remap REMAP, typename T>
    void transform3D(const T* input, size4_t input_stride,
                     T* output, size4_t output_stride, size4_t shape,
                     const float33_t* matrices, const float3_t* shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const size_t threads = stream.threads();

        switch (interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue(transformCenteredNormalized3D_<IS_DST_CENTERED, true, INTERP_NEAREST, T>,
                                      input, input_stride, output, output_stride, shape,
                                      matrices, shifts, cutoff, threads);
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return stream.enqueue(transformCenteredNormalized3D_<IS_DST_CENTERED, true, INTERP_LINEAR, T>,
                                      input, input_stride, output, output_stride, shape,
                                      matrices, shifts, cutoff, threads);
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return stream.enqueue(transformCenteredNormalized3D_<IS_DST_CENTERED, true, INTERP_COSINE, T>,
                                      input, input_stride, output, output_stride, shape,
                                      matrices, shifts, cutoff, threads);
            default:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    template<Remap REMAP, typename T>
    void transform3D(const T* input, size4_t input_stride,
                     T* output, size4_t output_stride, size4_t shape,
                     float33_t matrix, float3_t shift,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const size_t threads = stream.threads();

        switch (interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue([=]() {
                    transformCenteredNormalized3D_<IS_DST_CENTERED, false, INTERP_NEAREST, T>(
                            input, input_stride, output, output_stride, shape,
                            &matrix, all(shift == 0) ? nullptr : &shift, cutoff, threads);
                });
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return stream.enqueue([=]() {
                    transformCenteredNormalized3D_<IS_DST_CENTERED, false, INTERP_LINEAR, T>(
                            input, input_stride, output, output_stride, shape,
                            &matrix, all(shift == 0) ? nullptr : &shift, cutoff, threads);
                });
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return stream.enqueue([=]() {
                    transformCenteredNormalized3D_<IS_DST_CENTERED, false, INTERP_COSINE, T>(
                            input, input_stride, output, output_stride, shape,
                            &matrix, all(shift == 0) ? nullptr : &shift, cutoff, threads);
                });
            default:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    #define NOA_INSTANTIATE_TRANSFORM_(T)                                                                                                              \
    template void transform2D<Remap::HC2H, T>(const T*, size4_t, T*, size4_t, size4_t, const float22_t*, const float2_t*, float, InterpMode, Stream&); \
    template void transform2D<Remap::HC2HC, T>(const T*, size4_t, T*, size4_t, size4_t, const float22_t*, const float2_t*, float, InterpMode, Stream&);\
    template void transform2D<Remap::HC2H, T>(const T*, size4_t, T*, size4_t, size4_t, float22_t, float2_t, float, InterpMode, Stream&);               \
    template void transform2D<Remap::HC2HC, T>(const T*, size4_t, T*, size4_t, size4_t, float22_t, float2_t, float, InterpMode, Stream&);              \
    template void transform3D<Remap::HC2H, T>(const T*, size4_t, T*, size4_t, size4_t, const float33_t*, const float3_t*, float, InterpMode, Stream&); \
    template void transform3D<Remap::HC2HC, T>(const T*, size4_t, T*, size4_t, size4_t, const float33_t*, const float3_t*, float, InterpMode, Stream&);\
    template void transform3D<Remap::HC2H, T>(const T*, size4_t, T*, size4_t, size4_t, float33_t, float3_t, float, InterpMode, Stream&);               \
    template void transform3D<Remap::HC2HC, T>(const T*, size4_t, T*, size4_t, size4_t, float33_t, float3_t, float, InterpMode, Stream&)

    NOA_INSTANTIATE_TRANSFORM_(float);
    NOA_INSTANTIATE_TRANSFORM_(double);
    NOA_INSTANTIATE_TRANSFORM_(cfloat_t);
    NOA_INSTANTIATE_TRANSFORM_(cdouble_t);
}
