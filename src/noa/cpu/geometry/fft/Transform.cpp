#include "noa/common/Assert.h"
#include "noa/common/Exception.h"

#include "noa/cpu/geometry/Interpolator.h"
#include "noa/cpu/geometry/fft/Transform.h"

// Note: To support rectangular shapes, the kernels compute the transformation using normalized frequencies.
//       One other solution could have been to use an affine transform encoding the appropriate scaling to effectively
//       normalize the frequencies. Both options are fine and probably equivalent performance-wise.

// FIXME    For even sizes, there's an asymmetry due to the fact that there's only one Nyquist. After fftshift,
//          this extra frequency is on the left side of the axis. However, since we work with non-redundant
//          transforms, there's a slight error that can be introduced. For the elements close (+-1 element) to x=0
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
    void transformCenteredNormalized2D_(const T* input, size3_t input_strides,
                                        T* output, size3_t output_strides, size3_t shape,
                                        const float22_t* matrices, size_t matrices_stride,
                                        const float2_t* shifts, size_t shifts_stride,
                                        float cutoff, size_t threads) {
        const size_t offset = input_strides[0];
        const size2_t strides(input_strides.get(1));
        const cpu::geometry::Interpolator2D<T> interp(input, strides, size2_t(shape.get(1)).fft(), T(0));

        const size_t batches = shape[0];
        const long2_t l_shape(shape.get(1));
        const float2_t f_shape(l_shape / 2 * 2 + long2_t(l_shape == 1)); // if odd, n-1

        [[maybe_unused]] const float2_t pre_shift = math::Constants<float>::PI2 / float2_t(l_shape);

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        #pragma omp parallel for default(none) num_threads(threads) collapse(3)                  \
        shared(output, output_strides, matrices, matrices_stride, shifts, shifts_stride, cutoff, \
               batches, l_shape, f_shape, interp, offset, pre_shift)

        for (size_t i = 0; i < batches; ++i) {
            for (int64_t y = 0; y < l_shape[0]; ++y) {
                for (int64_t x = 0; x < l_shape[1] / 2 + 1; ++x) {

                    const float2_t freq{getFrequency_<IS_DST_CENTERED>(y, l_shape[0]), x};
                    float2_t coordinates = freq / f_shape; // [-0.5, 0.5]
                    if (math::dot(coordinates, coordinates) > cutoff) {
                        output[indexing::at(i, y, x, output_strides)] = 0;
                        continue;
                    }

                    coordinates = matrices[matrices_stride * i] * coordinates;
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
                            value *= getPhaseShift_<T>(shifts[shifts_stride * i] * pre_shift, freq);
                    }
                    output[indexing::at(i, y, x, output_strides)] = value;
                }
            }
        }
    }

    // 3D, centered input.
    template<bool IS_DST_CENTERED, InterpMode INTERP, typename T>
    void transformCenteredNormalized3D_(const T* input, size4_t input_strides,
                                        T* output, size4_t output_strides, size4_t shape,
                                        const float33_t* matrices, size_t matrices_stride,
                                        const float3_t* shifts, size_t shifts_stride,
                                        float cutoff, size_t threads) {
        const size_t offset = input_strides[0];
        const size3_t strides(input_strides.get(1));
        const cpu::geometry::Interpolator3D<T> interp(input, strides, size3_t(shape.get(1)).fft(), T(0));

        const size_t batches = shape[0];
        const long3_t l_shape(shape.get(1));
        const float3_t f_shape(l_shape / 2 * 2 + long3_t(l_shape == 1)); // if odd, n-1

        [[maybe_unused]] const float3_t pre_shift = math::Constants<float>::PI2 / float3_t(l_shape);

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        #pragma omp parallel for default(none) num_threads(threads) collapse(4)                  \
        shared(output, output_strides, matrices, matrices_stride, shifts, shifts_stride, cutoff, \
               batches, l_shape, f_shape, interp, offset, pre_shift)

        for (size_t i = 0; i < batches; ++i) {
            for (int64_t z = 0; z < l_shape[0]; ++z) {
                for (int64_t y = 0; y < l_shape[1]; ++y) {
                    for (int64_t x = 0; x < l_shape[2] / 2 + 1; ++x) {

                        const float3_t freq{getFrequency_<IS_DST_CENTERED>(z, l_shape[0]),
                                            getFrequency_<IS_DST_CENTERED>(y, l_shape[1]),
                                            x};
                        float3_t coordinates = freq / f_shape;
                        if (math::dot(coordinates, coordinates) > cutoff) {
                            output[indexing::at(i, z, y, x, output_strides)] = 0;
                            continue;
                        }

                        coordinates = matrices[matrices_stride * i] * coordinates;
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
                                value *= getPhaseShift_<T>(shifts[shifts_stride * i] * pre_shift, freq);
                        }
                        output[indexing::at(i, z, y, x, output_strides)] = value;
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
    template<Remap REMAP, typename T, typename M, typename S, typename>
    void transform2D(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                     const M& matrices, const S& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(input != output);
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const size_t threads = stream.threads();

        const size3_t i_strides{input_strides[0], input_strides[2], input_strides[3]};
        const size3_t o_strides{output_strides[0], output_strides[2], output_strides[3]};
        const size3_t shape_2d{shape[0], shape[2], shape[3]};

        const float22_t* matrices_;
        const float2_t* shifts_;
        constexpr size_t MATRICES_STRIDE = !traits::is_floatXX_v<M>;
        constexpr size_t SHIFTS_STRIDE = !traits::is_floatX_v<S>;
        if constexpr (MATRICES_STRIDE)
            matrices_ = matrices.get();
        else
            matrices_ = &matrices;
        if constexpr (SHIFTS_STRIDE)
            shifts_ = shifts.get();
        else
            shifts_ = all(shifts == 0) ? nullptr : &shifts;

        switch (interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue([=]() {
                    transformCenteredNormalized2D_<IS_DST_CENTERED, INTERP_NEAREST, T>(
                            input.get(), i_strides, output.get(), o_strides, shape_2d,
                            matrices_, MATRICES_STRIDE, shifts_, SHIFTS_STRIDE, cutoff, threads);
                });
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return stream.enqueue([=]() {
                    transformCenteredNormalized2D_<IS_DST_CENTERED, INTERP_LINEAR, T>(
                            input.get(), i_strides, output.get(), o_strides, shape_2d,
                            matrices_, MATRICES_STRIDE, shifts_, SHIFTS_STRIDE, cutoff, threads);
                });
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return stream.enqueue([=]() {
                    transformCenteredNormalized2D_<IS_DST_CENTERED, INTERP_COSINE, T>(
                            input.get(), i_strides, output.get(), o_strides, shape_2d,
                            matrices_, MATRICES_STRIDE, shifts_, SHIFTS_STRIDE, cutoff, threads);
                });
            default:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    template<Remap REMAP, typename T, typename M, typename S, typename>
    void transform3D(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                     const M& matrices, const S& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(input != output);
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const size_t threads = stream.threads();

        const float33_t* matrices_;
        const float3_t* shifts_;
        constexpr size_t MATRICES_STRIDE = !traits::is_floatXX_v<M>;
        constexpr size_t SHIFTS_STRIDE = !traits::is_floatX_v<S>;
        if constexpr (MATRICES_STRIDE)
            matrices_ = matrices.get();
        else
            matrices_ = &matrices;
        if constexpr (SHIFTS_STRIDE)
            shifts_ = shifts.get();
        else
            shifts_ = all(shifts == 0) ? nullptr : &shifts;

        switch (interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue([=]() {
                    transformCenteredNormalized3D_<IS_DST_CENTERED, INTERP_NEAREST, T>(
                            input.get(), input_strides, output.get(), output_strides, shape,
                            matrices_, MATRICES_STRIDE, shifts_, SHIFTS_STRIDE, cutoff, threads);
                });
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return stream.enqueue([=]() {
                    transformCenteredNormalized3D_<IS_DST_CENTERED, INTERP_LINEAR, T>(
                            input.get(), input_strides, output.get(), output_strides, shape,
                            matrices_, MATRICES_STRIDE, shifts_, SHIFTS_STRIDE, cutoff, threads);
                });
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return stream.enqueue([=]() {
                    transformCenteredNormalized3D_<IS_DST_CENTERED, INTERP_COSINE, T>(
                            input.get(), input_strides, output.get(), output_strides, shape,
                            matrices_, MATRICES_STRIDE, shifts_, SHIFTS_STRIDE, cutoff, threads);
                });
            default:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    #define NOA_INSTANTIATE_TRANSFORM2D_(T, M, S) \
    template void transform2D<Remap::HC2H,  T, M, S, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const M&, const S&, float, InterpMode, Stream&); \
    template void transform2D<Remap::HC2HC, T, M, S, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const M&, const S&, float, InterpMode, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM3D_(T, M, S) \
    template void transform3D<Remap::HC2H,  T, M, S, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const M&, const S&, float, InterpMode, Stream&); \
    template void transform3D<Remap::HC2HC, T, M, S, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const M&, const S&, float, InterpMode, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM2D_ALL_(T)                         \
    NOA_INSTANTIATE_TRANSFORM2D_(T, float22_t, float2_t);               \
    NOA_INSTANTIATE_TRANSFORM2D_(T, shared_t<float22_t[]>, float2_t);   \
    NOA_INSTANTIATE_TRANSFORM2D_(T, float22_t, shared_t<float2_t[]>);   \
    NOA_INSTANTIATE_TRANSFORM2D_(T, shared_t<float22_t[]>, shared_t<float2_t[]>)

    #define NOA_INSTANTIATE_TRANSFORM3D_ALL_(T)                         \
    NOA_INSTANTIATE_TRANSFORM3D_(T, float33_t, float3_t);               \
    NOA_INSTANTIATE_TRANSFORM3D_(T, shared_t<float33_t[]>, float3_t);   \
    NOA_INSTANTIATE_TRANSFORM3D_(T, float33_t, shared_t<float3_t[]>);   \
    NOA_INSTANTIATE_TRANSFORM3D_(T, shared_t<float33_t[]>, shared_t<float3_t[]>)

    #define NOA_INSTANTIATE_TRANSFORM_ALL_(T)   \
    NOA_INSTANTIATE_TRANSFORM2D_ALL_(T);        \
    NOA_INSTANTIATE_TRANSFORM3D_ALL_(T)

    NOA_INSTANTIATE_TRANSFORM_ALL_(float);
    NOA_INSTANTIATE_TRANSFORM_ALL_(double);
    NOA_INSTANTIATE_TRANSFORM_ALL_(cfloat_t);
    NOA_INSTANTIATE_TRANSFORM_ALL_(cdouble_t);
}
