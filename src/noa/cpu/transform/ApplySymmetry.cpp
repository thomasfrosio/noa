#include "noa/common/Assert.h"
#include "noa/common/Types.h"
#include "noa/common/Exception.h"

#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/transform/Prefilter.h"
#include "noa/cpu/transform/Apply.h"

namespace {
    using namespace ::noa;

    template<InterpMode INTERP, typename T>
    void applyWithSymmetry2D_(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                              float2_t shift, float22_t matrix, const transform::Symmetry& symmetry, float2_t center,
                              bool normalize, size_t threads) {
        const size_t count = symmetry.count();
        cpu::memory::PtrHost<float22_t> buffer(count);
        const float22_t* matrices_combined = buffer.data();
        const float33_t* matrices = symmetry.matrices();
        for (size_t i = 0; i < buffer.size(); ++i)
            buffer[i] = float22_t(matrices[i]) * matrix;

        const cpu::transform::Interpolator2D<T> interp(input, {input_pitch, 0}, shape, 0);
        const float2_t center_shift = center + shift;
        const auto scaling = normalize ? 1 / static_cast<traits::value_type_t<T>>(count + 1) : 1;

        #pragma omp parallel for collapse(2) default(none) num_threads(threads) \
        shared(output, output_pitch, shape, center, matrix, interp, center_shift, count, matrices_combined, scaling)

        for (size_t y = 0; y < shape.y; ++y) {
            for (size_t x = 0; x < shape.x; ++x) {
                float2_t pos(x, y);

                pos -= center;
                float2_t coordinates = matrix * pos; // matrix already inverted
                coordinates += center_shift; // inverse shifts
                T value = interp.template get<INTERP, BORDER_ZERO>(coordinates);

                for (size_t i = 0; i < count; ++i) {
                    coordinates = matrices_combined[i] * pos;
                    coordinates += center_shift;
                    value += interp.template get<INTERP, BORDER_ZERO>(coordinates);
                }

                output[index(x, y, output_pitch)] = value * scaling;
            }
        }
    }

    template<InterpMode INTERP, typename T>
    void applyWithSymmetry3D_(const T* input, size2_t input_pitch, T* output, size2_t output_pitch, size3_t shape,
                              float3_t shift, float33_t matrix, const transform::Symmetry& symmetry, float3_t center,
                              bool normalize, size_t threads) {
        const size_t count = symmetry.count();
        cpu::memory::PtrHost<float33_t> buffer(count);
        const float33_t* matrices_combined = buffer.data();
        const float33_t* matrices = symmetry.matrices();
        for (size_t i = 0; i < buffer.size(); ++i)
            buffer[i] = matrices[i] * matrix;

        const cpu::transform::Interpolator3D<T> interp(input, {input_pitch, 0}, shape, 0);
        const float3_t center_shift = center + shift;
        const auto scaling = normalize ? 1 / static_cast<traits::value_type_t<T>>(count + 1) : 1;

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(output, output_pitch, shape, center, matrix, interp, center_shift, count, matrices_combined, scaling)

        for (size_t z = 0; z < shape.z; ++z) {
            for (size_t y = 0; y < shape.y; ++y) {
                for (size_t x = 0; x < shape.x; ++x) {
                    float3_t pos(x, y, z);

                    pos -= center;
                    float3_t coordinates = matrix * pos;
                    coordinates += center_shift;
                    T value = interp.template get<INTERP, BORDER_ZERO>(coordinates);

                    for (size_t i = 0; i < count; ++i) {
                        coordinates = matrices_combined[i] * pos;
                        coordinates += center_shift;
                        value += interp.template get<INTERP, BORDER_ZERO>(coordinates);
                    }

                    output[index(x, y, z, output_pitch.x, output_pitch.y)] = value * scaling;
                }
            }
        }
    }
}

namespace noa::cpu::transform {
    template<bool PREFILTER, typename T>
    void apply2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                 float2_t shifts, float22_t matrix, const Symmetry& symmetry, float2_t center,
                 InterpMode interp_mode, bool normalize, Stream& stream) {
        NOA_ASSERT(input != output);

        const size_t threads = stream.threads();
        stream.enqueue([=, &symmetry]() {
            memory::PtrHost<T> buffer;
            const T* tmp;
            size_t tmp_pitch;
            if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
                buffer.reset(elements(shape));
                bspline::details::prefilter2D(input, {input_pitch, shape.y}, buffer.get(), shape, shape, 1, threads);
                tmp = buffer.get();
                tmp_pitch = shape.x;
            } else {
                tmp = input;
                tmp_pitch = input_pitch;
            }
            switch (interp_mode) {
                case INTERP_NEAREST:
                    return applyWithSymmetry2D_<INTERP_NEAREST, T>(
                            tmp, tmp_pitch, output, output_pitch, shape,
                            shifts, matrix, symmetry, center, normalize, threads);
                case INTERP_LINEAR:
                    return applyWithSymmetry2D_<INTERP_LINEAR, T>(
                            tmp, tmp_pitch, output, output_pitch, shape,
                            shifts, matrix, symmetry, center, normalize, threads);
                case INTERP_COSINE:
                    return applyWithSymmetry2D_<INTERP_COSINE, T>(
                            tmp, tmp_pitch, output, output_pitch, shape,
                            shifts, matrix, symmetry, center, normalize, threads);
                case INTERP_CUBIC:
                    return applyWithSymmetry2D_<INTERP_CUBIC, T>(
                            tmp, tmp_pitch, output, output_pitch, shape,
                            shifts, matrix, symmetry, center, normalize, threads);
                case INTERP_CUBIC_BSPLINE:
                    return applyWithSymmetry2D_<INTERP_CUBIC_BSPLINE, T>(
                            tmp, tmp_pitch, output, output_pitch, shape,
                            shifts, matrix, symmetry, center, normalize, threads);
                default:
                    NOA_THROW("The interpolation/filter mode {} is not supported", interp_mode);
            }
        });
    }

    template<bool PREFILTER, typename T>
    void apply3D(const T* input, size2_t input_pitch, T* output, size2_t output_pitch, size3_t shape,
                 float3_t shifts, float33_t matrix, const Symmetry& symmetry, float3_t center,
                 InterpMode interp_mode, bool normalize, Stream& stream) {
        NOA_ASSERT(input != output);

        const size_t threads = stream.threads();
        stream.enqueue([=, &symmetry]() {
            memory::PtrHost<T> buffer;
            const T* tmp;
            size2_t tmp_pitch;
            if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
                buffer.reset(elements(shape));
                bspline::details::prefilter3D(input, {input_pitch, shape.z}, buffer.get(), shape, shape, 1, threads);
                tmp = buffer.get();
                tmp_pitch = {shape.x, shape.y};
            } else {
                tmp = input;
                tmp_pitch = input_pitch;
            }
            switch (interp_mode) {
                case INTERP_NEAREST:
                    return applyWithSymmetry3D_<INTERP_NEAREST, T>(
                            tmp, tmp_pitch, output, output_pitch, shape,
                            shifts, matrix, symmetry, center, normalize, threads);
                case INTERP_LINEAR:
                    return applyWithSymmetry3D_<INTERP_LINEAR, T>(
                            tmp, tmp_pitch, output, output_pitch, shape,
                            shifts, matrix, symmetry, center, normalize, threads);
                case INTERP_COSINE:
                    return applyWithSymmetry3D_<INTERP_COSINE, T>(
                            tmp, tmp_pitch, output, output_pitch, shape,
                            shifts, matrix, symmetry, center, normalize, threads);
                case INTERP_CUBIC:
                    return applyWithSymmetry3D_<INTERP_CUBIC, T>(
                            tmp, tmp_pitch, output, output_pitch, shape,
                            shifts, matrix, symmetry, center, normalize, threads);
                case INTERP_CUBIC_BSPLINE:
                    return applyWithSymmetry3D_<INTERP_CUBIC_BSPLINE, T>(
                            tmp, tmp_pitch, output, output_pitch, shape,
                            shifts, matrix, symmetry, center, normalize, threads);
                default:
                    NOA_THROW("The interpolation/filter mode {} is not supported", interp_mode);
            }
        });
    }

    #define NOA_INSTANTIATE_APPLY_(T)                                                                                                                   \
    template void apply2D<true, T>(const T*, size_t, T*, size_t, size2_t, float2_t, float22_t, const Symmetry&, float2_t, InterpMode, bool, Stream&);   \
    template void apply3D<true, T>(const T*, size2_t, T*, size2_t, size3_t, float3_t, float33_t, const Symmetry&, float3_t, InterpMode, bool, Stream&); \
    template void apply2D<false, T>(const T*, size_t, T*, size_t, size2_t, float2_t, float22_t, const Symmetry&, float2_t, InterpMode, bool, Stream&);  \
    template void apply3D<false, T>(const T*, size2_t, T*, size2_t, size3_t, float3_t, float33_t, const Symmetry&, float3_t, InterpMode, bool, Stream&)

    NOA_INSTANTIATE_APPLY_(float);
    NOA_INSTANTIATE_APPLY_(double);
    NOA_INSTANTIATE_APPLY_(cfloat_t);
    NOA_INSTANTIATE_APPLY_(cdouble_t);
}
