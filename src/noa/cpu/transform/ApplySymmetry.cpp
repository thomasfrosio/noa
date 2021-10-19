#include "noa/common/Types.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/math/Arithmetics.h"
#include "noa/cpu/transform/Interpolate.h"
#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/transform/Apply.h"

namespace {
    using namespace ::noa;

    template<typename T, InterpMode INTERP, bool ADD_TO_OUTPUT>
    void applyWithSymmetry_(const T* input, T* output, size2_t shape,
                            float2_t center, float2_t shift, float22_t matrix) {
        cpu::transform::Interpolator2D<T> interp(input, shape, shape.x, 0);
        float2_t coordinates;
        for (size_t y = 0; y < shape.y; ++y) {
            for (size_t x = 0; x < shape.x; ++x, ++output) {
                // Inverse of: shift, and then rotate around the center.
                coordinates = float2_t(x, y);
                coordinates -= center;
                coordinates = matrix * coordinates; // matrix already inverted
                coordinates += center;
                coordinates += shift; // inverse shifts

                T value = interp.template get<INTERP, BORDER_ZERO>(coordinates);
                if constexpr(ADD_TO_OUTPUT)
                    *output += value;
                else
                    *output = value;
            }
        }
    }

    template<typename T, InterpMode INTERP, bool ADD_TO_OUTPUT>
    void applyWithSymmetry_(const T* input, T* output, size3_t shape,
                            float3_t center, float3_t shift, float33_t matrix) {
        cpu::transform::Interpolator3D<T> interp(input, shape, shape.x, 0);
        float3_t coordinates;
        for (size_t z = 0; z < shape.z; ++z) {
            for (size_t y = 0; y < shape.y; ++y) {
                for (size_t x = 0; x < shape.x; ++x, ++output) {
                    // Inverse of: shift, and then rotate around the center.
                    coordinates = float3_t(x, y, z);
                    coordinates -= center;
                    coordinates = matrix * coordinates;
                    coordinates += center;
                    coordinates += shift;

                    T value = interp.template get<INTERP, BORDER_ZERO>(coordinates);
                    if constexpr(ADD_TO_OUTPUT)
                        *output += value;
                    else
                        *output = value;
                }
            }
        }
    }

    template<bool ADD_TO_OUTPUT, typename T, typename SHAPE, typename CENTER, typename MATRIX>
    void launch_(const T* input, T* output, SHAPE shape,
                 CENTER center, CENTER shift, MATRIX matrix, InterpMode interp_mode) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                applyWithSymmetry_<T, INTERP_NEAREST, ADD_TO_OUTPUT>(
                        input, output, shape, center, shift, matrix);
                break;
            case INTERP_LINEAR:
                applyWithSymmetry_<T, INTERP_LINEAR, ADD_TO_OUTPUT>(
                        input, output, shape, center, shift, matrix);
                break;
            case INTERP_COSINE:
                applyWithSymmetry_<T, INTERP_COSINE, ADD_TO_OUTPUT>(
                        input, output, shape, center, shift, matrix);
                break;
            case INTERP_CUBIC:
                applyWithSymmetry_<T, INTERP_CUBIC, ADD_TO_OUTPUT>(
                        input, output, shape, center, shift, matrix);
                break;
            case INTERP_CUBIC_BSPLINE:
                applyWithSymmetry_<T, INTERP_CUBIC_BSPLINE, ADD_TO_OUTPUT>(
                        input, output, shape, center, shift, matrix);
                break;
            default:
                NOA_THROW_FUNC("apply(2|3)D", "The interpolation/filter mode {} is not supported", interp_mode);
        }
    }
}

namespace noa::cpu::transform {
    template<bool PREFILTER, typename T>
    void apply2D(const T* input, T* output, size2_t shape,
                 float2_t shifts, float22_t matrix, const Symmetry& symmetry, float2_t center,
                 InterpMode interp_mode) {
        NOA_PROFILE_FUNCTION();
        const size_t elements = getElements(shape);
        memory::PtrHost<T> buffer;
        const T* tmp;
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            buffer.reset(elements);
            bspline::prefilter2D(input, buffer.get(), shape, 1);
            tmp = buffer.get();
        } else {
            tmp = input;
        }

        // Identity. So only shift/rotate.
        launch_<false>(tmp, output, shape, center, shifts, matrix, interp_mode);

        // Then for the symmetry matrices, pre-multiply with the matrix (rotate then symmetrize).
        // Each rotation adds to the output.
        const float33_t* matrices = symmetry.matrices();
        const uint count = symmetry.count();
        for (uint i = 0; i < count; ++i) {
            float22_t combined(float22_t(matrices[i]) * matrix);
            launch_<true>(tmp, output, shape, center, shifts, combined, interp_mode);
        }

        if (count) {
            using real_t = traits::value_type_t<T>;
            auto scaling = 1 / static_cast<real_t>(count + 1);
            math::multiplyByValue(output, scaling, output, elements);
        }
    }

    template<bool PREFILTER, typename T>
    void apply3D(const T* input, T* output, size3_t shape,
                 float3_t shifts, float33_t matrix, const Symmetry& symmetry, float3_t center,
                 InterpMode interp_mode) {
        NOA_PROFILE_FUNCTION();
        const size_t elements = getElements(shape);
        memory::PtrHost<T> buffer;
        const T* tmp;
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            buffer.reset(elements);
            bspline::prefilter3D(input, buffer.get(), shape, 1);
            tmp = buffer.get();
        } else {
            tmp = input;
        }

        launch_<false>(tmp, output, shape, center, shifts, matrix, interp_mode);

        const float33_t* matrices = symmetry.matrices();
        const uint count = symmetry.count();
        for (uint i = 0; i < count; ++i) {
            float33_t combined(matrices[i] * matrix);
            launch_<true>(tmp, output, shape, center, shifts, combined, interp_mode);
        }

        if (count) {
            using real_t = traits::value_type_t<T>;
            auto scaling = 1 / static_cast<real_t>(count + 1);
            math::multiplyByValue(output, scaling, output, elements);
        }
    }

    #define NOA_INSTANTIATE_APPLY_(T)                                                                                   \
    template void apply2D<true, T>(const T*, T*, size2_t, float2_t, float22_t, const Symmetry&, float2_t, InterpMode);  \
    template void apply3D<true, T>(const T*, T*, size3_t, float3_t, float33_t, const Symmetry&, float3_t, InterpMode);  \
    template void apply2D<false, T>(const T*, T*, size2_t, float2_t, float22_t, const Symmetry&, float2_t, InterpMode); \
    template void apply3D<false, T>(const T*, T*, size3_t, float3_t, float33_t, const Symmetry&, float3_t, InterpMode)

    NOA_INSTANTIATE_APPLY_(float);
    NOA_INSTANTIATE_APPLY_(double);
    NOA_INSTANTIATE_APPLY_(cfloat_t);
    NOA_INSTANTIATE_APPLY_(cdouble_t);
}
