#include "noa/common/Types.h"
#include "noa/common/Exception.h"

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
                // Go to the rotation center, add shift, then rotate and go back to origin.
                coordinates = float2_t(x, y);
                coordinates -= center + shift;
                coordinates = matrix * coordinates;
                coordinates += center;

                T value = interp.template get<INTERP, BORDER_ZERO>(coordinates.x, coordinates.y);
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
                    // Go to the rotation center, add shift, then rotate and go back to origin.
                    coordinates = float3_t(x, y, z);
                    coordinates -= center + shift;
                    coordinates = matrix * coordinates;
                    coordinates += center;

                    T value = interp.template get<INTERP, BORDER_ZERO>(
                            coordinates.x, coordinates.y, coordinates.z);
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
                 float2_t center, float2_t shifts, float22_t matrix, Symmetry symmetry,
                 InterpMode interp_mode) {
        size_t elements = getElements(shape);
        memory::PtrHost<T> buffer;
        const T* tmp;
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            buffer.reset(elements);
            bspline::prefilter2D(input, buffer.get(), shape, 1);
            tmp = buffer.get();
        } else {
            tmp = input;
        }

        // The identity matrix isn't included in Symmetry. So simply do the rotation.
        // The output isn't set at this point, so don't add but set the output.
        launch_<false>(tmp, output, shape, center, shifts, matrix, interp_mode);

        // Then for the symmetry matrices, pre-multiply with the matrix (rotate then symmetrize).
        // Each rotation adds to the output.
        const float33_t* matrices = symmetry.getMatrices();
        for (uint i = 0; i < symmetry.getCount(); ++i) {
            float22_t combined(matrix * float22_t(matrices[i]));
            launch_<true>(tmp, output, shape, center, shifts, combined, interp_mode);
        }

        using real_t = traits::value_type_t<T>;
        auto scaling = 1 / static_cast<real_t>(symmetry.getCount() + 1);
        math::multiplyByValue(output, scaling, output, elements);
    }

    template<bool PREFILTER, typename T>
    void apply3D(const T* input, T* output, size3_t shape,
                 float3_t center, float3_t shifts, float33_t matrix, Symmetry symmetry,
                 InterpMode interp_mode) {
        size_t elements = getElements(shape);
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

        const float33_t* matrices = symmetry.getMatrices();
        for (uint i = 0; i < symmetry.getCount(); ++i) {
            float33_t combined(matrix * matrices[i]);
            launch_<true>(tmp, output, shape, center, shifts, combined, interp_mode);
        }

        using real_t = traits::value_type_t<T>;
        auto scaling = 1 / static_cast<real_t>(symmetry.getCount() + 1);
        math::multiplyByValue(output, scaling, output, elements);
    }

    #define NOA_INSTANTIATE_APPLY_(T)                                                                               \
    template void apply2D<true, T>(const T*, T*, size2_t, float2_t, float2_t, float22_t, Symmetry, InterpMode);     \
    template void apply3D<true, T>(const T*, T*, size3_t, float3_t, float3_t, float33_t, Symmetry, InterpMode);     \
    template void apply2D<false, T>(const T*, T*, size2_t, float2_t, float2_t, float22_t, Symmetry, InterpMode);    \
    template void apply3D<false, T>(const T*, T*, size3_t, float3_t, float3_t, float33_t, Symmetry, InterpMode)

    NOA_INSTANTIATE_APPLY_(float);
    NOA_INSTANTIATE_APPLY_(double);
    NOA_INSTANTIATE_APPLY_(cfloat_t);
    NOA_INSTANTIATE_APPLY_(cdouble_t);
}
