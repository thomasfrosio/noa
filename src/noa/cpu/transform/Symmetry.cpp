#include "noa/common/Types.h"
#include "noa/common/Exception.h"

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/transform/Interpolate.h"
#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/transform/Symmetry.h"

namespace {
    using namespace ::noa;

    template<typename T, InterpMode INTERP>
    void symmetrize_(const T* input, T* output, size2_t shape, float2_t center,
                     const float33_t* matrices, uint nb_matrices) {
        cpu::transform::Interpolator2D<T> interp(input, shape, shape.x, 0);
        float2_t coordinates;
        for (uint i = 0; i < nb_matrices; ++i) {
            T* tmp = output;
            float22_t sym(matrices[i]);
            for (size_t y = 0; y < shape.y; ++y) {
                for (size_t x = 0; x < shape.x; ++x, ++tmp) {
                    coordinates = float2_t(x, y) - center;
                    coordinates = sym * coordinates;
                    *tmp += interp.template get<INTERP, BORDER_ZERO>(coordinates.x, coordinates.y);
                }
            }
        }
    }

    template<typename T, InterpMode INTERP>
    void symmetrize_(const T* input, T* output, size3_t shape, float3_t center,
                     const float33_t* matrices, uint nb_matrices) {
        cpu::memory::copy(input, output, getElements(shape));
        cpu::transform::Interpolator3D<T> interp(input, shape, shape.x, 0);
        float3_t coordinates;
        for (uint i = 0; i < nb_matrices; ++i) {
            T* tmp = output;
            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = 0; x < shape.x; ++x, ++tmp) {
                        coordinates = float3_t(x, y, z) - center;
                        coordinates = matrices[i] * coordinates;
                        *tmp += interp.template get<INTERP, BORDER_ZERO>(
                                coordinates.x, coordinates.y, coordinates.z);
                    }
                }
            }
        }
    }

    template<typename T, typename SIZE, typename CENTER>
    void launch_(const T* input, T* output, SIZE shape, CENTER center,
                 const float33_t* matrices, uint nb_matrices, InterpMode interp_mode) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                symmetrize_<T, INTERP_NEAREST>(input, output, shape, center, matrices, nb_matrices);
                break;
            case INTERP_LINEAR:
                symmetrize_<T, INTERP_LINEAR>(input, output, shape, center, matrices, nb_matrices);
                break;
            case INTERP_COSINE:
                symmetrize_<T, INTERP_COSINE>(input, output, shape, center, matrices, nb_matrices);
                break;
            case INTERP_CUBIC:
                symmetrize_<T, INTERP_CUBIC>(input, output, shape, center, matrices, nb_matrices);
                break;
            case INTERP_CUBIC_BSPLINE:
                symmetrize_<T, INTERP_CUBIC_BSPLINE>(input, output, shape, center, matrices, nb_matrices);
                break;
            default:
                NOA_THROW_FUNC("symmetrize(2|3)D", "The interpolation/filter mode {} is not supported", interp_mode);
        }
    }
}

namespace noa::cpu::transform {
    template<bool PREFILTER, typename T>
    void symmetrize2D(const T* inputs, T* outputs, size2_t shape, uint batches,
                      Symmetry symmetry, float2_t symmetry_center, InterpMode interp_mode) {
        // The identity is not included, do a copy instead since no interpolation is required.
        size_t elements = getElements(shape);
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE)
            bspline::prefilter2D(inputs, outputs, shape, batches);
        else
            cpu::memory::copy(inputs, outputs, elements * batches);

        const float33_t* matrices = symmetry.getMatrices();
        for (uint batch = 0; batch < batches; ++batch) {
            size_t offset = elements * batch;
            launch_(inputs + offset, outputs + offset, shape, symmetry_center,
                    matrices, symmetry.getCount(), interp_mode);
        }
    }

    template<bool PREFILTER, typename T>
    void symmetrize3D(const T* inputs, T* outputs, size3_t shape, uint batches,
                      Symmetry symmetry, float3_t symmetry_center, InterpMode interp_mode) {
        // The identity is not included, do a copy instead since no interpolation is required.
        size_t elements = getElements(shape);
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE)
            bspline::prefilter3D(inputs, outputs, shape, batches);
        else
            cpu::memory::copy(inputs, outputs, elements * batches);

        const float33_t* matrices = symmetry.getMatrices();
        for (uint batch = 0; batch < batches; ++batch) {
            size_t offset = elements * batch;
            launch_(inputs + offset, outputs + offset, shape, symmetry_center,
                    matrices, symmetry.getCount(), interp_mode);
        }
    }

    #define NOA_INSTANTIATE_SYM_(T)                                                                     \
    template void symmetrize2D<true, T>(const T*, T*, size2_t, uint, Symmetry, float2_t, InterpMode);   \
    template void symmetrize3D<true, T>(const T*, T*, size3_t, uint, Symmetry, float3_t, InterpMode);   \
    template void symmetrize2D<false, T>(const T*, T*, size2_t, uint, Symmetry, float2_t, InterpMode);  \
    template void symmetrize3D<false, T>(const T*, T*, size3_t, uint, Symmetry, float3_t, InterpMode)

    NOA_INSTANTIATE_SYM_(float);
    NOA_INSTANTIATE_SYM_(double);
    NOA_INSTANTIATE_SYM_(cfloat_t);
    NOA_INSTANTIATE_SYM_(cdouble_t);
}
