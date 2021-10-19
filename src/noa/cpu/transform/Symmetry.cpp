#include "noa/common/Types.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/math/Arithmetics.h"
#include "noa/cpu/transform/Interpolate.h"
#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/transform/Symmetry.h"

// Here symmetry matrices are applied one after the other. The other solution would be to loop through all
// symmetric elements in the input and add them up. Not sure if this is more efficient though, mostly because
// of the cache for the input: symmetric positions are very far away from each other; it's a miss everytime.

namespace {
    using namespace ::noa;

    template<typename T, InterpMode INTERP>
    void symmetrize_(const T* input, T* output, size2_t shape, float2_t center,
                     const float33_t* matrices, uint nb_matrices) {
        // We assume the input was already copied into the output.
        cpu::transform::Interpolator2D<T> interp(input, shape, shape.x, 0);
        float2_t coordinates;
        for (uint i = 0; i < nb_matrices; ++i) {
            T* tmp = output;
            float22_t matrix(matrices[i]);
            for (size_t y = 0; y < shape.y; ++y) {
                for (size_t x = 0; x < shape.x; ++x, ++tmp) {
                    coordinates = float2_t(x, y) - center;
                    coordinates = matrix * coordinates;
                    coordinates += center;
                    *tmp += interp.template get<INTERP, BORDER_ZERO>(coordinates);
                }
            }
        }
        // At this point, we should still scale the output. Do it on the main function so that
        // there's only one call for all batches.
    }

    template<typename T, InterpMode INTERP>
    void symmetrize_(const T* input, T* output, size3_t shape, float3_t center,
                     const float33_t* matrices, uint nb_matrices) {
        cpu::transform::Interpolator3D<T> interp(input, shape, shape.x, 0);
        float3_t coordinates;
        for (uint i = 0; i < nb_matrices; ++i) {
            T* tmp = output;
            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = 0; x < shape.x; ++x, ++tmp) {
                        coordinates = float3_t(x, y, z) - center;
                        coordinates = matrices[i] * coordinates;
                        coordinates += center;
                        *tmp += interp.template get<INTERP, BORDER_ZERO>(coordinates);
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

    template<bool PREFILTER, typename T, typename SIZE, typename CENTER>
    void symmetrizeND_(const T* inputs, T* outputs, SIZE shape, uint batches,
                       const transform::Symmetry& symmetry, CENTER center, InterpMode interp_mode) {
        const size_t elements = getElements(shape);
        const uint count = symmetry.count();
        if (!count) // there's no matrices to apply other than the identity
            return cpu::memory::copy(inputs, outputs, elements * batches);

        const float33_t* matrices = symmetry.matrices();

        // If prefiltering is required, allocate a buffer for one batch and compute one batch at a time
        // with this buffer used as input for the interpolation.
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            cpu::memory::PtrHost<T> buffer(elements);
            for (uint batch = 0; batch < batches; ++batch) {
                size_t offset = elements * batch;
                if constexpr (std::is_same_v<SIZE, size2_t>)
                    cpu::transform::bspline::prefilter2D(inputs + offset, buffer.get(), shape, 1);
                else
                    cpu::transform::bspline::prefilter3D(inputs + offset, buffer.get(), shape, 1);
                cpu::memory::copy(buffer.get(), outputs + offset, elements); // identity
                launch_(buffer.get(), outputs + offset, shape, center, matrices, count, interp_mode);
            }
        } else {
            cpu::memory::copy(inputs, outputs, elements * batches); // identity
            for (uint batch = 0; batch < batches; ++batch) {
                size_t offset = elements * batch;
                launch_(inputs + offset, outputs + offset, shape, center, matrices, count, interp_mode);
            }
        }

        using real_t = traits::value_type_t<T>;
        auto scaling = 1 / static_cast<real_t>(count + 1); // + 1 to account for the copy
        cpu::math::multiplyByValue(outputs, scaling, outputs, elements * batches);
    }
}

namespace noa::cpu::transform {
    template<bool PREFILTER, typename T>
    void symmetrize2D(const T* inputs, T* outputs, size2_t shape, uint batches,
                      const Symmetry& symmetry, float2_t center, InterpMode interp_mode) {
        NOA_PROFILE_FUNCTION();
        symmetrizeND_<PREFILTER>(inputs, outputs, shape, batches, symmetry, center, interp_mode);
    }

    template<bool PREFILTER, typename T>
    void symmetrize3D(const T* inputs, T* outputs, size3_t shape, uint batches,
                      const Symmetry& symmetry, float3_t center, InterpMode interp_mode) {
        NOA_PROFILE_FUNCTION();
        symmetrizeND_<PREFILTER>(inputs, outputs, shape, batches, symmetry, center, interp_mode);
    }

    #define NOA_INSTANTIATE_SYM_(T)                                                                             \
    template void symmetrize2D<true, T>(const T*, T*, size2_t, uint, const Symmetry&, float2_t, InterpMode);    \
    template void symmetrize3D<true, T>(const T*, T*, size3_t, uint, const Symmetry&, float3_t, InterpMode);    \
    template void symmetrize2D<false, T>(const T*, T*, size2_t, uint, const Symmetry&, float2_t, InterpMode);   \
    template void symmetrize3D<false, T>(const T*, T*, size3_t, uint, const Symmetry&, float3_t, InterpMode)

    NOA_INSTANTIATE_SYM_(float);
    NOA_INSTANTIATE_SYM_(double);
    NOA_INSTANTIATE_SYM_(cfloat_t);
    NOA_INSTANTIATE_SYM_(cdouble_t);
}
