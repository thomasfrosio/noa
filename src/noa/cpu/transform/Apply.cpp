#include "noa/common/Types.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/transform/Interpolate.h"
#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/transform/Apply.h"

namespace {
    using namespace ::noa;

    // 2D, 2x3 matrices
    template<typename T, InterpMode INTERP, BorderMode BORDER>
    void apply_(const T* input, size2_t input_shape, T* outputs, size2_t output_shape,
                const float23_t* transforms, uint nb_transforms, T value) {
        cpu::transform::Interpolator2D<T> interp(input, input_shape, input_shape.x, value);
        float2_t coordinates;
        for (uint i = 0; i < nb_transforms; ++i) {
            for (size_t y = 0; y < output_shape.y; ++y) {
                for (size_t x = 0; x < output_shape.x; ++x, ++outputs) {
                    coordinates = transforms[i] * float3_t(x, y, 1.f);
                    *outputs = interp.template get<INTERP, BORDER>(coordinates);
                }
            }
        }
    }

    // 2D, 3x3 matrices
    template<typename T, InterpMode INTERP, BorderMode BORDER>
    void apply_(const T* input, size2_t input_shape, T* outputs, size2_t output_shape,
                const float33_t* transforms, uint nb_transforms, T value) {
        cpu::transform::Interpolator2D<T> interp(input, input_shape, input_shape.x, value);
        for (uint i = 0; i < nb_transforms; ++i) {
            float23_t transform(transforms[i]);
            for (size_t y = 0; y < output_shape.y; ++y) {
                for (size_t x = 0; x < output_shape.x; ++x, ++outputs) {
                    float3_t v(x, y, 1.f);
                    float2_t coordinates(math::dot(transform[0], v),
                                         math::dot(transform[1], v));
                    *outputs = interp.template get<INTERP, BORDER>(coordinates);
                }
            }
        }
    }

    // 3D, 3x4 matrices
    template<typename T, InterpMode INTERP, BorderMode BORDER>
    void apply_(const T* input, size3_t input_shape, T* outputs, size3_t output_shape,
                const float34_t* transforms, uint nb_transforms, T value) {
        cpu::transform::Interpolator3D<T> interp(input, input_shape, input_shape.x, value);
        float3_t coordinates;
        for (uint i = 0; i < nb_transforms; ++i) {
            for (size_t z = 0; z < output_shape.z; ++z) {
                for (size_t y = 0; y < output_shape.y; ++y) {
                    for (size_t x = 0; x < output_shape.x; ++x, ++outputs) {
                        coordinates = transforms[i] * float4_t(x, y, z, 1.f);
                        *outputs = interp.template get<INTERP, BORDER>(coordinates);
                    }
                }
            }
        }
    }

    // 3D, 4x4 matrices
    template<typename T, InterpMode INTERP, BorderMode BORDER>
    void apply_(const T* input, size3_t input_shape, T* outputs, size3_t output_shape,
                const float44_t* transforms, uint nb_transforms, T value) {
        cpu::transform::Interpolator3D<T> interp(input, input_shape, input_shape.x, value);
        for (uint i = 0; i < nb_transforms; ++i) {
            float34_t transform(transforms[i]);
            for (size_t z = 0; z < output_shape.z; ++z) {
                for (size_t y = 0; y < output_shape.y; ++y) {
                    for (size_t x = 0; x < output_shape.x; ++x, ++outputs) {
                        float4_t v(x, y, z, 1.f);
                        float3_t coordinates(math::dot(transform[0], v),
                                             math::dot(transform[1], v),
                                             math::dot(transform[2], v));
                        *outputs = interp.template get<INTERP, BORDER>(coordinates);
                    }
                }
            }
        }
    }

    template<typename T, typename SHAPE, typename MATRIX, InterpMode INTERP>
    void launch_(const T* input, SHAPE input_shape, T* outputs, SHAPE output_shape,
                 const MATRIX* transforms, uint nb_transforms, T value, BorderMode border_mode) {
        switch (border_mode) {
            case BORDER_ZERO:
                apply_<T, INTERP, BORDER_ZERO>(
                        input, input_shape, outputs, output_shape, transforms, nb_transforms, value);
                break;
            case BORDER_VALUE:
                apply_<T, INTERP, BORDER_VALUE>(
                        input, input_shape, outputs, output_shape, transforms, nb_transforms, value);
                break;
            case BORDER_CLAMP:
                apply_<T, INTERP, BORDER_CLAMP>(
                        input, input_shape, outputs, output_shape, transforms, nb_transforms, value);
                break;
            case BORDER_PERIODIC:
                apply_<T, INTERP, BORDER_PERIODIC>(
                        input, input_shape, outputs, output_shape, transforms, nb_transforms, value);
                break;
            case BORDER_MIRROR:
                apply_<T, INTERP, BORDER_MIRROR>(
                        input, input_shape, outputs, output_shape, transforms, nb_transforms, value);
                break;
            case BORDER_REFLECT:
                apply_<T, INTERP, BORDER_REFLECT>(
                        input, input_shape, outputs, output_shape, transforms, nb_transforms, value);
                break;
            default:
                NOA_THROW_FUNC("apply(2|3)D", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<typename T, typename SHAPE, typename MATRIX>
    void launch_(const T* input, SHAPE input_shape, T* outputs, SHAPE output_shape,
                 const MATRIX* transforms, uint nb_transforms, T value,
                 InterpMode interp_mode, BorderMode border_mode) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                launch_<T, SHAPE, MATRIX, INTERP_NEAREST>(
                        input, input_shape, outputs, output_shape, transforms, nb_transforms, value, border_mode);
                break;
            case INTERP_LINEAR:
                launch_<T, SHAPE, MATRIX, INTERP_LINEAR>(
                        input, input_shape, outputs, output_shape, transforms, nb_transforms, value, border_mode);
                break;
            case INTERP_COSINE:
                launch_<T, SHAPE, MATRIX, INTERP_COSINE>(
                        input, input_shape, outputs, output_shape, transforms, nb_transforms, value, border_mode);
                break;
            case INTERP_CUBIC:
                launch_<T, SHAPE, MATRIX, INTERP_CUBIC>(
                        input, input_shape, outputs, output_shape, transforms, nb_transforms, value, border_mode);
                break;
            case INTERP_CUBIC_BSPLINE:
                launch_<T, SHAPE, MATRIX, INTERP_CUBIC_BSPLINE>(
                        input, input_shape, outputs, output_shape, transforms, nb_transforms, value, border_mode);
                break;
            default:
                NOA_THROW_FUNC("apply(2|3)D", "The interpolation/filter mode {} is not supported", interp_mode);
        }
    }
}

namespace noa::cpu::transform {
    template<bool PREFILTER, typename T, typename MATRIX>
    void apply2D(const T* input, size2_t input_shape, T* outputs, size2_t output_shape,
                 const MATRIX* transforms, uint nb_transforms,
                 InterpMode interp_mode, BorderMode border_mode, T value) {
        NOA_PROFILE_FUNCTION();
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            memory::PtrHost<T> tmp(getElements(input_shape));
            bspline::prefilter2D(input, tmp.get(), input_shape, 1);
            launch_(tmp.get(), input_shape, outputs, output_shape,
                    transforms, nb_transforms, value, interp_mode, border_mode);
        } else {
            launch_(input, input_shape, outputs, output_shape,
                    transforms, nb_transforms, value, interp_mode, border_mode);
        }
    }

    template<bool PREFILTER, typename T, typename MATRIX>
    void apply3D(const T* input, size3_t input_shape, T* outputs, size3_t output_shape,
                 const MATRIX* transforms, uint nb_transforms,
                 InterpMode interp_mode, BorderMode border_mode, T value) {
        NOA_PROFILE_FUNCTION();
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            memory::PtrHost<T> tmp(getElements(input_shape));
            bspline::prefilter3D(input, tmp.get(), input_shape, 1);
            launch_(tmp.get(), input_shape, outputs, output_shape,
                    transforms, nb_transforms, value, interp_mode, border_mode);
        } else {
            launch_(input, input_shape, outputs, output_shape,
                    transforms, nb_transforms, value, interp_mode, border_mode);
        }
    }

    #define NOA_INSTANTIATE_APPLY_(T)                                                                                               \
    template void apply2D<true, T, float23_t>(const T*, size2_t, T*, size2_t, const float23_t*, uint, InterpMode, BorderMode, T);   \
    template void apply2D<false, T, float23_t>(const T*, size2_t, T*, size2_t, const float23_t*, uint, InterpMode, BorderMode, T);  \
    template void apply2D<true, T, float33_t>(const T*, size2_t, T*, size2_t, const float33_t*, uint, InterpMode, BorderMode, T);   \
    template void apply2D<false, T, float33_t>(const T*, size2_t, T*, size2_t, const float33_t*, uint, InterpMode, BorderMode, T);  \
    template void apply3D<true, T, float34_t>(const T*, size3_t, T*, size3_t, const float34_t*, uint, InterpMode, BorderMode, T);   \
    template void apply3D<false, T, float34_t>(const T*, size3_t, T*, size3_t, const float34_t*, uint, InterpMode, BorderMode, T);  \
    template void apply3D<true, T, float44_t>(const T*, size3_t, T*, size3_t, const float44_t*, uint, InterpMode, BorderMode, T);   \
    template void apply3D<false, T, float44_t>(const T*, size3_t, T*, size3_t, const float44_t*, uint, InterpMode, BorderMode, T)

    NOA_INSTANTIATE_APPLY_(float);
    NOA_INSTANTIATE_APPLY_(double);
    NOA_INSTANTIATE_APPLY_(cfloat_t);
    NOA_INSTANTIATE_APPLY_(cdouble_t);
}
