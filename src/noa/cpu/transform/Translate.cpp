#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/memory/PtrHost.h"

namespace {
    using namespace ::noa;

    // 2D
    template<typename T, InterpMode INTERP, BorderMode BORDER>
    void translate_(const T* input, size2_t input_shape, T* outputs, size2_t output_shape,
                    const float2_t* translations, uint nb_translations, T value) {
        cpu::transform::Interpolator2D<T> interp(input, input_shape, input_shape.x, value);
        for (uint i = 0; i < nb_translations; ++i) {
            for (size_t y = 0; y < output_shape.y; ++y) {
                for (size_t x = 0; x < output_shape.x; ++x, ++outputs) {
                    float2_t coordinates(x, y);
                    coordinates -= translations[i]; // take the inverse transformation
                    *outputs = interp.template get<INTERP, BORDER>(coordinates);
                }
            }
        }
    }

    // 3D
    template<typename T, InterpMode INTERP, BorderMode BORDER>
    void translate_(const T* input, size3_t input_shape, T* outputs, size3_t output_shape,
                    const float3_t* translations, uint nb_translations, T value) {
        cpu::transform::Interpolator3D<T> interp(input, input_shape, input_shape.x, value);
        for (uint i = 0; i < nb_translations; ++i) {
            for (size_t z = 0; z < output_shape.z; ++z) {
                for (size_t y = 0; y < output_shape.y; ++y) {
                    for (size_t x = 0; x < output_shape.x; ++x, ++outputs) {
                        float3_t coordinates(x, y, z);
                        coordinates -= translations[i]; // take the inverse transformation
                        *outputs = interp.template get<INTERP, BORDER>(coordinates);
                    }
                }
            }
        }
    }

    template<typename T, typename SHAPE, typename VECTOR, InterpMode INTERP>
    void launch_(const T* input, SHAPE input_shape, T* outputs, SHAPE output_shape,
                 const VECTOR* translations, uint nb_translations, T value, BorderMode border_mode) {
        switch (border_mode) {
            case BORDER_ZERO:
                translate_<T, INTERP, BORDER_ZERO>(
                        input, input_shape, outputs, output_shape, translations, nb_translations, value);
                break;
            case BORDER_VALUE:
                translate_<T, INTERP, BORDER_VALUE>(
                        input, input_shape, outputs, output_shape, translations, nb_translations, value);
                break;
            case BORDER_CLAMP:
                translate_<T, INTERP, BORDER_CLAMP>(
                        input, input_shape, outputs, output_shape, translations, nb_translations, value);
                break;
            case BORDER_PERIODIC:
                translate_<T, INTERP, BORDER_PERIODIC>(
                        input, input_shape, outputs, output_shape, translations, nb_translations, value);
                break;
            case BORDER_MIRROR:
                translate_<T, INTERP, BORDER_MIRROR>(
                        input, input_shape, outputs, output_shape, translations, nb_translations, value);
                break;
            case BORDER_REFLECT:
                translate_<T, INTERP, BORDER_REFLECT>(
                        input, input_shape, outputs, output_shape, translations, nb_translations, value);
                break;
            default:
                NOA_THROW_FUNC("translate(2|3)D", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<typename T, typename SHAPE, typename VECTOR>
    void launch_(const T* input, SHAPE input_shape, T* outputs, SHAPE output_shape,
                 const VECTOR* translations, uint nb_translations, T value,
                 InterpMode interp_mode, BorderMode border_mode) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                launch_<T, SHAPE, VECTOR, INTERP_NEAREST>(
                        input, input_shape, outputs, output_shape, translations, nb_translations, value, border_mode);
                break;
            case INTERP_LINEAR:
                launch_<T, SHAPE, VECTOR, INTERP_LINEAR>(
                        input, input_shape, outputs, output_shape, translations, nb_translations, value, border_mode);
                break;
            case INTERP_COSINE:
                launch_<T, SHAPE, VECTOR, INTERP_COSINE>(
                        input, input_shape, outputs, output_shape, translations, nb_translations, value, border_mode);
                break;
            case INTERP_CUBIC:
                launch_<T, SHAPE, VECTOR, INTERP_CUBIC>(
                        input, input_shape, outputs, output_shape, translations, nb_translations, value, border_mode);
                break;
            case INTERP_CUBIC_BSPLINE:
                launch_<T, SHAPE, VECTOR, INTERP_CUBIC_BSPLINE>(
                        input, input_shape, outputs, output_shape, translations, nb_translations, value, border_mode);
                break;
            default:
                NOA_THROW_FUNC("translate(2|3)D", "The interpolation/filter mode {} is not supported", interp_mode);
        }
    }
}

namespace noa::cpu::transform {
    template<bool PREFILTER, typename T>
    void translate2D(const T* input, size2_t input_shape, T* outputs, size2_t output_shape,
                     const float2_t* translations, uint nb_translations,
                     InterpMode interp_mode, BorderMode border_mode, T value) {
        NOA_PROFILE_FUNCTION();
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            memory::PtrHost<T> tmp(getElements(input_shape));
            bspline::prefilter2D(input, tmp.get(), input_shape, 1);
            launch_(tmp.get(), input_shape, outputs, output_shape,
                    translations, nb_translations, value, interp_mode, border_mode);
        } else {
            launch_(input, input_shape, outputs, output_shape,
                    translations, nb_translations, value, interp_mode, border_mode);
        }
    }

    template<bool PREFILTER, typename T>
    void translate3D(const T* input, size3_t input_shape, T* outputs, size3_t output_shape,
                     const float3_t* translations, uint nb_translations,
                     InterpMode interp_mode, BorderMode border_mode, T value) {
        NOA_PROFILE_FUNCTION();
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            memory::PtrHost<T> tmp(getElements(input_shape));
            bspline::prefilter3D(input, tmp.get(), input_shape, 1);
            launch_(tmp.get(), input_shape, outputs, output_shape,
                    translations, nb_translations, value, interp_mode, border_mode);
        } else {
            launch_(input, input_shape, outputs, output_shape,
                    translations, nb_translations, value, interp_mode, border_mode);
        }
    }

    #define NOA_INSTANTIATE_TRANSLATE_(T)                                                                                   \
    template void translate2D<true, T>(const T*, size2_t, T*, size2_t, const float2_t*, uint, InterpMode, BorderMode, T);   \
    template void translate2D<false, T>(const T*, size2_t, T*, size2_t, const float2_t*, uint, InterpMode, BorderMode, T);  \
    template void translate3D<true, T>(const T*, size3_t, T*, size3_t, const float3_t*, uint, InterpMode, BorderMode, T);   \
    template void translate3D<false, T>(const T*, size3_t, T*, size3_t, const float3_t*, uint, InterpMode, BorderMode, T)

    NOA_INSTANTIATE_TRANSLATE_(float);
    NOA_INSTANTIATE_TRANSLATE_(double);
    NOA_INSTANTIATE_TRANSLATE_(cfloat_t);
    NOA_INSTANTIATE_TRANSLATE_(cdouble_t);
}
