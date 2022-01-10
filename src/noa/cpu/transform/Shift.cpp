#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/transform/Prefilter.h"
#include "noa/cpu/transform/Shift.h"

namespace {
    using namespace ::noa;

    // 2D
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void shift_(const T* inputs, size2_t input_pitch, size2_t input_shape,
                T* outputs, size2_t output_pitch, size2_t output_shape,
                const float2_t* shifts, size_t batches, T value, size_t threads) {
        const cpu::transform::Interpolator2D<T> interp(inputs, input_pitch, input_shape, value);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(outputs, output_pitch, output_shape, shifts, batches, interp)

        for (size_t i = 0; i < batches; ++i) {
            for (size_t y = 0; y < output_shape.y; ++y) {
                for (size_t x = 0; x < output_shape.x; ++x) {
                    float2_t coordinates(x, y);
                    coordinates -= shifts[i]; // take the inverse transformation
                    outputs[index(x, y, i, output_pitch.x, output_pitch.y)] =
                            interp.template get<INTERP, BORDER>(coordinates, i);
                }
            }
        }
    }

    // 3D
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void shift_(const T* inputs, size3_t input_pitch, size3_t input_shape,
                T* outputs, size3_t output_pitch, size3_t output_shape,
                const float3_t* shifts, size_t batches, T value, size_t threads) {
        const cpu::transform::Interpolator3D<T> interp(inputs, input_pitch, input_shape, value);

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(outputs, output_pitch, output_shape, shifts, batches, interp)

        for (size_t i = 0; i < batches; ++i) {
            for (size_t z = 0; z < output_shape.z; ++z) {
                for (size_t y = 0; y < output_shape.y; ++y) {
                    for (size_t x = 0; x < output_shape.x; ++x) {
                        float3_t coordinates(x, y, z);
                        coordinates -= shifts[i]; // take the inverse transformation
                        T* output = outputs + i * elements(output_pitch);
                        output[index(x, y, z, output_pitch)] =
                                interp.template get<INTERP, BORDER>(coordinates, i);
                    }
                }
            }
        }
    }

    template<InterpMode INTERP, typename T, typename U, typename V>
    void launch_(const T* input, U input_pitch, U input_shape,
                 T* outputs, U output_pitch, U output_shape,
                 const V* shifts, size_t batches,
                 T value, BorderMode border_mode, size_t threads) {
        switch (border_mode) {
            case BORDER_ZERO:
                return shift_<INTERP, BORDER_ZERO>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, threads);
            case BORDER_VALUE:
                return shift_<INTERP, BORDER_VALUE>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, threads);
            case BORDER_CLAMP:
                return shift_<INTERP, BORDER_CLAMP>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, threads);
            case BORDER_PERIODIC:
                return shift_<INTERP, BORDER_PERIODIC>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, threads);
            case BORDER_MIRROR:
                return shift_<INTERP, BORDER_MIRROR>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, threads);
            case BORDER_REFLECT:
                return shift_<INTERP, BORDER_REFLECT>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, threads);
            default:
                NOA_THROW_FUNC("shift(2|3)D", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<typename T, typename U, typename V>
    void launch_(const T* input, U input_pitch, U input_shape,
                 T* outputs, U output_pitch, U output_shape,
                 const V* shifts, size_t batches,
                 T value, InterpMode interp_mode, BorderMode border_mode, size_t threads) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                return launch_<INTERP_NEAREST>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, border_mode, threads);
            case INTERP_LINEAR:
                return launch_<INTERP_LINEAR>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, border_mode, threads);
            case INTERP_COSINE:
                return launch_<INTERP_COSINE>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, border_mode, threads);
            case INTERP_CUBIC:
                return launch_<INTERP_CUBIC>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, border_mode, threads);
            case INTERP_CUBIC_BSPLINE:
                return launch_<INTERP_CUBIC_BSPLINE>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, border_mode, threads);
            default:
                NOA_THROW_FUNC("shift(2|3)D", "The interpolation/filter mode {} is not supported", interp_mode);
        }
    }
}

namespace noa::cpu::transform {
    template<bool PREFILTER, typename T>
    void shift2D(const T* inputs, size2_t input_pitch, size2_t input_shape,
                 T* outputs, size2_t output_pitch, size2_t output_shape,
                 const float2_t* shifts, size_t batches,
                 InterpMode interp_mode, BorderMode border_mode, T value, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        const size_t threads = stream.threads();
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            stream.enqueue([=]() {
                // If there's one input, prepare only one prefiltered input.
                const size_t nb_inputs = input_pitch.y ? batches : 1;
                const size2_t buffer_pitch = input_pitch.y ? input_shape : size2_t{input_shape.x, 0};
                memory::PtrHost<T> buffer(elements(input_shape) * nb_inputs);
                bspline::details::prefilter2D(inputs, input_pitch, buffer.get(), buffer_pitch, input_shape,
                                              nb_inputs, threads);
                launch_(buffer.get(), buffer_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, interp_mode, border_mode, threads);
            });
        } else {
            stream.enqueue([=]() {
                launch_(inputs, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, interp_mode, border_mode, threads);
            });
        }
    }

    template<bool PREFILTER, typename T>
    void shift3D(const T* inputs, size3_t input_pitch, size3_t input_shape,
                 T* outputs, size3_t output_pitch, size3_t output_shape,
                 const float3_t* shifts, size_t batches,
                 InterpMode interp_mode, BorderMode border_mode, T value, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        const size_t threads = stream.threads();
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            stream.enqueue([=]() {
                const size_t nb_inputs = input_pitch.z ? batches : 1;
                const size3_t buffer_pitch = input_pitch.z ? input_shape : size3_t{input_shape.x, input_shape.y, 0};
                memory::PtrHost<T> buffer(elements(input_shape) * nb_inputs);
                bspline::details::prefilter3D(inputs, input_pitch, buffer.get(), buffer_pitch, input_shape,
                                              nb_inputs, threads);
                launch_(buffer.get(), buffer_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, interp_mode, border_mode, threads);
            });
        } else {
            stream.enqueue([=]() {
                launch_(inputs, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        shifts, batches, value, interp_mode, border_mode, threads);
            });
        }
    }

    #define NOA_INSTANTIATE_TRANSLATE_(T)                                                                                                               \
    template void shift2D<true, T>(const T*, size2_t, size2_t, T*, size2_t, size2_t, const float2_t*, size_t, InterpMode, BorderMode, T, Stream&);  \
    template void shift2D<false, T>(const T*, size2_t, size2_t, T*, size2_t, size2_t, const float2_t*, size_t, InterpMode, BorderMode, T, Stream&); \
    template void shift3D<true, T>(const T*, size3_t, size3_t, T*, size3_t, size3_t, const float3_t*, size_t, InterpMode, BorderMode, T, Stream&);  \
    template void shift3D<false, T>(const T*, size3_t, size3_t, T*, size3_t, size3_t, const float3_t*, size_t, InterpMode, BorderMode, T, Stream&)

    NOA_INSTANTIATE_TRANSLATE_(float);
    NOA_INSTANTIATE_TRANSLATE_(double);
    NOA_INSTANTIATE_TRANSLATE_(cfloat_t);
    NOA_INSTANTIATE_TRANSLATE_(cdouble_t);
}
