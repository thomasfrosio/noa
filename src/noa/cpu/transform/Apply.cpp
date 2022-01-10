#include "noa/common/Assert.h"
#include "noa/common/Types.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/transform/Apply.h"
#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/transform/Prefilter.h"

namespace {
    using namespace ::noa;

    // 2D, 2x3 matrices
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void apply_(const T* input, size2_t input_pitch, size2_t input_shape,
                T* outputs, size2_t output_pitch, size2_t output_shape,
                const float23_t* transforms, size_t batches, T value, size_t threads) {
        const cpu::transform::Interpolator2D<T> interp(input, input_pitch, input_shape, value);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(outputs, output_pitch, output_shape, transforms, batches, interp)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t y = 0; y < output_shape.y; ++y) {
                for (size_t x = 0; x < output_shape.x; ++x) {
                    float2_t coordinates = transforms[batch] * float3_t(x, y, 1.f);
                    outputs[index(x, y, batch, output_pitch.x, output_pitch.y)] =
                            interp.template get<INTERP, BORDER>(coordinates, batch);
                }
            }
        }
    }

    // 2D, 3x3 matrices
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void apply_(const T* input, size2_t input_pitch, size2_t input_shape,
                T* outputs, size2_t output_pitch, size2_t output_shape,
                const float33_t* transforms, size_t batches, T value, size_t threads) {
        const cpu::transform::Interpolator2D<T> interp(input, input_pitch, input_shape, value);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(outputs, output_pitch, output_shape, transforms, batches, interp)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t y = 0; y < output_shape.y; ++y) {
                for (size_t x = 0; x < output_shape.x; ++x) {
                    float23_t transform(transforms[batch]);
                    float3_t v(x, y, 1.f);
                    float2_t coordinates(math::dot(transform[0], v),
                                         math::dot(transform[1], v));
                    outputs[index(x, y, batch, output_pitch.x, output_pitch.y)] =
                            interp.template get<INTERP, BORDER>(coordinates, batch);
                }
            }
        }
    }

    // 3D, 3x4 matrices
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void apply_(const T* input, size3_t input_pitch, size3_t input_shape,
                T* outputs, size3_t output_pitch, size3_t output_shape,
                const float34_t* transforms, size_t batches, T value, size_t threads) {
        const cpu::transform::Interpolator3D<T> interp(input, input_pitch, input_shape, value);
        const size_t offset = elements(output_pitch);

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(outputs, output_pitch, output_shape, transforms, batches, interp, offset)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t z = 0; z < output_shape.z; ++z) {
                for (size_t y = 0; y < output_shape.y; ++y) {
                    for (size_t x = 0; x < output_shape.x; ++x) {
                        float3_t coordinates = transforms[batch] * float4_t(x, y, z, 1.f);
                        T* output = outputs + batch * offset;
                        output[index(x, y, z, output_pitch)] =
                                interp.template get<INTERP, BORDER>(coordinates, batch);
                    }
                }
            }
        }
    }

    // 3D, 4x4 matrices
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void apply_(const T* input, size3_t input_pitch, size3_t input_shape,
                T* outputs, size3_t output_pitch, size3_t output_shape,
                const float44_t* transforms, size_t batches, T value, size_t threads) {
        const cpu::transform::Interpolator3D<T> interp(input, input_pitch, input_shape, value);
        const size_t offset = elements(output_pitch);

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(outputs, output_pitch, output_shape, transforms, batches, interp, offset)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t z = 0; z < output_shape.z; ++z) {
                for (size_t y = 0; y < output_shape.y; ++y) {
                    for (size_t x = 0; x < output_shape.x; ++x) {
                        float4_t v(x, y, z, 1.f);
                        float34_t transform(transforms[batch]);
                        float3_t coordinates(math::dot(transform[0], v),
                                             math::dot(transform[1], v),
                                             math::dot(transform[2], v));
                        T* output = outputs + batch * offset;
                        output[index(x, y, z, output_pitch)] =
                                interp.template get<INTERP, BORDER>(coordinates, batch);
                    }
                }
            }
        }
    }

    template<InterpMode INTERP, typename T, typename U, typename V>
    void launch_(const T* input, U input_pitch, U input_shape,
                 T* outputs, U output_pitch, U output_shape,
                 const V* transforms, size_t batches,
                 T value, BorderMode border_mode, size_t threads) {
        switch (border_mode) {
            case BORDER_ZERO:
                return apply_<INTERP, BORDER_ZERO>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        transforms, batches, value, threads);
            case BORDER_VALUE:
                return apply_<INTERP, BORDER_VALUE>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        transforms, batches, value, threads);
            case BORDER_CLAMP:
                return apply_<INTERP, BORDER_CLAMP>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        transforms, batches, value, threads);
            case BORDER_PERIODIC:
                return apply_<INTERP, BORDER_PERIODIC>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        transforms, batches, value, threads);
            case BORDER_MIRROR:
                return apply_<INTERP, BORDER_MIRROR>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        transforms, batches, value, threads);
            case BORDER_REFLECT:
                return apply_<INTERP, BORDER_REFLECT>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        transforms, batches, value, threads);
            default:
                NOA_THROW_FUNC("apply(2|3)D", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<typename T, typename U, typename V>
    void launch_(const T* input, U input_pitch, U input_shape,
                 T* outputs, U output_pitch, U output_shape,
                 const V* transforms, size_t batches,
                 T value, InterpMode interp_mode, BorderMode border_mode, size_t threads) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                return launch_<INTERP_NEAREST>(input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                                               transforms, batches, value, border_mode, threads);
            case INTERP_LINEAR:
                return launch_<INTERP_LINEAR>(input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                                              transforms, batches, value, border_mode, threads);
            case INTERP_COSINE:
                return launch_<INTERP_COSINE>(input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                                              transforms, batches, value, border_mode, threads);
            case INTERP_CUBIC:
                return launch_<INTERP_CUBIC>(input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                                             transforms, batches, value, border_mode, threads);
            case INTERP_CUBIC_BSPLINE:
                return launch_<INTERP_CUBIC_BSPLINE>(
                        input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        transforms, batches, value, border_mode, threads);
            default:
                NOA_THROW_FUNC("apply(2|3)D", "The interpolation/filter mode {} is not supported", interp_mode);
        }
    }
}

namespace noa::cpu::transform {
    template<bool PREFILTER, typename T, typename MATRIX>
    void apply2D(const T* inputs, size2_t input_pitch, size2_t input_shape,
                 T* outputs, size2_t output_pitch, size2_t output_shape,
                 const MATRIX* transforms, size_t batches,
                 InterpMode interp_mode, BorderMode border_mode,
                 T value, Stream& stream) {
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
                        transforms, batches, value, interp_mode, border_mode, threads);
            });
        } else {
            stream.enqueue([=]() {
                launch_(inputs, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        transforms, batches, value, interp_mode, border_mode, threads);
            });
        }
    }

    template<bool PREFILTER, typename T, typename MATRIX>
    void apply3D(const T* input, size3_t input_pitch, size3_t input_shape,
                 T* outputs, size3_t output_pitch, size3_t output_shape,
                 const MATRIX* transforms, size_t nb_transforms,
                 InterpMode interp_mode, BorderMode border_mode,
                 T value, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != outputs);
        const size_t threads = stream.threads();
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            stream.enqueue([=]() {
                const size_t nb_inputs = input_pitch.z ? nb_transforms : 1;
                const size3_t buffer_pitch = input_pitch.z ? input_shape : size3_t{input_shape.x, input_shape.y, 0};
                memory::PtrHost<T> buffer(elements(input_shape) * nb_inputs);
                bspline::details::prefilter3D(input, input_pitch, buffer.get(), buffer_pitch, input_shape,
                                              nb_inputs, threads);
                launch_(buffer.get(), buffer_pitch, input_shape, outputs, output_pitch, output_shape,
                        transforms, nb_transforms, value, interp_mode, border_mode, threads);
            });
        } else {
            stream.enqueue([=]() {
                launch_(input, input_pitch, input_shape, outputs, output_pitch, output_shape,
                        transforms, nb_transforms, value, interp_mode, border_mode, threads);
            });
        }
    }

    #define NOA_INSTANTIATE_APPLY_(T)                                                                                                                           \
    template void apply2D<true, T, float23_t>(const T*, size2_t, size2_t, T*, size2_t, size2_t, const float23_t*, size_t, InterpMode, BorderMode, T, Stream&);  \
    template void apply2D<false, T, float23_t>(const T*, size2_t, size2_t, T*, size2_t, size2_t, const float23_t*, size_t, InterpMode, BorderMode, T, Stream&); \
    template void apply2D<true, T, float33_t>(const T*, size2_t, size2_t, T*, size2_t, size2_t, const float33_t*, size_t, InterpMode, BorderMode, T, Stream&);  \
    template void apply2D<false, T, float33_t>(const T*, size2_t, size2_t, T*, size2_t, size2_t, const float33_t*, size_t, InterpMode, BorderMode, T, Stream&); \
    template void apply3D<true, T, float34_t>(const T*, size3_t, size3_t, T*, size3_t, size3_t, const float34_t*, size_t, InterpMode, BorderMode, T, Stream&);  \
    template void apply3D<false, T, float34_t>(const T*, size3_t, size3_t, T*, size3_t, size3_t, const float34_t*, size_t, InterpMode, BorderMode, T, Stream&); \
    template void apply3D<true, T, float44_t>(const T*, size3_t, size3_t, T*, size3_t, size3_t, const float44_t*, size_t, InterpMode, BorderMode, T, Stream&);  \
    template void apply3D<false, T, float44_t>(const T*, size3_t, size3_t, T*, size3_t, size3_t, const float44_t*, size_t, InterpMode, BorderMode, T, Stream&)

    NOA_INSTANTIATE_APPLY_(float);
    NOA_INSTANTIATE_APPLY_(double);
    NOA_INSTANTIATE_APPLY_(cfloat_t);
    NOA_INSTANTIATE_APPLY_(cdouble_t);
}
