#include "noa/common/Assert.h"
#include "noa/common/Types.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/transform/Interpolator.h"
#include "noa/cpu/transform/Prefilter.h"
#include "noa/cpu/transform/Symmetry.h"

namespace {
    using namespace ::noa;

    template<InterpMode INTERP, typename T>
    void symmetrize_(const T* inputs, size2_t input_pitch,
                     T* outputs, size2_t output_pitch, size2_t shape, size_t batches,
                     const transform::Symmetry& symmetry, float2_t center, bool normalize, size_t threads) {
        const size_t count = symmetry.count();
        const float33_t* matrices = symmetry.matrices();
        const auto scaling = normalize ? 1 / static_cast<traits::value_type_t<T>>(count + 1) : 1;

        size_t iffset = elements(input_pitch);
        size_t offset = elements(output_pitch);
        cpu::transform::Interpolator2D<T> interp(inputs, input_pitch, shape, 0);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(inputs, input_pitch, outputs, output_pitch, shape, batches, center, \
               matrices, count, scaling, iffset, offset, interp)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t y = 0; y < shape.y; ++y) {
                for (size_t x = 0; x < shape.x; ++x) {
                    T value = inputs[batch * iffset + index(x, y, input_pitch.x)];

                    float2_t pos(x, y);
                    pos -= center;
                    for (size_t i = 0; i < count; ++i) {
                        float2_t coordinates = float22_t(matrices[i]) * pos;
                        coordinates += center;
                        value += interp.template get<INTERP, BORDER_ZERO>(coordinates, batch);
                    }
                    outputs[batch * offset + index(x, y, output_pitch.x)] = value * scaling;
                }
            }
        }
    }

    template<InterpMode INTERP, typename T>
    void symmetrize_(const T* inputs, size3_t input_pitch,
                     T* outputs, size3_t output_pitch, size3_t shape, size_t batches,
                     const transform::Symmetry& symmetry, float3_t center, bool normalize, size_t threads) {
        const size_t count = symmetry.count();
        const float33_t* matrices = symmetry.matrices();
        const auto scaling = normalize ? 1 / static_cast<traits::value_type_t<T>>(count + 1) : 1;

        size_t iffset = elements(input_pitch);
        size_t offset = elements(output_pitch);
        cpu::transform::Interpolator3D<T> interp(inputs, input_pitch, shape, 0);

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(inputs, input_pitch, outputs, output_pitch, shape, batches, center, \
               matrices, count, scaling, iffset, offset, interp)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = 0; x < shape.x; ++x) {
                        T value = inputs[batch * iffset + index(x, y, z, input_pitch.x, input_pitch.y)];

                        float3_t pos(x, y, z);
                        pos -= center;
                        for (size_t i = 0; i < count; ++i) {
                            float3_t coordinates = matrices[i] * pos;
                            coordinates += center;
                            value += interp.template get<INTERP, BORDER_ZERO>(coordinates, batch);
                        }
                        outputs[batch * offset + index(x, y, z, output_pitch)] = value * scaling;
                    }
                }
            }
        }
    }

    template<typename T, typename U, typename V>
    void launch_(const T* inputs, U input_pitch, T* outputs, U output_pitch, U shape, size_t batches,
                 const transform::Symmetry& symmetry, V center, InterpMode interp_mode,
                 bool normalize, size_t threads) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                return symmetrize_<INTERP_NEAREST>(inputs, input_pitch, outputs, output_pitch, shape, batches,
                                                   symmetry, center, normalize, threads);
            case INTERP_LINEAR:
                return symmetrize_<INTERP_LINEAR>(inputs, input_pitch, outputs, output_pitch, shape, batches,
                                                  symmetry, center, normalize, threads);
            case INTERP_COSINE:
                return symmetrize_<INTERP_COSINE>(inputs, input_pitch, outputs, output_pitch, shape, batches,
                                                  symmetry, center, normalize, threads);
            case INTERP_CUBIC:
                return symmetrize_<INTERP_CUBIC>(inputs, input_pitch, outputs, output_pitch, shape, batches,
                                                 symmetry, center, normalize, threads);
            case INTERP_CUBIC_BSPLINE:
                return symmetrize_<INTERP_CUBIC_BSPLINE>(inputs, input_pitch, outputs, output_pitch, shape, batches,
                                                         symmetry, center, normalize, threads);
            default:
                NOA_THROW_FUNC("symmetrize(2|3)D", "The interpolation/filter mode {} is not supported", interp_mode);
        }
    }

    template<bool PREFILTER, typename T, typename U, typename V>
    void symmetrizeND_(const T* inputs, U input_pitch, T* outputs, U output_pitch, U shape, size_t batches,
                       const transform::Symmetry& symmetry, V center, InterpMode interp_mode, bool normalize,
                       cpu::Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        // TODO If prefiltering is required, maybe allocate a buffer for ONE batch and compute one batch at a time?
        const size_t threads = stream.threads();
        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            stream.enqueue([=, &symmetry]() {
                if constexpr (std::is_same_v<U, size2_t>) {
                    const size_t nb_inputs = input_pitch.y ? batches : 1;
                    const size2_t buffer_pitch = input_pitch.y ? shape : size2_t{shape.x, 0};
                    cpu::memory::PtrHost<T> buffer(elements(shape) * nb_inputs);
                    cpu::transform::bspline::details::prefilter2D(
                            inputs, input_pitch, buffer.get(), buffer_pitch, shape, batches, threads);
                    launch_(buffer.get(), buffer_pitch, outputs, output_pitch, shape, batches,
                            symmetry, center, interp_mode, normalize, threads);
                } else {
                    const size_t nb_inputs = input_pitch.z ? batches : 1;
                    const size3_t buffer_pitch = input_pitch.z ? shape : size3_t{shape.x, shape.y, 0};
                    cpu::memory::PtrHost<T> buffer(elements(shape) * nb_inputs);
                    cpu::transform::bspline::details::prefilter3D(
                            inputs, input_pitch, buffer.get(), buffer_pitch, shape, batches, threads);
                    launch_(buffer.get(), buffer_pitch, outputs, output_pitch, shape, batches,
                            symmetry, center, interp_mode, normalize, threads);
                }
            });
        } else {
            stream.enqueue([=, &symmetry]() {
                launch_(inputs, input_pitch, outputs, output_pitch, shape, batches,
                        symmetry, center, interp_mode, normalize, threads);
            });
        }
    }
}

namespace noa::cpu::transform {
    template<bool PREFILTER, typename T>
    void symmetrize2D(const T* inputs, size2_t input_pitch, T* outputs, size2_t output_pitch, size2_t shape,
                      size_t batches, const Symmetry& symmetry, float2_t center, InterpMode interp_mode,
                      bool normalize, Stream& stream) {
        if (!symmetry.count())
            return cpu::memory::copy(inputs, {input_pitch, 1}, outputs, {output_pitch, 1}, {shape, 1}, batches, stream);
        symmetrizeND_<PREFILTER>(inputs, input_pitch, outputs, output_pitch, shape, batches,
                                 symmetry, center, interp_mode, normalize, stream);
    }

    template<bool PREFILTER, typename T>
    void symmetrize3D(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                      size_t batches, const Symmetry& symmetry, float3_t center, InterpMode interp_mode,
                      bool normalize, Stream& stream) {
        if (!symmetry.count())
            return cpu::memory::copy(inputs, input_pitch, outputs, output_pitch, shape, batches, stream);
        symmetrizeND_<PREFILTER>(inputs, input_pitch, outputs, output_pitch, shape, batches,
                                 symmetry, center, interp_mode, normalize, stream);
    }

    #define NOA_INSTANTIATE_SYM_(T)                                                                                                             \
    template void symmetrize2D<true, T>(const T*, size2_t, T*, size2_t, size2_t, size_t, const Symmetry&, float2_t, InterpMode, bool, Stream&); \
    template void symmetrize3D<true, T>(const T*, size3_t, T*, size3_t, size3_t, size_t, const Symmetry&, float3_t, InterpMode, bool, Stream&); \
    template void symmetrize2D<false, T>(const T*, size2_t, T*, size2_t, size2_t, size_t, const Symmetry&, float2_t, InterpMode, bool, Stream&);\
    template void symmetrize3D<false, T>(const T*, size3_t, T*, size3_t, size3_t, size_t, const Symmetry&, float3_t, InterpMode, bool, Stream&)

    NOA_INSTANTIATE_SYM_(float);
    NOA_INSTANTIATE_SYM_(double);
    NOA_INSTANTIATE_SYM_(cfloat_t);
    NOA_INSTANTIATE_SYM_(cdouble_t);
}
