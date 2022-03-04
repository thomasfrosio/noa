#include "noa/common/Assert.h"
#include "noa/common/Types.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Interpolator.h"
#include "noa/cpu/geometry/Prefilter.h"
#include "noa/cpu/geometry/Symmetry.h"

namespace {
    using namespace ::noa;

    template<InterpMode INTERP, typename T>
    void symmetrize_(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                     const geometry::Symmetry& symmetry, float2_t center, bool normalize, size_t threads) {
        const size_t count = symmetry.count();
        const float33_t* matrices = symmetry.matrices();
        const auto scaling = normalize ? 1 / static_cast<traits::value_type_t<T>>(count + 1) : 1;

        const size3_t istride{input_stride[0], input_stride[2], input_stride[3]};
        const size3_t ostride{output_stride[0], output_stride[2], output_stride[3]};
        const cpu::geometry::Interpolator2D<T> interp(
                input, size2_t{istride.get() + 1}, size2_t{shape[2], shape[3]}, 0);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(input, istride, output, ostride, shape, center, matrices, count, scaling, interp)

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t y = 0; y < shape[2]; ++y) {
                for (size_t x = 0; x < shape[3]; ++x) {
                    T value = input[at(i, y, x, istride)];

                    float2_t pos(y, x);
                    pos -= center;
                    for (size_t s = 0; s < count; ++s) {
                        const float33_t& m = matrices[s];
                        float2_t coordinates = float22_t{m[1][1], m[1][2],
                                                         m[2][1], m[2][2]} * pos;
                        coordinates += center;
                        value += interp.template get<INTERP, BORDER_ZERO>(coordinates, i * istride[0]);
                    }
                    output[at(i, y, x, ostride)] = value * scaling;
                }
            }
        }
    }

    template<InterpMode INTERP, typename T>
    void symmetrize_(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                     const geometry::Symmetry& symmetry, float3_t center, bool normalize, size_t threads) {
        const size_t count = symmetry.count();
        const float33_t* matrices = symmetry.matrices();
        const auto scaling = normalize ? 1 / static_cast<traits::value_type_t<T>>(count + 1) : 1;

        const cpu::geometry::Interpolator3D<T> interp(
                input, size3_t{input_stride.get() + 1}, size3_t{shape.get() + 1}, 0);

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(input, input_stride, output, output_stride, shape, center, \
               matrices, count, scaling, interp)

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t z = 0; z < shape[1]; ++z) {
                for (size_t y = 0; y < shape[2]; ++y) {
                    for (size_t x = 0; x < shape[3]; ++x) {
                        T value = input[at(i, z, y, x, input_stride)];

                        float3_t pos{z, y, x};
                        pos -= center;
                        for (size_t s = 0; s < count; ++s) {
                            float3_t coordinates = matrices[s] * pos;
                            coordinates += center;
                            value += interp.template get<INTERP, BORDER_ZERO>(coordinates, i * input_stride[0]);
                        }
                        output[at(i, z, y, x, output_stride)] = value * scaling;
                    }
                }
            }
        }
    }

    template<typename T, typename U, typename V>
    void launch_(const T* input, U input_stride, T* output, U output_stride, U shape,
                 const geometry::Symmetry& symmetry, V center, InterpMode interp_mode,
                 bool normalize, size_t threads) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                return symmetrize_<INTERP_NEAREST>(input, input_stride, output, output_stride, shape,
                                                   symmetry, center, normalize, threads);
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return symmetrize_<INTERP_LINEAR>(input, input_stride, output, output_stride, shape,
                                                  symmetry, center, normalize, threads);
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return symmetrize_<INTERP_COSINE>(input, input_stride, output, output_stride, shape,
                                                  symmetry, center, normalize, threads);
            case INTERP_CUBIC:
                return symmetrize_<INTERP_CUBIC>(input, input_stride, output, output_stride, shape,
                                                 symmetry, center, normalize, threads);
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST:
                return symmetrize_<INTERP_CUBIC_BSPLINE>(input, input_stride, output, output_stride, shape,
                                                         symmetry, center, normalize, threads);
            default:
                NOA_THROW_FUNC("symmetrize(2|3)D", "The interpolation/filter mode {} is not supported", interp_mode);
        }
    }

    template<bool PREFILTER, typename T, typename U, typename V>
    void symmetrizeND_(const T* input, U input_stride, T* output, U output_stride, U shape,
                       const geometry::Symmetry& symmetry, V center, InterpMode interp_mode, bool normalize,
                       cpu::Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        NOA_ASSERT((std::is_same_v<V, float3_t>) || shape[1] == 1);

        if (!symmetry.count())
            return cpu::memory::copy(input, input_stride, output, output_stride, shape, stream);

        const size_t threads = stream.threads();
        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=, &symmetry, &stream]() {
                U new_shape = shape;
                if (input_stride[0] == 0)
                    new_shape[0] = 1; // only one batch in input
                const size4_t stride = new_shape.strides();
                cpu::memory::PtrHost<T> buffer(new_shape.elements());
                cpu::geometry::bspline::prefilter(input, input_stride, buffer.get(), stride, new_shape, stream);
                launch_(buffer.get(), stride, output, output_stride, shape,
                        symmetry, center, interp_mode, normalize, threads);
            });
        } else {
            stream.enqueue([=]() {
                launch_(input, input_stride, output, output_stride, shape,
                        symmetry, center, interp_mode, normalize, threads);
            });
        }
    }
}

namespace noa::cpu::geometry {
    template<bool PREFILTER, typename T>
    void symmetrize2D(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                      const Symmetry& symmetry, float2_t center, InterpMode interp_mode,
                      bool normalize, Stream& stream) {
        symmetrizeND_<PREFILTER>(input, input_stride, output, output_stride, shape,
                                 symmetry, center, interp_mode, normalize, stream);
    }

    template<bool PREFILTER, typename T>
    void symmetrize3D(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                      const Symmetry& symmetry, float3_t center, InterpMode interp_mode,
                      bool normalize, Stream& stream) {
        symmetrizeND_<PREFILTER>(input, input_stride, output, output_stride, shape,
                                 symmetry, center, interp_mode, normalize, stream);
    }

    #define NOA_INSTANTIATE_SYM_(T)                                                                                                     \
    template void symmetrize2D<true, T>(const T*, size4_t, T*, size4_t, size4_t, const Symmetry&, float2_t, InterpMode, bool, Stream&); \
    template void symmetrize3D<true, T>(const T*, size4_t, T*, size4_t, size4_t, const Symmetry&, float3_t, InterpMode, bool, Stream&); \
    template void symmetrize2D<false, T>(const T*, size4_t, T*, size4_t, size4_t, const Symmetry&, float2_t, InterpMode, bool, Stream&);\
    template void symmetrize3D<false, T>(const T*, size4_t, T*, size4_t, size4_t, const Symmetry&, float3_t, InterpMode, bool, Stream&)

    NOA_INSTANTIATE_SYM_(float);
    NOA_INSTANTIATE_SYM_(double);
    NOA_INSTANTIATE_SYM_(cfloat_t);
    NOA_INSTANTIATE_SYM_(cdouble_t);
}
