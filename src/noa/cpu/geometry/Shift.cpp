#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Interpolator.h"
#include "noa/cpu/geometry/Prefilter.h"
#include "noa/cpu/geometry/Shift.h"

namespace {
    using namespace ::noa;

    // 2D
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void shift_(const T* input, size3_t input_stride, size3_t input_shape,
                T* output, size3_t output_stride, size3_t output_shape,
                const float2_t* shifts, T value, size_t threads) {
        // Broadcast the input if it is not batched.
        const size_t offset = input_shape[0] == 1 ? 0 : input_stride[0];
        const size2_t stride{input_stride.get() + 1};
        const size2_t shape{input_shape.get() + 1};
        const cpu::geometry::Interpolator2D<T> interp(input, stride, shape, value);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(output, output_stride, output_shape, shifts, offset, interp)

        for (size_t i = 0; i < output_shape[0]; ++i) {
            for (size_t y = 0; y < output_shape[1]; ++y) {
                for (size_t x = 0; x < output_shape[2]; ++x) {
                    float2_t coordinates{y, x};
                    coordinates -= shifts[i]; // take the inverse transformation
                    output[indexing::at(i, y, x, output_stride)] =
                            interp.template get<INTERP, BORDER>(coordinates, i * offset);
                }
            }
        }
    }

    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void shift_(const T* input, size3_t input_stride, size3_t input_shape,
                T* output, size3_t output_stride, size3_t output_shape,
                float2_t shift, T value, size_t threads) {
        // Broadcast the input if it is not batched.
        const size_t offset = input_shape[0] == 1 ? 0 : input_stride[0];
        const size2_t stride{input_stride.get() + 1};
        const size2_t shape{input_shape.get() + 1};
        const cpu::geometry::Interpolator2D<T> interp(input, stride, shape, value);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(output, output_stride, output_shape, shift, offset, interp)

        for (size_t i = 0; i < output_shape[0]; ++i) {
            for (size_t y = 0; y < output_shape[1]; ++y) {
                for (size_t x = 0; x < output_shape[2]; ++x) {
                    float2_t coordinates{y, x};
                    coordinates -= shift; // take the inverse transformation
                    output[indexing::at(i, y, x, output_stride)] =
                            interp.template get<INTERP, BORDER>(coordinates, i * offset);
                }
            }
        }
    }

    // 3D
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void shift_(const T* input, size4_t input_stride, size4_t input_shape,
                T* output, size4_t output_stride, size4_t output_shape,
                const float3_t* shifts, T value, size_t threads) {
        // Broadcast the input if it is not batched.
        const size_t offset = input_shape[0] == 1 ? 0 : input_stride[0];
        const size3_t stride{input_stride.get() + 1};
        const size3_t shape{input_shape.get() + 1};
        const cpu::geometry::Interpolator3D<T> interp(input, stride, shape, value);

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(output, output_stride, output_shape, shifts, offset, interp)

        for (size_t i = 0; i < output_shape[0]; ++i) {
            for (size_t z = 0; z < output_shape[1]; ++z) {
                for (size_t y = 0; y < output_shape[2]; ++y) {
                    for (size_t x = 0; x < output_shape[3]; ++x) {
                        float3_t coordinates{z, y, x};
                        coordinates -= shifts[i]; // take the inverse transformation
                        output[indexing::at(i, z, y, x, output_stride)] =
                                interp.template get<INTERP, BORDER>(coordinates, i * offset);
                    }
                }
            }
        }
    }

    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void shift_(const T* input, size4_t input_stride, size4_t input_shape,
                T* output, size4_t output_stride, size4_t output_shape,
                float3_t shift, T value, size_t threads) {
        // Broadcast the input if it is not batched.
        const size_t offset = input_shape[0] == 1 ? 0 : input_stride[0];
        const size3_t stride{input_stride.get() + 1};
        const size3_t shape{input_shape.get() + 1};
        const cpu::geometry::Interpolator3D<T> interp(input, stride, shape, value);

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(output, output_stride, output_shape, shift, offset, interp)

        for (size_t i = 0; i < output_shape[0]; ++i) {
            for (size_t z = 0; z < output_shape[1]; ++z) {
                for (size_t y = 0; y < output_shape[2]; ++y) {
                    for (size_t x = 0; x < output_shape[3]; ++x) {
                        float3_t coordinates{z, y, x};
                        coordinates -= shift; // take the inverse transformation
                        output[indexing::at(i, z, y, x, output_stride)] =
                                interp.template get<INTERP, BORDER>(coordinates, i * offset);
                    }
                }
            }
        }
    }

    template<InterpMode INTERP, typename T, typename U, typename V>
    void launch_(const T* input, U input_stride, U input_shape,
                 T* output, U output_stride, U output_shape,
                 V shifts, T value, BorderMode border_mode, size_t threads) {
        switch (border_mode) {
            case BORDER_ZERO:
                return shift_<INTERP, BORDER_ZERO>(
                        input, input_stride, input_shape, output, output_stride, output_shape,
                        shifts, value, threads);
            case BORDER_VALUE:
                return shift_<INTERP, BORDER_VALUE>(
                        input, input_stride, input_shape, output, output_stride, output_shape,
                        shifts, value, threads);
            case BORDER_CLAMP:
                return shift_<INTERP, BORDER_CLAMP>(
                        input, input_stride, input_shape, output, output_stride, output_shape,
                        shifts, value, threads);
            case BORDER_PERIODIC:
                return shift_<INTERP, BORDER_PERIODIC>(
                        input, input_stride, input_shape, output, output_stride, output_shape,
                        shifts, value, threads);
            case BORDER_MIRROR:
                return shift_<INTERP, BORDER_MIRROR>(
                        input, input_stride, input_shape, output, output_stride, output_shape,
                        shifts, value, threads);
            case BORDER_REFLECT:
                return shift_<INTERP, BORDER_REFLECT>(
                        input, input_stride, input_shape, output, output_stride, output_shape,
                        shifts, value, threads);
            default:
                NOA_THROW_FUNC("shift(2|3)D", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<typename T, typename U, typename V>
    void launch_(const T* input, U input_stride, U input_shape,
                 T* output, U output_stride, U output_shape,
                 V shifts, T value, InterpMode interp_mode, BorderMode border_mode, size_t threads) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                return launch_<INTERP_NEAREST>(
                        input, input_stride, input_shape, output, output_stride, output_shape,
                        shifts, value, border_mode, threads);
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return launch_<INTERP_LINEAR>(
                        input, input_stride, input_shape, output, output_stride, output_shape,
                        shifts, value, border_mode, threads);
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return launch_<INTERP_COSINE>(
                        input, input_stride, input_shape, output, output_stride, output_shape,
                        shifts, value, border_mode, threads);
            case INTERP_CUBIC:
                return launch_<INTERP_CUBIC>(
                        input, input_stride, input_shape, output, output_stride, output_shape,
                        shifts, value, border_mode, threads);
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST:
                return launch_<INTERP_CUBIC_BSPLINE>(
                        input, input_stride, input_shape, output, output_stride, output_shape,
                        shifts, value, border_mode, threads);
            default:
                NOA_THROW_FUNC("shift(2|3)D", "The interpolation/filter mode {} is not supported", interp_mode);
        }
    }
}

namespace noa::cpu::geometry {
    template<bool PREFILTER, typename T>
    void shift2D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                 const shared_t<float2_t[]>& shifts, InterpMode interp_mode, BorderMode border_mode,
                 T value, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1);

        const size3_t istride_2d{input_stride[0], input_stride[2], input_stride[3]};
        const size3_t ostride_2d{output_stride[0], output_stride[2], output_stride[3]};
        const size3_t ishape_2d{input_shape[0], input_shape[2], input_shape[3]};
        const size3_t oshape_2d{output_shape[0], output_shape[2], output_shape[3]};
        const size_t threads = stream.threads();

        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                size4_t shape = input_shape;
                if (input_stride[0] == 0)
                    shape[0] = 1; // there's actually only one input
                const size4_t stride = shape.stride();
                memory::PtrHost<T> buffer{shape.elements()};
                bspline::prefilter(input, input_stride, buffer.share(), stride, shape, stream);
                launch_(buffer.get(), size3_t{stride[0], stride[2], stride[3]}, ishape_2d,
                        output.get(), ostride_2d, oshape_2d,
                        shifts.get(), value, interp_mode, border_mode, threads);
            });
        } else {
            stream.enqueue([=]() {
                launch_(input.get(), istride_2d, ishape_2d, output.get(), ostride_2d, oshape_2d,
                        shifts.get(), value, interp_mode, border_mode, threads);
            });
        }
    }

    template<bool PREFILTER, typename T>
    void shift2D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                 float2_t shift, InterpMode interp_mode, BorderMode border_mode,
                 T value, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1);

        const size3_t istride_2d{input_stride[0], input_stride[2], input_stride[3]};
        const size3_t ostride_2d{output_stride[0], output_stride[2], output_stride[3]};
        const size3_t ishape_2d{input_shape[0], input_shape[2], input_shape[3]};
        const size3_t oshape_2d{output_shape[0], output_shape[2], output_shape[3]};
        const size_t threads = stream.threads();

        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                size4_t shape = input_shape;
                if (input_stride[0] == 0)
                    shape[0] = 1; // there's actually only one input
                const size4_t stride = shape.stride();
                memory::PtrHost<T> buffer{shape.elements()};
                bspline::prefilter(input, input_stride, buffer.share(), stride, shape, stream);
                launch_(buffer.get(), size3_t{stride[0], stride[2], stride[3]}, ishape_2d,
                        output.get(), ostride_2d, oshape_2d,
                        shift, value, interp_mode, border_mode, threads);
            });
        } else {
            stream.enqueue([=]() {
                launch_(input.get(), istride_2d, ishape_2d, output.get(), ostride_2d, oshape_2d,
                        shift, value, interp_mode, border_mode, threads);
            });
        }
    }

    template<bool PREFILTER, typename T>
    void shift3D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                 const shared_t<float3_t[]>& shifts, InterpMode interp_mode, BorderMode border_mode,
                 T value, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

        const size_t threads = stream.threads();
        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                size4_t shape = input_shape;
                if (input_stride[0] == 0)
                    shape[0] = 1; // there's actually only one input
                const size4_t stride = shape.stride();
                memory::PtrHost<T> buffer{shape.elements()};
                bspline::prefilter(input, input_stride, buffer.share(), stride, shape, stream);
                launch_(buffer.get(), stride, input_shape, output.get(), output_stride, output_shape,
                        shifts.get(), value, interp_mode, border_mode, threads);
            });
        } else {
            stream.enqueue([=]() {
                launch_(input.get(), input_stride, input_shape, output.get(), output_stride, output_shape,
                        shifts.get(), value, interp_mode, border_mode, threads);
            });
        }
    }

    template<bool PREFILTER, typename T>
    void shift3D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                 float3_t shift, InterpMode interp_mode, BorderMode border_mode,
                 T value, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

        const size_t threads = stream.threads();
        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                size4_t shape = input_shape;
                if (input_stride[0] == 0)
                    shape[0] = 1; // there's actually only one input
                const size4_t stride = shape.stride();
                memory::PtrHost<T> buffer{shape.elements()};
                bspline::prefilter(input, input_stride, buffer.share(), stride, shape, stream);
                launch_(buffer.get(), stride, input_shape, output.get(), output_stride, output_shape,
                        shift, value, interp_mode, border_mode, threads);
            });
        } else {
            stream.enqueue([=]() {
                launch_(input.get(), input_stride, input_shape, output.get(), output_stride, output_shape,
                        shift, value, interp_mode, border_mode, threads);
            });
        }
    }

    #define NOA_INSTANTIATE_TRANSLATE_(T)                                                                                                                                               \
    template void shift2D<true, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float2_t[]>&, InterpMode, BorderMode, T, Stream&);    \
    template void shift2D<false, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float2_t[]>&, InterpMode, BorderMode, T, Stream&);   \
    template void shift3D<true, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float3_t[]>&, InterpMode, BorderMode, T, Stream&);    \
    template void shift3D<false, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float3_t[]>&, InterpMode, BorderMode, T, Stream&);   \
    template void shift2D<true, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, InterpMode, BorderMode, T, Stream&);                       \
    template void shift2D<false, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, InterpMode, BorderMode, T, Stream&);                      \
    template void shift3D<true, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, InterpMode, BorderMode, T, Stream&);                       \
    template void shift3D<false, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, InterpMode, BorderMode, T, Stream&)

    NOA_INSTANTIATE_TRANSLATE_(float);
    NOA_INSTANTIATE_TRANSLATE_(double);
    NOA_INSTANTIATE_TRANSLATE_(cfloat_t);
    NOA_INSTANTIATE_TRANSLATE_(cdouble_t);
}
