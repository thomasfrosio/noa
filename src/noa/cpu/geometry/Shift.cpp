#include "noa/common/Exception.h"
#include "noa/common/Types.h"

#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Interpolator.h"
#include "noa/cpu/geometry/Prefilter.h"
#include "noa/cpu/geometry/Shift.h"

// TODO If column-major, we can reorder.
namespace {
    using namespace ::noa;

    // 2D
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void shift_(AccessorRestrict<const T, 3, dim_t> input, dim3_t input_shape,
                AccessorRestrict<T, 3, dim_t> output, dim3_t output_shape,
                const float2_t* shifts, T value, dim_t threads) {
        // Broadcast the input if it is not batched.
        const dim_t offset = input_shape[0] == 1 ? 0 : input.stride(0);
        const dim2_t shape(input_shape.get(1));
        const cpu::geometry::Interpolator2D interp(input[0], shape, value);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(output, output_shape, shifts, offset, interp)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t y = 0; y < output_shape[1]; ++y) {
                for (dim_t x = 0; x < output_shape[2]; ++x) {
                    float2_t coordinates{y, x};
                    coordinates -= shifts[i]; // take the inverse transformation
                    output(i, y, x) = interp.template get<INTERP, BORDER>(coordinates, i * offset);
                }
            }
        }
    }

    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void shift_(AccessorRestrict<const T, 3, dim_t> input, dim3_t input_shape,
                AccessorRestrict<T, 3, dim_t> output, dim3_t output_shape,
                float2_t shift, T value, dim_t threads) {
        // Broadcast the input if it is not batched.
        const dim_t offset = input_shape[0] == 1 ? 0 : input.stride(0);
        const dim2_t shape(input_shape.get(1));
        const cpu::geometry::Interpolator2D interp(input[0], shape, value);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(output, output_shape, shift, offset, interp)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t y = 0; y < output_shape[1]; ++y) {
                for (dim_t x = 0; x < output_shape[2]; ++x) {
                    float2_t coordinates{y, x};
                    coordinates -= shift; // take the inverse transformation
                    output(i, y, x) = interp.template get<INTERP, BORDER>(coordinates, i * offset);
                }
            }
        }
    }

    // 3D
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void shift_(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape,
                const float3_t* shifts, T value, dim_t threads) {
        // Broadcast the input if it is not batched.
        const dim_t offset = input_shape[0] == 1 ? 0 : input.stride(0);
        const dim3_t shape(input_shape.get(1));
        const cpu::geometry::Interpolator3D interp(input[0], shape, value);

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(output, output_shape, shifts, offset, interp)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t z = 0; z < output_shape[1]; ++z) {
                for (dim_t y = 0; y < output_shape[2]; ++y) {
                    for (dim_t x = 0; x < output_shape[3]; ++x) {
                        float3_t coordinates{z, y, x};
                        coordinates -= shifts[i]; // take the inverse transformation
                        output(i, z, y, x) = interp.template get<INTERP, BORDER>(coordinates, i * offset);
                    }
                }
            }
        }
    }

    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void shift_(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape,
                float3_t shift, T value, dim_t threads) {
        // Broadcast the input if it is not batched.
        const dim_t offset = input_shape[0] == 1 ? 0 : input.stride(0);
        const dim3_t shape(input_shape.get(1));
        const cpu::geometry::Interpolator3D interp(input[0], shape, value);

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(output, output_shape, shift, offset, interp)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t z = 0; z < output_shape[1]; ++z) {
                for (dim_t y = 0; y < output_shape[2]; ++y) {
                    for (dim_t x = 0; x < output_shape[3]; ++x) {
                        float3_t coordinates{z, y, x};
                        coordinates -= shift; // take the inverse transformation
                        output(i, z, y, x) = interp.template get<INTERP, BORDER>(coordinates, i * offset);
                    }
                }
            }
        }
    }

    template<InterpMode INTERP, int N, typename value_t, typename dimN_t, typename shift_t>
    void launch_(const AccessorRestrict<const value_t, N, dim_t>& input, dimN_t input_shape,
                 const AccessorRestrict<value_t, N, dim_t>& output, dimN_t output_shape,
                 shift_t shifts, value_t value, BorderMode border_mode, dim_t threads) {
        switch (border_mode) {
            case BORDER_ZERO:
                return shift_<INTERP, BORDER_ZERO>(
                        input, input_shape, output, output_shape,
                        shifts, value, threads);
            case BORDER_VALUE:
                return shift_<INTERP, BORDER_VALUE>(
                        input, input_shape, output, output_shape,
                        shifts, value, threads);
            case BORDER_CLAMP:
                return shift_<INTERP, BORDER_CLAMP>(
                        input, input_shape, output, output_shape,
                        shifts, value, threads);
            case BORDER_PERIODIC:
                return shift_<INTERP, BORDER_PERIODIC>(
                        input, input_shape, output, output_shape,
                        shifts, value, threads);
            case BORDER_MIRROR:
                return shift_<INTERP, BORDER_MIRROR>(
                        input, input_shape, output, output_shape,
                        shifts, value, threads);
            case BORDER_REFLECT:
                return shift_<INTERP, BORDER_REFLECT>(
                        input, input_shape, output, output_shape,
                        shifts, value, threads);
            default:
                NOA_THROW_FUNC("shift(2|3)D", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<int N, typename value_t, typename dimN_t, typename shift_t>
    void launch_(const AccessorRestrict<const value_t, N, dim_t>& input, dimN_t input_shape,
                 const AccessorRestrict<value_t, N, dim_t>& output, dimN_t output_shape,
                 shift_t shifts, value_t value, InterpMode interp_mode, BorderMode border_mode, dim_t threads) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                return launch_<INTERP_NEAREST>(
                        input, input_shape, output, output_shape,
                        shifts, value, border_mode, threads);
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return launch_<INTERP_LINEAR>(
                        input, input_shape, output, output_shape,
                        shifts, value, border_mode, threads);
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return launch_<INTERP_COSINE>(
                        input, input_shape, output, output_shape,
                        shifts, value, border_mode, threads);
            case INTERP_CUBIC:
                return launch_<INTERP_CUBIC>(
                        input, input_shape, output, output_shape,
                        shifts, value, border_mode, threads);
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST:
                return launch_<INTERP_CUBIC_BSPLINE>(
                        input, input_shape, output, output_shape,
                        shifts, value, border_mode, threads);
            default:
                NOA_THROW_FUNC("shift(2|3)D", "The interpolation/filter mode {} is not supported", interp_mode);
        }
    }
}

namespace noa::cpu::geometry {
    template<typename T, typename S, typename>
    void shift2D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                 const S& shifts, InterpMode interp_mode, BorderMode border_mode,
                 T value, bool prefilter, Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(dim3_t(input_shape.get(1)).ndim() <= 2);
        NOA_ASSERT(dim3_t(output_shape.get(1)).ndim() <= 2);

        const dim3_t istrides_2d{input_strides[0], input_strides[2], input_strides[3]};
        const dim3_t ostrides_2d{output_strides[0], output_strides[2], output_strides[3]};
        const dim3_t ishape_2d{input_shape[0], input_shape[2], input_shape[3]};
        const dim3_t oshape_2d{output_shape[0], output_shape[2], output_shape[3]};
        const dim_t threads = stream.threads();

        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                dim4_t shape = input_shape;
                if (input_strides[0] == 0)
                    shape[0] = 1; // there's actually only one input
                const dim4_t strides = shape.strides();
                memory::PtrHost<T> buffer(shape.elements());
                bspline::prefilter(input, input_strides, buffer.share(), strides, shape, stream);

                const dim3_t buffer_strides{strides[0], strides[2], strides[3]};
                if constexpr (traits::is_float2_v<S>) {
                    launch_<3>({buffer.get(), buffer_strides}, ishape_2d,
                               {output.get(), ostrides_2d}, oshape_2d,
                               shifts, value, interp_mode, border_mode, threads);
                } else {
                    NOA_ASSERT(shifts);
                    launch_<3>({buffer.get(), buffer_strides}, ishape_2d,
                               {output.get(), ostrides_2d}, oshape_2d,
                               shifts.get(), value, interp_mode, border_mode, threads);
                }
            });
        } else {
            stream.enqueue([=]() {
                if constexpr (traits::is_float2_v<S>) {
                    launch_<3>({input.get(), istrides_2d}, ishape_2d,
                               {output.get(), ostrides_2d}, oshape_2d,
                               shifts, value, interp_mode, border_mode, threads);
                } else {
                    NOA_ASSERT(shifts);
                    launch_<3>({input.get(), istrides_2d}, ishape_2d,
                               {output.get(), ostrides_2d}, oshape_2d,
                               shifts.get(), value, interp_mode, border_mode, threads);
                }
            });
        }
    }

    template<typename T, typename S, typename>
    void shift3D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                 const S& shifts, InterpMode interp_mode, BorderMode border_mode,
                 T value, bool prefilter, Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

        const dim_t threads = stream.threads();
        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                dim4_t shape = input_shape;
                if (input_strides[0] == 0)
                    shape[0] = 1; // there's actually only one input
                const dim4_t strides = shape.strides();
                memory::PtrHost<T> buffer(shape.elements());
                bspline::prefilter(input, input_strides, buffer.share(), strides, shape, stream);

                if constexpr (traits::is_float3_v<S>) {
                    launch_<4>({buffer.get(), strides}, input_shape,
                               {output.get(), output_strides}, output_shape,
                               shifts, value, interp_mode, border_mode, threads);
                } else {
                    NOA_ASSERT(shifts);
                    launch_<4>({buffer.get(), strides}, input_shape,
                               {output.get(), output_strides}, output_shape,
                               shifts.get(), value, interp_mode, border_mode, threads);
                }
            });
        } else {
            stream.enqueue([=]() {
                if constexpr (traits::is_float3_v<S>) {
                    launch_<4>({input.get(), input_strides}, input_shape,
                               {output.get(), output_strides}, output_shape,
                               shifts, value, interp_mode, border_mode, threads);
                } else {
                    NOA_ASSERT(shifts);
                    launch_<4>({input.get(), input_strides}, input_shape,
                               {output.get(), output_strides}, output_shape,
                               shifts.get(), value, interp_mode, border_mode, threads);
                }
            });
        }
    }

    #define NOA_INSTANTIATE_TRANSLATE_(T)                                                                                                                                                                    \
    template void shift2D<T, shared_t<float2_t[]>, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float2_t[]>&, InterpMode, BorderMode, T, bool, Stream&); \
    template void shift3D<T, shared_t<float3_t[]>, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float3_t[]>&, InterpMode, BorderMode, T, bool, Stream&); \
    template void shift2D<T, float2_t, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const float2_t&, InterpMode, BorderMode, T, bool, Stream&);                         \
    template void shift3D<T, float3_t, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const float3_t&, InterpMode, BorderMode, T, bool, Stream&)

    NOA_INSTANTIATE_TRANSLATE_(float);
    NOA_INSTANTIATE_TRANSLATE_(double);
    NOA_INSTANTIATE_TRANSLATE_(cfloat_t);
    NOA_INSTANTIATE_TRANSLATE_(cdouble_t);
}
