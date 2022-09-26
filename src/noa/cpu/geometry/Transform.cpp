#include "noa/common/Assert.h"
#include "noa/common/Types.h"
#include "noa/common/Exception.h"

#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Transform.h"
#include "noa/cpu/geometry/Interpolator.h"
#include "noa/cpu/geometry/Prefilter.h"

namespace {
    using namespace ::noa;

    // 2D, 2x3 matrices, strides/shape: B,Y,X
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void transform_(AccessorRestrict<const T, 3, dim_t> input, dim3_t input_shape,
                    AccessorRestrict<T, 3, dim_t> output, dim3_t output_shape,
                    const float23_t* matrices, T value, dim_t threads) {
        // Broadcast the input if it is not batched.
        const dim_t offset = input_shape[0] == 1 ? 0 : input.stride(0);
        const dim2_t shape(input_shape.get(1));
        const cpu::geometry::Interpolator2D interp(input[0], shape, value);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(output, output_shape, matrices, offset, interp)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t y = 0; y < output_shape[1]; ++y) {
                for (dim_t x = 0; x < output_shape[2]; ++x) {
                    const float2_t coordinates = matrices[i] * float3_t{y, x, 1.f};
                    output(i, y, x) = interp.template get<INTERP, BORDER>(coordinates, offset * i);
                }
            }
        }
    }

    // 2D, 2x3 matrix, strides/shape: B,Y,X
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void transform_(AccessorRestrict<const T, 3, dim_t> input, dim3_t input_shape,
                    AccessorRestrict<T, 3, dim_t> output, dim3_t output_shape,
                    float23_t matrix, T value, dim_t threads) {
        // Broadcast the input if it is not batched.
        const dim_t offset = input_shape[0] == 1 ? 0 : input.stride(0);
        const dim2_t shape(input_shape.get(1));
        const cpu::geometry::Interpolator2D interp(input[0], shape, value);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(output, output_shape, matrix, offset, interp)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t y = 0; y < output_shape[1]; ++y) {
                for (dim_t x = 0; x < output_shape[2]; ++x) {
                    const float2_t coordinates = matrix * float3_t{y, x, 1.f};
                    output(i, y, x) = interp.template get<INTERP, BORDER>(coordinates, offset * i);
                }
            }
        }
    }

    // 2D, 3x3 matrices, strides/shape: B,Y,X
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void transform_(AccessorRestrict<const T, 3, dim_t> input, dim3_t input_shape,
                    AccessorRestrict<T, 3, dim_t> output, dim3_t output_shape,
                    const float33_t* matrices, T value, dim_t threads) {
        // Broadcast the input if it is not batched.
        const dim_t offset = input_shape[0] == 1 ? 0 : input.stride(0);
        const dim2_t shape(input_shape.get(1));
        const cpu::geometry::Interpolator2D interp(input[0], shape, value);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(output, output_shape, matrices, offset, interp)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t y = 0; y < output_shape[1]; ++y) {
                for (dim_t x = 0; x < output_shape[2]; ++x) {
                    const float23_t matrix(matrices[i]);
                    const float3_t v{y, x, 1.f};
                    const float2_t coordinates{math::dot(matrix[0], v),
                                               math::dot(matrix[1], v)};
                    output(i, y, x) = interp.template get<INTERP, BORDER>(coordinates, offset * i);
                }
            }
        }
    }

    // 3D, 3x4 matrices, strides/shape: B,Z,Y,X
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void transform_(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                    AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape,
                    const float34_t* matrices, T value, dim_t threads) {
        // Broadcast the input if it is not batched.
        const dim_t offset = input_shape[0] == 1 ? 0 : input.stride(0);
        const dim3_t shape(input_shape.get(1));
        const cpu::geometry::Interpolator3D interp(input[0], shape, value);

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(output, output_shape, matrices, offset, interp)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t z = 0; z < output_shape[1]; ++z) {
                for (dim_t y = 0; y < output_shape[2]; ++y) {
                    for (dim_t x = 0; x < output_shape[3]; ++x) {
                        const float3_t coordinates = matrices[i] * float4_t{z, y, x, 1.f};
                        output(i, z, y, x) = interp.template get<INTERP, BORDER>(coordinates, offset * i);
                    }
                }
            }
        }
    }

    // 3D, 3x4 matrix, strides/shape: B,Z,Y,X
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void transform_(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                    AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape,
                    float34_t matrix, T value, dim_t threads) {
        // Broadcast the input if it is not batched.
        const dim_t offset = input_shape[0] == 1 ? 0 : input.stride(0);
        const dim3_t shape(input_shape.get(1));
        const cpu::geometry::Interpolator3D interp(input[0], shape, value);

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(output, output_shape, matrix, offset, interp)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t z = 0; z < output_shape[1]; ++z) {
                for (dim_t y = 0; y < output_shape[2]; ++y) {
                    for (dim_t x = 0; x < output_shape[3]; ++x) {
                        const float3_t coordinates = matrix * float4_t{z, y, x, 1.f};
                        output(i, z, y, x) = interp.template get<INTERP, BORDER>(coordinates, offset * i);
                    }
                }
            }
        }
    }

    // 3D, 4x4 matrices, strides/shape: B,Z,Y,X
    template<InterpMode INTERP, BorderMode BORDER, typename T>
    void transform_(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                    AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape,
                    const float44_t* matrices, T value, dim_t threads) {
        // Broadcast the input if it is not batched.
        const dim_t offset = input_shape[0] == 1 ? 0 : input.stride(0);
        const dim3_t shape(input_shape.get(1));
        const cpu::geometry::Interpolator3D interp(input[0], shape, value);

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(output, output_shape, matrices, offset, interp)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t z = 0; z < output_shape[1]; ++z) {
                for (dim_t y = 0; y < output_shape[2]; ++y) {
                    for (dim_t x = 0; x < output_shape[3]; ++x) {
                        const float4_t v{z, y, x, 1.f};
                        const float34_t matrix(matrices[i]);
                        const float3_t coordinates{math::dot(matrix[0], v),
                                                   math::dot(matrix[1], v),
                                                   math::dot(matrix[2], v)};
                        output(i, z, y, x) = interp.template get<INTERP, BORDER>(coordinates, offset * i);
                    }
                }
            }
        }
    }

    template<InterpMode INTERP, int N, typename T, typename U, typename V>
    void launch_(const AccessorRestrict<const T, N, dim_t>& input, U input_shape,
                 const AccessorRestrict<T, N, dim_t>& output, U output_shape,
                 V matrices, T value, BorderMode border_mode, dim_t threads) {
        switch (border_mode) {
            case BORDER_ZERO:
                return transform_<INTERP, BORDER_ZERO>(
                        input, input_shape, output, output_shape,
                        matrices, value, threads);
            case BORDER_VALUE:
                return transform_<INTERP, BORDER_VALUE>(
                        input, input_shape, output, output_shape,
                        matrices, value, threads);
            case BORDER_CLAMP:
                return transform_<INTERP, BORDER_CLAMP>(
                        input, input_shape, output, output_shape,
                        matrices, value, threads);
            case BORDER_PERIODIC:
                return transform_<INTERP, BORDER_PERIODIC>(
                        input, input_shape, output, output_shape,
                        matrices, value, threads);
            case BORDER_MIRROR:
                return transform_<INTERP, BORDER_MIRROR>(
                        input, input_shape, output, output_shape,
                        matrices, value, threads);
            case BORDER_REFLECT:
                return transform_<INTERP, BORDER_REFLECT>(
                        input, input_shape, output, output_shape,
                        matrices, value, threads);
            default:
                NOA_THROW_FUNC("transform(2|3)D", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<int N, typename T, typename U, typename V>
    void launch_(const AccessorRestrict<const T, N, dim_t>& input, U input_shape,
                 const AccessorRestrict<T, N, dim_t>& output, U output_shape,
                 V matrices, T value, InterpMode interp_mode, BorderMode border_mode, dim_t threads) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                return launch_<INTERP_NEAREST>(input, input_shape, output, output_shape,
                                               matrices, value, border_mode, threads);
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return launch_<INTERP_LINEAR>(input, input_shape, output, output_shape,
                                              matrices, value, border_mode, threads);
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return launch_<INTERP_COSINE>(input, input_shape, output, output_shape,
                                              matrices, value, border_mode, threads);
            case INTERP_CUBIC:
                return launch_<INTERP_CUBIC>(input, input_shape, output, output_shape,
                                             matrices, value, border_mode, threads);
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST:
                return launch_<INTERP_CUBIC_BSPLINE>(input, input_shape, output, output_shape,
                                                     matrices, value, border_mode, threads);
            default:
                NOA_THROW_FUNC("transform(2|3)D", "The interpolation/filter mode {} is not supported", interp_mode);
        }
    }
}

namespace noa::cpu::geometry {
    template<typename T, typename MAT, typename>
    void transform2D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const MAT& matrices, InterpMode interp_mode, BorderMode border_mode,
                     T value, bool prefilter, Stream& stream) {
        NOA_ASSERT(!indexing::isOverlap(input.get(), input_strides, input_shape,
                                        output.get(), output_strides, output_shape));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1);

        const dim3_t istrides_2d{input_strides[0], input_strides[2], input_strides[3]};
        const dim3_t ostrides_2d{output_strides[0], output_strides[2], output_strides[3]};
        const dim3_t ishape_2d{input_shape[0], input_shape[2], input_shape[3]};
        const dim3_t oshape_2d{output_shape[0], output_shape[2], output_shape[3]};
        const dim_t threads = stream.threads();

        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                dim4_t shape = input_shape;
                if (input_strides[0] == 0)
                    shape[0] = 1;
                const dim4_t strides = shape.strides();
                memory::PtrHost<T> buffer(shape.elements());
                bspline::prefilter(input, input_strides, buffer.share(), strides, shape, stream);

                const dim3_t strides_2d{strides[0], strides[2], strides[3]};
                if constexpr (traits::is_floatXX_v<MAT>) {
                    launch_<3>({buffer.get(), strides_2d}, ishape_2d,
                               {output.get(), ostrides_2d}, oshape_2d,
                               float23_t(matrices), value, interp_mode, border_mode, threads);
                } else {
                    launch_<3>({buffer.get(), strides_2d}, ishape_2d,
                               {output.get(), ostrides_2d}, oshape_2d,
                               matrices.get(), value, interp_mode, border_mode, threads);
                }
            });
        } else {
            stream.enqueue([=]() {
                if constexpr (traits::is_floatXX_v<MAT>) {
                    launch_<3>({input.get(), istrides_2d}, ishape_2d,
                               {output.get(), ostrides_2d}, oshape_2d,
                               float23_t(matrices), value, interp_mode, border_mode, threads);
                } else {
                    launch_<3>({input.get(), istrides_2d}, ishape_2d,
                               {output.get(), ostrides_2d}, oshape_2d,
                               matrices.get(), value, interp_mode, border_mode, threads);
                }
            });
        }
    }

    template<typename T, typename MAT, typename>
    void transform3D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const MAT& matrices, InterpMode interp_mode, BorderMode border_mode,
                     T value, bool prefilter, Stream& stream) {
        NOA_ASSERT(!indexing::isOverlap(input.get(), input_strides, input_shape,
                                        output.get(), output_strides, output_shape));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

        const dim_t threads = stream.threads();
        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                dim4_t shape = input_shape;
                if (input_strides[0] == 0)
                    shape[0] = 1;
                const dim4_t strides = shape.strides();
                memory::PtrHost<T> buffer(shape.elements());
                bspline::prefilter(input, input_strides, buffer.share(), strides, shape, stream);

                if constexpr (traits::is_floatXX_v<MAT>) {
                    launch_<4>({buffer.get(), strides}, input_shape,
                               {output.get(), output_strides}, output_shape,
                               float34_t(matrices), value, interp_mode, border_mode, threads);
                } else {
                    launch_<4>({buffer.get(), strides}, input_shape,
                               {output.get(), output_strides}, output_shape,
                               matrices.get(), value, interp_mode, border_mode, threads);
                }
            });
        } else {
            stream.enqueue([=]() {
                if constexpr (traits::is_floatXX_v<MAT>) {
                    launch_<4>({input.get(), input_strides}, input_shape,
                               {output.get(), output_strides}, output_shape,
                               float34_t(matrices), value, interp_mode, border_mode, threads);
                } else {
                    launch_<4>({input.get(), input_strides}, input_shape,
                               {output.get(), output_strides}, output_shape,
                               matrices.get(), value, interp_mode, border_mode, threads);
                }
            });
        }
    }

    #define NOA_INSTANTIATE_TRANSFORM2D_(T, M)  \
    template void transform2D<T, M, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, InterpMode, BorderMode, T, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM3D_(T, M)  \
    template void transform3D<T, M, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, InterpMode, BorderMode, T, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM2D_ALL_(T)             \
    NOA_INSTANTIATE_TRANSFORM2D_(T, float23_t);             \
    NOA_INSTANTIATE_TRANSFORM2D_(T, float33_t);             \
    NOA_INSTANTIATE_TRANSFORM2D_(T, shared_t<float23_t[]>); \
    NOA_INSTANTIATE_TRANSFORM2D_(T, shared_t<float33_t[]>)

    #define NOA_INSTANTIATE_TRANSFORM3D_ALL_(T)             \
    NOA_INSTANTIATE_TRANSFORM3D_(T, float34_t);             \
    NOA_INSTANTIATE_TRANSFORM3D_(T, float44_t);             \
    NOA_INSTANTIATE_TRANSFORM3D_(T, shared_t<float34_t[]>); \
    NOA_INSTANTIATE_TRANSFORM3D_(T, shared_t<float44_t[]>)

    #define NOA_INSTANTIATE_TRANSFORM_ALL_(T)   \
    NOA_INSTANTIATE_TRANSFORM2D_ALL_(T);        \
    NOA_INSTANTIATE_TRANSFORM3D_ALL_(T)

    NOA_INSTANTIATE_TRANSFORM_ALL_(float);
    NOA_INSTANTIATE_TRANSFORM_ALL_(double);
    NOA_INSTANTIATE_TRANSFORM_ALL_(cfloat_t);
    NOA_INSTANTIATE_TRANSFORM_ALL_(cdouble_t);
}
