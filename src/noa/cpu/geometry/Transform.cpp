#include "noa/common/Assert.h"
#include "noa/common/Types.h"
#include "noa/common/Exception.h"
#include "noa/common/geometry/Interpolator.h"

#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Transform.h"
#include "noa/cpu/geometry/Prefilter.h"

namespace {
    using namespace ::noa;

    template<InterpMode INTERP, BorderMode BORDER, typename T, typename matrix_t>
    void transform_(AccessorRestrict<const T, 3, int64_t> input, long2_t input_shape,
                    AccessorRestrict<T, 3, dim_t> output, dim3_t output_shape,
                    matrix_t matrices, T value, dim_t threads) {
        auto interpolator = geometry::interpolator2D<BORDER, INTERP>(input, input_shape, value);

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(output, output_shape, matrices, interpolator)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t y = 0; y < output_shape[1]; ++y) {
                for (dim_t x = 0; x < output_shape[2]; ++x) {

                    if constexpr (std::is_same_v<matrix_t, const float23_t*>) {
                        const float2_t coordinates = matrices[i] * float3_t{y, x, 1.f};
                        output(i, y, x) = interpolator(coordinates, i);
                    } else if constexpr (std::is_same_v<matrix_t, float23_t>) {
                        const float2_t coordinates = matrices * float3_t{y, x, 1.f};
                        output(i, y, x) = interpolator(coordinates, i);
                    } else if constexpr (std::is_same_v<matrix_t, const float33_t*>) {
                        const float23_t matrix(matrices[i]);
                        const float2_t coordinates = matrix * float3_t{y, x, 1.f};
                        output(i, y, x) = interpolator(coordinates, i);
                    } else {
                        static_assert(traits::always_false_v<T>);
                    }
                }
            }
        }
    }

    template<InterpMode INTERP, BorderMode BORDER, typename T, typename matrix_t>
    void transform_(AccessorRestrict<const T, 4, int64_t> input, long3_t input_shape,
                    AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape,
                    matrix_t matrices, T value, dim_t threads) {
        auto interpolator = geometry::interpolator3D<BORDER, INTERP>(input, input_shape, value);

        #pragma omp parallel for collapse(4) default(none) num_threads(threads) \
        shared(output, output_shape, matrices, interpolator)

        for (dim_t i = 0; i < output_shape[0]; ++i) {
            for (dim_t z = 0; z < output_shape[1]; ++z) {
                for (dim_t y = 0; y < output_shape[2]; ++y) {
                    for (dim_t x = 0; x < output_shape[3]; ++x) {

                        if constexpr (std::is_same_v<matrix_t, const float34_t*>) {
                            const float3_t coordinates = matrices[i] * float4_t{z, y, x, 1.f};
                            output(i, z, y, x) = interpolator(coordinates, i);
                        } else if constexpr (std::is_same_v<matrix_t, float34_t>) {
                            const float3_t coordinates = matrices * float4_t{z, y, x, 1.f};
                            output(i, z, y, x) = interpolator(coordinates, i);
                        } else if constexpr (std::is_same_v<matrix_t, const float44_t*>) {
                            const float34_t matrix(matrices[i]);
                            const float3_t coordinates = matrix * float4_t{z, y, x, 1.f};
                            output(i, z, y, x) = interpolator(coordinates, i);
                        } else {
                            static_assert(traits::always_false_v<T>);
                        }
                    }
                }
            }
        }
    }

    template<InterpMode INTERP, int N, typename T, typename ishape_t, typename oshape_t, typename matrix_t>
    void launch_(const AccessorRestrict<const T, N, int64_t>& input, ishape_t input_shape,
                 const AccessorRestrict<T, N, dim_t>& output, oshape_t output_shape,
                 matrix_t matrices, T value, BorderMode border_mode, dim_t threads) {
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
            case BORDER_NOTHING:
                NOA_THROW_FUNC("transform(2|3)D", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<int N, typename matrix_t, typename T, typename ishape_t, typename oshape_t>
    void launch_(const AccessorRestrict<const T, N, int64_t>& input, ishape_t input_shape,
                 const AccessorRestrict<T, N, dim_t>& output, oshape_t output_shape,
                 matrix_t matrices, T value, InterpMode interp_mode, BorderMode border_mode, dim_t threads) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                return launch_<INTERP_NEAREST>(
                        input, input_shape, output, output_shape,
                        matrices, value, border_mode, threads);
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return launch_<INTERP_LINEAR>(
                        input, input_shape, output, output_shape,
                        matrices, value, border_mode, threads);
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return launch_<INTERP_COSINE>(
                        input, input_shape, output, output_shape,
                        matrices, value, border_mode, threads);
            case INTERP_CUBIC:
                return launch_<INTERP_CUBIC>(
                        input, input_shape, output, output_shape,
                        matrices, value, border_mode, threads);
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST:
                return launch_<INTERP_CUBIC_BSPLINE>(
                        input, input_shape, output, output_shape,
                        matrices, value, border_mode, threads);
        }
    }

    template<typename T>
    auto matrixOrRawConstPtr(const T& v) {
        if constexpr (traits::is_float23_v<T> || traits::is_float33_v<T>) {
            return float23_t(v);
        } else if constexpr (traits::is_float34_v<T> || traits::is_float44_v<T>) {
            return float34_t(v);
        } else {
            NOA_ASSERT(v != nullptr);
            using clean_t = traits::remove_ref_cv_t<T>;
            using raw_const_ptr_t = const typename clean_t::element_type*;
            return static_cast<raw_const_ptr_t>(v.get());
        }
    }
}

namespace noa::cpu::geometry {
    template<typename T, typename MAT, typename>
    void transform2D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const MAT& matrices, InterpMode interp_mode, BorderMode border_mode,
                     T value, bool prefilter, Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1);

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;

        const long3_t istrides_2d{input_strides[0], input_strides[2], input_strides[3]};
        const long2_t ishape_2d{input_shape[2], input_shape[3]};

        const dim3_t ostrides_2d{output_strides[0], output_strides[2], output_strides[3]};
        const dim3_t oshape_2d{output_shape[0], output_shape[2], output_shape[3]};
        const dim_t threads = stream.threads();

        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                const dim4_t strides = input_shape.strides();
                memory::PtrHost<T> buffer(input_shape.elements());
                bspline::prefilter(input, input_strides, buffer.share(), strides, input_shape, stream);

                const dim3_t strides_2d{input_shape[0] == 1 ? 0 : strides[0], strides[2], strides[3]};
                launch_<3>({buffer.get(), strides_2d}, ishape_2d,
                           {output.get(), ostrides_2d}, oshape_2d,
                           matrixOrRawConstPtr(matrices), value, interp_mode, border_mode, threads);
            });
        } else {
            stream.enqueue([=]() {
                launch_<3>({input.get(), istrides_2d}, ishape_2d,
                           {output.get(), ostrides_2d}, oshape_2d,
                           matrixOrRawConstPtr(matrices), value, interp_mode, border_mode, threads);
            });
        }
    }

    template<typename T, typename MAT, typename>
    void transform3D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const MAT& matrices, InterpMode interp_mode, BorderMode border_mode,
                     T value, bool prefilter, Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;
        const long3_t ishape_3d(input_shape.get(1));

        const dim_t threads = stream.threads();
        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                dim4_t strides = input_shape.strides();
                if (input_shape[0] == 1)
                    strides[0] = 0;
                memory::PtrHost<T> buffer(input_shape.elements());
                bspline::prefilter(input, input_strides, buffer.share(), strides, input_shape, stream);

                launch_<4>({buffer.get(), strides}, ishape_3d,
                           {output.get(), output_strides}, output_shape,
                           matrixOrRawConstPtr(matrices), value, interp_mode, border_mode, threads);
            });
        } else {
            stream.enqueue([=]() {
                launch_<4>({input.get(), input_strides}, ishape_3d,
                           {output.get(), output_strides}, output_shape,
                           matrixOrRawConstPtr(matrices), value, interp_mode, border_mode, threads);
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
