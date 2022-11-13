#include "noa/common/Assert.h"
#include "noa/common/Types.h"
#include "noa/common/Exception.h"
#include "noa/common/geometry/Interpolator.h"
#include "noa/common/geometry/details/LinearTransform2D.h"
#include "noa/common/geometry/details/LinearTransform3D.h"

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/geometry/Transform.h"
#include "noa/cpu/geometry/Prefilter.h"
#include "noa/cpu/utils/Loops.h"

namespace {
    using namespace ::noa;

    template<InterpMode INTERP, typename Value, typename Matrix>
    void launchLinearTransformFinal_(const AccessorRestrict<const Value, 3, int64_t>& input, long2_t input_shape,
                                     const AccessorRestrict<Value, 3, int64_t>& output, long3_t output_shape,
                                     Matrix matrices, Value value, BorderMode border_mode, dim_t threads) {
        switch (border_mode) {
            case BORDER_ZERO: {
                const auto interpolator = geometry::interpolator2D<BORDER_ZERO, INTERP>(input, input_shape, value);
                const auto kernel = geometry::details::transform2D<int64_t>(interpolator, output, matrices);
                return cpu::utils::iwise3D(output_shape, kernel, threads);
            }
            case BORDER_VALUE: {
                const auto interpolator = geometry::interpolator2D<BORDER_VALUE, INTERP>(input, input_shape, value);
                const auto kernel = geometry::details::transform2D<int64_t>(interpolator, output, matrices);
                return cpu::utils::iwise3D(output_shape, kernel, threads);
            }
            case BORDER_CLAMP: {
                const auto interpolator = geometry::interpolator2D<BORDER_CLAMP, INTERP>(input, input_shape, value);
                const auto kernel = geometry::details::transform2D<int64_t>(interpolator, output, matrices);
                return cpu::utils::iwise3D(output_shape, kernel, threads);
            }
            case BORDER_PERIODIC: {
                const auto interpolator = geometry::interpolator2D<BORDER_PERIODIC, INTERP>(input, input_shape, value);
                const auto kernel = geometry::details::transform2D<int64_t>(interpolator, output, matrices);
                return cpu::utils::iwise3D(output_shape, kernel, threads);
            }
            case BORDER_MIRROR: {
                const auto interpolator = geometry::interpolator2D<BORDER_MIRROR, INTERP>(input, input_shape, value);
                const auto kernel = geometry::details::transform2D<int64_t>(interpolator, output, matrices);
                return cpu::utils::iwise3D(output_shape, kernel, threads);
            }
            case BORDER_REFLECT: {
                const auto interpolator = geometry::interpolator2D<BORDER_REFLECT, INTERP>(input, input_shape, value);
                const auto kernel = geometry::details::transform2D<int64_t>(interpolator, output, matrices);
                return cpu::utils::iwise3D(output_shape, kernel, threads);
            }
            case BORDER_NOTHING:
                NOA_THROW_FUNC("transform2D", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<InterpMode INTERP, typename Value, typename Matrix>
    void launchLinearTransformFinal_(const AccessorRestrict<const Value, 4, int64_t>& input, long3_t input_shape,
                                     const AccessorRestrict<Value, 4, int64_t>& output, long4_t output_shape,
                                     Matrix matrices, Value value, BorderMode border_mode, dim_t threads) {
        switch (border_mode) {
            case BORDER_ZERO: {
                const auto interpolator = geometry::interpolator3D<BORDER_ZERO, INTERP>(input, input_shape, value);
                const auto kernel = geometry::details::transform3D<int64_t>(interpolator, output, matrices);
                return cpu::utils::iwise4D(output_shape, kernel, threads);
            }
            case BORDER_VALUE: {
                const auto interpolator = geometry::interpolator3D<BORDER_VALUE, INTERP>(input, input_shape, value);
                const auto kernel = geometry::details::transform3D<int64_t>(interpolator, output, matrices);
                return cpu::utils::iwise4D(output_shape, kernel, threads);
            }
            case BORDER_CLAMP: {
                const auto interpolator = geometry::interpolator3D<BORDER_CLAMP, INTERP>(input, input_shape, value);
                const auto kernel = geometry::details::transform3D<int64_t>(interpolator, output, matrices);
                return cpu::utils::iwise4D(output_shape, kernel, threads);
            }
            case BORDER_PERIODIC: {
                const auto interpolator = geometry::interpolator3D<BORDER_PERIODIC, INTERP>(input, input_shape, value);
                const auto kernel = geometry::details::transform3D<int64_t>(interpolator, output, matrices);
                return cpu::utils::iwise4D(output_shape, kernel, threads);
            }
            case BORDER_MIRROR: {
                const auto interpolator = geometry::interpolator3D<BORDER_MIRROR, INTERP>(input, input_shape, value);
                const auto kernel = geometry::details::transform3D<int64_t>(interpolator, output, matrices);
                return cpu::utils::iwise4D(output_shape, kernel, threads);
            }
            case BORDER_REFLECT: {
                const auto interpolator = geometry::interpolator3D<BORDER_REFLECT, INTERP>(input, input_shape, value);
                const auto kernel = geometry::details::transform3D<int64_t>(interpolator, output, matrices);
                return cpu::utils::iwise4D(output_shape, kernel, threads);
            }
            case BORDER_NOTHING:
                NOA_THROW_FUNC("transform2D", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<int32_t NDIM, typename Value, typename IShape, typename OShape, typename Matrix>
    void launchLinearTransform_(const AccessorRestrict<const Value, NDIM + 1, int64_t>& input, IShape input_shape,
                                const AccessorRestrict<Value, NDIM + 1, int64_t>& output, OShape output_shape,
                                Matrix matrices, Value value, InterpMode interp_mode,
                                BorderMode border_mode, dim_t threads) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                return launchLinearTransformFinal_<INTERP_NEAREST>(
                        input, input_shape, output, output_shape,
                        matrices, value, border_mode, threads);
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return launchLinearTransformFinal_<INTERP_LINEAR>(
                        input, input_shape, output, output_shape,
                        matrices, value, border_mode, threads);
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return launchLinearTransformFinal_<INTERP_COSINE>(
                        input, input_shape, output, output_shape,
                        matrices, value, border_mode, threads);
            case INTERP_CUBIC:
                return launchLinearTransformFinal_<INTERP_CUBIC>(
                        input, input_shape, output, output_shape,
                        matrices, value, border_mode, threads);
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST:
                return launchLinearTransformFinal_<INTERP_CUBIC_BSPLINE>(
                        input, input_shape, output, output_shape,
                        matrices, value, border_mode, threads);
        }
    }

    template<typename Value>
    void launchSymmetry_(const Value* input, dim4_t input_strides,
                         Value* output, dim4_t output_strides, dim4_t shape,
                         const geometry::Symmetry& symmetry, float2_t center,
                         InterpMode interp_mode, bool normalize, dim_t threads) {
        using value_t = traits::value_type_t<Value>;
        const auto symmetry_count = static_cast<int64_t>(symmetry.count());
        const float33_t* symmetry_matrices = symmetry.get();
        const auto scaling = normalize ? 1 / static_cast<value_t>(symmetry_count + 1) : 1;

        const auto input_shape_2d = long2_t(shape.get(2));
        const auto output_shape_2d = long3_t(shape[0], shape[2], shape[3]);
        const auto input_strides_2d = long3_t(input_strides[0], input_strides[2], input_strides[3]);
        const auto output_strides_2d = long3_t(output_strides[0], output_strides[2], output_strides[3]);
        const auto input_accessor = AccessorRestrict<const Value, 3, int64_t>(input, input_strides_2d);
        const auto output_accessor = AccessorRestrict<Value, 3, int64_t>(output, output_strides_2d);

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator =
                geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(input_accessor, input_shape_2d);
                const auto kernel = geometry::details::symmetry2D(
                        interpolator, output_accessor, center, symmetry_matrices, symmetry_count, scaling);
                return cpu::utils::iwise3D(output_shape_2d, kernel, threads);
            }
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST: {
                const auto interpolator =
                geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(input_accessor, input_shape_2d);
                const auto kernel = geometry::details::symmetry2D(
                        interpolator, output_accessor, center, symmetry_matrices, symmetry_count, scaling);
                return cpu::utils::iwise3D(output_shape_2d, kernel, threads);
            }
            case INTERP_COSINE:
            case INTERP_COSINE_FAST: {
                const auto interpolator =
                geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(input_accessor, input_shape_2d);
                const auto kernel = geometry::details::symmetry2D(
                        interpolator, output_accessor, center, symmetry_matrices, symmetry_count, scaling);
                return cpu::utils::iwise3D(output_shape_2d, kernel, threads);
            }
            case INTERP_CUBIC: {
                const auto interpolator =
                geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC>(input_accessor, input_shape_2d);
                const auto kernel = geometry::details::symmetry2D(
                        interpolator, output_accessor, center, symmetry_matrices, symmetry_count, scaling);
                return cpu::utils::iwise3D(output_shape_2d, kernel, threads);
            }
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST: {
                const auto interpolator =
                geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC_BSPLINE>(input_accessor, input_shape_2d);
                const auto kernel = geometry::details::symmetry2D(
                        interpolator, output_accessor, center, symmetry_matrices, symmetry_count, scaling);
                return cpu::utils::iwise3D(output_shape_2d, kernel, threads);
            }
        }
    }

    template<typename Value>
    void launchSymmetry_(const Value* input, dim4_t input_strides,
                         Value* output, dim4_t output_strides, dim4_t shape,
                         const geometry::Symmetry& symmetry, float3_t center,
                         InterpMode interp_mode, bool normalize, dim_t threads) {
        using value_t = traits::value_type_t<Value>;
        const auto symmetry_count = static_cast<int64_t>(symmetry.count());
        const float33_t* symmetry_matrices = symmetry.get();
        const auto scaling = normalize ? 1 / static_cast<value_t>(symmetry_count + 1) : 1;

        const auto output_shape_3d = long4_t(shape);
        const auto input_shape_3d = long3_t(output_shape_3d.get(1));
        const auto input_accessor = AccessorRestrict<const Value, 4, int64_t>(input, input_strides);
        const auto output_accessor = AccessorRestrict<Value, 4, int64_t>(output, output_strides);

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator =
                        geometry::interpolator3D<BORDER_ZERO, INTERP_NEAREST>(input_accessor, input_shape_3d);
                const auto kernel = geometry::details::symmetry3D(
                        interpolator, output_accessor, center, symmetry_matrices, symmetry_count, scaling);
                return cpu::utils::iwise4D(output_shape_3d, kernel, threads);
            }
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST: {
                const auto interpolator =
                        geometry::interpolator3D<BORDER_ZERO, INTERP_LINEAR>(input_accessor, input_shape_3d);
                const auto kernel = geometry::details::symmetry3D(
                        interpolator, output_accessor, center, symmetry_matrices, symmetry_count, scaling);
                return cpu::utils::iwise4D(output_shape_3d, kernel, threads);
            }
            case INTERP_COSINE:
            case INTERP_COSINE_FAST: {
                const auto interpolator =
                        geometry::interpolator3D<BORDER_ZERO, INTERP_COSINE>(input_accessor, input_shape_3d);
                const auto kernel = geometry::details::symmetry3D(
                        interpolator, output_accessor, center, symmetry_matrices, symmetry_count, scaling);
                return cpu::utils::iwise4D(output_shape_3d, kernel, threads);
            }
            case INTERP_CUBIC: {
                const auto interpolator =
                        geometry::interpolator3D<BORDER_ZERO, INTERP_CUBIC>(input_accessor, input_shape_3d);
                const auto kernel = geometry::details::symmetry3D(
                        interpolator, output_accessor, center, symmetry_matrices, symmetry_count, scaling);
                return cpu::utils::iwise4D(output_shape_3d, kernel, threads);
            }
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST: {
                const auto interpolator =
                        geometry::interpolator3D<BORDER_ZERO, INTERP_CUBIC_BSPLINE>(input_accessor, input_shape_3d);
                const auto kernel = geometry::details::symmetry3D(
                        interpolator, output_accessor, center, symmetry_matrices, symmetry_count, scaling);
                return cpu::utils::iwise4D(output_shape_3d, kernel, threads);
            }
        }
    }

    template<typename Value, typename Center>
    void symmetrizeND_(const shared_t<Value[]>& input, dim4_t input_strides,
                       const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                       const geometry::Symmetry& symmetry, Center center, InterpMode interp_mode, bool prefilter,
                       bool normalize, cpu::Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(shape > 0));
        NOA_ASSERT((std::is_same_v<Center, float3_t> && dim3_t(shape.get(1)).ndim() <= 3) ||
                   (std::is_same_v<Center, float2_t> && dim3_t(shape.get(1)).ndim() <= 2));

        if (!symmetry.count())
            return cpu::memory::copy(input, input_strides, output, output_strides, shape, stream);

        const dim_t threads = stream.threads();
        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                dim4_t new_shape = shape;
                dim4_t new_strides = new_shape.strides();

                if (input_strides[0] == 0) {
                    new_shape[0] = 1; // only one batch in input
                    new_strides[0] = 0;
                }
                const auto unique_buffer = cpu::memory::PtrHost<Value>::alloc(new_shape.elements());
                cpu::geometry::bspline::prefilter(
                        input.get(), input_strides,
                        unique_buffer.get(), new_strides,
                        new_shape, threads);

                launchSymmetry_<Value>(
                        unique_buffer.get(), new_strides, output.get(), output_strides, shape,
                        symmetry, center, interp_mode, normalize, threads);
            });
        } else {
            stream.enqueue([=]() {
                launchSymmetry_<Value>(
                        input.get(), input_strides, output.get(), output_strides, shape,
                        symmetry, center, interp_mode, normalize, threads);
            });
        }
    }

    template<typename Matrix>
    auto matrixOrRawConstPtr(const Matrix& v) {
        if constexpr (traits::is_float23_v<Matrix> || traits::is_float33_v<Matrix>) {
            return float23_t(v);
        } else if constexpr (traits::is_float34_v<Matrix> || traits::is_float44_v<Matrix>) {
            return float34_t(v);
        } else {
            NOA_ASSERT(v != nullptr);
            using clean_t = traits::remove_ref_cv_t<Matrix>;
            using raw_const_ptr_t = const typename clean_t::element_type*;
            return static_cast<raw_const_ptr_t>(v.get());
        }
    }
}

namespace noa::cpu::geometry {
    template<typename Value, typename Matrix, typename>
    void transform2D(const shared_t<Value[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const Matrix& matrices, InterpMode interp_mode, BorderMode border_mode,
                     Value value, bool prefilter, Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1);

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;

        const auto input_strides_2d = long3_t{input_strides[0], input_strides[2], input_strides[3]};
        const auto input_shape_2d = long2_t{input_shape[2], input_shape[3]};
        const auto output_strides_2d = dim3_t{output_strides[0], output_strides[2], output_strides[3]};
        const auto output_shape_2d = long3_t{output_shape[0], output_shape[2], output_shape[3]};
        const auto threads = stream.threads();

        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                dim4_t buffer_strides = input_shape.strides();
                if (input_shape[0] == 1)
                    buffer_strides[0] = 0;
                const auto buffer_strides_2d = long3_t{buffer_strides[0], buffer_strides[2], buffer_strides[3]};
                const auto unique_buffer = memory::PtrHost<Value>::alloc(input_shape.elements());
                bspline::prefilter(input.get(), input_strides,
                                   unique_buffer.get(), buffer_strides,
                                   input_shape, threads);

                launchLinearTransform_<2>(
                        {unique_buffer.get(), buffer_strides_2d}, input_shape_2d,
                        {output.get(), output_strides_2d}, output_shape_2d,
                        matrixOrRawConstPtr(matrices), value, interp_mode, border_mode, threads);
            });
        } else {
            stream.enqueue([=]() {
                launchLinearTransform_<2>(
                        {input.get(), input_strides_2d}, input_shape_2d,
                        {output.get(), output_strides_2d}, output_shape_2d,
                        matrixOrRawConstPtr(matrices), value, interp_mode, border_mode, threads);
            });
        }
    }

    template<typename Value, typename Matrix, typename>
    void transform3D(const shared_t<Value[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const Matrix& matrices, InterpMode interp_mode, BorderMode border_mode,
                     Value value, bool prefilter, Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;

        const long3_t input_shape_3d(input_shape.get(1));
        const long4_t output_shape_3d(output_shape);
        const dim_t threads = stream.threads();

        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            stream.enqueue([=]() mutable {
                dim4_t buffer_strides = input_shape.strides();
                if (input_shape[0] == 1)
                    buffer_strides[0] = 0;
                const auto unique_buffer = memory::PtrHost<Value>::alloc(input_shape.elements());
                bspline::prefilter(input.get(), input_strides,
                                   unique_buffer.get(), buffer_strides,
                                   input_shape, threads);

                launchLinearTransform_<3>(
                        {unique_buffer.get(), buffer_strides}, input_shape_3d,
                        {output.get(), output_strides}, output_shape_3d,
                        matrixOrRawConstPtr(matrices), value, interp_mode, border_mode, threads);
            });
        } else {
            stream.enqueue([=]() {
                launchLinearTransform_<3>(
                        {input.get(), input_strides}, input_shape_3d,
                        {output.get(), output_strides}, output_shape_3d,
                        matrixOrRawConstPtr(matrices), value, interp_mode, border_mode, threads);
            });
        }
    }

    template<typename Value, typename>
    void transform2D(const shared_t<Value[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float2_t shift, float22_t matrix, const Symmetry& symmetry, float2_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(dim3_t(input_shape.get(1)).ndim() <= 2);
        NOA_ASSERT(dim3_t(output_shape.get(1)).ndim() <= 2);

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            using unique_ptr = typename memory::PtrHost<Value>::alloc_unique_t;
            unique_ptr buffer;
            const Value* input_ptr;
            dim3_t input_strides_2d; // assume Z == 1
            if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
                buffer = memory::PtrHost<Value>::alloc(input_shape.elements());
                dim4_t buffer_strides = input_shape.strides();
                if (input_shape[0] == 1)
                    buffer_strides[0] = 0;
                bspline::prefilter(input.get(), input_strides, buffer.get(), buffer_strides, input_shape, threads);
                input_ptr = buffer.get();
                input_strides_2d = {buffer_strides[0], buffer_strides[2], buffer_strides[3]};
            } else {
                input_ptr = input.get();
                input_strides_2d = {input_strides[0], input_strides[2], input_strides[3]};
            }

            const auto input_shape_2d = long2_t{input_shape[2], input_shape[3]};
            const auto output_shape_2d = long3_t{output_shape[0], output_shape[2], output_shape[3]};
            const auto output_strides_2d = long3_t{output_strides[0], output_strides[2], output_strides[3]};
            const auto input_accessor = AccessorRestrict<const Value, 3, int64_t>(input_ptr, input_strides_2d);
            const auto output_accessor = AccessorRestrict<Value, 3, int64_t>(output.get(), output_strides_2d);

            using real_t = traits::value_type_t<Value>;
            const auto symmetry_count = static_cast<int64_t>(symmetry.count());
            const float33_t* symmetry_matrices = symmetry.get();
            const auto scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;

            switch (interp_mode) {
                case INTERP_NEAREST: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(
                            input_accessor, input_shape_2d);
                    const auto kernel = noa::geometry::details::transformSymmetry2D(
                            interpolator, output_accessor, shift, matrix, center,
                            symmetry_matrices, symmetry_count, scaling);
                    return cpu::utils::iwise3D(output_shape_2d, kernel, threads);
                }
                case INTERP_LINEAR:
                case INTERP_LINEAR_FAST: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(
                            input_accessor, input_shape_2d);
                    const auto kernel = noa::geometry::details::transformSymmetry2D(
                            interpolator, output_accessor, shift, matrix, center,
                            symmetry_matrices, symmetry_count, scaling);
                    return cpu::utils::iwise3D(output_shape_2d, kernel, threads);
                }
                case INTERP_COSINE:
                case INTERP_COSINE_FAST: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(
                            input_accessor, input_shape_2d);
                    const auto kernel = noa::geometry::details::transformSymmetry2D(
                            interpolator, output_accessor, shift, matrix, center,
                            symmetry_matrices, symmetry_count, scaling);
                    return cpu::utils::iwise3D(output_shape_2d, kernel, threads);
                }
                case INTERP_CUBIC: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC>(
                            input_accessor, input_shape_2d);
                    const auto kernel = noa::geometry::details::transformSymmetry2D(
                            interpolator, output_accessor, shift, matrix, center,
                            symmetry_matrices, symmetry_count, scaling);
                    return cpu::utils::iwise3D(output_shape_2d, kernel, threads);
                }
                case INTERP_CUBIC_BSPLINE:
                case INTERP_CUBIC_BSPLINE_FAST: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC_BSPLINE>(
                            input_accessor, input_shape_2d);
                    const auto kernel = noa::geometry::details::transformSymmetry2D(
                            interpolator, output_accessor, shift, matrix, center,
                            symmetry_matrices, symmetry_count, scaling);
                    return cpu::utils::iwise3D(output_shape_2d, kernel, threads);
                }
            }
        });
    }

    template<typename Value, typename>
    void transform3D(const shared_t<Value[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            using unique_ptr = typename memory::PtrHost<Value>::alloc_unique_t;
            unique_ptr buffer;
            const Value* input_ptr;
            long4_t input_strides_3d;
            if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
                buffer = memory::PtrHost<Value>::alloc(input_shape.elements());
                dim4_t buffer_strides = input_shape.strides();
                if (input_shape[0] == 1)
                    buffer_strides[0] = 0;
                bspline::prefilter(input.get(), input_strides, buffer.get(), buffer_strides, input_shape, threads);
                input_ptr = buffer.get();
                input_strides_3d = long4_t(buffer_strides);
            } else {
                input_ptr = input.get();
                input_strides_3d = long4_t(input_strides);
            }

            const long3_t input_shape_3d(input_shape.get(1));
            const long4_t output_shape_3d(output_shape);
            const auto input_accessor = AccessorRestrict<const Value, 4, int64_t>(input_ptr, input_strides_3d);
            const auto output_accessor = AccessorRestrict<Value, 4, int64_t>(output.get(), output_strides);

            using real_t = traits::value_type_t<Value>;
            const auto symmetry_count = static_cast<int64_t>(symmetry.count());
            const float33_t* symmetry_matrices = symmetry.get();
            const auto scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;

            switch (interp_mode) {
                case INTERP_NEAREST: {
                    const auto interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP_NEAREST>(
                            input_accessor, input_shape_3d);
                    const auto kernel = noa::geometry::details::transformSymmetry3D(
                            interpolator, output_accessor, shift, matrix, center,
                            symmetry_matrices, symmetry_count, scaling);
                    return cpu::utils::iwise4D(output_shape_3d, kernel, threads);
                }
                case INTERP_LINEAR:
                case INTERP_LINEAR_FAST: {
                    const auto interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP_LINEAR>(
                            input_accessor, input_shape_3d);
                    const auto kernel = noa::geometry::details::transformSymmetry3D(
                            interpolator, output_accessor, shift, matrix, center,
                            symmetry_matrices, symmetry_count, scaling);
                    return cpu::utils::iwise4D(output_shape_3d, kernel, threads);
                }
                case INTERP_COSINE:
                case INTERP_COSINE_FAST: {
                    const auto interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP_COSINE>(
                            input_accessor, input_shape_3d);
                    const auto kernel = noa::geometry::details::transformSymmetry3D(
                            interpolator, output_accessor, shift, matrix, center,
                            symmetry_matrices, symmetry_count, scaling);
                    return cpu::utils::iwise4D(output_shape_3d, kernel, threads);
                }
                case INTERP_CUBIC: {
                    const auto interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP_CUBIC>(
                            input_accessor, input_shape_3d);
                    const auto kernel = noa::geometry::details::transformSymmetry3D(
                            interpolator, output_accessor, shift, matrix, center,
                            symmetry_matrices, symmetry_count, scaling);
                    return cpu::utils::iwise4D(output_shape_3d, kernel, threads);
                }
                case INTERP_CUBIC_BSPLINE:
                case INTERP_CUBIC_BSPLINE_FAST: {
                    const auto interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP_CUBIC_BSPLINE>(
                            input_accessor, input_shape_3d);
                    const auto kernel = noa::geometry::details::transformSymmetry3D(
                            interpolator, output_accessor, shift, matrix, center,
                            symmetry_matrices, symmetry_count, scaling);
                    return cpu::utils::iwise4D(output_shape_3d, kernel, threads);
                }
            }
        });
    }

    template<typename Value, typename>
    void symmetrize2D(const shared_t<Value[]>& input, dim4_t input_strides,
                      const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                      const Symmetry& symmetry, float2_t center, InterpMode interp_mode,
                      bool prefilter, bool normalize, Stream& stream) {
        symmetrizeND_(input, input_strides, output, output_strides, shape,
                      symmetry, center, interp_mode, prefilter, normalize, stream);
    }

    template<typename Value, typename>
    void symmetrize3D(const shared_t<Value[]>& input, dim4_t input_strides,
                      const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                      const Symmetry& symmetry, float3_t center, InterpMode interp_mode,
                      bool prefilter, bool normalize, Stream& stream) {
        symmetrizeND_(input, input_strides, output, output_strides, shape,
                      symmetry, center, interp_mode, prefilter, normalize, stream);
    }

    #define NOA_INSTANTIATE_SYMMETRY_(T)                                \
    template void symmetrize2D<T, void>(                                \
        const shared_t<T[]>&, dim4_t,                                   \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        const Symmetry&, float2_t, InterpMode, bool, bool, Stream&);    \
    template void symmetrize3D<T, void>(                                \
        const shared_t<T[]>&, dim4_t,                                   \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        const Symmetry&, float3_t, InterpMode, bool, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM2D_(T, M)  \
    template void transform2D<T, M, void>(      \
        const shared_t<T[]>&, dim4_t, dim4_t,   \
        const shared_t<T[]>&, dim4_t, dim4_t,   \
        const M&, InterpMode, BorderMode, T, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM3D_(T, M)  \
    template void transform3D<T, M, void>(      \
        const shared_t<T[]>&, dim4_t, dim4_t,   \
        const shared_t<T[]>&, dim4_t, dim4_t,   \
        const M&, InterpMode, BorderMode, T, bool, Stream&)

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

    #define NOA_INSTANTIATE_TRANSFORM_SYMMETRY_(T)      \
    template void transform2D<T, void>(                 \
        const shared_t<T[]>&, dim4_t, dim4_t,           \
        const shared_t<T[]>&, dim4_t, dim4_t,           \
        float2_t, float22_t, const Symmetry&, float2_t, \
        InterpMode, bool, bool, Stream&);               \
    template void transform3D<T, void>(                 \
        const shared_t<T[]>&, dim4_t, dim4_t,           \
        const shared_t<T[]>&, dim4_t, dim4_t,           \
        float3_t, float33_t, const Symmetry&, float3_t, \
        InterpMode, bool, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_ALL_(T)   \
    NOA_INSTANTIATE_TRANSFORM2D_ALL_(T);        \
    NOA_INSTANTIATE_TRANSFORM3D_ALL_(T);        \
    NOA_INSTANTIATE_TRANSFORM_SYMMETRY_(T);     \
    NOA_INSTANTIATE_SYMMETRY_(T)

    NOA_INSTANTIATE_TRANSFORM_ALL_(float);
    NOA_INSTANTIATE_TRANSFORM_ALL_(double);
    NOA_INSTANTIATE_TRANSFORM_ALL_(cfloat_t);
    NOA_INSTANTIATE_TRANSFORM_ALL_(cdouble_t);
}
