#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/geometry/Interpolator.h"
#include "noa/common/geometry/details/LinearTransform2D.h"
#include "noa/common/geometry/details/LinearTransform3D.h"

#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"

#include "noa/gpu/cuda/geometry/Prefilter.h"
#include "noa/gpu/cuda/geometry/Transform.h"

namespace {
    using namespace ::noa;

    template<InterpMode INTERP, typename Value, typename Matrix>
    void launchLinearTransformFinal_(const AccessorRestrict<const Value, 3, uint32_t>& input, int2_t input_shape,
                                     const AccessorRestrict<Value, 3, uint32_t>& output, int3_t output_shape,
                                     Matrix inv_matrices, Value value, BorderMode border_mode,
                                     cuda::Stream& stream) {
        switch (border_mode) {
            case BORDER_ZERO: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP>(input, input_shape, value);
                const auto kernel = noa::geometry::details::transform2D<uint32_t>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
            }
            case BORDER_VALUE: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_VALUE, INTERP>(input, input_shape, value);
                const auto kernel = noa::geometry::details::transform2D<uint32_t>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
            }
            case BORDER_CLAMP: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_CLAMP, INTERP>(input, input_shape, value);
                const auto kernel = noa::geometry::details::transform2D<uint32_t>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
            }
            case BORDER_PERIODIC: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_PERIODIC, INTERP>(input, input_shape, value);
                const auto kernel = noa::geometry::details::transform2D<uint32_t>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
            }
            case BORDER_MIRROR: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_MIRROR, INTERP>(input, input_shape, value);
                const auto kernel = noa::geometry::details::transform2D<uint32_t>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
            }
            case BORDER_REFLECT: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_REFLECT, INTERP>(input, input_shape, value);
                const auto kernel = noa::geometry::details::transform2D<uint32_t>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
            }
            case BORDER_NOTHING:
                NOA_THROW_FUNC("transform2D", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<InterpMode INTERP, typename Value, typename Matrix>
    void launchLinearTransformFinal_(const AccessorRestrict<const Value, 4, uint32_t>& input, int3_t input_shape,
                                     const AccessorRestrict<Value, 4, uint32_t>& output, int4_t output_shape,
                                     Matrix inv_matrices, Value value, BorderMode border_mode,
                                     cuda::Stream& stream) {
        switch (border_mode) {
            case BORDER_ZERO: {
                const auto interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP>(input, input_shape, value);
                const auto kernel = noa::geometry::details::transform3D<uint32_t>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
            }
            case BORDER_VALUE: {
                const auto interpolator = noa::geometry::interpolator3D<BORDER_VALUE, INTERP>(input, input_shape, value);
                const auto kernel = noa::geometry::details::transform3D<uint32_t>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
            }
            case BORDER_CLAMP: {
                const auto interpolator = noa::geometry::interpolator3D<BORDER_CLAMP, INTERP>(input, input_shape, value);
                const auto kernel = noa::geometry::details::transform3D<uint32_t>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
            }
            case BORDER_PERIODIC: {
                const auto interpolator = noa::geometry::interpolator3D<BORDER_PERIODIC, INTERP>(input, input_shape, value);
                const auto kernel = noa::geometry::details::transform3D<uint32_t>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
            }
            case BORDER_MIRROR: {
                const auto interpolator = noa::geometry::interpolator3D<BORDER_MIRROR, INTERP>(input, input_shape, value);
                const auto kernel = noa::geometry::details::transform3D<uint32_t>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
            }
            case BORDER_REFLECT: {
                const auto interpolator = noa::geometry::interpolator3D<BORDER_REFLECT, INTERP>(input, input_shape, value);
                const auto kernel = noa::geometry::details::transform3D<uint32_t>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
            }
            case BORDER_NOTHING:
                NOA_THROW_FUNC("transform3D", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<int32_t NDIM, typename Value, typename IShape, typename OShape, typename Matrix>
    void launchLinearTransform_(const AccessorRestrict<const Value, NDIM + 1, uint32_t>& input, IShape input_shape,
                                const AccessorRestrict<Value, NDIM + 1, uint32_t>& output, OShape output_shape,
                                Matrix inv_matrices, Value value, InterpMode interp_mode,
                                BorderMode border_mode, cuda::Stream& stream) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                return launchLinearTransformFinal_<INTERP_NEAREST>(
                        input, input_shape, output, output_shape,
                        inv_matrices, value, border_mode, stream);
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST:
                return launchLinearTransformFinal_<INTERP_LINEAR>(
                        input, input_shape, output, output_shape,
                        inv_matrices, value, border_mode, stream);
            case INTERP_COSINE:
            case INTERP_COSINE_FAST:
                return launchLinearTransformFinal_<INTERP_COSINE>(
                        input, input_shape, output, output_shape,
                        inv_matrices, value, border_mode, stream);
            case INTERP_CUBIC:
                return launchLinearTransformFinal_<INTERP_CUBIC>(
                        input, input_shape, output, output_shape,
                        inv_matrices, value, border_mode, stream);
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST:
                return launchLinearTransformFinal_<INTERP_CUBIC_BSPLINE>(
                        input, input_shape, output, output_shape,
                        inv_matrices, value, border_mode, stream);
        }
    }

    template<typename Value>
    void launchSymmetry_(const Value* input, dim4_t input_strides,
                         Value* output, dim4_t output_strides, dim4_t shape,
                         const geometry::Symmetry& symmetry, float2_t center,
                         InterpMode interp_mode, bool normalize, cuda::Stream& stream) {

        using real_t = traits::value_type_t<Value>;
        const auto symmetry_count = static_cast<int32_t>(symmetry.count());
        const auto scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;
        const auto symmetry_matrices = cuda::memory::PtrDevice<float33_t>::alloc(symmetry_count, stream);
        cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);

        const auto input_shape_2d = int2_t(shape.get(2));
        const auto output_shape_2d = int3_t{shape[0], shape[2], shape[3]};
        const auto input_strides_2d = safe_cast<uint3_t>(dim3_t{input_strides[0], input_strides[2], input_strides[3]});
        const auto output_strides_2d = safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});
        const auto input_accessor = AccessorRestrict<const Value, 3, uint32_t>(input, input_strides_2d);
        const auto output_accessor = AccessorRestrict<Value, 3, uint32_t>(output, output_strides_2d);

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::geometry::details::symmetry2D(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise3D("geometry::symmetry2D", output_shape_2d, kernel, stream);
            }
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(input_accessor, input_shape_2d);
                const auto kernel = noa::geometry::details::symmetry2D(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise3D("geometry::symmetry2D", output_shape_2d, kernel, stream);
            }
            case INTERP_COSINE:
            case INTERP_COSINE_FAST: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(input_accessor, input_shape_2d);
                const auto kernel = noa::geometry::details::symmetry2D(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise3D("geometry::symmetry2D", output_shape_2d, kernel, stream);
            }
            case INTERP_CUBIC: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC>(input_accessor, input_shape_2d);
                const auto kernel = noa::geometry::details::symmetry2D(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise3D("geometry::symmetry2D", output_shape_2d, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC_BSPLINE>(input_accessor, input_shape_2d);
                const auto kernel = noa::geometry::details::symmetry2D(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise3D("geometry::symmetry2D", output_shape_2d, kernel, stream);
            }
        }
    }

    template<typename Value>
    void launchSymmetry_(const Value* input, dim4_t input_strides,
                         Value* output, dim4_t output_strides, dim4_t shape,
                         const geometry::Symmetry& symmetry, float3_t center,
                         InterpMode interp_mode, bool normalize, cuda::Stream& stream) {

        using real_t = traits::value_type_t<Value>;
        const auto symmetry_count = static_cast<int32_t>(symmetry.count());
        const auto scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;
        const auto symmetry_matrices = cuda::memory::PtrDevice<float33_t>::alloc(symmetry_count, stream);
        cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);

        const auto output_shape_3d = static_cast<int4_t>(shape);
        const auto input_shape_3d = static_cast<int3_t>(output_shape_3d.get(1));
        const auto input_strides_3d = safe_cast<uint4_t>(input_strides);
        const auto output_strides_3d = safe_cast<uint4_t>(output_strides);
        const auto input_accessor = AccessorRestrict<const Value, 4, uint32_t>(input, input_strides_3d);
        const auto output_accessor = AccessorRestrict<Value, 4, uint32_t>(output, output_strides_3d);

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_NEAREST>(input_accessor, input_shape_3d);
                const auto kernel = noa::geometry::details::symmetry3D(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise4D("geometry::symmetry3D", output_shape_3d, kernel, stream);
            }
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_LINEAR>(input_accessor, input_shape_3d);
                const auto kernel = noa::geometry::details::symmetry3D(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise4D("geometry::symmetry3D", output_shape_3d, kernel, stream);
            }
            case INTERP_COSINE:
            case INTERP_COSINE_FAST: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_COSINE>(input_accessor, input_shape_3d);
                const auto kernel = noa::geometry::details::symmetry3D(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise4D("geometry::symmetry3D", output_shape_3d, kernel, stream);
            }
            case INTERP_CUBIC: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_CUBIC>(input_accessor, input_shape_3d);
                const auto kernel = noa::geometry::details::symmetry3D(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise4D("geometry::symmetry3D", output_shape_3d, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_CUBIC_BSPLINE>(input_accessor, input_shape_3d);
                const auto kernel = noa::geometry::details::symmetry3D(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise4D("geometry::symmetry3D", output_shape_3d, kernel, stream);
            }
        }
    }

    template<typename Value, typename Center>
    void symmetrizeND_(const shared_t<Value[]>& input, dim4_t input_strides,
                       const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                       const geometry::Symmetry& symmetry, Center center, InterpMode interp_mode, bool prefilter,
                       bool normalize, cuda::Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(input.get() != output.get() && all(shape > 0));
        NOA_ASSERT((std::is_same_v<Center, float3_t> && dim3_t(shape.get(1)).ndim() <= 3) ||
                   (std::is_same_v<Center, float2_t> && dim3_t(shape.get(1)).ndim() <= 2));

        if (!symmetry.count())
            return cuda::memory::copy(input, input_strides, output, output_strides, shape, stream);

        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST))
            cuda::geometry::bspline::prefilter(input.get(), input_strides, input.get(), input_strides, shape, stream);

        launchSymmetry_<Value>(
                input.get(), input_strides, output.get(), output_strides, shape,
                symmetry, center, interp_mode, normalize, stream);

        stream.attach(input, output, symmetry.share());
    }

    template<typename Matrix>
    auto matrixOrRawConstPtr(const Matrix& matrices) {
        if constexpr (traits::is_float23_v<Matrix> || traits::is_float33_v<Matrix>) {
            return float23_t(matrices);
        } else if constexpr (traits::is_float34_v<Matrix> || traits::is_float44_v<Matrix>) {
            return float34_t(matrices);
        } else {
            NOA_ASSERT(matrices != nullptr);
            using clean_t = traits::remove_ref_cv_t<Matrix>;
            using raw_const_ptr_t = const typename clean_t::element_type*;
            return static_cast<raw_const_ptr_t>(matrices.get());
        }
    }
}

namespace noa::cuda::geometry {
    template<typename Value, typename Matrix, typename>
    void transform2D(const shared_t<Value[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const Matrix& inv_matrices, InterpMode interp_mode, BorderMode border_mode,
                     Value cvalue, bool prefilter, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(input.get() != output.get());
        NOA_ASSERT(all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1 && output_shape[1] == 1);
        if constexpr (!traits::is_floatXX_v<Matrix>) {
            NOA_ASSERT_DEVICE_PTR(inv_matrices.get(), stream.device());
        }

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;

        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST))
            bspline::prefilter(input.get(), input_strides, input.get(), input_strides, input_shape, stream);

        const auto input_strides_ = safe_cast<uint4_t>(input_strides);
        const auto output_strides_ = safe_cast<uint4_t>(output_strides);
        const auto input_strides_2d = uint3_t{input_strides_[0], input_strides_[2], input_strides_[3]};
        const auto output_strides_2d = uint3_t{output_strides_[0], output_strides_[2], output_strides_[3]};

        const auto input_shape_2d = int2_t{input_shape[2], input_shape[3]};
        const auto output_shape_2d = int3_t{output_shape[0], output_shape[2], output_shape[3]};

        launchLinearTransform_<2>(
                {input.get(), input_strides_2d}, input_shape_2d,
                {output.get(), output_strides_2d}, output_shape_2d,
                matrixOrRawConstPtr(inv_matrices), cvalue,
                interp_mode, border_mode, stream);

        stream.attach(input, output);
        if constexpr (!traits::is_floatXX_v<Matrix>)
            stream.attach(inv_matrices);
    }

    template<typename Value, typename Matrix, typename>
    void transform3D(const shared_t<Value[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const Matrix& inv_matrices, InterpMode interp_mode, BorderMode border_mode,
                     Value cvalue, bool prefilter, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(input.get() != output.get());
        NOA_ASSERT(all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        if constexpr (!traits::is_floatXX_v<Matrix>) {
            NOA_ASSERT_DEVICE_PTR(inv_matrices.get(), stream.device());
        }

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;

        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST))
            bspline::prefilter(input.get(), input_strides, input.get(), input_strides, input_shape, stream);

        const auto input_strides_ = safe_cast<uint4_t>(input_strides);
        const auto output_strides_ = safe_cast<uint4_t>(output_strides);
        const auto input_shape_3d = int3_t(input_shape.get(1));
        const auto output_shape_3d = int4_t(output_shape);

        launchLinearTransform_<3>(
                {input.get(), input_strides_}, input_shape_3d,
                {output.get(), output_strides_}, output_shape_3d,
                matrixOrRawConstPtr(inv_matrices), cvalue,
                interp_mode, border_mode, stream);

        stream.attach(input, output);
        if constexpr (!traits::is_floatXX_v<Matrix>)
            stream.attach(inv_matrices);
    }

    template<typename Value, typename>
    void transform2D(const shared_t<Value[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float2_t shift, float22_t inv_matrix, const Symmetry& symmetry, float2_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream) {
        NOA_ASSERT(all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1 && output_shape[1] == 1);

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;

        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST))
            bspline::prefilter(input.get(), input_strides, input.get(), input_strides, input_shape, stream);

        const auto input_shape_2d = int2_t{input_shape[2], input_shape[3]};
        const auto output_shape_2d = int3_t{output_shape[0], output_shape[2], output_shape[3]};
        const auto input_strides_2d = safe_cast<uint3_t>(dim3_t{input_strides[0], input_strides[2], input_strides[3]});
        const auto output_strides_2d = safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});
        const auto input_accessor = AccessorRestrict<const Value, 3, uint32_t>(input.get(), input_strides_2d);
        const auto output_accessor = AccessorRestrict<Value, 3, uint32_t>(output.get(), output_strides_2d);

        using real_t = traits::value_type_t<Value>;
        const auto symmetry_count = static_cast<int32_t>(symmetry.count());
        const auto scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;
        const auto symmetry_matrices = cuda::memory::PtrDevice<float33_t>::alloc(symmetry_count, stream);
        cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::geometry::details::transformSymmetry2D(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                cuda::utils::iwise3D("geometry::transform2D", output_shape_2d, kernel, stream);
                break;
            }
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::geometry::details::transformSymmetry2D(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                cuda::utils::iwise3D("geometry::transform2D", output_shape_2d, kernel, stream);
                break;
            }
            case INTERP_COSINE:
            case INTERP_COSINE_FAST: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::geometry::details::transformSymmetry2D(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                cuda::utils::iwise3D("geometry::transform2D", output_shape_2d, kernel, stream);
                break;
            }
            case INTERP_CUBIC: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::geometry::details::transformSymmetry2D(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                cuda::utils::iwise3D("geometry::transform2D", output_shape_2d, kernel, stream);
                break;
            }
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC_BSPLINE>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::geometry::details::transformSymmetry2D(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                cuda::utils::iwise3D("geometry::transform2D", output_shape_2d, kernel, stream);
                break;
            }
        }
        stream.attach(input, output, symmetry.share());
    }


    template<typename Value, typename>
    void transform3D(const shared_t<Value[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float3_t shift, float33_t inv_matrix, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream) {
        NOA_ASSERT(all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;

        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST))
            bspline::prefilter(input.get(), input_strides, input.get(), input_strides, input_shape, stream);

        const auto input_shape_3d = int3_t(input_shape.get(1));
        const auto output_shape_3d = int4_t(output_shape);
        const auto input_strides_3d = safe_cast<uint4_t>(input_strides);
        const auto output_strides_3d = safe_cast<uint4_t>(output_strides);
        const auto input_accessor = AccessorRestrict<const Value, 4, uint32_t>(input.get(), input_strides_3d);
        const auto output_accessor = AccessorRestrict<Value, 4, uint32_t>(output.get(), output_strides_3d);

        using real_t = traits::value_type_t<Value>;
        const auto symmetry_count = static_cast<int32_t>(symmetry.count());
        const auto scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;
        const auto symmetry_matrices = cuda::memory::PtrDevice<float33_t>::alloc(symmetry_count, stream);
        cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP_NEAREST>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::geometry::details::transformSymmetry3D(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                noa::cuda::utils::iwise4D("geometry::transform3D", output_shape_3d, kernel, stream);
                break;
            }
            case INTERP_LINEAR:
            case INTERP_LINEAR_FAST: {
                const auto interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP_LINEAR>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::geometry::details::transformSymmetry3D(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                noa::cuda::utils::iwise4D("geometry::transform3D", output_shape_3d, kernel, stream);
                break;
            }
            case INTERP_COSINE:
            case INTERP_COSINE_FAST: {
                const auto interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP_COSINE>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::geometry::details::transformSymmetry3D(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                noa::cuda::utils::iwise4D("geometry::transform3D", output_shape_3d, kernel, stream);
                break;
            }
            case INTERP_CUBIC: {
                const auto interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP_CUBIC>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::geometry::details::transformSymmetry3D(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                noa::cuda::utils::iwise4D("geometry::transform3D", output_shape_3d, kernel, stream);
                break;
            }
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST: {
                const auto interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP_CUBIC_BSPLINE>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::geometry::details::transformSymmetry3D(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                noa::cuda::utils::iwise4D("geometry::transform3D", output_shape_3d, kernel, stream);
                break;
            }
        }

        stream.attach(input, output, symmetry.share());
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
