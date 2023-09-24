#include "noa/core/Assert.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/algorithms/geometry/Transform.hpp"

#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/memory/AllocatorDevice.hpp"
#include "noa/gpu/cuda/geometry/Transform.hpp"

namespace {
    using namespace ::noa;

    template<InterpMode INTERP, typename Value, typename Matrix>
    void launch_linear_transform_final_(
            const AccessorRestrict<const Value, 3, u32>& input, const Shape2<i32>& input_shape,
            const AccessorRestrict<Value, 3, u32>& output, const Shape3<i32>& output_shape,
            const Matrix& inv_matrices, Value value, BorderMode border_mode,
            noa::cuda::Stream& stream
    ) {
        switch (border_mode) {
            case BorderMode::ZERO: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, INTERP>(input, input_shape, value);
                const auto kernel = noa::algorithm::geometry::transform_2d<u32>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case BorderMode::VALUE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::VALUE, INTERP>(input, input_shape, value);
                const auto kernel = noa::algorithm::geometry::transform_2d<u32>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case BorderMode::CLAMP: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::CLAMP, INTERP>(input, input_shape, value);
                const auto kernel = noa::algorithm::geometry::transform_2d<u32>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case BorderMode::PERIODIC: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::PERIODIC, INTERP>(input, input_shape, value);
                const auto kernel = noa::algorithm::geometry::transform_2d<u32>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case BorderMode::MIRROR: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::MIRROR, INTERP>(input, input_shape, value);
                const auto kernel = noa::algorithm::geometry::transform_2d<u32>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case BorderMode::REFLECT: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::REFLECT, INTERP>(input, input_shape, value);
                const auto kernel = noa::algorithm::geometry::transform_2d<u32>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case BorderMode::NOTHING:
                NOA_THROW_FUNC("transform_2d", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<InterpMode INTERP, typename Value, typename Matrix>
    void launch_linear_transform_final_(
            const AccessorRestrict<const Value, 4, u32>& input, const Shape3<i32>& input_shape,
            const AccessorRestrict<Value, 4, u32>& output, const Shape4<i32>& output_shape,
            const Matrix& inv_matrices, Value value, BorderMode border_mode,
            noa::cuda::Stream& stream
    ) {
        switch (border_mode) {
            case BorderMode::ZERO: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, INTERP>(input, input_shape, value);
                const auto kernel = noa::algorithm::geometry::transform_3d<u32>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case BorderMode::VALUE: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::VALUE, INTERP>(input, input_shape, value);
                const auto kernel = noa::algorithm::geometry::transform_3d<u32>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case BorderMode::CLAMP: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::CLAMP, INTERP>(input, input_shape, value);
                const auto kernel = noa::algorithm::geometry::transform_3d<u32>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case BorderMode::PERIODIC: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::PERIODIC, INTERP>(input, input_shape, value);
                const auto kernel = noa::algorithm::geometry::transform_3d<u32>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case BorderMode::MIRROR: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::MIRROR, INTERP>(input, input_shape, value);
                const auto kernel = noa::algorithm::geometry::transform_3d<u32>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case BorderMode::REFLECT: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::REFLECT, INTERP>(input, input_shape, value);
                const auto kernel = noa::algorithm::geometry::transform_3d<u32>(interpolator, output, inv_matrices);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case BorderMode::NOTHING:
                NOA_THROW_FUNC("transform_3d", "The border/addressing mode {} is not supported", border_mode);
        }
    }

    template<i32 NDIM, typename Value, typename IShape, typename OShape, typename Matrix>
    void launch_linear_transform_(
            const AccessorRestrict<const Value, NDIM + 1, u32>& input, const IShape& input_shape,
            const AccessorRestrict<Value, NDIM + 1, u32>& output, const OShape& output_shape,
            const Matrix& inv_matrices, Value value, InterpMode interp_mode,
            BorderMode border_mode, noa::cuda::Stream& stream
    ) {
        switch (interp_mode) {
            case InterpMode::NEAREST:
                return launch_linear_transform_final_<InterpMode::NEAREST>(
                        input, input_shape, output, output_shape,
                        inv_matrices, value, border_mode, stream);
            case InterpMode::LINEAR:
            case InterpMode::LINEAR_FAST:
                return launch_linear_transform_final_<InterpMode::LINEAR>(
                        input, input_shape, output, output_shape,
                        inv_matrices, value, border_mode, stream);
            case InterpMode::COSINE:
            case InterpMode::COSINE_FAST:
                return launch_linear_transform_final_<InterpMode::COSINE>(
                        input, input_shape, output, output_shape,
                        inv_matrices, value, border_mode, stream);
            case InterpMode::CUBIC:
                return launch_linear_transform_final_<InterpMode::CUBIC>(
                        input, input_shape, output, output_shape,
                        inv_matrices, value, border_mode, stream);
            case InterpMode::CUBIC_BSPLINE:
            case InterpMode::CUBIC_BSPLINE_FAST:
                return launch_linear_transform_final_<InterpMode::CUBIC_BSPLINE>(
                        input, input_shape, output, output_shape,
                        inv_matrices, value, border_mode, stream);
        }
    }

    template<typename Value>
    void launch_symmetry_(
            const Value* input, const Strides4<u32>& input_strides,
            Value* output, const Strides4<u32>& output_strides, const Shape4<i32>& shape,
            const geometry::Symmetry& symmetry, const Vec2<f32>& center,
            InterpMode interp_mode, bool normalize, noa::cuda::Stream& stream
    ) {
        using real_t = nt::value_type_t<Value>;
        const auto symmetry_count = symmetry.count();
        const auto scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;
        const auto symmetry_matrices = noa::cuda::memory::AllocatorDevice<Float33>::allocate_async(symmetry_count, stream);
        noa::cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);

        const auto input_shape_2d = shape.filter(2, 3);
        const auto output_shape_2d = shape.filter(0, 2, 3);
        const auto input_accessor = AccessorRestrict<const Value, 3, u32>(input, input_strides.filter(0, 2, 3));
        const auto output_accessor = AccessorRestrict<Value, 3, u32>(output, output_strides.filter(0, 2, 3));

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::symmetry_2d<u32>(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d(output_shape_2d, kernel, stream);
            }
            case InterpMode::LINEAR:
            case InterpMode::LINEAR_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::symmetry_2d<u32>(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d(output_shape_2d, kernel, stream);
            }
            case InterpMode::COSINE:
            case InterpMode::COSINE_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::symmetry_2d<u32>(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d(output_shape_2d, kernel, stream);
            }
            case InterpMode::CUBIC: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::CUBIC>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::symmetry_2d<u32>(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d(output_shape_2d, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE:
            case InterpMode::CUBIC_BSPLINE_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::CUBIC_BSPLINE>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::symmetry_2d<u32>(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d(output_shape_2d, kernel, stream);
            }
        }
    }

    template<typename Value>
    void launch_symmetry_(
            const Value* input, const Strides4<u32>& input_strides,
            Value* output, const Strides4<u32>& output_strides, const Shape4<i32>& shape,
            const geometry::Symmetry& symmetry, const Vec3<f32>& center,
            InterpMode interp_mode, bool normalize, noa::cuda::Stream& stream
    ) {
        using real_t = traits::value_type_t<Value>;
        const auto symmetry_count = symmetry.count();
        const auto scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;
        const auto symmetry_matrices = noa::cuda::memory::AllocatorDevice<Float33>::allocate_async(symmetry_count, stream);
        noa::cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);

        const auto input_shape_3d = shape.pop_front();
        const auto input_accessor = AccessorRestrict<const Value, 4, u32>(input, input_strides);
        const auto output_accessor = AccessorRestrict<Value, 4, u32>(output, output_strides);

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::NEAREST>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::symmetry_3d<u32>(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d(shape, kernel, stream);
            }
            case InterpMode::LINEAR:
            case InterpMode::LINEAR_FAST: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::LINEAR>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::symmetry_3d<u32>(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d(shape, kernel, stream);
            }
            case InterpMode::COSINE:
            case InterpMode::COSINE_FAST: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::COSINE>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::symmetry_3d<u32>(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d(shape, kernel, stream);
            }
            case InterpMode::CUBIC: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::CUBIC>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::symmetry_3d<u32>(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d(shape, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE:
            case InterpMode::CUBIC_BSPLINE_FAST: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::CUBIC_BSPLINE>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::symmetry_3d<u32>(
                        interpolator, output_accessor, center, symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d(shape, kernel, stream);
            }
        }
    }

    template<typename Value, typename Center>
    void symmetrize_nd_(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const geometry::Symmetry& symmetry, const Center& center,
            InterpMode interp_mode, bool normalize, noa::cuda::Stream& stream
    ) {
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(input != output && noa::all(shape > 0));
        NOA_ASSERT((std::is_same_v<Center, Vec3<f32>> && shape.ndim() <= 3) ||
                   (std::is_same_v<Center, Vec2<f32>> && shape.ndim() <= 2));

        if (!symmetry.count())
            return noa::cuda::memory::copy(input, input_strides, output, output_strides, shape, stream);

        launch_symmetry_<Value>(
                input, input_strides.as_safe<u32>(),
                output, output_strides.as_safe<u32>(), shape.as_safe<i32>(),
                symmetry, center, interp_mode, normalize, stream);
    }

    template<typename Matrix>
    auto truncated_matrix_or_const_ptr_(const Matrix& matrix) {
        if constexpr (nt::is_mat33_v<Matrix> || nt::is_mat44_v<Matrix>) {
            return noa::geometry::affine2truncated(matrix);
        } else if constexpr (nt::is_matXX_v<Matrix>) {
            return matrix;
        } else {
            NOA_ASSERT(matrix != nullptr);
            return matrix;
        }
    }
}

namespace noa::cuda::geometry {
    template<typename Value, typename Matrix, typename>
    void transform_2d(
            const Value* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Matrix& inv_matrices, InterpMode interp_mode, BorderMode border_mode,
            Value cvalue, Stream& stream
    ) {
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(input != output);
        NOA_ASSERT(noa::all(input_shape > 0) && noa::all(output_shape > 0));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1 && output_shape[1] == 1);
        if constexpr (!nt::is_matXX_v<Matrix>) {
            NOA_ASSERT_DEVICE_PTR(inv_matrices, stream.device());
        }

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;

        launch_linear_transform_<2>(
                {input, input_strides.filter(0, 2, 3).as_safe<u32>()}, input_shape.filter(2, 3).as_safe<i32>(),
                {output, output_strides.filter(0, 2, 3).as_safe<u32>()}, output_shape.filter(0, 2, 3).as_safe<i32>(),
                truncated_matrix_or_const_ptr_(inv_matrices), cvalue,
                interp_mode, border_mode, stream);
    }

    template<typename Value, typename Matrix, typename>
    void transform_3d(
            const Value* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Matrix& inv_matrices, InterpMode interp_mode, BorderMode border_mode,
            Value cvalue, Stream& stream
    ) {
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(input != output);
        NOA_ASSERT(noa::all(input_shape > 0) && noa::all(output_shape > 0));
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        if constexpr (!nt::is_matXX_v<Matrix>) {
            NOA_ASSERT_DEVICE_PTR(inv_matrices, stream.device());
        }

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;

        launch_linear_transform_<3>(
                {input, input_strides.as_safe<u32>()}, input_shape.pop_front().as_safe<i32>(),
                {output, output_strides.as_safe<u32>()}, output_shape.as_safe<i32>(),
                truncated_matrix_or_const_ptr_(inv_matrices), cvalue,
                interp_mode, border_mode, stream);
    }

    template<typename Value, typename>
    void transform_and_symmetrize_2d(
            const Value* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Vec2<f32>& shift, const Float22& inv_matrix,
            const Symmetry& symmetry, const Vec2<f32>& center,
            InterpMode interp_mode, bool normalize, Stream& stream
    ) {
        NOA_ASSERT(noa::all(input_shape > 0) && noa::all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1 && output_shape[1] == 1);

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;

        const auto input_shape_2d = input_shape.filter(2, 3).as_safe<i32>();
        const auto output_shape_2d = output_shape.filter(0, 2, 3).as_safe<i32>();
        const auto input_accessor = AccessorRestrict<const Value, 3, u32>(input, input_strides.filter(0, 2, 3).as_safe<u32>());
        const auto output_accessor = AccessorRestrict<Value, 3, u32>(output, output_strides.filter(0, 2, 3).as_safe<u32>());

        using real_t = nt::value_type_t<Value>;
        const auto symmetry_count = symmetry.count();
        const auto scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;
        const auto symmetry_matrices = noa::cuda::memory::AllocatorDevice<Float33>::allocate_async(symmetry_count, stream);
        noa::cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_2d<u32>(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d(output_shape_2d, kernel, stream);
            }
            case InterpMode::LINEAR:
            case InterpMode::LINEAR_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_2d<u32>(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d(output_shape_2d, kernel, stream);
            }
            case InterpMode::COSINE:
            case InterpMode::COSINE_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_2d<u32>(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d(output_shape_2d, kernel, stream);
            }
            case InterpMode::CUBIC: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::CUBIC>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_2d<u32>(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d(output_shape_2d, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE:
            case InterpMode::CUBIC_BSPLINE_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::CUBIC_BSPLINE>(
                        input_accessor, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_2d<u32>(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d(output_shape_2d, kernel, stream);
            }
        }
    }

    template<typename Value, typename>
    void transform_and_symmetrize_3d(
            const Value* input, Strides4<i64> input_strides, Shape4<i64> input_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Vec3<f32>& shift, const Float33& inv_matrix,
            const Symmetry& symmetry, const Vec3<f32>& center,
            InterpMode interp_mode, bool normalize, Stream& stream
    ) {
        NOA_ASSERT(noa::all(input_shape > 0) && noa::all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

        // Broadcast the input to every output batch.
        if (input_shape[0] == 1)
            input_strides[0] = 0;
        else if (input_strides[0] == 0)
            input_shape[0] = 1;

        const auto input_shape_3d = input_shape.pop_front().as_safe<i32>();
        const auto output_shape_3d = output_shape.as_safe<i32>();
        const auto input_accessor = AccessorRestrict<const Value, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<Value, 4, u32>(output, output_strides.as_safe<u32>());

        using real_t = nt::value_type_t<Value>;
        const auto symmetry_count = symmetry.count();
        const auto scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;
        const auto symmetry_matrices = noa::cuda::memory::AllocatorDevice<Float33>::allocate_async(symmetry_count, stream);
        noa::cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::NEAREST>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_3d<u32>(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d(output_shape_3d, kernel, stream);
            }
            case InterpMode::LINEAR:
            case InterpMode::LINEAR_FAST: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::LINEAR>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_3d<u32>(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d(output_shape_3d, kernel, stream);
            }
            case InterpMode::COSINE:
            case InterpMode::COSINE_FAST: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::COSINE>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_3d<u32>(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d(output_shape_3d, kernel, stream);
            }
            case InterpMode::CUBIC: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::CUBIC>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_3d<u32>(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d(output_shape_3d, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE:
            case InterpMode::CUBIC_BSPLINE_FAST: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::CUBIC_BSPLINE>(
                        input_accessor, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_3d<u32>(
                        interpolator, output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d(output_shape_3d, kernel, stream);
            }
        }
    }

    template<typename Value, typename>
    void symmetrize_2d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, const Symmetry& symmetry, const Vec2<f32>& center,
            InterpMode interp_mode, bool normalize, Stream& stream
    ) {
        symmetrize_nd_(input, input_strides, output, output_strides, shape,
                       symmetry, center, interp_mode, normalize, stream);
    }

    template<typename Value, typename>
    void symmetrize_3d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, const Symmetry& symmetry, const Vec3<f32>& center,
            InterpMode interp_mode, bool normalize, Stream& stream
    ) {
        symmetrize_nd_(input, input_strides, output, output_strides, shape,
                       symmetry, center, interp_mode, normalize, stream);
    }

    #define NOA_INSTANTIATE_SYMMETRY_(T)                \
    template void symmetrize_2d<T, void>(               \
        const T*, const Strides4<i64>&,                 \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const Symmetry&, const Vec2<f32>&,              \
        InterpMode, bool, Stream&);                     \
    template void symmetrize_3d<T, void>(               \
        const T*, const Strides4<i64>&,                 \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const Symmetry&, const Vec3<f32>&,              \
        InterpMode, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM2D_(T, M)          \
    template void transform_2d<T, M, void>(             \
        const T*, Strides4<i64>, Shape4<i64>,           \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        M const&, InterpMode, BorderMode, T, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM3D_(T, M)          \
    template void transform_3d<T, M, void>(             \
        const T*, Strides4<i64>, Shape4<i64>,           \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        M const&, InterpMode, BorderMode, T, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM2D_ALL_(T)         \
    NOA_INSTANTIATE_TRANSFORM2D_(T, Float23);           \
    NOA_INSTANTIATE_TRANSFORM2D_(T, Float33);           \
    NOA_INSTANTIATE_TRANSFORM2D_(T, const Float23*);    \
    NOA_INSTANTIATE_TRANSFORM2D_(T, const Float33*)

    #define NOA_INSTANTIATE_TRANSFORM3D_ALL_(T)         \
    NOA_INSTANTIATE_TRANSFORM3D_(T, Float34);           \
    NOA_INSTANTIATE_TRANSFORM3D_(T, Float44);           \
    NOA_INSTANTIATE_TRANSFORM3D_(T, const Float34*);    \
    NOA_INSTANTIATE_TRANSFORM3D_(T, const Float44*)

    #define NOA_INSTANTIATE_TRANSFORM_SYMMETRY_(T)      \
    template void transform_and_symmetrize_2d<T, void>( \
        const T*, Strides4<i64>, Shape4<i64>,           \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const Vec2<f32>&, const Float22&,               \
        const Symmetry&, const Vec2<f32>&,              \
        InterpMode, bool, Stream&);                     \
    template void transform_and_symmetrize_3d<T, void>( \
        const T*, Strides4<i64>, Shape4<i64>,           \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const Vec3<f32>&, const Float33&,               \
        const Symmetry&, const Vec3<f32>&,              \
        InterpMode, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_ALL_(T)   \
    NOA_INSTANTIATE_TRANSFORM2D_ALL_(T);        \
    NOA_INSTANTIATE_TRANSFORM3D_ALL_(T);
//    NOA_INSTANTIATE_TRANSFORM_SYMMETRY_(T);     \
//    NOA_INSTANTIATE_SYMMETRY_(T)

    NOA_INSTANTIATE_TRANSFORM_ALL_(f32);
//    NOA_INSTANTIATE_TRANSFORM_ALL_(f64);
//    NOA_INSTANTIATE_TRANSFORM_ALL_(c32);
//    NOA_INSTANTIATE_TRANSFORM_ALL_(c64);
}
