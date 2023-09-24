#include "noa/core/Assert.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/algorithms/geometry/TransformRFFT.hpp"

#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/memory/AllocatorDevice.hpp"
#include "noa/gpu/cuda/geometry/fft/Transform.hpp"

namespace {
    using namespace ::noa;

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename ShiftOrEmpty>
    void launch_transform_rfft_final_(
            const AccessorRestrict<const Value, 3, u32>& input,
            const AccessorRestrict<Value, 3, u32>& output, const Shape4<i64>& shape,
            const Matrix& matrices, const ShiftOrEmpty& shifts, f32 cutoff,
            InterpMode interp_mode, noa::cuda::Stream& stream
    ) {
        NOA_ASSERT(shape[1] == 1);
        const auto i_shape = shape.as_safe<i32>();
        const auto input_shape_2d = i_shape.filter(2, 3).rfft();
        const auto output_shape = i_shape.filter(0, 2, 3).rfft();

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(
                        input, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_rfft_2d<REMAP>(
                        interpolator, output, shape, matrices, shifts, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST:
            case InterpMode::LINEAR: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                        input, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_rfft_2d<REMAP>(
                        interpolator, output, shape, matrices, shifts, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST:
            case InterpMode::COSINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(
                        input, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_rfft_2d<REMAP>(
                        interpolator, output, shape, matrices, shifts, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform_2d", "{} is not supported", interp_mode);
        }
    }

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename ShiftOrEmpty>
    void launch_transform_rfft_final_(
            const AccessorRestrict<const Value, 4, u32>& input,
            const AccessorRestrict<Value, 4, u32>& output, const Shape4<i64>& shape,
            const Matrix& matrices, const ShiftOrEmpty& shift, f32 cutoff,
            InterpMode interp_mode, noa::cuda::Stream& stream
    ) {
        const auto i_shape = shape.as_safe<i32>();
        const auto input_shape_3d = i_shape.pop_front().rfft();
        const auto output_shape = i_shape.rfft();

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::NEAREST>(
                        input, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_rfft_3d<REMAP>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST:
            case InterpMode::LINEAR: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::LINEAR>(
                        input, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_rfft_3d<REMAP>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST:
            case InterpMode::COSINE: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::COSINE>(
                        input, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_rfft_3d<REMAP>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform_3d", "{} is not supported", interp_mode);
        }
    }

    template<noa::fft::Remap REMAP, int32_t NDIM, typename Value, typename Matrix, typename Shift>
    void launch_transform_rfft_(
            const AccessorRestrict<const Value, NDIM + 1, u32>& input,
            const AccessorRestrict<Value, NDIM + 1, u32>& output, const Shape4<i64>& shape,
            const Matrix& matrices, const Shift& shift, f32 cutoff,
            InterpMode interp_mode, noa::cuda::Stream& stream
    ) {
        const bool do_shift = noa::any(shift != Shift{});
        if (do_shift)
            launch_transform_rfft_final_<REMAP>(
                    input, output, shape, matrices, shift, cutoff, interp_mode, stream);
        else
            launch_transform_rfft_final_<REMAP>(
                    input, output, shape, matrices, Empty{}, cutoff, interp_mode, stream);
    }

    template<noa::fft::Remap REMAP, typename Value, typename MatrixOrEmpty, typename ShiftOrEmpty>
    void launch_transform_symmetry_final_(
            const AccessorRestrict<const Value, 3, u32>& input,
            const AccessorRestrict<Value, 3, u32>& output, const Shape4<i64>& shape,
            const MatrixOrEmpty& matrix, const geometry::Symmetry& symmetry,
            const ShiftOrEmpty& shift, f32 cutoff, bool normalize,
            InterpMode interp_mode, noa::cuda::Stream& stream
    ) {
        NOA_ASSERT(shape[1] == 1);
        const auto input_shape_2d = shape.filter(2, 3).as_safe<i32>().rfft();
        const auto output_shape = shape.filter(0, 2, 3).as_safe<i32>().rfft();

        const auto symmetry_count = symmetry.count();
        const auto symmetry_matrices = noa::cuda::memory::AllocatorDevice<Float33>::allocate_async(symmetry_count, stream);
        noa::cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);
        using real_t = traits::value_type_t<Value>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(
                        input, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_2d<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices.get(),
                        symmetry_count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST:
            case InterpMode::LINEAR: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                        input, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_2d<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices.get(),
                        symmetry_count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST:
            case InterpMode::COSINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(
                        input, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_2d<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices.get(),
                        symmetry_count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform_2d", "{} is not supported", interp_mode);
        }
    }

    template<noa::fft::Remap REMAP, typename Value, typename MatrixOrEmpty, typename ShiftOrEmpty>
    void launch_transform_symmetry_final_(
            const AccessorRestrict<const Value, 4, u32>& input,
            const AccessorRestrict<Value, 4, u32>& output, const Shape4<i64>& shape,
            const MatrixOrEmpty& matrix, const geometry::Symmetry& symmetry,
            const ShiftOrEmpty& shift, f32 cutoff, bool normalize,
            InterpMode interp_mode, noa::cuda::Stream& stream
    ) {
        NOA_ASSERT(shape[1] == 1);
        const auto input_shape_3d = shape.pop_front().as_safe<i32>().rfft();
        const auto output_shape = shape.as_safe<i32>().rfft();

        const auto symmetry_count = symmetry.count();
        const auto symmetry_matrices = noa::cuda::memory::AllocatorDevice<Float33>::allocate_async(symmetry_count, stream);
        noa::cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);
        using real_t = traits::value_type_t<Value>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::NEAREST>(
                        input, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_3d<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices.get(),
                        symmetry_count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST:
            case InterpMode::LINEAR: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::LINEAR>(
                        input, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_3d<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices.get(),
                        symmetry_count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST:
            case InterpMode::COSINE: {
                const auto interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::COSINE>(
                        input, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_3d<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices.get(),
                        symmetry_count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform3D", "{} is not supported", interp_mode);
        }
    }

    template<noa::fft::Remap REMAP, int32_t NDIM, typename Value, typename Matrix, typename Shift>
    void launch_transform_symmetry_(
            const AccessorRestrict<const Value, NDIM + 1, u32>& input,
            const AccessorRestrict<Value, NDIM + 1, u32>& output, const Shape4<i64>& shape,
            const Matrix& matrix, const geometry::Symmetry& symmetry, const Shift& shift,
            f32 cutoff, bool normalize, InterpMode interp_mode, noa::cuda::Stream& stream
    ) {
        const bool apply_shift = noa::any(shift != Shift{});
        const bool apply_matrix = matrix != Matrix{};

        if (apply_shift && apply_matrix) {
            launch_transform_symmetry_final_<REMAP>(
                    input, output, shape,
                    matrix, symmetry, shift, cutoff,
                    normalize, interp_mode, stream);
        } else if (apply_shift) {
            launch_transform_symmetry_final_<REMAP>(
                    input, output, shape,
                    Empty{}, symmetry, shift, cutoff,
                    normalize, interp_mode, stream);
        } else if (apply_matrix) {
            launch_transform_symmetry_final_<REMAP>(
                    input, output, shape,
                    matrix, symmetry, Empty{}, cutoff,
                    normalize, interp_mode, stream);
        } else {
            launch_transform_symmetry_final_<REMAP>(
                    input, output, shape,
                    Empty{}, symmetry, Empty{}, cutoff,
                    normalize, interp_mode, stream);
        }
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform_2d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Matrix& inv_matrices, const Shift& shifts,
            f32 cutoff, InterpMode interp_mode, Stream& stream
    ) {
        NOA_ASSERT(noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        if constexpr (!nt::is_matXX_v<Matrix>) {
            NOA_ASSERT_DEVICE_PTR(inv_matrices, stream.device());
        }

        const auto input_accessor = AccessorRestrict<const Value, 3, u32>(input, input_strides.filter(0, 2, 3).as_safe<u32>());
        const auto output_accessor = AccessorRestrict<Value, 3, u32>(output, output_strides.filter(0, 2, 3).as_safe<u32>());
        launch_transform_rfft_<REMAP, 2>(
                input_accessor, output_accessor, shape,
                inv_matrices, shifts, cutoff, interp_mode, stream);
    }

    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform_3d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Matrix& inv_matrices, const Shift& shifts,
            f32 cutoff, InterpMode interp_mode, Stream& stream
    ) {
        NOA_ASSERT(noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        if constexpr (!nt::is_matXX_v<Matrix>) {
            NOA_ASSERT_DEVICE_PTR(inv_matrices, stream.device());
        }

        const auto input_accessor = AccessorRestrict<const Value, 4, uint32_t>(input, input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<Value, 4, uint32_t>(output, output_strides.as_safe<u32>());
        launch_transform_rfft_<REMAP, 3>(
                input_accessor, output_accessor, shape,
                inv_matrices, shifts, cutoff, interp_mode, stream);
    }

    template<Remap REMAP, typename Value, typename>
    void transform_and_symmetrize_2d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float22& inv_matrix, const Symmetry& symmetry, const Vec2<f32>& shift,
            f32 cutoff, InterpMode interp_mode, bool normalize, Stream& stream
    ) {
        if (!symmetry.count()) {
            return transform_2d<REMAP>(
                    input, input_strides, output, output_strides, shape,
                    inv_matrix, shift, cutoff, interp_mode, stream);
        }

        NOA_ASSERT(noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        const auto input_accessor = AccessorRestrict<const Value, 3, u32>(input, input_strides.filter(0, 2, 3).as_safe<u32>());
        const auto output_accessor = AccessorRestrict<Value, 3, u32>(output, output_strides.filter(0, 2, 3).as_safe<u32>());
        launch_transform_symmetry_<REMAP, 2>(
                input_accessor, output_accessor, shape,
                inv_matrix, symmetry, shift,
                cutoff, normalize, interp_mode, stream);
    }

    template<Remap REMAP, typename Value, typename>
    void transform_and_symmetrize_3d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float33& inv_matrix, const Symmetry& symmetry, const Vec3<f32>& shift,
            f32 cutoff, InterpMode interp_mode, bool normalize, Stream& stream
    ) {
        if (!symmetry.count()) {
            return transform_3d<REMAP>(
                    input, input_strides, output, output_strides, shape,
                    inv_matrix, shift, cutoff, interp_mode, stream);
        }

        NOA_ASSERT(noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        const auto input_accessor = AccessorRestrict<const Value, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<Value, 4, u32>(output, output_strides.as_safe<u32>());
        launch_transform_symmetry_<REMAP, 3>(
                input_accessor, output_accessor, shape,
                inv_matrix, symmetry, shift,
                cutoff, normalize, interp_mode, stream);
    }

    #define NOA_INSTANTIATE_TRANSFORM_SYMMETRY(T)                               \
    template void transform_and_symmetrize_2d<noa::fft::Remap::HC2HC, T, void>( \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Float22&, const Symmetry&,                    \
        const Vec2<f32>&, f32, InterpMode, bool, Stream&);                      \
    template void transform_and_symmetrize_2d<noa::fft::Remap::HC2H, T, void>(  \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Float22&, const Symmetry&,                    \
        const Vec2<f32>&, f32, InterpMode, bool, Stream&);                      \
    template void transform_and_symmetrize_3d<noa::fft::Remap::HC2HC, T, void>( \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Float33&, const Symmetry&,                    \
        const Vec3<f32>&, f32, InterpMode, bool, Stream&);                      \
    template void transform_and_symmetrize_3d<noa::fft::Remap::HC2H, T, void>(  \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Float33&, const Symmetry&,                    \
        const Vec3<f32>&, f32, InterpMode, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM2D_(T, M, S)                           \
    template void transform_2d<noa::fft::Remap::HC2H,  T, M, S, void>(      \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,           \
        const Shape4<i64>&, M const&, S const&, f32, InterpMode, Stream&);  \
    template void transform_2d<noa::fft::Remap::HC2HC, T, M, S, void>(      \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,           \
        const Shape4<i64>&, M const&, S const&, f32, InterpMode, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM3D_(T, M, S)                           \
    template void transform_3d<noa::fft::Remap::HC2H,  T, M, S, void>(      \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,           \
        const Shape4<i64>&, M const&, S const&, f32, InterpMode, Stream&);  \
    template void transform_3d<noa::fft::Remap::HC2HC, T, M, S, void>(      \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,           \
        const Shape4<i64>&, M const&, S const&, f32, InterpMode, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM2D_ALL_(T)                     \
    NOA_INSTANTIATE_TRANSFORM2D_(T, Float22, Vec2<f32>);            \
    NOA_INSTANTIATE_TRANSFORM2D_(T, const Float22*, Vec2<f32>);     \
    NOA_INSTANTIATE_TRANSFORM2D_(T, Float22, const Vec2<f32>*);     \
    NOA_INSTANTIATE_TRANSFORM2D_(T, const Float22*, const Vec2<f32>*)

    #define NOA_INSTANTIATE_TRANSFORM3D_ALL_(T)                     \
    NOA_INSTANTIATE_TRANSFORM3D_(T, Float33, Vec3<f32>);            \
    NOA_INSTANTIATE_TRANSFORM3D_(T, const Float33*, Vec3<f32>);     \
    NOA_INSTANTIATE_TRANSFORM3D_(T, Float33, const Vec3<f32>*);     \
    NOA_INSTANTIATE_TRANSFORM3D_(T, const Float33*, const Vec3<f32>*)

    #define NOA_INSTANTIATE_TRANSFORM_ALL_(T)   \
    NOA_INSTANTIATE_TRANSFORM2D_ALL_(T);
//    NOA_INSTANTIATE_TRANSFORM3D_ALL_(T);
//    NOA_INSTANTIATE_TRANSFORM_SYMMETRY(T)

    NOA_INSTANTIATE_TRANSFORM_ALL_(f32);
//    NOA_INSTANTIATE_TRANSFORM_ALL_(f64);
//    NOA_INSTANTIATE_TRANSFORM_ALL_(c32);
//    NOA_INSTANTIATE_TRANSFORM_ALL_(c64);
}
