#include "noa/core/Assert.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/algorithms/geometry/TransformRFFT.hpp"

#include "noa/cpu/geometry/fft/Transform.h"
#include "noa/cpu/utils/Iwise.hpp"

namespace {
    using namespace ::noa;

    template<fft::Remap REMAP, typename Value, typename Matrix, typename ShiftOrEmpty>
    void launch_transform_rfft_final_(
            const AccessorRestrict<const Value, 3, i64>& input,
            const AccessorRestrict<Value, 3, i64>& output, const Shape4<i64>& shape,
            Matrix matrices, ShiftOrEmpty shift, f32 cutoff,
            InterpMode interp_mode, i64 threads) {
        NOA_ASSERT(shape[1] == 1);
        const auto input_shape_2d = shape.filter(2, 3).fft();
        const auto iwise_shape = shape.filter(0, 2, 3).fft();

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator =
                        noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(input, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_rfft_2d<REMAP>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::LINEAR_FAST:
            case InterpMode::LINEAR: {
                const auto interpolator =
                        noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(input, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_rfft_2d<REMAP>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::COSINE_FAST:
            case InterpMode::COSINE: {
                const auto interpolator =
                        noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(input, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_rfft_2d<REMAP>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            default:
                NOA_THROW_FUNC("transform_2d", "{} is not supported", interp_mode);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename ShiftOrEmpty>
    void launch_transform_rfft_final_(
            const AccessorRestrict<const Value, 4, i64>& input,
            const AccessorRestrict<Value, 4, i64>& output, const Shape4<i64>& shape,
            Matrix matrices, ShiftOrEmpty shift, f32 cutoff,
            InterpMode interp_mode, i64 threads) {
        const auto input_shape_3d = shape.pop_front().fft();
        const auto iwise_shape = shape.fft();

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator =
                        noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::NEAREST>(input, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_rfft_3d<REMAP, i64>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case InterpMode::LINEAR_FAST:
            case InterpMode::LINEAR: {
                const auto interpolator =
                        noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::LINEAR>(input, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_rfft_3d<REMAP, i64>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case InterpMode::COSINE_FAST:
            case InterpMode::COSINE: {
                const auto interpolator =
                        noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::COSINE>(input, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_rfft_3d<REMAP, i64>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            default:
                NOA_THROW_FUNC("transform_3d", "{} is not supported", interp_mode);
        }
    }

    template<fft::Remap REMAP, size_t NDIM, typename Value, typename Matrix, typename Shift>
    void launch_transform_rfft_(
            const AccessorRestrict<const Value, NDIM + 1, i64>& input,
            const AccessorRestrict<Value, NDIM + 1, i64>& output,
            const Shape4<i64>& shape,
            Matrix matrices, Shift shift, f32 cutoff,
            InterpMode interp_mode, i64 threads) {
        const bool do_shift = noa::any(shift != Shift{});
        if (do_shift) {
            launch_transform_rfft_final_<REMAP>(
                    input, output, shape, matrices,
                    shift, cutoff, interp_mode, threads);
        } else {
            launch_transform_rfft_final_<REMAP>(
                    input, output, shape, matrices,
                    Empty{}, cutoff, interp_mode, threads);
        }
    }

    template<fft::Remap REMAP, typename Value, typename MatrixOrEmpty, typename ShiftOrEmpty>
    void launch_transform_symmetry_rfft_final_(
            const AccessorRestrict<const Value, 3, i64>& input,
            const AccessorRestrict<Value, 3, i64>& output, const Shape4<i64>& shape,
            MatrixOrEmpty matrix, const geometry::Symmetry& symmetry,
            ShiftOrEmpty shift, f32 cutoff, bool normalize,
            InterpMode interp_mode, i64 threads) {
        NOA_ASSERT(shape[1] == 1);
        const auto input_shape_2d = shape.filter(2, 3).fft();
        const auto iwise_shape = shape.filter(0, 2, 3).fft();

        const auto symmetry_count = symmetry.count();
        const Float33* symmetry_matrices = symmetry.get();
        using real_t = noa::traits::value_type_t<Value>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator =
                        noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(input, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_2d<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::LINEAR_FAST:
            case InterpMode::LINEAR: {
                const auto interpolator =
                        noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(input, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_2d<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::COSINE_FAST:
            case InterpMode::COSINE: {
                const auto interpolator =
                        noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(input, input_shape_2d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_2d<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            default:
                NOA_THROW_FUNC("transform_2d", "{} is not supported", interp_mode);
        }
    }

    template<fft::Remap REMAP, typename Value, typename MatrixOrEmpty, typename ShiftOrEmpty>
    void launch_transform_symmetry_rfft_final_(
            const AccessorRestrict<const Value, 4, i64>& input,
            const AccessorRestrict<Value, 4, i64>& output, const Shape4<i64>& shape,
            MatrixOrEmpty matrix, const geometry::Symmetry& symmetry,
            ShiftOrEmpty shift, f32 cutoff, bool normalize,
            InterpMode interp_mode, i64 threads) {
        NOA_ASSERT(shape[1] == 1);
        const auto input_shape_3d = shape.pop_front().fft();
        const auto iwise_shape = shape.fft();

        const auto symmetry_count = symmetry.count();
        const Float33* symmetry_matrices = symmetry.get();
        using real_t = noa::traits::value_type_t<Value>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator =
                        noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::NEAREST>(input, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_3d<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case InterpMode::LINEAR_FAST:
            case InterpMode::LINEAR: {
                const auto interpolator =
                        noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::LINEAR>(input, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_3d<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case InterpMode::COSINE_FAST:
            case InterpMode::COSINE: {
                const auto interpolator =
                        noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::COSINE>(input, input_shape_3d);
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_3d<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            default:
                NOA_THROW_FUNC("transform_3d", "{} is not supported", interp_mode);
        }
    }

    template<fft::Remap REMAP, size_t NDIM, typename Value, typename Matrix, typename Shift>
    void launch_transform_symmetry_rfft_(
            const AccessorRestrict<const Value, NDIM + 1, i64>& input,
            const AccessorRestrict<Value, NDIM + 1, i64>& output, const Shape4<i64>& shape,
            Matrix matrix, const geometry::Symmetry& symmetry, Shift shift,
            f32 cutoff, bool normalize, InterpMode interp_mode, i64 threads) {
        const bool apply_shift = noa::any(shift != Shift{});
        const bool apply_matrix = matrix != Matrix{};

        if (apply_shift && apply_matrix) {
            launch_transform_symmetry_rfft_final_<REMAP>(
                    input, output, shape,
                    matrix, symmetry, shift, cutoff,
                    normalize, interp_mode, threads);
        } else if (apply_shift) {
            launch_transform_symmetry_rfft_final_<REMAP>(
                    input, output, shape,
                    Empty{}, symmetry, shift, cutoff,
                    normalize, interp_mode, threads);
        } else if (apply_matrix) {
            launch_transform_symmetry_rfft_final_<REMAP>(
                    input, output, shape,
                    matrix, symmetry, Empty{}, cutoff,
                    normalize, interp_mode, threads);
        } else {
            launch_transform_symmetry_rfft_final_<REMAP>(
                    input, output, shape,
                    Empty{}, symmetry, Empty{}, cutoff,
                    normalize, interp_mode, threads);
        }
    }
}

namespace noa::cpu::geometry::fft {
    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform_2d(const Value* input, const Strides4<i64>& input_strides,
                      Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                      const Matrix& inv_matrices, const Shift& shifts,
                      f32 cutoff, InterpMode interp_mode, i64 threads) {
        NOA_ASSERT(input && output && input != output && noa::all(shape > 0));
        NOA_ASSERT(!noa::indexing::are_overlapped(input, input_strides, output, output_strides, shape.fft()));

        const auto input_accessor = AccessorRestrict<const Value, 3, i64>(input, input_strides.filter(0, 2, 3));
        const auto output_accessor = AccessorRestrict<Value, 3, i64>(output, output_strides.filter(0, 2, 3));
        launch_transform_rfft_<REMAP, 2>(
                input_accessor, output_accessor, shape,
                inv_matrices, shifts, cutoff, interp_mode, threads);
    }

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform_3d(const Value* input, const Strides4<i64>& input_strides,
                      Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                      const Matrix& inv_matrices, const Shift& shifts,
                      f32 cutoff, InterpMode interp_mode, i64 threads) {
        NOA_ASSERT(input && output && input != output && noa::all(shape > 0));
        NOA_ASSERT(!noa::indexing::are_overlapped(input, input_strides, output, output_strides, shape.fft()));

        const auto input_accessor = AccessorRestrict<const Value, 4, i64>(input, input_strides);
        const auto output_accessor = AccessorRestrict<Value, 4, i64>(output, output_strides);
        launch_transform_rfft_<REMAP, 3>(
                input_accessor, output_accessor, shape,
                inv_matrices, shifts, cutoff, interp_mode, threads);
    }

    template<noa::fft::Remap REMAP, typename Value, typename>
    void transform_and_symmetrize_2d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float22& inv_matrix, const Symmetry& symmetry, const Vec2<f32>& shift,
            f32 cutoff, InterpMode interp_mode, bool normalize, i64 threads) {
        NOA_ASSERT(input && output && input != output && all(shape > 0));
        NOA_ASSERT(!noa::indexing::are_overlapped(input, input_strides, output, output_strides, shape.fft()));

        if (!symmetry.count()) {
            return transform_2d<REMAP>(
                    input, input_strides, output, output_strides,
                    shape, inv_matrix, shift, cutoff, interp_mode, threads);
        }

        const auto input_accessor = AccessorRestrict<const Value, 3, i64>(input, input_strides.filter(0, 2, 3));
        const auto output_accessor = AccessorRestrict<Value, 3, i64>(output, output_strides.filter(0, 2, 3));
        launch_transform_symmetry_rfft_<REMAP, 2>(
                input_accessor, output_accessor, shape,
                inv_matrix, symmetry, shift, cutoff, normalize, interp_mode, threads);
    }

    template<noa::fft::Remap REMAP, typename Value, typename>
    void transform_and_symmetrize_3d(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float33& inv_matrix, const Symmetry& symmetry, const Vec3<f32>& shift,
            f32 cutoff, InterpMode interp_mode, bool normalize, i64 threads) {
        NOA_ASSERT(input && output && input != output && all(shape > 0));
        NOA_ASSERT(!noa::indexing::are_overlapped(input, input_strides, output, output_strides, shape.fft()));

        if (!symmetry.count()) {
            return transform_3d<REMAP>(
                    input, input_strides, output, output_strides,
                    shape, inv_matrix, shift, cutoff, interp_mode, threads);
        }

        const AccessorRestrict<const Value, 4, i64> input_accessor(input, input_strides);
        const AccessorRestrict<Value, 4, i64> output_accessor(output, output_strides);
        launch_transform_symmetry_rfft_<REMAP, 3>(
                input_accessor, output_accessor, shape,
                inv_matrix, symmetry, shift, cutoff, normalize, interp_mode, threads);
    }

    #define NOA_INSTANTIATE_TRANSFORM_SYMMETRY(T)                               \
    template void transform_and_symmetrize_2d<noa::fft::Remap::HC2HC, T, void>( \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Float22&, const Symmetry&,                    \
        const Vec2<f32>&, f32, InterpMode, bool, i64);                          \
    template void transform_and_symmetrize_2d<noa::fft::Remap::HC2H, T, void>(  \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Float22&, const Symmetry&,                    \
        const Vec2<f32>&, f32, InterpMode, bool, i64);                          \
    template void transform_and_symmetrize_3d<noa::fft::Remap::HC2HC, T, void>( \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Float33&, const Symmetry&,                    \
        const Vec3<f32>&, f32, InterpMode, bool, i64);                          \
    template void transform_and_symmetrize_3d<noa::fft::Remap::HC2H, T, void>(  \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Float33&, const Symmetry&,                    \
        const Vec3<f32>&, f32, InterpMode, bool, i64)

    #define NOA_INSTANTIATE_TRANSFORM2D_(T, M, S)                       \
    template void transform_2d<noa::fft::Remap::HC2H,  T, M, S, void>(  \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,       \
        const Shape4<i64>&, M const&, S const&, f32, InterpMode, i64);  \
    template void transform_2d<noa::fft::Remap::HC2HC, T, M, S, void>(  \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,       \
        const Shape4<i64>&, M const&, S const&, f32, InterpMode, i64)

    #define NOA_INSTANTIATE_TRANSFORM3D_(T, M, S)                       \
    template void transform_3d<noa::fft::Remap::HC2H,  T, M, S, void>(  \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,       \
        const Shape4<i64>&, M const&, S const&, f32, InterpMode, i64);  \
    template void transform_3d<noa::fft::Remap::HC2HC, T, M, S, void>(  \
        const T*, const Strides4<i64>&, T*, const Strides4<i64>&,       \
        const Shape4<i64>&, M const&, S const&, f32, InterpMode, i64)

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
    NOA_INSTANTIATE_TRANSFORM2D_ALL_(T);        \
    NOA_INSTANTIATE_TRANSFORM3D_ALL_(T);        \
    NOA_INSTANTIATE_TRANSFORM_SYMMETRY(T)

    NOA_INSTANTIATE_TRANSFORM_ALL_(f32);
    NOA_INSTANTIATE_TRANSFORM_ALL_(f64);
    NOA_INSTANTIATE_TRANSFORM_ALL_(c32);
    NOA_INSTANTIATE_TRANSFORM_ALL_(c64);
}
