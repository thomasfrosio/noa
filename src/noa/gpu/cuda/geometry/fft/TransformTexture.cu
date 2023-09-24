#include "noa/core/Assert.hpp"
#include "noa/algorithms/geometry/TransformRFFT.hpp"

#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/memory/AllocatorArray.hpp"
#include "noa/gpu/cuda/memory/AllocatorDevice.hpp"
#include "noa/gpu/cuda/memory/AllocatorTexture.hpp"
#include "noa/gpu/cuda/geometry/Interpolator.hpp"
#include "noa/gpu/cuda/geometry/fft/Transform.hpp"

namespace {
    using namespace ::noa;

    template<noa::fft::Remap REMAP, bool LAYERED, typename Value, typename Matrix, typename ShiftOrEmpty>
    void launch_transform_rfft_2d_final_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Matrix& matrix, const ShiftOrEmpty& shift, f32 cutoff,
            noa::cuda::Stream& stream
    ) {
        NOA_ASSERT(shape[1] == 1);
        const auto i_shape = shape.as_safe<i32>();
        const auto output_shape = i_shape.filter(0, 2, 3).rfft();
        const auto output_accessor = AccessorRestrict<Value, 3, i32>(output, output_strides.filter(0, 2, 3).as_safe<i32>());

        switch (texture_interp_mode) {
            case InterpMode::NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_rfft_2d<REMAP>(
                        interpolator_t(texture), output_accessor, i_shape, matrix, shift, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case InterpMode::LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_rfft_2d<REMAP>(
                        interpolator_t(texture), output_accessor, i_shape, matrix, shift, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case InterpMode::COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_rfft_2d<REMAP>(
                        interpolator_t(texture), output_accessor, i_shape, matrix, shift, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_rfft_2d<REMAP>(
                        interpolator_t(texture), output_accessor, i_shape, matrix, shift, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_rfft_2d<REMAP>(
                        interpolator_t(texture), output_accessor, i_shape, matrix, shift, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform_2d", "{} is not supported", texture_interp_mode);
        }
    }

    template<noa::fft::Remap REMAP, bool LAYERED, typename Value, typename Matrix, typename Shift>
    void launch_transform_rfft_2d_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Matrix& matrix, const Shift& shift, f32 cutoff,
            noa::cuda::Stream& stream
    ) {
        const bool do_shift = noa::any(shift != Shift{});
        if (do_shift) {
            launch_transform_rfft_2d_final_<REMAP, LAYERED>(
                    texture, texture_interp_mode,
                    output, output_strides, shape,
                    matrix, shift, cutoff, stream);
        } else {
            launch_transform_rfft_2d_final_<REMAP, LAYERED>(
                    texture, texture_interp_mode,
                    output, output_strides, shape,
                    matrix, Empty{}, cutoff, stream);
        }
    }

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename ShiftOrEmpty>
    void launch_transform_rfft_3d_final_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Matrix& inv_matrix, const ShiftOrEmpty& shift, f32 cutoff,
            noa::cuda::Stream& stream
    ) {
        const auto i_shape = shape.as_safe<i32>();
        const auto output_shape = i_shape.rfft();
        const auto output_accessor = AccessorRestrict<Value, 4, i32>(output, output_strides.as_safe<i32>());

        switch (texture_interp_mode) {
            case InterpMode::NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::NEAREST, Value>;
                const auto kernel = noa::algorithm::geometry::transform_rfft_3d<REMAP>(
                        interpolator_t(texture), output_accessor, i_shape, inv_matrix, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case InterpMode::LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::LINEAR, Value>;
                const auto kernel = noa::algorithm::geometry::transform_rfft_3d<REMAP>(
                        interpolator_t(texture), output_accessor, i_shape, inv_matrix, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case InterpMode::COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::COSINE, Value>;
                const auto kernel = noa::algorithm::geometry::transform_rfft_3d<REMAP>(
                        interpolator_t(texture), output_accessor, i_shape, inv_matrix, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::LINEAR_FAST, Value>;
                const auto kernel = noa::algorithm::geometry::transform_rfft_3d<REMAP>(
                        interpolator_t(texture), output_accessor, i_shape, inv_matrix, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::COSINE_FAST, Value>;
                const auto kernel = noa::algorithm::geometry::transform_rfft_3d<REMAP>(
                        interpolator_t(texture), output_accessor, i_shape, inv_matrix, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform_3d", "{} is not supported", texture_interp_mode);
        }
    }

    template<noa::fft::Remap REMAP, typename Value, typename Matrix, typename Shift>
    void launch_transform_rfft_3d_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Matrix& inv_matrix, const Shift& shift, f32 cutoff,
            noa::cuda::Stream& stream
    ) {
        const bool do_shift = noa::any(shift != Shift{});
        if (do_shift) {
            launch_transform_rfft_3d_final_<REMAP>(
                    texture, texture_interp_mode,
                    output, output_strides, shape,
                    inv_matrix, shift, cutoff, stream);
        } else {
            launch_transform_rfft_3d_final_<REMAP>(
                    texture, texture_interp_mode,
                    output, output_strides, shape,
                    inv_matrix, Empty{}, cutoff, stream);
        }
    }

    template<noa::fft::Remap REMAP, bool LAYERED, typename Value, typename MatrixOrEmpty, typename ShiftOrEmpty>
    void launch_transform_symmetry_rfft_2d_final_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const MatrixOrEmpty& matrix, const geometry::Symmetry& symmetry,
            const ShiftOrEmpty& shift, f32 cutoff, bool normalize,
            noa::cuda::Stream& stream
    ) {
        // TODO Move symmetry matrices to constant memory?
        const auto count = symmetry.count();
        const auto symmetry_matrices = noa::cuda::memory::AllocatorDevice<Float33>::allocate_async(count, stream);
        noa::cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), count, stream);
        const auto scaling = normalize ? 1 / static_cast<f32>(count + 1) : 1;

        NOA_ASSERT(shape[1] == 1);
        const auto i_shape = shape.as_safe<i32>();
        const auto output_shape = i_shape.filter(0, 2, 3).rfft();
        const auto output_accessor = AccessorRestrict<Value, 3, i32>(output, output_strides.filter(0, 2, 3).as_safe<i32>());

        switch (texture_interp_mode) {
            case InterpMode::NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_2d<REMAP, i32>(
                        interpolator_t(texture), output_accessor, i_shape,
                        matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case InterpMode::LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_2d<REMAP, i32>(
                        interpolator_t(texture), output_accessor, i_shape,
                        matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case InterpMode::COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_2d<REMAP, i32>(
                        interpolator_t(texture), output_accessor, i_shape,
                        matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_2d<REMAP, i32>(
                        interpolator_t(texture), output_accessor, i_shape,
                        matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_2d<REMAP, i32>(
                        interpolator_t(texture), output_accessor, i_shape,
                        matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_3d(output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform_2d", "{} is not supported", texture_interp_mode);
        }
    }

    template<noa::fft::Remap REMAP, bool LAYERED, typename Value>
    void launch_transform_symmetry_rfft_2d_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float22 matrix, const geometry::Symmetry& symmetry, const Vec2<f32>& shift,
            f32 cutoff, bool normalize, noa::cuda::Stream& stream
    ) {
        const bool apply_shift = noa::any(shift != 0.f);
        const bool apply_matrix = matrix != Float22{};

        if (apply_shift && apply_matrix) {
            launch_transform_symmetry_rfft_2d_final_<REMAP, LAYERED>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    matrix, symmetry, shift, cutoff, normalize, stream);
        } else if (apply_shift) {
            launch_transform_symmetry_rfft_2d_final_<REMAP, LAYERED>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    Empty{}, symmetry, shift, cutoff, normalize, stream);
        } else if (apply_matrix) {
            launch_transform_symmetry_rfft_2d_final_<REMAP, LAYERED>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    matrix, symmetry, Empty{}, cutoff, normalize, stream);
        } else {
            launch_transform_symmetry_rfft_2d_final_<REMAP, LAYERED>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    Empty{}, symmetry, Empty{}, cutoff, normalize, stream);
        }
    }

    template<noa::fft::Remap REMAP, typename Value, typename MatrixOrEmpty, typename ShiftOrEmpty>
    void launch_transform_symmetry_rfft_3d_final_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const MatrixOrEmpty& inv_matrix, const geometry::Symmetry& symmetry,
            const ShiftOrEmpty& shift, f32 cutoff, bool normalize,
            noa::cuda::Stream& stream
    ) {
        // TODO Move symmetry matrices to constant memory?
        const auto count = symmetry.count();
        const auto symmetry_matrices = noa::cuda::memory::AllocatorDevice<Float33>::allocate_async(count, stream);
        noa::cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), count, stream);
        const auto scaling = normalize ? 1 / static_cast<f32>(count + 1) : 1;

        const auto i_shape = shape.as_safe<i32>();
        const auto output_shape = i_shape.rfft();
        const auto output_accessor = AccessorRestrict<Value, 4, i32>(output, output_strides.as_safe<i32>());

        switch (texture_interp_mode) {
            case InterpMode::NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::NEAREST, Value>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_3d<REMAP, i32>(
                        interpolator_t(texture), output_accessor, i_shape,
                        inv_matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case InterpMode::LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::LINEAR, Value>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_3d<REMAP, i32>(
                        interpolator_t(texture), output_accessor, i_shape,
                        inv_matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case InterpMode::COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::COSINE, Value>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_3d<REMAP, i32>(
                        interpolator_t(texture), output_accessor, i_shape,
                        inv_matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::LINEAR_FAST, Value>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_3d<REMAP, i32>(
                        interpolator_t(texture), output_accessor, i_shape,
                        inv_matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::COSINE_FAST, Value>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_rfft_3d<REMAP, i32>(
                        interpolator_t(texture), output_accessor, i_shape,
                        inv_matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise_4d(output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform_3d", "{} is not supported", texture_interp_mode);
        }
    }

    template<noa::fft::Remap REMAP, typename Value>
    void launch_transform_symmetry_rfft_3d_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float33& inv_matrix, const geometry::Symmetry& symmetry, const Vec3<f32>& shift,
            f32 cutoff, bool normalize, noa::cuda::Stream& stream
    ) {
        const bool apply_shift = noa::any(shift != 0.f);
        const bool apply_inv_matrix = inv_matrix != Float33{};

        if (apply_shift && apply_inv_matrix) {
            launch_transform_symmetry_rfft_3d_final_<REMAP>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    inv_matrix, symmetry, shift, cutoff, normalize, stream);
        } else if (apply_shift) {
            launch_transform_symmetry_rfft_3d_final_<REMAP>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    Empty{}, symmetry, shift, cutoff, normalize, stream);
        } else if (apply_inv_matrix) {
            launch_transform_symmetry_rfft_3d_final_<REMAP>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    inv_matrix, symmetry, Empty{}, cutoff, normalize, stream);
        } else {
            launch_transform_symmetry_rfft_3d_final_<REMAP>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    Empty{}, symmetry, Empty{}, cutoff, normalize, stream);
        }
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform_2d(
            cudaArray* array, cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Matrix& inv_matrices, const Shift& shifts, f32 cutoff, Stream& stream
    ) {
        NOA_ASSERT(array && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(noa::cuda::memory::AllocatorTexture::array(texture) == array);
        const bool is_layered = noa::cuda::memory::AllocatorArray<Value>::is_layered(array);

        if (is_layered) {
            launch_transform_rfft_2d_<REMAP, true>(
                    texture, texture_interp_mode,
                    output, output_strides, shape,
                    inv_matrices, shifts, cutoff, stream);
        } else {
            launch_transform_rfft_2d_<REMAP, false>(
                    texture, texture_interp_mode,
                    output, output_strides, shape,
                    inv_matrices, shifts, cutoff, stream);
        }
    }

    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform_3d(
            cudaArray* array, cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Matrix& inv_matrices, const Shift& shifts, f32 cutoff, Stream& stream
    ) {
        NOA_ASSERT(array && noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(noa::cuda::memory::AllocatorTexture::array(texture) == array);

        launch_transform_rfft_3d_<REMAP>(
                texture, texture_interp_mode,
                output, output_strides, shape,
                inv_matrices, shifts, cutoff, stream);
    }

    template<Remap REMAP, typename Value, typename>
    void transform_and_symmetrize_2d(
            cudaArray* array, cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float22& inv_matrix, const Symmetry& symmetry, const Vec2<f32>& shift,
            f32 cutoff, bool normalize, Stream& stream
    ) {
        NOA_ASSERT(array && noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(noa::cuda::memory::AllocatorTexture::array(texture) == array);
        const bool is_layered = noa::cuda::memory::AllocatorArray<Value>::is_layered(array);

        if (is_layered) {
            launch_transform_symmetry_rfft_2d_<REMAP, true>(
                    texture, texture_interp_mode,
                    output, output_strides,
                    shape, inv_matrix, symmetry, shift,
                    cutoff, normalize, stream);
        } else {
            launch_transform_symmetry_rfft_2d_<REMAP, false>(
                    texture, texture_interp_mode,
                    output, output_strides,
                    shape, inv_matrix, symmetry, shift,
                    cutoff, normalize, stream);
        }
    }

    template<Remap REMAP, typename Value, typename>
    void transform_and_symmetrize_3d(
            cudaArray* array, cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const Float33& inv_matrix, const Symmetry& symmetry, const Vec3<f32>& shift,
            f32 cutoff, bool normalize, Stream& stream
    ) {
        NOA_ASSERT(array && noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(noa::cuda::memory::AllocatorTexture::array(texture) == array);
        launch_transform_symmetry_rfft_3d_<REMAP>(
                texture, texture_interp_mode,
                output, output_strides,  shape,
                inv_matrix, symmetry, shift, cutoff, normalize, stream);
    }

    #define NOA_INSTANTIATE_TRANSFORM_2D_(T, M, S)              \
    template void transform_2d<Remap::HC2H,  T, M, S, void>(    \
        cudaArray*, cudaTextureObject_t, InterpMode,            \
        T*, const Strides4<i64>&, const Shape4<i64>&,           \
        M const&, S const&, f32, Stream&);                      \
    template void transform_2d<Remap::HC2HC, T, M, S, void>(    \
        cudaArray*, cudaTextureObject_t, InterpMode,            \
        T*, const Strides4<i64>&, const Shape4<i64>&,           \
        M const&, S const&, f32, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_SYMMETRY_2D_(T)                   \
    template void transform_and_symmetrize_2d<Remap::HC2HC, T, void>(   \
        cudaArray*, cudaTextureObject_t, InterpMode,                    \
        T*, const Strides4<i64>&, const Shape4<i64>&,                   \
        const Float22&, const Symmetry&, const Vec2<f32>&,              \
        f32, bool, Stream&);                                            \
    template void transform_and_symmetrize_2d<Remap::HC2H, T, void>(    \
        cudaArray*, cudaTextureObject_t, InterpMode,                    \
        T*, const Strides4<i64>&, const Shape4<i64>&,                   \
        const Float22&, const Symmetry&, const Vec2<f32>&,              \
        f32, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_3D_(T, M, S)              \
    template void transform_3d<Remap::HC2H,  T, M, S, void>(    \
        cudaArray*, cudaTextureObject_t, InterpMode,            \
        T*, const Strides4<i64>&, const Shape4<i64>&,           \
        M const&, S const&, f32, Stream&);                      \
    template void transform_3d<Remap::HC2HC, T, M, S, void>(    \
        cudaArray*, cudaTextureObject_t, InterpMode,            \
        T*, const Strides4<i64>&, const Shape4<i64>&,           \
        M const&, S const&, f32, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_SYMMETRY_3D_(T)                   \
    template void transform_and_symmetrize_3d<Remap::HC2HC, T, void>(   \
        cudaArray*, cudaTextureObject_t, InterpMode,                    \
        T*, const Strides4<i64>&, const Shape4<i64>&,                   \
        const Float33&, const Symmetry&, const Vec3<f32>&,              \
        f32, bool, Stream&);                                            \
    template void transform_and_symmetrize_3d<Remap::HC2H, T, void>(    \
        cudaArray*, cudaTextureObject_t, InterpMode,                    \
        T*, const Strides4<i64>&, const Shape4<i64>&,                   \
        const Float33&, const Symmetry&, const Vec3<f32>&,              \
        f32, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_ALL_(T) \
    NOA_INSTANTIATE_TRANSFORM_2D_(T, const Float22*, const Vec2<f32>*); \
    NOA_INSTANTIATE_TRANSFORM_2D_(T, const Float22*, Vec2<f32>);        \
    NOA_INSTANTIATE_TRANSFORM_2D_(T, Float22, const Vec2<f32>*);        \
    NOA_INSTANTIATE_TRANSFORM_2D_(T, Float22, Vec2<f32>);
//    NOA_INSTANTIATE_TRANSFORM_3D_(T, const Float33*, const Vec3<f32>*); \
//    NOA_INSTANTIATE_TRANSFORM_3D_(T, const Float33*, Vec3<f32>);        \
//    NOA_INSTANTIATE_TRANSFORM_3D_(T, Float33, const Vec3<f32>*);        \
//    NOA_INSTANTIATE_TRANSFORM_3D_(T, Float33, Vec3<f32>);
//    NOA_INSTANTIATE_TRANSFORM_SYMMETRY_2D_(T);                          \
//    NOA_INSTANTIATE_TRANSFORM_SYMMETRY_3D_(T)

    NOA_INSTANTIATE_TRANSFORM_ALL_(f32);
//    NOA_INSTANTIATE_TRANSFORM_ALL_(c32);
}
