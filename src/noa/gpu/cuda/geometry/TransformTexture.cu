#include "noa/core/Assert.hpp"
#include "noa/core/Math.hpp"
#include "noa/algorithms/geometry/Transform.hpp"

#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/memory/PtrArray.hpp"
#include "noa/gpu/cuda/memory/PtrDevice.hpp"
#include "noa/gpu/cuda/memory/PtrTexture.hpp"

#include "noa/gpu/cuda/geometry/Interpolator.hpp"
#include "noa/gpu/cuda/geometry/Transform.hpp"

namespace {
    using namespace ::noa;

    template<bool LAYERED, typename Value, typename Matrix>
    void launch_transform_texture_2d_(
            cudaTextureObject_t texture, const Shape4<i64>& texture_shape,
            InterpMode texture_interp_mode, BorderMode texture_border_mode,
            const AccessorRestrict<Value, 3, i32>& output, const Shape3<i32>& output_shape,
            Matrix inv_matrices, cuda::Stream& stream) {

        if (texture_border_mode == BorderMode::PERIODIC || texture_border_mode == BorderMode::MIRROR) {
            const auto f_shape = texture_shape.filter(2, 3).vec().as<f32>();
            if (texture_interp_mode == InterpMode::NEAREST) {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::NEAREST, Value, true, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_2d<i32>(
                        interpolator_t(texture, f_shape), output, inv_matrices);
                noa::cuda::utils::iwise_3d("geometry::transform_2d", output_shape, kernel, stream);
            } else if (texture_interp_mode == InterpMode::LINEAR_FAST) {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Value, true, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_2d<i32>(
                        interpolator_t(texture, f_shape), output, inv_matrices);
                noa::cuda::utils::iwise_3d("geometry::transform_2d", output_shape, kernel, stream);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            switch (texture_interp_mode) {
                case InterpMode::NEAREST: {
                    using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::NEAREST, Value, false, LAYERED>;
                    const auto kernel = noa::algorithm::geometry::transform_2d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_3d("geometry::transform_2d", output_shape, kernel, stream);
                }
                case InterpMode::LINEAR: {
                    using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR, Value, false, LAYERED>;
                    const auto kernel = noa::algorithm::geometry::transform_2d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_3d("geometry::transform_2d", output_shape, kernel, stream);
                }
                case InterpMode::COSINE: {
                    using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE, Value, false, LAYERED>;
                    const auto kernel = noa::algorithm::geometry::transform_2d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_3d("geometry::transform_2d", output_shape, kernel, stream);
                }
                case InterpMode::CUBIC: {
                    using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC, Value, false, LAYERED>;
                    const auto kernel = noa::algorithm::geometry::transform_2d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_3d("geometry::transform_2d", output_shape, kernel, stream);
                }
                case InterpMode::CUBIC_BSPLINE: {
                    using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC_BSPLINE, Value, false, LAYERED>;
                    const auto kernel = noa::algorithm::geometry::transform_2d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_3d("geometry::transform_2d", output_shape, kernel, stream);
                }
                case InterpMode::LINEAR_FAST: {
                    using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Value, false, LAYERED>;
                    const auto kernel = noa::algorithm::geometry::transform_2d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_3d("geometry::transform_2d", output_shape, kernel, stream);
                }
                case InterpMode::COSINE_FAST: {
                    using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE_FAST, Value, false, LAYERED>;
                    const auto kernel = noa::algorithm::geometry::transform_2d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_3d("geometry::transform_2d", output_shape, kernel, stream);
                }
                case InterpMode::CUBIC_BSPLINE_FAST: {
                    using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC_BSPLINE_FAST, Value, false, LAYERED>;
                    const auto kernel = noa::algorithm::geometry::transform_2d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_3d("geometry::transform_2d", output_shape, kernel, stream);
                }
            }
        }
    }

    template<typename Value, typename Matrix>
    void launch_transform_texture_3d_(
            cudaTextureObject_t texture, const Shape4<i64>& texture_shape,
            InterpMode texture_interp_mode, BorderMode texture_border_mode,
            const AccessorRestrict<Value, 4, i32>& output, const Shape4<i32>& output_shape,
            Matrix inv_matrices, cuda::Stream& stream) {

        if (texture_border_mode == BorderMode::PERIODIC || texture_border_mode == BorderMode::MIRROR) {
            const auto f_shape = texture_shape.pop_front().vec().as<f32>();
            if (texture_interp_mode == InterpMode::NEAREST) {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::NEAREST, Value, true>;
                const auto kernel = noa::algorithm::geometry::transform_3d<i32>(
                        interpolator_t(texture, f_shape), output, inv_matrices);
                noa::cuda::utils::iwise_4d("geometry::transform_3d", output_shape, kernel, stream);
            } else if (texture_interp_mode == InterpMode::LINEAR_FAST) {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::LINEAR_FAST, Value, true>;
                const auto kernel = noa::algorithm::geometry::transform_3d<i32>(
                        interpolator_t(texture, f_shape), output, inv_matrices);
                noa::cuda::utils::iwise_4d("geometry::transform_3d", output_shape, kernel, stream);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            switch (texture_interp_mode) {
                case InterpMode::NEAREST: {
                    using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::NEAREST, Value, false>;
                    const auto kernel = noa::algorithm::geometry::transform_3d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_4d("geometry::transform_3d", output_shape, kernel, stream);
                }
                case InterpMode::LINEAR: {
                    using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::LINEAR, Value, false>;
                    const auto kernel = noa::algorithm::geometry::transform_3d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_4d("geometry::transform_3d", output_shape, kernel, stream);
                }
                case InterpMode::COSINE: {
                    using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::COSINE, Value, false>;
                    const auto kernel = noa::algorithm::geometry::transform_3d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_4d("geometry::transform_3d", output_shape, kernel, stream);
                }
                case InterpMode::CUBIC: {
                    using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::CUBIC, Value, false>;
                    const auto kernel = noa::algorithm::geometry::transform_3d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_4d("geometry::transform_3d", output_shape, kernel, stream);
                }
                case InterpMode::CUBIC_BSPLINE: {
                    using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::CUBIC_BSPLINE, Value, false>;
                    const auto kernel = noa::algorithm::geometry::transform_3d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_4d("geometry::transform_3d", output_shape, kernel, stream);
                }
                case InterpMode::LINEAR_FAST: {
                    using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::LINEAR_FAST, Value, false>;
                    const auto kernel = noa::algorithm::geometry::transform_3d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_4d("geometry::transform_3d", output_shape, kernel, stream);
                }
                case InterpMode::COSINE_FAST: {
                    using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::COSINE_FAST, Value, false>;
                    const auto kernel = noa::algorithm::geometry::transform_3d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_4d("geometry::transform_3d", output_shape, kernel, stream);
                }
                case InterpMode::CUBIC_BSPLINE_FAST: {
                    using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::CUBIC_BSPLINE_FAST, Value, false>;
                    const auto kernel = noa::algorithm::geometry::transform_3d<i32>(
                            interpolator_t(texture), output, inv_matrices);
                    return noa::cuda::utils::iwise_4d("geometry::transform_3d", output_shape, kernel, stream);
                }
            }
        }
    }

    template<bool LAYERED, typename Value>
    void launch_transform_symmetry_texture_2d_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Vec2<f32>& shift, const Float22& inv_matrix, const geometry::Symmetry& symmetry,
            const Vec2<f32>& center, bool normalize, cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const auto symmetry_count = symmetry.count();
        const auto symmetry_matrices = noa::cuda::memory::PtrDevice<Float33>::alloc(symmetry_count, stream);
        noa::cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);
        const auto scaling = normalize ? 1 / static_cast<f32>(symmetry_count + 1) : 1;

        const auto iwise_shape = output_shape.filter(0, 2, 3).as_safe<i32>();
        const auto output_accessor = AccessorRestrict<Value, 3, i32>(
                output, output_strides.filter(0, 2, 3).as_safe<i32>());

        switch (texture_interp_mode) {
            case InterpMode::NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::transform_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::transform_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::transform_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::transform_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC_BSPLINE, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::transform_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::transform_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::transform_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC_BSPLINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::transform_2d", iwise_shape, kernel, stream);
            }
        }
    }

    template<typename Value>
    void launch_transform_symmetry_texture_3d_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Vec3<f32>& shift, const Float33& matrix, const geometry::Symmetry& symmetry,
            const Vec3<f32>& center, bool normalize, cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const auto symmetry_count = symmetry.count();
        const auto symmetry_matrices = cuda::memory::PtrDevice<Float33>::alloc(symmetry_count, stream);
        cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);
        const f32 scaling = normalize ? 1 / static_cast<f32>(symmetry_count + 1) : 1;

        const auto iwise_shape = output_shape.as_safe<i32>();
        const auto output_accessor = AccessorRestrict<Value, 4, i32>(output, output_strides.as_safe<i32>());

        switch (texture_interp_mode) {
            case InterpMode::NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator3D<InterpMode::NEAREST, Value>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise_4d("geometry::transform_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator3D<InterpMode::LINEAR, Value>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise_4d("geometry::transform_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE: {
                using interpolator_t = cuda::geometry::Interpolator3D<InterpMode::COSINE, Value>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise_4d("geometry::transform_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC: {
                using interpolator_t = cuda::geometry::Interpolator3D<InterpMode::CUBIC, Value>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise_4d("geometry::transform_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE: {
                using interpolator_t = cuda::geometry::Interpolator3D<InterpMode::CUBIC_BSPLINE, Value>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise_4d("geometry::transform_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<InterpMode::LINEAR_FAST, Value>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise_4d("geometry::transform_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<InterpMode::COSINE_FAST, Value>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise_4d("geometry::transform_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<InterpMode::CUBIC_BSPLINE_FAST, Value>;
                const auto kernel = noa::algorithm::geometry::transform_symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise_4d("geometry::transform_3d", iwise_shape, kernel, stream);
            }
        }
    }

    template<bool LAYERED, typename Value>
    void launch_symmetrize_2d_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const geometry::Symmetry& symmetry, const Vec2<f32>& center, bool normalize,
            cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const auto symmetry_count = symmetry.count();
        const auto symmetry_matrices = noa::cuda::memory::PtrDevice<Float33>::alloc(symmetry_count, stream);
        noa::cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);
        const auto scaling = normalize ? 1 / static_cast<f32>(symmetry_count + 1) : 1;

        const auto iwise_shape = output_shape.filter(0, 2, 3).as_safe<i32>();
        const auto output_accessor = AccessorRestrict<Value, 3, i32>(
                output, output_strides.filter(0, 2, 3).as_safe<i32>());

        switch (texture_interp_mode) {
            case InterpMode::NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::symmetry_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::symmetry_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::symmetry_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::symmetry_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC_BSPLINE, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::symmetry_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::symmetry_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::symmetry_2d", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::CUBIC_BSPLINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::algorithm::geometry::symmetry_2d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_3d("geometry::symmetry_2d", iwise_shape, kernel, stream);
            }
        }
    }

    template<typename Value>
    void launch_symmetrize_3d_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const geometry::Symmetry& symmetry, const Vec3<f32>& center, bool normalize,
            cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const auto symmetry_count = symmetry.count();
        const auto symmetry_matrices = noa::cuda::memory::PtrDevice<Float33>::alloc(symmetry_count, stream);
        noa::cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);
        const auto scaling = normalize ? 1 / static_cast<f32>(symmetry_count + 1) : 1;

        const auto iwise_shape = output_shape.as_safe<i32>();
        const auto output_accessor = AccessorRestrict<Value, 4, i32>(output, output_strides.as_safe<i32>());

        switch (texture_interp_mode) {
            case InterpMode::NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::NEAREST, Value>;
                const auto kernel = noa::algorithm::geometry::symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d("geometry::symmetry_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::LINEAR, Value>;
                const auto kernel = noa::algorithm::geometry::symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d("geometry::symmetry_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::COSINE, Value>;
                const auto kernel = noa::algorithm::geometry::symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d("geometry::symmetry_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::CUBIC, Value>;
                const auto kernel = noa::algorithm::geometry::symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d("geometry::symmetry_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::CUBIC_BSPLINE, Value>;
                const auto kernel = noa::algorithm::geometry::symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d("geometry::symmetry_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::LINEAR_FAST, Value>;
                const auto kernel = noa::algorithm::geometry::symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d("geometry::symmetry_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::COSINE_FAST, Value>;
                const auto kernel = noa::algorithm::geometry::symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d("geometry::symmetry_3d", iwise_shape, kernel, stream);
            }
            case InterpMode::CUBIC_BSPLINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::CUBIC_BSPLINE_FAST, Value>;
                const auto kernel = noa::algorithm::geometry::symmetry_3d<i32>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return noa::cuda::utils::iwise_4d("geometry::symmetry_3d", iwise_shape, kernel, stream);
            }
        }
    }

    template<typename Matrix>
    auto truncated_matrix_or_const_ptr_(const Matrix& matrix) {
        if constexpr (noa::traits::is_mat23_v<Matrix> || noa::traits::is_mat33_v<Matrix>) {
            return Float23(matrix);
        } else if constexpr (noa::traits::is_mat34_v<Matrix> || noa::traits::is_mat44_v<Matrix>) {
            return Float34(matrix);
        } else {
            NOA_ASSERT(matrix != nullptr);
            return matrix;
        }
    }
}

namespace noa::cuda::geometry {
    template<typename Value, typename Matrix, typename>
    void transform_2d(cudaArray* array, cudaTextureObject_t texture, const Shape4<i64>& texture_shape,
                      InterpMode texture_interp_mode, BorderMode texture_border_mode,
                      Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                      const Matrix& inv_matrices, Stream& stream) {
        NOA_ASSERT(array && texture && noa::all(texture_shape > 0) && noa::all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        const bool is_layered = noa::cuda::memory::PtrArray<Value>::is_layered(array);
        NOA_ASSERT(noa::cuda::memory::PtrTexture::array(texture) == array);
        NOA_ASSERT(texture_shape[1] == 1 && output_shape[1] == 1);
        if constexpr (!noa::traits::is_matXX_v<Matrix>) {
            NOA_ASSERT_DEVICE_PTR(inv_matrices, stream.device());
        }

        const auto output_shape_2d = output_shape.filter(0, 2, 3).as_safe<i32>();
        const auto output_accessor = AccessorRestrict<Value, 3, i32>(output, output_strides.filter(0, 2, 3).as_safe<i32>());

        if (is_layered) {
            NOA_ASSERT(texture_shape[0] == output_shape[0]);
            launch_transform_texture_2d_<true>(
                    texture, texture_shape, texture_interp_mode, texture_border_mode,
                    output_accessor, output_shape_2d,
                    truncated_matrix_or_const_ptr_(inv_matrices), stream);
        } else {
            NOA_ASSERT(texture_shape[0] == 1);
            launch_transform_texture_2d_<false>(
                    texture, texture_shape, texture_interp_mode, texture_border_mode,
                    output_accessor, output_shape_2d,
                    truncated_matrix_or_const_ptr_(inv_matrices), stream);
        }
    }

    template<typename Value, typename Matrix, typename>
    void transform_3d(cudaArray* array, cudaTextureObject_t texture, const Shape4<i64>& texture_shape,
                      InterpMode texture_interp_mode, BorderMode texture_border_mode,
                      Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                      const Matrix& inv_matrices, Stream& stream) {
        NOA_ASSERT(array && texture && noa::all(texture_shape > 0) && noa::all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(noa::cuda::memory::PtrTexture::array(texture) == array);
        NOA_ASSERT(texture_shape[0] == 1);

        const auto output_accessor = AccessorRestrict<Value, 4, i32>(output, output_strides.as_safe<i32>());
        launch_transform_texture_3d_(
                texture, texture_shape, texture_interp_mode, texture_border_mode,
                output_accessor, output_shape.as_safe<i32>(),
                truncated_matrix_or_const_ptr_(inv_matrices), stream);
    }

    template<typename Value, typename>
    void transform_and_symmetrize_2d(
            cudaArray* array, cudaTextureObject_t texture,
            InterpMode texture_interp_mode, const Shape4<i64>& texture_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Vec2<f32>& shift, const Float22& inv_matrix,
            const Symmetry& symmetry, const Vec2<f32>& center,
            bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && noa::all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        const bool is_layered = noa::cuda::memory::PtrArray<Value>::is_layered(array);
        NOA_ASSERT(noa::cuda::memory::PtrTexture::array(texture) == array);
        NOA_ASSERT(texture_shape[1] == 1);
        NOA_ASSERT((is_layered && texture_shape[0] == output_shape[0]) ||
                   (!is_layered && texture_shape[0] == 1));
        (void) texture_shape;

        if (is_layered) {
            launch_transform_symmetry_texture_2d_<true>(
                    texture, texture_interp_mode,
                    output, output_strides, output_shape,
                    shift, inv_matrix, symmetry, center, normalize, stream);
        } else {
            launch_transform_symmetry_texture_2d_<false>(
                    texture, texture_interp_mode,
                    output, output_strides, output_shape,
                    shift, inv_matrix, symmetry, center, normalize, stream);
        }
    }

    template<typename Value, typename>
    void transform_and_symmetrize_3d(
            cudaArray* array, cudaTextureObject_t texture,
            InterpMode texture_interp_mode, const Shape4<i64>& texture_shape,
            Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            const Vec3<f32>& shift, const Float33& inv_matrix,
            const Symmetry& symmetry, const Vec3<f32>& center,
            bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(texture_shape[0] == 1);
        (void) texture_shape;

        launch_transform_symmetry_texture_3d_(
                texture, texture_interp_mode,
                output, output_strides, output_shape,
                shift, inv_matrix, symmetry, center, normalize, stream);
    }

    template<typename Value, typename>
    void symmetrize_2d(cudaArray* array,
                       cudaTextureObject_t texture,
                       InterpMode texture_interp_mode, const Shape4<i64>& texture_shape,
                       Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                       const Symmetry& symmetry, const Vec2<f32>& center, bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && noa::all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        const bool is_layered = noa::cuda::memory::PtrArray<Value>::is_layered(array);
        NOA_ASSERT(noa::cuda::memory::PtrTexture::array(texture) == array);
        NOA_ASSERT(texture_shape[1] == 1);
        NOA_ASSERT((is_layered && texture_shape[0] == output_shape[0]) ||
                   (!is_layered && texture_shape[0] == 1));
        (void) texture_shape;

        if (is_layered) {
            launch_symmetrize_2d_<true>(
                    texture, texture_interp_mode,
                    output, output_strides, output_shape,
                    symmetry, center, normalize, stream);
        } else {
            launch_symmetrize_2d_<false>(
                    texture, texture_interp_mode,
                    output, output_strides, output_shape,
                    symmetry, center, normalize, stream);
        }
    }

    template<typename Value, typename>
    void symmetrize_3d(cudaArray* array,
                       cudaTextureObject_t texture,
                       InterpMode texture_interp_mode, const Shape4<i64>& texture_shape,
                       Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                       const Symmetry& symmetry, const Vec3<f32>& center, bool normalize, Stream& stream) {
        NOA_ASSERT(noa::all(output_shape > 0) && array && texture);
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(texture_shape[0] == 1);
        (void) texture_shape;

        launch_symmetrize_3d_(
                texture, texture_interp_mode,
                output, output_strides, output_shape,
                symmetry, center, normalize, stream);
    }

    #define NOA_INSTANTIATE_TRANSFORM_2D_MATRIX(T, M)   \
    template void transform_2d<T, M, void>(             \
        cudaArray*, cudaTextureObject_t,                \
        const Shape4<i64>&, InterpMode, BorderMode,     \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        M const&, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, M)   \
    template void transform_3d<T, M, void>(             \
        cudaArray*, cudaTextureObject_t,                \
        const Shape4<i64>&, InterpMode, BorderMode,     \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        M const&, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_SYM_(T)               \
    template void transform_and_symmetrize_2d<T, void>(     \
        cudaArray*, cudaTextureObject_t,                    \
        InterpMode, const Shape4<i64>&,                     \
        T*, const Strides4<i64>&, const Shape4<i64>&,       \
        const Vec2<f32>&, const Float22&,                   \
        const Symmetry&, const Vec2<f32>&, bool, Stream&);  \
    template void symmetrize_2d<T, void>(                   \
        cudaArray*, cudaTextureObject_t,                    \
        InterpMode, const Shape4<i64>&,                     \
        T*, const Strides4<i64>&, const Shape4<i64>&,       \
        const Symmetry&, const Vec2<f32>&, bool, Stream&);  \
    template void transform_and_symmetrize_3d<T, void>(     \
        cudaArray*, cudaTextureObject_t,                    \
        InterpMode, const Shape4<i64>&,                     \
        T*, const Strides4<i64>&, const Shape4<i64>&,       \
        const Vec3<f32>&, const Float33&,                   \
        const Symmetry&, const Vec3<f32>&, bool, Stream&);  \
    template void symmetrize_3d<T, void>(                   \
        cudaArray*, cudaTextureObject_t,                    \
        InterpMode, const Shape4<i64>&,                     \
        T*, const Strides4<i64>&, const Shape4<i64>&,       \
        const Symmetry&, const Vec3<f32>&, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_(T)                   \
    NOA_INSTANTIATE_TRANSFORM_2D_MATRIX(T, Float23);        \
    NOA_INSTANTIATE_TRANSFORM_2D_MATRIX(T, Float33);        \
    NOA_INSTANTIATE_TRANSFORM_2D_MATRIX(T, const Float23*); \
    NOA_INSTANTIATE_TRANSFORM_2D_MATRIX(T, const Float33*); \
    NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, Float34);        \
    NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, Float44);        \
    NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, const Float34*); \
    NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, const Float44*); \
    NOA_INSTANTIATE_TRANSFORM_SYM_(T)

    NOA_INSTANTIATE_TRANSFORM_(f32);
    NOA_INSTANTIATE_TRANSFORM_(c32);
}
