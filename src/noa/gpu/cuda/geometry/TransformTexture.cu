#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/geometry/details/LinearTransform2D.h"
#include "noa/common/geometry/details/LinearTransform3D.h"

#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"

#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/geometry/Transform.h"

namespace {
    using namespace ::noa;

    template<bool LAYERED, typename Value, typename Matrix>
    void launchTransformTexture2D_(cudaTextureObject_t texture, dim4_t texture_shape,
                                   InterpMode texture_interp_mode, BorderMode texture_border_mode,
                                   const AccessorRestrict<Value, 3, uint32_t>& output, uint3_t output_shape,
                                   Matrix inv_matrices, cuda::Stream& stream) {

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            const auto f_shape = float2_t(texture_shape.get(2));
            if (texture_interp_mode == INTERP_NEAREST) {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, Value, true, LAYERED>;
                const auto kernel = noa::geometry::details::transform2D<uint32_t>(
                        interpolator_t(texture, f_shape), output, inv_matrices);
                cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, true, LAYERED>;
                const auto kernel = noa::geometry::details::transform2D<uint32_t>(
                        interpolator_t(texture, f_shape), output, inv_matrices);
                cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            switch (texture_interp_mode) {
                case INTERP_NEAREST: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, Value, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
                }
                case INTERP_LINEAR: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
                }
                case INTERP_COSINE: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, Value, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
                }
                case INTERP_CUBIC: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC, Value, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
                }
                case INTERP_CUBIC_BSPLINE: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE, Value, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
                }
                case INTERP_LINEAR_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
                }
                case INTERP_COSINE_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, Value, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
                }
                case INTERP_CUBIC_BSPLINE_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE_FAST, Value, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", output_shape, kernel, stream);
                }
            }
        }
    }

    template<typename Value, typename Matrix>
    void launchTransformTexture3D_(cudaTextureObject_t texture, dim4_t texture_shape,
                                   InterpMode texture_interp_mode, BorderMode texture_border_mode,
                                   const AccessorRestrict<Value, 4, uint32_t>& output, uint4_t output_shape,
                                   Matrix inv_matrices, cuda::Stream& stream) {

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            const auto f_shape = float3_t(texture_shape.get(1));
            if (texture_interp_mode == INTERP_NEAREST) {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_NEAREST, Value, true>;
                const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                        interpolator_t(texture, f_shape), output, inv_matrices);
                cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, Value, true>;
                const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                        interpolator_t(texture, f_shape), output, inv_matrices);
                cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            switch (texture_interp_mode) {
                case INTERP_NEAREST: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_NEAREST, Value, false>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
                }
                case INTERP_LINEAR: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR, Value, false>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
                }
                case INTERP_COSINE: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE, Value, false>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
                }
                case INTERP_CUBIC: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC, Value, false>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
                }
                case INTERP_CUBIC_BSPLINE: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE, Value, false>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
                }
                case INTERP_LINEAR_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, Value, false>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
                }
                case INTERP_COSINE_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE_FAST, Value, false>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
                }
                case INTERP_CUBIC_BSPLINE_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE_FAST, Value, false>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output, inv_matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", output_shape, kernel, stream);
                }
            }
        }
    }

    template<bool LAYERED, typename Value>
    void launchTransformSymmetryTexture2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                           Value* output, dim4_t output_strides, dim4_t output_shape,
                                           float2_t shift, float22_t inv_matrix, const geometry::Symmetry& symmetry,
                                           float2_t center, bool normalize, cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const dim_t symmetry_count = symmetry.count();
        const auto symmetry_matrices = cuda::memory::PtrDevice<float33_t>::alloc(symmetry_count, stream);
        cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(symmetry_count + 1) : 1;

        const auto iwise_shape = safe_cast<uint3_t>(dim3_t{output_shape[0], output_shape[2], output_shape[3]});
        const auto output_accessor = AccessorRestrict<Value, 3, uint32_t>(
                output, safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]}));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, inv_matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
        }
    }

    template<typename Value>
    void launchTransformSymmetryTexture3D_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            Value* output, dim4_t output_strides, dim4_t output_shape,
            float3_t shift, float33_t matrix, const geometry::Symmetry& symmetry,
            float3_t center, bool normalize, cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const dim_t symmetry_count = symmetry.count();
        const auto symmetry_matrices = cuda::memory::PtrDevice<float33_t>::alloc(symmetry_count, stream);
        cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(symmetry_count + 1) : 1;

        const auto iwise_shape = safe_cast<uint4_t>(output_shape);
        const auto output_accessor = AccessorRestrict<Value, 4, uint32_t>(output, safe_cast<uint4_t>(output_strides));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_NEAREST, Value>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR, Value>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE, Value>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC, Value>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE, Value>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, Value>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE_FAST, Value>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE_FAST, Value>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
            }
        }
    }

    template<bool LAYERED, typename Value>
    void launchSymmetrize2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                             Value* output, dim4_t output_strides, dim4_t output_shape,
                             const geometry::Symmetry& symmetry, float2_t center, bool normalize,
                             cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const dim_t symmetry_count = symmetry.count();
        const auto symmetry_matrices = cuda::memory::PtrDevice<float33_t>::alloc(symmetry_count, stream);
        cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(symmetry_count + 1) : 1;

        const auto iwise_shape = safe_cast<uint3_t>(dim3_t{output_shape[0], output_shape[2], output_shape[3]});
        const auto output_accessor = AccessorRestrict<Value, 3, uint32_t>(
                output, safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]}));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
        }
    }

    template<typename Value>
    void launchSymmetrize3D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                             Value* output, dim4_t output_strides, dim4_t output_shape,
                             const geometry::Symmetry& symmetry, float3_t center, bool normalize,
                             cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const dim_t symmetry_count = symmetry.count();
        const auto symmetry_matrices = cuda::memory::PtrDevice<float33_t>::alloc(symmetry_count, stream);
        cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(symmetry_count + 1) : 1;

        const auto iwise_shape = safe_cast<uint4_t>(output_shape);
        const auto output_accessor = AccessorRestrict<Value, 4, uint32_t>(output, safe_cast<uint4_t>(output_strides));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_NEAREST, Value>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR, Value>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE, Value>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC, Value>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE, Value>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, Value>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE_FAST, Value>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE_FAST, Value>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        symmetry_matrices.get(), symmetry_count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
        }
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
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, dim4_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const Matrix& inv_matrices, Stream& stream) {
        NOA_ASSERT(array && texture && all(texture_shape > 0) && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        const bool is_layered = memory::PtrArray<Value>::isLayered(array.get());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());
        NOA_ASSERT(texture_shape[1] == 1 && output_shape[1] == 1);
        if constexpr (!traits::is_floatXX_v<Matrix>) {
            NOA_ASSERT_DEVICE_PTR(inv_matrices.get(), stream.device());
        }

        const auto output_shape_ = safe_cast<uint3_t>(dim3_t{output_shape[0], output_shape[2], output_shape[3]});
        const auto output_strides_ = safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});
        const auto output_accessor = AccessorRestrict<Value, 3, uint32_t>(output.get(), output_strides_);

        if (is_layered) {
            NOA_ASSERT(texture_shape[0] == output_shape[0]);
            launchTransformTexture2D_<true>(
                    *texture, texture_shape, texture_interp_mode, texture_border_mode,
                    output_accessor, output_shape_,
                    matrixOrRawConstPtr(inv_matrices), stream);
        } else {
            NOA_ASSERT(texture_shape[0] == 1);
            launchTransformTexture2D_<false>(
                    *texture, texture_shape, texture_interp_mode, texture_border_mode,
                    output_accessor, output_shape_,
                    matrixOrRawConstPtr(inv_matrices), stream);
        }

        if constexpr (traits::is_floatXX_v<Matrix>)
            stream.attach(array, texture, output);
        else
            stream.attach(array, texture, output, inv_matrices);
    }

    template<typename Value, typename Matrix, typename>
    void transform3D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, dim4_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const Matrix& inv_matrices, Stream& stream) {
        NOA_ASSERT(array && texture && all(texture_shape > 0) && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());
        NOA_ASSERT(texture_shape[0] == 1);

        const auto output_shape_ = safe_cast<uint4_t>(output_shape);
        const auto output_strides_ = safe_cast<uint4_t>(output_strides);
        const auto output_accessor = AccessorRestrict<Value, 4, uint32_t>(output.get(), output_strides_);

        launchTransformTexture3D_(
                *texture, texture_shape, texture_interp_mode, texture_border_mode,
                output_accessor, output_shape_, matrixOrRawConstPtr(inv_matrices), stream);

        if constexpr (traits::is_floatXX_v<Matrix>)
            stream.attach(array, texture, output);
        else
            stream.attach(array, texture, output, inv_matrices);
    }

    template<typename Value, typename>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture,
                     InterpMode texture_interp_mode, dim4_t texture_shape,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float2_t shift, float22_t inv_matrix, const Symmetry& symmetry, float2_t center,
                     bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        const bool is_layered = memory::PtrArray<Value>::isLayered(array.get());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());
        NOA_ASSERT(texture_shape[1] == 1);
        NOA_ASSERT((is_layered && texture_shape[0] == output_shape[0]) ||
                   (!is_layered && texture_shape[0] == 1));
        (void) texture_shape;

        if (is_layered) {
            launchTransformSymmetryTexture2D_<true>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides, output_shape,
                    shift, inv_matrix, symmetry, center, normalize, stream);
        } else {
            launchTransformSymmetryTexture2D_<false>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides, output_shape,
                    shift, inv_matrix, symmetry, center, normalize, stream);
        }
        stream.attach(array, texture, output, symmetry.share());
    }

    template<typename Value, typename>
    void transform3D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture,
                     InterpMode texture_interp_mode, dim4_t texture_shape,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float3_t shift, float33_t inv_matrix, const Symmetry& symmetry, float3_t center,
                     bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(texture_shape[0] == 1);
        (void) texture_shape;

        launchTransformSymmetryTexture3D_(
                *texture, texture_interp_mode,
                output.get(), output_strides, output_shape,
                shift, inv_matrix, symmetry, center, normalize, stream);
        stream.attach(array, texture, output, symmetry.share());
    }

    template<typename Value, typename>
    void symmetrize2D(const shared_t<cudaArray>& array,
                      const shared_t<cudaTextureObject_t>& texture,
                      InterpMode texture_interp_mode, dim4_t texture_shape,
                      const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                      const Symmetry& symmetry, float2_t center, bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        const bool is_layered = memory::PtrArray<Value>::isLayered(array.get());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());
        NOA_ASSERT(texture_shape[1] == 1);
        NOA_ASSERT((is_layered && texture_shape[0] == output_shape[0]) ||
                   (!is_layered && texture_shape[0] == 1));
        (void) texture_shape;

        if (is_layered) {
            launchSymmetrize2D_<true>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides, output_shape,
                    symmetry, center, normalize, stream);
        } else {
            launchSymmetrize2D_<false>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides, output_shape,
                    symmetry, center, normalize, stream);
        }
        stream.attach(array, texture, output, symmetry.share());
    }

    template<typename Value, typename>
    void symmetrize3D(const shared_t<cudaArray>& array,
                      const shared_t<cudaTextureObject_t>& texture,
                      InterpMode texture_interp_mode, dim4_t texture_shape,
                      const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                      const Symmetry& symmetry, float3_t center, bool normalize, Stream& stream) {
        NOA_ASSERT(all(output_shape > 0) && array && texture);
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(texture_shape[0] == 1);
        (void) texture_shape;

        launchSymmetrize3D_(
                *texture, texture_interp_mode,
                output.get(), output_strides, output_shape,
                symmetry, center, normalize, stream);
        stream.attach(array, texture, output, symmetry.share());
    }

    #define NOA_INSTANTIATE_TRANSFORM_2D_MATRIX(T, M)                        \
    template void transform2D<T, shared_t<M[]>, void>(                       \
        const shared_t<cudaArray>&,                                          \
        const shared_t<cudaTextureObject_t>&, dim4_t, InterpMode, BorderMode,\
        const shared_t<T[]>&, dim4_t, dim4_t,                                \
        const shared_t<M[]>&, Stream&);                                      \
    template void transform2D<T, M, void>(                                   \
        const shared_t<cudaArray>&,                                          \
        const shared_t<cudaTextureObject_t>&, dim4_t, InterpMode, BorderMode,\
        const shared_t<T[]>&, dim4_t, dim4_t,                                \
        const M&, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, M)                        \
    template void transform3D<T, shared_t<M[]>, void>(                       \
        const shared_t<cudaArray>&,                                          \
        const shared_t<cudaTextureObject_t>&, dim4_t, InterpMode, BorderMode,\
        const shared_t<T[]>&, dim4_t, dim4_t,                                \
        const shared_t<M[]>&, Stream&);                                      \
    template void transform3D<T, M, void>(                                   \
        const shared_t<cudaArray>&,                                          \
        const shared_t<cudaTextureObject_t>&, dim4_t, InterpMode, BorderMode,\
        const shared_t<T[]>&, dim4_t, dim4_t,                                \
        const M&, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_SYM_(T)                           \
    template void transform2D<T, void>(                                 \
        const shared_t<cudaArray>&,                                     \
        const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t,       \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        float2_t, float22_t, const Symmetry&, float2_t, bool, Stream&); \
    template void symmetrize2D<T, void>(                                \
        const shared_t<cudaArray>&,                                     \
        const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t,       \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        const Symmetry&, float2_t, bool, Stream&);                      \
    template void transform3D<T, void>(                                 \
        const shared_t<cudaArray>&,                                     \
        const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t,       \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        float3_t, float33_t, const Symmetry&, float3_t, bool, Stream&); \
    template void symmetrize3D<T, void>(                                \
        const shared_t<cudaArray>&,                                     \
        const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t,       \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        const Symmetry&, float3_t, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_(T)               \
    NOA_INSTANTIATE_TRANSFORM_2D_MATRIX(T, float23_t);  \
    NOA_INSTANTIATE_TRANSFORM_2D_MATRIX(T, float33_t);  \
    NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, float34_t);  \
    NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, float44_t);  \
    NOA_INSTANTIATE_TRANSFORM_SYM_(T)

    NOA_INSTANTIATE_TRANSFORM_(float);
    NOA_INSTANTIATE_TRANSFORM_(cfloat_t);
}
