#include "noa/common/Assert.h"
#include "noa/common/geometry/details/LinearTransform2DFourier.h"
#include "noa/common/geometry/details/LinearTransform3DFourier.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/geometry/fft/Transform.h"

namespace {
    using namespace ::noa;

    template<bool IS_OPTIONAL, typename Wrapper>
    auto matrixOrShiftOrRawConstPtr(const Wrapper& matrix_or_shift) {
        using output_t = std::conditional_t<traits::is_floatXX_v<Wrapper> || traits::is_floatX_v<Wrapper>,
                traits::remove_ref_cv_t<Wrapper>,
        const traits::element_type_t<Wrapper>*>;
        if constexpr (traits::is_floatXX_v<Wrapper> || traits::is_floatX_v<Wrapper>) {
            return output_t(matrix_or_shift);
        } else {
            if (IS_OPTIONAL && matrix_or_shift.get() == nullptr)
                return output_t{};
            NOA_ASSERT(matrix_or_shift != nullptr);
            using clean_t = traits::remove_ref_cv_t<Wrapper>;
            using raw_const_ptr_t = const typename clean_t::element_type*;
            return static_cast<raw_const_ptr_t>(matrix_or_shift.get());
        }
    }

    template<fft::Remap REMAP, bool LAYERED, typename Value, typename Matrix, typename ShiftOrEmpty>
    void linearTransform2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                            Value* output, dim4_t output_strides, dim4_t shape,
                            Matrix matrix, ShiftOrEmpty shift, float cutoff,
                            cuda::Stream& stream) {
        NOA_ASSERT(shape[1] == 1);
        const auto output_shape = safe_cast<int3_t>(dim3_t{shape[0], shape[2], shape[3]}).fft();
        const auto output_accessor = AccessorRestrict<Value, 3, uint32_t>(
                output, safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]}));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                return noa::cuda::utils::iwise3D("geometry::fft::transform2D", output_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                return noa::cuda::utils::iwise3D("geometry::fft::transform2D", output_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_COSINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                return noa::cuda::utils::iwise3D("geometry::fft::transform2D", output_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                return noa::cuda::utils::iwise3D("geometry::fft::transform2D", output_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                return noa::cuda::utils::iwise3D("geometry::fft::transform2D", output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform2D", "{} is not supported", texture_interp_mode);
        }
    }

    template<fft::Remap REMAP, bool LAYERED, typename Value, typename Matrix, typename Shift>
    void launchLinearTransform2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                  Value* output, dim4_t output_strides, dim4_t shape,
                                  Matrix matrix, Shift shift, float cutoff,
                                  cuda::Stream& stream) {
        const bool do_shift = noa::any(shift != Shift{});
        if (do_shift) {
            linearTransform2D_<REMAP, LAYERED>(
                    texture, texture_interp_mode,
                    output, output_strides, shape,
                    matrix, shift, cutoff, stream);
        } else {
            linearTransform2D_<REMAP, LAYERED>(
                    texture, texture_interp_mode,
                    output, output_strides, shape,
                    matrix, empty_t{}, cutoff, stream);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename ShiftOrEmpty>
    void linearTransform3D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                            Value* output, dim4_t output_strides, dim4_t shape,
                            Matrix inv_matrix, ShiftOrEmpty shift, float cutoff,
                            cuda::Stream& stream) {
        const auto output_shape = safe_cast<int4_t>(shape).fft();
        const auto output_accessor = AccessorRestrict<Value, 4, uint32_t>(output, safe_cast<uint4_t>(output_strides));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<INTERP_NEAREST, Value>;
                const auto kernel = noa::geometry::fft::details::transform3D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, inv_matrix, shift, cutoff);
                return noa::cuda::utils::iwise4D("geometry::fft::transform3D", output_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<INTERP_LINEAR, Value>;
                const auto kernel = noa::geometry::fft::details::transform3D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, inv_matrix, shift, cutoff);
                return noa::cuda::utils::iwise4D("geometry::fft::transform3D", output_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<INTERP_COSINE, Value>;
                const auto kernel = noa::geometry::fft::details::transform3D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, inv_matrix, shift, cutoff);
                return noa::cuda::utils::iwise4D("geometry::fft::transform3D", output_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, Value>;
                const auto kernel = noa::geometry::fft::details::transform3D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, inv_matrix, shift, cutoff);
                return noa::cuda::utils::iwise4D("geometry::fft::transform3D", output_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator3D<INTERP_COSINE_FAST, Value>;
                const auto kernel = noa::geometry::fft::details::transform3D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, inv_matrix, shift, cutoff);
                return noa::cuda::utils::iwise4D("geometry::fft::transform3D", output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform3D", "{} is not supported", texture_interp_mode);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename Shift>
    void launchLinearTransform3D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                  Value* output, dim4_t output_strides, dim4_t shape,
                                  Matrix inv_matrix, Shift shift, float cutoff,
                                  cuda::Stream& stream) {
        const bool do_shift = noa::any(shift != Shift{});
        if (do_shift) {
            linearTransform3D_<REMAP>(
                    texture, texture_interp_mode,
                    output, output_strides, shape,
                    inv_matrix, shift, cutoff, stream);
        } else {
            linearTransform3D_<REMAP>(
                    texture, texture_interp_mode,
                    output, output_strides, shape,
                    inv_matrix, empty_t{}, cutoff, stream);
        }
    }

    template<fft::Remap REMAP, bool LAYERED, typename Value, typename MatrixOrEmpty, typename ShiftOrEmpty>
    void linearTransformSymmetry2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                    Value* output, dim4_t output_strides, dim4_t shape,
                                    MatrixOrEmpty matrix, const geometry::Symmetry& symmetry,
                                    ShiftOrEmpty shift, float cutoff, bool normalize,
                                    cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const dim_t count = symmetry.count();
        const auto symmetry_matrices = cuda::memory::PtrDevice<float33_t>::alloc(count, stream);
        cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(count + 1) : 1;

        NOA_ASSERT(shape[1] == 1);
        const auto output_shape = safe_cast<int3_t>(dim3_t{shape[0], shape[2], shape[3]}).fft();
        const auto output_accessor = AccessorRestrict<Value, 3, uint32_t>(
                output, safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]}));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise3D("geometry::fft::transform2D", output_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise3D("geometry::fft::transform2D", output_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_COSINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise3D("geometry::fft::transform2D", output_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise3D("geometry::fft::transform2D", output_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise3D("geometry::fft::transform2D", output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform2D", "{} is not supported", texture_interp_mode);
        }
    }

    template<fft::Remap REMAP, bool LAYERED, typename Value>
    void launchLinearTransformSymmetry2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                          Value* output, dim4_t output_strides, dim4_t shape,
                                          float22_t matrix, const geometry::Symmetry& symmetry, float2_t shift,
                                          float cutoff, bool normalize, cuda::Stream& stream) {
        const bool apply_shift = any(shift != 0.f);
        const bool apply_matrix = matrix != float22_t{};

        if (apply_shift && apply_matrix) {
            linearTransformSymmetry2D_<REMAP, LAYERED>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    matrix, symmetry, shift, cutoff, normalize, stream);
        } else if (apply_shift) {
            linearTransformSymmetry2D_<REMAP, LAYERED>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    empty_t{}, symmetry, shift, cutoff, normalize, stream);
        } else if (apply_matrix) {
            linearTransformSymmetry2D_<REMAP, LAYERED>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    matrix, symmetry, empty_t{}, cutoff, normalize, stream);
        } else {
            linearTransformSymmetry2D_<REMAP, LAYERED>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    empty_t{}, symmetry, empty_t{}, cutoff, normalize, stream);
        }
    }


    template<fft::Remap REMAP, typename Value, typename MatrixOrEmpty, typename ShiftOrEmpty>
    void linearTransformSymmetry3D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                    Value* output, dim4_t output_strides, dim4_t shape,
                                    MatrixOrEmpty inv_matrix, const geometry::Symmetry& symmetry,
                                    ShiftOrEmpty shift, float cutoff, bool normalize,
                                    cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const dim_t count = symmetry.count();
        const auto symmetry_matrices = cuda::memory::PtrDevice<float33_t>::alloc(count, stream);
        cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(count + 1) : 1;

        const auto output_shape = safe_cast<int4_t>(shape).fft();
        const auto output_accessor = AccessorRestrict<Value, 4, uint32_t>(output, safe_cast<uint4_t>(output_strides));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_NEAREST, Value>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry3D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        inv_matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return cuda::utils::iwise4D("geometry::fft::transform3D", output_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR, Value>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry3D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        inv_matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return cuda::utils::iwise4D("geometry::fft::transform3D", output_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE, Value>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry3D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        inv_matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return cuda::utils::iwise4D("geometry::fft::transform3D", output_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, Value>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry3D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        inv_matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return cuda::utils::iwise4D("geometry::fft::transform3D", output_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE_FAST, Value>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry3D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        inv_matrix, symmetry_matrices.get(), count, scaling, shift, cutoff);
                return cuda::utils::iwise4D("geometry::fft::transform3D", output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform3D", "{} is not supported", texture_interp_mode);
        }
    }

    template<fft::Remap REMAP, typename Value>
    void launchLinearTransformSymmetry3D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                          Value* output, dim4_t output_strides, dim4_t shape,
                                          float33_t inv_matrix, const geometry::Symmetry& symmetry, float3_t shift,
                                          float cutoff, bool normalize, cuda::Stream& stream) {
        const bool apply_shift = any(shift != 0.f);
        const bool apply_inv_matrix = inv_matrix != float33_t{};

        if (apply_shift && apply_inv_matrix) {
            linearTransformSymmetry3D_<REMAP>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    inv_matrix, symmetry, shift, cutoff, normalize, stream);
        } else if (apply_shift) {
            linearTransformSymmetry3D_<REMAP>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    empty_t{}, symmetry, shift, cutoff, normalize, stream);
        } else if (apply_inv_matrix) {
            linearTransformSymmetry3D_<REMAP>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    inv_matrix, symmetry, empty_t{}, cutoff, normalize, stream);
        } else {
            linearTransformSymmetry3D_<REMAP>(
                    texture, texture_interp_mode, output, output_strides, shape,
                    empty_t{}, symmetry, empty_t{}, cutoff, normalize, stream);
        }
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& shifts, float cutoff, Stream& stream) {
        NOA_ASSERT(array && texture && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());
        const bool is_layered = memory::PtrArray<Value>::isLayered(array.get());

        if (is_layered) {
            launchLinearTransform2D_<REMAP, true>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides, shape,
                    matrixOrShiftOrRawConstPtr<false>(inv_matrices),
                    matrixOrShiftOrRawConstPtr<true>(shifts),
                    cutoff, stream);
        } else {
            launchLinearTransform2D_<REMAP, false>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides, shape,
                    matrixOrShiftOrRawConstPtr<false>(inv_matrices),
                    matrixOrShiftOrRawConstPtr<true>(shifts),
                    cutoff, stream);
        }

        stream.attach(array, texture, output);
        if constexpr (!traits::is_floatXX_v<Matrix>)
            stream.attach(inv_matrices);
        if constexpr (!traits::is_floatX_v<Shift>)
            stream.attach(shifts);
    }

    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform3D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& shifts, float cutoff, Stream& stream) {
        NOA_ASSERT(array && texture && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());

        launchLinearTransform3D_<REMAP>(
                *texture, texture_interp_mode,
                output.get(), output_strides,
                shape,
                matrixOrShiftOrRawConstPtr<false>(inv_matrices),
                matrixOrShiftOrRawConstPtr<true>(shifts),
                cutoff, stream);

        stream.attach(array, texture, output);
        if constexpr (!traits::is_floatXX_v<Matrix>)
            stream.attach(inv_matrices);
        if constexpr (!traits::is_floatX_v<Shift>)
            stream.attach(shifts);
    }

    template<Remap REMAP, typename Value, typename>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     float22_t inv_matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());
        const bool is_layered = memory::PtrArray<Value>::isLayered(array.get());

        if (is_layered) {
            launchLinearTransformSymmetry2D_<REMAP, true>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides,
                    shape, inv_matrix, symmetry, shift,
                    cutoff, normalize, stream);
        } else {
            launchLinearTransformSymmetry2D_<REMAP, false>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides,
                    shape, inv_matrix, symmetry, shift,
                    cutoff, normalize, stream);
        }
        stream.attach(array, texture, output, symmetry.share());
    }

    template<Remap REMAP, typename Value, typename>
    void transform3D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     float33_t inv_matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());
        launchLinearTransformSymmetry3D_<REMAP>(
                *texture, texture_interp_mode,
                output.get(), output_strides,  shape,
                inv_matrix, symmetry, shift, cutoff, normalize, stream);
        stream.attach(array, texture, output, symmetry.share());
    }

    #define NOA_INSTANTIATE_TRANSFORM_2D_(T, M, S)                                      \
    template void transform2D<Remap::HC2H,  T, M, S, void>(                             \
        const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode,   \
        const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, Stream&);      \
    template void transform2D<Remap::HC2HC, T, M, S, void>(                             \
        const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode,   \
        const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_SYMMETRY_2D_(T)                                   \
    template void transform2D<Remap::HC2HC, T, void>(                                   \
        const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode,   \
        const shared_t<T[]>&, dim4_t, dim4_t, float22_t,                                \
        const Symmetry&, float2_t, float, bool, Stream&);                               \
    template void transform2D<Remap::HC2H, T, void>(                                    \
        const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode,   \
        const shared_t<T[]>&, dim4_t, dim4_t, float22_t,                                \
        const Symmetry&, float2_t, float, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_3D_(T, M, S)                                      \
    template void transform3D<Remap::HC2H,  T, M, S, void>(                             \
        const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode,   \
        const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, Stream&);      \
    template void transform3D<Remap::HC2HC, T, M, S, void>(                             \
        const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode,   \
        const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_SYMMETRY_3D_(T)                                   \
    template void transform3D<Remap::HC2HC, T, void>(                                   \
        const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode,   \
        const shared_t<T[]>&, dim4_t, dim4_t, float33_t,                                \
        const Symmetry&, float3_t, float, bool, Stream&);                               \
    template void transform3D<Remap::HC2H, T, void>(                                    \
        const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode,   \
        const shared_t<T[]>&, dim4_t, dim4_t, float33_t,                                \
        const Symmetry&, float3_t, float, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_ALL_(T) \
    NOA_INSTANTIATE_TRANSFORM_2D_(T, shared_t<float22_t[]>, shared_t<float2_t[]>);  \
    NOA_INSTANTIATE_TRANSFORM_2D_(T, shared_t<float22_t[]>, float2_t);              \
    NOA_INSTANTIATE_TRANSFORM_2D_(T, float22_t, shared_t<float2_t[]>);              \
    NOA_INSTANTIATE_TRANSFORM_2D_(T, float22_t, float2_t);                          \
    NOA_INSTANTIATE_TRANSFORM_3D_(T, shared_t<float33_t[]>, shared_t<float3_t[]>);  \
    NOA_INSTANTIATE_TRANSFORM_3D_(T, shared_t<float33_t[]>, float3_t);              \
    NOA_INSTANTIATE_TRANSFORM_3D_(T, float33_t, shared_t<float3_t[]>);              \
    NOA_INSTANTIATE_TRANSFORM_3D_(T, float33_t, float3_t);                          \
    NOA_INSTANTIATE_TRANSFORM_SYMMETRY_2D_(T);                                      \
    NOA_INSTANTIATE_TRANSFORM_SYMMETRY_3D_(T)

    NOA_INSTANTIATE_TRANSFORM_ALL_(float);
    NOA_INSTANTIATE_TRANSFORM_ALL_(cfloat_t);
}
