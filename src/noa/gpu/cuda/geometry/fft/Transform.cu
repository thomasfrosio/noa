#include "noa/common/Assert.h"
#include "noa/common/geometry/Interpolator.h"
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


    template<fft::Remap REMAP, typename Value, typename Matrix, typename ShiftOrEmpty>
    void linearTransform_(const AccessorRestrict<const Value, 3, uint32_t>& input,
                          const AccessorRestrict<Value, 3, uint32_t>& output, dim4_t shape,
                          Matrix matrices, ShiftOrEmpty shifts, float cutoff,
                          InterpMode interp_mode, cuda::Stream& stream) {
        NOA_ASSERT(shape[1] == 1);
        const auto input_shape_2d = int2_t(shape.get(2)).fft();
        const auto output_shape = safe_cast<int3_t>(dim3_t{shape[0], shape[2], shape[3]}).fft();

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(input, input_shape_2d);
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int32_t>(
                        interpolator, output, shape, matrices, shifts, cutoff);
                return noa::cuda::utils::iwise3D("transform2D", output_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(input, input_shape_2d);
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int32_t>(
                        interpolator, output, shape, matrices, shifts, cutoff);
                return noa::cuda::utils::iwise3D("transform2D", output_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST:
            case INTERP_COSINE: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(input, input_shape_2d);
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int32_t>(
                        interpolator, output, shape, matrices, shifts, cutoff);
                return noa::cuda::utils::iwise3D("transform2D", output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform2D", "{} is not supported", interp_mode);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Matrix, typename ShiftOrEmpty>
    void linearTransform_(const AccessorRestrict<const Value, 4, uint32_t>& input,
                          const AccessorRestrict<Value, 4, uint32_t>& output, dim4_t shape,
                          Matrix matrices, ShiftOrEmpty shift, float cutoff,
                          InterpMode interp_mode, cuda::Stream& stream) {
        const auto input_shape_3d = int3_t(shape.get(1)).fft();
        const auto output_shape = safe_cast<int4_t>(shape).fft();

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_NEAREST>(input, input_shape_3d);
                const auto kernel = noa::geometry::fft::details::transform3D<REMAP, int32_t>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return noa::cuda::utils::iwise4D("transform3D", output_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_LINEAR>(input, input_shape_3d);
                const auto kernel = noa::geometry::fft::details::transform3D<REMAP, int32_t>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return noa::cuda::utils::iwise4D("transform3D", output_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST:
            case INTERP_COSINE: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_COSINE>(input, input_shape_3d);
                const auto kernel = noa::geometry::fft::details::transform3D<REMAP, int32_t>(
                        interpolator, output, shape, matrices, shift, cutoff);
                return noa::cuda::utils::iwise4D("transform3D", output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform3D", "{} is not supported", interp_mode);
        }
    }

    template<fft::Remap REMAP, int32_t NDIM, typename Value, typename Matrix, typename Shift>
    void launchLinearTransform_(const AccessorRestrict<const Value, NDIM + 1, uint32_t>& input,
                                const AccessorRestrict<Value, NDIM + 1, uint32_t>& output, dim4_t shape,
                                Matrix matrices, Shift shift, float cutoff,
                                InterpMode interp_mode, cuda::Stream& stream) {
        const bool do_shift = noa::any(shift != Shift{});
        if (do_shift)
            linearTransform_<REMAP>(input, output, shape, matrices, shift, cutoff, interp_mode, stream);
        else
            linearTransform_<REMAP>(input, output, shape, matrices, empty_t{}, cutoff, interp_mode, stream);
    }

    template<fft::Remap REMAP, typename Value, typename MatrixOrEmpty, typename ShiftOrEmpty>
    void linearTransformSymmetry_(const AccessorRestrict<const Value, 3, uint32_t>& input,
                                  const AccessorRestrict<Value, 3, uint32_t>& output, dim4_t shape,
                                  MatrixOrEmpty matrix, const geometry::Symmetry& symmetry,
                                  ShiftOrEmpty shift, float cutoff, bool normalize,
                                  InterpMode interp_mode, cuda::Stream& stream) {
        NOA_ASSERT(shape[1] == 1);
        const auto input_shape_2d = int2_t(shape.get(2)).fft();
        const auto output_shape = safe_cast<int3_t>(dim3_t{shape[0], shape[2], shape[3]}).fft();

        const auto symmetry_count = static_cast<int64_t>(symmetry.count());
        const auto symmetry_matrices = noa::cuda::memory::PtrDevice<float33_t>::alloc(symmetry_count, stream);
        noa::cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);
        using real_t = traits::value_type_t<Value>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(input, input_shape_2d);
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices.get(),
                        symmetry_count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise3D("transform2D", output_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(input, input_shape_2d);
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices.get(),
                        symmetry_count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise3D("transform2D", output_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST:
            case INTERP_COSINE: {
                const auto interpolator =
                        noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(input, input_shape_2d);
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices.get(),
                        symmetry_count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise3D("transform2D", output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform2D", "{} is not supported", interp_mode);
        }
    }

    template<fft::Remap REMAP, typename Value, typename MatrixOrEmpty, typename ShiftOrEmpty>
    void linearTransformSymmetry_(const AccessorRestrict<const Value, 4, uint32_t>& input,
                                  const AccessorRestrict<Value, 4, uint32_t>& output, dim4_t shape,
                                  MatrixOrEmpty matrix, const geometry::Symmetry& symmetry,
                                  ShiftOrEmpty shift, float cutoff, bool normalize,
                                  InterpMode interp_mode, cuda::Stream& stream) {
        NOA_ASSERT(shape[1] == 1);
        const auto input_shape_3d = int3_t(shape.get(1)).fft();
        const auto output_shape = safe_cast<int4_t>(shape).fft();

        const auto symmetry_count = static_cast<int64_t>(symmetry.count());
        const auto symmetry_matrices = noa::cuda::memory::PtrDevice<float33_t>::alloc(symmetry_count, stream);
        noa::cuda::memory::copy(symmetry.get(), symmetry_matrices.get(), symmetry_count, stream);
        using real_t = traits::value_type_t<Value>;
        const real_t scaling = normalize ? 1 / static_cast<real_t>(symmetry_count + 1) : 1;

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_NEAREST>(input, input_shape_3d);
                const auto kernel = noa::geometry::fft::details::transformSymmetry3D<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices.get(),
                        symmetry_count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise4D("transform3D", output_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_LINEAR>(input, input_shape_3d);
                const auto kernel = noa::geometry::fft::details::transformSymmetry3D<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices.get(),
                        symmetry_count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise4D("transform3D", output_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST:
            case INTERP_COSINE: {
                const auto interpolator =
                        noa::geometry::interpolator3D<BORDER_ZERO, INTERP_COSINE>(input, input_shape_3d);
                const auto kernel = noa::geometry::fft::details::transformSymmetry3D<REMAP>(
                        interpolator, output, shape, matrix, symmetry_matrices.get(),
                        symmetry_count, scaling, shift, cutoff);
                return noa::cuda::utils::iwise4D("transform3D", output_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform3D", "{} is not supported", interp_mode);
        }
    }

    template<fft::Remap REMAP, int32_t NDIM, typename Value, typename Matrix, typename Shift>
    void launchLinearTransformSymmetry_(const AccessorRestrict<const Value, NDIM + 1, uint32_t>& input,
                                        const AccessorRestrict<Value, NDIM + 1, uint32_t>& output, dim4_t shape,
                                        Matrix matrix, const geometry::Symmetry& symmetry, Shift shift,
                                        float cutoff, bool normalize, InterpMode interp_mode, cuda::Stream& stream) {
        const bool apply_shift = noa::any(shift != Shift{});
        const bool apply_matrix = matrix != Matrix{};

        if (apply_shift && apply_matrix) {
            linearTransformSymmetry_<REMAP>(
                    input, output, shape,
                    matrix, symmetry, shift, cutoff,
                    normalize, interp_mode, stream);
        } else if (apply_shift) {
            linearTransformSymmetry_<REMAP>(
                    input, output, shape,
                    empty_t{}, symmetry, shift, cutoff,
                    normalize, interp_mode, stream);
        } else if (apply_matrix) {
            linearTransformSymmetry_<REMAP>(
                    input, output, shape,
                    matrix, symmetry, empty_t{}, cutoff,
                    normalize, interp_mode, stream);
        } else {
            linearTransformSymmetry_<REMAP>(
                    input, output, shape,
                    empty_t{}, symmetry, empty_t{}, cutoff,
                    normalize, interp_mode, stream);
        }
    }



    template<fft::Remap REMAP, bool LAYERED, typename Value, typename Matrix, typename ShiftOrEmpty>
    void linearTransform2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                            Value* output, dim4_t output_strides, dim4_t shape,
                            Matrix matrix, ShiftOrEmpty shift, float cutoff,
                            cuda::Stream& stream) {
        NOA_ASSERT(shape[1] == 1);
        const auto iwise_shape = safe_cast<int3_t>(dim3_t{shape[0], shape[2], shape[3]}).fft();
        const auto output_accessor = AccessorRestrict<Value, 3, uint32_t>(
                output, safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]}));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                return cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                return cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                return cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                return cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                return cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
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

    template<fft::Remap REMAP, bool LAYERED, typename Value, typename MatrixOrEmpty, typename ShiftOrEmpty>
    void linearTransformSymmetry2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                    Value* output, dim4_t output_strides, dim4_t shape,
                                    MatrixOrEmpty matrix, const geometry::Symmetry& symmetry,
                                    ShiftOrEmpty shift, float cutoff, bool normalize,
                                    cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const dim_t count = symmetry.count();
        const float33_t* symmetry_matrices = symmetry.get();
        using unique_ptr_t = cuda::memory::PtrDevice<float33_t>::alloc_unique_t;
        unique_ptr_t d_matrices = cuda::memory::PtrDevice<float33_t>::alloc(count, stream);
        cuda::memory::copy(symmetry_matrices, d_matrices.get(), count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(count + 1) : 1;

        NOA_ASSERT(shape[1] == 1);
        const auto iwise_shape = safe_cast<int3_t>(dim3_t{shape[0], shape[2], shape[3]}).fft();
        const auto output_accessor = AccessorRestrict<Value, 3, uint32_t>(
                output, safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]}));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, d_matrices.get(), count, scaling, shift, cutoff);
                return cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, d_matrices.get(), count, scaling, shift, cutoff);
                return cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, d_matrices.get(), count, scaling, shift, cutoff);
                return cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, d_matrices.get(), count, scaling, shift, cutoff);
                return cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, d_matrices.get(), count, scaling, shift, cutoff);
                return cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform2D", "{} is not supported", texture_interp_mode);
        }
    }

    template<fft::Remap REMAP, bool LAYERED, typename Value>
    void launchLinearTransformSymmetry2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                          Value* output, dim4_t output_strides, dim4_t output_shape,
                                          float22_t matrix, const geometry::Symmetry& symmetry, float2_t shift,
                                          float cutoff, bool normalize, cuda::Stream& stream) {
        const bool apply_shift = any(shift != 0.f);
        const bool apply_matrix = matrix != float22_t{};

        if (apply_shift && apply_matrix) {
            linearTransformSymmetry2D_<REMAP, LAYERED>(
                    texture, texture_interp_mode, output, output_strides, output_shape,
                    matrix, symmetry, shift, cutoff, normalize, stream);
        } else if (apply_shift) {
            linearTransformSymmetry2D_<REMAP, LAYERED>(
                    texture, texture_interp_mode, output, output_strides, output_shape,
                    empty_t{}, symmetry, shift, cutoff, normalize, stream);
        } else if (apply_matrix) {
            linearTransformSymmetry2D_<REMAP, LAYERED>(
                    texture, texture_interp_mode, output, output_strides, output_shape,
                    matrix, symmetry, empty_t{}, cutoff, normalize, stream);
        } else {
            linearTransformSymmetry2D_<REMAP, LAYERED>(
                    texture, texture_interp_mode, output, output_strides, output_shape,
                    empty_t{}, symmetry, empty_t{}, cutoff, normalize, stream);
        }
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform2D(const shared_t<Value[]>& input, dim4_t input_strides,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        if constexpr (!traits::is_floatXX_v<Matrix>) {
            NOA_ASSERT_DEVICE_PTR(inv_matrices.get(), stream.device());
        }

        const auto input_strides_2d = safe_cast<uint3_t>(dim3_t{input_strides[0], input_strides[2], input_strides[3]});
        const auto output_strides_2d = safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});
        const auto input_accessor = AccessorRestrict<const Value, 3, uint32_t>(input.get(), input_strides_2d);
        const auto output_accessor = AccessorRestrict<Value, 3, uint32_t>(output.get(), output_strides_2d);

        launchLinearTransform_<REMAP, 2>(
                input_accessor, output_accessor, shape,
                matrixOrShiftOrRawConstPtr<false>(inv_matrices),
                matrixOrShiftOrRawConstPtr<true>(shifts),
                cutoff, interp_mode, stream);

        stream.attach(input, output);
        if constexpr (!traits::is_floatXX_v<Matrix>)
            stream.attach(inv_matrices);
        if constexpr (!traits::is_floatX_v<Shift>)
            stream.attach(shifts);
    }

    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform3D(const shared_t<Value[]>& input, dim4_t input_strides,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     const Matrix& inv_matrices, const Shift& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        if constexpr (!traits::is_floatXX_v<Matrix>) {
            NOA_ASSERT_DEVICE_PTR(inv_matrices.get(), stream.device());
        }

        const auto input_strides_3d = safe_cast<uint4_t>(input_strides);
        const auto output_strides_3d = safe_cast<uint4_t>(output_strides);
        const auto input_accessor = AccessorRestrict<const Value, 4, uint32_t>(input.get(), input_strides_3d);
        const auto output_accessor = AccessorRestrict<Value, 4, uint32_t>(output.get(), output_strides_3d);

        launchLinearTransform_<REMAP, 3>(
                input_accessor, output_accessor, shape,
                matrixOrShiftOrRawConstPtr<false>(inv_matrices),
                matrixOrShiftOrRawConstPtr<true>(shifts),
                cutoff, interp_mode, stream);

        stream.attach(input, output);
        if constexpr (!traits::is_floatXX_v<Matrix>)
            stream.attach(inv_matrices);
        if constexpr (!traits::is_floatX_v<Shift>)
            stream.attach(shifts);
    }

    template<Remap REMAP, typename Value, typename>
    void transform2D(const shared_t<Value[]>& input, dim4_t input_strides,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     float22_t inv_matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        if (!symmetry.count()) {
            return transform2D<REMAP>(input, input_strides, output, output_strides, shape,
                                      inv_matrix, shift, cutoff, interp_mode, stream);
        }

        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const auto input_strides_2d = safe_cast<uint3_t>(dim3_t{input_strides[0], input_strides[2], input_strides[3]});
        const auto output_strides_2d = safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});
        const auto input_accessor = AccessorRestrict<const Value, 3, uint32_t>(input.get(), input_strides_2d);
        const auto output_accessor = AccessorRestrict<Value, 3, uint32_t>(output.get(), output_strides_2d);

        launchLinearTransformSymmetry_<REMAP, 2>(
                input_accessor, output_accessor, shape,
                inv_matrix, symmetry, shift,
                cutoff, normalize, interp_mode, stream);

        stream.attach(input, output, symmetry.share());
    }

    template<Remap REMAP, typename Value, typename>
    void transform3D(const shared_t<Value[]>& input, dim4_t input_strides,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     float33_t inv_matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        if (!symmetry.count()) {
            return transform3D<REMAP>(input, input_strides, output, output_strides, shape,
                                      inv_matrix, shift, cutoff, interp_mode, stream);
        }

        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const auto input_strides_3d = safe_cast<uint4_t>(input_strides);
        const auto output_strides_3d = safe_cast<uint4_t>(output_strides);
        const auto input_accessor = AccessorRestrict<const Value, 4, uint32_t>(input.get(), input_strides_3d);
        const auto output_accessor = AccessorRestrict<Value, 4, uint32_t>(output.get(), output_strides_3d);

        launchLinearTransformSymmetry_<REMAP, 3>(
                input_accessor, output_accessor, shape,
                inv_matrix, symmetry, shift,
                cutoff, normalize, interp_mode, stream);

        stream.attach(input, output, symmetry.share());
    }

    #define NOA_INSTANTIATE_TRANSFORM_SYMMETRY(T)                                                                                                                                                   \
    template void transform2D<Remap::HC2HC, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&l);  \
    template void transform2D<Remap::HC2H, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&);    \
    template void transform3D<Remap::HC2HC, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&);   \
    template void transform3D<Remap::HC2H, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM2D_(T, M, S) \
    template void transform2D<Remap::HC2H,  T, M, S, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, InterpMode, Stream&); \
    template void transform2D<Remap::HC2HC, T, M, S, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, InterpMode, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM3D_(T, M, S) \
    template void transform3D<Remap::HC2H,  T, M, S, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, InterpMode, Stream&); \
    template void transform3D<Remap::HC2HC, T, M, S, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, InterpMode, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM2D_ALL_(T)                         \
    NOA_INSTANTIATE_TRANSFORM2D_(T, float22_t, float2_t);               \
    NOA_INSTANTIATE_TRANSFORM2D_(T, shared_t<float22_t[]>, float2_t);   \
    NOA_INSTANTIATE_TRANSFORM2D_(T, float22_t, shared_t<float2_t[]>);   \
    NOA_INSTANTIATE_TRANSFORM2D_(T, shared_t<float22_t[]>, shared_t<float2_t[]>)

    #define NOA_INSTANTIATE_TRANSFORM3D_ALL_(T)                         \
    NOA_INSTANTIATE_TRANSFORM3D_(T, float33_t, float3_t);               \
    NOA_INSTANTIATE_TRANSFORM3D_(T, shared_t<float33_t[]>, float3_t);   \
    NOA_INSTANTIATE_TRANSFORM3D_(T, float33_t, shared_t<float3_t[]>);   \
    NOA_INSTANTIATE_TRANSFORM3D_(T, shared_t<float33_t[]>, shared_t<float3_t[]>)

    #define NOA_INSTANTIATE_TRANSFORM_ALL_(T)   \
    NOA_INSTANTIATE_TRANSFORM2D_ALL_(T);        \
    NOA_INSTANTIATE_TRANSFORM3D_ALL_(T);        \
    NOA_INSTANTIATE_TRANSFORM_SYMMETRY(T)

    NOA_INSTANTIATE_TRANSFORM_ALL_(float);
    NOA_INSTANTIATE_TRANSFORM_ALL_(double);
    NOA_INSTANTIATE_TRANSFORM_ALL_(cfloat_t);
    NOA_INSTANTIATE_TRANSFORM_ALL_(cdouble_t);
}
