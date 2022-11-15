#include "noa/common/Assert.h"
#include "noa/common/geometry/details/LinearTransform2DFourier.h"

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

    template<bool IS_OPTIONAL, typename Wrapper, typename Value>
    auto matrixOrShiftOrRawConstPtrOnDevice_(Wrapper wrapper, size_t count,
                                             cuda::memory::PtrDevice<Value>& buffer,
                                             cuda::Stream& stream) {
        using output_t = std::conditional_t<traits::is_floatXX_v<Wrapper> || traits::is_floatX_v<Wrapper>,
                                            traits::remove_ref_cv_t<Wrapper>,
                                            const traits::element_type_t<Wrapper>*>;
        if constexpr (traits::is_floatXX_v<Wrapper> || traits::is_floatX_v<Wrapper>) {
            return output_t(wrapper);
        } else {
            if (IS_OPTIONAL && wrapper.get() == nullptr)
                return output_t{};
            return output_t(cuda::utils::ensureDeviceAccess(wrapper.get(), stream, buffer, count));
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
        NOA_ASSERT(input && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        // Ensure transformation parameters are accessible to the GPU:
        memory::PtrDevice<float22_t> matrix_buffer;
        memory::PtrDevice<float2_t> shift_buffer;
        auto inv_matrices_ = matrixOrShiftOrRawConstPtrOnDevice_<false>(inv_matrices, shape[0], matrix_buffer, stream);
        auto shifts_ = matrixOrShiftOrRawConstPtrOnDevice_<true>(shifts, shape[0], shift_buffer, stream);

        memory::PtrArray<Value> array({1, 1, shape[2], shape[3] / 2 + 1});
        memory::PtrTexture texture(array.get(), interp_mode, BORDER_ZERO);

        dim_t iterations;
        dim4_t output_shape;
        if (input_strides[0] == 0) {
            iterations = 1;
            output_shape = {shape[0], 1, shape[2], shape[3]};
        } else {
            iterations = shape[0];
            output_shape = {1, 1, shape[2], shape[3]};
        }

        for (dim_t i = 0; i < iterations; ++i) {
            memory::copy(input.get() + i * input_strides[0], input_strides,
                         array.get(), array.shape(), stream);
            launchLinearTransform2D_<REMAP, false>(
                    texture.get(), interp_mode,
                    output.get() + i * output_strides[0], output_strides,
                    output_shape, inv_matrices_, shifts_, cutoff, stream);

            if constexpr (!traits::is_float22_v<Matrix>)
                ++inv_matrices_;
            if constexpr (!traits::is_float2_v<Shift>)
                ++shifts_;
        }

        stream.attach(input, output, array.share(), texture.share());
        if constexpr (!traits::is_float22_v<Matrix>)
            stream.attach(inv_matrices);
        if constexpr (!traits::is_float2_v<Shift>)
            stream.attach(shifts);
    }

    template<Remap REMAP, typename Value, typename Matrix, typename Shift, typename>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const Matrix& inv_matrices, const Shift& shifts, float cutoff, Stream& stream) {
        NOA_ASSERT(array && texture && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        // Ensure transformation parameters are accessible to the GPU:
        memory::PtrDevice<float22_t> matrix_buffer;
        memory::PtrDevice<float2_t> shift_buffer;
        auto inv_matrices_ = matrixOrShiftOrRawConstPtrOnDevice_<false>(inv_matrices, output_shape[0], matrix_buffer, stream);
        auto shifts_ = matrixOrShiftOrRawConstPtrOnDevice_<true>(shifts, output_shape[0], shift_buffer, stream);

        const bool is_layered = memory::PtrArray<Value>::isLayered(array.get());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());

        if (is_layered) {
            launchLinearTransform2D_<REMAP, true>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides,
                    output_shape, inv_matrices_, shifts_, cutoff, stream);
        } else {
            launchLinearTransform2D_<REMAP, false>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides,
                    output_shape, inv_matrices_, shifts_, cutoff, stream);
        }

        stream.attach(array, texture, output);
        if constexpr (!traits::is_float22_v<Matrix>)
            stream.attach(inv_matrices);
        if constexpr (!traits::is_float2_v<Shift>)
            stream.attach(shifts);
    }

    template<Remap REMAP, typename Value, typename>
    void transform2D(const shared_t<Value[]>& input, dim4_t input_strides,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t shape,
                     float22_t inv_matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        if (!symmetry.count())
            return transform2D<REMAP>(input, input_strides, output, output_strides, shape,
                                      inv_matrix, shift, cutoff, interp_mode, stream);

        NOA_ASSERT(input && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        memory::PtrArray<Value> array(dim4_t{1, 1, shape[2], shape[3]}.fft());
        memory::PtrTexture texture(array.get(), interp_mode, BORDER_ZERO);

        dim_t iterations;
        dim4_t output_shape;
        if (input_strides[0] == 0) {
            iterations = 1;
            output_shape = {shape[0], 1, shape[2], shape[3]};
        } else {
            iterations = shape[0];
            output_shape = {1, 1, shape[2], shape[3]};
        }
        for (dim_t i = 0; i < iterations; ++i) {
            memory::copy(input.get() + i * input_strides[0], input_strides, array.get(), array.shape(), stream);
            launchLinearTransformSymmetry2D_<REMAP, false>(
                    texture.get(), interp_mode,
                    output.get() + i * output_strides[0], output_strides, output_shape,
                    inv_matrix, symmetry, shift, cutoff, normalize, stream);
        }
        stream.attach(input, output, symmetry.share(), array.share(), texture.share());
    }

    template<Remap REMAP, typename Value, typename>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<Value[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float22_t inv_matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        const bool is_layered = memory::PtrArray<Value>::isLayered(array.get());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());

        if (is_layered) {
            launchLinearTransformSymmetry2D_<REMAP, true>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides,
                    output_shape, inv_matrix, symmetry, shift,
                    cutoff, normalize, stream);
        } else {
            launchLinearTransformSymmetry2D_<REMAP, false>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides,
                    output_shape, inv_matrix, symmetry, shift,
                    cutoff, normalize, stream);
        }
        stream.attach(array, texture, output, symmetry.share());
    }

    #define NOA_INSTANTIATE_TRANSFORM_2D_(T, M, S)                                                                                                                                                                  \
    template void transform2D<Remap::HC2H,  T, M, S, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, InterpMode, Stream&);                                     \
    template void transform2D<Remap::HC2HC, T, M, S, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, InterpMode, Stream&);                                     \
    template void transform2D<Remap::HC2H,  T, M, S, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, Stream&); \
    template void transform2D<Remap::HC2HC, T, M, S, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, const shared_t<T[]>&, dim4_t, dim4_t, const M&, const S&, float, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_SYMMETRY_2D_(T)                                                                                                                                                                                   \
    template void transform2D<Remap::HC2HC, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&);                                       \
    template void transform2D<Remap::HC2H, T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, InterpMode, bool, Stream&);                                        \
    template void transform2D<Remap::HC2HC, T, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, bool, Stream&);   \
    template void transform2D<Remap::HC2H, T, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, const shared_t<T[]>&, dim4_t, dim4_t, float22_t, const Symmetry&, float2_t, float, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_2D_ALL_(T)                                    \
    NOA_INSTANTIATE_TRANSFORM_2D_(T, shared_t<float22_t[]>, shared_t<float2_t[]>);  \
    NOA_INSTANTIATE_TRANSFORM_2D_(T, shared_t<float22_t[]>, float2_t);              \
    NOA_INSTANTIATE_TRANSFORM_2D_(T, float22_t, shared_t<float2_t[]>);              \
    NOA_INSTANTIATE_TRANSFORM_2D_(T, float22_t, float2_t);                          \
    NOA_INSTANTIATE_TRANSFORM_SYMMETRY_2D_(T)

    NOA_INSTANTIATE_TRANSFORM_2D_ALL_(float);
    NOA_INSTANTIATE_TRANSFORM_2D_ALL_(cfloat_t);
}
