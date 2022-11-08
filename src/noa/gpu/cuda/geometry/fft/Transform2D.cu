#include "noa/common/Assert.h"
#include "noa/common/geometry/details/LinearTransformations2D.h"

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

    template<bool IS_OPTIONAL, typename wrapper_t, typename value_t>
    auto matrixOrShiftOrRawConstPtrOnDevice_(wrapper_t wrapper, size_t count,
                                             cuda::memory::PtrDevice<value_t>& buffer,
                                             cuda::Stream& stream) {
        using output_t = std::conditional_t<traits::is_floatXX_v<wrapper_t> || traits::is_floatX_v<wrapper_t>,
                                            traits::remove_ref_cv_t<wrapper_t>,
                                            const traits::element_type_t<wrapper_t>*>;
        if constexpr (traits::is_floatXX_v<wrapper_t> || traits::is_floatX_v<wrapper_t>) {
            return output_t(wrapper);
        } else {
            if (IS_OPTIONAL && wrapper.get() == nullptr)
                return output_t{};
            return output_t(cuda::utils::ensureDeviceAccess(wrapper.get(), stream, buffer, count));
        }
    }

    template<fft::Remap REMAP, bool LAYERED,
             typename data_t, typename matrix_t, typename shift_or_empty_t>
    void linearTransform2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 data_t* output, dim4_t output_strides, dim4_t shape,
                 matrix_t matrix, shift_or_empty_t shift, float cutoff,
                 cuda::Stream& stream) {
        NOA_ASSERT(shape[1] == 1);
        const auto iwise_shape = safe_cast<int3_t>(dim3_t{shape[0], shape[2], shape[3]}).fft();
        const auto output_accessor = AccessorRestrict<data_t, 3, uint32_t>(
                output, safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]}));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transform2D<REMAP, LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shape, matrix, shift, cutoff);
                cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform2D", "{} is not supported", texture_interp_mode);
        }
    }

    template<fft::Remap REMAP, bool LAYERED, typename data_t, typename matrix_t, typename shift_t>
    void launchLinearTransform2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                  data_t* output, dim4_t output_strides, dim4_t shape,
                                  matrix_t matrix, shift_t shift, float cutoff,
                                  cuda::Stream& stream) {
        const bool do_shift = noa::any(shift != shift_t{});
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

    template<fft::Remap REMAP, bool LAYERED, typename data_t, typename matrix_or_empty_t, typename shift_or_empty_t>
    void linearTransformSymmetry2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                    data_t* output, dim4_t output_strides, dim4_t shape,
                                    matrix_or_empty_t matrix, const geometry::Symmetry& symmetry,
                                    shift_or_empty_t shift, float cutoff, bool normalize,
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
        const auto output_accessor = AccessorRestrict<data_t, 3, uint32_t>(
                output, safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]}));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, d_matrices.get(), count, scaling, shift, cutoff);
                cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, d_matrices.get(), count, scaling, shift, cutoff);
                cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, d_matrices.get(), count, scaling, shift, cutoff);
                cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, d_matrices.get(), count, scaling, shift, cutoff);
                cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::transformSymmetry2D<REMAP, LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shape,
                        matrix, d_matrices.get(), count, scaling, shift, cutoff);
                cuda::utils::iwise3D("geometry::fft::transform2D", iwise_shape, kernel, stream);
            }
            default:
                NOA_THROW_FUNC("transform2D", "{} is not supported", texture_interp_mode);
        }
    }

    template<fft::Remap REMAP, bool LAYERED, typename data_t>
    void launchLinearTransformSymmetry2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                          data_t* output, dim4_t output_strides, dim4_t output_shape,
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
    template<Remap REMAP, typename data_t, typename matrix_t, typename shift_t, typename>
    void transform2D(const shared_t<data_t[]>& input, dim4_t input_strides,
                     const shared_t<data_t[]>& output, dim4_t output_strides, dim4_t shape,
                     const matrix_t& matrices, const shift_t& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(input && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        // Ensure transformation parameters are accessible to the GPU:
        memory::PtrDevice<float22_t> matrix_buffer;
        memory::PtrDevice<float2_t> shift_buffer;
        auto matrices_ = matrixOrShiftOrRawConstPtrOnDevice_<false>(matrices, shape[0], matrix_buffer, stream);
        auto shifts_ = matrixOrShiftOrRawConstPtrOnDevice_<true>(shifts, shape[0], shift_buffer, stream);

        memory::PtrArray<data_t> array({1, 1, shape[2], shape[3] / 2 + 1});
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
                    output_shape, matrices_, shifts_, cutoff, stream);

            if constexpr (!traits::is_float22_v<matrix_t>)
                ++matrices_;
            if constexpr (!traits::is_float2_v<shift_t>)
                ++shifts_;
        }

        stream.attach(input, output, array.share(), texture.share());
        if constexpr (!traits::is_float22_v<matrix_t>)
            stream.attach(matrices);
        if constexpr (!traits::is_float2_v<shift_t>)
            stream.attach(shifts);
    }

    template<Remap REMAP, typename data_t, typename matrix_t, typename shift_t, typename>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<data_t[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const matrix_t& matrices, const shift_t& shifts, float cutoff, Stream& stream) {
        NOA_ASSERT(array && texture && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        // Ensure transformation parameters are accessible to the GPU:
        memory::PtrDevice<float22_t> matrix_buffer;
        memory::PtrDevice<float2_t> shift_buffer;
        auto matrices_ = matrixOrShiftOrRawConstPtrOnDevice_<false>(matrices, output_shape[0], matrix_buffer, stream);
        auto shifts_ = matrixOrShiftOrRawConstPtrOnDevice_<true>(shifts, output_shape[0], shift_buffer, stream);

        const bool is_layered = memory::PtrArray<data_t>::isLayered(array.get());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());

        if (is_layered) {
            launchLinearTransform2D_<REMAP, true>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides,
                    output_shape, matrices_, shifts_, cutoff, stream);
        } else {
            launchLinearTransform2D_<REMAP, false>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides,
                    output_shape, matrices_, shifts_, cutoff, stream);
        }

        stream.attach(array, texture, output);
        if constexpr (!traits::is_float22_v<matrix_t>)
            stream.attach(matrices);
        if constexpr (!traits::is_float2_v<shift_t>)
            stream.attach(shifts);
    }

    template<Remap REMAP, typename data_t, typename>
    void transform2D(const shared_t<data_t[]>& input, dim4_t input_strides,
                     const shared_t<data_t[]>& output, dim4_t output_strides, dim4_t shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream) {
        if (!symmetry.count())
            return transform2D<REMAP>(input, input_strides, output, output_strides, shape,
                                      matrix, shift, cutoff, interp_mode, stream);

        NOA_ASSERT(input && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        memory::PtrArray<data_t> array(dim4_t{1, 1, shape[2], shape[3]}.fft());
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
                    matrix, symmetry, shift, cutoff, normalize, stream);
        }
        stream.attach(input, output, symmetry.share(), array.share(), texture.share());
    }

    template<Remap REMAP, typename data_t, typename>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<data_t[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        const bool is_layered = memory::PtrArray<data_t>::isLayered(array.get());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());

        if (is_layered) {
            launchLinearTransformSymmetry2D_<REMAP, true>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides,
                    output_shape, matrix, symmetry, shift,
                    cutoff, normalize, stream);
        } else {
            launchLinearTransformSymmetry2D_<REMAP, false>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides,
                    output_shape, matrix, symmetry, shift,
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
