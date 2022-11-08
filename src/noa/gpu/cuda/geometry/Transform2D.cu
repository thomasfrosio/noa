#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/geometry/details/LinearTransformations2D.h"

#include "noa/gpu/cuda/util/Iwise.cuh"
#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"

#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/geometry/Prefilter.h"
#include "noa/gpu/cuda/geometry/Transform.h"

namespace {
    using namespace ::noa;

    template<bool LAYERED, typename data_t, typename matrix_t>
    void launchTransformTexture2D_(cudaTextureObject_t texture, dim4_t texture_shape,
                                   InterpMode texture_interp_mode, BorderMode texture_border_mode,
                                   data_t* output, dim4_t output_strides, dim4_t output_shape,
                                   matrix_t matrices, cuda::Stream& stream) {
        NOA_ASSERT(output_shape[1] == 1);
        NOA_ASSERT(texture_shape[1] == 1);
        NOA_ASSERT((LAYERED && texture_shape[0] == output_shape[0]) ||
                   (!LAYERED && texture_shape[0] == 1));

        const auto iwise_shape = safe_cast<uint3_t>(dim3_t{output_shape[0], output_shape[2], output_shape[3]});
        const auto output_accessor = AccessorRestrict<data_t, 3, uint32_t>(
                output, safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]}));

        // Copy matrices to device if not available yet.
        using value_t = std::remove_cv_t<std::remove_pointer_t<matrix_t>>;
        cuda::memory::PtrDevice<value_t> buffer;
        if constexpr (std::is_pointer_v<matrix_t>)
            matrices = cuda::util::ensureDeviceAccess(matrices, stream, buffer, output_shape[0]);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            const float2_t i_shape(texture_shape.get(2));

            if (texture_interp_mode == INTERP_NEAREST) {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, data_t, true, LAYERED>;
                const auto kernel = noa::geometry::details::transform2D<LAYERED, uint32_t>(
                        interpolator_t(texture, i_shape), output_accessor, matrices);
                cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, data_t, true, LAYERED>;
                const auto kernel = noa::geometry::details::transform2D<LAYERED, uint32_t>(
                        interpolator_t(texture, i_shape), output_accessor, matrices);
                cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }

        } else {
            switch (texture_interp_mode) {
                case INTERP_NEAREST: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, data_t, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<LAYERED, uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
                }
                case INTERP_LINEAR: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, data_t, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<LAYERED, uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
                }
                case INTERP_COSINE: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, data_t, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<LAYERED, uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
                }
                case INTERP_CUBIC: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC, data_t, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<LAYERED, uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
                }
                case INTERP_CUBIC_BSPLINE: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE, data_t, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<LAYERED, uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
                }
                case INTERP_LINEAR_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, data_t, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<LAYERED, uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
                }
                case INTERP_COSINE_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, data_t, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<LAYERED, uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
                }
                case INTERP_CUBIC_BSPLINE_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE_FAST, data_t, false, LAYERED>;
                    const auto kernel = noa::geometry::details::transform2D<LAYERED, uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
                }
            }
        }
    }

    template<bool LAYERED, typename data_t>
    void launchTransformSymmetryTexture2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                           data_t* output, dim4_t output_strides, dim4_t output_shape,
                                           float2_t shift, float22_t matrix, const geometry::Symmetry& symmetry,
                                           float2_t center, bool normalize, cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const dim_t count = symmetry.count();
        const float33_t* symmetry_matrices = symmetry.get();
        using unique_ptr = cuda::memory::PtrDevice<float33_t>::alloc_unique_t;
        unique_ptr d_matrices = cuda::memory::PtrDevice<float33_t>::alloc(count, stream);
        cuda::memory::copy(symmetry_matrices, d_matrices.get(), count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(count + 1) : 1;

        const auto iwise_shape = safe_cast<uint3_t>(dim3_t{output_shape[0], output_shape[2], output_shape[3]});
        const auto output_accessor = AccessorRestrict<data_t, 3, uint32_t>(
                output, safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]}));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE_FAST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::transformSymmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::transform2D", iwise_shape, kernel, stream);
            }
        }
    }

    template<bool LAYERED, typename data_t>
    void launchSymmetrize2D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                             data_t* output, dim4_t output_strides, dim4_t output_shape,
                             const geometry::Symmetry& symmetry, float2_t center, bool normalize,
                             cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const dim_t count = symmetry.count();
        const float33_t* symmetry_matrices = symmetry.get();
        using unique_ptr = cuda::memory::PtrDevice<float33_t>::alloc_unique_t;
        unique_ptr d_matrices = cuda::memory::PtrDevice<float33_t>::alloc(count, stream);
        cuda::memory::copy(symmetry_matrices, d_matrices.get(), count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(count + 1) : 1;

        const auto iwise_shape = safe_cast<uint3_t>(dim3_t{output_shape[0], output_shape[2], output_shape[3]});
        const auto output_accessor = AccessorRestrict<data_t, 3, uint32_t>(
                output, safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]}));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE_FAST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::details::symmetry2D<LAYERED, int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise3D("geometry::symmetry2D", iwise_shape, kernel, stream);
            }
        }
    }

    // Updates the input and output shape to correctly broadcast the input.
    // Prefilter the input if needed.
    template<typename T>
    auto preprocess2D_(const shared_t<T[]>& input, dim4_t& input_strides, dim4_t& input_shape,
                      const shared_t<T[]>& output, dim4_t output_strides, dim4_t& output_shape,
                      InterpMode interp_mode, bool prefilter, cuda::Stream& stream) {

        // Be careful about the symmetry case, where the input and output shape are the same objects.
        // In this case, there's no need to update the shapes.
        if (&input_shape != &output_shape) {
            // If the output is batched, the input is allowed to either have the same number of batches
            // or have a single batch. In the later case, the single batch is used to compute all output
            // batches. The following makes sure to correctly identify whether the input is batched.
            // We update this before doing the prefilter, since it can simplify the processing.
            if (input_strides[0] == 0)
                input_shape[0] = 1;
            else if (input_shape[0] == 1)
                input_strides[0] = 0;

            // If the input is not batched, then we need to ensure that the processing loop will compute
            // one batch at a time, for both the input and the output. Otherwise, the processing loop
            // should run once, processing all output batches at the same time using the unique input batch.
            if (input_shape[0] > 1)
                output_shape[0] = 1;
        }

        shared_t<T[]> buffer;
        const T* buffer_ptr;
        dim4_t buffer_strides;
        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            if (input_shape[2] != output_shape[2] || input_shape[3] != output_shape[3]) {
                buffer = cuda::memory::PtrDevice<T>::alloc(input_shape.elements(), stream);
                const dim4_t contiguous_strides = input_shape.strides();
                cuda::geometry::bspline::prefilter(
                        input, input_strides, buffer, contiguous_strides, input_shape, stream);
                buffer_ptr = buffer.get();
                buffer_strides = contiguous_strides;
            } else {
                // Whether input is batched or not, since we copy to the CUDA array, we can use the output as buffer.
                cuda::geometry::bspline::prefilter(input, input_strides, output, output_strides, input_shape, stream);
                buffer_ptr = output.get();
                buffer_strides = output_strides;
            }
        } else {
            buffer_ptr = input.get();
            buffer_strides = input_strides;
        }

        return std::tuple<shared_t<T[]>, const T*, dim4_t>(buffer, buffer_ptr, buffer_strides);
    }

    template<typename T>
    auto matrixOrRawConstPtr(const T& v, size_t index = 0) {
        if constexpr (traits::is_float23_v<T> || traits::is_float33_v<T>) {
            return float23_t(v);
        } else {
            NOA_ASSERT(v != nullptr);
            using clean_t = traits::remove_ref_cv_t<T>;
            using raw_const_ptr_t = const typename clean_t::element_type*;
            return static_cast<raw_const_ptr_t>(v.get() + index);
        }
    }
}

namespace noa::cuda::geometry {
    template<typename T, typename MAT, typename>
    void transform2D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const MAT& matrices, InterpMode interp_mode, BorderMode border_mode,
                     bool prefilter, Stream& stream) {
        NOA_ASSERT(input && all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1 && output_shape[1] == 1);

        // Prepare the input array:
        auto [buffer, buffer_ptr, buffer_strides] = preprocess2D_(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                interp_mode, prefilter, stream);

        // Copy to texture and launch (per input batch):
        cuda::memory::PtrArray<T> array({1, 1, input_shape[2], input_shape[3]});
        cuda::memory::PtrTexture texture(array.get(), interp_mode, border_mode);
        for (dim_t i = 0; i < input_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_strides[0],
                               buffer_strides, array.get(), array.shape(), stream);
            launchTransformTexture2D_<false>(
                    texture.get(), array.shape(), interp_mode, border_mode,
                    output.get() + i * output_strides[0], output_strides, output_shape,
                    matrixOrRawConstPtr(matrices, i), stream);
        }
        stream.attach(input, output, array.share(), texture.share());
        if constexpr (!traits::is_floatXX_v<MAT>)
            stream.attach(matrices);
    }

    template<typename T, typename MAT, typename>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, dim4_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const MAT& matrices, Stream& stream) {
        NOA_ASSERT(array && texture && all(texture_shape > 0) && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        const bool is_layered = memory::PtrArray<T>::isLayered(array.get());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());

        if (is_layered) {
            launchTransformTexture2D_<true>(
                    *texture, texture_shape, texture_interp_mode, texture_border_mode,
                    output.get(), output_strides, output_shape,
                    matrixOrRawConstPtr(matrices, 0), stream);
        } else {
            launchTransformTexture2D_<false>(
                    *texture, texture_shape, texture_interp_mode, texture_border_mode,
                    output.get(), output_strides, output_shape,
                    matrixOrRawConstPtr(matrices, 0), stream);
        }

        if constexpr (traits::is_floatXX_v<MAT>)
            stream.attach(array, texture, output);
        else
            stream.attach(array, texture, output, matrices);
    }

    template<typename T, typename>
    void transform2D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float2_t shift, float22_t matrix, const Symmetry& symmetry, float2_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream) {
        NOA_ASSERT(input && all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1 && output_shape[1] == 1);

        // Prepare the input array:
        auto [buffer, buffer_ptr, buffer_strides] = preprocess2D_(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                interp_mode, prefilter, stream);

        // Copy to texture and launch (per input batch):
        cuda::memory::PtrArray<T> array({1, 1, input_shape[2], input_shape[3]});
        cuda::memory::PtrTexture texture(array.get(), interp_mode, BORDER_ZERO);
        for (dim_t i = 0; i < input_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_strides[0],
                               buffer_strides, array.get(), array.shape(), stream);
            launchTransformSymmetryTexture2D_<false>(
                    texture.get(), interp_mode,
                    output.get() + i * output_strides[0], output_strides, output_shape,
                    shift, matrix, symmetry, center, normalize, stream);
        }
        stream.attach(input, output, symmetry.share(), array.share(), texture.share());
        if (buffer)
            stream.attach(buffer);
    }

    template<typename T, typename>
    void transform2D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture,
                     InterpMode texture_interp_mode, dim4_t texture_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float2_t shift, float22_t matrix, const Symmetry& symmetry, float2_t center,
                     bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        const bool is_layered = memory::PtrArray<T>::isLayered(array.get());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());

        NOA_ASSERT(texture_shape[1] == 1);
        NOA_ASSERT((is_layered && texture_shape[0] == output_shape[0]) ||
                   (!is_layered && texture_shape[0] == 1));
        (void) texture_shape;

        if (is_layered) {
            launchTransformSymmetryTexture2D_<true>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides, output_shape,
                    shift, matrix, symmetry, center, normalize, stream);
        } else {
            launchTransformSymmetryTexture2D_<false>(
                    *texture, texture_interp_mode,
                    output.get(), output_strides, output_shape,
                    shift, matrix, symmetry, center, normalize, stream);
        }
        stream.attach(array, texture, output, symmetry.share());
    }

    template<typename T, typename>
    void symmetrize2D(const shared_t<T[]>& input, dim4_t input_strides,
                      const shared_t<T[]>& output, dim4_t output_strides,
                      dim4_t shape, const Symmetry& symmetry, float2_t center,
                      InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream) {
        NOA_ASSERT(all(shape > 0) && input);
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(shape[1] == 1);

        if (!symmetry.count()) {
            if (input != output)
                memory::copy(input, input_strides, output, output_strides, shape, stream);
            return;
        }

        auto [buffer, buffer_ptr, buffer_strides] = preprocess2D_(
                input, input_strides, shape,
                output, output_strides, shape,
                interp_mode, prefilter, stream);

        // Copy to texture and launch (per input batch):
        cuda::memory::PtrArray<T> array({1, 1, shape[2], shape[3]});
        cuda::memory::PtrTexture texture(array.get(), interp_mode, BORDER_ZERO);
        for (dim_t i = 0; i < shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_strides[0], buffer_strides, array.get(), shape, stream);
            launchSymmetrize2D_<false>(
                    texture.get(), interp_mode,
                    output.get() + i * output_strides[0], output_strides, shape,
                    symmetry, center, normalize, stream);
        }
        stream.attach(input, output, symmetry.share(), array.share(), texture.share());
    }

    template<typename T, typename>
    void symmetrize2D(const shared_t<cudaArray>& array,
                      const shared_t<cudaTextureObject_t>& texture,
                      InterpMode texture_interp_mode, dim4_t texture_shape,
                      const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                      const Symmetry& symmetry, float2_t center, bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        const bool is_layered = memory::PtrArray<T>::isLayered(array.get());
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

    #define NOA_INSTANTIATE_TRANSFORM_2D_MATRIX(T, M)                                                                                                                                                                           \
    template void transform2D<T, shared_t<M[]>, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<M[]>&, InterpMode, BorderMode, bool, Stream&);                                 \
    template void transform2D<T, M, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, InterpMode, BorderMode, bool, Stream&);                                                         \
    template void transform2D<T, shared_t<M[]>, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, dim4_t, InterpMode, BorderMode, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<M[]>&, Stream&);   \
    template void transform2D<T, M, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, dim4_t, InterpMode, BorderMode, const shared_t<T[]>&, dim4_t, dim4_t, const M&, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_SYM_(T)                                                                                                                                                           \
    template void transform2D<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float22_t, const Symmetry&, float2_t, InterpMode, bool, bool, Stream&);    \
    template void transform2D<T, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float22_t, const Symmetry&, float2_t, bool, Stream&)

    #define NOA_INSTANTIATE_SYM_(T)                                                                                                                                         \
    template void symmetrize2D<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const Symmetry&, float2_t, InterpMode, bool, bool, Stream&);    \
    template void symmetrize2D<T, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const Symmetry&, float2_t, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_2D_(T)            \
    NOA_INSTANTIATE_TRANSFORM_2D_MATRIX(T, float23_t);  \
    NOA_INSTANTIATE_TRANSFORM_2D_MATRIX(T, float33_t);  \
    NOA_INSTANTIATE_TRANSFORM_SYM_(T);                  \
    NOA_INSTANTIATE_SYM_(T)

    NOA_INSTANTIATE_TRANSFORM_2D_(float);
    NOA_INSTANTIATE_TRANSFORM_2D_(cfloat_t);
}
