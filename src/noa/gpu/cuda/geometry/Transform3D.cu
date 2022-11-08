#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/geometry/details/LinearTransform3D.h"

#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"

#include "noa/gpu/cuda/geometry/Transform.h"
#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/geometry/Prefilter.h"

namespace {
    using namespace ::noa;

    template<typename data_t, typename matrix_t>
    void launchTransformTexture3D_(cudaTextureObject_t texture, dim4_t texture_shape,
                                   InterpMode texture_interp_mode, BorderMode texture_border_mode,
                                   data_t* output, dim4_t output_strides, dim4_t output_shape,
                                   matrix_t matrices, cuda::Stream& stream) {
        NOA_ASSERT(texture_shape[0] == 1);

        const auto iwise_shape = safe_cast<uint4_t>(output_shape);
        const auto output_accessor = AccessorRestrict<data_t, 4, uint32_t>(output, safe_cast<uint4_t>(output_strides));

        // Copy matrices to device if not available yet.
        using value_t = std::remove_cv_t<std::remove_pointer_t<matrix_t>>;
        cuda::memory::PtrDevice<value_t> buffer;
        if constexpr (std::is_pointer_v<matrix_t>)
            matrices = cuda::utils::ensureDeviceAccess(matrices, stream, buffer, output_shape[0]);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            const float3_t i_shape(texture_shape.get(1));

            if (texture_interp_mode == INTERP_NEAREST) {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_NEAREST, data_t, true>;
                const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                        interpolator_t(texture, i_shape), output_accessor, matrices);
                cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, data_t, true>;
                const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                        interpolator_t(texture, i_shape), output_accessor, matrices);
                cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            switch (texture_interp_mode) {
                case INTERP_NEAREST: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_NEAREST, data_t>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                }
                case INTERP_LINEAR: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR, data_t>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                }
                case INTERP_COSINE: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE, data_t>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                }
                case INTERP_CUBIC: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC, data_t>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                }
                case INTERP_CUBIC_BSPLINE: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE, data_t>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                }
                case INTERP_LINEAR_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, data_t>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                }
                case INTERP_COSINE_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE_FAST, data_t>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                }
                case INTERP_CUBIC_BSPLINE_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE_FAST, data_t>;
                    const auto kernel = noa::geometry::details::transform3D<uint32_t>(
                            interpolator_t(texture), output_accessor, matrices);
                    return cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                }
            }
        }
    }


    template<typename data_t>
    void launchTransformSymmetryTexture3D_(
            cudaTextureObject_t texture, InterpMode texture_interp_mode,
            data_t* output, dim4_t output_strides, dim4_t output_shape,
            float3_t shift, float33_t matrix, const geometry::Symmetry& symmetry,
            float3_t center, bool normalize, cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const dim_t count = symmetry.count();
        const float33_t* symmetry_matrices = symmetry.get();
        using unique_ptr = cuda::memory::PtrDevice<float33_t>::alloc_unique_t;
        unique_ptr d_matrices = cuda::memory::PtrDevice<float33_t>::alloc(count, stream);
        cuda::memory::copy(symmetry_matrices, d_matrices.get(), count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(count + 1) : 1;

        const auto iwise_shape = safe_cast<uint4_t>(output_shape);
        const auto output_accessor = AccessorRestrict<data_t, 4, uint32_t>(output, safe_cast<uint4_t>(output_strides));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_NEAREST, data_t>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                return;
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR, data_t>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                return;
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE, data_t>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                return;
            }
            case INTERP_CUBIC: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC, data_t>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                return;
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE, data_t>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                return;
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, data_t>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                return;
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE_FAST, data_t>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                return;
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE_FAST, data_t>;
                const auto kernel = noa::geometry::details::transformSymmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, shift, matrix, center,
                        d_matrices.get(), count, scaling);
                cuda::utils::iwise4D("geometry::transform3D", iwise_shape, kernel, stream);
                return;
            }
        }
    }

    template<typename data_t>
    void launchSymmetrize3D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                             data_t* output, dim4_t output_strides, dim4_t output_shape,
                             const geometry::Symmetry& symmetry, float3_t center, bool normalize,
                             cuda::Stream& stream) {
        // TODO Move symmetry matrices to constant memory?
        const dim_t count = symmetry.count();
        const float33_t* symmetry_matrices = symmetry.get();
        using unique_ptr = cuda::memory::PtrDevice<float33_t>::alloc_unique_t;
        unique_ptr d_matrices = cuda::memory::PtrDevice<float33_t>::alloc(count, stream);
        cuda::memory::copy(symmetry_matrices, d_matrices.get(), count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(count + 1) : 1;

        const auto iwise_shape = safe_cast<uint4_t>(output_shape);
        const auto output_accessor = AccessorRestrict<data_t, 4, uint32_t>(output, safe_cast<uint4_t>(output_strides));

        switch (texture_interp_mode) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_NEAREST, data_t>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR, data_t>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE, data_t>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC, data_t>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE, data_t>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, data_t>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE_FAST, data_t>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE_FAST, data_t>;
                const auto kernel = noa::geometry::details::symmetry3D<int32_t>(
                        interpolator_t(texture), output_accessor, center,
                        d_matrices.get(), count, scaling);
                return cuda::utils::iwise4D("geometry::symmetry3D", iwise_shape, kernel, stream);
            }
        }
    }

    // Updates the input and output shape to correctly broadcast the input.
    // Prefilter the input if needed.
    template<typename T>
    auto preprocess3D_(const shared_t<T[]>& input, dim4_t& input_strides, dim4_t& input_shape,
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
            if (input_shape[1] != output_shape[1] ||
                input_shape[2] != output_shape[2] ||
                input_shape[3] != output_shape[3]) {
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
        if constexpr (traits::is_float34_v<T> || traits::is_float44_v<T>) {
            return float34_t(v);
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
    void transform3D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const MAT& matrices, InterpMode interp_mode, BorderMode border_mode,
                     bool prefilter, Stream& stream) {
        NOA_ASSERT(input && all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

        auto [buffer, buffer_ptr, buffer_strides] = preprocess3D_(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                interp_mode, prefilter, stream);

        // Copy to texture and launch (per input batch):
        cuda::memory::PtrArray<T> array({1, input_shape[1], input_shape[2], input_shape[3]});
        cuda::memory::PtrTexture texture(array.get(), interp_mode, border_mode);
        for (dim_t i = 0; i < input_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_strides[0], buffer_strides,
                               array.get(), array.shape(), stream);
            launchTransformTexture3D_(
                    texture.get(), array.shape(), interp_mode, border_mode,
                    output.get() + i * output_strides[0], output_strides, output_shape,
                    matrixOrRawConstPtr(matrices, i), stream);
        }
        stream.attach(input, output, array.share(), texture.share());
        if constexpr (!traits::is_floatXX_v<MAT>)
            stream.attach(matrices);
    }

    template<typename data_t, typename matrix_t, typename>
    void transform3D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, dim4_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     const shared_t<data_t[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const matrix_t& matrices, Stream& stream) {
        NOA_ASSERT(array && texture && all(texture_shape > 0) && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(memory::PtrTexture::array(*texture) == array.get());

        launchTransformTexture3D_(
                *texture, texture_shape, texture_interp_mode, texture_border_mode,
                output.get(), output_strides, output_shape,
                matrixOrRawConstPtr(matrices), stream);

        if constexpr (traits::is_floatXX_v<matrix_t>)
            stream.attach(array, texture, output);
        else
            stream.attach(array, texture, output, matrices);
    }

    template<typename T, typename>
    void transform3D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream) {
        NOA_ASSERT(input && all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

        auto [buffer, buffer_ptr, buffer_strides] = preprocess3D_(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                interp_mode, prefilter, stream);

        // Copy to texture and launch (per input batch):
        cuda::memory::PtrArray<T> array({1, input_shape[1], input_shape[2], input_shape[3]});
        cuda::memory::PtrTexture texture(array.get(), interp_mode, BORDER_ZERO);
        for (dim_t i = 0; i < input_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_strides[0], buffer_strides,
                               array.get(), array.shape(), stream);
            launchTransformSymmetryTexture3D_(
                    texture.get(), interp_mode,
                    output.get() + i * output_strides[0], output_strides, output_shape,
                    shift, matrix, symmetry, center, normalize, stream);
        }
        stream.attach(input, output, symmetry.share(), array.share(), texture.share());
    }

    template<typename T, typename>
    void transform3D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture,
                     InterpMode texture_interp_mode, dim4_t texture_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     bool normalize, Stream& stream) {
        NOA_ASSERT(array && texture && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(texture_shape[0] == 1);
        (void) texture_shape;

        launchTransformSymmetryTexture3D_(
                *texture, texture_interp_mode,
                output.get(), output_strides, output_shape,
                shift, matrix, symmetry, center, normalize, stream);
        stream.attach(array, texture, output, symmetry.share());
    }

    template<typename T, typename>
    void symmetrize3D(const shared_t<T[]>& input, dim4_t input_strides,
                      const shared_t<T[]>& output, dim4_t output_strides,
                      dim4_t shape, const Symmetry& symmetry, float3_t center,
                      InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream) {
        NOA_ASSERT(all(shape > 0) && input);
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(shape[1] > 1);

        if (!symmetry.count()) {
            if (input != output)
                memory::copy(input, input_strides, output, output_strides, shape, stream);
            return;
        }

        auto [buffer, buffer_ptr, buffer_strides] = preprocess3D_(
                input, input_strides, shape,
                output, output_strides, shape,
                interp_mode, prefilter, stream);

        // Copy to texture and launch (per input batch):
        memory::PtrArray<T> array({1, shape[1], shape[2], shape[3]});
        memory::PtrTexture texture(array.get(), interp_mode, BORDER_ZERO);
        for (dim_t i = 0; i < shape[0]; ++i) {
            memory::copy(buffer_ptr + i * buffer_strides[0], buffer_strides, array.get(), array.shape(), stream);
            launchSymmetrize3D_(
                    texture.get(), interp_mode,
                    output.get() + i * output_strides[0], output_strides, shape,
                    symmetry, center, normalize, stream);
        }
        stream.attach(input, output, symmetry.share(), array.share(), texture.share());
    }

    template<typename T, typename>
    void symmetrize3D(const shared_t<cudaArray>& array,
                      const shared_t<cudaTextureObject_t>& texture,
                      InterpMode texture_interp_mode, dim4_t texture_shape,
                      const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
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

    #define NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, M)                                                                                                                                                                           \
    template void transform3D<T, shared_t<M[]>, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<M[]>&, InterpMode, BorderMode, bool, Stream&);                                 \
    template void transform3D<T, M, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, InterpMode, BorderMode, bool, Stream&);                                                         \
    template void transform3D<T, shared_t<M[]>, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, dim4_t, InterpMode, BorderMode, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<M[]>&, Stream&);   \
    template void transform3D<T, M, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, dim4_t, InterpMode, BorderMode, const shared_t<T[]>&, dim4_t, dim4_t, const M&, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_SYM_(T)                                                                                                                                                           \
    template void transform3D<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float33_t, const Symmetry&, float3_t, InterpMode, bool, bool, Stream&);    \
    template void transform3D<T, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float33_t, const Symmetry&, float3_t, bool, Stream&)

    #define NOA_INSTANTIATE_SYM_(T)                                                                                                                                         \
    template void symmetrize3D<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const Symmetry&, float3_t, InterpMode, bool, bool, Stream&);    \
    template void symmetrize3D<T, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const Symmetry&, float3_t, bool, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_3D_(T)            \
    NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, float34_t);  \
    NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, float44_t);  \
    NOA_INSTANTIATE_TRANSFORM_SYM_(T);                  \
    NOA_INSTANTIATE_SYM_(T)

    NOA_INSTANTIATE_TRANSFORM_3D_(float);
    NOA_INSTANTIATE_TRANSFORM_3D_(cfloat_t);
}
