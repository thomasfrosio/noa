#include "noa/common/Assert.h"
#include "noa/common/geometry/details/PolarTransform.h"

#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/geometry/Polar.h"
#include "noa/gpu/cuda/geometry/Prefilter.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace {
    using namespace ::noa;

    template<bool LAYERED, typename Value>
    void launchCartesianPolarTexture_(cudaTextureObject_t cartesian,
                                      dim4_t cartesian_shape,
                                      InterpMode cartesian_interp,
                                      Value* polar, dim4_t polar_strides, dim4_t polar_shape,
                                      float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                                      bool log, cuda::Stream& stream) {
        NOA_ASSERT(polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[1] == 1);
        NOA_ASSERT((LAYERED && cartesian_shape[0] == polar_shape[0]) ||
                   (!LAYERED && cartesian_shape[0] == 1));
        (void) cartesian_shape;

        const auto iwise_shape = safe_cast<uint3_t>(dim3_t(polar_shape[0], polar_shape[2], polar_shape[3]));
        const auto polar_accessor = AccessorRestrict<Value, 3, uint32_t>(
                polar, safe_cast<uint3_t>(dim3_t{polar_strides[0], polar_strides[2], polar_strides[3]}));

        switch (cartesian_interp) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
        }
    }

    template<bool LAYERED, typename Value>
    void launchPolarCartesianTexture_(cudaTextureObject_t polar, InterpMode polar_interp, dim4_t polar_shape,
                                      Value* cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                                      float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                                      bool log, cuda::Stream& stream) {
        NOA_ASSERT(polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[1] == 1);
        NOA_ASSERT((LAYERED && polar_shape[0] == cartesian_shape[0]) ||
                   (!LAYERED && polar_shape[0] == 1));

        const auto iwise_shape = safe_cast<uint3_t>(dim3_t(cartesian_shape[0], cartesian_shape[2], cartesian_shape[3]));
        const auto cartesian_accessor = AccessorRestrict<Value, 3, uint32_t>(
                cartesian, safe_cast<uint3_t>(dim3_t{cartesian_strides[0], cartesian_strides[2], cartesian_strides[3]}));

        switch (polar_interp) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
        }
    }

    // Updates the input and output shape to correctly broadcast the input.
    // Prefilter the input if needed.
    template<typename Value>
    auto preprocess2D_(const shared_t<Value[]>& input, dim4_t& input_strides, dim4_t& input_shape,
                       const shared_t<Value[]>& output, dim4_t output_strides, dim4_t& output_shape,
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

            // If the input is batched, then we need to ensure that the processing loop will compute
            // one batch at a time, for both the input and the output. Otherwise, the processing loop
            // should run once, processing all output batches at the same time using the unique input batch.
            if (input_shape[0] > 1)
                output_shape[0] = 1;
        }

        using unique_ptr_t = typename cuda::memory::PtrDevice<Value>::alloc_unique_t;
        unique_ptr_t buffer;
        Value* buffer_ptr;
        dim4_t buffer_strides;
        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            if (input_shape[2] != output_shape[2] || input_shape[3] != output_shape[3]) {
                buffer = cuda::memory::PtrDevice<Value>::alloc(input_shape.elements(), stream);
                buffer_ptr = buffer.get();
                buffer_strides = input_shape.strides();
            } else {
                // Whether input is batched or not, since we copy
                // to the CUDA array, we can use the output as buffer.
                buffer_ptr = output.get();
                buffer_strides = output_strides;
            }
            cuda::geometry::bspline::prefilter(
                    input.get(), input_strides,
                    buffer_ptr, buffer_strides,
                    input_shape, stream);
        } else {
            buffer_ptr = input.get();
            buffer_strides = input_strides;
        }

        return std::tuple<unique_ptr_t, const Value*, dim4_t>(std::move(buffer), buffer_ptr, buffer_strides);
    }
}

namespace noa::cuda::geometry {
    template<typename Value, typename>
    void cartesian2polar(const shared_t<Value[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         const shared_t<Value[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp_mode, bool prefilter, Stream& stream) {
        NOA_ASSERT(cartesian && polar && all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(polar.get(), stream.device());
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);

        // Prepare the input array:
        auto [buffer, buffer_ptr, buffer_strides] = preprocess2D_(
                cartesian, cartesian_strides, cartesian_shape,
                polar, polar_strides, polar_shape,
                interp_mode, prefilter, stream);

        // Copy to texture and launch (per input batch):
        memory::PtrArray<Value> array({1, 1, cartesian_shape[2], cartesian_shape[3]});
        memory::PtrTexture texture(array.get(), interp_mode, BORDER_ZERO);
        for (dim_t i = 0; i < cartesian_shape[0]; ++i) {
            memory::copy(buffer_ptr + i * buffer_strides[0], buffer_strides, array.get(), array.shape(), stream);
            launchCartesianPolarTexture_<false>(
                    texture.get(), cartesian_shape, interp_mode,
                    polar.get() + i * polar_strides[0], polar_strides, polar_shape,
                    cartesian_center, radius_range, angle_range, log, stream);
        }
        stream.attach(cartesian, polar, array.share(), texture.share());
    }

    template<typename Value, typename>
    void cartesian2polar(const shared_t<cudaArray>& array,
                         const shared_t<cudaTextureObject_t>& cartesian,
                         InterpMode cartesian_interp, dim4_t cartesian_shape,
                         const shared_t<Value[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, Stream& stream) {
        NOA_ASSERT(array && cartesian && all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(polar.get(), stream.device());
        const bool is_layered = memory::PtrArray<Value>::isLayered(array.get());
        NOA_ASSERT(memory::PtrTexture::array(*cartesian) == array.get());

        if (is_layered) {
            launchCartesianPolarTexture_<true>(
                    *cartesian, cartesian_shape, cartesian_interp,
                    polar.get(), polar_strides, polar_shape,
                    cartesian_center, radius_range, angle_range, log, stream);
        } else {
            launchCartesianPolarTexture_<false>(
                    *cartesian, cartesian_shape, cartesian_interp,
                    polar.get(), polar_strides, polar_shape,
                    cartesian_center, radius_range, angle_range, log, stream);
        }

        stream.attach(array, cartesian, polar);
    }

    template<typename Value, typename>
    void polar2cartesian(const shared_t<Value[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         const shared_t<Value[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp_mode, bool prefilter, Stream& stream) {
        NOA_ASSERT(cartesian && polar && all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(cartesian.get(), stream.device());
        NOA_ASSERT(polar_shape[0] == 1 || polar_shape[0] == cartesian_shape[0]);
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);

        // Prepare the input array:
        auto [buffer, buffer_ptr, buffer_strides] = preprocess2D_(
                cartesian, cartesian_strides, cartesian_shape,
                polar, polar_strides, polar_shape,
                interp_mode, prefilter, stream);

        // Copy to texture and launch (per input batch):
        memory::PtrArray<Value> array({1, 1, polar_shape[2], polar_shape[3]});
        memory::PtrTexture texture(array.get(), interp_mode, BORDER_ZERO);
        for (dim_t i = 0; i < polar_shape[0]; ++i) {
            memory::copy(buffer_ptr + i * buffer_strides[0], buffer_strides, array.get(), array.shape(), stream);
            launchPolarCartesianTexture_<false>(
                    texture.get(), interp_mode, polar_shape,
                    cartesian.get() + i * cartesian_strides[0], cartesian_strides, cartesian_shape,
                    cartesian_center, radius_range, angle_range, log, stream);
        }
        stream.attach(polar, cartesian, array.share(), texture.share());
    }



    template<typename Value, typename>
    void polar2cartesian(const shared_t<cudaArray>& array,
                         const shared_t<cudaTextureObject_t>& polar,
                         InterpMode polar_interp, dim4_t polar_shape,
                         const shared_t<Value[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, Stream& stream) {
        NOA_ASSERT(array && cartesian && all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(cartesian.get(), stream.device());
        const bool is_layered = memory::PtrArray<Value>::isLayered(array.get());
        NOA_ASSERT(memory::PtrTexture::array(*polar) == array.get());

        if (is_layered) {
            launchPolarCartesianTexture_<true>(
                    *polar, polar_interp, polar_shape,
                    cartesian.get(), cartesian_strides, cartesian_shape,
                    cartesian_center, radius_range, angle_range, log, stream);
        } else {
            launchPolarCartesianTexture_<false>(
                    *polar, polar_interp, polar_shape,
                    cartesian.get(), cartesian_strides, cartesian_shape,
                    cartesian_center, radius_range, angle_range, log, stream);
        }

        stream.attach(array, cartesian, polar);
    }

    #define INSTANTIATE_POLAR(T) \
    template void cartesian2polar<T,void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, float2_t, bool, InterpMode, bool, Stream&);                       \
    template void polar2cartesian<T,void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, float2_t, bool, InterpMode, bool, Stream&);                       \
    template void cartesian2polar<T,void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, float2_t, bool, Stream&); \
    template void polar2cartesian<T,void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, float2_t, bool, Stream&)

    INSTANTIATE_POLAR(float);
    INSTANTIATE_POLAR(cfloat_t);
}
