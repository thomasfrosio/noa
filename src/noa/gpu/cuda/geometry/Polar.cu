#include "noa/common/Assert.h"
#include "noa/common/geometry/Interpolator.h"
#include "noa/common/geometry/details/PolarTransform.h"

#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/geometry/Polar.h"
#include "noa/gpu/cuda/geometry/Prefilter.h"
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

        const auto iwise_shape = safe_cast<uint3_t>(dim3_t(polar_shape[0], polar_shape[2], polar_shape[3]));
        const auto polar_accessor = AccessorRestrict<Value, 3, uint32_t>(
                polar, safe_cast<uint3_t>(dim3_t{polar_strides[0], polar_strides[2], polar_strides[3]}));

        switch (cartesian_interp) {
            case INTERP_NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_COSINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_CUBIC, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), polar_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::cartesian2polar", iwise_shape, kernel, stream);
            }
        }
    }

    template<bool LAYERED, typename Value>
    void launchPolarCartesianTexture_(cudaTextureObject_t polar, InterpMode polar_interp, dim4_t polar_shape,
                                      Value* cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                                      float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                                      bool log, cuda::Stream& stream) {

        const auto iwise_shape = safe_cast<uint3_t>(dim3_t(cartesian_shape[0], cartesian_shape[2], cartesian_shape[3]));
        const auto cartesian_accessor = AccessorRestrict<Value, 3, uint32_t>(
                cartesian, safe_cast<uint3_t>(dim3_t{cartesian_strides[0], cartesian_strides[2], cartesian_strides[3]}));

        switch (polar_interp) {
            case INTERP_NEAREST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_COSINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_CUBIC, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC_BSPLINE_FAST: {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::details::polar2cartesian<uint32_t>(
                        interpolator_t(polar), polar_shape, cartesian_accessor,
                        cartesian_center, radius_range, angle_range, log);
                return noa::cuda::utils::iwise3D("geometry::polar2cartesian", iwise_shape, kernel, stream);
            }
        }
    }
}

namespace noa::cuda::geometry {
    template<typename Value, typename>
    void cartesian2polar(const shared_t<Value[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         const shared_t<Value[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp_mode, bool prefilter, Stream& stream) {
        NOA_ASSERT(cartesian.get() != polar.get() && all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(cartesian.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(polar.get(), stream.device());
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);

        // Broadcast the input to every output batch.
        if (cartesian_shape[0] == 1)
            cartesian_strides[0] = 0;
        else if (cartesian_strides[0] == 0)
            cartesian_shape[0] = 1;

        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            noa::cuda::geometry::bspline::prefilter(
                    cartesian.get(), cartesian_strides,
                    cartesian.get(), cartesian_strides,
                    cartesian_shape, stream);
        }

        const auto cartesian_shape_2d = int2_t{cartesian_shape[2], cartesian_shape[3]};
        const auto polar_shape_2d = int3_t{polar_shape[0], polar_shape[2], polar_shape[3]};
        const auto cartesian_strides_2d = safe_cast<uint3_t>(dim3_t{cartesian_strides[0], cartesian_strides[2], cartesian_strides[3]});
        const auto polar_strides_2d = safe_cast<uint3_t>(dim3_t{polar_strides[0], polar_strides[2], polar_strides[3]});
        const auto input_accessor = AccessorRestrict<const Value, 3, uint32_t>(cartesian.get(), cartesian_strides_2d);
        const auto output_accessor = AccessorRestrict<Value, 3, uint32_t>(polar.get(), polar_strides_2d);

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::geometry::details::cartesian2polar<int32_t>(
                        interpolator, output_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                noa::cuda::utils::iwise3D("geometry::cartesian2polar", polar_shape_2d, kernel, stream);
                break;
            }
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::geometry::details::cartesian2polar<int32_t>(
                        interpolator, output_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                noa::cuda::utils::iwise3D("geometry::cartesian2polar", polar_shape_2d, kernel, stream);
                break;
            }
            case INTERP_COSINE_FAST:
            case INTERP_COSINE: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::geometry::details::cartesian2polar<int32_t>(
                        interpolator, output_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                noa::cuda::utils::iwise3D("geometry::cartesian2polar", polar_shape_2d, kernel, stream);
                break;
            }
            case INTERP_CUBIC: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::geometry::details::cartesian2polar<int32_t>(
                        interpolator, output_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                noa::cuda::utils::iwise3D("geometry::cartesian2polar", polar_shape_2d, kernel, stream);
                break;
            }
            case INTERP_CUBIC_BSPLINE_FAST:
            case INTERP_CUBIC_BSPLINE: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC_BSPLINE>(
                        input_accessor, cartesian_shape_2d);
                const auto kernel = noa::geometry::details::cartesian2polar<int32_t>(
                        interpolator, output_accessor, polar_shape,
                        cartesian_center, radius_range, angle_range, log);
                noa::cuda::utils::iwise3D("geometry::cartesian2polar", polar_shape_2d, kernel, stream);
                break;
            }
        }
        stream.attach(cartesian, polar);
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
        NOA_ASSERT(memory::PtrTexture::array(*cartesian) == array.get());
        NOA_ASSERT(polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[1] == 1);
        const bool is_layered = memory::PtrArray<Value>::isLayered(array.get());

        if (is_layered) {
            NOA_ASSERT(cartesian_shape[0] == polar_shape[0]);
            launchCartesianPolarTexture_<true>(
                    *cartesian, cartesian_shape, cartesian_interp,
                    polar.get(), polar_strides, polar_shape,
                    cartesian_center, radius_range, angle_range, log, stream);
        } else {
            NOA_ASSERT(cartesian_shape[0] == 1);
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

        // Broadcast the input to every output batch.
        if (polar_shape[0] == 1)
            polar_strides[0] = 0;
        else if (polar_strides[0] == 0)
            polar_shape[0] = 1;

        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            noa::cuda::geometry::bspline::prefilter(
                    polar.get(), polar_strides,
                    polar.get(), polar_strides,
                    polar_shape, stream);
        }

        const auto polar_shape_2d = int2_t{polar_shape[2], polar_shape[3]};
        const auto cartesian_shape_2d = int3_t{cartesian_shape[0], cartesian_shape[2], cartesian_shape[3]};
        const auto polar_strides_2d = safe_cast<uint3_t>(dim3_t{polar_strides[0], polar_strides[2], polar_strides[3]});
        const auto cartesian_strides_2d = safe_cast<uint3_t>(dim3_t{cartesian_strides[0], cartesian_strides[2], cartesian_strides[3]});
        const auto input_accessor = AccessorRestrict<const Value, 3, int32_t>(polar.get(), polar_strides_2d);
        const auto output_accessor = AccessorRestrict<Value, 3, int32_t>(cartesian.get(), cartesian_strides_2d);

        switch (interp_mode) {
            case INTERP_NEAREST: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(
                        input_accessor, polar_shape_2d);
                const auto kernel = noa::geometry::details::polar2cartesian<int32_t>(
                        interpolator, polar_shape, output_accessor,
                        cartesian_center, radius_range, angle_range, log);
                noa::cuda::utils::iwise3D("geometry::polar2cartesian", cartesian_shape_2d, kernel, stream);
                break;
            }
            case INTERP_LINEAR_FAST:
            case INTERP_LINEAR: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(
                        input_accessor, polar_shape_2d);
                const auto kernel = noa::geometry::details::polar2cartesian<int32_t>(
                        interpolator, polar_shape, output_accessor,
                        cartesian_center, radius_range, angle_range, log);
                noa::cuda::utils::iwise3D("geometry::polar2cartesian", cartesian_shape_2d, kernel, stream);
                break;
            }
            case INTERP_COSINE_FAST:
            case INTERP_COSINE: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(
                        input_accessor, polar_shape_2d);
                const auto kernel = noa::geometry::details::polar2cartesian<int32_t>(
                        interpolator, polar_shape, output_accessor,
                        cartesian_center, radius_range, angle_range, log);
                noa::cuda::utils::iwise3D("geometry::polar2cartesian", cartesian_shape_2d, kernel, stream);
                break;
            }
            case INTERP_CUBIC: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC>(
                        input_accessor, polar_shape_2d);
                const auto kernel = noa::geometry::details::polar2cartesian<int32_t>(
                        interpolator, polar_shape, output_accessor,
                        cartesian_center, radius_range, angle_range, log);
                noa::cuda::utils::iwise3D("geometry::polar2cartesian", cartesian_shape_2d, kernel, stream);
                break;
            }
            case INTERP_CUBIC_BSPLINE_FAST:
            case INTERP_CUBIC_BSPLINE: {
                const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_CUBIC_BSPLINE>(
                        input_accessor, polar_shape_2d);
                const auto kernel = noa::geometry::details::polar2cartesian<int32_t>(
                        interpolator, polar_shape, output_accessor,
                        cartesian_center, radius_range, angle_range, log);
                noa::cuda::utils::iwise3D("geometry::polar2cartesian", cartesian_shape_2d, kernel, stream);
                break;
            }
        }
        stream.attach(polar, cartesian);
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
        NOA_ASSERT(polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[1] == 1);

        if (is_layered) {
            NOA_ASSERT(polar_shape[0] == cartesian_shape[0]);
            launchPolarCartesianTexture_<true>(
                    *polar, polar_interp, polar_shape,
                    cartesian.get(), cartesian_strides, cartesian_shape,
                    cartesian_center, radius_range, angle_range, log, stream);
        } else {
            NOA_ASSERT(polar_shape[0] == 1);
            launchPolarCartesianTexture_<false>(
                    *polar, polar_interp, polar_shape,
                    cartesian.get(), cartesian_strides, cartesian_shape,
                    cartesian_center, radius_range, angle_range, log, stream);
        }

        stream.attach(array, cartesian, polar);
    }

    #define INSTANTIATE_POLAR(T)                                        \
    template void cartesian2polar<T,void>(                              \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        float2_t, float2_t, float2_t, bool, InterpMode, bool, Stream&); \
    template void polar2cartesian<T,void>(                              \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        float2_t, float2_t, float2_t, bool, InterpMode, bool, Stream&)

    #define INSTANTIATE_POLAR_TEXTURE(T)                            \
    template void cartesian2polar<T,void>(                          \
        const shared_t<cudaArray>&,                                 \
        const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t,   \
        const shared_t<T[]>&, dim4_t, dim4_t,                       \
        float2_t, float2_t, float2_t, bool, Stream&);               \
    template void polar2cartesian<T,void>(                          \
        const shared_t<cudaArray>&,                                 \
        const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t,   \
        const shared_t<T[]>&, dim4_t, dim4_t,                       \
        float2_t, float2_t, float2_t, bool, Stream&)

    INSTANTIATE_POLAR(float);
    INSTANTIATE_POLAR(double);
    INSTANTIATE_POLAR(cfloat_t);
    INSTANTIATE_POLAR(cdouble_t);
    INSTANTIATE_POLAR_TEXTURE(float);
    INSTANTIATE_POLAR_TEXTURE(cfloat_t);
}
