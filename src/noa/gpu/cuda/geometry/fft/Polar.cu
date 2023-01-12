#include "noa/common/Assert.h"
#include "noa/common/geometry/Interpolator.h"
#include "noa/common/geometry/details/PolarTransformFourier.h"

#include "noa/gpu/cuda/geometry/fft/Polar.h"
#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace {
    using namespace ::noa;

    template<bool LAYERED, typename Value>
    void launchCartesian2Polar_(cudaTextureObject_t cartesian, InterpMode cartesian_interp, dim4_t cartesian_shape,
                                Value* polar, dim4_t polar_strides, dim4_t polar_shape,
                                float2_t frequency_range, float2_t angle_range, bool log, cuda::Stream& stream) {
        NOA_ASSERT(polar_shape[1] == 1);
        NOA_ASSERT((LAYERED && polar_shape[0] == cartesian_shape[0]) ||
                   (!LAYERED && cartesian_shape[0] == 1));

        const auto iwise_shape = safe_cast<uint3_t>(dim3_t(polar_shape[0], polar_shape[2], polar_shape[3]));
        const auto polar_accessor = AccessorRestrict<Value, 3, uint32_t>(
                polar, safe_cast<uint3_t>(dim3_t{polar_strides[0], polar_strides[2], polar_strides[3]}));

        switch (cartesian_interp) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::fft::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::fft::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::fft::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::fft::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, Value, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::cartesian2polar<uint32_t>(
                        interpolator_t(cartesian), cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                return cuda::utils::iwise3D("geometry::fft::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC:
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST:
                NOA_THROW_FUNC("cartesian2polar", "{} is not supported", cartesian_interp);
        }
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename Value, typename>
    void cartesian2polar(const shared_t<Value[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         const shared_t<Value[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(cartesian && all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(polar.get(), stream.device());
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        // Broadcast the input to every output batch.
        if (cartesian_shape[0] == 1)
            cartesian_strides[0] = 0;
        else if (cartesian_strides[0] == 0)
            cartesian_shape[0] = 1;

        const auto cartesian_shape_2d = int2_t{cartesian_shape[2], cartesian_shape[3]};
        const auto polar_shape_2d = int3_t(polar_shape[0], polar_shape[2], polar_shape[3]);
        const auto cartesian_strides_2d = safe_cast<uint3_t>(dim3_t{cartesian_strides[0], cartesian_strides[2], cartesian_strides[3]});
        const auto polar_strides_2d = safe_cast<uint3_t>(dim3_t{polar_strides[0], polar_strides[2], polar_strides[3]});
        const auto cartesian_accessor = AccessorRestrict<const Value, 3, uint32_t>(cartesian.get(),cartesian_strides_2d);
        const auto polar_accessor = AccessorRestrict<Value, 3, uint32_t>(polar.get(), polar_strides_2d);

            switch (interp_mode) {
                case INTERP_NEAREST: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_NEAREST>(
                            cartesian_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::fft::details::cartesian2polar<int32_t>(
                            interpolator, cartesian_shape, polar_accessor, polar_shape,
                            frequency_range, angle_range, log);
                    return noa::cuda::utils::iwise3D("geometry::fft::cartesian2polar", polar_shape_2d, kernel, stream);
                }
                case INTERP_LINEAR: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(
                            cartesian_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::fft::details::cartesian2polar<int32_t>(
                            interpolator, cartesian_shape, polar_accessor, polar_shape,
                            frequency_range, angle_range, log);
                    return noa::cuda::utils::iwise3D("geometry::fft::cartesian2polar", polar_shape_2d, kernel, stream);
                }
                case INTERP_COSINE: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE>(
                            cartesian_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::fft::details::cartesian2polar<int32_t>(
                            interpolator, cartesian_shape, polar_accessor, polar_shape,
                            frequency_range, angle_range, log);
                    return noa::cuda::utils::iwise3D("geometry::fft::cartesian2polar", polar_shape_2d, kernel, stream);
                }
                case INTERP_LINEAR_FAST: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR_FAST>(
                            cartesian_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::fft::details::cartesian2polar<int32_t>(
                            interpolator, cartesian_shape, polar_accessor, polar_shape,
                            frequency_range, angle_range, log);
                    return noa::cuda::utils::iwise3D("geometry::fft::cartesian2polar", polar_shape_2d, kernel, stream);
                }
                case INTERP_COSINE_FAST: {
                    const auto interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_COSINE_FAST>(
                            cartesian_accessor, cartesian_shape_2d);
                    const auto kernel = noa::geometry::fft::details::cartesian2polar<int32_t>(
                            interpolator, cartesian_shape, polar_accessor, polar_shape,
                            frequency_range, angle_range, log);
                    return noa::cuda::utils::iwise3D("geometry::fft::cartesian2polar", polar_shape_2d, kernel, stream);
                }
                case INTERP_CUBIC:
                case INTERP_CUBIC_BSPLINE:
                case INTERP_CUBIC_BSPLINE_FAST:
                    NOA_THROW("{} is not supported", interp_mode);
            }
        stream.attach(cartesian, polar);
    }

    template<typename Value, typename>
    void cartesian2polar(const shared_t<cudaArray>& array,
                         const shared_t<cudaTextureObject_t>& cartesian,
                         InterpMode cartesian_interp, dim4_t cartesian_shape,
                         const shared_t<Value[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, Stream& stream) {
        NOA_ASSERT(array && cartesian && all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(polar.get(), stream.device());

        const bool is_layered = memory::PtrArray<Value>::isLayered(array.get());
        NOA_ASSERT(memory::PtrTexture::array(*cartesian) == array.get());

        if (is_layered) {
            launchCartesian2Polar_<true>(
                    *cartesian, cartesian_interp, cartesian_shape,
                    polar.get(), polar_strides, polar_shape,
                    frequency_range, angle_range, log, stream);
        } else {
            launchCartesian2Polar_<false>(
                    *cartesian, cartesian_interp, cartesian_shape,
                    polar.get(), polar_strides, polar_shape,
                    frequency_range, angle_range, log, stream);
        }
        stream.attach(array, cartesian, polar);
    }

    #define INSTANTIATE_POLAR(T) \
    template void cartesian2polar<Remap::HC2FC, T, void>( \
        const shared_t<T[]>&, dim4_t, dim4_t,             \
        const shared_t<T[]>&, dim4_t, dim4_t,             \
        float2_t, float2_t, bool, InterpMode, Stream&)

    #define INSTANTIATE_POLAR_TEXTURE(T) \
    template void cartesian2polar<T, void>(               \
        const shared_t<cudaArray>&,                       \
        const shared_t<cudaTextureObject_t>&, InterpMode, \
        dim4_t, const shared_t<T[]>&, dim4_t, dim4_t,     \
        float2_t, float2_t, bool, Stream&)

    INSTANTIATE_POLAR(float);
    INSTANTIATE_POLAR(cfloat_t);
    INSTANTIATE_POLAR(double);
    INSTANTIATE_POLAR(cdouble_t);
    INSTANTIATE_POLAR_TEXTURE(float);
    INSTANTIATE_POLAR_TEXTURE(cfloat_t);
}
