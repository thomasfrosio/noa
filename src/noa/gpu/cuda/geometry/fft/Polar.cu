#include "noa/common/Assert.h"
#include "noa/common/geometry/details/PolarTransformations.h"

#include "noa/gpu/cuda/geometry/fft/Polar.h"
#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace {
    using namespace ::noa;

    template<bool LAYERED, typename data_t>
    void launchCartesian2Polar_(cudaTextureObject_t cartesian, InterpMode cartesian_interp, dim4_t cartesian_shape,
                                data_t* polar, dim4_t polar_strides, dim4_t polar_shape,
                                float2_t frequency_range, float2_t angle_range, bool log, cuda::Stream& stream) {
        NOA_ASSERT(polar_shape[1] == 1);
        NOA_ASSERT((LAYERED && polar_shape[0] == cartesian_shape[0]) ||
                   (!LAYERED && cartesian_shape[0] == 1));

        const auto iwise_shape = safe_cast<uint3_t>(dim3_t(polar_shape[0], polar_shape[2], polar_shape[3]));
        const auto polar_accessor = AccessorRestrict<data_t, 3, uint32_t>(
                polar, safe_cast<uint3_t>(dim3_t{polar_strides[0], polar_strides[2], polar_strides[3]}));

        switch (cartesian_interp) {
            case INTERP_NEAREST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::cartesian2polar<LAYERED, uint32_t>(
                        interpolator_t(cartesian), cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                cuda::utils::iwise3D("geometry::fft::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::cartesian2polar<LAYERED, uint32_t>(
                        interpolator_t(cartesian), cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                cuda::utils::iwise3D("geometry::fft::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::cartesian2polar<LAYERED, uint32_t>(
                        interpolator_t(cartesian), cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                cuda::utils::iwise3D("geometry::fft::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_LINEAR_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::cartesian2polar<LAYERED, uint32_t>(
                        interpolator_t(cartesian), cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                cuda::utils::iwise3D("geometry::fft::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_COSINE_FAST: {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, data_t, false, LAYERED>;
                const auto kernel = noa::geometry::fft::details::cartesian2polar<LAYERED, uint32_t>(
                        interpolator_t(cartesian), cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, angle_range, log);
                cuda::utils::iwise3D("geometry::fft::cartesian2polar", iwise_shape, kernel, stream);
            }
            case INTERP_CUBIC:
            case INTERP_CUBIC_BSPLINE:
            case INTERP_CUBIC_BSPLINE_FAST:
                NOA_THROW_FUNC("cartesian2polar", "{} is not supported", cartesian_interp);
        }
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename T, typename>
    void cartesian2polar(const shared_t<T[]>& cartesian, dim4_t cartesian_strides, dim4_t cartesian_shape,
                         const shared_t<T[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, InterpMode interp_mode, Stream& stream) {
        NOA_ASSERT(cartesian && all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(polar.get(), stream.device());
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        // Broadcast input if it is not batched:
        if (cartesian_strides[0] == 0)
            cartesian_shape[0] = 1;
        else if (cartesian_shape[0] == 1)
            cartesian_strides[0] = 0;
        const dim4_t output_shape{cartesian_shape[0] > 1 ? 1 : polar_shape[0],
                                  polar_shape[1], polar_shape[2], polar_shape[3]};

        // Copy to texture and launch (per input batch):
        memory::PtrArray<T> array(dim4_t{1, 1, cartesian_shape[2], cartesian_shape[3] / 2 + 1});
        memory::PtrTexture texture(array.get(), interp_mode, BORDER_ZERO);
        for (dim_t i = 0; i < cartesian_shape[0]; ++i) {
            memory::copy(cartesian.get() + i * cartesian_strides[0], cartesian_strides,
                         array.get(), array.shape(), stream);
            launchCartesian2Polar_<false>(texture.get(), interp_mode, cartesian_shape,
                    polar.get() + i * polar_strides[0], polar_strides, output_shape,
                    frequency_range, angle_range, log, stream);
        }
        stream.attach(cartesian, polar, array.share(), texture.share());
    }

    template<typename data_t, typename>
    void cartesian2polar(const shared_t<cudaArray>& array,
                         const shared_t<cudaTextureObject_t>& cartesian,
                         InterpMode cartesian_interp, dim4_t cartesian_shape,
                         const shared_t<data_t[]>& polar, dim4_t polar_strides, dim4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, Stream& stream) {
        NOA_ASSERT(array && cartesian && all(cartesian_shape > 0) && all(polar_shape > 0));
        NOA_ASSERT_DEVICE_PTR(polar.get(), stream.device());

        const bool is_layered = memory::PtrArray<data_t>::isLayered(array.get());
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
    template void cartesian2polar<Remap::HC2FC, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, bool, InterpMode, Stream&); \
    template void cartesian2polar<T, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float2_t, bool, Stream&)

    INSTANTIATE_POLAR(float);
    INSTANTIATE_POLAR(cfloat_t);
}
