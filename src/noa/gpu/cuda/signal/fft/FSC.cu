#include "noa/common/signal/details/FSC.h"

#include "noa/gpu/cuda/signal/fft/FSC.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace noa::cuda::signal::fft {
    template<noa::fft::Remap REMAP, typename Real, typename>
    void isotropicFSC(const shared_t<Complex<Real>[]>& lhs, dim4_t lhs_strides,
                      const shared_t<Complex<Real>[]>& rhs, dim4_t rhs_strides,
                      const shared_t<Real[]>& fsc,
                      dim4_t shape, cuda::Stream& stream) {
        NOA_ASSERT(lhs && rhs && fsc);

        const auto s_shape = safe_cast<int4_t>(shape);
        const auto s_shape_3d = int3_t(s_shape.get(1));
        const auto batches = s_shape[0];
        const auto shell_count = noa::math::min(s_shape_3d) / 2 + 1;

        const auto denominator_lhs = noa::cuda::memory::PtrDevice<Real>(batches * shell_count, stream); // allocates
        const auto denominator_rhs = noa::cuda::memory::PtrDevice<Real>(batches * shell_count, stream); // allocates

        const auto fsc_reduction = noa::signal::fft::details::isotropicFSC<REMAP>(
                lhs.get(), safe_cast<uint4_t>(lhs_strides),
                rhs.get(), safe_cast<uint4_t>(rhs_strides),
                fsc.get(), denominator_lhs.get(), denominator_rhs.get(),
                s_shape_3d, shell_count);

        noa::cuda::utils::iwise4D("signal::fft::isotropicFFT", s_shape, fsc_reduction, stream);

        using iwise2D_config = noa::cuda::utils::IwiseStaticConfig<noa::cuda::Limits::WARP_SIZE, 1>;
        noa::cuda::utils::iwise2D<iwise2D_config>(
                "signal::fft::isotropicFFT", int2_t{batches, shell_count}, fsc_reduction, stream);

        stream.attach(lhs, rhs, fsc);
    }

    template<noa::fft::Remap REMAP, typename Real, typename>
    void anisotropicFSC(const shared_t<Complex<Real>[]>& lhs, dim4_t lhs_strides,
                        const shared_t<Complex<Real>[]>& rhs, dim4_t rhs_strides,
                        const shared_t<Real[]>& fsc,
                        dim4_t shape,
                        const shared_t<float3_t[]>& normalized_cone_directions,
                        dim_t cone_count, float cone_aperture,
                        cuda::Stream& stream) {

        const auto s_shape = safe_cast<int4_t>(shape);
        const auto s_shape_3d = int3_t(s_shape.get(1));
        const auto batches = s_shape[0];
        const auto shell_count = noa::math::min(s_shape_3d) / 2 + 1;
        const auto cones = safe_cast<int32_t>(cone_count);

        const auto buffer_size = batches * cones * shell_count;
        const auto denominator_lhs = noa::cuda::memory::PtrDevice<Real>(buffer_size, stream); // allocates
        const auto denominator_rhs = noa::cuda::memory::PtrDevice<Real>(buffer_size, stream); // allocates

        const auto fsc_reduction = noa::signal::fft::details::anisotropicFSC<REMAP>(
                lhs.get(), safe_cast<uint4_t>(lhs_strides),
                rhs.get(), safe_cast<uint4_t>(rhs_strides),
                fsc.get(), denominator_lhs.get(), denominator_rhs.get(),
                s_shape_3d, shell_count,
                normalized_cone_directions.get(),
                cones, cone_aperture);

        noa::cuda::utils::iwise4D("signal::fft::anisotropicFFT", s_shape, fsc_reduction, stream);

        using iwise3D_config = noa::cuda::utils::IwiseStaticConfig<noa::cuda::Limits::WARP_SIZE, 1>;
        noa::cuda::utils::iwise3D<iwise3D_config>(
                "signal::fft::anisotropicFFT", int3_t{batches, cones, shell_count}, fsc_reduction, stream);

        stream.attach(lhs, rhs, fsc, normalized_cone_directions);
    }

    #define NOA_INSTANTIATE_FSC(R, T)               \
    template void isotropicFSC<R, T, void>(         \
            const shared_t<Complex<T>[]>&, dim4_t,  \
            const shared_t<Complex<T>[]>&, dim4_t,  \
            const shared_t<T[]>&, dim4_t,           \
            cuda::Stream&);                         \
    template void anisotropicFSC<R, T, void>(       \
            const shared_t<Complex<T>[]>&, dim4_t,  \
            const shared_t<Complex<T>[]>&, dim4_t,  \
            const shared_t<T[]>&, dim4_t,           \
            const shared_t<float3_t[]>&,            \
            dim_t, float,                           \
            cuda::Stream&)

    #define NOA_INSTANTIATE_FSC_ALL(T)      \
    NOA_INSTANTIATE_FSC(noa::fft::H2H, T);  \
    NOA_INSTANTIATE_FSC(noa::fft::HC2HC, T)

    NOA_INSTANTIATE_FSC_ALL(float);
    NOA_INSTANTIATE_FSC_ALL(double);
}
