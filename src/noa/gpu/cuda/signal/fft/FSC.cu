#include "noa/algorithms/signal/FSC.hpp"
#include "noa/gpu/cuda/signal/fft/FSC.hpp"
#include "noa/gpu/cuda/memory/AllocatorDevice.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"

namespace noa::cuda::signal::fft {
    template<noa::fft::Remap REMAP, typename Real, typename>
    void isotropic_fsc(
            const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
            const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
            Real* fsc,
            const Shape4<i64>& shape, cuda::Stream& stream
    ) {
        NOA_ASSERT(lhs && rhs && fsc);

        const auto s_shape = shape.as_safe<i32>();
        const auto s_shape_3d = s_shape.pop_front();
        const auto batches = s_shape[0];
        const auto shell_count = noa::math::min(s_shape_3d) / 2 + 1;

        using namespace noa::cuda::memory;
        const auto denominator_lhs = AllocatorDevice<Real>::allocate_async(batches * shell_count, stream);
        const auto denominator_rhs = AllocatorDevice<Real>::allocate_async(batches * shell_count, stream);

        const auto fsc_reduction = noa::algorithm::signal::isotropic_fsc<REMAP>(
                lhs, lhs_strides.as_safe<u32>(),
                rhs, rhs_strides.as_safe<u32>(),
                fsc, denominator_lhs.get(), denominator_rhs.get(),
                s_shape_3d, shell_count);

        noa::cuda::utils::iwise_4d(s_shape, fsc_reduction, stream);

        using iwise2D_config = noa::cuda::utils::IwiseStaticConfig<noa::cuda::Constant::WARP_SIZE, 1>;
        noa::cuda::utils::iwise_2d<iwise2D_config>(Shape2<i32>{batches, shell_count}, fsc_reduction, stream);
    }

    template<noa::fft::Remap REMAP, typename Real, typename>
    void anisotropic_fsc(
            const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
            const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
            Real* fsc,
            const Shape4<i64>& shape,
            const Vec3<f32>* normalized_cone_directions,
            i64 cone_count, f32 cone_aperture,
            cuda::Stream& stream
    ) {
        const auto s_shape = shape.as_safe<i32>();
        const auto s_shape_3d = s_shape.pop_front();
        const auto batches = s_shape[0];
        const auto shell_count = noa::math::min(s_shape_3d) / 2 + 1;
        const auto cones = safe_cast<i32>(cone_count);

        using namespace noa::cuda::memory;
        const auto buffer_size = batches * cones * shell_count;
        const auto denominator_lhs = AllocatorDevice<Real>::allocate_async(buffer_size, stream);
        const auto denominator_rhs = AllocatorDevice<Real>::allocate_async(buffer_size, stream);

        const auto fsc_reduction = noa::algorithm::signal::anisotropic_fsc<REMAP>(
                lhs, lhs_strides,
                rhs, rhs_strides,
                fsc, denominator_lhs.get(), denominator_rhs.get(),
                s_shape_3d, shell_count,
                normalized_cone_directions, cones, cone_aperture);

        noa::cuda::utils::iwise_4d(s_shape, fsc_reduction, stream);

        using iwise3D_config = noa::cuda::utils::IwiseStaticConfig<noa::cuda::Constant::WARP_SIZE, 1>;
        noa::cuda::utils::iwise_3d<iwise3D_config>(Shape3<i32>{batches, cones, shell_count}, fsc_reduction, stream);
    }

    #define NOA_INSTANTIATE_FSC(R, T)                   \
    template void isotropic_fsc<R, T, void>(            \
            const Complex<T>*, const Strides4<i64>&,    \
            const Complex<T>*, const Strides4<i64>&,    \
            T*, const Shape4<i64>&, Stream&);           \
    template void anisotropic_fsc<R, T, void>(          \
            const Complex<T>*, const Strides4<i64>&,    \
            const Complex<T>*, const Strides4<i64>&,    \
            T*, const Shape4<i64>&,                     \
            const Vec3<f32>*, i64, f32, Stream&)

    #define NOA_INSTANTIATE_FSC_ALL(T)              \
    NOA_INSTANTIATE_FSC(noa::fft::Remap::H2H, T);   \
    NOA_INSTANTIATE_FSC(noa::fft::Remap::HC2HC, T)

//    NOA_INSTANTIATE_FSC_ALL(f32);
//    NOA_INSTANTIATE_FSC_ALL(f64);
}
