#include "noa/algorithms/signal/FSC.hpp"
#include "noa/cpu/memory/PtrHost.hpp"
#include "noa/cpu/signal/fft/FSC.hpp"
#include "noa/cpu/utils/Iwise.hpp"

namespace noa::cpu::signal::fft {
    template<noa::fft::Remap REMAP, typename Real, typename>
    void isotropic_fsc(const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
                       const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
                       Real* fsc,
                       const Shape4<i64>& shape,
                       i64 threads) {
        NOA_ASSERT(lhs && rhs && fsc); // TODO

        const auto shape_3d = shape.pop_front();
        const auto batches = shape[0];
        const auto shell_count = noa::math::min(shape_3d) / 2 + 1;
        const auto s_shape_2d = Shape2<i64>{batches, shell_count};

        const auto denominator_lhs = noa::cpu::memory::PtrHost<Real>::alloc(batches * shell_count); // allocates
        const auto denominator_rhs = noa::cpu::memory::PtrHost<Real>::alloc(batches * shell_count); // allocates

        const auto fsc_reduction = noa::signal::fft::details::isotropic_fsc<REMAP>(
                lhs, lhs_strides,
                rhs, rhs_strides,
                fsc, denominator_lhs.get(), denominator_rhs.get(),
                shape_3d, shell_count);
        noa::cpu::utils::iwise_4d(shape, fsc_reduction, threads);
        noa::cpu::utils::iwise_2d(s_shape_2d, fsc_reduction, 1);
    }

    template<noa::fft::Remap REMAP, typename Real, typename>
    void anisotropic_fsc(const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
                         const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
                         Real* fsc,
                         const Shape4<i64>& shape,
                         const Vec3<f32>* normalized_cone_directions,
                         i64 cone_count, f32 cone_aperture,
                         i64 threads) {

        const auto shape_3d = shape.pop_front();
        const auto batches = shape[0];
        const auto shell_count = noa::math::min(shape_3d) / 2 + 1;
        const auto fsc_shape_3d = Shape3<i64>{batches, cone_count, shell_count};

        const auto buffer_size = batches * cone_count * shell_count;
        const auto denominator_lhs = noa::cpu::memory::PtrHost<Real>::alloc(buffer_size); // allocates
        const auto denominator_rhs = noa::cpu::memory::PtrHost<Real>::alloc(buffer_size); // allocates

        const auto fsc_reduction = noa::signal::fft::details::anisotropic_fsc<REMAP>(
                lhs, lhs_strides,
                rhs, rhs_strides,
                fsc, denominator_lhs.get(), denominator_rhs.get(),
                shape_3d, shell_count,
                normalized_cone_directions, cone_count, cone_aperture);
        noa::cpu::utils::iwise_4d(shape, fsc_reduction, threads);
        noa::cpu::utils::iwise_3d(fsc_shape_3d, fsc_reduction, 1);
    }

    #define NOA_INSTANTIATE_FSC(R, T)                   \
    template void isotropic_fsc<R, T, void>(            \
            const Complex<T>*, const Strides4<i64>&,    \
            const Complex<T>*, const Strides4<i64>&,    \
            T*, const Shape4<i64>&, i64);               \
    template void anisotropic_fsc<R, T, void>(          \
            const Complex<T>*, const Strides4<i64>&,    \
            const Complex<T>*, const Strides4<i64>&,    \
            T*, const Shape4<i64>&,                     \
            const Vec3<f32>*, i64, f32, i64)

    #define NOA_INSTANTIATE_FSC_ALL(T)              \
    NOA_INSTANTIATE_FSC(noa::fft::Remap::H2H, T);   \
    NOA_INSTANTIATE_FSC(noa::fft::Remap::HC2HC, T)

    NOA_INSTANTIATE_FSC_ALL(f32);
    NOA_INSTANTIATE_FSC_ALL(f64);
}
