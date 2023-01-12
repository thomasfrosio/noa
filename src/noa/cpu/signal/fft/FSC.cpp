#include "noa/common/signal/details/FSC.h"

#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/signal/fft/FSC.h"
#include "noa/cpu/utils/Iwise.h"

namespace noa::cpu::signal::fft {
    template<noa::fft::Remap REMAP, typename Real, typename>
    void isotropicFSC(const shared_t<Complex<Real>[]>& lhs, dim4_t lhs_strides,
                      const shared_t<Complex<Real>[]>& rhs, dim4_t rhs_strides,
                      const shared_t<Real[]>& fsc,
                      dim4_t shape, cpu::Stream& stream) {
        NOA_ASSERT(lhs && rhs && fsc); // TODO

        const auto s_shape_4d = safe_cast<long4_t>(shape);
        const auto threads = stream.threads();

        stream.enqueue([=]() {
            const auto s_shape_3d = long3_t(s_shape_4d.get(1));
            const auto batches = s_shape_4d[0];
            const auto shell_count = noa::math::min(s_shape_3d) / 2 + 1;
            const auto s_shape_2d = long2_t{batches, shell_count};

            const auto denominator_lhs = noa::cpu::memory::PtrHost<Real>(batches * shell_count); // allocates
            const auto denominator_rhs = noa::cpu::memory::PtrHost<Real>(batches * shell_count); // allocates

            const auto fsc_reduction = noa::signal::fft::details::isotropicFSC<REMAP>(
                    lhs.get(), safe_cast<long4_t>(lhs_strides),
                    rhs.get(), safe_cast<long4_t>(rhs_strides),
                    fsc.get(), denominator_lhs.get(), denominator_rhs.get(),
                    s_shape_3d, shell_count);
            noa::cpu::utils::iwise4D(s_shape_4d, fsc_reduction, threads);
            noa::cpu::utils::iwise2D(s_shape_2d, fsc_reduction, 1);
        });
    }

    template<noa::fft::Remap REMAP, typename Real, typename>
    void anisotropicFSC(const shared_t<Complex<Real>[]>& lhs, dim4_t lhs_strides,
                        const shared_t<Complex<Real>[]>& rhs, dim4_t rhs_strides,
                        const shared_t<Real[]>& fsc,
                        dim4_t shape,
                        const shared_t<float3_t[]>& normalized_cone_directions,
                        dim_t cone_count, float cone_aperture,
                        cpu::Stream& stream) {

        const auto s_shape_4d = safe_cast<long4_t>(shape);
        const auto threads = stream.threads();
        const auto cones = safe_cast<int64_t>(cone_count);

        stream.enqueue([=]() {
            const auto s_shape_3d = long3_t(s_shape_4d.get(1));
            const auto batches = s_shape_4d[0];
            const auto shell_count = noa::math::min(s_shape_3d) / 2 + 1;
            const auto fsc_shape_3d = long3_t{batches, cones, shell_count};

            const auto buffer_size = batches * cones * shell_count;
            const auto denominator_lhs = noa::cpu::memory::PtrHost<Real>(buffer_size); // allocates
            const auto denominator_rhs = noa::cpu::memory::PtrHost<Real>(buffer_size); // allocates

            const auto fsc_reduction = noa::signal::fft::details::anisotropicFSC<REMAP>(
                    lhs.get(), safe_cast<long4_t>(lhs_strides),
                    rhs.get(), safe_cast<long4_t>(rhs_strides),
                    fsc.get(), denominator_lhs.get(), denominator_rhs.get(),
                    s_shape_3d, shell_count,
                    normalized_cone_directions.get(), cones, cone_aperture);
            noa::cpu::utils::iwise4D(s_shape_4d, fsc_reduction, threads);
            noa::cpu::utils::iwise3D(fsc_shape_3d, fsc_reduction, 1);
        });
    }

    #define NOA_INSTANTIATE_FSC(R, T)               \
    template void isotropicFSC<R, T, void>(         \
            const shared_t<Complex<T>[]>&, dim4_t,  \
            const shared_t<Complex<T>[]>&, dim4_t,  \
            const shared_t<T[]>&, dim4_t,           \
            cpu::Stream&);                          \
    template void anisotropicFSC<R, T, void>(       \
            const shared_t<Complex<T>[]>&, dim4_t,  \
            const shared_t<Complex<T>[]>&, dim4_t,  \
            const shared_t<T[]>&, dim4_t,           \
            const shared_t<float3_t[]>&,            \
            dim_t, float,                           \
            cpu::Stream&)

    #define NOA_INSTANTIATE_FSC_ALL(T)      \
    NOA_INSTANTIATE_FSC(noa::fft::H2H, T);  \
    NOA_INSTANTIATE_FSC(noa::fft::HC2HC, T)

    NOA_INSTANTIATE_FSC_ALL(float);
    NOA_INSTANTIATE_FSC_ALL(double);
}
