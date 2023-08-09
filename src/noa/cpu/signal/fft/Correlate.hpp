#pragma once

#include "noa/core/Types.hpp"

namespace noa::cpu::signal::fft::details {
    using Remap = ::noa::fft::Remap;

    template<typename Real>
    Real xcorr(const Complex<Real>* lhs, const Strides3<i64>& lhs_strides,
               const Complex<Real>* rhs, const Strides3<i64>& rhs_strides,
               const Shape3<i64>& shape, i64 threads);

    template<Remap REMAP, typename Real>
    constexpr bool is_valid_xmap_v =
            noa::traits::is_any_v<Real, f32, f64> &&
            (REMAP == Remap::H2F || REMAP == Remap::H2FC);

    template<Remap REMAP, typename Real>
    constexpr bool is_valid_xpeak_v =
            noa::traits::is_any_v<Real, f32, f64> &&
            (REMAP == Remap::F2F || REMAP == Remap::FC2FC);

    template<Remap REMAP, typename Real>
    constexpr bool is_valid_xcorr_v =
            noa::traits::is_any_v<Real, f32, f64> &&
            (REMAP == Remap::H2H || REMAP == Remap::HC2HC ||
             REMAP == Remap::F2F || REMAP == Remap::FC2FC);
}

namespace noa::cpu::signal::fft {
    using Remap = ::noa::fft::Remap;
    using Norm = ::noa::fft::Norm;
    using CorrelationMode = ::noa::signal::CorrelationMode;
    using PeakMode = ::noa::signal::PeakMode;

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xmap_v<REMAP, Real>>>
    void xmap(const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
              Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
              Real* output, const Strides4<i64>& output_strides,
              const Shape4<i64>& shape, CorrelationMode correlation_mode, Norm norm,
              Complex<Real>* tmp, Strides4<i64> tmp_strides, i64 threads);

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, Real>>>
    void xcorr(const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
               const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
               const Shape4<i64>& shape, Real* coefficients, i64 threads) {
        NOA_ASSERT(lhs && rhs && noa::all(shape > 0));
        constexpr bool SRC_IS_HALF = noa::to_underlying(REMAP) & noa::fft::Layout::SRC_HALF;
        const auto batches = shape[0];
        const auto input_shape = SRC_IS_HALF ? shape.pop_front() : shape.pop_front().rfft();
        const auto lhs_strides_3d = lhs_strides.pop_front();
        const auto rhs_strides_3d = rhs_strides.pop_front();

        for (i64 batch = 0; batch < batches; ++batch) {
            const Complex<Real>* lhs_ptr = lhs + batch * lhs_strides[0];
            const Complex<Real>* rhs_ptr = rhs + batch * rhs_strides[0];
            Real* coefficient = coefficients + batch;
            *coefficient = details::xcorr(lhs_ptr, lhs_strides_3d, rhs_ptr, rhs_strides_3d, input_shape, threads);
        }
    }

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, Real>>>
    Real xcorr(const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
               const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
               const Shape4<i64>& shape, i64 threads) {
        NOA_ASSERT(lhs && rhs && noa::all(shape > 0));
        NOA_ASSERT(shape[0] == 1);
        constexpr bool SRC_IS_HALF = noa::to_underlying(REMAP) & noa::fft::Layout::SRC_HALF;
        const auto input_shape = SRC_IS_HALF ? shape.pop_front() : shape.pop_front().rfft();
        const auto lhs_strides_3d = lhs_strides.pop_front();
        const auto rhs_strides_3d = rhs_strides.pop_front();
        return details::xcorr(lhs, lhs_strides_3d, rhs, rhs_strides_3d, input_shape, threads);
    }
}

namespace noa::cpu::signal::fft {
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    void xpeak_1d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, Vec1<f32> xmap_ellipse_radius,
                  Vec1<f32>* output_peak_coordinates, Real* output_peak_values,
                  PeakMode peak_mode, Vec1<i64> peak_radius, i64 threads);

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    auto xpeak_1d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, Vec1<f32> xmap_ellipse_radius,
                  PeakMode peak_mode, Vec1<i64> peak_radius, i64 threads) -> std::pair<Vec1<f32>, Real>;

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    void xpeak_2d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec2<f32>& xmap_ellipse_radius,
                  Vec2<f32>* output_peak_coordinates, Real* output_peak_values,
                  PeakMode peak_mode, const Vec2<i64>& peak_radius, i64 threads);

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    auto xpeak_2d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec2<f32>& xmap_ellipse_radius,
                  PeakMode peak_mode, const Vec2<i64>& peak_radius, i64 threads) -> std::pair<Vec2<f32>, Real>;

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    void xpeak_3d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec3<f32>& xmap_ellipse_radius,
                  Vec3<f32>* output_peak_coordinates, Real* output_peak_values,
                  PeakMode peak_mode, const Vec3<i64>& peak_radius, i64 threads);

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    auto xpeak_3d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec3<f32>& xmap_ellipse_radius,
                  PeakMode peak_mode, const Vec3<i64>& peak_radius, i64 threads) -> std::pair<Vec3<f32>, Real>;
}
