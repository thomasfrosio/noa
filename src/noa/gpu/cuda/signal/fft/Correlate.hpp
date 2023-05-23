#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::signal::fft::details {
    using Remap = ::noa::fft::Remap;

    template<typename Real>
    void xcorr(const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
               const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
               const Shape4<i64>& shape, Real* coefficients, Stream& stream);

    template<typename Real>
    Real xcorr(const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
               const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
               const Shape4<i64>& shape, Stream& stream);

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

namespace noa::cuda::signal::fft {
    using Remap = ::noa::fft::Remap;
    using Norm = ::noa::fft::Norm;
    using CorrelationMode = ::noa::signal::CorrelationMode;
    using PeakMode = ::noa::signal::PeakMode;

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xmap_v<REMAP, Real>>>
    void xmap(const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
              Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
              Real* output, const Strides4<i64>& output_strides,
              const Shape4<i64>& shape, CorrelationMode correlation_mode, Norm norm,
              Complex<Real>* tmp, Strides4<i64> tmp_strides, Stream& stream);

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, Real>>>
    void xcorr(const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
               const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
               const Shape4<i64>& shape, Real* coefficients, Stream& stream) {
        constexpr bool SRC_IS_HALF = noa::traits::to_underlying(REMAP) & noa::fft::Layout::SRC_HALF;
        const auto shape_fft = SRC_IS_HALF ? shape.rfft() : shape;
        details::xcorr(lhs, lhs_strides, rhs, rhs_strides, shape_fft, coefficients, stream);
    }

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, Real>>>
    Real xcorr(const Complex<Real>* lhs, const Strides4<i64>& lhs_strides,
               const Complex<Real>* rhs, const Strides4<i64>& rhs_strides,
               const Shape4<i64>& shape, Stream& stream) {
        constexpr bool SRC_IS_HALF = noa::traits::to_underlying(REMAP) & noa::fft::Layout::SRC_HALF;
        const auto shape_fft = SRC_IS_HALF ? shape.rfft() : shape;
        return details::xcorr(lhs, lhs_strides, rhs, rhs_strides, shape_fft, stream);
    }
}

namespace noa::cuda::signal::fft {
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    void xpeak_1d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, Vec1<f32> xmap_ellipse_radius,
                  Vec1<f32>* output_peak_coordinates, Real* output_peak_values,
                  PeakMode peak_mode, Vec1<i64> peak_radius, Stream& stream);

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    auto xpeak_1d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, Vec1<f32> xmap_ellipse_radius,
                  PeakMode peak_mode, Vec1<i64> peak_radius, Stream& stream) -> std::pair<Vec1<f32>, Real>;

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    void xpeak_2d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec2<f32>& xmap_ellipse_radius,
                  Vec2<f32>* output_peak_coordinates, Real* output_peak_values,
                  PeakMode peak_mode, const Vec2<i64>& peak_radius, Stream& stream);

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    auto xpeak_2d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec2<f32>& xmap_ellipse_radius,
                  PeakMode peak_mode, const Vec2<i64>& peak_radius, Stream& stream) -> std::pair<Vec2<f32>, Real>;

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    void xpeak_3d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec3<f32>& xmap_ellipse_radius,
                  Vec3<f32>* output_peak_coordinates, Real* output_peak_values,
                  PeakMode peak_mode, const Vec3<i64>& peak_radius, Stream& stream);

    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    auto xpeak_3d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec3<f32>& xmap_ellipse_radius,
                  PeakMode peak_mode, const Vec3<i64>& peak_radius, Stream& stream) -> std::pair<Vec3<f32>, Real>;
}
