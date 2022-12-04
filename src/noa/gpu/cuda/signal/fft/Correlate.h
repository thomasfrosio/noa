#pragma once

#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::signal::fft::details {
    using Remap = ::noa::fft::Remap;

    template<typename Real>
    void xcorr(const shared_t<Complex<Real>[]>& lhs, dim4_t lhs_strides,
               const shared_t<Complex<Real>[]>& rhs, dim4_t rhs_strides,
               dim4_t shape, const shared_t<Real[]>& coefficients,
               Stream& stream, bool is_half);

    template<typename Real>
    Real xcorr(const shared_t<Complex<Real>[]>& lhs, dim4_t lhs_strides,
               const shared_t<Complex<Real>[]>& rhs, dim4_t rhs_strides,
               dim4_t shape, Stream& stream, bool is_half);

    template<Remap REMAP, typename Real>
    constexpr bool is_valid_xmap_v = traits::is_any_v<Real, float, double> &&
                                     (REMAP == Remap::H2F || REMAP == Remap::H2FC);

    template<Remap REMAP, typename Real>
    constexpr bool is_valid_xpeak_v = traits::is_any_v<Real, float, double> &&
                                      (REMAP == Remap::F2F || REMAP == Remap::FC2FC);

    template<Remap REMAP, typename Real>
    constexpr bool is_valid_xcorr_v = traits::is_any_v<Real, float, double> &&
                                      (REMAP == Remap::H2H || REMAP == Remap::HC2HC ||
                                       REMAP == Remap::F2F || REMAP == Remap::FC2FC);
}

namespace noa::cuda::signal::fft {
    using Remap = ::noa::fft::Remap;
    using Norm = ::noa::fft::Norm;
    using CorrelationMode = ::noa::signal::CorrelationMode;
    using PeakMode = ::noa::signal::PeakMode;

    // Computes the phase cross-correlation map.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xmap_v<REMAP, Real>>>
    void xmap(const shared_t<Complex<Real>[]>& lhs, dim4_t lhs_strides,
              const shared_t<Complex<Real>[]>& rhs, dim4_t rhs_strides,
              const shared_t<Real[]>& output, dim4_t output_strides,
              dim4_t shape, CorrelationMode correlation_mode, Norm norm, Stream& stream,
              const shared_t<Complex<Real>[]>& tmp = nullptr, dim4_t tmp_strides = {});

    // Find the highest peak in a cross-correlation line.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    void xpeak1D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float xmap_ellipse_radius,
                 const shared_t<float[]>& peak_coordinates, PeakMode peak_mode, int64_t peak_radius, Stream& stream);

    // Returns the coordinates of the highest peak in a cross-correlation line.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    float xpeak1D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float xmap_ellipse_radius,
                  PeakMode peak_mode, int64_t peak_radius, Stream& stream);

    // Find the highest peak in a cross-correlation map.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    void xpeak2D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float2_t xmap_ellipse_radius,
                 const shared_t<float2_t[]>& peak_coordinates, PeakMode peak_mode, long2_t peak_radius, Stream& stream);

    // Returns the HW coordinates of the highest peak in a cross-correlation map.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    float2_t xpeak2D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float2_t xmap_ellipse_radius,
                     PeakMode peak_mode, long2_t peak_radius, Stream& stream);

    // Find the highest peak in a cross-correlation map.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    void xpeak3D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float3_t xmap_ellipse_radius,
                 const shared_t<float3_t[]>& peak_coordinates, PeakMode peak_mode, long3_t peak_radius, Stream& stream);

    // Returns the DHW coordinates of the highest peak in a cross-correlation map.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    float3_t xpeak3D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float3_t xmap_ellipse_radius,
                     PeakMode peak_mode, long3_t peak_radius, Stream& stream);

    // Computes the cross-correlation coefficient(s).
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, Real>>>
    void xcorr(const shared_t<Complex<Real>[]>& lhs, dim4_t lhs_strides,
               const shared_t<Complex<Real>[]>& rhs, dim4_t rhs_strides,
               dim4_t shape, const shared_t<Real[]>& coeffs,
               Stream& stream) {
        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        details::xcorr(lhs, lhs_strides, rhs, rhs_strides, shape, coeffs, stream, SRC_IS_HALF);
    }

    // Computes the cross-correlation coefficient.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, Real>>>
    Real xcorr(const shared_t<Complex<Real>[]>& lhs, dim4_t lhs_strides,
               const shared_t<Complex<Real>[]>& rhs, dim4_t rhs_strides,
               dim4_t shape, Stream& stream) {
        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        return details::xcorr(lhs, lhs_strides, rhs, rhs_strides, shape, stream, SRC_IS_HALF);
    }
}
