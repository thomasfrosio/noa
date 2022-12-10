#pragma once

#include "noa/unified/Array.h"
#include "noa/unified/fft/Transform.h"

namespace noa::signal::fft::details {
    using Remap = ::noa::fft::Remap;

    template<Remap REMAP, typename Real>
    constexpr bool is_valid_xmap_v =
            traits::is_any_v<Real, float, double> &&
            (REMAP == Remap::H2F || REMAP == Remap::H2FC);

    template<Remap REMAP, typename Real>
    constexpr bool is_valid_xpeak_v =
            traits::is_any_v<Real, float, double> &&
            (REMAP == Remap::F2F || REMAP == Remap::FC2FC);

    template<Remap REMAP, typename Real>
    constexpr bool is_valid_xcorr_v =
            traits::is_any_v<Real, float, double> &&
            (REMAP == Remap::H2H || REMAP == Remap::HC2HC ||
             REMAP == Remap::F2F || REMAP == Remap::FC2FC);
}

namespace noa::signal::fft {
    using Remap = ::noa::fft::Remap;
    using Norm = ::noa::fft::Norm;

    /// Computes the cross-correlation map.
    /// \tparam REMAP           Whether the output map should be centered. Should be H2F or H2FC.
    /// \tparam Real            float or double.
    /// \param[in] lhs          Left-hand side non-redundant and non-centered FFT argument.
    /// \param[in,out] rhs      Right-hand side non-redundant and non-centered FFT argument.
    ///                         Overwritten by default (see \p buffer).
    /// \param[out] output      Cross-correlation map.
    ///                         If REMAP is H2F, the central peak is at indexes {n, 0, 0, 0}.
    ///                         If REMAP is H2FC, the central peal is at indexes {n, shape[1]/2, shape[2]/2, shape[3]/2}.
    /// \param correlation_mode Correlation mode to use. Remember that DOUBLE_PHASE_CORRELATION doubles the shifts.
    /// \param fft_norm         Normalization mode to use for the C2R transform producing the final output.
    ///                         This should match the mode that was used to compute the input transforms.
    /// \param[out] buffer      Buffer that can fit \p shape.fft() complex elements. It is overwritten.
    ///                         Can be \p lhs or \p rhs. If empty, use \p rhs instead.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xmap_v<REMAP, Real>>>
    void xmap(const Array<Complex<Real>>& lhs, const Array<Complex<Real>>& rhs, const Array<Real>& output,
              CorrelationMode correlation_mode = CONVENTIONAL_CORRELATION,
              Norm fft_norm = noa::fft::NORM_DEFAULT,
              const Array<Complex<Real>>& buffer = {});

    /// Find the highest peak in a cross-correlation line.
    /// \tparam REMAP               Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam Real                float or double.
    /// \param[in] xmap             1D cross-correlation map. Should be a column or row vector.
    ///                             It can be overwritten depending on \p peak_radius.
    /// \param[out] peaks           Output coordinates of the highest peak. One per batch.
    /// \param xmap_ellipse_radius  Radius of the smooth mask to apply (in-place) on \p xmap.
    ///                             This is used to restrict the peak position relative to the center
    ///                             of \p xmap. If negative or 0, it is ignored.
    /// \param peak_mode            Registration mode to use for subpixel accuracy.
    /// \param peak_radius          Radius of the registration window, centered on the peak.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    void xpeak1D(const Array<Real>& xmap, const Array<float>& peaks,
                 float xmap_ellipse_radius = 0,
                 PeakMode peak_mode = PEAK_PARABOLA_1D,
                 int64_t peak_radius = 1);

    /// Returns the coordinates of the highest peak in a cross-correlation map.
    /// \tparam REMAP               Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam Real                float or double.
    /// \param[in] xmap             1D cross-correlation map. Should be a column or row vector.
    ///                             It can be overwritten depending on \p peak_radius.
    /// \param xmap_ellipse_radius  Radius of the smooth mask to apply (in-place) on \p xmap.
    ///                             This is used to restrict the peak position relative to the center
    ///                             of \p xmap. If negative or 0, it is ignored.
    /// \param peak_mode            Registration mode to use for subpixel accuracy.
    /// \param peak_radius          Radius of the registration window, centered on the peak.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    [[nodiscard]] float xpeak1D(const Array<Real>& xmap,
                                float xmap_ellipse_radius = 0,
                                PeakMode peak_mode = PEAK_PARABOLA_1D,
                                int64_t peak_radius = 1);

    /// Find the highest peak in a cross-correlation map.
    /// \tparam REMAP               Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam Real                float or double.
    /// \param[in,out] xmap         2D cross-correlation map. It can be overwritten depending on \p peak_radius.
    /// \param[out] peaks           Output HW coordinates of the highest peak. One per batch.
    /// \param xmap_ellipse_radius  HW radius of the smooth elliptic mask to apply (in-place) on \p xmap.
    ///                             This is used to restrict the peak position relative to the center
    ///                             of \p xmap. If negative or 0, it is ignored.
    /// \param peak_mode            Registration mode to use for subpixel accuracy.
    /// \param peak_radius          HW radius of the registration window, centered on the peak.
    /// \note On the GPU, \p peak_radius is limited to 8 with \p peak_mode PEAK_PARABOLA_COM.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    void xpeak2D(const Array<Real>& xmap, const Array<float2_t>& peaks,
                 float2_t xmap_ellipse_radius = float2_t{0},
                 PeakMode peak_mode = PEAK_PARABOLA_1D,
                 long2_t peak_radius = long2_t{1});

    /// Returns the HW coordinates of the highest peak in a cross-correlation map.
    /// \tparam REMAP               Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam Real                float or double.
    /// \param[in,out] xmap         2D cross-correlation map. It can be overwritten depending on \p peak_radius.
    /// \param xmap_ellipse_radius  HW radius of the smooth elliptic mask to apply (in-place) on \p xmap.
    ///                             This is used to restrict the peak position relative to the center
    ///                             of \p xmap. If negative or 0, it is ignored.
    /// \param peak_mode            Registration mode to use for subpixel accuracy.
    /// \param peak_radius          HW radius of the registration window, centered on the peak.
    /// \note On the GPU, \p peak_radius is limited to 8 with \p peak_mode PEAK_PARABOLA_COM.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    [[nodiscard]] float2_t xpeak2D(const Array<Real>& xmap,
                                   float2_t xmap_ellipse_radius = float2_t{0},
                                   PeakMode peak_mode = PEAK_PARABOLA_1D,
                                   long2_t peak_radius = long2_t{1});

    /// Find the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP               Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam Real                float or double.
    /// \param[in,out] xmap         3D cross-correlation map. It can be overwritten depending on \p peak_radius.
    /// \param[out] peaks           Output DHW coordinates of the highest peak. One per batch.
    /// \param xmap_ellipse_radius  DHW radius of the smooth elliptic mask to apply (in-place) on \p xmap.
    ///                             This is used to restrict the peak position relative to the center
    ///                             of \p xmap. If negative or 0, it is ignored.
    /// \param peak_mode            Registration mode to use for subpixel accuracy.
    /// \param peak_radius          DHW radius of the registration window, centered on the peak.
    /// \note On the GPU, \p peak_radius is limited to 2 with \p peak_mode PEAK_PARABOLA_COM.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    void xpeak3D(const Array<Real>& xmap, const Array<float3_t>& peak,
                 float3_t xmap_ellipse_radius = float3_t{0},
                 PeakMode peak_mode = PEAK_PARABOLA_1D,
                 long3_t peak_radius = long3_t{1});

    /// Returns the DHW coordinates of the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP               Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam Real                float or double.
    /// \param[in,out] xmap         3D cross-correlation map. It can be overwritten depending on \p peak_radius.
    /// \param xmap_ellipse_radius  DHW radius of the smooth elliptic mask to apply (in-place) on \p xmap.
    ///                             This is used to restrict the peak position relative to the center
    ///                             of \p xmap. If negative or 0, it is ignored.
    /// \param peak_mode            Registration mode to use for subpixel accuracy.
    /// \param peak_radius          DHW radius of the registration window, centered on the peak.
    /// \note On the GPU, \p peak_radius is limited to 2 with \p peak_mode PEAK_PARABOLA_COM.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, Real>>>
    [[nodiscard]] float3_t xpeak3D(const Array<Real>& xmap,
                                   float3_t xmap_ellipse_radius = float3_t{0},
                                   PeakMode peak_mode = PEAK_PARABOLA_1D,
                                   long3_t peak_radius = long3_t{1});

    /// Computes the cross-correlation coefficient(s).
    /// \tparam REMAP               Layout of \p lhs and \p rhs. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam Real                float or double.
    /// \param[in] lhs              Left-hand side FFT.
    /// \param[in] rhs              Right-hand side FFT.
    /// \param shape                BDHW logical shape.
    /// \param[out] coefficients    Cross-correlation coefficient(s). One per batch.
    ///                             It should be dereferenceable by the CPU.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, Real>>>
    void xcorr(const Array<Complex<Real>>& lhs, const Array<Complex<Real>>& rhs,
               dim4_t shape, const Array<Real>& coefficients);

    /// Computes the cross-correlation coefficient.
    /// \tparam REMAP   Layout of \p lhs and \p rhs. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam Real    float or double.
    /// \param[in] lhs  Left-hand side FFT.
    /// \param[in] rhs  Right-hand side FFT.
    /// \param shape    BDHW logical shape. Should not be batched.
    template<Remap REMAP, typename Real, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, Real>>>
    [[nodiscard]] Real xcorr(const Array<Complex<Real>>& lhs, const Array<Complex<Real>>& rhs, dim4_t shape);
}
