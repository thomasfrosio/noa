#pragma once

#include "noa/unified/Array.h"
#include "noa/unified/fft/Transform.h"

namespace noa::signal::fft::details {
    using Remap = ::noa::fft::Remap;

    template<Remap REMAP, typename T>
    constexpr bool is_valid_xmap_v = traits::is_any_v<T, float, double> &&
                                     (REMAP == Remap::H2F || REMAP == Remap::H2FC);

    template<Remap REMAP, typename T>
    constexpr bool is_valid_xpeak_v = traits::is_any_v<T, float, double> &&
                                      (REMAP == Remap::F2F || REMAP == Remap::FC2FC);

    template<Remap REMAP, typename T>
    constexpr bool is_valid_xcorr_v = traits::is_any_v<T, float, double> &&
                                      (REMAP == Remap::H2H || REMAP == Remap::HC2HC ||
                                       REMAP == Remap::F2F || REMAP == Remap::FC2FC);
}

namespace noa::signal::fft {
    using Remap = ::noa::fft::Remap;
    using Norm = ::noa::fft::Norm;

    /// Computes the phase cross-correlation map.
    /// \tparam REMAP       Whether the output map should be centered. Should be H2F or H2FC.
    /// \tparam T           float or double.
    /// \param[in] lhs      Left-hand side non-redundant and non-centered FFT argument.
    /// \param[in,out] rhs  Right-hand side non-redundant and non-centered FFT argument.
    ///                     Overwritten by default (see \p tmp).
    /// \param[out] output  Cross-correlation map.
    ///                     If REMAP is H2F, the central peak is at indexes {n, 0, 0, 0}.
    ///                     If REMAP is H2FC, the central peal is at indexes {n, shape[1]/2, shape[2]/2, shape[3]/2}.
    /// \param normalize    Whether the normalized cross-correlation should be returned.
    /// \param norm         Normalization mode to use for the C2R transform producing the final output.
    /// \param[out] tmp     Buffer that can fit \p shape.fft() complex elements. It is overwritten.
    ///                     Can be \p lhs or \p rhs. If empty, use \p rhs instead.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xmap_v<REMAP, T>>>
    void xmap(const Array<Complex<T>>& lhs, const Array<Complex<T>>& rhs, const Array<T>& output,
              bool normalize = true, Norm norm = noa::fft::NORM_DEFAULT, const Array<Complex<T>>& tmp = {});

    /// Find the highest peak in a cross-correlation line.
    /// \tparam REMAP               Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T                   float, double.
    /// \param[in] xmap             1D cross-correlation map. Should be a column or row vector.
    /// \param[out] peaks           Coordinates of the highest peak. One per batch.
    ///                             If \p xmap is on the CPU, it should be dereferenceable by the CPU.
    ///                             If \p xmap is on the GPU, it can be on any device, including the CPU.
    /// \param ellipse_radius       Radius of the smooth elliptic mask to apply (in-place) on \p xmap.
    ///                             This is used to restrict the peak position relative to the center of \p xmap.
    ///                             If negative or 0, it is ignored.
    /// \param registration_radius  Radius of the window, centered on the peak, for subpixel registration.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak1D(const Array<T>& xmap, const Array<float>& peaks,
                 float ellipse_radius = -1,
                 int64_t registration_radius = 1);

    /// Returns the coordinates of the highest peak in a cross-correlation map.
    /// \tparam REMAP               Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T                   float, double.
    /// \param xmap                 Unbatched 1D cross-correlation map. Should be a column or row vector.
    /// \param ellipse_radius       Radius of the smooth elliptic mask to apply (in-place) on \p xmap.
    ///                             This is used to restrict the peak position relative to the center of \p xmap.
    ///                             If negative or 0, it is ignored.
    /// \param registration_radius  Radius of the window, centered on the peak, for subpixel registration.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    [[nodiscard]] float xpeak1D(const Array<T>& xmap,
                                float ellipse_radius = -1,
                                int64_t registration_radius = 1);

    /// Find the highest peak in a cross-correlation map.
    /// \tparam REMAP               Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T                   float, double.
    /// \param[in,out] xmap         2D cross-correlation map. It can be overwritten depending on \p max_radius.
    /// \param[out] peaks           HW coordinates of the highest peak. One per batch.
    ///                             If \p xmap is on the CPU, it should be dereferenceable by the CPU.
    ///                             If \p xmap is on the GPU, it can be on any device, including the CPU.
    /// \param ellipse_radius       HW radius of the smooth elliptic mask to apply (in-place) on \p xmap.
    ///                             This is used to restrict the peak position relative to the center of \p xmap.
    ///                             If negative or 0, it is ignored.
    /// \param registration_radius  HW radius of the window, centered on the peak, for subpixel registration.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak2D(const Array<T>& xmap, const Array<float2_t>& peaks,
                 float2_t ellipse_radius = float2_t{-1},
                 long2_t registration_radius = long2_t{1});

    /// Returns the HW coordinates of the highest peak in a cross-correlation map.
    /// \tparam REMAP               Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T                   float, double.
    /// \param xmap                 Unbatched 2D cross-correlation map.
    /// \param ellipse_radius       HW radius of the smooth elliptic mask to apply (in-place) on \p xmap.
    ///                             This is used to restrict the peak position relative to the center of \p xmap.
    ///                             If negative or 0, it is ignored.
    /// \param registration_radius  HW radius of the window, centered on the peak, for subpixel registration.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    [[nodiscard]] float2_t xpeak2D(const Array<T>& xmap,
                                   float2_t ellipse_radius = float2_t{-1},
                                   long2_t registration_radius = long2_t{1});

    /// Find the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP               Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T                   float, double.
    /// \param[in] xmap             3D cross-correlation map.
    /// \param[out] peak            DHW coordinates of the highest peak. One per batch.
    ///                             If \p xmap is on the CPU, it should be dereferenceable by the CPU.
    ///                             If \p xmap is on the GPU, it can be on any device, including the CPU.
    /// \param ellipse_radius       DHW radius of the smooth elliptic mask to apply (in-place) on \p xmap.
    ///                             This is used to restrict the peak position relative to the center of \p xmap.
    ///                             If negative or 0, it is ignored.
    /// \param registration_radius  DHW radius of the window, centered on the peak, for subpixel registration.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak3D(const Array<T>& xmap, const Array<float3_t>& peak,
                 float3_t ellipse_radius = float3_t{-1},
                 long3_t registration_radius = long3_t{1});

    /// Returns the DHW coordinates of the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP               Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T                   float, double.
    /// \param xmap                 Unbatched 3D cross-correlation map.
    /// \param ellipse_radius       DHW radius of the smooth elliptic mask to apply (in-place) on \p xmap.
    ///                             This is used to restrict the peak position relative to the center of \p xmap.
    ///                             If negative or 0, it is ignored.
    /// \param registration_radius  DHW radius of the window, centered on the peak, for subpixel registration.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    [[nodiscard]] float3_t xpeak3D(const Array<T>& xmap,
                                   float3_t ellipse_radius = float3_t{-1},
                                   long3_t registration_radius = long3_t{1});

    /// Computes the cross-correlation coefficient(s).
    /// \tparam REMAP       Layout of \p lhs and \p rhs. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam T           cfloat_t or cdouble_t.
    /// \param[in] lhs      Left-hand side FFT.
    /// \param[in] rhs      Right-hand side FFT.
    /// \param shape        BDHW logical shape.
    /// \param[out] coeffs  Cross-correlation coefficient(s). One per batch.
    ///                     It should be dereferenceable by the CPU.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, T>>>
    void xcorr(const Array<Complex<T>>& lhs, const Array<Complex<T>>& rhs,
               dim4_t shape, const Array<T>& coeffs);

    /// Computes the cross-correlation coefficient.
    /// \tparam REMAP   Layout of \p lhs and \p rhs. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam T       cfloat_t or cdouble_t.
    /// \param[in] lhs  Left-hand side FFT.
    /// \param[in] rhs  Right-hand side FFT.
    /// \param shape    BDHW logical shape. Should not be batched.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, T>>>
    [[nodiscard]] T xcorr(const Array<Complex<T>>& lhs, const Array<Complex<T>>& rhs, dim4_t shape);
}

#define NOA_UNIFIED_FFT_CORRELATE
#include "noa/unified/signal/fft/Correlate.inl"
#undef NOA_UNIFIED_FFT_CORRELATE
