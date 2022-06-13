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
    /// \param shape        Rightmost logical shape.
    /// \param normalize    Whether the normalized cross-correlation should be returned.
    /// \param norm         Normalization mode to use for the C2R transform producing the final output.
    /// \param[out] tmp     Contiguous buffer that can fit \p shape.fft() complex elements.
    ///                     Can be \p lhs or \p rhs. If empty, use \p rhs instead.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xmap_v<REMAP, T>>>
    void xmap(const Array<Complex<T>>& lhs, const Array<Complex<T>>& rhs, const Array<T>& output, size4_t shape,
              bool normalize = true, Norm norm = noa::fft::NORM_DEFAULT, const Array<Complex<T>>& tmp = {});

    /// Find the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately to the peak and 2 adjacent points.
    /// \tparam REMAP       Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T           float, double.
    /// \param[in] xmap     1D cross-correlation map.
    /// \param[out] peaks   Row vector with the rightmost coordinates of the highest peak. One per batch.
    ///                     If \p xmap is on the CPU, it should be dereferencable by the CPU.
    ///                     If \p xmap is on the CPU, it can be on any device, including the CPU.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak1D(const Array<T>& xmap, const Array<float>& peaks);

    /// Returns the rightmost coordinates of the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately to the peak and 2 adjacent points.
    /// \tparam REMAP   Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T       float, double.
    /// \param xmap     Unbatched 1D cross-correlation map.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    float xpeak1D(const Array<T>& xmap);

    /// Find the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP       Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T           float, double.
    /// \param[in] xmap     2D cross-correlation map.
    /// \param[out] peaks   Row vector with the rightmost coordinates of the highest peak. One per batch.
    ///                     If \p xmap is on the CPU, it should be dereferencable by the CPU.
    ///                     If \p xmap is on the CPU, it can be on any device, including the CPU.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak2D(const Array<T>& xmap, const Array<float2_t>& peaks);

    /// Returns the rightmost coordinates of the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP   Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T       float, double.
    /// \param xmap     Unbatched 2D cross-correlation map.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    float2_t xpeak2D(const Array<T>& xmap);

    /// Find the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP       Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T           float, double.
    /// \param[in] xmap     3D cross-correlation map.
    /// \param[out] peak    Row vector with the rightmost coordinates of the highest peak. One per batch.
    ///                     If \p xmap is on the CPU, it should be dereferencable by the CPU.
    ///                     If \p xmap is on the CPU, it can be on any device, including the CPU.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak3D(const Array<T>& xmap, const Array<float3_t>& peak);

    /// Returns the rightmost coordinates of the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP   Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T       float, double.
    /// \param xmap     Unbatched 3D cross-correlation map.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    float3_t xpeak3D(const Array<T>& xmap);

    /// Computes the cross-correlation coefficient(s).
    /// \tparam REMAP       Layout of \p lhs and \p rhs. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam T           cfloat_t or cdouble_t.
    /// \param[in] lhs      Left-hand side FFT.
    /// \param[in] rhs      Right-hand side FFT.
    /// \param shape        Rightmost logical shape.
    /// \param[out] coeffs  Row vector with the cross-correlation coefficient(s). One per batch.
    ///                     It should be dereferencable by the CPU.
    /// \param[out] tmp     Contiguous temporary array with the same shape as \p lhs and \p rhs.
    ///                     Only used by the CUDA backend. If empty, a temporary array is allocated.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, T>>>
    void xcorr(const Array<Complex<T>>& lhs, const Array<Complex<T>>& rhs,
               size4_t shape, const Array<T>& coeffs,
               const Array<T>& tmp = {});

    /// Computes the cross-correlation coefficient.
    /// \tparam REMAP       Layout of \p lhs and \p rhs. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam T           cfloat_t or cdouble_t.
    /// \param[in] lhs      Left-hand side FFT.
    /// \param[in,out] rhs  Right-hand side FFT.
    /// \param shape        Rightmost logical shape. Should be unbatched.
    /// \param[out] tmp     Contiguous temporary array with the same shape as \p lhs and \p rhs.
    ///                     Only used by the CUDA backend. If empty, a temporary array is allocated.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, T>>>
    T xcorr(const Array<Complex<T>>& lhs, const Array<Complex<T>>& rhs, size4_t shape);
}

#define NOA_UNIFIED_FFT_CORRELATE
#include "noa/unified/signal/fft/Correlate.inl"
#undef NOA_UNIFIED_FFT_CORRELATE
