#pragma once

#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::signal::fft::details {
    using Remap = ::noa::fft::Remap;

    template<typename T>
    void xcorr(const shared_t<Complex<T>[]>& lhs, size4_t lhs_stride,
               const shared_t<Complex<T>[]>& rhs, size4_t rhs_stride,
               size4_t shape, const shared_t<T[]>& coefficients,
               Stream& stream, const shared_t<T[]>& tmp, bool is_half);

    template<typename T>
    T xcorr(const shared_t<Complex<T>[]>& lhs, size4_t lhs_stride,
            const shared_t<Complex<T>[]>& rhs, size4_t rhs_stride,
            size4_t shape, Stream& stream, const shared_t<T[]>& tmp, bool is_half);

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

namespace noa::cuda::signal::fft {
    using Remap = ::noa::fft::Remap;
    using Norm = ::noa::fft::Norm;

    /// Computes the phase cross-correlation map.
    /// \tparam REMAP           Whether the output map should be centered. Should be H2F or H2FC.
    /// \tparam T               float or double.
    /// \param[in] lhs          On the \b device. Left-hand side non-redundant and non-centered FFT argument.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param[in,out] rhs      On the \b device. Right-hand side non-redundant and non-centered FFT argument.
    ///                         Can be overwritten (see \p tmp).
    /// \param rhs_stride       Rightmost stride of \p rhs.
    /// \param[out] output      On the \b device. Cross-correlation map.
    ///                         If REMAP is H2F, the central peak is at indexes {n, 0, 0, 0}.
    ///                         If REMAP is H2FC, the central peal is at indexes {n, shape[1]/2, shape[2]/2, shape[3]/2}.
    /// \param output_stride    Rightmost stride of \p output.
    /// \param shape            Rightmost logical shape.
    /// \param normalize        Whether the normalized cross-correlation should be returned.
    /// \param norm             Normalization mode to use for the C2R transform producing the final output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \param[out] tmp         On the \b device. Buffer that can fit \p shape.fft() complex elements.
    ///                         Can overlap with \p lhs or \p rhs. If nullptr, use \p rhs instead.
    /// \param tmp_stride       Rightmost stride of \p tmp. If \p tmp is nullptr, use \p rhs_stride instead.
    /// \note This function is asynchronous and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xmap_v<REMAP, T>>>
    void xmap(const shared_t<Complex<T>[]>& lhs, size4_t lhs_stride,
              const shared_t<Complex<T>[]>& rhs, size4_t rhs_stride,
              const shared_t<T[]>& output, size4_t output_stride,
              size4_t shape, bool normalize, Norm norm, Stream& stream,
              const shared_t<Complex<T>[]>& tmp = nullptr, size4_t tmp_stride = {});

    /// Find the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP           Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T               float, double.
    /// \param xmap             On the \b device. 2D cross-correlation map.
    /// \param stride           Rightmost stride of \p map.
    ///                         The second-most dimension should have a non-zero stride.
    /// \param shape            Rightmost shape of \p map.
    /// \param[out] peaks       On the \b host or \b device. Rightmost coordinates of the highest peak. One per batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak2D(const shared_t<T[]>& xmap, size4_t stride, size4_t shape,
                 const shared_t<float2_t[]>& peaks, Stream& stream);

    /// Returns the rightmost coordinates of the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP           Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T               float, double.
    /// \param xmap             On the \b device. Unbatched 2D cross-correlation map.
    /// \param stride           Rightmost stride of \p map.
    ///                         The second-most dimension should have a non-zero stride.
    /// \param shape            Rightmost shape of \p map.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    float2_t xpeak2D(const shared_t<T[]>& xmap, size4_t stride, size4_t shape, Stream& stream);

    /// Find the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP           Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T               float, double.
    /// \param xmap             On the \b device. Cross-correlation map.
    /// \param stride           Rightmost stride of \p map.
    ///                         The third-most and second-most dimension should have a non-zero stride.
    /// \param shape            Rightmost shape of \p map.
    /// \param[out] peaks       On the \b host or \b device. Rightmost coordinates of the highest peak. One per batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak3D(const shared_t<T[]>& xmap, size4_t stride, size4_t shape,
                 const shared_t<float3_t[]>& peaks, Stream& stream);

    /// Returns the rightmost coordinates of the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP           Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T               float, double.
    /// \param xmap             On the \b device. Unbatched cross-correlation map.
    /// \param stride           Rightmost stride of \p map.
    ///                         The third-most and second-most dimension should have a non-zero stride.
    /// \param shape            Rightmost shape of \p map.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    float3_t xpeak3D(const shared_t<T[]>& xmap, size4_t stride, size4_t shape, Stream& stream);

    /// Computes the cross-correlation coefficient(s).
    /// \tparam REMAP           Layout of \p lhs and \p rhs. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam T               cfloat_t or cdouble_t.
    /// \param[in] lhs          On the \b device. Left-hand side FFT.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param[in,out] rhs      On the \b device. Right-hand side FFT.
    /// \param rhs_stride       Rightmost stride of \p rhs.
    /// \param shape            Rightmost logical shape.
    /// \param[out] coeffs      On the \b host or \b device. Cross-correlation coefficient(s). One per batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \param[out] tmp         Contiguous temporary buffer able to fit the real components of \p lhs.
    ///                         If nullptr, a temporary array is stream-allocated.
    /// \note This function is asynchronous and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, T>>>
    void xcorr(const shared_t<Complex<T>[]>& lhs, size4_t lhs_stride,
               const shared_t<Complex<T>[]>& rhs, size4_t rhs_stride,
               size4_t shape, const shared_t<T[]>& coeffs,
               Stream& stream, const shared_t<T[]>& tmp = nullptr) {
        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        details::xcorr(lhs, lhs_stride, rhs, rhs_stride, shape, coeffs, stream, tmp, SRC_IS_HALF);
    }

    /// Computes the cross-correlation coefficient.
    /// \tparam REMAP           Layout of \p lhs and \p rhs. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam T               cfloat_t or cdouble_t.
    /// \param[in] lhs          On the \b device. Left-hand side non-redundant FFT.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param[in,out] rhs      On the \b device. Right-hand side non-redundant FFT.
    /// \param rhs_stride       Rightmost stride of \p rhs.
    /// \param shape            Rightmost logical shape. Should be unbatched.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \param[out] tmp         Contiguous temporary buffer able to fit the real components of \p lhs.
    ///                         If nullptr, a temporary array is stream-allocated.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, T>>>
    T xcorr(const shared_t<Complex<T>[]>& lhs, size4_t lhs_stride,
            const shared_t<Complex<T>[]>& rhs, size4_t rhs_stride,
            size4_t shape, Stream& stream, const shared_t<T[]>& tmp = nullptr) {
        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        return details::xcorr(lhs, lhs_stride, rhs, rhs_stride, shape, stream, tmp, SRC_IS_HALF);
    }
}
