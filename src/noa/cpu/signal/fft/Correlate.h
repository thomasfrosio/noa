#pragma once

#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::signal::fft::details {
    using Remap = ::noa::fft::Remap;

    template<typename T>
    T xcorr(const Complex<T>* lhs, size3_t lhs_stride,
            const Complex<T>* rhs, size3_t rhs_stride,
            size3_t shape, size_t threads);

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

namespace noa::cpu::signal::fft {
    using Remap = ::noa::fft::Remap;
    using Norm = ::noa::fft::Norm;

    /// Computes the phase cross-correlation map.
    /// \tparam REMAP           Whether the output map should be centered. Should be H2F or H2FC.
    /// \tparam T               float or double.
    /// \param[in] lhs          On the \b host. Left-hand side non-redundant and non-centered FFT argument.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param[in,out] rhs      On the \b host. Right-hand side non-redundant and non-centered FFT argument.
    ///                         Can be overwritten (see \p tmp).
    /// \param rhs_stride       Rightmost stride of \p rhs.
    /// \param[out] output      On the \b host. Cross-correlation map.
    ///                         If REMAP is H2F, the central peak is at indexes {n, 0, 0, 0}.
    ///                         If REMAP is H2FC, the central peal is at indexes {n, shape[1]/2, shape[2]/2, shape[3]/2}.
    /// \param output_stride    Rightmost stride of \p output.
    /// \param shape            Rightmost logical shape.
    /// \param normalize        Whether the normalized cross-correlation should be returned.
    /// \param norm             Normalization mode to use for the C2R transform producing the final output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \param[out] tmp         On the \b host. Buffer that can fit \p shape.fft() complex elements.
    ///                         Can overlap with \p lhs or \p rhs. If nullptr, use \p rhs instead.
    /// \param tmp_stride       Rightmost stride of \p tmp. If \p tmp is nullptr, use \p rhs_stride instead.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xmap_v<REMAP, T>>>
    void xmap(const shared_t<Complex<T>[]>& lhs, size4_t lhs_stride,
              const shared_t<Complex<T>[]>& rhs, size4_t rhs_stride,
              const shared_t<T[]>& output, size4_t output_stride,
              size4_t shape, bool normalize, Norm norm, Stream& stream,
              const shared_t<Complex<T>[]>& tmp = nullptr, size4_t tmp_stride = {});

    /// Find the highest peak in a cross-correlation line.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately to the peak and 2 adjacent points.
    /// \tparam REMAP           Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T               float, double.
    /// \param xmap             On the \b host. 1D cross-correlation map.
    /// \param stride           Rightmost stride of \p map.
    ///                         The innermost dimension should have a non-zero stride.
    /// \param shape            Rightmost shape of \p map.
    /// \param[out] peaks       On the \b host. Rightmost coordinates of the highest peak. One per batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak1D(const shared_t<T[]>& xmap, size4_t stride, size4_t shape,
                 const shared_t<float[]>& peaks, Stream& stream);

    /// Returns the rightmost coordinates of the highest peak in a cross-correlation line.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately to the peak and 2 adjacent points.
    /// \tparam REMAP           Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T               float, double.
    /// \param xmap             On the \b host. Unbatched 1D cross-correlation map.
    /// \param stride           Rightmost stride of \p map.
    ///                         The innermost dimension should have a non-zero stride.
    /// \param shape            Rightmost shape of \p map.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    float xpeak1D(const shared_t<T[]>& xmap, size4_t stride, size4_t shape, Stream& stream);

    /// Find the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP           Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T               float, double.
    /// \param xmap             On the \b host. 2D cross-correlation map.
    /// \param stride           Rightmost stride of \p map.
    ///                         The second-most dimension should have a non-zero stride.
    /// \param shape            Rightmost shape of \p map.
    /// \param[out] peaks       On the \b host. Rightmost coordinates of the highest peak. One per batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak2D(const shared_t<T[]>& xmap, size4_t stride, size4_t shape,
                 const shared_t<float2_t[]>& peaks, Stream& stream);

    /// Returns the rightmost coordinates of the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP           Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T               float, double.
    /// \param xmap             On the \b host. Unbatched 2D cross-correlation map.
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
    /// \param xmap             On the \b host. Cross-correlation map.
    /// \param stride           Rightmost stride of \p map.
    ///                         The third-most and second-most dimension should have a non-zero stride.
    /// \param shape            Rightmost shape of \p map.
    /// \param[out] peaks       On the \b host. Rightmost coordinates of the highest peak. One per batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak3D(const shared_t<T[]>& xmap, size4_t stride, size4_t shape,
                 const shared_t<float3_t[]>& peaks, Stream& stream);

    /// Returns the rightmost coordinates of the highest peak in a cross-correlation map.
    /// \details The highest value of the map is found. Then the sub-pixel position is determined
    ///          by fitting a parabola separately in each dimension to the peak and 2 adjacent points.
    /// \tparam REMAP           Whether \p xmap is centered. Should be F2F or FC2FC.
    /// \tparam T               float, double.
    /// \param xmap             On the \b host. Unbatched cross-correlation map.
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
    /// \param[in] lhs          On the \b host. Left-hand side FFT.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param[in,out] rhs      On the \b host. Right-hand side FFT.
    /// \param rhs_stride       Rightmost stride of \p rhs.
    /// \param shape            Rightmost logical shape.
    /// \param[out] coeffs      On the \b host. Cross-correlation coefficient(s). One per batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, T>>>
    void xcorr(const shared_t<Complex<T>[]>& lhs, size4_t lhs_stride,
               const shared_t<Complex<T>[]>& rhs, size4_t rhs_stride,
               size4_t shape, const shared_t<T[]>& coeffs, Stream& stream) {
        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        stream.enqueue([=]() {
            const size_t batches = shape[0];
            const size3_t input_shape{shape[1], shape[2], SRC_IS_HALF ? shape[3] : shape[3] / 2 + 1};
            const size3_t lhs_stride_{lhs_stride.get() + 1};
            const size3_t rhs_stride_{rhs_stride.get() + 1};
            const size_t threads = stream.threads();

            for (size_t batch = 0; batch < batches; ++batch) {
                const Complex<T>* lhs_ptr = lhs.get() + batch * lhs_stride[0];
                const Complex<T>* rhs_ptr = rhs.get() + batch * rhs_stride[0];
                T* coefficient = coeffs.get() + batch;
                *coefficient = details::xcorr(lhs_ptr, lhs_stride_, rhs_ptr, rhs_stride_,
                                              input_shape, stream.threads());
            }
        });
    }

    /// Computes the cross-correlation coefficient.
    /// \tparam REMAP           Layout of \p lhs and \p rhs. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam T               cfloat_t or cdouble_t.
    /// \param[in] lhs          On the \b host. Left-hand side non-redundant FFT.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param[in,out] rhs      On the \b host. Right-hand side non-redundant FFT.
    /// \param rhs_stride       Rightmost stride of \p rhs.
    /// \param shape            Rightmost logical shape. Should be unbatched.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, T>>>
    T xcorr(const shared_t<Complex<T>[]>& lhs, size4_t lhs_stride,
            const shared_t<Complex<T>[]>& rhs, size4_t rhs_stride,
            size4_t shape, Stream& stream) {
        NOA_ASSERT(shape[0] == 1);
        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        const size3_t input_shape{shape[1], shape[2], SRC_IS_HALF ? shape[3] : shape[3] / 2 + 1};
        const size3_t lhs_stride_{lhs_stride.get() + 1};
        const size3_t rhs_stride_{rhs_stride.get() + 1};
        stream.synchronize();
        return details::xcorr(lhs.get(), lhs_stride_, rhs.get(), rhs_stride_, input_shape, stream.threads());
    }
}
