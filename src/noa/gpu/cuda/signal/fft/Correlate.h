#pragma once

#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::signal::fft::details {
    using Remap = ::noa::fft::Remap;

    template<typename T>
    void xcorr(const shared_t<Complex<T>[]>& lhs, size4_t lhs_strides,
               const shared_t<Complex<T>[]>& rhs, size4_t rhs_strides,
               size4_t shape, const shared_t<T[]>& coefficients,
               Stream& stream, bool is_half);

    template<typename T>
    T xcorr(const shared_t<Complex<T>[]>& lhs, size4_t lhs_strides,
            const shared_t<Complex<T>[]>& rhs, size4_t rhs_strides,
            size4_t shape, Stream& stream, bool is_half);

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

    // Computes the phase cross-correlation map.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xmap_v<REMAP, T>>>
    void xmap(const shared_t<Complex<T>[]>& lhs, size4_t lhs_strides,
              const shared_t<Complex<T>[]>& rhs, size4_t rhs_strides,
              const shared_t<T[]>& output, size4_t output_strides,
              size4_t shape, bool normalize, Norm norm, Stream& stream,
              const shared_t<Complex<T>[]>& tmp = nullptr, size4_t tmp_strides = {});

    // Find the highest peak in a cross-correlation line.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak1D(const shared_t<T[]>& xmap, size4_t strides, size4_t shape,
                 const shared_t<float[]>& peaks, Stream& stream);

    // Returns the coordinates of the highest peak in a cross-correlation map.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    float xpeak1D(const shared_t<T[]>& xmap, size4_t strides, size4_t shape, Stream& stream);

    // Find the highest peak in a cross-correlation map.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak2D(const shared_t<T[]>& xmap, size4_t strides, size4_t shape,
                 const shared_t<float2_t[]>& peaks, Stream& stream);

    // Returns the HW coordinates of the highest peak in a cross-correlation map.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    float2_t xpeak2D(const shared_t<T[]>& xmap, size4_t strides, size4_t shape, Stream& stream);

    // Find the highest peak in a cross-correlation map.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak3D(const shared_t<T[]>& xmap, size4_t strides, size4_t shape,
                 const shared_t<float3_t[]>& peaks, Stream& stream);

    // Returns the DHW coordinates of the highest peak in a cross-correlation map.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    float3_t xpeak3D(const shared_t<T[]>& xmap, size4_t strides, size4_t shape, Stream& stream);

    // Computes the cross-correlation coefficient(s).
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, T>>>
    void xcorr(const shared_t<Complex<T>[]>& lhs, size4_t lhs_strides,
               const shared_t<Complex<T>[]>& rhs, size4_t rhs_strides,
               size4_t shape, const shared_t<T[]>& coeffs,
               Stream& stream) {
        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        details::xcorr(lhs, lhs_strides, rhs, rhs_strides, shape, coeffs, stream, SRC_IS_HALF);
    }

    // Computes the cross-correlation coefficient.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, T>>>
    T xcorr(const shared_t<Complex<T>[]>& lhs, size4_t lhs_strides,
            const shared_t<Complex<T>[]>& rhs, size4_t rhs_strides,
            size4_t shape, Stream& stream) {
        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        return details::xcorr(lhs, lhs_strides, rhs, rhs_strides, shape, stream, SRC_IS_HALF);
    }
}
