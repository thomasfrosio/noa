#pragma once

#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::signal::fft::details {
    using Remap = ::noa::fft::Remap;

    template<typename T>
    void xcorr(const shared_t<Complex<T>[]>& lhs, dim4_t lhs_strides,
               const shared_t<Complex<T>[]>& rhs, dim4_t rhs_strides,
               dim4_t shape, const shared_t<T[]>& coefficients,
               Stream& stream, bool is_half);

    template<typename T>
    T xcorr(const shared_t<Complex<T>[]>& lhs, dim4_t lhs_strides,
            const shared_t<Complex<T>[]>& rhs, dim4_t rhs_strides,
            dim4_t shape, Stream& stream, bool is_half);

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
    void xmap(const shared_t<Complex<T>[]>& lhs, dim4_t lhs_strides,
              const shared_t<Complex<T>[]>& rhs, dim4_t rhs_strides,
              const shared_t<T[]>& output, dim4_t output_strides,
              dim4_t shape, bool normalize, Norm norm, Stream& stream,
              const shared_t<Complex<T>[]>& tmp = nullptr, dim4_t tmp_strides = {});

    // Find the highest peak in a cross-correlation line.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak1D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, const shared_t<float[]>& peaks,
                 float max_radius, int64_t registration_radius, Stream& stream);

    // Returns the coordinates of the highest peak in a cross-correlation line.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    float xpeak1D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                  float max_radius, int64_t registration_radius, Stream& stream);

    // Find the highest peak in a cross-correlation map.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak2D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, const shared_t<float2_t[]>& peaks,
                 float2_t max_radius, long2_t registration_radius, Stream& stream);

    // Returns the HW coordinates of the highest peak in a cross-correlation map.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    float2_t xpeak2D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                     float2_t max_radius, long2_t registration_radius, Stream& stream);

    // Find the highest peak in a cross-correlation map.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    void xpeak3D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, const shared_t<float3_t[]>& peaks,
                 float3_t max_radius, long3_t registration_radius, Stream& stream);

    // Returns the DHW coordinates of the highest peak in a cross-correlation map.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xpeak_v<REMAP, T>>>
    float3_t xpeak3D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                     float3_t max_radius, long3_t registration_radius, Stream& stream);

    // Computes the cross-correlation coefficient(s).
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, T>>>
    void xcorr(const shared_t<Complex<T>[]>& lhs, dim4_t lhs_strides,
               const shared_t<Complex<T>[]>& rhs, dim4_t rhs_strides,
               dim4_t shape, const shared_t<T[]>& coeffs,
               Stream& stream) {
        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        details::xcorr(lhs, lhs_strides, rhs, rhs_strides, shape, coeffs, stream, SRC_IS_HALF);
    }

    // Computes the cross-correlation coefficient.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, T>>>
    T xcorr(const shared_t<Complex<T>[]>& lhs, dim4_t lhs_strides,
            const shared_t<Complex<T>[]>& rhs, dim4_t rhs_strides,
            dim4_t shape, Stream& stream) {
        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        return details::xcorr(lhs, lhs_strides, rhs, rhs_strides, shape, stream, SRC_IS_HALF);
    }
}
