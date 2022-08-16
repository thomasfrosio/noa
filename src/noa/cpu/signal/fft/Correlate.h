#pragma once

#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::signal::fft::details {
    using Remap = ::noa::fft::Remap;

    template<typename T>
    T xcorr(const Complex<T>* lhs, size3_t lhs_strides,
            const Complex<T>* rhs, size3_t rhs_strides,
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

    // Returns the coordinates of the highest peak in a cross-correlation line.
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
               size4_t shape, const shared_t<T[]>& coeffs, Stream& stream) {
        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        stream.enqueue([=]() {
            const size_t batches = shape[0];
            const size3_t input_shape(shape[1], shape[2], SRC_IS_HALF ? shape[3] : shape[3] / 2 + 1);
            const size3_t lhs_strides_(lhs_strides.get(1));
            const size3_t rhs_strides_(rhs_strides.get(1));
            const size_t threads = stream.threads();

            for (size_t batch = 0; batch < batches; ++batch) {
                const Complex<T>* lhs_ptr = lhs.get() + batch * lhs_strides[0];
                const Complex<T>* rhs_ptr = rhs.get() + batch * rhs_strides[0];
                T* coefficient = coeffs.get() + batch;
                *coefficient = details::xcorr(lhs_ptr, lhs_strides_, rhs_ptr, rhs_strides_,
                                              input_shape, threads);
            }
        });
    }

    // Computes the cross-correlation coefficient.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_xcorr_v<REMAP, T>>>
    T xcorr(const shared_t<Complex<T>[]>& lhs, size4_t lhs_strides,
            const shared_t<Complex<T>[]>& rhs, size4_t rhs_strides,
            size4_t shape, Stream& stream) {
        NOA_ASSERT(shape[0] == 1);
        constexpr bool SRC_IS_HALF = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::SRC_HALF;
        const size3_t input_shape(shape[1], shape[2], SRC_IS_HALF ? shape[3] : shape[3] / 2 + 1);
        const size3_t lhs_strides_(lhs_strides.get(1));
        const size3_t rhs_strides_(rhs_strides.get(1));
        stream.synchronize();
        return details::xcorr(lhs.get(), lhs_strides_, rhs.get(), rhs_strides_, input_shape, stream.threads());
    }
}
