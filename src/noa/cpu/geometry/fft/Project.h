#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry::fft::details {
    using Remap = noa::fft::Remap;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_insert_v = traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                       (REMAP == Remap::H2H || REMAP == Remap::H2HC ||
                                        REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename T>
    constexpr bool is_valid_extract_v = traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                        (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);
}

namespace noa::cpu::geometry::fft {
    using Remap = noa::fft::Remap;

    // Inserts 2D Fourier slice(s) into a 3D Fourier volume, using tri-linear interpolation.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_insert_v<REMAP, T>>>
    void insert3D(const shared_t<T[]>& slice, size4_t slice_strides, size4_t slice_shape,
                  const shared_t<T[]>& grid, size4_t grid_strides, size4_t grid_shape,
                  const shared_t<float22_t[]>& scaling_factors,
                  const shared_t<float33_t[]>& rotations,
                  float cutoff, float sampling_factor, float2_t ews_radius, Stream& stream);

    // Extracts 2D Fourier slice(s) from a Fourier volume using tri-linear interpolation.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_extract_v<REMAP, T>>>
    void extract3D(const shared_t<T[]>& grid, size4_t grid_strides, size4_t grid_shape,
                   const shared_t<T[]>& slice, size4_t slice_strides, size4_t slice_shape,
                   const shared_t<float22_t[]>& scaling_factors,
                   const shared_t<float33_t[]>& rotations,
                   float cutoff, float sampling_factor, float2_t ews_radius, Stream& stream);

    // Corrects for the gridding, assuming tri-linear interpolation is used during the insertion or extraction.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    void griddingCorrection(const shared_t<T[]>& input, size4_t input_strides,
                            const shared_t<T[]>& output, size4_t output_strides,
                            size4_t shape, bool post_correction, Stream& stream);
}
