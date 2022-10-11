#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::geometry::fft::details {
    using Remap = noa::fft::Remap;
    template<Remap REMAP, typename T, typename S, typename R>
    constexpr bool is_valid_insert_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<S, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<R, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::H2H || REMAP == Remap::H2HC ||
             REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename T, typename S, typename R>
    constexpr bool is_valid_extract_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<S, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<R, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename T, typename S, typename R>
    constexpr bool is_valid_extract_texture_v =
            is_valid_extract_v<REMAP, T, S, R> &&
            !traits::is_any_v<T, double, cdouble_t>;
}

namespace noa::cuda::geometry::fft {
    using Remap = noa::fft::Remap;

    // Inserts 2D Fourier slice(s) into a 3D Fourier volume, using tri-linear interpolation.
    template<Remap REMAP, typename T, typename S, typename R,
             typename = std::enable_if_t<details::is_valid_insert_v<REMAP, T, S, R>>>
    void insert3D(const shared_t<T[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<T[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const S& scaling_factors, const R& rotations,
                  float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream);

    // Extracts 2D Fourier slice(s) from a Fourier volume using tri-linear interpolation.
    template<Remap REMAP, typename T, typename S, typename R,
             typename = std::enable_if_t<details::is_valid_extract_v<REMAP, T, S, R>>>
    void extract3D(const shared_t<T[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                   const shared_t<T[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                   const S& scaling_factors, const R& rotations,
                   float cutoff, dim4_t target_shape, float2_t ews_radius, bool use_texture, Stream& stream);

    // Extracts 2D Fourier slice(s) from a Fourier volume using tri-linear interpolation.
    template<Remap REMAP, typename T, typename S, typename R,
             typename = std::enable_if_t<details::is_valid_extract_texture_v<REMAP, T, S, R>>>
    void extract3D(const shared_t<cudaArray>& array,
                   const shared_t<cudaTextureObject_t>& grid, int3_t grid_shape,
                   const shared_t<T[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                   const S& scaling_factors, const R& rotations,
                   float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream);

    // Corrects for the gridding, assuming tri-linear interpolation is used during the insertion or extraction.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    void griddingCorrection(const shared_t<T[]>& input, dim4_t input_strides,
                            const shared_t<T[]>& output, dim4_t output_strides,
                            dim4_t shape, bool post_correction, Stream& stream);
}
