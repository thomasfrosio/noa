#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry::fft::details {
    using Remap = noa::fft::Remap;
    template<Remap REMAP, typename T, typename S, typename R>
    constexpr bool is_valid_insert_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<S, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<R, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::H2H || REMAP == Remap::H2HC ||
             REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename T, typename S, typename R>
    constexpr bool is_valid_insert_thick_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<S, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<R, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename T, typename S, typename R>
    constexpr bool is_valid_extract_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<S, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<R, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename T, typename S0, typename S1, typename R0, typename R1>
    constexpr bool is_valid_insert_insert_extract_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<S0, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<S1, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<R0, shared_t<float33_t[]>, float33_t> &&
            traits::is_any_v<R1, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);
}

namespace noa::cpu::geometry::fft {
    using Remap = noa::fft::Remap;

    // Inserts 2D Fourier slice(s) into a 3D Fourier volume, using tri-linear interpolation.
    template<Remap REMAP, typename T, typename S, typename R,
             typename = std::enable_if_t<details::is_valid_insert_v<REMAP, T, S, R>>>
    void insert3D(const shared_t<T[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<T[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const S& scaling_matrices, const R& rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream);

    template<Remap REMAP, typename T, typename S, typename R,
             typename = std::enable_if_t<details::is_valid_insert_thick_v<REMAP, T, S, R>>>
    void insert3D(const shared_t<T[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<T[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const S& scaling_matrices, const R& rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius, float slice_z_radius, Stream& stream);

    // Extracts 2D Fourier slice(s) from a Fourier volume using tri-linear interpolation.
    template<Remap REMAP, typename T, typename S, typename R,
             typename = std::enable_if_t<details::is_valid_extract_v<REMAP, T, S, R>>>
    void extract3D(const shared_t<T[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                   const shared_t<T[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                   const S& scaling_matrices, const R& rotation_matrices,
                   float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream);

    // Inserts 2D central slice(s) into a "virtual" 3D Fourier volume and immediately extracts 2D central slices.
    template<Remap REMAP, typename T, typename S0, typename S1, typename R0, typename R1,
             typename = std::enable_if_t<details::is_valid_insert_insert_extract_v<REMAP, T, S0, S1, R0, R1>>>
    void extract3D(const shared_t<T[]>& input_slice, dim4_t input_slice_strides, dim4_t input_slice_shape,
                   const shared_t<T[]>& output_slice, dim4_t output_slice_strides, dim4_t output_slice_shape,
                   const S0& insert_inv_scaling_matrices, const R0& insert_fwd_rotation_matrices,
                   const S1& extract_inv_scaling_matrices, const R1& extract_fwd_rotation_matrices,
                   float cutoff, float2_t ews_radius, float slice_z_radius, Stream& stream);

    // Corrects for the gridding, assuming tri-linear interpolation is used during the insertion or extraction.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    void griddingCorrection(const shared_t<T[]>& input, dim4_t input_strides,
                            const shared_t<T[]>& output, dim4_t output_strides,
                            dim4_t shape, bool post_correction, Stream& stream);
}
