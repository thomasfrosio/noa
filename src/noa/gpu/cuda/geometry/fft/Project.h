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
    constexpr bool is_valid_insert_thick_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<S, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<R, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename T, typename S, typename R>
    constexpr bool is_valid_insert_thick_texture_v =
            traits::is_any_v<T, float, cfloat_t> &&
            traits::is_any_v<S, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<R, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

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

    template<Remap REMAP, typename T, typename S0, typename S1, typename R0, typename R1>
    constexpr bool is_valid_insert_insert_extract_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<S0, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<S1, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<R0, shared_t<float33_t[]>, float33_t> &&
            traits::is_any_v<R1, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename T, typename S0, typename S1, typename R0, typename R1>
    constexpr bool is_valid_insert_insert_extract_texture_v =
            traits::is_any_v<T, float, cfloat_t> &&
            traits::is_any_v<S0, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<S1, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<R0, shared_t<float33_t[]>, float33_t> &&
            traits::is_any_v<R1, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);
}

namespace noa::cuda::geometry::fft {
    using Remap = noa::fft::Remap;

    // Inserts 2D central slice(s) into a 3D Fourier volume, using tri-linear interpolation.
    template<Remap REMAP, typename data_t, typename scale_t, typename rotate_t,
             typename = std::enable_if_t<details::is_valid_insert_v<REMAP, data_t, scale_t, rotate_t>>>
    void insert3D(const shared_t<data_t[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<data_t[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const scale_t& scaling_factors, const rotate_t& rotations,
                  float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream);

    // Inserts 2D central slice(s) into a 3D Fourier volume, using tri-linear interpolation.
    template<Remap REMAP, typename data_t, typename scale_t, typename rotate_t,
             typename = std::enable_if_t<details::is_valid_insert_thick_v<REMAP, data_t, scale_t, rotate_t>>>
    void insert3D(const shared_t<data_t[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<data_t[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const scale_t& scaling_matrices, const rotate_t& rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius,
                  float slice_z_radius, bool, Stream& stream);

    // Inserts 2D central slice(s) into a 3D Fourier volume, using tri-linear interpolation.
    // Uses a 2D LAYERED texture, with INTERP_LINEAR or INTERP_LINEAR_FAST mode.
    template<Remap REMAP, typename data_t, typename scale_t, typename rotate_t,
             typename = std::enable_if_t<details::is_valid_insert_thick_texture_v<REMAP, data_t, scale_t, rotate_t>>>
    void insert3D(const shared_t<cudaArray>& array,
                  const shared_t<cudaTextureObject_t>& slice, InterpMode slice_interpolation_mode, dim4_t slice_shape,
                  const shared_t<data_t[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const scale_t& scaling_matrices, const rotate_t& rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius, float slice_z_radius, Stream& stream);

    // Extracts 2D central slice(s) from a Fourier volume using tri-linear interpolation.
    template<Remap REMAP, typename data_t, typename scale_t, typename rotate_t,
             typename = std::enable_if_t<details::is_valid_extract_v<REMAP, data_t, scale_t, rotate_t>>>
    void extract3D(const shared_t<data_t[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                   const shared_t<data_t[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                   const scale_t& scaling_factors, const rotate_t& rotations,
                   float cutoff, dim4_t target_shape, float2_t ews_radius, bool use_texture, Stream& stream);

    // Extracts 2D central slice(s) from a Fourier volume using tri-linear interpolation.
    // Uses a 3D texture, with INTERP_LINEAR or INTERP_LINEAR_FAST mode.
    template<Remap REMAP, typename data_t, typename scale_t, typename rotate_t,
             typename = std::enable_if_t<details::is_valid_extract_texture_v<REMAP, data_t, scale_t, rotate_t>>>
    void extract3D(const shared_t<cudaArray>& array,
                   const shared_t<cudaTextureObject_t>& grid, InterpMode slice_interpolation_mode, dim4_t grid_shape,
                   const shared_t<data_t[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                   const scale_t& scaling_factors, const rotate_t& rotations,
                   float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream);

    // Inserts 2D central slice(s) into a "virtual" 3D Fourier volume and immediately extracts 2D central slices.
    template<Remap REMAP, typename data_t, typename scale0_t, typename scale1_t, typename rotate0_t, typename rotate1_t,
             typename = std::enable_if_t<details::is_valid_insert_insert_extract_v<
                     REMAP, data_t, scale0_t, scale1_t, rotate0_t, rotate1_t>>>
    void extract3D(const shared_t<data_t[]>& input_slice, dim4_t input_slice_strides, dim4_t input_slice_shape,
                   const shared_t<data_t[]>& output_slice, dim4_t output_slice_strides, dim4_t output_slice_shape,
                   const scale0_t& insert_inv_scaling_matrices, const rotate0_t& insert_fwd_rotation_matrices,
                   const scale1_t& extract_inv_scaling_matrices, const rotate1_t& extract_fwd_rotation_matrices,
                   float cutoff, float2_t ews_radius, float slice_z_radius, Stream& stream);

    // Inserts 2D central slice(s) into a "virtual" 3D Fourier volume and immediately extracts 2D central slices.
    // Uses a 2D LAYERED texture, with INTERP_LINEAR or INTERP_LINEAR_FAST mode.
    template<Remap REMAP, typename T, typename scale0_t, typename scale1_t, typename rotate0_t, typename rotate1_t,
             typename = std::enable_if_t<details::is_valid_insert_insert_extract_texture_v<
                     REMAP, T, scale0_t, scale1_t, rotate0_t, rotate1_t>>>
    void extract3D(const shared_t<cudaArray>& input_slice_array,
                   const shared_t<cudaTextureObject_t>& input_slice_texture,
                   InterpMode input_slice_interpolation_mode, dim4_t input_slice_shape,
                   const shared_t<T[]>& output_slice, dim4_t output_slice_strides, dim4_t output_slice_shape,
                   const scale0_t& insert_inv_scaling_matrices, const rotate0_t& insert_fwd_rotation_matrices,
                   const scale1_t& extract_inv_scaling_matrices, const rotate1_t& extract_fwd_rotation_matrices,
                   float cutoff, float2_t ews_radius, float slice_z_radius, Stream& stream);

    // Corrects for the gridding, assuming tri-linear interpolation is used during the insertion or extraction.
    template<typename data_t, typename = std::enable_if_t<traits::is_any_v<data_t, float, double>>>
    void griddingCorrection(const shared_t<data_t[]>& input, dim4_t input_strides,
                            const shared_t<data_t[]>& output, dim4_t output_strides,
                            dim4_t shape, bool post_correction, Stream& stream);
}
