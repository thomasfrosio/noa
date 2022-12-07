#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::geometry::fft::details {
    using Remap = noa::fft::Remap;
    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_v =
            traits::is_any_v<Value, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<Scale, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<Rotate, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::H2H || REMAP == Remap::H2HC ||
             REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_thick_v =
            traits::is_any_v<Value, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<Scale, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<Rotate, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_thick_texture_v =
            traits::is_any_v<Value, float, cfloat_t> &&
            traits::is_any_v<Scale, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<Rotate, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_extract_v =
            traits::is_any_v<Value, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<Scale, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<Rotate, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_extract_texture_v =
            is_valid_extract_v<REMAP, Value, Scale, Rotate> &&
            !traits::is_any_v<Value, double, cdouble_t>;

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1>
    constexpr bool is_valid_insert_insert_extract_v =
            traits::is_any_v<Value, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<Scale0, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<Scale1, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<Rotate0, shared_t<float33_t[]>, float33_t> &&
            traits::is_any_v<Rotate1, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1>
    constexpr bool is_valid_insert_insert_extract_texture_v =
            traits::is_any_v<Value, float, cfloat_t> &&
            traits::is_any_v<Scale0, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<Scale1, shared_t<float22_t[]>, float22_t> &&
            traits::is_any_v<Rotate0, shared_t<float33_t[]>, float33_t> &&
            traits::is_any_v<Rotate1, shared_t<float33_t[]>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);
}

namespace noa::cuda::geometry::fft {
    using Remap = noa::fft::Remap;

    // Inserts 2D central slice(s) into a 3D Fourier volume, using tri-linear rasterization.
    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_v<REMAP, Value, Scale, Rotate>>>
    void insert3D(const shared_t<Value[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream);

    // Inserts 2D central slice(s) into a 3D Fourier volume, using tri-linear rasterization.
    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_v<REMAP, Value, Scale, Rotate>>>
    void insert3D(Value slice, dim4_t slice_shape,
                  const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream);

    // Inserts 2D central slice(s) into a 3D Fourier volume, using tri-linear interpolation.
    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_thick_v<REMAP, Value, Scale, Rotate>>>
    void insert3D(const shared_t<Value[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius,
                  float slice_z_radius, Stream& stream);

    // Inserts 2D central slice(s) into a 3D Fourier volume, using tri-linear interpolation.
    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_thick_v<REMAP, Value, Scale, Rotate>>>
    void insert3D(Value slice, dim4_t slice_shape,
                  const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius,
                  float slice_z_radius, Stream& stream);

    // Inserts 2D central slice(s) into a 3D Fourier volume, using tri-linear interpolation.
    // Uses a 2D texture, with INTERP_LINEAR or INTERP_LINEAR_FAST mode. Can be layered.
    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_thick_texture_v<REMAP, Value, Scale, Rotate>>>
    void insert3D(const shared_t<cudaArray>& array,
                  const shared_t<cudaTextureObject_t>& slice, InterpMode slice_interpolation_mode, dim4_t slice_shape,
                  const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius, float slice_z_radius, Stream& stream);

    // Extracts 2D central slice(s) from a Fourier volume using tri-linear interpolation.
    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_extract_v<REMAP, Value, Scale, Rotate>>>
    void extract3D(const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                   const shared_t<Value[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                   const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                   float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream);

    // Extracts 2D central slice(s) from a Fourier volume using tri-linear interpolation.
    // Uses a 3D texture, with INTERP_LINEAR or INTERP_LINEAR_FAST mode.
    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_extract_texture_v<REMAP, Value, Scale, Rotate>>>
    void extract3D(const shared_t<cudaArray>& array,
                   const shared_t<cudaTextureObject_t>& grid, InterpMode slice_interpolation_mode, dim4_t grid_shape,
                   const shared_t<Value[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                   const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                   float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream);

    // Inserts 2D central slice(s) into a "virtual" 3D Fourier volume and immediately extracts 2D central slices.
    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1,
             typename = std::enable_if_t<details::is_valid_insert_insert_extract_v<
                     REMAP, Value, Scale0, Scale1, Rotate0, Rotate1>>>
    void extract3D(const shared_t<Value[]>& input_slice, dim4_t input_slice_strides, dim4_t input_slice_shape,
                   const shared_t<Value[]>& output_slice, dim4_t output_slice_strides, dim4_t output_slice_shape,
                   const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
                   const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
                   float cutoff, float2_t ews_radius, float slice_z_radius, Stream& stream);

    // Inserts 2D central slice(s) into a "virtual" 3D Fourier volume and immediately extracts 2D central slices.
    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1,
             typename = std::enable_if_t<details::is_valid_insert_insert_extract_v<
                     REMAP, Value, Scale0, Scale1, Rotate0, Rotate1>>>
    void extract3D(Value input_slice, dim4_t input_slice_shape,
                   const shared_t<Value[]>& output_slice, dim4_t output_slice_strides, dim4_t output_slice_shape,
                   const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
                   const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
                   float cutoff, float2_t ews_radius, float slice_z_radius, Stream& stream);

    // Inserts 2D central slice(s) into a "virtual" 3D Fourier volume and immediately extracts 2D central slices.
    // Uses a 2D texture, with INTERP_LINEAR or INTERP_LINEAR_FAST mode. Can be layered.
    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1,
             typename = std::enable_if_t<details::is_valid_insert_insert_extract_texture_v<
                     REMAP, Value, Scale0, Scale1, Rotate0, Rotate1>>>
    void extract3D(const shared_t<cudaArray>& input_slice_array,
                   const shared_t<cudaTextureObject_t>& input_slice_texture,
                   InterpMode input_slice_interpolation_mode, dim4_t input_slice_shape,
                   const shared_t<Value[]>& output_slice, dim4_t output_slice_strides, dim4_t output_slice_shape,
                   const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
                   const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
                   float cutoff, float2_t ews_radius, float slice_z_radius, Stream& stream);

    // Corrects for the gridding, assuming tri-linear interpolation is used during the insertion or extraction.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, double>>>
    void griddingCorrection(const shared_t<Value[]>& input, dim4_t input_strides,
                            const shared_t<Value[]>& output, dim4_t output_strides,
                            dim4_t shape, bool post_correction, Stream& stream);
}
