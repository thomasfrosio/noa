#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/algorithms/geometry/ProjectionsFFT.hpp"

namespace noa::cuda::geometry::fft {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename = std::enable_if_t<
             na::geometry::details::is_valid_insert_rasterize_v<REMAP, Value, Scale, Rotate>>>
    void insert_rasterize_3d(
            const Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
            Value* volume, const Strides4<i64>& volume_strides, const Shape4<i64>& volume_shape,
            const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            f32 fftfreq_cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, Stream& stream
    );

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename = std::enable_if_t<
             na::geometry::details::is_valid_insert_rasterize_v<REMAP, Value, Scale, Rotate>>>
    void insert_rasterize_3d(
            Value slice, const Shape4<i64>& slice_shape,
            Value* volume, const Strides4<i64>& volume_strides, const Shape4<i64>& volume_shape,
            const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            f32 fftfreq_cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, Stream& stream
    );

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename = std::enable_if_t<
             na::geometry::details::is_valid_insert_interpolate_v<REMAP, Value, Scale, Rotate>>>
    void insert_interpolate_3d(
            const Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
            Value* volume, const Strides4<i64>& volume_strides, const Shape4<i64>& volume_shape,
            const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
            f32 fftfreq_cutoff, f32 fftfreq_sinc, f32 fftfreq_blackman,
            const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, Stream& stream
    );

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename = std::enable_if_t<
             na::geometry::details::is_valid_insert_interpolate_v<REMAP, Value, Scale, Rotate>>>
    void insert_interpolate_3d(
            Value slice, const Shape4<i64>& slice_shape,
            Value* volume, const Strides4<i64>& volume_strides, const Shape4<i64>& volume_shape,
            const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
            f32 fftfreq_cutoff, f32 fftfreq_sinc, f32 fftfreq_blackman,
            const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, Stream& stream
    );

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename = std::enable_if_t<
             na::geometry::details::is_valid_insert_interpolate_texture_v<REMAP, Value, Scale, Rotate>>>
    void insert_interpolate_3d(
            cudaArray* array, cudaTextureObject_t slice,
            InterpMode slice_interpolation_mode, const Shape4<i64>& slice_shape,
            Value* volume, const Strides4<i64>& volume_strides, const Shape4<i64>& volume_shape,
            const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
            f32 fftfreq_cutoff, f32 fftfreq_sinc, f32 fftfreq_blackman,
            const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, Stream& stream
    );

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename = std::enable_if_t<
             na::geometry::details::is_valid_extract_v<REMAP, Value, Scale, Rotate>>>
    void extract_3d(
            const Value* volume, const Strides4<i64>& volume_strides, const Shape4<i64>& volume_shape,
            Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
            const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            f32 fftfreq_cutoff, f32 fftfreq_z_sinc, f32 fftfreq_z_blackman,
            const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, Stream& stream
    );

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename = std::enable_if_t<
             na::geometry::details::is_valid_extract_texture_v<REMAP, Value, Scale, Rotate>>>
    void extract_3d(
            cudaArray* array, cudaTextureObject_t volume,
            InterpMode volume_interpolation_mode, const Shape4<i64>& volume_shape,
            Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
            const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            f32 fftfreq_cutoff, f32 fftfreq_z_sinc, f32 fftfreq_z_blackman,
            const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, Stream& stream
    );

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1,
             typename = std::enable_if_t<na::geometry::details::is_valid_insert_insert_extract_v<
                     REMAP, Value, Scale0, Scale1, Rotate0, Rotate1>>>
    void insert_interpolate_and_extract_3d(
            const Value* input_slice, const Strides4<i64>& input_slice_strides, const Shape4<i64>& input_slice_shape,
            Value* output_slice, const Strides4<i64>& output_slice_strides, const Shape4<i64>& output_slice_shape,
            const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
            const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
            f32 fftfreq_cutoff, f32 fftfreq_input_sinc, f32 fftfreq_input_blackman,
            f32 fftfreq_z_sinc, f32 fftfreq_z_blackman, bool add_to_output,
            const Vec2<f32>& ews_radius, Stream& stream
    );

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1,
             typename = std::enable_if_t<na::geometry::details::is_valid_insert_insert_extract_v<
                     REMAP, Value, Scale0, Scale1, Rotate0, Rotate1>>>
    void insert_interpolate_and_extract_3d(
            Value input_slice, const Shape4<i64>& input_slice_shape,
            Value* output_slice, const Strides4<i64>& output_slice_strides, const Shape4<i64>& output_slice_shape,
            const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
            const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
            f32 fftfreq_cutoff, f32 fftfreq_input_sinc, f32 fftfreq_input_blackman,
            f32 fftfreq_z_sinc, f32 fftfreq_z_blackman, bool add_to_output,
            const Vec2<f32>& ews_radius, Stream& stream
    );

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1,
             typename = std::enable_if_t<na::geometry::details::is_valid_insert_insert_extract_texture_v<
                     REMAP, Value, Scale0, Scale1, Rotate0, Rotate1>>>
    void insert_interpolate_and_extract_3d(
            cudaArray* input_slice_array, cudaTextureObject_t input_slice_texture,
            InterpMode input_slice_interpolation_mode, const Shape4<i64>& input_slice_shape,
            Value* output_slice, const Strides4<i64>& output_slice_strides, const Shape4<i64>& output_slice_shape,
            const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
            const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
            f32 fftfreq_cutoff, f32 fftfreq_input_sinc, f32 fftfreq_input_blackman,
            f32 fftfreq_z_sinc, f32 fftfreq_z_blackman, bool add_to_output,
            const Vec2<f32>& ews_radius, Stream& stream
    );

    template<typename Value, typename = std::enable_if_t<nt::is_any_v<Value, f32, f64>>>
    void gridding_correction(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, bool post_correction, Stream& stream
    );
}
