#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::geometry::fft {
    using Remap = noa::fft::Remap;

    template<noa::fft::Remap REMAP,
             typename InputValue, typename InputWeight,
             typename OutputValue, typename OutputWeight,
             typename Scale, typename Rotate>
    void insert_rasterize_3d(
            InputValue input_slice, const Strides4<i64>& input_slice_strides,
            InputWeight input_weight, const Strides4<i64>& input_weight_strides,
            const Shape4<i64>& input_slice_shape,
            OutputValue output_volume, const Strides4<i64>& output_volume_strides,
            OutputWeight output_weight, const Strides4<i64>& output_weight_strides,
            const Shape4<i64>& output_volume_shape,
            const Scale& inv_scaling, const Rotate& fwd_rotation,
            f32 fftfreq_cutoff, const Shape4<i64>& target_shape,
            const Vec2<f32>& ews_radius, Stream& stream
    );

    template<noa::fft::Remap REMAP,
             typename InputValue, typename InputWeight,
             typename OutputValue, typename OutputWeight,
             typename Scale, typename Rotate>
    void insert_interpolate_3d(
            InputValue input_slice, const Strides4<i64>& input_slice_strides,
            InputWeight input_weight, const Strides4<i64>& input_weight_strides,
            const Shape4<i64>& input_slice_shape,
            OutputValue output_volume, const Strides4<i64>& output_volume_strides,
            OutputWeight output_weight, const Strides4<i64>& output_weight_strides,
            const Shape4<i64>& output_volume_shape,
            const Scale& fwd_scaling, const Rotate& inv_rotation,
            f32 fftfreq_sinc, f32 fftfreq_blackman, f32 fftfreq_cutoff,
            const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, Stream& stream
    );

    template<noa::fft::Remap REMAP,
             typename InputValue, typename InputWeight,
             typename OutputValue, typename OutputWeight,
             typename Scale, typename Rotate>
    void extract_3d(
            InputValue input_volume, const Strides4<i64>& input_volume_strides,
            InputWeight input_weight, const Strides4<i64>& input_weight_strides,
            const Shape4<i64>& input_volume_shape,
            OutputValue output_slice, const Strides4<i64>& output_slice_strides,
            OutputWeight output_weight, const Strides4<i64>& output_weight_strides,
            const Shape4<i64>& output_slice_shape,
            const Scale& inv_scaling, const Rotate& fwd_rotation,
            f32 fftfreq_z_sinc, f32 fftfreq_z_blackman, f32 fftfreq_cutoff,
            const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, Stream& stream
    );

    template<noa::fft::Remap REMAP,
             typename InputValue, typename InputWeight,
             typename OutputValue, typename OutputWeight,
             typename Scale0, typename Scale1,
             typename Rotate0, typename Rotate1>
    void insert_interpolate_and_extract_3d(
            InputValue input_slice, const Strides4<i64>& input_slice_strides,
            InputWeight input_weight, const Strides4<i64>& input_weight_strides,
            const Shape4<i64>& input_slice_shape,
            OutputValue output_slice, const Strides4<i64>& output_slice_strides,
            OutputWeight output_weight, const Strides4<i64>& output_weight_strides,
            const Shape4<i64>& output_slice_shape,
            const Scale0& insert_fwd_scaling, const Rotate0& insert_inv_rotation,
            const Scale1& extract_inv_scaling, const Rotate1& extract_fwd_rotation,
            f32 fftfreq_input_sinc, f32 fftfreq_input_blackman,
            f32 fftfreq_z_sinc, f32 fftfreq_z_blackman,
            f32 fftfreq_cutoff, bool add_to_output, bool correct_multiplicity,
            const Vec2<f32>& ews_radius, Stream& stream
    );

    template<typename Value>
    void gridding_correction(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, bool post_correction, Stream& stream
    );
}
