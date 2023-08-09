#pragma once

#include "noa/core/Types.hpp"

namespace noa::cpu::geometry::fft::details {
    using Remap = noa::fft::Remap;
    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_rasterize_v =
            nt::is_any_v<Value, f32, f64, c32, c64> &&
            nt::is_any_v<Scale, const Float22*, Float22> &&
            nt::is_any_v<Rotate, const Float33*, Float33> &&
            (REMAP == Remap::H2H || REMAP == Remap::H2HC ||
             REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_interpolate_v =
            nt::is_any_v<Value, f32, f64, c32, c64> &&
            nt::is_any_v<Scale, const Float22*, Float22> &&
            nt::is_any_v<Rotate, const Float33*, Float33> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_extract_v =
            nt::is_any_v<Value, f32, f64, c32, c64> &&
            nt::is_any_v<Scale, const Float22*, Float22> &&
            nt::is_any_v<Rotate, const Float33*, Float33> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1>
    constexpr bool is_valid_insert_insert_extract_v =
            nt::is_any_v<Value, f32, f64, c32, c64> &&
            nt::is_any_v<Scale0, const Float22*, Float22> &&
            nt::is_any_v<Scale1, const Float22*, Float22> &&
            nt::is_any_v<Rotate0, const Float33*, Float33> &&
            nt::is_any_v<Rotate1, const Float33*, Float33> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);
}

namespace noa::cpu::geometry::fft {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_rasterize_v<REMAP, Value, Scale, Rotate>>>
    void insert_rasterize_3d(
            const Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
            Value* grid, const Strides4<i64>& grid_strides, const Shape4<i64>& grid_shape,
            const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, i64 threads);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_rasterize_v<REMAP, Value, Scale, Rotate>>>
    void insert_rasterize_3d(
            Value slice, const Shape4<i64>& slice_shape,
            Value* grid, const Strides4<i64>& grid_strides, const Shape4<i64>& grid_shape,
            const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, i64 threads);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_interpolate_v<REMAP, Value, Scale, Rotate>>>
    void insert_interpolate_3d(
            const Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
            Value* grid, const Strides4<i64>& grid_strides, const Shape4<i64>& grid_shape,
            const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
            f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius,
            f32 slice_z_radius, i64 threads);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_interpolate_v<REMAP, Value, Scale, Rotate>>>
    void insert_interpolate_3d(
            Value slice, const Shape4<i64>& slice_shape,
            Value* grid, const Strides4<i64>& grid_strides, const Shape4<i64>& grid_shape,
            const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
            f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius,
            f32 slice_z_radius, i64 threads);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_extract_v<REMAP, Value, Scale, Rotate>>>
    void extract_3d(const Value* grid, const Strides4<i64>& grid_strides, const Shape4<i64>& grid_shape,
                    Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
                    const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                    f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, i64 threads);

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1,
             typename = std::enable_if_t<details::is_valid_insert_insert_extract_v<
                     REMAP, Value, Scale0, Scale1, Rotate0, Rotate1>>>
    void insert_interpolate_and_extract_3d(
            const Value* input_slice, const Strides4<i64>& input_slice_strides, const Shape4<i64>& input_slice_shape,
            Value* output_slice, const Strides4<i64>& output_slice_strides, const Shape4<i64>& output_slice_shape,
            const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
            const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
            f32 cutoff, const Vec2<f32>& ews_radius, f32 slice_z_radius, bool add_to_output, i64 threads);

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1,
             typename = std::enable_if_t<details::is_valid_insert_insert_extract_v<
                      REMAP, Value, Scale0, Scale1, Rotate0, Rotate1>>>
    void insert_interpolate_and_extract_3d(
            Value input_slice, const Shape4<i64>& input_slice_shape,
            Value* output_slice, const Strides4<i64>& output_slice_strides, const Shape4<i64>& output_slice_shape,
            const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
            const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
            f32 cutoff, const Vec2<f32>& ews_radius, f32 slice_z_radius, bool add_to_output, i64 threads);

    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, f32, f64>>>
    void gridding_correction(const Value* input, const Strides4<i64>& input_strides,
                             Value* output, const Strides4<i64>& output_strides,
                             const Shape4<i64>& shape, bool post_correction, i64 threads);
}
