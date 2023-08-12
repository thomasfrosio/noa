#include "noa/algorithms/geometry/ProjectionsFFT.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/core/geometry/InterpolatorValue.hpp"
#include "noa/cpu/geometry/fft/Project.hpp"
#include "noa/cpu/memory/Set.hpp"
#include "noa/cpu/utils/Iwise.hpp"

namespace noa::cpu::geometry::fft {
    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert_rasterize_3d(
            const Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
            Value* volume, const Strides4<i64>& volume_strides, const Shape4<i64>& volume_shape,
            const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            f32 fftfreq_cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, i64 threads
    ) {
        const auto slice_accessor = AccessorRestrict<const Value, 3, i64>(slice, slice_strides.filter(0, 2, 3));
        const auto volume_accessor = AccessorRestrict<Value, 3, i64>(volume, volume_strides.filter(1, 2, 3));
        const auto iwise_shape = slice_shape.filter(0, 2, 3).rfft();

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = inv_scaling_matrices != Scale{};

        if (apply_ews || apply_scale) {
            const auto op = na::geometry::fourier_insert_rasterize<REMAP>(
                    slice_accessor, slice_shape, volume_accessor, volume_shape,
                    inv_scaling_matrices, fwd_rotation_matrices,
                    fftfreq_cutoff, target_shape, ews_radius);
            noa::cpu::utils::iwise_3d(iwise_shape, op, threads);
        } else {
            const auto op = na::geometry::fourier_insert_rasterize<REMAP>(
                    slice_accessor, slice_shape, volume_accessor, volume_shape,
                    Empty{}, fwd_rotation_matrices,
                    fftfreq_cutoff, target_shape, Empty{});
            noa::cpu::utils::iwise_3d(iwise_shape, op, threads);
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert_rasterize_3d(
            Value slice, const Shape4<i64>& slice_shape,
            Value* volume, const Strides4<i64>& volume_strides, const Shape4<i64>& volume_shape,
            const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            f32 fftfreq_cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, i64 threads
    ) {
        const auto volume_accessor = AccessorRestrict<Value, 3, i64>(volume, volume_strides.filter(1, 2, 3));
        const auto iwise_shape = slice_shape.filter(0, 2, 3).rfft();

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = inv_scaling_matrices != Scale{};

        if (apply_ews || apply_scale) {
            const auto op = na::geometry::fourier_insert_rasterize<REMAP>(
                    slice, slice_shape, volume_accessor, volume_shape,
                    inv_scaling_matrices, fwd_rotation_matrices,
                    fftfreq_cutoff, target_shape, ews_radius);
            noa::cpu::utils::iwise_3d(iwise_shape, op, threads);
        } else {
            const auto op = na::geometry::fourier_insert_rasterize<REMAP>(
                    slice, slice_shape, volume_accessor, volume_shape,
                    Empty{}, fwd_rotation_matrices,
                    fftfreq_cutoff, target_shape, Empty{});
            noa::cpu::utils::iwise_3d(iwise_shape, op, threads);
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert_interpolate_3d(
            const Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
            Value* volume, const Strides4<i64>& volume_strides, const Shape4<i64>& volume_shape,
            const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
            f32 fftfreq_cutoff, f32 fftfreq_sinc, f32 fftfreq_blackman,
            const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, i64 threads
    ) {
        const auto slice_accessor = AccessorRestrict<const Value, 3, i64>(slice, slice_strides.filter(0, 2, 3));
        const auto volume_accessor = AccessorRestrict<Value, 3, i64>(volume, volume_strides.filter(1, 2, 3));
        const auto iwise_shape = volume_shape.pop_front().rfft();
        const auto slice_interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                slice_accessor, slice_shape.filter(2, 3).rfft(), Value{0});

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = fwd_scaling_matrices != Scale{};

        if (apply_ews || apply_scale) {
            const auto op = na::geometry::fourier_insert_interpolate<REMAP>(
                    slice_interpolator, slice_shape, volume_accessor, volume_shape,
                    fwd_scaling_matrices, inv_rotation_matrices,
                    fftfreq_cutoff, fftfreq_sinc, fftfreq_blackman, target_shape, ews_radius);
            noa::cpu::utils::iwise_3d(iwise_shape, op, threads);
        } else {
            const auto op = na::geometry::fourier_insert_interpolate<REMAP>(
                    slice_interpolator, slice_shape, volume_accessor, volume_shape,
                    Empty{}, inv_rotation_matrices,
                    fftfreq_cutoff, fftfreq_sinc, fftfreq_blackman, target_shape, Empty{});
            noa::cpu::utils::iwise_3d(iwise_shape, op, threads);
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert_interpolate_3d(
            Value slice, const Shape4<i64>& slice_shape,
            Value* volume, const Strides4<i64>& volume_strides, const Shape4<i64>& volume_shape,
            const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
            f32 fftfreq_cutoff, f32 fftfreq_sinc, f32 fftfreq_blackman,
            const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, i64 threads
    ) {
        const auto volume_accessor = AccessorRestrict<Value, 3, i64>(volume, volume_strides.filter(1, 2, 3));
        const auto iwise_shape = volume_shape.pop_front().rfft();
        const auto slice_interpolator = noa::geometry::interpolator_value_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                slice, slice_shape.filter(2, 3).rfft(), Value{0});

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = fwd_scaling_matrices != Scale{};

        if (apply_ews || apply_scale) {
            const auto op = na::geometry::fourier_insert_interpolate<REMAP>(
                    slice_interpolator, slice_shape, volume_accessor, volume_shape,
                    fwd_scaling_matrices, inv_rotation_matrices,
                    fftfreq_cutoff, fftfreq_sinc, fftfreq_blackman,
                    target_shape, ews_radius);
            noa::cpu::utils::iwise_3d(iwise_shape, op, threads);
        } else {
            const auto op = na::geometry::fourier_insert_interpolate<REMAP>(
                    slice_interpolator, slice_shape, volume_accessor, volume_shape,
                    Empty{}, inv_rotation_matrices,
                    fftfreq_cutoff, fftfreq_sinc, fftfreq_blackman,
                    target_shape, Empty{});
            noa::cpu::utils::iwise_3d(iwise_shape, op, threads);
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void extract_3d(
            const Value* volume, const Strides4<i64>& volume_strides, const Shape4<i64>& volume_shape,
            Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
            const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            f32 fftfreq_cutoff, f32 fftfreq_z_sinc, f32 fftfreq_z_blackman,
            const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, i64 threads
    ) {
        const auto slice_accessor = AccessorRestrict<Value, 3, i64>(slice, slice_strides.filter(0, 2, 3));
        const auto volume_accessor = AccessorRestrict<const Value, 3, i64>(volume, volume_strides.filter(1, 2, 3));
        const auto volume_interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::LINEAR>(
                volume_accessor, volume_shape.filter(1, 2, 3).rfft(), Value{0});

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = inv_scaling_matrices != Scale{};

        if (apply_ews || apply_scale) {
            const auto op = na::geometry::fourier_extract<REMAP>(
                    volume_interpolator, volume_shape, slice_accessor, slice_shape,
                    inv_scaling_matrices, fwd_rotation_matrices,
                    fftfreq_cutoff, fftfreq_z_sinc, fftfreq_z_blackman, target_shape, ews_radius);
            if (op.windowed_sinc_size() > 1) {
                noa::cpu::memory::set(slice, slice_strides, slice_shape.rfft(), Value{0}, /*threads=*/ 1);
                noa::cpu::utils::iwise_4d(slice_shape.rfft().set<1>(op.windowed_sinc_size()), op, threads);
            } else {
                noa::cpu::utils::iwise_3d(slice_shape.filter(0, 2, 3).rfft(), op, threads);
            }
        } else {
            const auto op = na::geometry::fourier_extract<REMAP>(
                    volume_interpolator, volume_shape, slice_accessor, slice_shape,
                    Empty{}, fwd_rotation_matrices,
                    fftfreq_cutoff, fftfreq_z_sinc, fftfreq_z_blackman, target_shape, Empty{});
            if (op.windowed_sinc_size() > 1) {
                noa::cpu::memory::set(slice, slice_strides, slice_shape.rfft(), Value{0}, /*threads=*/ 1);
                noa::cpu::utils::iwise_4d(slice_shape.rfft().set<1>(op.windowed_sinc_size()), op, threads);
            } else {
                noa::cpu::utils::iwise_3d(slice_shape.filter(0, 2, 3).rfft(), op, threads);
            }
        }
    }

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1, typename>
    void insert_interpolate_and_extract_3d(
            const Value* input_slice, const Strides4<i64>& input_slice_strides, const Shape4<i64>& input_slice_shape,
            Value* output_slice, const Strides4<i64>& output_slice_strides, const Shape4<i64>& output_slice_shape,
            const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
            const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
            f32 fftfreq_cutoff, f32 fftfreq_input_sinc, f32 fftfreq_input_blackman,
            f32 fftfreq_z_sinc, f32 fftfreq_z_blackman,
            bool add_to_output, const Vec2<f32>& ews_radius, i64 threads
    ) {
        const auto input_slice_accessor = AccessorRestrict<const Value, 3, i64>(input_slice, input_slice_strides.filter(0, 2, 3));
        const auto output_slice_accessor = AccessorRestrict<Value, 3, i64>(output_slice, output_slice_strides.filter(0, 2, 3));
        const auto input_slice_interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                input_slice_accessor, input_slice_shape.filter(2, 3).rfft(), Value{0});

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = insert_fwd_scaling_matrices != Scale0{};

        if (apply_ews || apply_scale) {
            const auto op = na::geometry::fourier_insert_and_extract<REMAP>(
                    input_slice_interpolator, input_slice_shape,
                    output_slice_accessor, output_slice_shape,
                    insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                    extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                    fftfreq_cutoff, fftfreq_input_sinc, fftfreq_input_blackman,
                    fftfreq_z_sinc, fftfreq_z_blackman,
                    add_to_output, ews_radius);
            if (op.windowed_sinc_size() > 1) {
                if (!add_to_output)
                    noa::cpu::memory::set(output_slice, output_slice_strides, output_slice_shape.rfft(), Value{0}, /*threads=*/ 1);
                noa::cpu::utils::iwise_4d(output_slice_shape.rfft().set<1>(op.windowed_sinc_size()), op, threads);
            } else {
                noa::cpu::utils::iwise_3d(output_slice_shape.filter(0, 2, 3).rfft(), op, threads);
            }
        } else {
            const auto op = na::geometry::fourier_insert_and_extract<REMAP>(
                    input_slice_interpolator, input_slice_shape,
                    output_slice_accessor, output_slice_shape,
                    Empty{}, insert_inv_rotation_matrices,
                    extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                    fftfreq_cutoff, fftfreq_input_sinc, fftfreq_input_blackman,
                    fftfreq_z_sinc, fftfreq_z_blackman, add_to_output, Empty{});
            if (op.windowed_sinc_size() > 1) {
                if (!add_to_output)
                    noa::cpu::memory::set(output_slice, output_slice_strides, output_slice_shape.rfft(), Value{0}, /*threads=*/ 1);
                noa::cpu::utils::iwise_4d(output_slice_shape.rfft().set<1>(op.windowed_sinc_size()), op, threads);
            } else {
                noa::cpu::utils::iwise_3d(output_slice_shape.filter(0, 2, 3).rfft(), op, threads);
            }
        }
    }

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1, typename>
    void insert_interpolate_and_extract_3d(
            Value input_slice, const Shape4<i64>& input_slice_shape,
            Value* output_slice, const Strides4<i64>& output_slice_strides, const Shape4<i64>& output_slice_shape,
            const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
            const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
            f32 fftfreq_cutoff, f32 fftfreq_input_sinc, f32 fftfreq_input_blackman,
            f32 fftfreq_z_sinc, f32 fftfreq_z_blackman,
            bool add_to_output, const Vec2<f32>& ews_radius, i64 threads
    ) {
        const auto output_slice_accessor = AccessorRestrict<Value, 3, i64>(output_slice, output_slice_strides.filter(0, 2, 3));
        const auto input_slice_interpolator = noa::geometry::interpolator_value_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                input_slice, input_slice_shape.filter(2, 3).rfft(), Value{0});

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = insert_fwd_scaling_matrices != Scale0{};

        if (apply_ews || apply_scale) {
            const auto op = na::geometry::fourier_insert_and_extract<REMAP>(
                    input_slice_interpolator, input_slice_shape,
                    output_slice_accessor, output_slice_shape,
                    insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                    extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                    fftfreq_cutoff, fftfreq_input_sinc, fftfreq_input_blackman,
                    fftfreq_z_sinc, fftfreq_z_blackman,
                    add_to_output, ews_radius);
            if (op.windowed_sinc_size() > 1) {
                if (!add_to_output)
                    noa::cpu::memory::set(output_slice, output_slice_strides, output_slice_shape.rfft(), Value{0}, /*threads=*/ 1);
                noa::cpu::utils::iwise_4d(output_slice_shape.rfft().set<1>(op.windowed_sinc_size()), op, threads);
            } else {
                noa::cpu::utils::iwise_3d(output_slice_shape.filter(0, 2, 3).rfft(), op, threads);
            }
        } else {
            const auto op = na::geometry::fourier_insert_and_extract<REMAP>(
                    input_slice_interpolator, input_slice_shape,
                    output_slice_accessor, output_slice_shape,
                    Empty{}, insert_inv_rotation_matrices,
                    extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                    fftfreq_cutoff, fftfreq_input_sinc, fftfreq_input_blackman,
                    fftfreq_z_sinc, fftfreq_z_blackman, add_to_output, Empty{});
            if (op.windowed_sinc_size() > 1) {
                if (!add_to_output)
                    noa::cpu::memory::set(output_slice, output_slice_strides, output_slice_shape.rfft(), Value{0}, /*threads=*/ 1);
                noa::cpu::utils::iwise_4d(output_slice_shape.rfft().set<1>(op.windowed_sinc_size()), op, threads);
            } else {
                noa::cpu::utils::iwise_3d(output_slice_shape.filter(0, 2, 3).rfft(), op, threads);
            }
        }
    }

    template<typename Value, typename>
    void gridding_correction(
            const Value* input, const Strides4<i64>& input_strides,
            Value* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, bool post_correction, i64 threads
    ) {
        NOA_ASSERT(input && input && all(shape > 0));
        const auto input_accessor = Accessor<const Value, 4, i64>(input, input_strides);
        const auto output_accessor = Accessor<Value, 4, i64>(output, output_strides);

        if (post_correction) {
            const auto op = na::geometry::gridding_correction<true>(
                    input_accessor, output_accessor, shape);
            noa::cpu::utils::iwise_4d(shape, op, threads);
        } else {
            const auto op = na::geometry::gridding_correction<false>(
                    input_accessor, output_accessor, shape);
            noa::cpu::utils::iwise_4d(shape, op, threads);
        }
    }
    template void gridding_correction<f32, void>(const f32*, const Strides4<i64>&, f32*, const Strides4<i64>&, const Shape4<i64>&, bool, i64);
    template void gridding_correction<f64, void>(const f64*, const Strides4<i64>&, f64*, const Strides4<i64>&, const Shape4<i64>&, bool, i64);

    #define NOA_INSTANTIATE_INSERT_RASTERIZE_(T, REMAP, S, R)                   \
    template void insert_rasterize_3d<REMAP, T, S, R, void>(                    \
        const T*, const Strides4<i64>&, const Shape4<i64>&,                     \
        T*, const Strides4<i64>&, const Shape4<i64>&,                           \
        S const&, R const&, f32, const Shape4<i64>&, const Vec2<f32>&, i64);    \
    template void insert_rasterize_3d<REMAP, T, S, R, void>(                    \
        T, const Shape4<i64>&,                                                  \
        T*, const Strides4<i64>&, const Shape4<i64>&,                           \
        S const&, R const&, f32, const Shape4<i64>&, const Vec2<f32>&, i64)

    #define NOA_INSTANTIATE_INSERT_INTERPOLATE_(T, REMAP, S, R) \
    template void insert_interpolate_3d<REMAP, T, S, R, void>(  \
        const T*, const Strides4<i64>&, const Shape4<i64>&,     \
        T*, const Strides4<i64>&, const Shape4<i64>&,           \
        S const&, R const&, f32, f32, f32,                      \
        const Shape4<i64>&, const Vec2<f32>&, i64);             \
    template void insert_interpolate_3d<REMAP, T, S, R, void>(  \
        T, const Shape4<i64>&,                                  \
        T*, const Strides4<i64>&, const Shape4<i64>&,           \
        S const&, R const&, f32, f32, f32,                      \
        const Shape4<i64>&, const Vec2<f32>&, i64)

    #define NOA_INSTANTIATE_EXTRACT_(T, REMAP, S, R)        \
    template void extract_3d<REMAP, T, S, R, void>(         \
        const T*, const Strides4<i64>&, const Shape4<i64>&, \
        T*, const Strides4<i64>&, const Shape4<i64>&,       \
        S const&, R const&, f32, f32, f32,                  \
        const Shape4<i64>&, const Vec2<f32>&, i64)

    #define NOA_INSTANTIATE_INSERT_EXTRACT_(T, REMAP, S0, S1, R0, R1)                   \
    template void insert_interpolate_and_extract_3d<REMAP, T, S0, S1, R0, R1, void>(    \
        const T*, const Strides4<i64>&, const Shape4<i64>&,                             \
        T*, const Strides4<i64>&, const Shape4<i64>&,                                   \
        S0 const&, R0 const&, S1 const&, R1 const&,                                     \
        f32, f32, f32, f32, f32, bool, const Vec2<f32>&, i64);                          \
    template void insert_interpolate_and_extract_3d<REMAP, T, S0, S1, R0, R1, void>(    \
        T, const Shape4<i64>&,                                                          \
        T*, const Strides4<i64>&, const Shape4<i64>&,                                   \
        S0 const&, R0 const&, S1 const&, R1 const&,                                     \
        f32, f32, f32, f32, f32, bool, const Vec2<f32>&, i64)

    #define NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, S, R)          \
    NOA_INSTANTIATE_INSERT_RASTERIZE_(T, Remap::H2H, S, R);     \
    NOA_INSTANTIATE_INSERT_RASTERIZE_(T, Remap::H2HC, S, R);    \
    NOA_INSTANTIATE_INSERT_RASTERIZE_(T, Remap::HC2H, S, R);    \
    NOA_INSTANTIATE_INSERT_RASTERIZE_(T, Remap::HC2HC, S, R);   \
    NOA_INSTANTIATE_INSERT_INTERPOLATE_(T, Remap::HC2H, S, R);  \
    NOA_INSTANTIATE_INSERT_INTERPOLATE_(T, Remap::HC2HC, S, R); \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2H, S, R);             \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2HC, S, R)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, S0, S1, R0, R1)  \
    NOA_INSTANTIATE_INSERT_EXTRACT_(T, Remap::HC2H, S0, S1, R0, R1);    \
    NOA_INSTANTIATE_INSERT_EXTRACT_(T, Remap::HC2HC, S0, S1, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, R0, R1)                      \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, Float22, Float22, R0, R1);           \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, const Float22*, Float22, R0, R1);    \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, Float22, const Float22*, R0, R1);    \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, const Float22*, const Float22*, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(T)                     \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, Float33, Float33);           \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, const Float33*, Float33);    \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, Float33, const Float33*);    \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, const Float33*, const Float33*)

    #define NOA_INSTANTIATE_PROJECT_ALL_(T)                                 \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, Float22, Float33);                 \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, const Float22*, Float33);          \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, Float22, const Float33*);          \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, const Float22*, const Float33*);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(T)

    NOA_INSTANTIATE_PROJECT_ALL_(f32);
    NOA_INSTANTIATE_PROJECT_ALL_(f64);
    NOA_INSTANTIATE_PROJECT_ALL_(c32);
    NOA_INSTANTIATE_PROJECT_ALL_(c64);
}
