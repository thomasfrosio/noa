#include "noa/core/geometry/Interpolator.hpp"
#include "noa/core/geometry/InterpolatorValue.hpp"
#include "noa/algorithms/geometry/ProjectionsFFT.hpp"

#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/memory/PtrArray.hpp"
#include "noa/gpu/cuda/memory/PtrTexture.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/geometry/fft/Project.h"

namespace {
    using namespace ::noa;

    template<fft::Remap REMAP, typename Interpolator, typename Value, typename Scale, typename Rotate>
    void insert_interpolate_3d_(
            Interpolator slice_interpolator, const Shape4<i64>& slice_shape,
            Value* grid, const Strides4<i64>& grid_strides, const Shape4<i64>& grid_shape,
            const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
            f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius,
            f32 slice_z_radius, cuda::Stream& stream) {

        const auto i_grid_shape = grid_shape.as_safe<i32>();
        const auto grid_strides_3d = grid_strides.filter(1, 2, 3).as_safe<u32>();
        const auto grid_accessor = AccessorRestrict<Value, 3, u32>(grid, grid_strides_3d);
        const auto iwise_shape = i_grid_shape.pop_front().fft();

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = fwd_scaling_matrices != Scale{};

        if (apply_ews || apply_scale) {
            const auto kernel = noa::algorithm::geometry::fourier_insertion_interpolate<REMAP, i32>(
                    slice_interpolator, slice_shape.as_safe<i32>(), grid_accessor, i_grid_shape,
                    fwd_scaling_matrices, inv_rotation_matrices,
                    cutoff, target_shape.as_safe<i32>(), ews_radius, slice_z_radius);
            return cuda::utils::iwise_3d("geometry::fft::insert_interpolate_3d", iwise_shape, kernel, stream);
        } else {
            const auto kernel = noa::algorithm::geometry::fourier_insertion_interpolate<REMAP, i32>(
                    slice_interpolator, slice_shape.as_safe<i32>(), grid_accessor, i_grid_shape,
                    Empty{}, inv_rotation_matrices,
                    cutoff, target_shape.as_safe<i32>(), Empty{}, slice_z_radius);
            return cuda::utils::iwise_3d("geometry::fft::insert_interpolate_3d", iwise_shape, kernel, stream);
        }
    }

    template<fft::Remap REMAP, typename Value, typename Interpolator, typename Scale, typename Rotate>
    void extract_3d_(
            Interpolator grid, const Shape4<i64>& grid_shape,
            Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
            const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, cuda::Stream& stream) {

        const auto i_slice_shape = slice_shape.as_safe<i32>();
        const auto iwise_shape = i_slice_shape.filter(0, 2, 3).fft();
        const auto slice_strides_3d = slice_strides.filter(0, 2, 3).as_safe<u32>();
        const auto slice_accessor = AccessorRestrict<Value, 3, u32>(slice, slice_strides_3d);

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = inv_scaling_matrices != Scale{};

        if (apply_ews || apply_scale) {
            const auto functor = noa::algorithm::geometry::fourier_extraction<REMAP>(
                    grid, grid_shape.as_safe<i32>(), slice_accessor, i_slice_shape,
                    inv_scaling_matrices, fwd_rotation_matrices,
                    cutoff, target_shape.as_safe<i32>(), ews_radius);
            cuda::utils::iwise_3d("geometry::fft::extract_3d", iwise_shape, functor, stream);
        } else {
            const auto functor = noa::algorithm::geometry::fourier_extraction<REMAP>(
                    grid, grid_shape.as_safe<i32>(), slice_accessor, i_slice_shape,
                    Empty{}, fwd_rotation_matrices,
                    cutoff, target_shape.as_safe<i32>(), Empty{});
            cuda::utils::iwise_3d("geometry::fft::extract_3d", iwise_shape, functor, stream);
        }
    }

    template<fft::Remap REMAP, typename Interpolator, typename Value,
             typename Scale0, typename Scale1, typename Rotate0, typename Rotate1>
    void insert_interpolate_and_extract_3d_(
            Interpolator input_slice_interpolator, const Shape4<i64>& input_slice_shape,
            Value* output_slice, const Strides4<i64>& output_slice_strides, const Shape4<i64>& output_slice_shape,
            const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
            const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
            f32 cutoff, const Vec2<f32>& ews_radius, f32 slice_z_radius, cuda::Stream& stream) {

        const auto output_slice_strides_2d = output_slice_strides.filter(0, 2, 3).as_safe<u32>();
        const auto output_slice_accessor = AccessorRestrict<Value, 3, u32>(output_slice, output_slice_strides_2d);
        const auto i_output_slice_shape = output_slice_shape.as_safe<i32>();
        const auto iwise_shape = i_output_slice_shape.filter(0, 2, 3).fft();

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = insert_fwd_scaling_matrices != Scale0{};

        if (apply_ews || apply_scale) {
            const auto functor = noa::algorithm::geometry::fourier_insert_and_extraction<REMAP, i32>(
                    input_slice_interpolator, input_slice_shape.as_safe<i32>(),
                    output_slice_accessor, i_output_slice_shape,
                    insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                    extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                    cutoff, ews_radius, slice_z_radius);
            return noa::cuda::utils::iwise_3d("insert_interpolate_and_extract_3d", iwise_shape, functor, stream);
        } else {
            const auto functor = noa::algorithm::geometry::fourier_insert_and_extraction<REMAP, i32>(
                    input_slice_interpolator, input_slice_shape.as_safe<i32>(),
                    output_slice_accessor, i_output_slice_shape,
                    Empty{}, insert_inv_rotation_matrices,
                    extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                    cutoff, Empty{}, slice_z_radius);
            return noa::cuda::utils::iwise_3d("insert_interpolate_and_extract_3d", iwise_shape, functor, stream);
        }
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert_rasterize_3d(
            const Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
            Value* grid, const Strides4<i64>& grid_strides, const Shape4<i64>& grid_shape,
            const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, Stream& stream) {

        NOA_ASSERT_DEVICE_PTR(slice, stream.device());
        NOA_ASSERT_DEVICE_PTR(grid, stream.device());

        const auto slice_strides_3d = slice_strides.filter(0, 2, 3).as_safe<u32>();
        const auto grid_strides_3d = grid_strides.pop_front().as_safe<u32>();
        const auto slice_accessor = AccessorRestrict<const Value, 3, u32>(slice, slice_strides_3d);
        const auto grid_accessor = AccessorRestrict<Value, 3, u32>(grid, grid_strides_3d);
        const auto i_slice_shape = slice_shape.as_safe<i32>();
        const auto iwise_shape = i_slice_shape.filter(0, 2, 3).fft();

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = inv_scaling_matrices != Scale{};

        if (apply_ews || apply_scale) {
            const auto functor = noa::algorithm::geometry::fourier_insertion_rasterize<REMAP, i32>(
                    slice_accessor, i_slice_shape, grid_accessor, grid_shape.as_safe<i32>(),
                    inv_scaling_matrices, fwd_rotation_matrices,
                    cutoff, target_shape.as_safe<i32>(), ews_radius);
            return noa::cuda::utils::iwise_3d("insert_rasterize_3d", iwise_shape, functor, stream);
        } else {
            const auto functor = noa::algorithm::geometry::fourier_insertion_rasterize<REMAP, i32>(
                    slice_accessor, i_slice_shape, grid_accessor, grid_shape.as_safe<i32>(),
                    Empty{}, fwd_rotation_matrices,
                    cutoff, target_shape.as_safe<i32>(), Empty{});
            return noa::cuda::utils::iwise_3d("insert_rasterize_3d", iwise_shape, functor, stream);
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert_rasterize_3d(
            Value slice, const Shape4<i64>& slice_shape,
            Value* grid, const Strides4<i64>& grid_strides, const Shape4<i64>& grid_shape,
            const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
            f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, Stream& stream) {

        NOA_ASSERT_DEVICE_PTR(grid, stream.device());

        const auto i_slice_shape = slice_shape.as_safe<i32>();
        const auto iwise_shape = i_slice_shape.filter(0, 2, 3).fft();
        const auto grid_accessor = AccessorRestrict<Value, 3, uint32_t>(grid, grid_strides.pop_front().as_safe<u32>());

        const auto apply_ews = noa::any(ews_radius != 0);
        const bool apply_scale = inv_scaling_matrices != Scale{};

        if (apply_ews || apply_scale) {
            const auto functor = noa::algorithm::geometry::fourier_insertion_rasterize<REMAP>(
                    slice, i_slice_shape, grid_accessor, grid_shape.as_safe<i32>(),
                    inv_scaling_matrices, fwd_rotation_matrices,
                    cutoff, target_shape.as_safe<i32>(), ews_radius);
            return noa::cuda::utils::iwise_3d("insert_rasterize_3d", iwise_shape, functor, stream);
        } else {
            const auto functor = noa::algorithm::geometry::fourier_insertion_rasterize<REMAP>(
                    slice, i_slice_shape, grid_accessor, grid_shape.as_safe<i32>(),
                    Empty{}, fwd_rotation_matrices,
                    cutoff, target_shape.as_safe<i32>(), Empty{});
            return noa::cuda::utils::iwise_3d("insert_rasterize_3d", iwise_shape, functor, stream);
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert_interpolate_3d(
            const Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
            Value* grid, const Strides4<i64>& grid_strides, const Shape4<i64>& grid_shape,
            const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
            f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius,
            f32 slice_z_radius, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(grid, stream.device());
        NOA_ASSERT_DEVICE_PTR(slice, stream.device());

        const auto slice_accessor = AccessorRestrict<const Value, 3, u32>(
                slice, slice_strides.filter(0, 2, 3).as_safe<u32>());
        const auto slice_interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                slice_accessor, slice_shape.filter(2, 3).as_safe<i32>().fft(), Value{0});

        insert_interpolate_3d_<REMAP>(
                slice_interpolator, slice_shape, grid, grid_strides, grid_shape,
                fwd_scaling_matrices, inv_rotation_matrices, cutoff, target_shape, ews_radius,
                slice_z_radius, stream);
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert_interpolate_3d(
            Value slice, const Shape4<i64>& slice_shape,
            Value* grid, const Strides4<i64>& grid_strides, const Shape4<i64>& grid_shape,
            const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
            f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius,
            f32 slice_z_radius, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(grid, stream.device());

        const auto slice_interpolator = noa::geometry::interpolator_value_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                slice, slice_shape.filter(2, 3).as_safe<i32>().fft(), Value{0});

        insert_interpolate_3d_<REMAP>(
                slice_interpolator, slice_shape, grid, grid_strides, grid_shape,
                fwd_scaling_matrices, inv_rotation_matrices, cutoff, target_shape, ews_radius,
                slice_z_radius, stream);
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert_interpolate_3d(
            cudaArray* array, cudaTextureObject_t slice,
            InterpMode slice_interpolation_mode, const Shape4<i64>& slice_shape,
            Value* grid, const Strides4<i64>& grid_strides, const Shape4<i64>& grid_shape,
            const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
            f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius,
            f32 slice_z_radius, Stream& stream) {

        NOA_ASSERT(noa::cuda::memory::PtrTexture::array(slice) == array);
        NOA_ASSERT_DEVICE_PTR(grid, stream.device());
        NOA_CHECK(slice_interpolation_mode == InterpMode::LINEAR || slice_interpolation_mode == InterpMode::LINEAR_FAST,
                  "The interpolation mode should be {} or {}, got {}",
                  InterpMode::LINEAR, InterpMode::LINEAR_FAST, slice_interpolation_mode);

        const bool is_layered = noa::cuda::memory::PtrArray<Value>::is_layered(array);
        if (is_layered) {
            if (slice_interpolation_mode == InterpMode::LINEAR) {
                using interpolator_t = cuda::geometry::Interpolator2D<InterpMode::LINEAR, Value, false, true>;
                insert_interpolate_3d_<REMAP>(
                        interpolator_t(slice), slice_shape, grid, grid_strides, grid_shape,
                        fwd_scaling_matrices, inv_rotation_matrices, cutoff, target_shape, ews_radius,
                        slice_z_radius, stream);
            } else {
                using interpolator_t = cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Value, false, true>;
                insert_interpolate_3d_<REMAP>(
                        interpolator_t(slice), slice_shape, grid, grid_strides, grid_shape,
                        fwd_scaling_matrices, inv_rotation_matrices, cutoff, target_shape, ews_radius,
                        slice_z_radius, stream);
            }
        } else {
            if (slice_interpolation_mode == InterpMode::LINEAR) {
                using interpolator_t = cuda::geometry::Interpolator2D<InterpMode::LINEAR, Value>;
                insert_interpolate_3d_<REMAP>(
                        interpolator_t(slice), slice_shape, grid, grid_strides, grid_shape,
                        fwd_scaling_matrices, inv_rotation_matrices, cutoff, target_shape, ews_radius,
                        slice_z_radius, stream);
            } else {
                using interpolator_t = cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Value>;
                insert_interpolate_3d_<REMAP>(
                        interpolator_t(slice), slice_shape, grid, grid_strides, grid_shape,
                        fwd_scaling_matrices, inv_rotation_matrices, cutoff, target_shape, ews_radius,
                        slice_z_radius, stream);
            }
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void extract_3d(const Value* grid, const Strides4<i64>& grid_strides, const Shape4<i64>& grid_shape,
                    Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
                    const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                    f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(slice, stream.device());
        NOA_ASSERT_DEVICE_PTR(grid, stream.device());

        const auto grid_accessor = AccessorRestrict<const Value, 3, u32>(grid, grid_strides.pop_front().as_safe<u32>());
        const auto grid_interpolator = noa::geometry::interpolator_3d<BorderMode::ZERO, InterpMode::LINEAR>(
                grid_accessor, grid_shape.pop_front().as_safe<i32>().fft(), Value{0});

        extract_3d_<REMAP>(grid_interpolator, grid_shape, slice, slice_strides, slice_shape,
                           inv_scaling_matrices, fwd_rotation_matrices, cutoff, target_shape,
                           ews_radius, stream);
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void extract_3d(cudaArray* array, cudaTextureObject_t grid,
                    InterpMode grid_interpolation_mode, const Shape4<i64>& grid_shape,
                    Value* slice, const Strides4<i64>& slice_strides, const Shape4<i64>& slice_shape,
                    const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                    f32 cutoff, const Shape4<i64>& target_shape, const Vec2<f32>& ews_radius, Stream& stream) {
        NOA_ASSERT(noa::cuda::memory::PtrTexture::array(grid) == array);
        NOA_ASSERT_DEVICE_PTR(slice, stream.device());
        NOA_CHECK(grid_interpolation_mode == InterpMode::LINEAR || grid_interpolation_mode == InterpMode::LINEAR_FAST,
                  "The interpolation mode should be {} or {}, got {}",
                  InterpMode::LINEAR, InterpMode::LINEAR_FAST, grid_interpolation_mode);

        if (grid_interpolation_mode == InterpMode::LINEAR) {
            using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::LINEAR, Value>;
            extract_3d_<REMAP>(interpolator_t(grid), grid_shape, slice, slice_strides, slice_shape,
                               inv_scaling_matrices, fwd_rotation_matrices, cutoff, target_shape,
                               ews_radius, stream);
        } else if (grid_interpolation_mode == InterpMode::LINEAR_FAST) {
            using interpolator_t = noa::cuda::geometry::Interpolator3D<InterpMode::LINEAR_FAST, Value>;
            extract_3d_<REMAP>(interpolator_t(grid), grid_shape, slice, slice_strides, slice_shape,
                               inv_scaling_matrices, fwd_rotation_matrices, cutoff, target_shape,
                               ews_radius, stream);
        } else {
            NOA_THROW("The interpolation mode should be {} or {}, got {}",
                      InterpMode::LINEAR, InterpMode::LINEAR_FAST, grid_interpolation_mode);
        }
    }

    template<Remap REMAP, typename Value,
             typename Scale0, typename Scale1,
             typename Rotate0, typename Rotate1, typename>
    void insert_interpolate_and_extract_3d(
            const Value* input_slice, const Strides4<i64>& input_slice_strides, const Shape4<i64>& input_slice_shape,
            Value* output_slice, const Strides4<i64>& output_slice_strides, const Shape4<i64>& output_slice_shape,
            const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
            const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
            f32 cutoff, const Vec2<f32>& ews_radius, f32 slice_z_radius, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(output_slice, stream.device());
        NOA_ASSERT_DEVICE_PTR(input_slice, stream.device());

        const auto input_slice_accessor = AccessorRestrict<const Value, 3, u32>(
                input_slice, input_slice_strides.filter(0, 2, 3).as_safe<u32>());
        const auto input_slice_interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                input_slice_accessor, input_slice_shape.filter(2, 3).as_safe<i32>().fft(), Value{0});

        insert_interpolate_and_extract_3d_<REMAP>(
                input_slice_interpolator, input_slice_shape,
                output_slice, output_slice_strides, output_slice_shape,
                insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                cutoff, ews_radius, slice_z_radius, stream);
    }

    template<Remap REMAP, typename Value,
             typename Scale0, typename Scale1,
             typename Rotate0, typename Rotate1, typename>
    void insert_interpolate_and_extract_3d(
            Value input_slice, const Shape4<i64>& input_slice_shape,
            Value* output_slice, const Strides4<i64>& output_slice_strides, const Shape4<i64>& output_slice_shape,
            const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
            const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
            f32 cutoff, const Vec2<f32>& ews_radius, f32 slice_z_radius, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(output_slice, stream.device());

        const auto input_slice_interpolator = noa::geometry::interpolator_value_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                input_slice, input_slice_shape.filter(2, 3).as_safe<i32>().fft(), Value{0});

        insert_interpolate_and_extract_3d_<REMAP>(
                input_slice_interpolator, input_slice_shape,
                output_slice, output_slice_strides, output_slice_shape,
                insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                cutoff, ews_radius, slice_z_radius, stream);
    }

    template<Remap REMAP, typename Value,
             typename Scale0, typename Scale1,
             typename Rotate0, typename Rotate1, typename>
    void insert_interpolate_and_extract_3d(
            cudaArray* input_slice_array, cudaTextureObject_t input_slice_texture,
            InterpMode input_slice_interpolation_mode, const Shape4<i64>& input_slice_shape,
            Value* output_slice, const Strides4<i64>& output_slice_strides, const Shape4<i64>& output_slice_shape,
            const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
            const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
            f32 cutoff, const Vec2<f32>& ews_radius, f32 slice_z_radius, Stream& stream) {

        // Input texture requirements:
        NOA_ASSERT(noa::cuda::memory::PtrTexture::array(input_slice_texture) == input_slice_array);
        NOA_ASSERT_DEVICE_PTR(output_slice, stream.device());
        NOA_CHECK(input_slice_interpolation_mode == InterpMode::LINEAR ||
                  input_slice_interpolation_mode == InterpMode::LINEAR_FAST,
                  "The interpolation mode should be {} or {}, got {}",
                  InterpMode::LINEAR, InterpMode::LINEAR_FAST, input_slice_interpolation_mode);

        const bool is_layered = noa::cuda::memory::PtrArray<Value>::is_layered(input_slice_array);
        if (is_layered) {
            if (input_slice_interpolation_mode == InterpMode::LINEAR) {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR, Value, false, true>;
                insert_interpolate_and_extract_3d_<REMAP>(
                        interpolator_t(input_slice_texture), input_slice_shape,
                        output_slice, output_slice_strides, output_slice_shape,
                        insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                        extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                        cutoff, ews_radius, slice_z_radius, stream);
            } else if (input_slice_interpolation_mode == InterpMode::LINEAR_FAST) {
                using interpolator_t = noa::cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Value, false, true>;
                insert_interpolate_and_extract_3d_<REMAP>(
                        interpolator_t(input_slice_texture), input_slice_shape,
                        output_slice, output_slice_strides, output_slice_shape,
                        insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                        extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                        cutoff, ews_radius, slice_z_radius, stream);
            }
        } else {
            if (input_slice_interpolation_mode == InterpMode::LINEAR) {
                using interpolator_t = cuda::geometry::Interpolator2D<InterpMode::LINEAR, Value>;
                insert_interpolate_and_extract_3d_<REMAP>(
                        interpolator_t(input_slice_texture), input_slice_shape,
                        output_slice, output_slice_strides, output_slice_shape,
                        insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                        extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                        cutoff, ews_radius, slice_z_radius, stream);
            } else if (input_slice_interpolation_mode == InterpMode::LINEAR_FAST) {
                using interpolator_t = cuda::geometry::Interpolator2D<InterpMode::LINEAR_FAST, Value>;
                insert_interpolate_and_extract_3d_<REMAP>(
                        interpolator_t(input_slice_texture), input_slice_shape,
                        output_slice, output_slice_strides, output_slice_shape,
                        insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                        extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                        cutoff, ews_radius, slice_z_radius, stream);
            }
        }
    }

    template<typename Value, typename>
    void gridding_correction(const Value* input, const Strides4<i64>& input_strides,
                             Value* output, const Strides4<i64>& output_strides,
                             const Shape4<i64>& shape, bool post_correction, Stream& stream) {
        NOA_ASSERT(noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        const auto i_shape = shape.as_safe<u32>();
        const auto input_accessor = Accessor<const Value, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = Accessor<Value, 4, u32>(output, output_strides.as_safe<u32>());

        if (post_correction) {
            const auto kernel = noa::algorithm::geometry::gridding_correction<true>(
                    input_accessor, output_accessor, i_shape);
            noa::cuda::utils::iwise_4d("geometry::fft::gridding_correction", i_shape, kernel, stream);
        } else {
            const auto kernel = noa::algorithm::geometry::gridding_correction<false>(
                    input_accessor, output_accessor, i_shape);
            noa::cuda::utils::iwise_4d("geometry::fft::gridding_correction", i_shape, kernel, stream);
        }
    }
    template void gridding_correction<f32, void>(const f32*, const Strides4<i64>&, f32*, const Strides4<i64>&, const Shape4<i64>&, bool, Stream&);
    template void gridding_correction<f64, void>(const f64*, const Strides4<i64>&, f64*, const Strides4<i64>&, const Shape4<i64>&, bool, Stream&);

    #define NOA_INSTANTIATE_INSERT_RASTERIZE_(T, REMAP, S, R)                       \
    template void insert_rasterize_3d<REMAP, T, S, R, void>(                        \
        const T*, const Strides4<i64>&, const Shape4<i64>&,                         \
        T*, const Strides4<i64>&, const Shape4<i64>&,                               \
        S const&, R const&, f32, const Shape4<i64>&, const Vec2<f32>&, Stream&);    \
    template void insert_rasterize_3d<REMAP, T, S, R, void>(                        \
        T, const Shape4<i64>&,                                                      \
        T*, const Strides4<i64>&, const Shape4<i64>&,                               \
        S const&, R const&, f32, const Shape4<i64>&, const Vec2<f32>&, Stream&)

    #define NOA_INSTANTIATE_INSERT_INTERPOLATE_(T, REMAP, S, R)                         \
    template void insert_interpolate_3d<REMAP, T, S, R, void>(                          \
        const T*, const Strides4<i64>&, const Shape4<i64>&,                             \
        T*, const Strides4<i64>&, const Shape4<i64>&,                                   \
        S const&, R const&, f32, const Shape4<i64>&, const Vec2<f32>&, f32, Stream&);   \
    template void insert_interpolate_3d<REMAP, T, S, R, void>(                          \
        T, const Shape4<i64>&,                                                          \
        T*, const Strides4<i64>&, const Shape4<i64>&,                                   \
        S const&, R const&, f32, const Shape4<i64>&, const Vec2<f32>&, f32, Stream&)

    #define NOA_INSTANTIATE_EXTRACT_(T, REMAP, S, R)        \
    template void extract_3d<REMAP, T, S, R, void>(         \
        const T*, const Strides4<i64>&, const Shape4<i64>&, \
        T*, const Strides4<i64>&, const Shape4<i64>&,       \
        S const&, R const&, f32, const Shape4<i64>&, const Vec2<f32>&, Stream&)

    #define NOA_INSTANTIATE_INSERT_EXTRACT_(T, REMAP, S0, S1, R0, R1)                       \
    template void insert_interpolate_and_extract_3d<REMAP, T, S0, S1, R0, R1, void>(        \
        const T*, const Strides4<i64>&, const Shape4<i64>&,                                 \
        T*, const Strides4<i64>&, const Shape4<i64>&,                                       \
        S0 const&, R0 const&, S1 const&, R1 const&, f32, const Vec2<f32>&, f32, Stream&);   \
    template void insert_interpolate_and_extract_3d<REMAP, T, S0, S1, R0, R1, void>(        \
        T, const Shape4<i64>&,                                                              \
        T*, const Strides4<i64>&, const Shape4<i64>&,                                       \
        S0 const&, R0 const&, S1 const&, R1 const&, f32, const Vec2<f32>&, f32, Stream&)

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

    #define NOA_INSTANTIATE_INSERT_THICK_TEXTURE(T, REMAP, S, R)            \
    template void insert_interpolate_3d<REMAP, T, S, R, void>(              \
        cudaArray*, cudaTextureObject_t, InterpMode, const Shape4<i64>&,    \
        T*, const Strides4<i64>&, const Shape4<i64>&,                       \
        S const&, R const&, f32, const Shape4<i64>&, const Vec2<f32>&, f32, Stream&)

    #define NOA_INSTANTIATE_EXTRACT_TEXTURE(T, REMAP, S, R)                 \
    template void extract_3d<REMAP, T, S, R, void>(                         \
        cudaArray*, cudaTextureObject_t, InterpMode, const Shape4<i64>&,    \
        T*, const Strides4<i64>&, const Shape4<i64>&,                       \
        S const&, R const&, f32, const Shape4<i64>&, const Vec2<f32>&, Stream&)

    #define NOA_INSTANTIATE_INSERT_EXTRACT_TEXTURE_(T, REMAP, S0, S1, R0, R1)       \
    template void insert_interpolate_and_extract_3d<REMAP, T, S0, S1, R0, R1, void>(\
        cudaArray*, cudaTextureObject_t, InterpMode, const Shape4<i64>&,            \
        T*, const Strides4<i64>&, const Shape4<i64>&,                               \
        S0 const&, R0 const&, S1 const&, R1 const&, f32,                            \
        const Vec2<f32>&, f32, Stream&)

    #define NOA_INSTANTIATE_PROJECT_TEXTURE_ALL_REMAP(T, S, R)  \
    NOA_INSTANTIATE_INSERT_THICK_TEXTURE(T, Remap::HC2H, S, R); \
    NOA_INSTANTIATE_INSERT_THICK_TEXTURE(T, Remap::HC2HC, S, R);\
    NOA_INSTANTIATE_EXTRACT_TEXTURE(T, Remap::HC2H, S, R);      \
    NOA_INSTANTIATE_EXTRACT_TEXTURE(T, Remap::HC2HC, S, R)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP_TEXTURE(T, S0, S1, R0, R1)  \
    NOA_INSTANTIATE_INSERT_EXTRACT_TEXTURE_(T, Remap::HC2H, S0, S1, R0, R1);    \
    NOA_INSTANTIATE_INSERT_EXTRACT_TEXTURE_(T, Remap::HC2HC, S0, S1, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE_TEXTURE(T, R0, R1)                      \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP_TEXTURE(T, Float22, Float22, R0, R1);           \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP_TEXTURE(T, const Float22*, Float22, R0, R1);    \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP_TEXTURE(T, Float22, const Float22*, R0, R1);    \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP_TEXTURE(T, const Float22*, const Float22*, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE_TEXTURE(T)                     \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE_TEXTURE(T, Float33, Float33);           \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE_TEXTURE(T, const Float33*, Float33);    \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE_TEXTURE(T, Float33, const Float33*);    \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE_TEXTURE(T, const Float33*, const Float33*)

    #define NOA_INSTANTIATE_PROJECT_TEXTURE_ALL(T)                                  \
    NOA_INSTANTIATE_PROJECT_TEXTURE_ALL_REMAP(T, Float22, Float33);                 \
    NOA_INSTANTIATE_PROJECT_TEXTURE_ALL_REMAP(T, const Float22*, Float33);          \
    NOA_INSTANTIATE_PROJECT_TEXTURE_ALL_REMAP(T, Float22, const Float33*);          \
    NOA_INSTANTIATE_PROJECT_TEXTURE_ALL_REMAP(T, const Float22*, const Float33*);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE_TEXTURE(T)

    NOA_INSTANTIATE_PROJECT_TEXTURE_ALL(f32);
    NOA_INSTANTIATE_PROJECT_TEXTURE_ALL(c32);
}
