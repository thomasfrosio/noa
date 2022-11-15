#include "noa/common/geometry/Interpolator.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"
#include "noa/gpu/cuda/utils/Atomic.cuh"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/geometry/fft/Project.h"

// The Project.h header contains the implementation, which includes
// a call to cuda's atomicAdd. For now, leave it like this, but make
// sure the Atomic.cub header is included before this point.
#include "noa/common/geometry/details/FourierProjections.h"

namespace {
    using namespace ::noa;

    template<typename T>
    using matrixOrRawConstPtr_t = std::conditional_t<
            traits::is_floatXX_v<T>,
            traits::remove_ref_cv_t<T>,
            const traits::element_type_t<T>*>;

    template<bool ASSERT_VALID_PTR, typename T>
    auto matrixOrRawConstPtr_(T v) {
        using output_t = matrixOrRawConstPtr_t<T>;
        if constexpr (traits::is_floatXX_v<T>) {
            return output_t(v);
        } else {
            NOA_ASSERT(!ASSERT_VALID_PTR || v.get() != nullptr);
            return static_cast<output_t>(v.get());
        }
    }

    template<bool ASSERT_VALID_PTR, typename MatrixWrapper, typename MatrixValue>
    auto matrixOrRawConstPtrOnDevice_(MatrixWrapper matrices, size_t count,
                                      cuda::memory::PtrDevice<MatrixValue>& buffer, cuda::Stream& stream) {
        using output_t = matrixOrRawConstPtr_t<MatrixWrapper>;
        if constexpr (traits::is_floatXX_v<MatrixWrapper>) {
            return output_t(matrices);
        } else {
            NOA_ASSERT(!ASSERT_VALID_PTR || matrices.get() != nullptr);
            return output_t(cuda::utils::ensureDeviceAccess(matrices.get(), stream, buffer, count));
        }
    }

    template<typename MatrixWrapper, typename MatrixValue>
    auto inverseMatrices_(MatrixWrapper matrices, size_t count,
                          cuda::memory::PtrPinned<MatrixValue>& buffer) {
        if constexpr (traits::is_floatXX_v<MatrixWrapper>) {
            return math::inverse(matrices);
        } else {
            NOA_ASSERT(count == 0 || cuda::utils::hostPointer(matrices) != nullptr);
            buffer = cuda::memory::PtrPinned<MatrixValue>(count);
            for (size_t i = 0; i < count; ++i)
                buffer[i] = math::inverse(matrices[i]);
            using output_t = const MatrixValue*;
            return output_t(buffer.get());
        }
    }

    template<fft::Remap REMAP, typename Interpolator, typename Value, typename Scale, typename Rotate>
    void insert3D_(Interpolator slice_interpolator, dim4_t slice_shape,
                   Value* grid, dim4_t grid_strides, dim4_t grid_shape,
                   const Scale& inv_scaling_matrices,
                   const Rotate& fwd_rotation_matrices,
                   float cutoff, dim4_t target_shape, float2_t ews_radius,
                   float slice_z_radius, cuda::Stream& stream) {

        const auto slice_count = static_cast<size_t>(slice_shape[0]);
        const auto grid_strides_3d = safe_cast<uint3_t>(dim3_t{grid_strides[1], grid_strides[2], grid_strides[3]});
        const auto grid_accessor = AccessorRestrict<Value, 3, uint32_t>(grid, grid_strides_3d);
        const auto iwise_shape = safe_cast<int3_t>(dim3_t{grid_shape.get(1)}).fft();

        const auto apply_ews = any(ews_radius != 0);
        const bool apply_scale = inv_scaling_matrices != Scale{};

        cuda::memory::PtrPinned<float22_t> fwd_scaling_matrices_buffer;
        cuda::memory::PtrPinned<float33_t> inv_rotation_matrices_buffer;
        const auto fwd_scaling_matrices = inverseMatrices_(
                matrixOrRawConstPtr_<true>(inv_scaling_matrices), slice_count, fwd_scaling_matrices_buffer);
        const auto inv_rotation_matrices = inverseMatrices_(
                matrixOrRawConstPtr_<true>(fwd_rotation_matrices), slice_count, inv_rotation_matrices_buffer);

        using namespace noa::geometry::fft::details;
        if (apply_ews && apply_scale) { // TODO Benchmark to see if it's worth it
            const auto functor = fourierInsertionExplicitThickness<REMAP, int32_t>(
                    slice_interpolator, slice_shape, grid_accessor, grid_shape,
                    fwd_scaling_matrices, inv_rotation_matrices,
                    cutoff, target_shape, ews_radius, slice_z_radius);
            cuda::utils::iwise3D("geometry::fft::insert3D", iwise_shape, functor, stream);
        } else if (apply_ews) {
            const auto functor = fourierInsertionExplicitThickness<REMAP, int32_t>(
                    slice_interpolator, slice_shape, grid_accessor, grid_shape,
                    empty_t{}, inv_rotation_matrices,
                    cutoff, target_shape, ews_radius, slice_z_radius);
            cuda::utils::iwise3D("geometry::fft::insert3D", iwise_shape, functor, stream);
        } else if (apply_scale) {
            const auto functor = fourierInsertionExplicitThickness<REMAP, int32_t>(
                    slice_interpolator, slice_shape, grid_accessor, grid_shape,
                    fwd_scaling_matrices, inv_rotation_matrices,
                    cutoff, target_shape, empty_t{}, slice_z_radius);
            cuda::utils::iwise3D("geometry::fft::insert3D", iwise_shape, functor, stream);
        } else {
            const auto functor = fourierInsertionExplicitThickness<REMAP, int32_t>(
                    slice_interpolator, slice_shape, grid_accessor, grid_shape,
                    empty_t{}, inv_rotation_matrices,
                    cutoff, target_shape, empty_t{}, slice_z_radius);
            cuda::utils::iwise3D("geometry::fft::insert3D", iwise_shape, functor, stream);
        }

        if constexpr (!traits::is_floatXX_v<Scale>)
            if (inv_scaling_matrices)
                stream.attach(inv_scaling_matrices, fwd_scaling_matrices_buffer.share());
        if constexpr (!traits::is_floatXX_v<Rotate>)
            stream.attach(fwd_rotation_matrices, inv_rotation_matrices_buffer.share());
    }


    template<fft::Remap REMAP, typename Value, typename Interpolator, typename Scale, typename Rotate>
    void extract3D_(Interpolator grid, dim4_t grid_shape,
                    Value* slice, dim4_t slice_strides, dim4_t slice_shape,
                    const Scale& inv_scaling_matrices,
                    const Rotate& fwd_rotation_matrices,
                    float cutoff, dim4_t target_shape, float2_t ews_radius, cuda::Stream& stream) {

        const auto slice_count = static_cast<size_t>(slice_shape[0]);
        const auto iwise_shape = safe_cast<int3_t>(dim3_t{slice_shape[0], slice_shape[2], slice_shape[3]}).fft();
        const auto slice_strides_3d = safe_cast<uint3_t>(dim3_t{slice_strides[0], slice_strides[2], slice_strides[3]});
        const auto slice_accessor = AccessorRestrict<Value, 3, uint32_t>(slice, slice_strides_3d);

        const auto apply_ews = any(ews_radius != 0);
        const bool apply_scale = inv_scaling_matrices != Scale{};

        // Ensure transformation parameters are accessible to the GPU:
        cuda::memory::PtrDevice<float22_t> inv_scaling_matrices_buffer;
        cuda::memory::PtrDevice<float33_t> fwd_rotation_matrices_buffer;
        const auto inv_scaling_matrices_ = matrixOrRawConstPtrOnDevice_<false>(
                inv_scaling_matrices, slice_count, inv_scaling_matrices_buffer, stream);
        const auto fwd_rotation_matrices_ = matrixOrRawConstPtrOnDevice_<true>(
                fwd_rotation_matrices, slice_count, fwd_rotation_matrices_buffer, stream);

        using namespace noa::geometry::fft::details;
        if (apply_ews && apply_scale) { // TODO Benchmark to see if it's worth it
            const auto functor = fourierExtraction<REMAP, int32_t>(
                    grid, grid_shape, slice_accessor, slice_shape,
                    inv_scaling_matrices_, fwd_rotation_matrices_,
                    cutoff, target_shape, ews_radius);
            cuda::utils::iwise3D("geometry::fft::extract3D", iwise_shape, functor, stream);
        } else if (apply_ews) {
            const auto functor = fourierExtraction<REMAP, int32_t>(
                    grid, grid_shape, slice_accessor, slice_shape,
                    empty_t{}, fwd_rotation_matrices_,
                    cutoff, target_shape, ews_radius);
            cuda::utils::iwise3D("geometry::fft::extract3D", iwise_shape, functor, stream);
        } else if (apply_scale) {
            const auto functor = fourierExtraction<REMAP, int32_t>(
                    grid, grid_shape, slice_accessor, slice_shape,
                    inv_scaling_matrices_, fwd_rotation_matrices_,
                    cutoff, target_shape, empty_t{});
            cuda::utils::iwise3D("geometry::fft::extract3D", iwise_shape, functor, stream);
        } else {
            const auto functor = fourierExtraction<REMAP, int32_t>(
                    grid, grid_shape, slice_accessor, slice_shape,
                    empty_t{}, fwd_rotation_matrices_,
                    cutoff, target_shape, empty_t{});
            cuda::utils::iwise3D("geometry::fft::extract3D", iwise_shape, functor, stream);
        }
        if constexpr (!traits::is_floatXX_v<Scale>)
            if (inv_scaling_matrices)
                stream.attach(inv_scaling_matrices);
        if constexpr (!traits::is_floatXX_v<Rotate>)
            stream.attach(fwd_rotation_matrices);
    }

    template<fft::Remap REMAP, typename Interpolator, typename Value,
             typename Scale0, typename Scale1, typename Rotate0, typename Rotate1>
    void extract3D_(Interpolator input_slice_interpolator, dim4_t input_slice_shape,
                    Value* output_slice, dim4_t output_slice_strides, dim4_t output_slice_shape,
                    const Scale0& insert_inv_scaling_matrices, const Rotate0& insert_fwd_rotation_matrices,
                    const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
                    float cutoff, float2_t ews_radius, float slice_z_radius, cuda::Stream& stream) {

        const auto output_slice_strides_2d = safe_cast<uint3_t>(
                dim3_t{output_slice_strides[0], output_slice_strides[2], output_slice_strides[3]});
        const auto output_slice_accessor = AccessorRestrict<Value, 3, uint32_t>(
                output_slice, output_slice_strides_2d);
        const auto iwise_shape = safe_cast<int3_t>(
                dim3_t{output_slice_shape[0], output_slice_shape[2], output_slice_shape[3]}).fft();

        const auto apply_ews = any(ews_radius != 0);
        const bool apply_scale = insert_inv_scaling_matrices != Scale0{};

        // The transformation for the insertion needs to be inverted.
        cuda::memory::PtrPinned<float22_t> insert_fwd_scaling_matrices_buffer;
        cuda::memory::PtrPinned<float33_t> insert_inv_rotation_matrices_buffer;
        const auto insert_fwd_scaling_matrices = inverseMatrices_(
                matrixOrRawConstPtr_<false>(insert_inv_scaling_matrices),
                input_slice_shape[0], insert_fwd_scaling_matrices_buffer);
        const auto insert_inv_rotation_matrices = inverseMatrices_(
                matrixOrRawConstPtr_<true>(insert_fwd_rotation_matrices),
                input_slice_shape[0], insert_inv_rotation_matrices_buffer);

        // Ensure transformation parameters are accessible to the GPU:
        cuda::memory::PtrDevice<float22_t> extract_inv_scaling_matrices_buffer;
        cuda::memory::PtrDevice<float33_t> extract_fwd_rotation_matrices_buffer;
        const auto extract_inv_scaling_matrices_ = matrixOrRawConstPtrOnDevice_<false>(
                extract_inv_scaling_matrices, output_slice_shape[0],
                extract_inv_scaling_matrices_buffer, stream);
        const auto extract_fwd_rotation_matrices_ = matrixOrRawConstPtrOnDevice_<true>(
                extract_fwd_rotation_matrices, output_slice_shape[0],
                extract_fwd_rotation_matrices_buffer, stream);

        using namespace noa::geometry::fft::details;
        if (apply_ews && apply_scale) { // TODO Benchmark to see if it's worth it
            const auto functor = fourierInsertExtraction<REMAP, int32_t>(
                    input_slice_interpolator, input_slice_shape,
                    output_slice_accessor, output_slice_shape,
                    insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                    extract_inv_scaling_matrices_, extract_fwd_rotation_matrices_,
                    cutoff, ews_radius, slice_z_radius);
            cuda::utils::iwise3D("geometry::fft::extract3D", iwise_shape, functor, stream);
        } else if (apply_ews) {
            const auto functor = fourierInsertExtraction<REMAP, int32_t>(
                    input_slice_interpolator, input_slice_shape,
                    output_slice_accessor, output_slice_shape,
                    empty_t{}, insert_inv_rotation_matrices,
                    extract_inv_scaling_matrices_, extract_fwd_rotation_matrices_,
                    cutoff, ews_radius, slice_z_radius);
            cuda::utils::iwise3D("geometry::fft::extract3D", iwise_shape, functor, stream);
        } else if (apply_scale) {
            const auto functor = fourierInsertExtraction<REMAP, int32_t>(
                    input_slice_interpolator, input_slice_shape,
                    output_slice_accessor, output_slice_shape,
                    insert_fwd_scaling_matrices, insert_inv_rotation_matrices,
                    extract_inv_scaling_matrices_, extract_fwd_rotation_matrices_,
                    cutoff, empty_t{}, slice_z_radius);
            cuda::utils::iwise3D("geometry::fft::extract3D", iwise_shape, functor, stream);
        } else {
            const auto functor = fourierInsertExtraction<REMAP, int32_t>(
                    input_slice_interpolator, input_slice_shape,
                    output_slice_accessor, output_slice_shape,
                    empty_t{}, insert_inv_rotation_matrices,
                    extract_inv_scaling_matrices_, extract_fwd_rotation_matrices_,
                    cutoff, empty_t{}, slice_z_radius);
            cuda::utils::iwise3D("geometry::fft::extract3D", iwise_shape, functor, stream);
        }

        if constexpr (!traits::is_floatXX_v<Scale0>)
            if (insert_inv_scaling_matrices)
                stream.attach(insert_inv_scaling_matrices, insert_fwd_scaling_matrices_buffer.share());
        if constexpr (!traits::is_floatXX_v<Rotate0>)
            stream.attach(insert_fwd_rotation_matrices, insert_inv_rotation_matrices_buffer.share());
        if constexpr (!traits::is_floatXX_v<Scale1>)
            if (extract_inv_scaling_matrices)
                stream.attach(extract_inv_scaling_matrices);
        if constexpr (!traits::is_floatXX_v<Rotate1>)
            stream.attach(extract_fwd_rotation_matrices);
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert3D(const shared_t<Value[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream) {

        NOA_ASSERT_DEVICE_PTR(slice.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(grid.get(), stream.device());

        const auto slice_strides_3d = safe_cast<uint3_t>(dim3_t{slice_strides[0], slice_strides[2], slice_strides[3]});
        const auto grid_strides_3d = safe_cast<uint3_t>(dim3_t(grid_strides.get(1)));
        const auto slice_accessor = AccessorRestrict<const Value, 3, uint32_t>(slice.get(), slice_strides_3d);
        const auto grid_accessor = AccessorRestrict<Value, 3, uint32_t>(grid.get(), grid_strides_3d);
        const auto iwise_shape = safe_cast<int3_t>(dim3_t{slice_shape[0], slice_shape[2], slice_shape[3]}).fft();

        const auto apply_ews = any(ews_radius != 0);
        const bool apply_scale = inv_scaling_matrices != Scale{};

        // Ensure transformation parameters are accessible to the GPU:
        memory::PtrDevice<float22_t> inv_scaling_matrices_buffer;
        memory::PtrDevice<float33_t> fwd_rotation_matrices_buffer;
        const auto inv_scaling_matrices_ = matrixOrRawConstPtrOnDevice_<false>(
                inv_scaling_matrices, iwise_shape[0], inv_scaling_matrices_buffer, stream);
        const auto fwd_rotation_matrices_ = matrixOrRawConstPtrOnDevice_<true>(
                fwd_rotation_matrices, iwise_shape[0], fwd_rotation_matrices_buffer, stream);

        using namespace noa::geometry::fft::details;
        if (apply_ews && apply_scale) { // TODO Benchmark to see if it's worth it
            const auto functor = fourierInsertionByGridding<REMAP, int32_t>(
                    slice_accessor, slice_shape, grid_accessor, grid_shape,
                    inv_scaling_matrices_, fwd_rotation_matrices_,
                    cutoff, target_shape, ews_radius);
            utils::iwise3D("geometry::fft::insert3D", iwise_shape, functor, stream);
        } else if (apply_ews) {
            const auto functor = fourierInsertionByGridding<REMAP, int32_t>(
                    slice_accessor, slice_shape, grid_accessor, grid_shape,
                    empty_t{}, fwd_rotation_matrices_,
                    cutoff, target_shape, ews_radius);
            utils::iwise3D("geometry::fft::insert3D", iwise_shape, functor, stream);
        } else if (apply_scale) {
            const auto functor = fourierInsertionByGridding<REMAP, int32_t>(
                    slice_accessor, slice_shape, grid_accessor, grid_shape,
                    inv_scaling_matrices_, fwd_rotation_matrices_,
                    cutoff, target_shape, empty_t{});
            utils::iwise3D("geometry::fft::insert3D", iwise_shape, functor, stream);
        } else {
            const auto functor = fourierInsertionByGridding<REMAP, int32_t>(
                    slice_accessor, slice_shape, grid_accessor, grid_shape,
                    empty_t{}, fwd_rotation_matrices_,
                    cutoff, target_shape, empty_t{});
            utils::iwise3D("geometry::fft::insert3D", iwise_shape, functor, stream);
        }

        stream.attach(slice, grid);
        if constexpr (!traits::is_floatXX_v<Scale>)
            if (inv_scaling_matrices)
                stream.attach(inv_scaling_matrices);
        if constexpr (!traits::is_floatXX_v<Rotate>)
            stream.attach(fwd_rotation_matrices);
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert3D(const shared_t<Value[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius,
                  float slice_z_radius, bool use_texture, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(grid.get(), stream.device());

        if (use_texture) {
            if constexpr (traits::is_any_v<Value, double, cdouble_t>) {
                NOA_THROW("Double precision is not supported in this mode. Use use_texture=false instead");
            } else {
                // Be conservative on the memory that is allocated, do one slice at a time.
                // Users can use the overload with the texture anyway, so they still have
                // the choice to batch everything.
                dim_t slice_count = slice_shape[0];
                if (slice_strides[0] == 0)
                    slice_count = 1;
                memory::PtrArray<Value> array({1, 1, slice_shape[2], slice_shape[3]}, cudaArrayLayered);
                memory::PtrTexture texture(array.get(), INTERP_LINEAR, BORDER_ZERO);
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, true>;
                const auto slice_interpolator = interpolator_t(texture.get());

                for (dim_t i = 0; i < slice_count; ++i) {
                    memory::copy(slice.get() + slice_strides[0] * i, slice_strides,
                                 array.get(), array.shape(), stream);
                    insert3D_<REMAP>(slice_interpolator, slice_shape, grid.get(), grid_strides, grid_shape,
                                     inv_scaling_matrices, fwd_rotation_matrices, cutoff, target_shape, ews_radius,
                                     slice_z_radius, stream);
                }
                stream.attach(slice, grid, array.share(), texture.share());
            }
        } else {
            NOA_ASSERT_DEVICE_PTR(slice.get(), stream.device());
            const auto slice_shape_2d = safe_cast<int2_t>(dim2_t(slice_shape.get(2)));
            const auto slice_strides_3d = safe_cast<uint3_t>(dim3_t{slice_strides[0], slice_strides[2], slice_strides[3]});
            const auto slice_accessor = AccessorRestrict<const Value, 3, uint32_t>(slice.get(), slice_strides_3d);
            const auto slice_interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(
                    slice_accessor, slice_shape_2d.fft(), Value{0});

            insert3D_<REMAP>(slice_interpolator, slice_shape, grid.get(), grid_strides, grid_shape,
                             inv_scaling_matrices, fwd_rotation_matrices, cutoff, target_shape, ews_radius,
                             slice_z_radius, stream);
            stream.attach(slice, grid);
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert3D(const shared_t<cudaArray>& array,
                  const shared_t<cudaTextureObject_t>& slice, InterpMode slice_interpolation_mode, dim4_t slice_shape,
                  const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius,
                  float slice_z_radius, Stream& stream) {

        // Input texture requirements:
        constexpr bool NORMALIZED = false;
        constexpr bool LAYERED = true;
        NOA_ASSERT(memory::PtrTexture::array(*slice) == array.get());

        if (slice_interpolation_mode == INTERP_LINEAR) {
            using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, NORMALIZED, LAYERED>;
            const auto slice_interpolator = interpolator_t(*slice);
            insert3D_<REMAP>(slice_interpolator, slice_shape, grid.get(), grid_strides, grid_shape,
                             inv_scaling_matrices, fwd_rotation_matrices, cutoff, target_shape, ews_radius,
                             slice_z_radius, stream);
        } else if (slice_interpolation_mode == INTERP_LINEAR_FAST) {
            using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, NORMALIZED, LAYERED>;
            const auto slice_interpolator = interpolator_t(*slice);
            insert3D_<REMAP>(slice_interpolator, slice_shape, grid.get(), grid_strides, grid_shape,
                             inv_scaling_matrices, fwd_rotation_matrices, cutoff, target_shape, ews_radius,
                             slice_z_radius, stream);
        } else {
            NOA_THROW("The interpolation mode should be {} or {}, got {}",
                      INTERP_LINEAR, INTERP_LINEAR_FAST, slice_interpolation_mode);
        }
        stream.attach(array, grid, slice);
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void extract3D(const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                   const shared_t<Value[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                   const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                   float cutoff, dim4_t target_shape, float2_t ews_radius,
                   bool use_texture, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(slice.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(grid.get(), stream.device());

        if (use_texture) {
            if constexpr (traits::is_any_v<Value, double, cdouble_t>) {
                NOA_THROW("Double precision is not supported in this mode. Use use_texture=false instead");
            } else {
                memory::PtrArray<Value> array(grid_shape);
                memory::PtrTexture texture(array.get(), INTERP_LINEAR, BORDER_ZERO);
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR, Value>;
                const auto grid_interpolator = interpolator_t(texture.get());

                memory::copy(grid.get(), grid_strides, array.get(), array.shape(), stream);
                extract3D_<REMAP>(grid_interpolator, grid_shape, slice.get(), slice_strides, slice_shape,
                                  inv_scaling_matrices, fwd_rotation_matrices, cutoff, target_shape,
                                  ews_radius, stream);
                stream.attach(slice, grid, array.share(), texture.share());
            }
        } else {
            const auto grid_shape_3d = safe_cast<int3_t>(dim3_t(grid_shape.get(1)));
            const auto grid_strides_3d = safe_cast<uint3_t>(dim3_t(grid_strides.get(1)));
            const auto grid_accessor = AccessorRestrict<const Value, 3, uint32_t>(grid.get(), grid_strides_3d);
            const auto grid_interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP_LINEAR>(
                    grid_accessor, grid_shape_3d.fft(), Value{0});

            extract3D_<REMAP>(grid_interpolator, grid_shape, slice.get(), slice_strides, slice_shape,
                              inv_scaling_matrices, fwd_rotation_matrices, cutoff, target_shape,
                              ews_radius, stream);
            stream.attach(slice, grid);
        }
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void extract3D(const shared_t<cudaArray>& array,
                   const shared_t<cudaTextureObject_t>& grid, InterpMode grid_interpolation_mode, dim4_t grid_shape,
                   const shared_t<Value[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                   const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                   float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream) {
        NOA_ASSERT(memory::PtrTexture::array(*grid) == array.get());

        if (grid_interpolation_mode == INTERP_LINEAR) {
            using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR, Value>;
            const auto grid_interpolator = interpolator_t(*grid);
            extract3D_<REMAP>(grid_interpolator, grid_shape, slice.get(), slice_strides, slice_shape,
                              inv_scaling_matrices, fwd_rotation_matrices, cutoff, target_shape,
                              ews_radius, stream);
        } else if (grid_interpolation_mode == INTERP_LINEAR_FAST) {
            using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, Value>;
            const auto grid_interpolator = interpolator_t(*grid);
            extract3D_<REMAP>(grid_interpolator, grid_shape, slice.get(), slice_strides, slice_shape,
                              inv_scaling_matrices, fwd_rotation_matrices, cutoff, target_shape,
                              ews_radius, stream);
        } else {
            NOA_THROW("The interpolation mode should be {} or {}, got {}",
                      INTERP_LINEAR, INTERP_LINEAR_FAST, grid_interpolation_mode);
        }
        stream.attach(array, grid, slice);
    }

    template<Remap REMAP, typename Value,
             typename Scale0, typename Scale1,
             typename Rotate0, typename Rotate1, typename>
    void extract3D(const shared_t<Value[]>& input_slice, dim4_t input_slice_strides, dim4_t input_slice_shape,
                   const shared_t<Value[]>& output_slice, dim4_t output_slice_strides, dim4_t output_slice_shape,
                   const Scale0& insert_inv_scaling_matrices, const Rotate0& insert_fwd_rotation_matrices,
                   const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
                   float cutoff, float2_t ews_radius, float slice_z_radius, bool use_texture, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(output_slice.get(), stream.device());

        if (use_texture) {
            if constexpr (traits::is_any_v<Value, double, cdouble_t>) {
                NOA_THROW("Double precision is not supported in this mode. Use use_texture=false instead");
            } else {
                // Be conservative on the memory that is allocated, do one slice at a time.
                // Users can use the overload with the texture anyway, so they still
                // have the choice to batch everything.
                dim_t input_slice_count = input_slice_shape[0];
                if (input_slice_strides[0] == 0)
                    input_slice_count = 1;
                memory::PtrArray<Value> array({1, 1, input_slice_shape[2], input_slice_shape[3]}, cudaArrayLayered);
                memory::PtrTexture texture(array.get(), INTERP_LINEAR, BORDER_ZERO);
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, false, true>;
                const auto input_slice_interpolator = interpolator_t(texture.get());

                for (dim_t i = 0; i < input_slice_count; ++i) {
                    memory::copy(input_slice.get() + input_slice_strides[0] * i, input_slice_strides,
                                 array.get(), array.shape(), stream);
                    extract3D_<REMAP>(input_slice_interpolator, input_slice_shape,
                                      output_slice.get(), output_slice_strides, output_slice_shape,
                                      insert_inv_scaling_matrices, insert_fwd_rotation_matrices,
                                      extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                                      cutoff, ews_radius, slice_z_radius, stream);
                }
                stream.attach(input_slice, output_slice, array.share(), texture.share());
            }
        } else {
            NOA_ASSERT_DEVICE_PTR(input_slice.get(), stream.device());
            const dim3_t input_slice_strides_2d{input_slice_strides[0], input_slice_strides[2], input_slice_strides[3]};
            const auto input_slice_accessor = AccessorRestrict<const Value, 3, uint32_t>(
                    input_slice.get(), input_slice_strides_2d);
            const auto input_slice_interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(
                    input_slice_accessor, safe_cast<int2_t>(dim2_t(input_slice_shape.get(2))).fft(), Value{0});

            extract3D_<REMAP>(input_slice_interpolator, input_slice_shape,
                              output_slice.get(), output_slice_strides, output_slice_shape,
                              insert_inv_scaling_matrices, insert_fwd_rotation_matrices,
                              extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                              cutoff, ews_radius, slice_z_radius, stream);
            stream.attach(input_slice, output_slice);
        }
    }

    template<Remap REMAP, typename Value,
             typename Scale0, typename Scale1,
             typename Rotate0, typename Rotate1, typename>
    void extract3D(const shared_t<cudaArray>& input_slice_array,
                   const shared_t<cudaTextureObject_t>& input_slice_texture,
                   InterpMode input_slice_interpolation_mode, dim4_t input_slice_shape,
                   const shared_t<Value[]>& output_slice, dim4_t output_slice_strides, dim4_t output_slice_shape,
                   const Scale0& insert_inv_scaling_matrices, const Rotate0& insert_fwd_rotation_matrices,
                   const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
                   float cutoff, float2_t ews_radius, float slice_z_radius, Stream& stream) {

        // Input texture requirements:
        constexpr bool NORMALIZED = false;
        constexpr bool LAYERED = true;
        NOA_ASSERT(memory::PtrTexture::array(*input_slice_texture) == input_slice_array.get());

        if (input_slice_interpolation_mode == INTERP_LINEAR) {
            using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, Value, NORMALIZED, LAYERED>;
            const auto input_slice_interpolator = interpolator_t(*input_slice_texture);
            extract3D_<REMAP>(input_slice_interpolator, input_slice_shape,
                              output_slice.get(), output_slice_strides, output_slice_shape,
                              insert_inv_scaling_matrices, insert_fwd_rotation_matrices,
                              extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                              cutoff, ews_radius, slice_z_radius, stream);
        } else if (input_slice_interpolation_mode == INTERP_LINEAR_FAST) {
            using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, Value, NORMALIZED, LAYERED>;
            const auto input_slice_interpolator = interpolator_t(*input_slice_texture);
            extract3D_<REMAP>(input_slice_interpolator, input_slice_shape,
                              output_slice.get(), output_slice_strides, output_slice_shape,
                              insert_inv_scaling_matrices, insert_fwd_rotation_matrices,
                              extract_inv_scaling_matrices, extract_fwd_rotation_matrices,
                              cutoff, ews_radius, slice_z_radius, stream);
        } else {
            NOA_THROW("The interpolation mode should be {} or {}, got {}",
                      INTERP_LINEAR, INTERP_LINEAR_FAST, input_slice_interpolation_mode);
        }
        stream.attach(input_slice_array, input_slice_texture, output_slice);
    }

    template<typename Value, typename>
    void griddingCorrection(const shared_t<Value[]>& input, dim4_t input_strides,
                            const shared_t<Value[]>& output, dim4_t output_strides,
                            dim4_t shape, bool post_correction, Stream& stream) {
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const auto iwise_shape = safe_cast<uint4_t>(shape);
        const auto input_accessor = Accessor<const Value, 4, uint32_t>(input.get(), safe_cast<uint4_t>(input_strides));
        const auto output_accessor = Accessor<Value, 4, uint32_t>(output.get(), safe_cast<uint4_t>(output_strides));

        if (post_correction) {
            const auto kernel = noa::geometry::fft::details::griddingCorrection<true>(
                    input_accessor, output_accessor, shape);
            utils::iwise4D("geometry::fft::griddingCorrection", iwise_shape, kernel, stream);
        } else {
            const auto kernel = noa::geometry::fft::details::griddingCorrection<false>(
                    input_accessor, output_accessor, shape);
            utils::iwise4D("geometry::fft::griddingCorrection", iwise_shape, kernel, stream);
        }
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_INSERT_(T, REMAP, S, R) \
    template void insert3D<REMAP, T, S, R, void>(   \
        const shared_t<T[]>&, dim4_t, dim4_t,       \
        const shared_t<T[]>&, dim4_t, dim4_t,       \
        const S&, const R&, float, dim4_t, float2_t, Stream&)

    #define NOA_INSTANTIATE_INSERT_THICK_(T, REMAP, S, R)                   \
    template void insert3D<REMAP, T, S, R, void>(                           \
        const shared_t<T[]>&, dim4_t, dim4_t,                               \
        const shared_t<T[]>&, dim4_t, dim4_t,                               \
        const S&, const R&, float, dim4_t, float2_t, float, bool, Stream&)

    #define NOA_INSTANTIATE_EXTRACT_(T, REMAP, S, R)    \
    template void extract3D<REMAP, T, S, R, void>(      \
        const shared_t<T[]>&, dim4_t, dim4_t,           \
        const shared_t<T[]>&, dim4_t, dim4_t,           \
        const S&, const R&, float, dim4_t, float2_t, bool, Stream&)

    #define NOA_INSTANTIATE_INSERT_EXTRACT_(T, REMAP, S0, S1, R0, R1)   \
    template void extract3D<REMAP, T, S0, S1, R0, R1, void>(            \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        const S0&, const R0&, const S1&, const R1&, float,              \
        float2_t, float, bool, Stream&)

    #define NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, S, R)      \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2H, S, R);           \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2HC, S, R);          \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2H, S, R);          \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2HC, S, R);         \
    NOA_INSTANTIATE_INSERT_THICK_(T, Remap::HC2H, S, R);    \
    NOA_INSTANTIATE_INSERT_THICK_(T, Remap::HC2HC, S, R);   \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2H, S, R);         \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2HC, S, R)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, S0, S1, R0, R1)  \
    NOA_INSTANTIATE_INSERT_EXTRACT_(T, Remap::HC2H, S0, S1, R0, R1);    \
    NOA_INSTANTIATE_INSERT_EXTRACT_(T, Remap::HC2HC, S0, S1, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, R0, R1)                              \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, float22_t, float22_t, R0, R1);               \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, shared_t<float22_t[]>, float22_t, R0, R1);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, float22_t, shared_t<float22_t[]>, R0, R1);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, shared_t<float22_t[]>, shared_t<float22_t[]>, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(T)                             \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, float33_t, float33_t);               \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, shared_t<float33_t[]>, float33_t);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, float33_t, shared_t<float33_t[]>);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, shared_t<float33_t[]>, shared_t<float33_t[]>)

    #define NOA_INSTANTIATE_PROJECT_ALL(T)                                  \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, float22_t, float33_t);             \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, shared_t<float22_t[]>, float33_t); \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, float22_t, shared_t<float33_t[]>); \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, shared_t<float22_t[]>, shared_t<float33_t[]>);\
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(T)

    NOA_INSTANTIATE_PROJECT_ALL(float);
    NOA_INSTANTIATE_PROJECT_ALL(cfloat_t);
    NOA_INSTANTIATE_PROJECT_ALL(double);
    NOA_INSTANTIATE_PROJECT_ALL(cdouble_t);

    #define NOA_INSTANTIATE_INSERT_THICK_TEXTURE(T, REMAP, S, R)            \
    template void insert3D<REMAP, T, S, R, void>(                           \
        const shared_t<cudaArray>&,                                         \
        const shared_t<cudaTextureObject_t>& slice, InterpMode, dim4_t,     \
        const shared_t<T[]>&, dim4_t, dim4_t,                               \
        const S&, const R&, float, dim4_t, float2_t, float, Stream&)

    #define NOA_INSTANTIATE_EXTRACT_TEXTURE(T, REMAP, S, R)         \
    template void extract3D<REMAP, T, S, R, void>(                  \
        const shared_t<cudaArray>&,                                 \
        const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t,   \
        const shared_t<T[]>&, dim4_t, dim4_t,                       \
        const S&, const R&, float, dim4_t, float2_t, Stream&)

    #define NOA_INSTANTIATE_INSERT_EXTRACT_TEXTURE_(T, REMAP, S0, S1, R0, R1)   \
    template void extract3D<REMAP, T, S0, S1, R0, R1, void>(                    \
        const shared_t<cudaArray>&,                                             \
        const shared_t<cudaTextureObject_t>&, InterpMode, dim4_t,               \
        const shared_t<T[]>&, dim4_t, dim4_t,                                   \
        const S0&, const R0&, const S1&, const R1&, float,                      \
        float2_t, float, Stream&)

    #define NOA_INSTANTIATE_PROJECT_TEXTURE_ALL_REMAP(T, S, R)  \
    NOA_INSTANTIATE_INSERT_THICK_TEXTURE(T, Remap::HC2H, S, R); \
    NOA_INSTANTIATE_INSERT_THICK_TEXTURE(T, Remap::HC2HC, S, R);\
    NOA_INSTANTIATE_EXTRACT_TEXTURE(T, Remap::HC2H, S, R);      \
    NOA_INSTANTIATE_EXTRACT_TEXTURE(T, Remap::HC2HC, S, R)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP_TEXTURE(T, S0, S1, R0, R1)  \
    NOA_INSTANTIATE_INSERT_EXTRACT_TEXTURE_(T, Remap::HC2H, S0, S1, R0, R1);    \
    NOA_INSTANTIATE_INSERT_EXTRACT_TEXTURE_(T, Remap::HC2HC, S0, S1, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE_TEXTURE(T, R0, R1)                              \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP_TEXTURE(T, float22_t, float22_t, R0, R1);               \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP_TEXTURE(T, shared_t<float22_t[]>, float22_t, R0, R1);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP_TEXTURE(T, float22_t, shared_t<float22_t[]>, R0, R1);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP_TEXTURE(T, shared_t<float22_t[]>, shared_t<float22_t[]>, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE_TEXTURE(T)                             \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE_TEXTURE(T, float33_t, float33_t);               \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE_TEXTURE(T, shared_t<float33_t[]>, float33_t);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE_TEXTURE(T, float33_t, shared_t<float33_t[]>);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE_TEXTURE(T, shared_t<float33_t[]>, shared_t<float33_t[]>)

    #define NOA_INSTANTIATE_PROJECT_TEXTURE_ALL(T)                                              \
    NOA_INSTANTIATE_PROJECT_TEXTURE_ALL_REMAP(T, float22_t, float33_t);                         \
    NOA_INSTANTIATE_PROJECT_TEXTURE_ALL_REMAP(T, shared_t<float22_t[]>, float33_t);             \
    NOA_INSTANTIATE_PROJECT_TEXTURE_ALL_REMAP(T, float22_t, shared_t<float33_t[]>);             \
    NOA_INSTANTIATE_PROJECT_TEXTURE_ALL_REMAP(T, shared_t<float22_t[]>, shared_t<float33_t[]>); \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE_TEXTURE(T)

    NOA_INSTANTIATE_PROJECT_TEXTURE_ALL(float);
    NOA_INSTANTIATE_PROJECT_TEXTURE_ALL(cfloat_t);

    template void griddingCorrection<float, void>(const shared_t<float[]>&, dim4_t, const shared_t<float[]>&, dim4_t, dim4_t, bool, Stream&);
    template void griddingCorrection<double, void>(const shared_t<double[]>&, dim4_t, const shared_t<double[]>&, dim4_t, dim4_t, bool, Stream&);
}
