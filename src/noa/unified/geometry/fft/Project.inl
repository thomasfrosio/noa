#pragma once

#ifndef NOA_UNIFIED_GEOMETRY_FFT_PROJECT_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/geometry/fft/Project.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Project.h"
#endif

namespace noa::geometry::fft::details {
    template<typename T, typename U, typename S, typename R>
    auto parseProjectInputs(const T& slice, dim4_t slice_shape,
                            const U& grid, dim4_t grid_shape,
                            const S& scaling_factors, const R& rotations,
                            dim4_t target_shape) {
        const Device device = grid.device();
        NOA_CHECK(slice.device() == device,
                  "The slices and the grid should be on the same device but got slice:{} and grid:{}",
                  slice.device(), device);

        NOA_CHECK(all(slice.shape() == slice_shape.fft()),
                  "The shape of the non-redundant slices do not match the expected shape. Got {} and expected {}",
                  slice.shape(), slice_shape.fft());
        NOA_CHECK(all(grid.shape() == grid_shape.fft()),
                  "The shape of the non-redundant grid do not match the expected shape. Got {} and expected {}",
                  grid.shape(), grid_shape.fft());
        NOA_CHECK(slice.shape()[1] == 1, "2D slices are expected but got shape {}", slice.shape());

        if (any(target_shape == 0)) {
            NOA_CHECK(grid.shape()[0] == 1 && grid_shape.ndim() == 3,
                      "A single 3D grid is expected but got shape {}", grid.shape());
        } else {
            NOA_CHECK(grid.shape()[0] == 1 && target_shape[0] == 1 && target_shape.ndim() == 3,
                      "A single grid is expected, with a target shape describing a single 3D volume, "
                      "but got grid shape {} and target shape", grid.shape(), target_shape);
        }

        [[maybe_unused]] const size_t slices = slice.shape()[0];
        bool sync_cpu{false};

        constexpr bool SINGLE_SCALING = traits::is_float22_v<S>;
        using scaling_t = traits::shared_type_t<S>;
        const scaling_t* scaling_factors_;
        if constexpr (!SINGLE_SCALING) {
            NOA_CHECK(scaling_factors.empty() ||
                      (indexing::isVector(scaling_factors.shape()) &&
                       scaling_factors.elements() == slices &&
                       scaling_factors.contiguous()),
                      "The number of scaling factors, specified as a contiguous vector, should be equal to the number "
                      "of slices, but got {} scaling factors and {} slices", scaling_factors.elements(), slices);
            scaling_factors_ = &scaling_factors.share();

            if (device.cpu()) {
                NOA_CHECK(scaling_factors.empty() || scaling_factors.dereferenceable(),
                          "The transform parameters should be accessible to the CPU");
                if (!scaling_factors.empty() && scaling_factors.device().gpu())
                    Stream::current(scaling_factors.device()).synchronize();
            } else {
                if (!scaling_factors.empty() && scaling_factors.device().cpu())
                    sync_cpu = true;
            }
        } else {
            scaling_factors_ = &scaling_factors;
        }

        constexpr bool SINGLE_ROTATION = traits::is_float33_v<R>;
        using rotation_t = traits::shared_type_t<R>;
        const rotation_t* rotations_;
        if constexpr (!SINGLE_ROTATION) {
            NOA_CHECK(indexing::isVector(rotations.shape()) &&
                      rotations.elements() == slices &&
                      rotations.contiguous(),
                      "The number of rotations, specified as a contiguous vector, should be equal to the number "
                      "of slices, but got {} rotations and {} slices", rotations.elements(), slices);
            rotations_ = &rotations.share();

            if (device.cpu()) {
                NOA_CHECK(rotations.dereferenceable(), "The rotation matrices should be accessible to the CPU");
                if constexpr (SINGLE_SCALING) {
                    if (rotations.device().gpu())
                        Stream::current(rotations.device()).synchronize();
                } else {
                    if (rotations.device().gpu() && rotations.device() != scaling_factors.device())
                        Stream::current(rotations.device()).synchronize();
                }
            } else {
                if (rotations.device().cpu() || sync_cpu)
                    Stream::current(Device{}).synchronize();
            }
        } else {
            rotations_ = &rotations;
        }

        return std::pair{*scaling_factors_, *rotations_};
    }
}

namespace noa::geometry::fft {
    template<Remap REMAP, typename T, typename S, typename R, typename>
    void insert3D(const Array<T>& slice, dim4_t slice_shape,
                  const Array<T>& grid, dim4_t grid_shape,
                  const S& scaling_factors, const R& rotations,
                  float cutoff, dim4_t target_shape, float2_t ews_radius) {
        NOA_CHECK(!slice.empty() && !grid.empty(), "Empty array detected");
        NOA_CHECK(!indexing::isOverlap(slice, grid), "Input and output arrays should not overlap");

        const Device device = grid.device();
        NOA_CHECK(slice.device() == device,
                  "The slices and the grid should be on the same device but got slice:{} and grid:{}",
                  slice.device(), device);

        const auto[scaling_factors_, rotations_] = details::parseProjectInputs(
                slice, slice_shape, grid, grid_shape, scaling_factors, rotations, target_shape);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::insert3D<REMAP>(
                    slice.share(), slice.strides(), slice_shape,
                    grid.share(), grid.strides(), grid_shape,
                    scaling_factors_, rotations_,
                    cutoff, target_shape, ews_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::fft::insert3D<REMAP>(
                    slice.share(), slice.strides(), slice_shape,
                    grid.share(), grid.strides(), grid_shape,
                    scaling_factors_, rotations_,
                    cutoff, target_shape, ews_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename S, typename R, typename>
    void extract3D(const Array<T>& grid, dim4_t grid_shape,
                   const Array<T>& slice, dim4_t slice_shape,
                   const S& scaling_factors, const R& rotations,
                   float cutoff, dim4_t target_shape, float2_t ews_radius) {
        NOA_CHECK(!slice.empty() && !grid.empty(), "Empty array detected");
        NOA_CHECK(!indexing::isOverlap(slice, grid), "Input and output arrays should not overlap");

        const Device device = slice.device();
        NOA_CHECK(grid.device() == device,
                  "The slices and the grid should be on the same device but got slice:{} and grid:{}",
                  device, grid.device());

        const auto[scaling_factors_, rotations_] = details::parseProjectInputs(
                slice, slice_shape, grid, grid_shape, scaling_factors, rotations, target_shape);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::extract3D<REMAP>(
                    grid.share(), grid.strides(), grid_shape,
                    slice.share(), slice.strides(), slice_shape,
                    scaling_factors_, rotations_,
                    cutoff, target_shape, ews_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            constexpr bool USE_TEXTURE = false;
            cuda::geometry::fft::extract3D<REMAP>(
                    grid.share(), grid.strides(), grid_shape,
                    slice.share(), slice.strides(), slice_shape,
                    scaling_factors_, rotations_,
                    cutoff, target_shape, ews_radius, USE_TEXTURE, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename S, typename R, typename>
    void extract3D(const Texture<T>& grid, dim4_t grid_shape,
                   const Array<T>& slice, dim4_t slice_shape,
                   const S& scaling_factors, const R& rotations,
                   float cutoff, dim4_t target_shape, float2_t ews_radius) {
        NOA_CHECK(!slice.empty() && !grid.empty(), "Empty array detected");

        if (grid.device().cpu()) {
            const cpu::Texture<T>& texture = grid.cpu();
            extract3D<REMAP>(Array<T>(texture.ptr, grid.shape(), texture.strides, grid.options()), grid_shape,
                             slice, slice_shape, scaling_factors, rotations, cutoff, target_shape, ews_radius);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
            NOA_THROW("In the CUDA backend, double-precision floating-points are not supported by this function");
        } else {
            const Device device = slice.device();
            NOA_CHECK(device == grid.device(),
                      "The input and output must be on the same device, "
                      "but got input:{} and output:{}", grid.device(), device);

            const auto[scaling_factors_, rotations_] = details::parseProjectInputs(
                    slice, slice_shape, grid, grid_shape, scaling_factors, rotations, target_shape);

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = grid.cuda();
            cuda::geometry::fft::extract3D<REMAP>(
                    texture.array, texture.texture, int3_t(grid.shape().get(1)),
                    slice.share(), slice.strides(), slice.shape(),
                    scaling_factors_, rotations_,
                    cutoff, target_shape, ews_radius, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename T, typename>
    void griddingCorrection(const Array<T>& input, const Array<T>& output, bool post_correction) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");

        dim4_t input_strides = input.strides();
        if (!indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::griddingCorrection(
                    input.share(), input_strides,
                    output.share(), output.strides(),
                    output.shape(), post_correction, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::fft::griddingCorrection(
                    input.share(), input_strides,
                    output.share(), output.strides(),
                    output.shape(), post_correction, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
