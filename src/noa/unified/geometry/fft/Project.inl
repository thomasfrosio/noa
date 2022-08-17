#pragma once

#ifndef NOA_UNIFIED_GEOMETRY_FFT_PROJECT_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/geometry/fft/Project.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Project.h"
#endif

namespace noa::geometry::fft {
    template<Remap REMAP, typename T, typename>
    void insert3D(const Array<T>& slice, size4_t slice_shape,
                  const Array<T>& grid, size4_t grid_shape,
                  const Array<float22_t>& scaling_factors,
                  const Array<float33_t>& rotations,
                  float cutoff, float sampling_factor, float2_t ews_radius) {
        NOA_CHECK(all(slice.shape() == slice_shape.fft()),
                  "The shape of the non-redundant slices do not match the expected shape. Got {} and expected {}",
                  slice.shape(), slice_shape.fft());
        NOA_CHECK(all(grid.shape() == grid_shape.fft()),
                  "The shape of the non-redundant grid do not match the expected shape. Got {} and expected {}",
                  grid.shape(), grid_shape.fft());
        NOA_CHECK(slice.shape()[1] == 1, "2D slices are expected but got shape {}", slice.shape());
        NOA_CHECK(grid.shape()[0] == 1 && grid_shape.ndim() == 3,
                  "A single 3D grid is expected but got shape {}", grid.shape());

        [[maybe_unused]] const size_t slices = slice.shape()[0];
        NOA_CHECK(scaling_factors.empty() ||
                  (indexing::isVector(scaling_factors.shape()) &&
                   scaling_factors.shape().elements() == slices &&
                   scaling_factors.contiguous()),
                  "The number of scaling factors, specified as a contiguous vector, should be equal to the number "
                  "of slices, but got {} scaling factors and {} slices", scaling_factors.shape().elements(), slices);
        NOA_CHECK(indexing::isVector(rotations.shape()) &&
                  rotations.shape().elements() == slices &&
                  rotations.contiguous(),
                  "The number of rotations, specified as a contiguous vector, should be equal to the number "
                  "of slices, but got {} rotations and {} slices", rotations.shape().elements(), slices);

        const Device device = grid.device();
        NOA_CHECK(slice.device() == device,
                  "The slices and the grid should be on the same device but got slice:{} and grid:{}",
                  slice.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(rotations.dereferenceable() &&
                      (scaling_factors.empty() || scaling_factors.dereferenceable()),
                      "The transform parameters should be accessible to the CPU");

            if (rotations.device().gpu())
                Stream::current(rotations.device()).synchronize();
            if (!scaling_factors.empty() && scaling_factors.device().gpu() &&
                rotations.device() != scaling_factors.device())
                Stream::current(scaling_factors.device()).synchronize();

            cpu::geometry::fft::insert3D<REMAP>(
                    slice.share(), slice.strides(), slice_shape,
                    grid.share(), grid.strides(), grid_shape,
                    scaling_factors.share(), rotations.share(),
                    cutoff, sampling_factor, ews_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if (rotations.device().cpu() ||
                (!scaling_factors.empty() && scaling_factors.device().cpu()))
                Stream::current(Device{}).synchronize();

            cuda::geometry::fft::insert3D<REMAP>(
                    slice.share(), slice.strides(), slice_shape,
                    grid.share(), grid.strides(), grid_shape,
                    scaling_factors.share(), rotations.share(),
                    cutoff, sampling_factor, ews_radius, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void extract3D(const Array<T>& grid, size4_t grid_shape,
                   const Array<T>& slice, size4_t slice_shape,
                   const Array<float22_t>& scaling_factors,
                   const Array<float33_t>& rotations,
                   float cutoff, float sampling_factor, float2_t ews_radius) {
        NOA_CHECK(all(slice.shape() == slice_shape.fft()),
                  "The shape of the non-redundant slices do not match the expected shape. Got {} and expected {}",
                  slice.shape(), slice_shape.fft());
        NOA_CHECK(all(grid.shape() == grid_shape.fft()),
                  "The shape of the non-redundant grid do not match the expected shape. Got {} and expected {}",
                  grid.shape(), grid_shape.fft());
        NOA_CHECK(slice.shape()[1] == 1, "2D slices are expected but got shape {}", slice.shape());
        NOA_CHECK(grid.shape()[0] == 1 && grid_shape.ndim() == 3,
                  "A single 3D grid is expected but got shape {}", grid.shape());

        [[maybe_unused]] const size_t slices = slice.shape()[0];
        NOA_CHECK(scaling_factors.empty() ||
                  (indexing::isVector(scaling_factors.shape()) &&
                   scaling_factors.shape().elements() == slices &&
                   scaling_factors.contiguous()),
                  "The number of scaling factors, specified as a contiguous vector, should be equal to the number "
                  "of slices, but got {} scaling factors and {} slices", scaling_factors.shape().elements(), slices);
        NOA_CHECK(indexing::isVector(rotations.shape()) &&
                  rotations.shape().elements() == slices &&
                  rotations.contiguous(),
                  "The number of rotations, specified as a contiguous vector, should be equal to the number "
                  "of slices, but got {} rotations and {} slices", rotations.shape().elements(), slices);

        const Device device = slice.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(grid.device() == device,
                      "The slices and the grid should be on the same device but got slice:{} and grid:{}",
                      device, grid.device());
            NOA_CHECK(rotations.dereferenceable() &&
                      (scaling_factors.empty() || scaling_factors.dereferenceable()),
                      "The transform parameters should be accessible to the CPU");

            if (rotations.device().gpu())
                Stream::current(rotations.device()).synchronize();
            if (!scaling_factors.empty() && scaling_factors.device().gpu() &&
                rotations.device() != scaling_factors.device())
                Stream::current(scaling_factors.device()).synchronize();

            cpu::geometry::fft::extract3D<REMAP>(
                    grid.share(), grid.strides(), grid_shape,
                    slice.share(), slice.strides(), slice_shape,
                    scaling_factors.share(), rotations.share(),
                    cutoff, sampling_factor, ews_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(indexing::isRightmost(grid.strides()) &&
                          indexing::isContiguous(grid.strides(), grid.shape())[1] && grid.strides()[3] == 1,
                          "The grid should be in the rightmost order and the depth and width dimension should be "
                          "contiguous, but got shape {} and strides {}", grid.shape(), grid.strides());
                if (rotations.device().cpu() ||
                    (!scaling_factors.empty() && scaling_factors.device().cpu()))
                    Stream::current(Device{}).synchronize();

                cuda::geometry::fft::extract3D<REMAP>(
                        grid.share(), grid.strides(), grid_shape,
                        slice.share(), slice.strides(), slice_shape,
                        scaling_factors.share(), rotations.share(),
                        cutoff, sampling_factor, ews_radius, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename T, typename>
    void extract3D(const Texture<T>& grid, size4_t grid_shape,
                   const Array<T>& slice, size4_t slice_shape,
                   const Array<float22_t>& scaling_factors,
                   const Array<float33_t>& rotations,
                   float cutoff, float sampling_factor, float2_t ews_radius) {
        if (grid.device().cpu()) {
            const cpu::Texture<T>& texture = grid.cpu();
            extract3D<REMAP>(Array<T>(texture.ptr, grid.shape(), texture.strides, grid.options()), grid_shape,
                             slice, slice_shape, scaling_factors, rotations, cutoff, sampling_factor, ews_radius);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
            NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
        } else {
            NOA_CHECK(all(slice.shape() == slice_shape.fft()),
                      "The shape of the non-redundant slices do not match the expected shape. Got {} and expected {}",
                      slice.shape(), slice_shape.fft());
            NOA_CHECK(all(grid.shape() == grid_shape.fft()),
                      "The shape of the non-redundant grid do not match the expected shape. Got {} and expected {}",
                      grid.shape(), grid_shape.fft());
            NOA_CHECK(grid.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", grid.shape()[0]);

            [[maybe_unused]] const size_t slices = slice.shape()[0];
            NOA_CHECK(scaling_factors.empty() ||
                      (indexing::isVector(scaling_factors.shape()) &&
                       scaling_factors.shape().elements() == slices &&
                       scaling_factors.contiguous()),
                      "The number of scaling factors, specified as a contiguous vector, should be equal to the number "
                      "of slices, but got {} scaling factors and {} slices", scaling_factors.shape().elements(), slices);
            NOA_CHECK(indexing::isVector(rotations.shape()) &&
                      rotations.shape().elements() == slices &&
                      rotations.contiguous(),
                      "The number of rotations, specified as a contiguous vector, should be equal to the number "
                      "of slices, but got {} rotations and {} slices", rotations.shape().elements(), slices);

            const Device device = slice.device();
            NOA_CHECK(device == grid.device(),
                      "The input and output must be on the same device, "
                      "but got input:{} and output:{}", grid.device(), device);

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = grid.cuda();
            cuda::geometry::fft::extract3D<REMAP>(
                    texture.array, texture.texture, int3_t(grid.shape().get(1)),
                    slice.share(), slice.strides(), slice.shape(),
                    scaling_factors.share(), rotations.share(),
                    cutoff, sampling_factor, ews_radius, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename T, typename>
    void griddingCorrection(const Array<T>& input, const Array<T>& output, bool post_correction) {
        size4_t input_strides = input.strides();
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
