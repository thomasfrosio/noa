#pragma once

#ifndef NOA_UNIFIED_GEOMETRY_FFT_PROJECT_
#error "This is a private header"
#endif

#include "noa/cpu/geometry/fft/Project.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Project.h"
#endif

namespace noa::geometry::fft {
    template<Remap REMAP, typename T>
    void insert3D(const Array<T>& slice, size4_t slice_shape,
                  const Array<T>& grid, size4_t grid_shape,
                  const Array<float22_t>& scaling_factors,
                  const Array<float33_t>& rotations,
                  float cutoff, float2_t ews_radius) {
        NOA_CHECK(all(slice.shape() == slice_shape.fft()),
                  "The shape of the non-redundant slices do not match the expected shape. Got {} and expected {}",
                  slice.shape(), slice_shape.fft());
        NOA_CHECK(all(grid.shape() == grid_shape.fft()),
                  "The shape of the non-redundant grid do not match the expected shape. Got {} and expected {}",
                  grid.shape(), grid_shape.fft());
        NOA_CHECK(slice.shape()[1] == 1, "2D slices are expected but got shape {}", slice.shape());
        NOA_CHECK(grid.shape()[0] == 1, "A single 3D grid is expected but got shape {}", grid.shape());

        const size_t slices = slice.shape()[0];
        NOA_CHECK(scaling_factors.empty() ||
                  (scaling_factors.shape()[3] == slices &&
                   scaling_factors.shape().ndim() == 1 && all(scaling_factors.contiguous())),
                  "The number of scaling factors, specified as a contiguous row vector, should be equal to the number "
                  "of slices, but got {} scaling factors and {} slices", scaling_factors.shape()[3], slices);
        NOA_CHECK(rotations.shape()[3] == slices && rotations.shape().ndim() == 1 && all(rotations.contiguous()),
                  "The number of rotations, specified as a contiguous row vector, should be equal to the number "
                  "of slices, but got {} rotations and {} slices", rotations.shape()[3], slices);

        const Device device = grid.device();
        NOA_CHECK(slice.device() == device,
                  "The slices and the grid should be on the same device but got slice:{} and grid:{}",
                  slice.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(rotations.dereferencable() &&
                      (scaling_factors.empty() || scaling_factors.dereferencable()),
                      "The transform parameters should be accessible to the CPU");

            if (rotations.device().gpu())
                Stream::current(rotations.device()).synchronize();
            if (!scaling_factors.empty() && scaling_factors.device().gpu() &&
                rotations.device() != scaling_factors.device())
                Stream::current(scaling_factors.device()).synchronize();

            cpu::geometry::fft::insert3D<REMAP>(
                    slice.share(), slice.stride(), slice_shape,
                    grid.share(), grid.stride(), grid_shape,
                    scaling_factors.share(), rotations.share(),
                    cutoff, ews_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if (rotations.device().cpu() ||
                (!scaling_factors.empty() && scaling_factors.device().cpu()))
                Stream::current(Device{}).synchronize();

            cuda::geometry::fft::insert3D<REMAP>(
                    slice.share(), slice.stride(), slice_shape,
                    grid.share(), grid.stride(), grid_shape,
                    scaling_factors.share(), rotations.share(),
                    cutoff, ews_radius, stream.cuda());
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
                   float cutoff, float2_t ews_radius) {
        NOA_CHECK(all(slice.shape() == slice_shape.fft()),
                  "The shape of the non-redundant slices do not match the expected shape. Got {} and expected {}",
                  slice.shape(), slice_shape.fft());
        NOA_CHECK(all(grid.shape() == grid_shape.fft()),
                  "The shape of the non-redundant grid do not match the expected shape. Got {} and expected {}",
                  grid.shape(), grid_shape.fft());
        NOA_CHECK(slice.shape()[1] == 1, "2D slices are expected but got shape {}", slice.shape());
        NOA_CHECK(grid.shape()[0] == 1, "A single 3D grid is expected but got shape {}", grid.shape());

        const size_t slices = slice.shape()[0];
        NOA_CHECK(scaling_factors.empty() ||
                  (scaling_factors.shape()[3] == slices &&
                   scaling_factors.shape().ndim() == 1 && all(scaling_factors.contiguous())),
                  "The number of scaling factors, specified as a contiguous row vector, should be equal to the number "
                  "of slices, but got {} scaling factors and {} slices", scaling_factors.shape()[3], slices);
        NOA_CHECK(rotations.shape()[3] == slices && rotations.shape().ndim() == 1 && all(rotations.contiguous()),
                  "The number of rotations, specified as a contiguous row vector, should be equal to the number "
                  "of slices, but got {} rotations and {} slices", rotations.shape()[3], slices);

        const Device device = slice.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(grid.device() == device,
                      "The slices and the grid should be on the same device but got slice:{} and grid:{}",
                      grid.device(), device);
            NOA_CHECK(rotations.dereferencable() &&
                      (scaling_factors.empty() || scaling_factors.dereferencable()),
                      "The transform parameters should be accessible to the CPU");

            if (rotations.device().gpu())
                Stream::current(rotations.device()).synchronize();
            if (!scaling_factors.empty() && scaling_factors.device().gpu() &&
                rotations.device() != scaling_factors.device())
                Stream::current(scaling_factors.device()).synchronize();

            cpu::geometry::fft::extract3D<REMAP>(
                    grid.share(), grid.stride(), grid_shape,
                    slice.share(), slice.stride(), slice_shape,
                    scaling_factors.share(), rotations.share(),
                    cutoff, ews_radius, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                NOA_CHECK(indexing::isContiguous(grid.stride(), grid.shape())[1] && grid.stride()[3] == 1,
                          "The third-most and innermost dimension of the grid should be contiguous, but got shape {} "
                          "and stride {}", grid.shape(), grid.stride());
                if (rotations.device().cpu() ||
                    (!scaling_factors.empty() && scaling_factors.device().cpu()))
                    Stream::current(Device{}).synchronize();

                cuda::geometry::fft::extract3D<REMAP>(
                        grid.share(), grid.stride(), grid_shape,
                        slice.share(), slice.stride(), slice_shape,
                        scaling_factors.share(), rotations.share(),
                        cutoff, ews_radius, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void griddingCorrection(const Array<T>& input, const Array<T>& output, bool post_correction) {
        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
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
                    input.share(), input_stride,
                    output.share(), output.stride(),
                    output.shape(), post_correction, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::fft::griddingCorrection(
                    input.share(), input_stride,
                    output.share(), output.stride(),
                    output.shape(), post_correction, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
