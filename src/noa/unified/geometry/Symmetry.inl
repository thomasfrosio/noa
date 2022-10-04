#pragma once

#ifndef NOA_UNIFIED_SYMMETRY_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/geometry/Symmetry.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Symmetry.h"
#endif

namespace noa::geometry {
    template<typename T, typename>
    void symmetrize2D(const Array<T>& input, const Array<T>& output,
                      const Symmetry& symmetry, float2_t center,
                      InterpMode interp_mode, bool prefilter, bool normalize) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");
        NOA_CHECK(input.shape()[3] == output.shape()[3] &&
                  input.shape()[2] == output.shape()[2],
                  "The input {} and output {} shapes don't match", input.shape(), output.shape());
        NOA_CHECK(input.shape()[1] == 1 && output.shape()[1] == 1,
                  "The input and output arrays should be 2D, but got shape input:{}, output:{}",
                  input.shape(), output.shape());
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);

        size4_t input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(!indexing::isOverlap(input, output), "Input and output arrays should not overlap");

            cpu::geometry::symmetrize2D(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    symmetry, center, interp_mode, prefilter, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == input.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", input.device(), device);
                NOA_CHECK(indexing::isRightmost(input_strides) && input_strides[3] == 1,
                          "The input should be in the rightmost order and the width dimension should be contiguous, "
                          "but got shape {} and stride {}", input.shape(), input_strides);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::symmetrize2D(
                        input.share(), input_strides,
                        output.share(), output.strides(), output.shape(),
                        symmetry, center, interp_mode, prefilter, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void symmetrize2D(const Texture<T>& input, const Array<T>& output,
                      const Symmetry& symmetry, float2_t center,
                      bool normalize) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");

        if (input.device().cpu()) {
            const cpu::Texture<T>& texture = input.cpu();
            symmetry2D(Array<T>(texture.ptr, input.shape(), texture.strides, input.options()), output,
                       symmetry, center, input.interp(), false, normalize);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
            NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
        } else {
            NOA_CHECK(input.shape()[3] == output.shape()[3] &&
                      input.shape()[2] == output.shape()[2],
                      "The input {} and output {} shapes don't match", input.shape(), output.shape());
            NOA_CHECK(input.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", input.shape()[0]);
            NOA_CHECK(input.shape()[1] == 1 && output.shape()[1] == 1,
                      "The input and output arrays should be 2D, but got shape input:{}, output:{}",
                      input.shape(), output.shape());

            const Device device = output.device();
            NOA_CHECK(device == input.device(),
                      "The input and output must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = input.cuda();
            cuda::geometry::symmetrize2D(
                    texture.array, texture.texture, input.interp(), output.share(), output.strides(), output.shape(),
                    symmetry, center, normalize, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }

    template<typename T, typename>
    void symmetrize3D(const Array<T[]>& input, const Array<T[]>& output,
                      const Symmetry& symmetry, float3_t center,
                      InterpMode interp_mode, bool prefilter, bool normalize) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");
        NOA_CHECK(input.shape()[3] == input.shape()[3] &&
                  input.shape()[2] == input.shape()[2] &&
                  input.shape()[1] == input.shape()[1],
                  "The input {} and output {} shapes don't match", input.shape(), output.shape());
        NOA_CHECK(input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                  "The number of batches in the input ({}) is not compatible with the number of "
                  "batches in the output ({})", input.shape()[0], output.shape()[0]);

        size4_t input_strides = input.strides();
        if (input.shape()[0] == 1)
            input_strides[0] = 0;

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == input.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);
            NOA_CHECK(!indexing::isOverlap(input, output), "Input and output arrays should not overlap");

            cpu::geometry::symmetrize3D(
                    input.share(), input_strides,
                    output.share(), output.strides(), output.shape(),
                    symmetry, center, interp_mode, prefilter, normalize, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (sizeof(traits::value_type_t<T>) >= 8) {
                NOA_THROW("Double-precision floating-points are not supported");
            } else {
                const bool do_prefilter =
                        prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST);
                NOA_CHECK(!do_prefilter || device == input.device(),
                          "The input and output arrays must be on the same device, "
                          "but got input:{} and output:{}", input.device(), device);
                NOA_CHECK(indexing::isRightmost(input_strides) &&
                          indexing::isContiguous(input_strides, input.shape())[1] && input_strides[3] == 1,
                          "The input should be in the rightmost order and the height and width should be contiguous, "
                          "but got shape {} and strides {}", input.shape(), input_strides);

                if (input.device().cpu())
                    Stream::current(Device{}).synchronize();
                cuda::geometry::symmetrize3D(
                        input.share(), input_strides,
                        output.share(), output.strides(), output.shape(),
                        symmetry, center, interp_mode, prefilter, normalize, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename T, typename>
    void symmetrize3D(const Texture<T>& input, const Array<T>& output,
                      const Symmetry& symmetry, float3_t center,
                      bool normalize) {
        NOA_CHECK(!input.empty() && !output.empty(), "Empty array detected");

        if (input.device().cpu()) {
            const cpu::Texture<T>& texture = input.cpu();
            symmetry3D(Array<T>(texture.ptr, input.shape(), texture.strides, input.options()), output,
                       symmetry, center, input.interp(), false, normalize);
            return;
        }

        #ifdef NOA_ENABLE_CUDA
        if constexpr (!traits::is_any_v<T, float, cfloat_t>) {
            NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
        } else {
            NOA_CHECK(input.shape()[3] == input.shape()[3] &&
                      input.shape()[2] == input.shape()[2] &&
                      input.shape()[1] == input.shape()[1],
                      "The input {} and output {} shapes don't match", input.shape(), output.shape());
            NOA_CHECK(input.shape()[0] == 1,
                      "The number of batches in the texture ({}) should be 1, got {}", input.shape()[0]);

            const Device device = output.device();
            NOA_CHECK(device == input.device(),
                      "The input and output must be on the same device, "
                      "but got input:{} and output:{}", input.device(), device);

            Stream& stream = Stream::current(device);
            const cuda::Texture<T>& texture = input.cuda();
            cuda::geometry::symmetrize3D(
                    texture.array, texture.texture, input.interp(), output.share(), output.strides(), output.shape(),
                    symmetry, center, normalize, stream.cuda());
        }
        #else
        NOA_THROW("No GPU backend detected");
        #endif
    }
}
