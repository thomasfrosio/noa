#pragma once

#include "noa/core/geometry/Prefilter.hpp"
#include "noa/gpu/cuda/Types.hpp"

// The implementation requires a single thread to go through the entire 1D array. This is not very efficient
// compared to the CPU implementation. However, when multiple batches are processes, a warp can simultaneously process
//  as many batches as it has threads, which is more efficient.
namespace noa::cuda::geometry::guts {
    template<typename T>
    __global__ void cubic_bspline_prefilter_1d_x_inplace(T* input, Strides2<u32> strides, Shape2<u32> shape) {
        // process lines in x-direction
        const u32 batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= shape[0])
            return;
        input += batch * strides[0];
        noa::geometry::BSplinePrefilter1d<T, i32>::filter_inplace(input, strides[1], shape[1]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_1d_x(
            const T* __restrict__ input, Strides2<u32> input_strides,
            T* __restrict__ output, Strides2<u32> output_strides,
            Shape2<u32> shape
    ) {
        // process lines in x-direction
        const u32 batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= shape[0])
            return;
        input += batch * input_strides[0];
        output += batch * output_strides[0];
        noa::geometry::BSplinePrefilter1d<T, i32>::filter(
                input, input_strides[1], output, output_strides[1], shape[1]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_2d_x_inplace(Accessor<T, 3, u32> input, Shape2<u32> shape) {
        // process lines in x-direction
        const u32 y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= shape[0])
            return;
        const auto input_1d = input[blockIdx.y][y]; // blockIdx.y == batch
        noa::geometry::BSplinePrefilter1d<T, i32>::filter_inplace(
                input_1d.get(), input_1d.template stride<0>(), shape[1]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_2d_x(
            AccessorRestrict<const T, 3, u32> input,
            AccessorRestrict<T, 3, u32> output,
            Shape2<u32> shape
    ) {
        // process lines in x-direction
        const u32 y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= shape[0])
            return;
        const auto input_1d = input[blockIdx.y][y];
        const auto output_1d = output[blockIdx.y][y];
        noa::geometry::BSplinePrefilter1d<T, i32>::filter(
                input_1d.get(), input_1d.template stride<0>(),
                output.get(), output_1d.template stride<0>(), shape[1]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_2d_y(T* input, Strides3<u32> strides, Shape2<u32> shape) {
        // process lines in y-direction
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= shape[1])
            return;
        input += blockIdx.y * strides[0] + x * strides[2];
        noa::geometry::BSplinePrefilter1d<T, i32>::filter_inplace(input, strides[1], shape[0]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_3d_x_inplace(Accessor<T, 4, u32> input, Shape3<u32> shape) {
        // process lines in x-direction
        const u32 y = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] || y >= shape[1])
            return;
        const auto input_1d = input[blockIdx.z][z][y];
        noa::geometry::BSplinePrefilter1d<T, i32>::filter_inplace(
                input_1d.get(), input_1d.template stride<0>(), shape[2]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_3d_x(
            AccessorRestrict<const T, 4, u32> input,
            AccessorRestrict<T, 4, u32> output,
            Shape3<u32> shape
    ) {
        // process lines in x-direction
        const u32 y = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] || y >= shape[1])
            return;
        const auto input_1d = input[blockIdx.z][z][y];
        const auto output_1d = output[blockIdx.z][z][y];
        noa::geometry::BSplinePrefilter1d<T, i32>::filter(
                input_1d.get(), input_1d.template stride<0>(),
                output_1d.get(), output_1d.template stride<0>(), shape[2]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_3d_y(T* input, Strides4<u32> strides, Shape3<u32> shape) {
        // process lines in y-direction
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] || x >= shape[2])
            return;
        input += ni::offset_at(blockIdx.z, z, strides) + x * strides[3];
        noa::geometry::BSplinePrefilter1d<T, i32>::filter_inplace(input, strides[2], shape[1]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_3d_z(T* input, Strides4<u32> strides, Shape3<u32> shape) {
        // process lines in z-direction
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y >= shape[1] || x >= shape[2])
            return;
        input += blockIdx.z * strides[0] + y * strides[2] + x * strides[3];
        noa::geometry::BSplinePrefilter1d<T, i32>::filter_inplace(input, strides[1], shape[0]);
    }
}
