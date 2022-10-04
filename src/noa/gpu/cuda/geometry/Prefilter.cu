#include "noa/common/Math.h"

#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/geometry/Interpolate.h"

// This is adapted from https://github.com/DannyRuijters/CubicInterpolationCUDA
// See licences/CubicInterpolationCUDA.txt
// Pitch/step was switched to number of elements. const was added when necessary. Out-of-place filtering was added.

// The implementation requires a single thread to go through the entire 1D array. This is not very efficient
// compared to the CPU implementation. However, when multiple batches are processes, a warp can process
// simultaneously as many batches as it has threads, which is more efficient.
namespace {
    using namespace ::noa;
    constexpr float POLE = -0.2679491924311228f; // math::sqrt(3.0f)-2.0f; pole for cubic b-spline

    template<typename T> // float or float2
    __device__ T initialCausalCoefficient_(const T* c, uint32_t stride, uint32_t shape) {
        const uint32_t horizon = math::min(12U, shape);

        // this initialization corresponds to clamping boundaries accelerated loop
        float zn = POLE;
        T sum = *c;
        for (uint32_t n = 0; n < horizon; n++) {
            sum += zn * *c;
            zn *= POLE;
            c += stride;
        }
        return sum;
    }

    template<typename T>
    __forceinline__ __device__ T initialAntiCausalCoefficient_(const T* c) {
        // this initialization corresponds to clamping boundaries
        return ((POLE / (POLE - 1.0f)) * *c);
    }

    template<typename T>
    __device__ void toCoeffs_(T* output, uint32_t stride, uint32_t shape) {
        // compute the overall gain
        const float lambda = (1.0f - POLE) * (1.0f - 1.0f / POLE);

        // causal initialization and recursion
        T* c = output;
        T previous_c;  //cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = lambda * initialCausalCoefficient_(c, stride, shape);
        for (uint32_t n = 1; n < shape; n++) {
            c += stride;
            *c = previous_c = lambda * *c + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initialAntiCausalCoefficient_(c);
        for (int32_t n = static_cast<int32_t>(shape) - 2; 0 <= n; n--) {
            c -= stride;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }

    template<typename T>
    __device__ void toCoeffs_(const T* __restrict__ input, uint32_t input_stride,
                              T* __restrict__ output, uint32_t output_stride,
                              uint32_t shape) {
        // compute the overall gain
        const float lambda = (1.0f - POLE) * (1.0f - 1.0f / POLE);

        // causal initialization and recursion
        T* c = output;
        T previous_c;  // cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = lambda * initialCausalCoefficient_(input, input_stride, shape);
        for (uint32_t n = 1; n < shape; n++) {
            input += input_stride;
            c += output_stride;
            *c = previous_c = lambda * *input + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initialAntiCausalCoefficient_(c);
        for (int32_t n = static_cast<int32_t>(shape) - 2; 0 <= n; n--) {
            c -= output_stride;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }

    // -- 1D -- //

    template<typename T>
    __global__ void toCoeffs1DX_inplace_(T* input, uint2_t strides, uint2_t shape) {
        // process lines in x-direction
        const uint32_t batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= shape[0])
            return;
        input += batch * strides[0];
        toCoeffs_(input, strides[1], shape[1]);
    }

    template<typename T>
    __global__ void toCoeffs1DX_(const T* __restrict__ input, uint2_t input_strides,
                                 T* __restrict__ output, uint2_t output_strides,
                                 uint2_t shape) {
        // process lines in x-direction
        const uint32_t batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= shape[0])
            return;
        input += batch * input_strides[0];
        output += batch * output_strides[0];
        toCoeffs_(input, input_strides[1], output, output_strides[1], shape[1]);
    }

    // -- 2D -- //

    template<typename T>
    __global__ void toCoeffs2DX_inplace_(Accessor<T, 3, uint32_t> input, uint2_t shape) {
        // process lines in x-direction
        const uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= shape[0])
            return;
        const auto input_1d = input[blockIdx.y][y]; // blockIdx.y == batch
        toCoeffs_(input_1d.get(), input_1d.stride(0), shape[1]);
    }

    template<typename T>
    __global__ void toCoeffs2DX_(AccessorRestrict<const T, 3, uint32_t> input,
                                 AccessorRestrict<T, 3, uint32_t> output,
                                 uint2_t shape) {
        // process lines in x-direction
        const uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= shape[0])
            return;
        const auto input_1d = input[blockIdx.y][y];
        const auto output_1d = output[blockIdx.y][y];
        toCoeffs_(input_1d.get(), input_1d.stride(0), output.get(), output_1d.stride(0), shape[1]);
    }

    template<typename T>
    __global__ void toCoeffs2DY_(T* input, uint3_t strides, uint2_t shape) {
        // process lines in y-direction
        const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= shape[1])
            return;
        input += blockIdx.y * strides[0] + x * strides[2];
        toCoeffs_(input, strides[1], shape[0]);
    }

    // -- 3D -- //

    template<typename T>
    __global__ void toCoeffs3DX_inplace_(Accessor<T, 4, uint32_t> input, uint3_t shape) {
        // process lines in x-direction
        const uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] || y >= shape[1])
            return;
        const auto input_1d = input[blockIdx.z][z][y];
        toCoeffs_(input_1d.get(), input_1d.stride(0), shape[2]);
    }

    template<typename T>
    __global__ void toCoeffs3DX_(AccessorRestrict<const T, 4, uint32_t> input,
                                 AccessorRestrict<T, 4, uint32_t> output,
                                 uint3_t shape) {
        // process lines in x-direction
        const uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] || y >= shape[1])
            return;
        const auto input_1d = input[blockIdx.z][z][y];
        const auto output_1d = output[blockIdx.z][z][y];
        toCoeffs_(input_1d.get(), input_1d.stride(0), output_1d.get(), output_1d.stride(0), shape[2]);
    }

    template<typename T>
    __global__ void toCoeffs3DY_(T* input, uint4_t strides, uint3_t shape) {
        // process lines in y-direction
        const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] || x >= shape[2])
            return;
        input += indexing::at(blockIdx.z, z, strides) + x * strides[3];
        toCoeffs_(input, strides[2], shape[1]);
    }

    template<typename T>
    __global__ void toCoeffs3DZ_(T* input, uint4_t strides, uint3_t shape) {
        // process lines in z-direction
        const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y >= shape[1] || x >= shape[2])
            return;
        input += blockIdx.z * strides[0] + y * strides[2] + x * strides[3];
        toCoeffs_(input, strides[1], shape[0]);
    }

    void getLaunchConfig3D(uint32_t dim0, uint32_t dim1, dim3* threads, dim3* blocks) {
        threads->x = dim0 <= 32U ? 32U : 64U; // either 32 or 64 threads in the first dimension
        threads->y = math::min(math::nextMultipleOf(dim1, 32U), 512U / threads->x); // 2D block up to 512 threads
        blocks->x = math::divideUp(dim0, threads->x);
        blocks->y = math::divideUp(dim1, threads->y);
    }

    template<typename T>
    void prefilter1D_(const T* input, uint2_t input_strides, T* output, uint2_t output_strides,
                      uint2_t shape, cuda::Stream& stream) {
        // Each threads processes an entire batch.
        // This has the same problem as the toCoeffs2DX_ and toCoeffs3DX_, memory reads/writes are not coalesced.
        const uint32_t threads = math::nextMultipleOf(shape[0], 32U);
        const uint32_t blocks = math::divideUp(shape[0], threads);
        const cuda::LaunchConfig config{blocks, threads};

        if (input == output) {
            stream.enqueue("geometry::bspline::prefilter1D", toCoeffs1DX_inplace_<T>, config,
                           output, output_strides, shape);
        } else {
            stream.enqueue("geometry::bspline::prefilter1D", toCoeffs1DX_<T>, config,
                           input, input_strides, output, output_strides, shape);
        }
    }

    template<typename T>
    void prefilter2D_(const T* input, uint3_t input_strides, T* output, uint3_t output_strides,
                      uint3_t shape, cuda::Stream& stream) {
        // Each threads processes an entire line. The line is first x, then y.
        const uint32_t threads_x = shape[1] <= 32U ? 32U : 64U;
        const uint32_t threads_y = shape[2] <= 32U ? 32U : 64U;
        const dim3 blocks_x(math::divideUp(shape[1], threads_x), shape[0]);
        const dim3 blocks_y(math::divideUp(shape[2], threads_y), shape[0]);
        const cuda::LaunchConfig config_x{blocks_x, threads_x};
        const cuda::LaunchConfig config_y{blocks_y, threads_y};

        if (input == output) {
            const Accessor<T, 3, uint32_t> accessor(output, output_strides);
            stream.enqueue("geometry::bspline::prefilter2D_x", toCoeffs2DX_inplace_<T>, config_x,
                           accessor, uint2_t{shape[1], shape[2]});
        } else {
            const AccessorRestrict<const T, 3, uint32_t> input_accessor(input, input_strides);
            const AccessorRestrict<T, 3, uint32_t> output_accessor(output, output_strides);
            stream.enqueue("geometry::bspline::prefilter2D_x", toCoeffs2DX_<T>, config_x,
                           input_accessor, output_accessor, uint2_t{shape[1], shape[2]});
        }
        stream.enqueue("geometry::bspline::prefilter2D_y", toCoeffs2DY_<T>, config_y,
                       output, output_strides, uint2_t{shape[1], shape[2]});
    }

    template<typename T>
    void prefilter3D_(const T* input, uint4_t input_strides, T* output, uint4_t output_strides,
                      uint4_t shape, cuda::Stream& stream) {
        // Try to determine the optimal block dimensions
        dim3 threads;
        dim3 blocks;
        threads.z = 1;
        blocks.z = shape[0];

        getLaunchConfig3D(shape[2], shape[1], &threads, &blocks);
        if (input == output) {
            const Accessor<T, 4, uint32_t> accessor(output, output_strides);
            stream.enqueue("geometry::bspline::prefilter3D_x", toCoeffs3DX_inplace_<T>, {blocks, threads},
                           accessor, uint3_t{shape[1], shape[2], shape[3]});
        } else {
            const AccessorRestrict<const T, 4, uint32_t> input_accessor(input, input_strides);
            const AccessorRestrict<T, 4, uint32_t> output_accessor(output, output_strides);
            stream.enqueue("geometry::bspline::prefilter3D_x", toCoeffs3DX_<T>, {blocks, threads},
                           input_accessor, output_accessor, uint3_t{shape[1], shape[2], shape[3]});
        }

        getLaunchConfig3D(shape[3], shape[1], &threads, &blocks);
        stream.enqueue("geometry::bspline::prefilter3D_y", toCoeffs3DY_<T>, {blocks, threads},
                       output, output_strides, uint3_t{shape[1], shape[2], shape[3]});

        getLaunchConfig3D(shape[3], shape[2], &threads, &blocks);
        stream.enqueue("geometry::bspline::prefilter3D_z", toCoeffs3DZ_<T>, {blocks, threads},
                       output, output_strides, uint3_t{shape[1], shape[2], shape[3]});
    }
}

namespace noa::cuda::geometry::bspline {
    template<typename T, typename>
    void prefilter(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides,
                   dim4_t shape, Stream& stream) {
        const auto input_strides_ = safe_cast<uint4_t>(input_strides);
        const auto output_strides_ = safe_cast<uint4_t>(output_strides);
        const auto shape_ = safe_cast<uint4_t>(shape);

        const dim_t ndim = dim3_t(shape.get(1)).ndim();
        if (ndim == 3) {
            prefilter3D_<T>(input.get(), input_strides_,
                            output.get(), output_strides_, shape_, stream);
        } else if (ndim == 2) {
            prefilter2D_<T>(input.get(), uint3_t{input_strides_[0], input_strides_[2], input_strides_[3]},
                            output.get(), uint3_t{output_strides_[0], output_strides_[2], output_strides_[3]},
                            uint3_t{shape_[0], shape_[2], shape_[3]}, stream);
        } else {
            const bool is_column = shape_[3] == 1;
            prefilter1D_<T>(input.get(), uint2_t{input_strides_[0], input_strides_[3 - is_column]},
                            output.get(), uint2_t{output_strides_[0], output_strides_[3 - is_column]},
                            uint2_t{shape_[0], shape_[3 - is_column]}, stream);
        }
        if (input == output)
            stream.attach(output);
        else
            stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_PREFILTER_(T) \
    template void prefilter<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&)

    NOA_INSTANTIATE_PREFILTER_(float);
    NOA_INSTANTIATE_PREFILTER_(double);
    NOA_INSTANTIATE_PREFILTER_(cfloat_t);
    NOA_INSTANTIATE_PREFILTER_(cdouble_t);
}
