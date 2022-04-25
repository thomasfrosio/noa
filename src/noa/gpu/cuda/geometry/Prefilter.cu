#include "noa/common/Math.h"
#include "noa/common/Profiler.h"

#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/geometry/Interpolate.h"

// This is adapted from https://github.com/DannyRuijters/CubicInterpolationCUDA
// See licences/CubicInterpolationCUDA.txt
// Pitch/step was switched to number of elements. const was added when necessary. Out-of-place filtering was added.

/// \note The implementation requires a single thread to go through the entire 1D array. This is not very efficient
///       compared to the CPU implementation. However, when multiple batches are processes, a warp can process
///       simultaneously as many batches as it has threads, which is more efficient.
namespace {
    using namespace ::noa;
    constexpr float POLE = -0.2679491924311228f; // math::sqrt(3.0f)-2.0f; pole for cubic b-spline

    template<typename T> // float or float2
    __device__ T initialCausalCoefficient_(const T* c, uint stride, uint shape) {
        const uint horizon = math::min(12U, shape);

        // this initialization corresponds to clamping boundaries accelerated loop
        float zn = POLE;
        T sum = *c;
        for (uint n = 0; n < horizon; n++) {
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
    __device__ void toCoeffs_(T* output, uint stride, uint shape) {
        // compute the overall gain
        const float lambda = (1.0f - POLE) * (1.0f - 1.0f / POLE);

        // causal initialization and recursion
        T* c = output;
        T previous_c;  //cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = lambda * initialCausalCoefficient_(c, stride, shape);
        for (uint n = 1; n < shape; n++) {
            c += stride;
            *c = previous_c = lambda * *c + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initialAntiCausalCoefficient_(c);
        for (int n = static_cast<int>(shape) - 2; 0 <= n; n--) {
            c -= stride;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }

    template<typename T>
    __device__ void toCoeffs_(const T* __restrict__ input, uint input_stride,
                              T* __restrict__ output, uint output_stride,
                              uint shape) {
        // compute the overall gain
        const float lambda = (1.0f - POLE) * (1.0f - 1.0f / POLE);

        // causal initialization and recursion
        T* c = output;
        T previous_c;  // cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = lambda * initialCausalCoefficient_(input, input_stride, shape);
        for (uint n = 1; n < shape; n++) {
            input += input_stride;
            c += output_stride;
            *c = previous_c = lambda * *input + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initialAntiCausalCoefficient_(c);
        for (int n = static_cast<int>(shape) - 2; 0 <= n; n--) {
            c -= output_stride;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }

    // -- 1D -- //

    template<typename T>
    __global__ void toCoeffs1DX_inplace_(T* input, uint2_t stride, uint2_t shape) {
        // process lines in x-direction
        const uint batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= shape[0])
            return;
        input += batch * stride[0];
        toCoeffs_(input, stride[1], shape[1]);
    }

    template<typename T>
    __global__ void toCoeffs1DX_(const T* __restrict__ input, uint2_t input_stride,
                                 T* __restrict__ output, uint2_t output_stride,
                                 uint2_t shape) {
        // process lines in x-direction
        const uint batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= shape[0])
            return;
        input += batch * input_stride[0];
        output += batch * output_stride[0];
        toCoeffs_(input, input_stride[1], output, output_stride[1], shape[1]);
    }

    // -- 2D -- //

    template<typename T>
    __global__ void toCoeffs2DX_inplace_(T* input, uint3_t stride, uint2_t shape) {
        // process lines in x-direction
        const uint y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= shape[0])
            return;
        input += blockIdx.y * stride[0] + y * stride[1]; // blockIdx.y == batch
        toCoeffs_(input, stride[2], shape[1]);
    }

    template<typename T>
    __global__ void toCoeffs2DX_(const T* __restrict__ input, uint3_t input_stride,
                                 T* __restrict__ output, uint3_t output_stride,
                                 uint2_t shape) {
        // process lines in x-direction
        const uint y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= shape[0])
            return;
        input += blockIdx.y * input_stride[0] + y * input_stride[1];
        output += blockIdx.y * output_stride[0] + y * output_stride[1];
        toCoeffs_(input, input_stride[2], output, output_stride[2], shape[1]);
    }

    template<typename T>
    __global__ void toCoeffs2DY_(T* input, uint3_t stride, uint2_t shape) {
        // process lines in y-direction
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= shape[1])
            return;
        input += blockIdx.y * stride[0] + x * stride[2];
        toCoeffs_(input, stride[1], shape[0]);
    }

    // -- 3D -- //

    template<typename T>
    __global__ void toCoeffs3DX_inplace_(T* input, uint4_t stride, uint3_t shape) {
        // process lines in x-direction
        const uint y = blockIdx.x * blockDim.x + threadIdx.x;
        const uint z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] || y >= shape[1])
            return;
        input += indexing::at(blockIdx.z, z, y, stride);
        toCoeffs_(input, stride[3], shape[2]);
    }

    template<typename T>
    __global__ void toCoeffs3DX_(const T* __restrict__ input, uint4_t input_stride,
                                 T* __restrict__ output, uint4_t output_stride,
                                 uint3_t shape) {
        // process lines in x-direction
        const uint y = blockIdx.x * blockDim.x + threadIdx.x;
        const uint z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] || y >= shape[1])
            return;
        input += indexing::at(blockIdx.z, z, y, input_stride);
        output += indexing::at(blockIdx.z, z, y, output_stride);
        toCoeffs_(input, input_stride[3], output, output_stride[3], shape[2]);
    }

    template<typename T>
    __global__ void toCoeffs3DY_(T* input, uint4_t stride, uint3_t shape) {
        // process lines in y-direction
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] || x >= shape[2])
            return;
        input += indexing::at(blockIdx.z, z, stride) + x * stride[3];
        toCoeffs_(input, stride[2], shape[1]);
    }

    template<typename T>
    __global__ void toCoeffs3DZ_(T* input, uint4_t stride, uint3_t shape) {
        // process lines in z-direction
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y >= shape[1] || x >= shape[2])
            return;
        input += blockIdx.z * stride[0] + y * stride[2] + x * stride[3];
        toCoeffs_(input, stride[1], shape[0]);
    }

    void getLaunchConfig3D(uint dim0, uint dim1, dim3* threads, dim3* blocks) {
        threads->x = dim0 <= 32U ? 32U : 64U; // either 32 or 64 threads in the first dimension
        threads->y = math::min(math::nextMultipleOf(dim1, 32U), 512U / threads->x); // 2D block up to 512 threads
        blocks->x = math::divideUp(dim0, threads->x);
        blocks->y = math::divideUp(dim1, threads->y);
    }

    template<typename T>
    void prefilter1D_(const T* input, uint2_t input_stride, T* output, uint2_t output_stride,
                      uint2_t shape, cuda::Stream& stream) {
        NOA_PROFILE_FUNCTION();
        // Each threads processes an entire batch.
        // This has the same problem as the toCoeffs2DX_ and toCoeffs3DX_, memory reads/writes are not coalesced.
        const uint threads = math::nextMultipleOf(shape[0], 32U);
        const uint blocks = math::divideUp(shape[0], threads);
        const cuda::LaunchConfig config{blocks, threads};

        if (input == output) {
            stream.enqueue("geometry::bspline::prefilter1D", toCoeffs1DX_inplace_<T>, config,
                           output, output_stride, shape);
        } else {
            stream.enqueue("geometry::bspline::prefilter1D", toCoeffs1DX_<T>, config,
                           input, input_stride, output, output_stride, shape);
        }
    }

    template<typename T>
    void prefilter2D_(const T* input, uint3_t input_stride, T* output, uint3_t output_stride,
                      uint3_t shape, cuda::Stream& stream) {
        NOA_PROFILE_FUNCTION();
        // Each threads processes an entire line. The line is first x, then y.
        const uint threads_x = shape[1] <= 32U ? 32U : 64U;
        const uint threads_y = shape[2] <= 32U ? 32U : 64U;
        const dim3 blocks_x(math::divideUp(shape[1], threads_x), shape[0]);
        const dim3 blocks_y(math::divideUp(shape[2], threads_y), shape[0]);
        const cuda::LaunchConfig config_x{blocks_x, threads_x};
        const cuda::LaunchConfig config_y{blocks_y, threads_y};

        if (input == output) {
            stream.enqueue("geometry::bspline::prefilter2D_x", toCoeffs2DX_inplace_<T>, config_x,
                           output, output_stride, uint2_t{shape[1], shape[2]});
        } else {
            stream.enqueue("geometry::bspline::prefilter2D_x", toCoeffs2DX_<T>, config_x,
                           input, input_stride, output, output_stride, uint2_t{shape[1], shape[2]});
        }
        stream.enqueue("geometry::bspline::prefilter2D_y", toCoeffs2DY_<T>, config_y,
                       output, output_stride, uint2_t{shape[1], shape[2]});
    }

    template<typename T>
    void prefilter3D_(const T* input, uint4_t input_stride, T* output, uint4_t output_stride,
                      uint4_t shape, cuda::Stream& stream) {
        NOA_PROFILE_FUNCTION();
        // Try to determine the optimal block dimensions
        dim3 threads;
        dim3 blocks;
        threads.z = 1;
        blocks.z = shape[0];

        getLaunchConfig3D(shape[2], shape[1], &threads, &blocks);
        if (input == output) {
            stream.enqueue("geometry::bspline::prefilter3D_x", toCoeffs3DX_inplace_<T>, {blocks, threads},
                           output, output_stride, uint3_t{shape[1], shape[2], shape[3]});
        } else {
            stream.enqueue("geometry::bspline::prefilter3D_x", toCoeffs3DX_<T>, {blocks, threads},
                           input, input_stride, output, output_stride, uint3_t{shape[1], shape[2], shape[3]});
        }

        getLaunchConfig3D(shape[3], shape[1], &threads, &blocks);
        stream.enqueue("geometry::bspline::prefilter3D_y", toCoeffs3DY_<T>, {blocks, threads},
                       output, output_stride, uint3_t{shape[1], shape[2], shape[3]});

        getLaunchConfig3D(shape[3], shape[2], &threads, &blocks);
        stream.enqueue("geometry::bspline::prefilter3D_z", toCoeffs3DZ_<T>, {blocks, threads},
                       output, output_stride, uint3_t{shape[1], shape[2], shape[3]});
    }
}

namespace noa::cuda::geometry::bspline {
    template<typename T>
    void prefilter(const shared_t<T[]>& input, size4_t input_stride,
                   const shared_t<T[]>& output, size4_t output_stride,
                   size4_t shape, Stream& stream) {
        const size_t ndim = size3_t{shape.get() + 1}.ndim();
        if (ndim == 3) {
            prefilter3D_<T>(input.get(), uint4_t{input_stride},
                            output.get(), uint4_t{output_stride}, uint4_t{shape}, stream);
        } else if (ndim == 2) {
            prefilter2D_<T>(input.get(), uint3_t{input_stride[0], input_stride[2], input_stride[3]},
                            output.get(), uint3_t{output_stride[0], output_stride[2], output_stride[3]},
                            uint3_t{shape[0], shape[2], shape[3]}, stream);
        } else {
            prefilter1D_<T>(input.get(), uint2_t{input_stride[0], input_stride[3]},
                            output.get(), uint2_t{output_stride[0], output_stride[3]},
                            uint2_t{shape[0], shape[3]}, stream);
        }
        if (input == output)
            stream.attach(output);
        else
            stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_PREFILTER_(T) \
    template void prefilter<T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_PREFILTER_(float);
    NOA_INSTANTIATE_PREFILTER_(double);
    NOA_INSTANTIATE_PREFILTER_(cfloat_t);
    NOA_INSTANTIATE_PREFILTER_(cdouble_t);
}
