#include "noa/common/Math.h"
#include "noa/common/Profiler.h"

#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/transform/Interpolate.h"

// This is adapted from https://github.com/DannyRuijters/CubicInterpolationCUDA
// See licences/CubicInterpolationCUDA.txt
// Pitch/step was switched to number of elements. const was added when necessary. Out-of-place filtering was added.
namespace {
    using namespace ::noa;
    constexpr float POLE = -0.2679491924311228f; // math::sqrt(3.0f)-2.0f; pole for cubic b-spline

    template<typename T> // float or float2
    __device__ T initialCausalCoefficient_(const T* c, uint step_increment, uint steps) {
        const uint horizon = math::min(12U, steps);

        // this initialization corresponds to clamping boundaries accelerated loop
        float zn = POLE;
        T sum = *c;
        for (uint n = 0; n < horizon; n++) {
            sum += zn * *c;
            zn *= POLE;
            c += step_increment;
        }
        return sum;
    }

    template<typename T>
    __forceinline__ __device__ T initialAntiCausalCoefficient_(const T* c) {
        // this initialization corresponds to clamping boundaries
        return ((POLE / (POLE - 1.0f)) * *c);
    }

    template<typename T>
    __device__ void toCoeffs_(T* output, uint step_increment, uint steps) {
        // compute the overall gain
        const float lambda = (1.0f - POLE) * (1.0f - 1.0f / POLE);

        // causal initialization and recursion
        T* c = output;
        T previous_c;  //cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = lambda * initialCausalCoefficient_(c, step_increment, steps);
        for (uint n = 1; n < steps; n++) {
            c += step_increment;
            *c = previous_c = lambda * *c + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initialAntiCausalCoefficient_(c);
        for (int n = static_cast<int>(steps) - 2; 0 <= n; n--) {
            c -= step_increment;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }

    template<typename T>
    __device__ void toCoeffs_(const T* __restrict__ input, uint input_step_increment,
                              T* __restrict__ output, uint output_step_increment,
                              uint steps) {
        // compute the overall gain
        const float lambda = (1.0f - POLE) * (1.0f - 1.0f / POLE);

        // causal initialization and recursion
        T* c = output;
        T previous_c;  // cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = lambda * initialCausalCoefficient_(input, input_step_increment, steps);
        for (uint n = 1; n < steps; n++) {
            input += input_step_increment;
            c += output_step_increment;
            *c = previous_c = lambda * *input + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initialAntiCausalCoefficient_(c);
        for (int n = static_cast<int>(steps) - 2; 0 <= n; n--) {
            c -= output_step_increment;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }

    // -- 1D -- //

    template<typename T>
    __global__ void toCoeffs1DX_(T* input, uint input_pitch, uint size, uint batches) {
        // process lines in x-direction
        const uint batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= batches)
            return;
        input += batch * input_pitch;
        toCoeffs_(input, 1, size);
    }

    template<typename T>
    __global__ void toCoeffs1DX_(const T* __restrict__ input, uint input_pitch,
                                 T* __restrict__ output, uint output_pitch,
                                 uint size, uint batches) {
        // process lines in x-direction
        const uint batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= batches)
            return;
        input += batch * input_pitch;
        output += batch * output_pitch;
        toCoeffs_(input, 1, output, 1, size);
    }

    // -- 2D -- //

    template<typename T>
    __global__ void toCoeffs2DX_(T* input, uint input_pitch, uint2_t shape) {
        // process lines in x-direction
        const uint y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= shape.y)
            return;
        input += (blockIdx.y * shape.y + y) * input_pitch; // blockIdx.y == batch
        toCoeffs_(input, 1, shape.x);
    }

    template<typename T>
    __global__ void toCoeffs2DX_(const T* __restrict__ input, uint input_pitch,
                                 T* __restrict__ output, uint output_pitch,
                                 uint2_t shape) {
        // process lines in x-direction
        const uint y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= shape.y)
            return;
        input += (blockIdx.y * shape.y + y) * input_pitch;
        output += (blockIdx.y * shape.y + y) * output_pitch;
        toCoeffs_(input, 1, output, 1, shape.x);
    }

    template<typename T>
    __global__ void toCoeffs2DY_(T* input, uint input_pitch, uint2_t shape) {
        // process lines in y-direction
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= shape.x)
            return;
        input += blockIdx.y * shape.y * input_pitch + x;
        toCoeffs_(input, input_pitch, shape.y);
    }

    // -- 3D -- //

    template<typename T>
    __global__ void toCoeffs3DX_(T* input, uint input_pitch, uint3_t shape) {
        // process lines in x-direction
        const uint y = blockIdx.x * blockDim.x + threadIdx.x;
        const uint z = blockIdx.y * blockDim.y + threadIdx.y;
        if (y >= shape.y || z >= shape.z)
            return;
        input += blockIdx.z * getRows(shape) * input_pitch;
        input += (z * shape.y + y) * input_pitch;
        toCoeffs_(input, 1, shape.x);
    }

    template<typename T>
    __global__ void toCoeffs3DX_(const T* __restrict__ input, uint input_pitch,
                                 T* __restrict__ output, uint output_pitch,
                                 uint3_t shape) {
        // process lines in x-direction
        const uint y = blockIdx.x * blockDim.x + threadIdx.x;
        const uint z = blockIdx.y * blockDim.y + threadIdx.y;
        if (y >= shape.y || z >= shape.z)
            return;
        const uint batch_rows = blockIdx.z * getRows(shape);
        const uint offset = (z * shape.y + y);
        input += batch_rows * input_pitch;
        output += batch_rows * output_pitch;
        input += offset * input_pitch;
        output += offset * output_pitch;
        toCoeffs_(input, 1, output, 1, shape.x);
    }

    template<typename T>
    __global__ void toCoeffs3DY_(T* input, uint input_pitch, uint3_t shape) {
        // process lines in y-direction
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint z = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= shape.x || z >= shape.z)
            return;
        input += blockIdx.z * getRows(shape) * input_pitch;
        input += z * shape.y * input_pitch + x;
        toCoeffs_(input, input_pitch, shape.y);
    }

    template<typename T>
    __global__ void toCoeffs3DZ_(T* input, uint input_pitch, uint3_t shape) {
        // process lines in z-direction
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= shape.x || y >= shape.y)
            return;
        input += blockIdx.z * getRows(shape) * input_pitch;
        input += y * input_pitch + x;
        toCoeffs_(input, input_pitch * shape.y, shape.z);
    }

    void getLaunchConfig3D(uint dim0, uint dim1, dim3* threads, dim3* blocks) {
        threads->x = dim0 <= 32U ? 32U : 64U; // either 32 or 64 threads in the first dimension
        threads->y = math::min(math::nextMultipleOf(dim1, 32U), 512U / threads->x); // 2D block up to 512 threads
        blocks->x = math::divideUp(dim0, threads->x);
        blocks->y = math::divideUp(dim1, threads->y);
    }
}

namespace noa::cuda::transform::bspline {
    template<typename T>
    void prefilter1D(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                     size_t size, uint batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const uint tmp(size);
        // Each threads processes an entire batch.
        // This has the same problem than the toCoeffs2DX_ and toCoeffs3DX_, memory reads/writes are not coalesced.
        dim3 threadsX(math::nextMultipleOf(batches, 32U));
        dim3 blocksX(math::divideUp(batches, threadsX.x));

        if (inputs == outputs)
            toCoeffs1DX_<<<blocksX, threadsX, 0, stream.id()>>>(outputs, outputs_pitch, tmp, batches);
        else
            toCoeffs1DX_<<<blocksX, threadsX, 0, stream.id()>>>(inputs, inputs_pitch, outputs, outputs_pitch,
                                                                tmp, batches);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void prefilter2D(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                     size2_t shape, uint batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const uint2_t tmp(shape);
        // Each threads processes an entire line. The line is first x, then y.
        dim3 threadsX(tmp.y <= 32U ? 32U : 64U);
        dim3 threadsY(tmp.x <= 32U ? 32U : 64U);
        dim3 blocksX(math::divideUp(tmp.y, threadsX.x), batches);
        dim3 blocksY(math::divideUp(tmp.x, threadsY.x), batches);

        if (inputs == outputs)
            toCoeffs2DX_<<<blocksX, threadsX, 0, stream.id()>>>(outputs, outputs_pitch, tmp);
        else
            toCoeffs2DX_<<<blocksX, threadsX, 0, stream.id()>>>(inputs, inputs_pitch, outputs, outputs_pitch, tmp);
        NOA_THROW_IF(cudaPeekAtLastError());
        toCoeffs2DY_<<<blocksY, threadsY, 0, stream.id()>>>(outputs, outputs_pitch, tmp);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void prefilter3D(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                     size3_t shape, uint batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        // Try to determine the optimal block dimensions
        const uint3_t tmp_shape(shape);
        dim3 threads;
        dim3 blocks;
        threads.z = 1;
        blocks.z = batches;

        getLaunchConfig3D(tmp_shape.y, tmp_shape.z, &threads, &blocks);
        if (inputs == outputs)
            toCoeffs3DX_<<<blocks, threads, 0, stream.id()>>>(outputs, outputs_pitch, tmp_shape);
        else
            toCoeffs3DX_<<<blocks, threads, 0, stream.id()>>>(inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape);
        NOA_THROW_IF(cudaPeekAtLastError());

        getLaunchConfig3D(tmp_shape.x, tmp_shape.z, &threads, &blocks);
        toCoeffs3DY_<<<blocks, threads, 0, stream.id()>>>(outputs, outputs_pitch, tmp_shape);
        NOA_THROW_IF(cudaPeekAtLastError());

        getLaunchConfig3D(tmp_shape.x, tmp_shape.y, &threads, &blocks);
        toCoeffs3DZ_<<<blocks, threads, 0, stream.id()>>>(outputs, outputs_pitch, tmp_shape);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define INSTANTIATE_PREFILTER(T)                                                    \
    template void prefilter2D<T>(const T*, size_t, T*, size_t, size2_t, uint, Stream&); \
    template void prefilter3D<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&)

    INSTANTIATE_PREFILTER(float);
    INSTANTIATE_PREFILTER(cfloat_t);
}
