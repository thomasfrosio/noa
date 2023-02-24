#include "noa/core/Math.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/geometry/Prefilter.h"

// This is adapted from https://github.com/DannyRuijters/CubicInterpolationCUDA
// See licences/CubicInterpolationCUDA.txt
// Pitch/step was switched to number of elements. const was added when necessary. Out-of-place filtering was added.

// The implementation requires a single thread to go through the entire 1D array. This is not very efficient
// compared to the CPU implementation. However, when multiple batches are processes, a warp can process
// simultaneously as many batches as it has threads, which is more efficient.
namespace {
    using namespace ::noa;
    constexpr f32 POLE = -0.2679491924311228f; // math::sqrt(3.0f)-2.0f; pole for cubic b-spline

    template<typename T>
    __device__ T initial_causal_coefficient_(const T* c, i32 stride, i32 shape) {
        const i32 horizon = noa::math::min(i32{12}, shape);

        // this initialization corresponds to clamping boundaries accelerated loop
        f32 zn = POLE;
        T sum = *c;
        for (i32 n = 0; n < horizon; n++) {
            sum += zn * *c;
            zn *= POLE;
            c += stride;
        }
        return sum;
    }

    template<typename T>
    __forceinline__ __device__ T initial_anticausal_coefficient_(const T* c) {
        // this initialization corresponds to clamping boundaries
        return ((POLE / (POLE - 1.0f)) * *c);
    }

    template<typename T>
    __device__ void to_coeffs_(T* output, i32 stride, i32 shape) {
        // compute the overall gain
        const f32 lambda = (1.0f - POLE) * (1.0f - 1.0f / POLE);

        // causal initialization and recursion
        T* c = output;
        T previous_c;  //cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = lambda * initial_causal_coefficient_(c, stride, shape);
        for (i32 n = 1; n < shape; n++) {
            c += stride;
            *c = previous_c = lambda * *c + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initial_anticausal_coefficient_(c);
        for (i32 n = shape - 2; 0 <= n; n--) {
            c -= stride;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }

    template<typename T>
    __device__ void to_coeffs_(const T* __restrict__ input, i32 input_stride,
                               T* __restrict__ output, i32 output_stride,
                               i32 shape) {
        // compute the overall gain
        const f32 lambda = (1.0f - POLE) * (1.0f - 1.0f / POLE);

        // causal initialization and recursion
        T* c = output;
        T previous_c;  // cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = lambda * initial_causal_coefficient_(input, input_stride, shape);
        for (i32 n = 1; n < shape; n++) {
            input += input_stride;
            c += output_stride;
            *c = previous_c = lambda * *input + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initial_anticausal_coefficient_(c);
        for (i32 n = shape - 2; 0 <= n; n--) {
            c -= output_stride;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }

    // -- 1D -- //

    template<typename T>
    __global__ void to_coeffs_1d_x_inplace_(T* input, Strides2<u32> strides, Shape2<u32> shape) {
        // process lines in x-direction
        const u32 batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= shape[0])
            return;
        input += batch * strides[0];
        to_coeffs_(input, strides[1], shape[1]);
    }

    template<typename T>
    __global__ void to_coeffs_1d_x_(const T* __restrict__ input, Strides2<u32> input_strides,
                                    T* __restrict__ output, Strides2<u32> output_strides,
                                    Shape2<u32> shape) {
        // process lines in x-direction
        const u32 batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= shape[0])
            return;
        input += batch * input_strides[0];
        output += batch * output_strides[0];
        to_coeffs_(input, input_strides[1], output, output_strides[1], shape[1]);
    }

    // -- 2D -- //

    template<typename T>
    __global__ void to_coeffs_2d_x_inplace_(Accessor<T, 3, u32> input, Shape2<u32> shape) {
        // process lines in x-direction
        const u32 y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= shape[0])
            return;
        const auto input_1d = input[blockIdx.y][y]; // blockIdx.y == batch
        to_coeffs_(input_1d.get(), input_1d.template stride<0>(), shape[1]);
    }

    template<typename T>
    __global__ void to_coeffs_2d_x_(AccessorRestrict<const T, 3, u32> input,
                                    AccessorRestrict<T, 3, u32> output,
                                    Shape2<u32> shape) {
        // process lines in x-direction
        const u32 y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= shape[0])
            return;
        const auto input_1d = input[blockIdx.y][y];
        const auto output_1d = output[blockIdx.y][y];
        to_coeffs_(input_1d.get(), input_1d.template stride<0>(),
                   output.get(), output_1d.template stride<0>(), shape[1]);
    }

    template<typename T>
    __global__ void to_coeffs_2d_y_(T* input, Strides3<u32> strides, Shape2<u32> shape) {
        // process lines in y-direction
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= shape[1])
            return;
        input += blockIdx.y * strides[0] + x * strides[2];
        to_coeffs_(input, strides[1], shape[0]);
    }

    // -- 3D -- //

    template<typename T>
    __global__ void to_coeffs_3d_x_inplace_(Accessor<T, 4, u32> input, Shape3<u32> shape) {
        // process lines in x-direction
        const u32 y = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] || y >= shape[1])
            return;
        const auto input_1d = input[blockIdx.z][z][y];
        to_coeffs_(input_1d.get(), input_1d.template stride<0>(), shape[2]);
    }

    template<typename T>
    __global__ void to_coeffs_3d_x_(AccessorRestrict<const T, 4, u32> input,
                                    AccessorRestrict<T, 4, u32> output,
                                    Shape3<u32> shape) {
        // process lines in x-direction
        const u32 y = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] || y >= shape[1])
            return;
        const auto input_1d = input[blockIdx.z][z][y];
        const auto output_1d = output[blockIdx.z][z][y];
        to_coeffs_(input_1d.get(), input_1d.template stride<0>(),
                   output_1d.get(), output_1d.template stride<0>(), shape[2]);
    }

    template<typename T>
    __global__ void to_coeffs_3d_y_(T* input, Strides4<u32> strides, Shape3<u32> shape) {
        // process lines in y-direction
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] || x >= shape[2])
            return;
        input += indexing::at(blockIdx.z, z, strides) + x * strides[3];
        to_coeffs_(input, strides[2], shape[1]);
    }

    template<typename T>
    __global__ void to_coeffs_3d_z_(T* input, Strides4<u32> strides, Shape3<u32> shape) {
        // process lines in z-direction
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y >= shape[1] || x >= shape[2])
            return;
        input += blockIdx.z * strides[0] + y * strides[2] + x * strides[3];
        to_coeffs_(input, strides[1], shape[0]);
    }

    void get_launch_config_3d(u32 dim0, u32 dim1, dim3* threads, dim3* blocks) {
        threads->x = dim0 <= 32U ? 32U : 64U; // either 32 or 64 threads in the first dimension
        threads->y = std::min(noa::math::next_multiple_of(dim1, 32U), 512U / threads->x); // 2D block up to 512 threads
        blocks->x = noa::math::divide_up(dim0, threads->x);
        blocks->y = noa::math::divide_up(dim1, threads->y);
    }

    template<typename T>
    void prefilter_1d_(const T* input, const Strides2<u32>& input_strides,
                       T* output, const Strides2<u32>& output_strides,
                       const Shape2<u32>& shape, cuda::Stream& stream) {
        // Each threads processes an entire batch.
        // This has the same problem as the toCoeffs2DX_ and toCoeffs3DX_, memory reads/writes are not coalesced.
        const u32 threads = noa::math::next_multiple_of(shape[0], 32U);
        const u32 blocks = noa::math::divide_up(shape[0], threads);
        const auto config = cuda::LaunchConfig{blocks, threads};

        if (input == output) {
            stream.enqueue("cubic_bspline_prefilter", to_coeffs_1d_x_inplace_<T>, config,
                           output, output_strides, shape);
        } else {
            stream.enqueue("cubic_bspline_prefilter", to_coeffs_1d_x_<T>, config,
                           input, input_strides, output, output_strides, shape);
        }
    }

    template<typename T>
    void prefilter_2d_(const T* input, const Strides3<u32>& input_strides,
                       T* output, const Strides3<u32>& output_strides,
                       const Shape3<u32>& shape, cuda::Stream& stream) {
        // Each threads processes an entire line. The line is first x, then y.
        const u32 threads_x = shape[1] <= 32U ? 32U : 64U;
        const u32 threads_y = shape[2] <= 32U ? 32U : 64U;
        const dim3 blocks_x(noa::math::divide_up(shape[1], threads_x), shape[0]);
        const dim3 blocks_y(noa::math::divide_up(shape[2], threads_y), shape[0]);
        const auto config_x = cuda::LaunchConfig{blocks_x, threads_x};
        const auto config_y = cuda::LaunchConfig{blocks_y, threads_y};

        if (input == output) {
            const auto accessor = Accessor<T, 3, u32>(output, output_strides);
            stream.enqueue("cubic_bspline_prefilter", to_coeffs_2d_x_inplace_<T>, config_x,
                           accessor, shape.pop_front());
        } else {
            const auto input_accessor = AccessorRestrict<const T, 3, u32> (input, input_strides);
            const auto output_accessor = AccessorRestrict<T, 3, u32> (output, output_strides);
            stream.enqueue("cubic_bspline_prefilter", to_coeffs_2d_x_<T>, config_x,
                           input_accessor, output_accessor, shape.pop_front());
        }
        stream.enqueue("cubic_bspline_prefilter", to_coeffs_2d_y_<T>, config_y,
                       output, output_strides, shape.pop_front());
    }

    template<typename T>
    void prefilter_3d_(const T* input, const Strides4<u32>& input_strides,
                       T* output, const Strides4<u32>& output_strides,
                       const Shape4<u32>& shape, cuda::Stream& stream) {
        // Try to determine the optimal block dimensions
        dim3 threads;
        dim3 blocks;
        threads.z = 1;
        blocks.z = shape[0];

        get_launch_config_3d(shape[2], shape[1], &threads, &blocks);
        if (input == output) {
            const auto accessor = Accessor<T, 4, u32> (output, output_strides);
            stream.enqueue("cubic_bspline_prefilter", to_coeffs_3d_x_inplace_<T>, {blocks, threads},
                           accessor, shape.filter(1, 2, 3));
        } else {
            const auto input_accessor = AccessorRestrict<const T, 4, u32> (input, input_strides);
            const auto output_accessor = AccessorRestrict<T, 4, u32> (output, output_strides);
            stream.enqueue("cubic_bspline_prefilter", to_coeffs_3d_x_<T>, {blocks, threads},
                           input_accessor, output_accessor, shape.filter(1, 2, 3));
        }

        get_launch_config_3d(shape[3], shape[1], &threads, &blocks);
        stream.enqueue("cubic_bspline_prefilter", to_coeffs_3d_y_<T>, {blocks, threads},
                       output, output_strides, shape.filter(1, 2, 3));

        get_launch_config_3d(shape[3], shape[2], &threads, &blocks);
        stream.enqueue("cubic_bspline_prefilter", to_coeffs_3d_z_<T>, {blocks, threads},
                       output, output_strides, shape.filter(1, 2, 3));
    }
}

namespace noa::cuda::geometry {
    template<typename Value, typename>
    void cubic_bspline_prefilter(const Value* input, Strides4<i64> input_strides,
                                 Value* output, Strides4<i64> output_strides,
                                 Shape4<i64> shape, Stream& stream) {
        NOA_ASSERT(noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        // Reorder to rightmost.
        const auto order = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
        if (noa::any(order != Vec3<i64>{0, 1, 2})) {
            const auto order_4d = (order + 1).push_front(0);
            input_strides = input_strides.reorder(order_4d);
            output_strides = output_strides.reorder(order_4d);
            shape = shape.reorder(order_4d);
        }

        const auto ndim = shape.ndim();
        if (ndim == 3) {
            prefilter_3d_<Value>(
                    input, input_strides.as_safe<u32>(),
                    output, output_strides.as_safe<u32>(),
                    shape.as_safe<u32>(), stream);
        } else if (ndim == 2) {
            prefilter_2d_<Value>(
                    input, input_strides.filter(0, 2, 3).as_safe<u32>(),
                    output, output_strides.filter(0, 2, 3).as_safe<u32>(),
                    shape.filter(0, 2, 3).as_safe<u32>(), stream);
        } else {
            prefilter_1d_<Value>(
                    input, input_strides.filter(0, 3).as_safe<u32>(),
                    output, output_strides.filter(0, 3).as_safe<u32>(),
                    shape.filter(0, 3).as_safe<u32>(), stream);
        }
    }

    #define NOA_INSTANTIATE_PREFILTER_(T) \
    template void cubic_bspline_prefilter<T,void>(const T*, Strides4<i64>, T*, Strides4<i64>, Shape4<i64>, Stream&)

    NOA_INSTANTIATE_PREFILTER_(f32);
    NOA_INSTANTIATE_PREFILTER_(f64);
    NOA_INSTANTIATE_PREFILTER_(c32);
    NOA_INSTANTIATE_PREFILTER_(c64);
}
