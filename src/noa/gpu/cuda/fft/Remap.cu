#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/fft/Exception.h"
#include "noa/gpu/cuda/fft/Remap.h"
#include "noa/gpu/cuda/util/Block.cuh"

namespace {
    using namespace noa;
    constexpr uint MAX_THREADS = 256;

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void hc2h_(const T* __restrict__ input, uint4_t input_stride,
               T* __restrict__ output, uint4_t output_stride, uint3_t shape_fft) {
        const uint batch = blockIdx.z;
        const uint2_t gid(blockIdx.y, blockIdx.x);
        const uint iz = math::FFTShift(gid[0], shape_fft[0]);
        const uint iy = math::FFTShift(gid[1], shape_fft[1]);
        output += indexing::at(batch, gid[0], gid[1], output_stride);
        input += indexing::at(batch, iz, iy, input_stride);

        for (uint x = threadIdx.x; x < shape_fft[2]; x += blockDim.x)
            output[x * output_stride[3]] = input[x * input_stride[3]];
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void h2hc_(const T* __restrict__ input, uint4_t input_stride,
               T* __restrict__ output, uint4_t output_stride, uint3_t shape_fft) {
        const uint batch = blockIdx.z;
        const uint2_t gid(blockIdx.y, blockIdx.x);
        const uint iz = math::iFFTShift(gid[0], shape_fft[0]);
        const uint iy = math::iFFTShift(gid[1], shape_fft[1]);
        output += indexing::at(batch, gid[0], gid[1], output_stride);
        input += indexing::at(batch, iz, iy, input_stride);

        for (uint x = threadIdx.x; x < shape_fft[2]; x += blockDim.x)
            output[x * output_stride[3]] = input[x * input_stride[3]];
    }

    // In-place, Y and Z dimensions have both an even number of elements.
    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void h2hcInPlace_(T* output, uint4_t output_stride, uint3_t shape_fft) {
        const uint batch = blockIdx.z;
        const uint2_t gid(blockIdx.y, blockIdx.x);
        const uint iz = math::iFFTShift(gid[0], shape_fft[0]);
        const uint iy = math::iFFTShift(gid[1], shape_fft[1]);
        T* input = output + indexing::at(batch, iz, iy, output_stride);
        output += indexing::at(batch, gid[0], gid[1], output_stride);

        T* shared = cuda::util::block::dynamicSharedResource<T>();
        int count = 0;
        for (uint x = threadIdx.x; x < shape_fft[2]; x += blockDim.x, ++count) {
            shared[x - count * blockDim.x] = output[x * output_stride[3]];
            output[x * output_stride[3]] = input[x * output_stride[3]];
            input[x * output_stride[3]] = shared[x - count * blockDim.x];
        }
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void f2fc_(const T* __restrict__ input, uint4_t input_stride,
               T* __restrict__ output, uint4_t output_stride, uint3_t shape) {
        const uint batch = blockIdx.z;
        const uint2_t gid(blockIdx.y, blockIdx.x);
        const uint iz = math::iFFTShift(gid[0], shape[0]);
        const uint iy = math::iFFTShift(gid[1], shape[1]);
        input += indexing::at(batch, iz, iy, input_stride);
        output += indexing::at(batch, gid[0], gid[1], output_stride);

        for (uint x = threadIdx.x; x < shape[2]; x += blockDim.x)
            output[x * output_stride[3]] = input[math::iFFTShift(x, shape[2]) * input_stride[3]];
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void fc2f_(const T* __restrict__ input, uint4_t input_stride,
               T* __restrict__ output, uint4_t output_stride, uint3_t shape) {
        const uint batch = blockIdx.z;
        const uint2_t gid(blockIdx.y, blockIdx.x);
        const uint iz = math::FFTShift(gid[0], shape[0]);
        const uint iy = math::FFTShift(gid[1], shape[1]);
        input += indexing::at(batch, iz, iy, input_stride);
        output += indexing::at(batch, gid[0], gid[1], output_stride);

        for (uint x = threadIdx.x; x < shape[2]; x += blockDim.x)
            output[x * output_stride[3]] = input[math::FFTShift(x, shape[2]) * input_stride[3]];
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void f2h_(const T* __restrict__ input, uint4_t input_stride,
              T* __restrict__ output, uint4_t output_stride, uint3_t shape_fft) {
        const uint batch = blockIdx.z;
        const uint2_t gid(blockIdx.y, blockIdx.x);
        input += indexing::at(batch, gid[0], gid[1], input_stride);
        output += indexing::at(batch, gid[0], gid[1], output_stride);

        for (uint x = threadIdx.x; x < shape_fft[2]; x += blockDim.x)
            output[x * output_stride[3]] = input[x * input_stride[3]];
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void h2f_(const T* __restrict__ input, uint4_t input_stride,
              T* __restrict__ output, uint4_t output_stride, uint3_t shape) {
        const uint batch = blockIdx.z;
        const uint2_t gid(blockIdx.y, blockIdx.x);
        const uint half = shape[2] / 2 + 1;
        output += indexing::at(batch, gid[0], gid[1], output_stride);

        // Copy first half:
        const T* in = input + indexing::at(batch, gid[0], gid[1], input_stride);
        for (uint x = threadIdx.x; x < half; x += blockDim.x)
            output[x * output_stride[3]] = in[x * input_stride[3]];

        // Rebase to the symmetric row in the non-redundant array corresponding to the redundant elements.
        // Then copy in reverse order.
        in = input + indexing::at(batch,
                        gid[0] ? shape[0] - gid[0] : gid[0],
                        gid[1] ? shape[1] - gid[1] : gid[1],
                        input_stride);
        for (uint x = half + threadIdx.x; x < shape[2]; x += blockDim.x) {
            if constexpr (noa::traits::is_complex_v<T>)
                output[x * output_stride[3]] = math::conj(in[(shape[2] - x) * input_stride[3]]);
            else
                output[x * output_stride[3]] = in[(shape[2] - x) * input_stride[3]];
        }
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void f2hc_(const T* __restrict__ input, uint4_t input_stride,
               T* __restrict__ output, uint4_t output_stride, uint3_t shape_fft) {
        const uint batch = blockIdx.z;
        const uint2_t gid(blockIdx.y, blockIdx.x);
        const uint iz = math::iFFTShift(gid[0], shape_fft[0]);
        const uint iy = math::iFFTShift(gid[1], shape_fft[1]);
        input += indexing::at(batch, iz, iy, input_stride);
        output += indexing::at(batch, gid[0], gid[1], output_stride);

        for (uint x = threadIdx.x; x < shape_fft[2]; x += blockDim.x)
            output[x * output_stride[3]] = input[x * input_stride[3]];
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void hc2f_(const T* __restrict__ input, uint4_t input_stride,
               T* __restrict__ output, uint4_t output_stride, uint3_t shape) {
        const uint batch = blockIdx.z;
        const uint2_t gid(blockIdx.y, blockIdx.x);
        const uint half = shape[2] / 2 + 1;
        output += indexing::at(batch, gid[0], gid[1], output_stride);

        // Copy first half:
        const T* in = input + indexing::at(batch,
                                 math::FFTShift(gid[0], shape[0]),
                                 math::FFTShift(gid[1], shape[1]),
                                 input_stride);
        for (uint x = threadIdx.x; x < half; x += blockDim.x)
            output[x * output_stride[3]] = in[x * input_stride[3]];

        // Rebase to the symmetric row in the non-redundant array corresponding to the redundant elements.
        // Then copy in reverse order.
        in = input + indexing::at(batch,
                        math::FFTShift(gid[0] ? shape[0] - gid[0] : gid[0], shape[0]),
                        math::FFTShift(gid[1] ? shape[1] - gid[1] : gid[1], shape[1]),
                        input_stride);
        for (uint x = half + threadIdx.x; x < shape[2]; x += blockDim.x) {
            if constexpr (noa::traits::is_complex_v<T>)
                output[x * output_stride[3]] = math::conj(in[(shape[2] - x) * input_stride[3]]);
            else
                output[x * output_stride[3]] = in[(shape[2] - x) * input_stride[3]];
        }
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void fc2h_(const T* __restrict__ input, uint4_t input_stride,
               T* __restrict__ output, uint4_t output_stride, uint3_t shape) {
        const uint batch = blockIdx.z;
        const uint2_t gid(blockIdx.y, blockIdx.x);
        const uint iz = math::FFTShift(gid[0], shape[0]);
        const uint iy = math::FFTShift(gid[1], shape[1]);
        input += indexing::at(batch, iz, iy, input_stride);
        output += indexing::at(batch, gid[0], gid[1], output_stride);

        for (uint x = threadIdx.x; x < shape[2] / 2 + 1; x += blockDim.x)
            output[x * output_stride[3]] = input[math::FFTShift(x, shape[2]) * input_stride[3]];
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void fc2hc_(const T* __restrict__ input, uint4_t input_stride,
               T* __restrict__ output, uint4_t output_stride, uint3_t shape) {
        const uint batch = blockIdx.z;
        const uint2_t gid(blockIdx.y, blockIdx.x);
        input += indexing::at(batch, gid[0], gid[1], input_stride);
        output += indexing::at(batch, gid[0], gid[1], output_stride);

        for (uint x = threadIdx.x; x < shape[2] / 2 + 1; x += blockDim.x)
            output[x * output_stride[3]] = input[math::FFTShift(x, shape[2]) * input_stride[3]];
    }
}

namespace noa::cuda::fft::details {
    template<typename T>
    void hc2h(const shared_t<T[]>& input, size4_t input_stride,
              const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream) {
        NOA_ASSERT(input != output);
        const uint3_t shape_fft{shape.fft().get(1)};
        const uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_fft[2], Limits::WARP_SIZE));
        const dim3 blocks(shape_fft[1], shape_fft[0], shape[0]);
        stream.enqueue("hc2h_", hc2h_<T>, {blocks, threads},
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride}, shape_fft);
        stream.attach(input, output);
    }

    template<typename T>
    void h2hc(const shared_t<T[]>& input, size4_t input_stride,
              const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream) {
        const uint3_t shape_fft{shape.fft().get(1)};
        const uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_fft[2], Limits::WARP_SIZE));

        if (input == output) {
            if ((shape[2] != 1 && shape[2] % 2) || (shape[1] != 1 && shape[1] % 2))
                NOA_THROW("In-place remapping is only available when dim 1 and 2 have an even number of elements");
            const dim3 blocks(noa::math::max(shape_fft[1] / 2, 1U), shape_fft[0], shape[0]);
            stream.enqueue("h2hcInPlace_", h2hcInPlace_<T>, {blocks, threads, threads * sizeof(T)},
                           output.get(), uint4_t{output_stride}, shape_fft);
            stream.attach(output);
        } else {
            const dim3 blocks(shape_fft[1], shape_fft[0], shape[0]);
            stream.enqueue("h2hc_", h2hc_<T>, {blocks, threads},
                           input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride}, shape_fft);
            stream.attach(input, output);
        }
    }

    template<typename T>
    void f2fc(const shared_t<T[]>& input, size4_t input_stride,
              const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream) {
        NOA_ASSERT(input != output);
        const uint3_t shape_full{shape.get(1)};
        const uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full[2], Limits::WARP_SIZE));
        const dim3 blocks(shape_full[1], shape_full[0], shape[0]);
        stream.enqueue("f2fc_", f2fc_<T>, {blocks, threads},
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride}, shape_full);
        stream.attach(input, output);
    }

    template<typename T>
    void fc2f(const shared_t<T[]>& input, size4_t input_stride,
              const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream) {
        NOA_ASSERT(input != output);
        const uint3_t shape_full{shape.get(1)};
        const uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full[2], Limits::WARP_SIZE));
        const dim3 blocks(shape_full[1], shape_full[0], shape[0]);
        stream.enqueue("fc2f_", fc2f_<T>, {blocks, threads},
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride}, shape_full);
        stream.attach(input, output);
    }

    template<typename T>
    void f2h(const shared_t<T[]>& input, size4_t input_stride,
             const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream) {
        NOA_ASSERT(input != output);
        const uint3_t shape_fft{shape.fft().get(1)};
        const uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_fft[2], Limits::WARP_SIZE));
        const dim3 blocks(shape_fft[1], shape_fft[0], shape[0]);
        stream.enqueue("f2h_", f2h_<T>, {blocks, threads},
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride}, shape_fft);
        stream.attach(input, output);
    }

    template<typename T>
    void h2f(const shared_t<T[]>& input, size4_t input_stride,
             const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream) {
        NOA_ASSERT(input != output);
        const uint3_t shape_full{shape.get(1)};
        const uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full[2] / 2 + 1, Limits::WARP_SIZE));
        const dim3 blocks(shape_full[1], shape_full[0], shape[0]);
        stream.enqueue("h2f_", h2f_<T>, {blocks, threads},
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride}, shape_full);
        stream.attach(input, output);
    }

    template<typename T>
    void f2hc(const shared_t<T[]>& input, size4_t input_stride,
              const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream) {
        NOA_ASSERT(input != output);
        const uint3_t shape_fft{shape.fft().get(1)};
        const uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_fft[2], Limits::WARP_SIZE));
        const dim3 blocks(shape_fft[1], shape_fft[0], shape[0]);
        stream.enqueue("f2hc_", f2hc_<T>, {blocks, threads},
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride}, shape_fft);
        stream.attach(input, output);
    }

    template<typename T>
    void hc2f(const shared_t<T[]>& input, size4_t input_stride,
              const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream) {
        NOA_ASSERT(input != output);
        const uint3_t shape_full{shape.get(1)};
        const uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full[2] / 2 + 1, Limits::WARP_SIZE));
        const dim3 blocks(shape_full[1], shape_full[0], shape[0]);
        stream.enqueue("hc2f_", hc2f_<T>, {blocks, threads},
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride}, shape_full);
        stream.attach(input, output);
    }

    template<typename T>
    void fc2h(const shared_t<T[]>& input, size4_t input_stride,
              const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream) {
        NOA_ASSERT(input != output);
        const uint3_t shape_full{shape.get(1)};
        const uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full[2] / 2 + 1, Limits::WARP_SIZE));
        const dim3 blocks(shape_full[1], shape_full[0], shape[0]);
        stream.enqueue("fc2h_", fc2h_<T>, {blocks, threads},
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride}, shape_full);
        stream.attach(input, output);
    }

    template<typename T>
    void fc2hc(const shared_t<T[]>& input, size4_t input_stride,
               const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream) {
        NOA_ASSERT(input != output);
        const uint3_t shape_full{shape.get(1)};
        const uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full[2] / 2 + 1, Limits::WARP_SIZE));
        const dim3 blocks(shape_full[1], shape_full[0], shape[0]);
        stream.enqueue("fc2hc_", fc2hc_<T>, {blocks, threads},
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride}, shape_full);
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_REMAPS_(T)                                                                      \
    template void hc2h<T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);  \
    template void h2hc<T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);  \
    template void f2fc<T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);  \
    template void fc2f<T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);  \
    template void f2h<T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);   \
    template void h2f<T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);   \
    template void f2hc<T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);  \
    template void hc2f<T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);  \
    template void fc2h<T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);  \
    template void fc2hc<T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_REMAPS_(half_t);
    NOA_INSTANTIATE_REMAPS_(float);
    NOA_INSTANTIATE_REMAPS_(double);
    NOA_INSTANTIATE_REMAPS_(chalf_t);
    NOA_INSTANTIATE_REMAPS_(cfloat_t);
    NOA_INSTANTIATE_REMAPS_(cdouble_t);
}
