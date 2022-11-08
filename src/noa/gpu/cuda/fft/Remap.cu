#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/fft/Exception.h"
#include "noa/gpu/cuda/fft/Remap.h"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/Pointers.h"

namespace {
    using namespace noa;
    constexpr uint32_t MAX_THREADS = 256;

    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void hc2h_(AccessorRestrict<const T, 4, uint32_t> input,
               AccessorRestrict<T, 4, uint32_t> output,
               uint3_t shape_fft) {

        const uint32_t batch = blockIdx.z;
        const uint2_t gid{blockIdx.y, blockIdx.x};
        const uint32_t iz = math::FFTShift(gid[0], shape_fft[0]);
        const uint32_t iy = math::FFTShift(gid[1], shape_fft[1]);
        const auto input_row = input[batch][iz][iy];
        const auto output_row = output[batch][gid[0]][gid[1]];

        for (uint32_t x = threadIdx.x; x < shape_fft[2]; x += blockDim.x)
            output_row[x] = input_row[x];
    }

    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void h2hc_(AccessorRestrict<const T, 4, uint32_t> input,
               AccessorRestrict<T, 4, uint32_t> output, uint3_t shape_fft) {
        const uint32_t batch = blockIdx.z;
        const uint2_t gid{blockIdx.y, blockIdx.x};
        const uint32_t iz = math::iFFTShift(gid[0], shape_fft[0]);
        const uint32_t iy = math::iFFTShift(gid[1], shape_fft[1]);
        const auto input_row = input[batch][iz][iy];
        const auto output_row = output[batch][gid[0]][gid[1]];

        for (uint32_t x = threadIdx.x; x < shape_fft[2]; x += blockDim.x)
            output_row[x] = input_row[x];
    }

    // In-place, Y and Z dimensions have both an even number of elements.
    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void h2hcInPlace_(Accessor<T, 4, uint32_t> output, uint3_t shape_fft) {
        const uint32_t batch = blockIdx.z;
        const uint2_t gid{blockIdx.y, blockIdx.x};
        const uint32_t iz = math::iFFTShift(gid[0], shape_fft[0]);
        const uint32_t iy = math::iFFTShift(gid[1], shape_fft[1]);
        const auto input_row = output[batch][iz][iy];
        const auto output_row = output[batch][gid[0]][gid[1]];

        T* shared = cuda::utils::block::dynamicSharedResource<T>();
        int32_t count = 0;
        for (uint32_t x = threadIdx.x; x < shape_fft[2]; x += blockDim.x, ++count) {
            shared[x - count * blockDim.x] = output_row[x];
            output_row[x] = input_row[x];
            input_row[x] = shared[x - count * blockDim.x];
        }
    }

    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void f2fc_(AccessorRestrict<const T, 4, uint32_t> input,
               AccessorRestrict<T, 4, uint32_t> output, uint3_t shape) {
        const uint32_t batch = blockIdx.z;
        const uint2_t gid{blockIdx.y, blockIdx.x};
        const uint32_t iz = math::iFFTShift(gid[0], shape[0]);
        const uint32_t iy = math::iFFTShift(gid[1], shape[1]);
        const auto input_row = input[batch][iz][iy];
        const auto output_row = output[batch][gid[0]][gid[1]];

        for (uint32_t x = threadIdx.x; x < shape[2]; x += blockDim.x)
            output_row[x] = input_row[math::iFFTShift(x, shape[2])];
    }

    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void fc2f_(AccessorRestrict<const T, 4, uint32_t> input,
               AccessorRestrict<T, 4, uint32_t> output, uint3_t shape) {
        const uint32_t batch = blockIdx.z;
        const uint2_t gid{blockIdx.y, blockIdx.x};
        const uint32_t iz = math::FFTShift(gid[0], shape[0]);
        const uint32_t iy = math::FFTShift(gid[1], shape[1]);
        const auto input_row = input[batch][iz][iy];
        const auto output_row = output[batch][gid[0]][gid[1]];

        for (uint32_t x = threadIdx.x; x < shape[2]; x += blockDim.x)
            output_row[x] = input_row[math::FFTShift(x, shape[2])];
    }

    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void f2h_(AccessorRestrict<const T, 4, uint32_t> input,
              AccessorRestrict<T, 4, uint32_t> output, uint3_t shape_fft) {
        const uint32_t batch = blockIdx.z;
        const uint2_t gid{blockIdx.y, blockIdx.x};
        const auto input_row = input[batch][gid[0]][gid[1]];
        const auto output_row = output[batch][gid[0]][gid[1]];

        for (uint32_t x = threadIdx.x; x < shape_fft[2]; x += blockDim.x)
            output_row[x] = input_row[x];
    }

    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void h2f_(AccessorRestrict<const T, 4, uint32_t> input,
              AccessorRestrict<T, 4, uint32_t> output, uint3_t shape) {
        const uint32_t batch = blockIdx.z;
        const uint2_t gid{blockIdx.y, blockIdx.x};
        const uint32_t half = shape[2] / 2 + 1;
        const auto output_row = output[batch][gid[0]][gid[1]];

        // Copy first half:
        auto input_row = input[batch][gid[0]][gid[1]];
        for (uint32_t x = threadIdx.x; x < half; x += blockDim.x)
            output_row[x] = input_row[x];

        // Rebase to the symmetric row in the non-redundant array corresponding to the redundant elements.
        // Then copy in reverse order.
        const uint32_t idx_z = gid[0] ? shape[0] - gid[0] : gid[0];
        const uint32_t idx_y = gid[1] ? shape[1] - gid[1] : gid[1];
        input_row = input[batch][idx_z][idx_y];
        for (uint32_t x = half + threadIdx.x; x < shape[2]; x += blockDim.x) {
            if constexpr (traits::is_complex_v<T>)
                output_row[x] = math::conj(input_row[(shape[2] - x)]);
            else
                output_row[x] = input_row[(shape[2] - x)];
        }
    }

    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void f2hc_(AccessorRestrict<const T, 4, uint32_t> input,
               AccessorRestrict<T, 4, uint32_t> output, uint3_t shape_fft) {
        const uint32_t batch = blockIdx.z;
        const uint2_t gid{blockIdx.y, blockIdx.x};
        const uint32_t iz = math::iFFTShift(gid[0], shape_fft[0]);
        const uint32_t iy = math::iFFTShift(gid[1], shape_fft[1]);
        const auto input_row = input[batch][iz][iy];
        const auto output_row = output[batch][gid[0]][gid[1]];

        for (uint32_t x = threadIdx.x; x < shape_fft[2]; x += blockDim.x)
            output_row[x] = input_row[x];
    }

    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void hc2f_(AccessorRestrict<const T, 4, uint32_t> input,
               AccessorRestrict<T, 4, uint32_t> output, uint3_t shape) {
        const uint32_t batch = blockIdx.z;
        const uint2_t gid{blockIdx.y, blockIdx.x};
        const uint32_t half = shape[2] / 2 + 1;
        const auto output_row = output[batch][gid[0]][gid[1]];

        // Copy first half:
        uint32_t idx_z = math::FFTShift(gid[0], shape[0]);
        uint32_t idx_y = math::FFTShift(gid[1], shape[1]);
        auto input_row = input[batch][idx_z][idx_y];
        for (uint32_t x = threadIdx.x; x < half; x += blockDim.x)
            output_row[x] = input_row[x];

        // Rebase to the symmetric row in the non-redundant array corresponding to the redundant elements.
        // Then copy in reverse order.
        idx_z = math::FFTShift(gid[0] ? shape[0] - gid[0] : gid[0], shape[0]);
        idx_y = math::FFTShift(gid[1] ? shape[1] - gid[1] : gid[1], shape[1]);
        input_row = input[batch][idx_z][idx_y];
        for (uint32_t x = half + threadIdx.x; x < shape[2]; x += blockDim.x) {
            if constexpr (traits::is_complex_v<T>)
                output_row[x] = math::conj(input_row[(shape[2] - x)]);
            else
                output_row[x] = input_row[(shape[2] - x)];
        }
    }

    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void fc2h_(AccessorRestrict<const T, 4, uint32_t> input,
               AccessorRestrict<T, 4, uint32_t> output, uint3_t shape) {
        const uint32_t batch = blockIdx.z;
        const uint2_t gid{blockIdx.y, blockIdx.x};
        const uint32_t iz = math::FFTShift(gid[0], shape[0]);
        const uint32_t iy = math::FFTShift(gid[1], shape[1]);
        const auto input_row = input[batch][iz][iy];
        const auto output_row = output[batch][gid[0]][gid[1]];

        for (uint32_t x = threadIdx.x; x < shape[2] / 2 + 1; x += blockDim.x)
            output_row[x] = input_row[math::FFTShift(x, shape[2])];
    }

    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void fc2hc_(AccessorRestrict<const T, 4, uint32_t> input,
                AccessorRestrict<T, 4, uint32_t> output, uint3_t shape) {
        const uint32_t batch = blockIdx.z;
        const uint2_t gid{blockIdx.y, blockIdx.x};
        const auto input_row = input[batch][gid[0]][gid[1]];
        const auto output_row = output[batch][gid[0]][gid[1]];

        for (uint32_t x = threadIdx.x; x < shape[2] / 2 + 1; x += blockDim.x)
            output_row[x] = input_row[math::FFTShift(x, shape[2])];
    }
}

namespace noa::cuda::fft::details {
    template<typename T>
    void hc2h(const shared_t<T[]>& input, dim4_t input_strides,
              const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT(input.get() != output.get() && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const auto shape_fft = safe_cast<uint3_t>(dim3_t(shape.fft().get(1)));
        const uint32_t threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_fft[2], Limits::WARP_SIZE));
        const dim3 blocks(shape_fft[1], shape_fft[0], shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("hc2h_", hc2h_<T>, {blocks, threads}, input_, output_, shape_fft);
        stream.attach(input, output);
    }

    template<typename T>
    void h2hc(const shared_t<T[]>& input, dim4_t input_strides,
              const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const auto shape_fft = safe_cast<uint3_t>(dim3_t(shape.fft().get(1)));
        const uint32_t threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_fft[2], Limits::WARP_SIZE));

        if (input == output) {
            NOA_ASSERT((shape[2] == 1 || !(shape[2] % 2)) && (shape[1] == 1 || !(shape[1] % 2)));
            NOA_ASSERT(all(input_strides == output_strides));

            const dim3 blocks(noa::math::max(shape_fft[1] / 2, 1U), shape_fft[0], shape[0]);
            const Accessor<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

            stream.enqueue("h2hcInPlace_", h2hcInPlace_<T>, {blocks, threads, threads * sizeof(T)}, output_, shape_fft);
            stream.attach(output);
        } else {
            const dim3 blocks(shape_fft[1], shape_fft[0], shape[0]);
            const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
            const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

            stream.enqueue("h2hc_", h2hc_<T>, {blocks, threads}, input_, output_, shape_fft);
            stream.attach(input, output);
        }
    }

    template<typename T>
    void f2fc(const shared_t<T[]>& input, dim4_t input_strides,
              const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT(input.get() != output.get() && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        if (indexing::isColMajor(input_strides) && indexing::isColMajor(output_strides)) {
            std::swap(shape[2], shape[3]);
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
        }

        const auto shape_full = safe_cast<uint3_t>(dim3_t(shape.get(1)));
        const uint32_t threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full[2], Limits::WARP_SIZE));
        const dim3 blocks(shape_full[1], shape_full[0], shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("f2fc_", f2fc_<T>, {blocks, threads}, input_, output_, shape_full);
        stream.attach(input, output);
    }

    template<typename T>
    void fc2f(const shared_t<T[]>& input, dim4_t input_strides,
              const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT(input.get() != output.get() && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        if (indexing::isColMajor(input_strides) && indexing::isColMajor(output_strides)) {
            std::swap(shape[2], shape[3]);
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
        }

        const auto shape_full = safe_cast<uint3_t>(dim3_t(shape.get(1)));
        const uint32_t threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full[2], Limits::WARP_SIZE));
        const dim3 blocks(shape_full[1], shape_full[0], shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("fc2f_", fc2f_<T>, {blocks, threads}, input_, output_, shape_full);
        stream.attach(input, output);
    }

    template<typename T>
    void f2h(const shared_t<T[]>& input, dim4_t input_strides,
             const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT(input.get() != output.get() && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const auto shape_fft = safe_cast<uint3_t>(dim3_t(shape.fft().get(1)));
        const uint32_t threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_fft[2], Limits::WARP_SIZE));
        const dim3 blocks(shape_fft[1], shape_fft[0], shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("f2h_", f2h_<T>, {blocks, threads}, input_, output_, shape_fft);
        stream.attach(input, output);
    }

    template<typename T>
    void h2f(const shared_t<T[]>& input, dim4_t input_strides,
             const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT(input.get() != output.get() && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const auto shape_full = safe_cast<uint3_t>(dim3_t(shape.get(1)));
        const uint32_t threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full[2] / 2 + 1, Limits::WARP_SIZE));
        const dim3 blocks(shape_full[1], shape_full[0], shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("h2f_", h2f_<T>, {blocks, threads}, input_, output_, shape_full);
        stream.attach(input, output);
    }

    template<typename T>
    void f2hc(const shared_t<T[]>& input, dim4_t input_strides,
              const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT(input.get() != output.get() && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const auto shape_fft = safe_cast<uint3_t>(dim3_t(shape.fft().get(1)));
        const uint32_t threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_fft[2], Limits::WARP_SIZE));
        const dim3 blocks(shape_fft[1], shape_fft[0], shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("f2hc_", f2hc_<T>, {blocks, threads}, input_, output_, shape_fft);
        stream.attach(input, output);
    }

    template<typename T>
    void hc2f(const shared_t<T[]>& input, dim4_t input_strides,
              const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT(input.get() != output.get() && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const auto shape_full = safe_cast<uint3_t>(dim3_t(shape.get(1)));
        const uint32_t threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full[2] / 2 + 1, Limits::WARP_SIZE));
        const dim3 blocks(shape_full[1], shape_full[0], shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("hc2f_", hc2f_<T>, {blocks, threads}, input_, output_, shape_full);
        stream.attach(input, output);
    }

    template<typename T>
    void fc2h(const shared_t<T[]>& input, dim4_t input_strides,
              const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT(input.get() != output.get() && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const auto shape_full = safe_cast<uint3_t>(dim3_t(shape.get(1)));
        const uint32_t threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full[2] / 2 + 1, Limits::WARP_SIZE));
        const dim3 blocks(shape_full[1], shape_full[0], shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("fc2h_", fc2h_<T>, {blocks, threads}, input_, output_, shape_full);
        stream.attach(input, output);
    }

    template<typename T>
    void fc2hc(const shared_t<T[]>& input, dim4_t input_strides,
               const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT(input.get() != output.get() && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const auto shape_full = safe_cast<uint3_t>(dim3_t(shape.get(1)));
        const uint32_t threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full[2] / 2 + 1, Limits::WARP_SIZE));
        const dim3 blocks(shape_full[1], shape_full[0], shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("fc2hc_", fc2hc_<T>, {blocks, threads}, input_, output_, shape_full);
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_REMAPS_(T)                                                                  \
    template void hc2h<T>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&); \
    template void h2hc<T>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&); \
    template void f2fc<T>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&); \
    template void fc2f<T>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&); \
    template void f2h<T>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&);  \
    template void h2f<T>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&);  \
    template void f2hc<T>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&); \
    template void hc2f<T>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&); \
    template void fc2h<T>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&); \
    template void fc2hc<T>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&)

    NOA_INSTANTIATE_REMAPS_(half_t);
    NOA_INSTANTIATE_REMAPS_(float);
    NOA_INSTANTIATE_REMAPS_(double);
    NOA_INSTANTIATE_REMAPS_(chalf_t);
    NOA_INSTANTIATE_REMAPS_(cfloat_t);
    NOA_INSTANTIATE_REMAPS_(cdouble_t);
}
