#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/signal/Convolve.h"
#include "noa/gpu/cuda/util/Block.cuh"
#include "noa/gpu/cuda/util/Pointers.h"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    constexpr dim3 BLOCK_SIZE(16, 16);
    constexpr int32_t MAX_FILTER_SIZE = 129;
    constexpr int32_t MAX_FILTER_BYTES = MAX_FILTER_SIZE * sizeof(double);
    __constant__ char cfilter[MAX_FILTER_BYTES];

    // This is identical to the convolve1_ kernel.
    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolveSeparableX_(AccessorRestrict<const T, 4, uint32_t> input,
                             AccessorRestrict<T, 4, uint32_t> output,
                             uint2_t shape /* YX */, int32_t filter_size, uint32_t blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int2_t tid{threadIdx.y, threadIdx.x};
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_SIZE.y * index[0] + tid[0],
                         BLOCK_SIZE.x * index[1] + tid[1]};

        const auto input_ = input[gid[0]][gid[1]][gid[2]];

        const int32_t padding = filter_size - 1;
        const int32_t halo = padding / 2;
        const int32_t shared_len = static_cast<int32_t>(BLOCK_SIZE.x) + padding;

        // Filter along x.
        T* shared = util::block::dynamicSharedResource<T>();
        if (gid[2] < shape[0]) {
            shared += tid[0] * shared_len; // focus on current row

            // Load shared memory. Loop to take into account padding.
            for (int32_t lx = tid[1], gx = gid[3]; lx < shared_len; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x) {
                const int32_t i_x = gx - halo;
                shared[lx] = i_x >= 0 && i_x < shape[1] ? input_[i_x] : T{0};
            }
            util::block::synchronize();

            if (gid[3] < shape[1]) {
                // Weighted sum.
                const T* window = reinterpret_cast<T*>(cfilter);
                T result{0};
                for (int32_t idx = 0; idx < filter_size; ++idx)
                    result += shared[tid[1] + idx] * window[idx];
                output(gid) = result;
            }
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolveSeparableY_(AccessorRestrict<const T, 4, uint32_t> input,
                             AccessorRestrict<T, 4, uint32_t> output,
                             uint2_t shape /* YX */, int32_t filter_size, uint32_t blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int2_t tid{threadIdx.y, threadIdx.x};
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_SIZE.y * index[0] + tid[0],
                         BLOCK_SIZE.x * index[1] + tid[1]};

        const auto input_ = input[gid[0]][gid[1]];

        const int32_t padding = filter_size - 1;
        const int32_t halo = padding / 2;
        const int32_t shared_len_y = static_cast<int32_t>(BLOCK_SIZE.y) + padding;

        // Filter along y.
        T* shared = util::block::dynamicSharedResource<T>();
        if (gid[3] < shape[1]) {
            for (int32_t ly = tid[0], gy = gid[2]; ly < shared_len_y; ly += BLOCK_SIZE.y, gy += BLOCK_SIZE.y) {
                const int32_t i_y = gy - halo;
                shared[ly * BLOCK_SIZE.x + tid[1]] = i_y >= 0 && i_y < shape[0] ? input_(i_y, gid[3]) : T{0};
            }
            util::block::synchronize();

            if (gid[2] < shape[0]) {
                const T* window = reinterpret_cast<T*>(cfilter);
                T result{0};
                for (int32_t idx = 0; idx < filter_size; ++idx)
                    result += shared[(tid[0] + idx) * BLOCK_SIZE.x + tid[1]] * window[idx];
                output(gid) = result;
            }
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolveSeparableZ_(AccessorRestrict<const T, 4, uint32_t> input,
                             AccessorRestrict<T, 4, uint32_t> output,
                             uint2_t shape /* ZX */, int32_t filter_size, uint32_t blocks_x) {
        T* shared = util::block::dynamicSharedResource<T>();

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int2_t tid{threadIdx.y, threadIdx.x};
        const int4_t gid{blockIdx.z,
                         BLOCK_SIZE.y * index[0] + tid[0],
                         blockIdx.y,
                         BLOCK_SIZE.x * index[1] + tid[1]};

        const auto input_ = input[gid[0]];

        const int32_t padding = filter_size - 1;
        const int32_t halo = padding / 2;
        const int32_t shared_len_z = static_cast<int32_t>(BLOCK_SIZE.y) + padding;

        if (gid[3] < shape[1]) {
            for (int32_t lz = tid[0], gz = gid[1]; lz < shared_len_z; lz += BLOCK_SIZE.y, gz += BLOCK_SIZE.y) {
                const int32_t i_z = gz - halo;
                shared[lz * BLOCK_SIZE.x + tid[1]] = i_z >= 0 && i_z < shape[0] ? input_(i_z, gid[2], gid[3]) : T{0};
            }
            util::block::synchronize();

            // Weighted sum.
            if (gid[1] < shape[0]) {
                const T* window = reinterpret_cast<T*>(cfilter);
                T result{0};
                for (int32_t idx = 0; idx < filter_size; ++idx)
                    result += shared[(tid[0] + idx) * BLOCK_SIZE.x + tid[1]] * window[idx];
                output(gid) = result;
            }
        }
    }

    template<typename T>
    void launchX(const T* input, uint4_t input_strides, T* output, uint4_t output_strides, uint4_t shape,
                 const T* filter, uint32_t filter_size, cuda::Stream& stream) {
        if (filter_size <= 1)
            return;

        NOA_ASSERT(filter_size * sizeof(T) <= MAX_FILTER_BYTES);
        NOA_THROW_IF(cudaMemcpyToSymbolAsync(cfilter, filter, filter_size * sizeof(T),
                                             0, cudaMemcpyDefault, stream.get()));

        const uint2_t u_shape(shape.get(2));
        const uint32_t blocks_x = noa::math::divideUp(u_shape[1], BLOCK_SIZE.x);
        const uint32_t blocks_y = noa::math::divideUp(u_shape[0], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const uint32_t shared_bytes = (BLOCK_SIZE.x + filter_size - 1) * BLOCK_SIZE.y * sizeof(T);
        const LaunchConfig config{blocks, BLOCK_SIZE, shared_bytes};

        const AccessorRestrict<const T, 4, uint32_t> input_accessor(input, input_strides);
        const AccessorRestrict<T, 4, uint32_t> output_accessor(output, output_strides);
        stream.enqueue("memory::convolveX", convolveSeparableX_<T>, config,
                       input_accessor, output_accessor, u_shape, filter_size, blocks_x);
    }

    template<typename T>
    void launchY(const T* input, uint4_t input_strides, T* output, uint4_t output_strides, uint4_t shape,
                 const T* filter, uint32_t filter_size, cuda::Stream& stream) {
        if (filter_size <= 1)
            return;

        NOA_ASSERT(filter_size * sizeof(T) <= MAX_FILTER_BYTES);
        NOA_THROW_IF(cudaMemcpyToSymbolAsync(cfilter, filter, filter_size * sizeof(T),
                                             0, cudaMemcpyDefault, stream.get()));

        const uint2_t u_shape(shape.get(2));
        const uint32_t blocks_x = noa::math::divideUp(u_shape[1], BLOCK_SIZE.x);
        const uint32_t blocks_y = noa::math::divideUp(u_shape[0], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const uint32_t shared_bytes = BLOCK_SIZE.x * (BLOCK_SIZE.y + filter_size - 1) * sizeof(T);
        const LaunchConfig config{blocks, BLOCK_SIZE, shared_bytes};

        const AccessorRestrict<const T, 4, uint32_t> input_accessor(input, input_strides);
        const AccessorRestrict<T, 4, uint32_t> output_accessor(output, output_strides);
        stream.enqueue("memory::convolveY", convolveSeparableY_<T>, config,
                       input_accessor, output_accessor, u_shape, filter_size, blocks_x);
    }

    template<typename T>
    void launchZ(const T* input, uint4_t input_strides, T* output, uint4_t output_strides, uint4_t shape,
                 const T* filter, uint32_t filter_size, cuda::Stream& stream) {
        if (filter_size <= 1)
            return;

        NOA_ASSERT(filter_size * sizeof(T) <= MAX_FILTER_BYTES);
        NOA_THROW_IF(cudaMemcpyToSymbolAsync(cfilter, filter, filter_size * sizeof(T),
                                             0, cudaMemcpyDefault, stream.get()));

        const uint2_t u_shape{shape[1], shape[3]};
        const uint32_t blocks_x = noa::math::divideUp(u_shape[1], BLOCK_SIZE.x);
        const uint32_t blocks_z = noa::math::divideUp(u_shape[0], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_z, shape[2], shape[0]);
        const uint32_t shared_bytes = BLOCK_SIZE.x * (BLOCK_SIZE.y + filter_size - 1) * sizeof(T);
        const LaunchConfig config{blocks, BLOCK_SIZE, shared_bytes};

        const AccessorRestrict<const T, 4, uint32_t> input_accessor(input, input_strides);
        const AccessorRestrict<T, 4, uint32_t> output_accessor(output, output_strides);
        stream.enqueue("memory::convolveZ", convolveSeparableZ_<T>, config,
                       input_accessor, output_accessor, u_shape, filter_size, blocks_x);
    }
}

namespace noa::cuda::signal {
    template<typename T, typename U, typename>
    void convolve(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  const shared_t<U[]>& filter0, dim_t filter0_size,
                  const shared_t<U[]>& filter1, dim_t filter1_size,
                  const shared_t<U[]>& filter2, dim_t filter2_size, Stream& stream,
                  const shared_t<T[]>& tmp, dim4_t tmp_strides) {
        NOA_ASSERT(input != output);

        const auto input_strides_ = safe_cast<uint4_t>(input_strides);
        const auto output_strides_ = safe_cast<uint4_t>(output_strides);
        const auto shape_ = safe_cast<uint4_t>(shape);

        int32_t count = 0;
        if (filter0)
            count += 1;
        if (filter1)
            count += 1;
        if (filter2)
            count += 1;
        const bool allocate = !tmp && count > 1;
        const shared_t<T[]> buf = allocate ? memory::PtrDevice<T>::alloc(shape.elements(), stream) : tmp;
        const auto buf_strides = safe_cast<uint4_t>(allocate ? shape.strides() : tmp_strides);

        if (filter0 && filter1 && filter2) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter1_size % 2);
            NOA_ASSERT(filter2_size % 2);
            launchZ(input.get(), input_strides_, output.get(), output_strides_, shape_, filter0.get(), filter0_size, stream);
            launchY(output.get(), output_strides_, buf.get(), buf_strides, shape_, filter1.get(), filter1_size, stream);
            launchX(buf.get(), buf_strides, output.get(), output_strides_, shape_, filter2.get(), filter2_size, stream);
            stream.attach(input, output, buf, filter0, filter1, filter2);

        } else if (filter0 && filter1) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter1_size % 2);
            launchZ(input.get(), input_strides_, buf.get(), buf_strides, shape_, filter0.get(), filter0_size, stream);
            launchY(buf.get(), buf_strides, output.get(), output_strides_, shape_, filter1.get(), filter1_size, stream);
            stream.attach(input, output, buf, filter0, filter1);

        } else if (filter0 && filter2) {
            NOA_ASSERT(filter0_size % 2);
            NOA_ASSERT(filter2_size % 2);
            launchZ(input.get(), input_strides_, buf.get(), buf_strides, shape_, filter0.get(), filter0_size, stream);
            launchX(buf.get(), buf_strides, output.get(), output_strides_, shape_, filter2.get(), filter2_size, stream);
            stream.attach(input, output, buf, filter0, filter2);

        } else if (filter1 && filter2) {
            NOA_ASSERT(filter1_size % 2);
            NOA_ASSERT(filter2_size % 2);
            launchY(input.get(), input_strides_, buf.get(), buf_strides, shape_, filter1.get(), filter1_size, stream);
            launchX(buf.get(), buf_strides, output.get(), output_strides_, shape_, filter2.get(), filter2_size, stream);
            stream.attach(input, output, buf, filter1, filter2);

        } else if (filter0) {
            NOA_ASSERT(filter0_size % 2);
            launchZ(input.get(), input_strides_, output.get(), output_strides_, shape_, filter0.get(), filter0_size, stream);
            stream.attach(input, output, filter0);
        } else if (filter1) {
            NOA_ASSERT(filter1_size % 2);
            launchY(input.get(), input_strides_, output.get(), output_strides_, shape_, filter1.get(), filter1_size, stream);
            stream.attach(input, output, filter1);
        } else if (filter2) {
            NOA_ASSERT(filter2_size % 2);
            launchX(input.get(), input_strides_, output.get(), output_strides_, shape_, filter2.get(), filter2_size, stream);
            stream.attach(input, output, filter2);
        }
    }

    template<typename T, typename U, typename>
    void convolve(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  const shared_t<U[]>& filter, dim3_t filter_shape, Stream& stream) {
        NOA_ASSERT(all(filter_shape >= 1));
        const dim_t ndim = filter_shape.ndim();

        // If there's a single dimension, use separable convolution kernels:
        if (ndim == 1 || (ndim == 3 && filter_shape[1] == 1 && filter_shape[2])) {
            if (all(filter_shape == 1)) {
                // Make sure the (single) value is dereferenceable.
                U filter_value;
                U* filter_ = util::hostPointer(filter.get());
                if (filter_) {
                    filter_value = filter_[0];
                } else {
                    memory::copy(filter.get(), &filter_value, 1, stream);
                    stream.attach(filter);
                }
                math::ewise(input, input_strides, static_cast<T>(filter_value),
                            output, output_strides, shape,
                            noa::math::multiply_t{}, stream);
            } else if (filter_shape[2] > 1) {
                convolve<T, U>(input, input_strides, output, output_strides, shape,
                               nullptr, 0, nullptr, 0, filter, filter_shape[2], stream);
            } else if (filter_shape[1] > 1) {
                convolve<T, U>(input, input_strides, output, output_strides, shape,
                               nullptr, 0, filter, filter_shape[1], nullptr, 0, stream);
            } else {
                convolve<T, U>(input, input_strides, output, output_strides, shape,
                               filter, filter_shape[0], nullptr, 0, nullptr, 0, stream);
            }
            return;
        } else if (ndim == 2) {
            return convolve2(input, input_strides, output, output_strides,
                             shape, filter, {filter_shape[1], filter_shape[2]}, stream);
        } else {
            return convolve3(input, input_strides, output, output_strides,
                             shape, filter, filter_shape, stream);
        }
    }

    #define NOA_INSTANTIATE_CONV_(T)                                        \
    template void convolve<T,T,void>(const shared_t<T[]>&, dim4_t,          \
                                     const shared_t<T[]>&, dim4_t, dim4_t,  \
                                     const shared_t<T[]>&, dim_t,           \
                                     const shared_t<T[]>&, dim_t,           \
                                     const shared_t<T[]>&, dim_t,  Stream&, \
                                     const shared_t<T[]>&, dim4_t);         \
    template void convolve<T,T,void>(const shared_t<T[]>&, dim4_t,          \
                                     const shared_t<T[]>&, dim4_t, dim4_t,  \
                                     const shared_t<T[]>&, dim3_t,  Stream&)

    NOA_INSTANTIATE_CONV_(half_t);
    NOA_INSTANTIATE_CONV_(float);
    NOA_INSTANTIATE_CONV_(double);
}
