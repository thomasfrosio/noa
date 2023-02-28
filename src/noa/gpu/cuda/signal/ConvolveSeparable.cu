#include "noa/core/Math.hpp"
#include "noa/gpu/cuda/Ewise.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/memory/PtrDevice.hpp"
#include "noa/gpu/cuda/signal/Convolve.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/Pointers.hpp"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    constexpr dim3 BLOCK_SIZE(16, 16);
    constexpr i32 MAX_FILTER_SIZE = 129;
    constexpr i32 MAX_FILTER_BYTES = MAX_FILTER_SIZE * sizeof(f64);
    __constant__ char cfilter[MAX_FILTER_BYTES];

    // This is identical to the convolve1_ kernel.
    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolve_separable_x_(AccessorRestrict<const T, 4, u32> input,
                               AccessorRestrict<T, 4, u32> output,
                               Shape2<i32> shape_yx, i32 filter_size, u32 blocks_x) {

        const auto index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>{threadIdx.y, threadIdx.x};
        const auto gid = Vec4<i32>{blockIdx.z,
                                   blockIdx.y,
                                   BLOCK_SIZE.y * index[0] + tid[0],
                                   BLOCK_SIZE.x * index[1] + tid[1]};
        const auto input_x = input[gid[0]][gid[1]][gid[2]];

        const i32 padding = filter_size - 1;
        const i32 halo = padding / 2;
        const i32 shared_len = static_cast<i32>(BLOCK_SIZE.x) + padding;

        // Filter along x.
        T* shared = noa::cuda::utils::block_dynamic_shared_resource<T>();
        if (gid[2] < shape_yx[0]) {
            shared += tid[0] * shared_len; // focus on current row

            // Load shared memory. Loop to take into account padding.
            for (i32 lx = tid[1], gx = gid[3]; lx < shared_len; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x) {
                const i32 i_x = gx - halo;
                shared[lx] = i_x >= 0 && i_x < shape_yx[1] ? input_x[i_x] : T{0};
            }
            noa::cuda::utils::block_synchronize();

            if (gid[3] < shape_yx[1]) {
                // Weighted sum.
                const T* window = reinterpret_cast<T*>(cfilter);
                T result{0};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += shared[tid[1] + idx] * window[idx];
                output(gid) = result;
            }
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolve_separable_y_(AccessorRestrict<const T, 4, u32> input,
                               AccessorRestrict<T, 4, u32> output,
                               Shape2<i32> shape_yx, i32 filter_size, u32 blocks_x) {

        const auto index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>{threadIdx.y, threadIdx.x};
        const auto gid = Vec4<i32>{blockIdx.z,
                                   blockIdx.y,
                                   BLOCK_SIZE.y * index[0] + tid[0],
                                   BLOCK_SIZE.x * index[1] + tid[1]};
        const auto input_yx = input[gid[0]][gid[1]];

        const i32 padding = filter_size - 1;
        const i32 halo = padding / 2;
        const i32 shared_len_y = static_cast<i32>(BLOCK_SIZE.y) + padding;

        // Filter along y.
        T* shared = noa::cuda::utils::block_dynamic_shared_resource<T>();
        if (gid[3] < shape_yx[1]) {
            for (i32 ly = tid[0], gy = gid[2]; ly < shared_len_y; ly += BLOCK_SIZE.y, gy += BLOCK_SIZE.y) {
                const i32 i_y = gy - halo;
                shared[ly * BLOCK_SIZE.x + tid[1]] =
                        i_y >= 0 && i_y < shape_yx[0] ?
                        input_yx(i_y, gid[3]) : T{0};
            }
            noa::cuda::utils::block_synchronize();

            if (gid[2] < shape_yx[0]) {
                const T* window = reinterpret_cast<T*>(cfilter);
                T result{0};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += shared[(tid[0] + idx) * BLOCK_SIZE.x + tid[1]] * window[idx];
                output(gid) = result;
            }
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolve_separable_z_(AccessorRestrict<const T, 4, u32> input,
                               AccessorRestrict<T, 4, u32> output,
                               Shape2<i32> shape_zx, i32 filter_size, u32 blocks_x) {
        T* shared = noa::cuda::utils::block_dynamic_shared_resource<T>();

        const auto index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>{threadIdx.y, threadIdx.x};
        const auto gid = Vec4<i32>{blockIdx.z,
                                   BLOCK_SIZE.y * index[0] + tid[0],
                                   blockIdx.y,
                                   BLOCK_SIZE.x * index[1] + tid[1]};
        const auto input_3d = input[gid[0]];

        const i32 padding = filter_size - 1;
        const i32 halo = padding / 2;
        const i32 shared_len_z = static_cast<i32>(BLOCK_SIZE.y) + padding;

        if (gid[3] < shape_zx[1]) {
            for (i32 lz = tid[0], gz = gid[1]; lz < shared_len_z; lz += BLOCK_SIZE.y, gz += BLOCK_SIZE.y) {
                const i32 i_z = gz - halo;
                shared[lz * BLOCK_SIZE.x + tid[1]] =
                        i_z >= 0 && i_z < shape_zx[0] ?
                        input_3d(i_z, gid[2], gid[3]) : T{0};
            }
            noa::cuda::utils::block_synchronize();

            // Weighted sum.
            if (gid[1] < shape_zx[0]) {
                const T* window = reinterpret_cast<T*>(cfilter);
                T result{0};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += shared[(tid[0] + idx) * BLOCK_SIZE.x + tid[1]] * window[idx];
                output(gid) = result;
            }
        }
    }

    template<typename T>
    void launch_convolve_separable_x_(
            const T* input, const Strides4<u32>& input_strides,
            T* output, const Strides4<u32>& output_strides, const Shape4<u32>& shape,
            const T* filter, u32 filter_size, cuda::Stream& stream) {
        if (filter_size == 1) {
            T filter_value;
            noa::cuda::memory::copy(filter, &filter_value, 1, stream);
            return noa::cuda::ewise_binary(
                    input, input_strides.as<i64>(), filter_value,
                    output, output_strides.as<i64>(), shape.as<i64>(),
                    noa::multiply_t{}, stream);
        }

        NOA_ASSERT(filter_size * sizeof(T) <= MAX_FILTER_BYTES);
        NOA_THROW_IF(cudaMemcpyToSymbolAsync(
                cfilter, filter, filter_size * sizeof(T),
                0, cudaMemcpyDefault, stream.get()));

        const u32 blocks_x = noa::math::divide_up(shape[3], BLOCK_SIZE.x);
        const u32 blocks_y = noa::math::divide_up(shape[2], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const u32 shared_bytes = (BLOCK_SIZE.x + filter_size - 1) * BLOCK_SIZE.y * static_cast<u32>(sizeof(T));
        const auto config = cuda::LaunchConfig{blocks, BLOCK_SIZE, shared_bytes};

        const auto input_accessor = AccessorRestrict<const T, 4, u32> (input, input_strides);
        const auto output_accessor = AccessorRestrict<T, 4, u32> (output, output_strides);
        const auto shape_2d = shape.filter(2, 3).as<i32>();
        stream.enqueue("convolve_separable_x", convolve_separable_x_<T>, config,
                       input_accessor, output_accessor, shape_2d, filter_size, blocks_x);
    }

    template<typename T>
    void launch_convolve_separable_y_(
            const T* input, const Strides4<u32>& input_strides,
            T* output, const Strides4<u32>& output_strides, const Shape4<u32>& shape,
            const T* filter, u32 filter_size, cuda::Stream& stream) {
        if (filter_size == 1) {
            T filter_value;
            noa::cuda::memory::copy(filter, &filter_value, 1, stream);
            return noa::cuda::ewise_binary(
                    input, input_strides.as<i64>(), filter_value,
                    output, output_strides.as<i64>(), shape.as<i64>(),
                    noa::multiply_t{}, stream);
        }

        NOA_ASSERT(filter_size * sizeof(T) <= MAX_FILTER_BYTES);
        NOA_THROW_IF(cudaMemcpyToSymbolAsync(
                cfilter, filter, filter_size * sizeof(T),
                0, cudaMemcpyDefault, stream.get()));

        const u32 blocks_x = noa::math::divide_up(shape[3], BLOCK_SIZE.x);
        const u32 blocks_y = noa::math::divide_up(shape[2], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const u32 shared_bytes = BLOCK_SIZE.x * (BLOCK_SIZE.y + filter_size - 1) * static_cast<u32>(sizeof(T));
        const auto config = cuda::LaunchConfig{blocks, BLOCK_SIZE, shared_bytes};

        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides);
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides);
        const auto shape_2d = shape.filter(2, 3).as<i32>();
        stream.enqueue("convolve_separable_y", convolve_separable_y_<T>, config,
                       input_accessor, output_accessor, shape_2d, filter_size, blocks_x);
    }

    template<typename T>
    void launch_convolve_separable_z_(
            const T* input, const Strides4<u32>& input_strides,
            T* output, const Strides4<u32>& output_strides, const Shape4<u32>& shape,
            const T* filter, u32 filter_size, cuda::Stream& stream) {
        if (filter_size == 1) {
            T filter_value;
            noa::cuda::memory::copy(filter, &filter_value, 1, stream);
            return noa::cuda::ewise_binary(
                    input, input_strides.as<i64>(), filter_value,
                    output, output_strides.as<i64>(), shape.as<i64>(),
                    noa::multiply_t{}, stream);
        }

        NOA_ASSERT(filter_size * sizeof(T) <= MAX_FILTER_BYTES);
        NOA_THROW_IF(cudaMemcpyToSymbolAsync(
                cfilter, filter, filter_size * sizeof(T),
                0, cudaMemcpyDefault, stream.get()));

        const u32 blocks_x = noa::math::divide_up(shape[3], BLOCK_SIZE.x);
        const u32 blocks_z = noa::math::divide_up(shape[1], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_z, shape[2], shape[0]);
        const u32 shared_bytes = BLOCK_SIZE.x * (BLOCK_SIZE.y + filter_size - 1) * static_cast<u32>(sizeof(T));
        const auto config = cuda::LaunchConfig{blocks, BLOCK_SIZE, shared_bytes};

        const auto input_accessor = AccessorRestrict<const T, 4, u32> (input, input_strides);
        const auto output_accessor = AccessorRestrict<T, 4, u32> (output, output_strides);
        const auto shape_2d = shape.filter(1,3).as_safe<i32>();
        stream.enqueue("convolve_separable_z", convolve_separable_z_<T>, config,
                       input_accessor, output_accessor, shape_2d, filter_size, blocks_x);
    }
}

namespace noa::cuda::signal {
    template<typename T, typename U, typename>
    void convolve_separable(const T* input, const Strides4<i64>& input_strides,
                            T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                            const U* filter_depth, i64 filter_depth_size,
                            const U* filter_height, i64 filter_height_size,
                            const U* filter_width, i64 filter_width_size,
                            T* tmp, Strides4<i64> tmp_strides, Stream& stream) {
        NOA_ASSERT(input != output && noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT_DEVICE_OR_NULL_PTR(tmp, stream.device());
        NOA_ASSERT_DEVICE_OR_NULL_PTR(filter_depth, stream.device());
        NOA_ASSERT_DEVICE_OR_NULL_PTR(filter_height, stream.device());
        NOA_ASSERT_DEVICE_OR_NULL_PTR(filter_width, stream.device());

        const auto u_input_strides = input_strides.as_safe<u32>();
        const auto u_output_strides = output_strides.as_safe<u32>();
        const auto u_shape = shape.as_safe<u32>();

        if (filter_depth_size <= 0)
            filter_depth = nullptr;
        if (filter_height_size <= 0)
            filter_height = nullptr;
        if (filter_width_size <= 0)
            filter_width = nullptr;

        // Allocate temp buffer if necessary.
        i32 count = 0;
        if (filter_depth)
            count += 1;
        if (filter_height)
            count += 1;
        if (filter_width)
            count += 1;
        using allocator_t = noa::cuda::memory::PtrDevice<T>;
        typename allocator_t::unique_type buffer{};
        if (!tmp && count > 1) {
            buffer = allocator_t::alloc(shape.elements(), stream);
            tmp = buffer.get();
            tmp_strides = shape.strides();
        }
        const auto u_tmp_strides = tmp_strides.as_safe<u32>();

        NOA_ASSERT(!filter_depth || filter_depth_size % 2);
        NOA_ASSERT(!filter_height || filter_height_size % 2);
        NOA_ASSERT(!filter_width || filter_width_size % 2);

        if (filter_depth && filter_height && filter_width) {
            launch_convolve_separable_z_(
                    input, u_input_strides, output, u_output_strides, u_shape,
                    filter_depth, filter_depth_size, stream);
            launch_convolve_separable_y_(
                    output, u_output_strides, tmp, u_tmp_strides, u_shape,
                    filter_height, filter_height_size, stream);
            launch_convolve_separable_x_(
                    tmp, u_tmp_strides, output, u_output_strides, u_shape,
                    filter_width, filter_width_size, stream);

        } else if (filter_depth && filter_height) {
            launch_convolve_separable_z_(
                    input, u_input_strides, tmp, u_tmp_strides, u_shape,
                    filter_depth, filter_depth_size, stream);
            launch_convolve_separable_y_(
                    tmp, u_tmp_strides, output, u_output_strides, u_shape,
                    filter_height, filter_height_size, stream);

        } else if (filter_depth && filter_width) {
            launch_convolve_separable_z_(
                    input, u_input_strides, tmp, u_tmp_strides, u_shape,
                    filter_depth, filter_depth_size, stream);
            launch_convolve_separable_x_(
                    tmp, u_tmp_strides, output, u_output_strides, u_shape,
                    filter_width, filter_width_size, stream);

        } else if (filter_height && filter_width) {
            launch_convolve_separable_y_(
                    input, u_input_strides, tmp, u_tmp_strides, u_shape,
                    filter_height, filter_height_size, stream);
            launch_convolve_separable_x_(
                    tmp, u_tmp_strides, output, u_output_strides, u_shape,
                    filter_width, filter_width_size, stream);

        } else if (filter_depth) {
            launch_convolve_separable_z_(
                    input, u_input_strides, output, u_output_strides, u_shape,
                    filter_depth, filter_depth_size, stream);
        } else if (filter_height) {
            launch_convolve_separable_y_(
                    input, u_input_strides, output, u_output_strides, u_shape,
                    filter_height, filter_height_size, stream);
        } else if (filter_width) {
            launch_convolve_separable_x_(
                    input, u_input_strides, output, u_output_strides, u_shape,
                    filter_width, filter_width_size, stream);
        }
    }

    template<typename T, typename U, typename>
    void convolve(const T* input, const Strides4<i64>& input_strides,
                  T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                  const U* filter, const Shape3<i64>& filter_shape, Stream& stream) {
        NOA_ASSERT(noa::all(filter_shape > 0) && noa::all(shape > 0));

        // If there's a single dimension, use separable convolution kernels:
        const auto dimensions_to_convolve = noa::math::sum(filter_shape > 1);
        const auto ndim = filter_shape.ndim();

        if (dimensions_to_convolve == 1) {
            if (filter_shape[0] > 1) {
                launch_convolve_separable_z_(
                        input, input_strides.as_safe<u32>(),
                        output, output_strides.as_safe<u32>(), shape.as_safe<u32>(),
                        filter, filter_shape[0], stream);
            } else if (filter_shape[1] > 1) {
                launch_convolve_separable_y_(
                        input, input_strides.as_safe<u32>(),
                        output, output_strides.as_safe<u32>(), shape.as_safe<u32>(),
                        filter, filter_shape[1], stream);
            } else {
                launch_convolve_separable_x_(
                        input, input_strides.as_safe<u32>(),
                        output, output_strides.as_safe<u32>(), shape.as_safe<u32>(),
                        filter, filter_shape[2], stream);
            }
        } else if (ndim == 2) {
            return convolve_2d(input, input_strides, output, output_strides,
                               shape, filter, filter_shape.pop_front(), stream);
        } else if (ndim == 3) {
            return convolve_3d(input, input_strides, output, output_strides,
                               shape, filter, filter_shape, stream);
        } else if (noa::all(filter_shape == 1)) {
            T filter_value;
            noa::cuda::memory::copy(filter, &filter_value, 1, stream);
            return noa::cuda::ewise_binary(
                    input, input_strides, filter_value,
                    output, output_strides, shape,
                    noa::multiply_t{}, stream);
        } else {
            NOA_THROW("DEV: unreachable");
        }
    }

    #define NOA_INSTANTIATE_CONV_(T, U)                 \
    template void convolve_separable<T, U, void>(       \
        const T*, const Strides4<i64>&,                 \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const U*, i64, const U*, i64, const U*, i64,    \
        T*, Strides4<i64>, Stream&);                    \
    template void convolve<T, U, void>(                 \
        const T*, const Strides4<i64>&,                 \
        T*, const Strides4<i64>&, const Shape4<i64>&,   \
        const U*, const Shape3<i64>&, Stream&)

    NOA_INSTANTIATE_CONV_(f16, f16);
    NOA_INSTANTIATE_CONV_(f32, f32);
    NOA_INSTANTIATE_CONV_(f64, f64);
}
