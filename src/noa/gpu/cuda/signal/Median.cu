#include "noa/common/Math.h"

#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/signal/Median.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/util/Block.cuh"

// The current implementations only supports small squared windows. This allows to:
//  1)  Load the windows for all threads in a block in shared memory. This is useful because windows overlap.
//  2)  The exchange search can be done on the per thread registers. Only about half of the window needs to
//      be on the registers at a single time. The rest stays in shared memory. This also requires the indexing
//      to be constant, i.e. the window size should be a template argument.
// TODO Maybe look at other implementations for larger windows (e.g. with textures)?

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;
    constexpr dim3 BLOCK_SIZE(16, 16);

    // Ensures a < b. If not, swap.
    template<typename T>
    __forceinline__ __device__ void swap_(T& a, T& b) {
        if (b < a) {
            T tmp = a;
            a = b;
            b = tmp;
        }
    }

    // Sets the array v, so that the min is at v[0] and max is at v[length-1].
    // TODO make sure the force inline still allows constant indexing, otherwise replace with a macro.
    template<typename T>
    __forceinline__ __device__ void order_(T* v, int32_t length) {
        for (int32_t i = 0; i < length / 2; i++) // ensure min in first half, max in second half
            swap_(v[i], v[length - 1 - i]);
        for (int32_t i = 1; i < (length + 1) / 2; i++) // move min in first half to first pos
            swap_(v[0], v[i]);
        for (int32_t i = length - 2; i >= length / 2; i--) // move max in second half to last pos
            swap_(v[i], v[length - 1]);
    }

    // Loads the shared memory array according to the BORDER_MODE.
    // input_row:   device memory. Should point to the beginning of the current row.
    // stride:      innermost stride of the input.
    // s_mem:       shared memory. Should point to the current element.
    // shape_x:     number of logical elements in x, x being the dimension of the 1D window.
    // gx:          index of the current element in x. If out of bound, add padding according to the BORDER_MODE.
    template<typename T, int32_t BORDER_MODE, int32_t HALO>
    __device__ void loadToShared1D_(AccessorReference<const T, 1, uint32_t> input_row,
                                    T* s_mem, int32_t shape_x, int32_t gx) {
        static_assert(BORDER_MODE == BORDER_REFLECT || BORDER_MODE == BORDER_ZERO);
        if constexpr (BORDER_MODE == BORDER_REFLECT) {
            if (gx < 0)
                *s_mem = input_row[-gx]; // pad left; requires shape_x >= HALO + 1, since gx >= -HALO
            else if (gx < shape_x)
                *s_mem = input_row[gx]; // in array, no padding
            else if (gx < shape_x + HALO)
                *s_mem = input_row[(2 * (shape_x - 1) - gx)]; // pad right; requires shape_x >= HALO  + 1
            // Otherwise, don't do anything since the *s_mem element will not be used anyway.

        } else if constexpr (BORDER_MODE == BORDER_ZERO) {
            if (gx < 0 || gx >= shape_x)
                *s_mem = static_cast<T>(0);
            else
                *s_mem = input_row[gx];
        }
    }

    template<typename T, int32_t BORDER_MODE, int32_t WINDOW_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void medfilt1_(Accessor<const T, 4, uint32_t> input,
                   Accessor<T, 4, uint32_t> output,
                   uint2_t shape, uint32_t blocks_x) {
        static_assert(WINDOW_SIZE % 2); // only support odd windows.
        constexpr int32_t PADDING = WINDOW_SIZE - 1; // assume odd
        constexpr int32_t HALO = PADDING / 2;

        // The shared memory is the shape of the block + the padding in the first dimension.
        using uninit_t = util::traits::uninitialized_type_t<T>;
        constexpr int2_t SHARED_SIZE(BLOCK_SIZE.y, BLOCK_SIZE.x + PADDING);
        __shared__ uninit_t buffer[math::prod(SHARED_SIZE)];
        T* shared_mem = reinterpret_cast<T*>(buffer);

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int2_t tid{threadIdx.y, threadIdx.x};
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_SIZE.y * index[0] + tid[0],
                         BLOCK_SIZE.x * index[1] + tid[1]};

        const auto input_ = input[gid[0]][gid[1]][gid[2]];

        // There's no padding in y, so if out of bounds, stop.
        if (gid[2] < shape[0]) {
            // Load shared memory. Loop to take into account padding.
            for (int32_t lx = tid[1], gx = gid[3]; lx < SHARED_SIZE[1]; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x) {
                loadToShared1D_<T, BORDER_MODE, HALO>(input_,
                                                      shared_mem + tid[0] * SHARED_SIZE[1] + lx,
                                                      shape[1], gx - HALO);
            }
            util::block::synchronize();

            // Only continue if not out of bound.
            if (gid[3] < shape[1]) {
                // The goal is to reduce register pressure as much as possible, but still use registers
                // to do the exchange sort. The window is divided into two half: the first "activate" half,
                // which is where the sorting happens, and the second half, which is the pool of contestants
                // that will be progressively added.
                constexpr int32_t ACTIVE = WINDOW_SIZE / 2 + 2; // the final size should be at least 3.

                // Load active half to, hopefully, the local registers (otherwise spill to device memory).
                T v[ACTIVE]; // all indexing are known at compile, so registers should be used
                for (int32_t x = 0; x < ACTIVE; ++x)
                    v[x] = shared_mem[tid[0] * SHARED_SIZE[1] + tid[1] + x];

                order_(v, ACTIVE); // ensure min at 0, max at ACTIVE - 1

                // We can replace the min by a new contestant and then reorder.
                // We also don't have to compare the max at ACTIVE - 1, we just need to stack the new max at the end.
                // As such, as we add new contestants, we right-truncate the active half to ignore the previous max.
                int32_t length = ACTIVE;
                for (int32_t k = ACTIVE; k < WINDOW_SIZE; ++k) {
                    v[0] = shared_mem[tid[0] * SHARED_SIZE[1] + tid[1] + k]; // replace min by new contestant
                    --length; // ignore the previous max at the end
                    order_(v, length); // min at 0, max at length - 1
                }

                // Since we know the min and max are at v[0] and v[length-1] respectively, to find the median,
                // we simply need to (exchange) sort the elements from v[1] to v[length-2] included.
                // The median will then be at length/2.
                for (int32_t k = 1; k < length - 1; k++)
                    order_(v + k, length - k);
                output(gid) = v[length / 2];
            }
        }
    }

    // Loads the (gx,gy) element according to the BORDER_MODE.
    // input_slice: device memory. Should point to the beginning of the current slice (z).
    // stride_y/x:  stride in y and x.
    // s_mem:       shared memory. Should point to the current element.
    // shape_y/x:   number of logical elements in y/x.
    // gy/x:        index of the current element in y/x. If out of bound, add padding according to the BORDER_MODE.
    template<typename T, int32_t BORDER_MODE, int32_t HALO>
    __device__ void loadToShared2D_(AccessorReference<const T, 2, uint32_t> input_slice, T* s_mem,
                                    int32_t shape_y, int32_t gy, int32_t shape_x, int32_t gx) {
        static_assert(BORDER_MODE == BORDER_REFLECT || BORDER_MODE == BORDER_ZERO);
        if constexpr (BORDER_MODE == BORDER_REFLECT) {
            if (gx < 0)
                gx *= -1;
            else if (gx >= shape_x) {
                if (gx >= shape_x + HALO)
                    return; // don't do anything; gx is out of the shape + window so *s_mem won't be used anyway
                gx = 2 * (shape_x - 1) - gx;
            }
            if (gy < 0)
                gy *= -1;
            else if (gy >= shape_y) {
                if (gy >= shape_y + HALO)
                    return;
                gy = 2 * (shape_y - 1) - gy;
            }
            *s_mem = input_slice(gy, gx);

        } else if constexpr (BORDER_MODE == BORDER_ZERO) {
            if (gx < 0 || gx >= shape_x || gy < 0 || gy >= shape_y)
                *s_mem = static_cast<T>(0);
            else
                *s_mem = input_slice(gy, gx);
        }
    }

    template<typename T, int32_t BORDER_MODE, int32_t WINDOW_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void medfilt2_(Accessor<const T, 4, uint32_t> input,
                   Accessor<T, 4, uint32_t> output,
                   uint2_t shape, uint32_t blocks_x) {
        static_assert(WINDOW_SIZE % 2); // only support odd windows.
        constexpr int32_t TILE_SIZE = WINDOW_SIZE * WINDOW_SIZE;
        constexpr int32_t PADDING = WINDOW_SIZE - 1; // assume odd
        constexpr int32_t HALO = PADDING / 2;

        // The shared memory is the shape of the block + the padding the first and second dimension.
        using uninit_t = util::traits::uninitialized_type_t<T>;
        constexpr int2_t SHARED_SIZE(BLOCK_SIZE.y + PADDING, BLOCK_SIZE.x + PADDING);
        __shared__ uninit_t buffer[math::prod(SHARED_SIZE)];
        T* shared_mem = reinterpret_cast<T*>(buffer);

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int2_t tid{threadIdx.y, threadIdx.x};
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_SIZE.y * index[0] + tid[0],
                         BLOCK_SIZE.x * index[1] + tid[1]};

        const auto input_ = input[gid[0]][gid[1]];

        // Load shared memory. Loop to account for the halo.
        for (int32_t ly = tid[0], gy = gid[2]; ly < SHARED_SIZE[0]; ly += BLOCK_SIZE.y, gy += BLOCK_SIZE.y)
            for (int32_t lx = tid[1], gx = gid[3]; lx < SHARED_SIZE[1]; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x)
                loadToShared2D_<T, BORDER_MODE, HALO>(input_,
                                                      shared_mem + ly * SHARED_SIZE[1] + lx,
                                                      shape[0], gy - HALO,
                                                      shape[1], gx - HALO);
        util::block::synchronize();

        // Only continue if not out of bound. gid.z cannot be out of bound.
        if (gid[2] < shape[0] && gid[3] < shape[1]) {
            constexpr int32_t ACTIVE = TILE_SIZE / 2 + 2;

            // Load active window from shared memory into this 1D array.
            T v[ACTIVE];
            for (int32_t count = 0, y = 0; y < WINDOW_SIZE; ++y)
                for (int32_t x = 0; count < ACTIVE && x < WINDOW_SIZE; ++x, ++count)
                    v[count] = shared_mem[(tid[0] + y) * SHARED_SIZE[1] + tid[1] + x];

            order_(v, ACTIVE);
            int32_t length = ACTIVE;
            for (int32_t k = ACTIVE; k < TILE_SIZE; ++k) {
                // k is the index, but we want the corresponding (x,y) coordinates in the 2D window.
                // Then offset these coordinates to the current thread.
                const int32_t coord_y = k / WINDOW_SIZE;
                const int32_t coord_x = k - coord_y * WINDOW_SIZE;
                v[0] = shared_mem[(coord_y + tid[0]) * SHARED_SIZE[1] + coord_x + tid[1]];

                --length;
                order_(v, length);
            }

            // Sort the final elements.
            for (int32_t k = 1; k < length - 1; k++)
                order_(v + k, length - k);
            output(gid) = v[length / 2];
        }
    }

    // Loads the (gx,gy,gz) element according to the BORDER_MODE.
    // input:        device memory. Should point to the beginning of the batch.
    // stride_z/y/x: strides
    // s_mem:        shared memory. Should point to the current element.
    // shape_z/y/x:  number of logical elements in z/y/x.
    // gz/y/x:       index of the current element in z/y/x. If out of bound, add padding according to the BORDER_MODE.
    template<typename T, int32_t BORDER_MODE, int32_t HALO>
    __device__ void loadToShared3D_(AccessorReference<const T, 3, uint32_t> input, T* s_mem,
                                    int32_t shape_z, int32_t gz,
                                    int32_t shape_y, int32_t gy,
                                    int32_t shape_x, int32_t gx) {
        static_assert(BORDER_MODE == BORDER_REFLECT || BORDER_MODE == BORDER_ZERO);
        if constexpr (BORDER_MODE == BORDER_REFLECT) {
            if (gx < 0) {
                gx *= -1;
            } else if (gx >= shape_x) {
                if (gx >= shape_x + HALO)
                    return;
                gx = 2 * (shape_x - 1) - gx;
            }
            if (gy < 0) {
                gy *= -1;
            } else if (gy >= shape_y) {
                if (gy >= shape_y + HALO)
                    return;
                gy = 2 * (shape_y - 1) - gy;
            }
            if (gz < 0) {
                gz *= -1;
            } else if (gz >= shape_z) {
                if (gz >= shape_z + HALO)
                    return;
                gz = 2 * (shape_z - 1) - gz;
            }
            *s_mem = input(gz, gy, gx);

        } else if constexpr (BORDER_MODE == BORDER_ZERO) {
            if (gx < 0 || gx >= shape_x ||
                gy < 0 || gy >= shape_y ||
                gz < 0 || gz >= shape_z)
                *s_mem = static_cast<T>(0);
            else
                *s_mem = input(gz, gy, gx);
        }
    }

    // The launch config and block size is like medfilt1_.
    template<typename T, int32_t BORDER_MODE, uint32_t WINDOW_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void medfilt3_(Accessor<const T, 4, uint32_t> input,
                   Accessor<T, 4, uint32_t> output,
                   uint3_t shape, uint32_t blocks_x) {
        static_assert(WINDOW_SIZE % 2); // only support odd windows.
        constexpr int32_t TILE_SIZE = WINDOW_SIZE * WINDOW_SIZE * WINDOW_SIZE;
        constexpr int32_t PADDING = WINDOW_SIZE - 1; // assume odd
        constexpr int32_t HALO = PADDING / 2;

        // The shared memory is the shape of the block + the padding all 3 dimensions.
        // For the largest supported window (5), this goes up to 20*20*5=2205 elements.
        using uninit_t = util::traits::uninitialized_type_t<T>;
        constexpr int3_t SHARED_SIZE(WINDOW_SIZE, BLOCK_SIZE.y + PADDING, BLOCK_SIZE.x + PADDING);
        __shared__ uninit_t buffer[math::prod(SHARED_SIZE)];
        T* shared_mem = reinterpret_cast<T*>(buffer);

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int2_t tid{threadIdx.y, threadIdx.x};
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_SIZE.y * index[0] + tid[0],
                         BLOCK_SIZE.x * index[1] + tid[1]};

        const auto input_ = input[gid[0]];

        // Load shared memory.
        // Each thread processes at least WINDOW_SIZE elements (the z dimension).
        for (int32_t lz = 0, gz = gid[1]; lz < SHARED_SIZE[0]; ++lz, ++gz)
            for (int32_t ly = tid[0], gy = gid[2]; ly < SHARED_SIZE[1]; ly += BLOCK_SIZE.y, gy += BLOCK_SIZE.y)
                for (int32_t lx = tid[1], gx = gid[3]; lx < SHARED_SIZE[2]; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x)
                    loadToShared3D_<T, BORDER_MODE, HALO>(
                            input_,
                            shared_mem + (lz * SHARED_SIZE[1] + ly) * SHARED_SIZE[2] + lx,
                            shape[0], gz - HALO, shape[1], gy - HALO, shape[2], gx - HALO);
        util::block::synchronize();

        // Only continue if not out of bound. gid.z cannot be out of bound.
        if (gid[2] < shape[1] && gid[3] < shape[2]) {
            constexpr int32_t ACTIVE = TILE_SIZE / 2 + 2;

            // Load active window from shared memory into this 1D array.
            T v[ACTIVE];
            for (int32_t count = 0, z = 0; z < WINDOW_SIZE; ++z)
                for (int32_t y = 0; count < ACTIVE && y < WINDOW_SIZE; ++y)
                    for (int32_t x = 0; count < ACTIVE && x < WINDOW_SIZE; ++x, ++count)
                        v[count] = shared_mem[(z * SHARED_SIZE[1] + tid[0] + y) * SHARED_SIZE[2] + tid[1] + x];

            order_(v, ACTIVE);
            int32_t length = ACTIVE;
            for (int32_t k = ACTIVE; k < TILE_SIZE; ++k) {
                // k is the index, but we want the corresponding (x,y,z) coordinates in the 3D window.
                // Then offset these coordinates to the current thread.
                const int32_t coord_z = k / (WINDOW_SIZE * WINDOW_SIZE);
                const int32_t tmp = k - coord_z * WINDOW_SIZE * WINDOW_SIZE;
                const int32_t coord_y = tmp / WINDOW_SIZE;
                const int32_t coord_x = tmp - coord_y * WINDOW_SIZE;
                v[0] = shared_mem[(coord_z * SHARED_SIZE[1] + coord_y + tid[0]) * SHARED_SIZE[2] + coord_x + tid[1]];
                --length;
                order_(v, length);
            }

            // Sort the final elements.
            for (int32_t k = 1; k < length - 1; k++)
                order_(v + k, length - k);
            output(gid) = v[length / 2];
        }
    }
}

namespace noa::cuda::signal {
    template<typename T, typename>
    void median1(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides,
                 dim4_t shape, BorderMode border_mode, dim_t window_size, Stream& stream) {
        NOA_ASSERT(input != output && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        if (window_size <= 1)
            return memory::copy(input, input_strides, output, output_strides, shape, stream);

        const auto uint_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
        const auto uint_input_strides = safe_cast<uint4_t>(input_strides);
        const auto uint_output_strides = safe_cast<uint4_t>(output_strides);
        const uint32_t blocks_x = math::divideUp(uint_shape[1], BLOCK_SIZE.x);
        const uint32_t blocks_y = math::divideUp(uint_shape[0], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const LaunchConfig config{blocks, BLOCK_SIZE};

        const Accessor<const T, 4, uint32_t> input_accessor(input.get(), uint_input_strides);
        const Accessor<T, 4, uint32_t> output_accessor(output.get(), uint_output_strides);

        NOA_ASSERT(border_mode != BORDER_REFLECT || window_size / 2 + 1 <= shape[3]);
        if (border_mode != BORDER_REFLECT && border_mode != BORDER_ZERO) {
            NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                      BORDER_REFLECT, BORDER_ZERO, border_mode);
        }

        switch (window_size) {
            case 3:
                stream.enqueue(
                        "signal::medfilt1",
                        border_mode == BORDER_REFLECT ? medfilt1_<T, BORDER_REFLECT, 3> : medfilt1_<T, BORDER_ZERO, 3>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 5:
                stream.enqueue(
                        "signal::medfilt1",
                        border_mode == BORDER_REFLECT ? medfilt1_<T, BORDER_REFLECT, 5> : medfilt1_<T, BORDER_ZERO, 5>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 7:
                stream.enqueue(
                        "signal::medfilt1",
                        border_mode == BORDER_REFLECT ? medfilt1_<T, BORDER_REFLECT, 7> : medfilt1_<T, BORDER_ZERO, 7>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 9:
                stream.enqueue(
                        "signal::medfilt1",
                        border_mode == BORDER_REFLECT ? medfilt1_<T, BORDER_REFLECT, 9> : medfilt1_<T, BORDER_ZERO, 9>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 11:
                stream.enqueue(
                        "signal::medfilt1",
                        border_mode == BORDER_REFLECT ? medfilt1_<T, BORDER_REFLECT, 11> : medfilt1_<T, BORDER_ZERO, 11>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 13:
                stream.enqueue(
                        "signal::medfilt1",
                        border_mode == BORDER_REFLECT ? medfilt1_<T, BORDER_REFLECT, 13> : medfilt1_<T, BORDER_ZERO, 13>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 15:
                stream.enqueue(
                        "signal::medfilt1",
                        border_mode == BORDER_REFLECT ? medfilt1_<T, BORDER_REFLECT, 15> : medfilt1_<T, BORDER_ZERO, 15>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 17:
                stream.enqueue(
                        "signal::medfilt1",
                        border_mode == BORDER_REFLECT ? medfilt1_<T, BORDER_REFLECT, 17> : medfilt1_<T, BORDER_ZERO, 17>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 19:
                stream.enqueue(
                        "signal::medfilt1",
                        border_mode == BORDER_REFLECT ? medfilt1_<T, BORDER_REFLECT, 19> : medfilt1_<T, BORDER_ZERO, 19>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 21:
                stream.enqueue(
                        "signal::medfilt1",
                        border_mode == BORDER_REFLECT ? medfilt1_<T, BORDER_REFLECT, 21> : medfilt1_<T, BORDER_ZERO, 21>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            default:
                NOA_THROW("Unsupported window size. It should be an odd number from 1 to 21, got {}", window_size);
        }
        stream.attach(input, output);
    }

    template<typename T, typename>
    void median2(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides,
                 dim4_t shape, BorderMode border_mode, dim_t window_size, Stream& stream) {
        NOA_ASSERT(input != output && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        if (window_size <= 1)
            return memory::copy(input, input_strides, output, output_strides, shape, stream);

        const dim2_t order_2d = indexing::order(dim2_t(output_strides.get(2)), dim2_t(shape.get(2)));
        if (any(order_2d != dim2_t{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
        }

        const auto uint_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
        const auto uint_input_strides = safe_cast<uint4_t>(input_strides);
        const auto uint_output_strides = safe_cast<uint4_t>(output_strides);
        const uint32_t blocks_x = math::divideUp(uint_shape[1], BLOCK_SIZE.x);
        const uint32_t blocks_y = math::divideUp(uint_shape[0], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const LaunchConfig config{blocks, BLOCK_SIZE};

        const Accessor<const T, 4, uint32_t> input_accessor(input.get(), uint_input_strides);
        const Accessor<T, 4, uint32_t> output_accessor(output.get(), uint_output_strides);

        NOA_ASSERT(border_mode != BORDER_REFLECT || all(window_size / 2 + 1 <= uint_shape));
        if (border_mode != BORDER_REFLECT && border_mode != BORDER_ZERO) {
            NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                      BORDER_REFLECT, BORDER_ZERO, border_mode);
        }

        switch (window_size) {
            case 3:
                stream.enqueue(
                        "signal::medfilt2",
                        border_mode == BORDER_REFLECT ? medfilt2_<T, BORDER_REFLECT, 3> : medfilt2_<T, BORDER_ZERO, 3>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 5:
                stream.enqueue(
                        "signal::medfilt2",
                        border_mode == BORDER_REFLECT ? medfilt2_<T, BORDER_REFLECT, 5> : medfilt2_<T, BORDER_ZERO, 5>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 7:
                stream.enqueue(
                        "signal::medfilt2",
                        border_mode == BORDER_REFLECT ? medfilt2_<T, BORDER_REFLECT, 7> : medfilt2_<T, BORDER_ZERO, 7>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 9:
                stream.enqueue(
                        "signal::medfilt2",
                        border_mode == BORDER_REFLECT ? medfilt2_<T, BORDER_REFLECT, 9> : medfilt2_<T, BORDER_ZERO, 9>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 11:
                stream.enqueue(
                        "signal::medfilt2",
                        border_mode == BORDER_REFLECT ? medfilt2_<T, BORDER_REFLECT, 11> : medfilt2_<T, BORDER_ZERO, 11>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            default:
                NOA_THROW("Unsupported window size. It should be an odd number from 1 to 11, got {}", window_size);
        }
        stream.attach(input, output);
    }

    template<typename T, typename>
    void median3(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides,
                 dim4_t shape, BorderMode border_mode, dim_t window_size, Stream& stream) {
        NOA_ASSERT(input != output && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        if (window_size <= 1)
            return memory::copy(input, input_strides, output, output_strides, shape, stream);

        const dim3_t order_3d = indexing::order(dim3_t(output_strides.get(1)), dim3_t(shape.get(1))) + 1;
        if (any(order_3d != dim3_t{1, 2, 3})) {
            const dim4_t order{0, order_3d[0], order_3d[1], order_3d[2]};
            input_strides = indexing::reorder(input_strides, order);
            output_strides = indexing::reorder(output_strides, order);
            shape = indexing::reorder(shape, order);
        }

        const auto uint_shape = safe_cast<uint3_t>(dim3_t(shape.get(1)));
        const auto uint_input_strides = safe_cast<uint4_t>(input_strides);
        const auto uint_output_strides = safe_cast<uint4_t>(output_strides);
        const uint32_t blocks_x = math::divideUp(uint_shape[2], BLOCK_SIZE.x);
        const uint32_t blocks_y = math::divideUp(uint_shape[1], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, uint_shape[0], shape[0]);
        const LaunchConfig config{blocks, BLOCK_SIZE};

        const Accessor<const T, 4, uint32_t> input_accessor(input.get(), uint_input_strides);
        const Accessor<T, 4, uint32_t> output_accessor(output.get(), uint_output_strides);

        NOA_ASSERT(border_mode != BORDER_REFLECT || all(window_size / 2 + 1 <= uint_shape));
        if (border_mode != BORDER_REFLECT && border_mode != BORDER_ZERO) {
            NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                      BORDER_REFLECT, BORDER_ZERO, border_mode);
        }

        switch (window_size) {
            case 3:
                stream.enqueue(
                        "signal::medfilt3",
                        border_mode == BORDER_REFLECT ? medfilt3_<T, BORDER_REFLECT, 3> : medfilt3_<T, BORDER_ZERO, 3>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            case 5:
                stream.enqueue(
                        "signal::medfilt3",
                        border_mode == BORDER_REFLECT ? medfilt3_<T, BORDER_REFLECT, 5> : medfilt3_<T, BORDER_ZERO, 5>,
                        config, input_accessor, output_accessor, uint_shape, blocks_x);
                break;
            default:
                NOA_THROW("Unsupported window size. It should be an odd number from 1 to 11, got {}", window_size);
        }
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_MEDFILT_(T)                                                                                             \
    template void median1<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, BorderMode, dim_t, Stream&); \
    template void median2<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, BorderMode, dim_t, Stream&); \
    template void median3<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, BorderMode, dim_t, Stream&)

    NOA_INSTANTIATE_MEDFILT_(half_t);
    NOA_INSTANTIATE_MEDFILT_(float);
    NOA_INSTANTIATE_MEDFILT_(double);
    NOA_INSTANTIATE_MEDFILT_(int32_t);
    NOA_INSTANTIATE_MEDFILT_(int64_t);
    NOA_INSTANTIATE_MEDFILT_(uint32_t);
    NOA_INSTANTIATE_MEDFILT_(uint64_t);
}
