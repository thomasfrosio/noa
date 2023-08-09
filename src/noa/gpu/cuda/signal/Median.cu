#include "noa/core/Math.hpp"

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/signal/Median.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"

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
    __forceinline__ __device__ void sort_swap_(T& a, T& b) {
        if (b < a) {
            // TODO Use XOR swap?
            T tmp = a;
            a = b;
            b = tmp;
        }
    }

    // Sets the array v, so that the min is at v[0] and max is at v[length-1].
    // TODO make sure the force inline still allows constant indexing, otherwise replace with a macro.
    template<typename T>
    __forceinline__ __device__ void order_(T* v, i32 length) {
        for (i32 i = 0; i < length / 2; i++) // ensure min in first half, max in second half
            sort_swap_(v[i], v[length - 1 - i]);
        for (i32 i = 1; i < (length + 1) / 2; i++) // move min in first half to first pos
            sort_swap_(v[0], v[i]);
        for (i32 i = length - 2; i >= length / 2; i--) // move max in second half to last pos
            sort_swap_(v[i], v[length - 1]);
    }

    // Loads the shared memory array according to the BORDER_MODE.
    // input_row:   device memory. Should point to the beginning of the current row.
    // stride:      innermost stride of the input.
    // s_mem:       shared memory. Should point to the current element.
    // shape_x:     number of logical elements in x, x being the dimension of the 1D window.
    // gx:          index of the current element in x. If out of bound, add padding according to the BORDER_MODE.
    template<typename T, BorderMode BORDER_MODE, i32 HALO>
    __device__ void load_to_shared_1d_(
            AccessorReference<const T, 1, u32> input_row,
            T* s_mem, i32 shape_x, i32 gx
    ) {
        if constexpr (BORDER_MODE == BorderMode::REFLECT) {
            if (gx < 0)
                *s_mem = input_row[-gx]; // pad left; requires shape_x >= HALO + 1, since gx >= -HALO
            else if (gx < shape_x)
                *s_mem = input_row[gx]; // in array, no padding
            else if (gx < shape_x + HALO)
                *s_mem = input_row[(2 * (shape_x - 1) - gx)]; // pad right; requires shape_x >= HALO  + 1
            // Otherwise, don't do anything since the *s_mem element will not be used anyway.

        } else if constexpr (BORDER_MODE == BorderMode::ZERO) {
            if (gx < 0 || gx >= shape_x)
                *s_mem = T{0};
            else
                *s_mem = input_row[gx];
        } else {
            static_assert(nt::always_false_v<T>);
        }
    }

    template<typename T, BorderMode BORDER_MODE, i32 WINDOW_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void median_filter_1d_(
            Accessor<const T, 4, u32> input,
            Accessor<T, 4, u32> output,
            Shape2<i32> shape, u32 blocks_x
    ) {
        static_assert(WINDOW_SIZE % 2); // only support odd windows.
        constexpr i32 PADDING = WINDOW_SIZE - 1; // assume odd
        constexpr i32 HALO = PADDING / 2;

        // The shared memory is the shape of the block + the padding in the first dimension.
        using uninit_t = noa::cuda::utils::uninitialized_type_t<T>;
        constexpr auto SHARED_SHAPE = Shape2<i32>{BLOCK_SIZE.y, BLOCK_SIZE.x + PADDING};
        constexpr auto SHARED_SIZE = SHARED_SHAPE.elements();
        __shared__ uninit_t buffer[SHARED_SIZE];
        T* shared_mem = reinterpret_cast<T*>(buffer);

        const auto index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>{threadIdx.y, threadIdx.x};
        const auto gid = Vec4<i32>{blockIdx.z,
                                   blockIdx.y,
                                   BLOCK_SIZE.y * index[0] + tid[0],
                                   BLOCK_SIZE.x * index[1] + tid[1]};

        const auto input_1d = input[gid[0]][gid[1]][gid[2]];

        // There's no padding in y, so if out of bounds, stop.
        if (gid[2] < shape[0]) {
            // Load shared memory. Loop to take into account padding.
            for (i32 lx = tid[1], gx = gid[3]; lx < SHARED_SHAPE[1]; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x) {
                load_to_shared_1d_<T, BORDER_MODE, HALO>(
                        input_1d, shared_mem + tid[0] * SHARED_SHAPE[1] + lx, shape[1], gx - HALO);
            }
            noa::cuda::utils::block_synchronize();

            // Only continue if not out of bound.
            if (gid[3] < shape[1]) {
                // The goal is to reduce register pressure as much as possible, but still use registers
                // to do the exchange sort. The window is divided into two half: the first "activated" half,
                // which is where the sorting happens, and the second half, which is the pool of contestants
                // that will be progressively added.
                constexpr i32 ACTIVE = WINDOW_SIZE / 2 + 2; // the final size should be at least 3.

                // Load active half to, hopefully, the local registers (otherwise spill to device memory).
                T v[ACTIVE]; // all indexing are known at compile, so registers should be used
                for (i32 x = 0; x < ACTIVE; ++x)
                    v[x] = shared_mem[tid[0] * SHARED_SHAPE[1] + tid[1] + x];

                order_(v, ACTIVE); // ensure min at 0, max at ACTIVE - 1

                // We can replace the min by a new contestant and then reorder.
                // We also don't have to compare the max at ACTIVE - 1, we just need to stack the new max at the end.
                // As such, as we add new contestants, we right-truncate the active half to ignore the previous max.
                i32 length = ACTIVE;
                for (i32 k = ACTIVE; k < WINDOW_SIZE; ++k) {
                    v[0] = shared_mem[tid[0] * SHARED_SHAPE[1] + tid[1] + k]; // replace min by new contestant
                    --length; // ignore the previous max at the end
                    order_(v, length); // min at 0, max at length - 1
                }

                // Since we know the min and max are at v[0] and v[length-1] respectively, to find the median,
                // we simply need to (exchange) sort the elements from v[1] to v[length-2] included.
                // The median will then be at length/2.
                for (i32 k = 1; k < length - 1; k++)
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
    template<typename T, BorderMode BORDER_MODE, i32 HALO>
    __device__ void load_to_shared_2d_(
            AccessorReference<const T, 2, u32> input_slice, T* s_mem,
            i32 shape_y, i32 gy, i32 shape_x, i32 gx
    ) {
        if constexpr (BORDER_MODE == BorderMode::REFLECT) {
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

        } else if constexpr (BORDER_MODE == BorderMode::ZERO) {
            if (gx < 0 || gx >= shape_x || gy < 0 || gy >= shape_y)
                *s_mem = T{0};
            else
                *s_mem = input_slice(gy, gx);
        } else {
            static_assert(nt::always_false_v<T>);
        }
    }

    template<typename T, BorderMode BORDER_MODE, i32 WINDOW_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void median_filter_2d_(
            Accessor<const T, 4, u32> input,
            Accessor<T, 4, u32> output,
            Shape2<i32> shape, u32 blocks_x
    ) {
        static_assert(WINDOW_SIZE % 2); // only support odd windows.
        constexpr i32 TILE_SIZE = WINDOW_SIZE * WINDOW_SIZE;
        constexpr i32 PADDING = WINDOW_SIZE - 1; // assume odd
        constexpr i32 HALO = PADDING / 2;

        // The shared memory is the shape of the block + the padding the first and second dimension.
        using uninit_t = noa::cuda::utils::uninitialized_type_t<T>;
        constexpr auto SHARED_SHAPE = Shape2<u32>{BLOCK_SIZE.y + PADDING, BLOCK_SIZE.x + PADDING};
        constexpr auto SHARED_SIZE = SHARED_SHAPE.elements();
        __shared__ uninit_t buffer[SHARED_SIZE];
        T* shared_mem = reinterpret_cast<T*>(buffer);

        const auto index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>{threadIdx.y, threadIdx.x};
        const auto gid = Vec4<i32>{blockIdx.z,
                                   blockIdx.y,
                                   BLOCK_SIZE.y * index[0] + tid[0],
                                   BLOCK_SIZE.x * index[1] + tid[1]};

        const auto input_2d = input[gid[0]][gid[1]];

        // Load shared memory. Loop to account for the halo.
        for (i32 ly = tid[0], gy = gid[2]; ly < SHARED_SHAPE[0]; ly += BLOCK_SIZE.y, gy += BLOCK_SIZE.y)
            for (i32 lx = tid[1], gx = gid[3]; lx < SHARED_SHAPE[1]; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x)
                load_to_shared_2d_<T, BORDER_MODE, HALO>(
                        input_2d, shared_mem + ly * SHARED_SHAPE[1] + lx,
                        shape[0], gy - HALO, shape[1], gx - HALO);
        noa::cuda::utils::block_synchronize();

        // Only continue if not out of bound. gid.z cannot be out of bound.
        if (gid[2] < shape[0] && gid[3] < shape[1]) {
            constexpr i32 ACTIVE = TILE_SIZE / 2 + 2;

            // Load active window from shared memory into this 1D array.
            T v[ACTIVE];
            for (i32 count = 0, y = 0; y < WINDOW_SIZE; ++y)
                for (i32 x = 0; count < ACTIVE && x < WINDOW_SIZE; ++x, ++count)
                    v[count] = shared_mem[(tid[0] + y) * SHARED_SHAPE[1] + tid[1] + x];

            order_(v, ACTIVE);
            i32 length = ACTIVE;
            for (i32 k = ACTIVE; k < TILE_SIZE; ++k) {
                // k is the index, but we want the corresponding (x,y) coordinates in the 2D window.
                // Then offset these coordinates to the current thread.
                const i32 coord_y = k / WINDOW_SIZE;
                const i32 coord_x = k - coord_y * WINDOW_SIZE;
                v[0] = shared_mem[(coord_y + tid[0]) * SHARED_SHAPE[1] + coord_x + tid[1]];

                --length;
                order_(v, length);
            }

            // Sort the final elements.
            for (i32 k = 1; k < length - 1; k++)
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
    template<typename T, BorderMode BORDER_MODE, i32 HALO>
    __device__ void load_to_shared_3d(
            AccessorReference<const T, 3, u32> input, T* s_mem,
            i32 shape_z, i32 gz, i32 shape_y, i32 gy, i32 shape_x, i32 gx
    ) {
        if constexpr (BORDER_MODE == BorderMode::REFLECT) {
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

        } else if constexpr (BORDER_MODE == BorderMode::ZERO) {
            if (gx < 0 || gx >= shape_x ||
                gy < 0 || gy >= shape_y ||
                gz < 0 || gz >= shape_z)
                *s_mem = T{0};
            else
                *s_mem = input(gz, gy, gx);
        } else {
            static_assert(nt::always_false_v<T>);
        }
    }

    // The launch config and block size is like median_filter_1d_.
    template<typename T, BorderMode BORDER_MODE, u32 WINDOW_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void median_filter_3d_(
            Accessor<const T, 4, u32> input,
            Accessor<T, 4, u32> output,
            Shape3<i32> shape, u32 blocks_x
    ) {
        static_assert(WINDOW_SIZE % 2); // only support odd windows.
        constexpr i32 TILE_SIZE = WINDOW_SIZE * WINDOW_SIZE * WINDOW_SIZE;
        constexpr i32 PADDING = WINDOW_SIZE - 1; // assume odd
        constexpr i32 HALO = PADDING / 2;

        // The shared memory is the shape of the block + the padding all 3 dimensions.
        // For the largest supported window (5), this goes up to 20*20*5=2205 elements.
        using uninit_t = noa::cuda::utils::uninitialized_type_t<T>;
        constexpr auto SHARED_SHAPE = Shape3<i32>{WINDOW_SIZE, BLOCK_SIZE.y + PADDING, BLOCK_SIZE.x + PADDING};
        constexpr auto SHARED_SIZE = SHARED_SHAPE.elements();
        __shared__ uninit_t buffer[SHARED_SIZE];
        T* shared_mem = reinterpret_cast<T*>(buffer);

        const auto index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>{threadIdx.y, threadIdx.x};
        const auto gid = Vec4<i32>{blockIdx.z,
                                   blockIdx.y,
                                   BLOCK_SIZE.y * index[0] + tid[0],
                                   BLOCK_SIZE.x * index[1] + tid[1]};

        const auto input_3d = input[gid[0]];

        // Load shared memory.
        // Each thread processes at least WINDOW_SIZE elements (the z dimension).
        for (i32 lz = 0, gz = gid[1]; lz < SHARED_SHAPE[0]; ++lz, ++gz)
            for (i32 ly = tid[0], gy = gid[2]; ly < SHARED_SHAPE[1]; ly += BLOCK_SIZE.y, gy += BLOCK_SIZE.y)
                for (i32 lx = tid[1], gx = gid[3]; lx < SHARED_SHAPE[2]; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x)
                    load_to_shared_3d<T, BORDER_MODE, HALO>(
                            input_3d,
                            shared_mem + (lz * SHARED_SHAPE[1] + ly) * SHARED_SHAPE[2] + lx,
                            shape[0], gz - HALO, shape[1], gy - HALO, shape[2], gx - HALO);
        noa::cuda::utils::block_synchronize();

        // Only continue if not out of bound. gid.z cannot be out of bound.
        if (gid[2] < shape[1] && gid[3] < shape[2]) {
            constexpr i32 ACTIVE = TILE_SIZE / 2 + 2;

            // Load active window from shared memory into this 1D array.
            T v[ACTIVE];
            for (i32 count = 0, z = 0; z < WINDOW_SIZE; ++z)
                for (i32 y = 0; count < ACTIVE && y < WINDOW_SIZE; ++y)
                    for (i32 x = 0; count < ACTIVE && x < WINDOW_SIZE; ++x, ++count)
                        v[count] = shared_mem[(z * SHARED_SHAPE[1] + tid[0] + y) * SHARED_SHAPE[2] + tid[1] + x];

            order_(v, ACTIVE);
            i32 length = ACTIVE;
            for (i32 k = ACTIVE; k < TILE_SIZE; ++k) {
                // k is the index, but we want the corresponding (x,y,z) coordinates in the 3D window.
                // Then offset these coordinates to the current thread.
                const i32 coord_z = k / (WINDOW_SIZE * WINDOW_SIZE);
                const i32 tmp = k - coord_z * WINDOW_SIZE * WINDOW_SIZE;
                const i32 coord_y = tmp / WINDOW_SIZE;
                const i32 coord_x = tmp - coord_y * WINDOW_SIZE;
                v[0] = shared_mem[(coord_z * SHARED_SHAPE[1] + coord_y + tid[0]) * SHARED_SHAPE[2] + coord_x + tid[1]];
                --length;
                order_(v, length);
            }

            // Sort the final elements.
            for (i32 k = 1; k < length - 1; k++)
                order_(v + k, length - k);
            output(gid) = v[length / 2];
        }
    }
}

namespace noa::cuda::signal {
    template<typename T, typename>
    void median_filter_1d(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, BorderMode border_mode, i64 window_size, Stream& stream
    ) {
        NOA_ASSERT(input != output && noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        if (window_size <= 1)
            return noa::cuda::memory::copy(input, input_strides, output, output_strides, shape, stream);

        const auto shape_2d = shape.filter(2, 3).as_safe<i32>();
        const u32 blocks_x = noa::math::divide_up(static_cast<u32>(shape_2d[1]), BLOCK_SIZE.x);
        const u32 blocks_y = noa::math::divide_up(static_cast<u32>(shape_2d[0]), BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const auto config = LaunchConfig{blocks, BLOCK_SIZE};

        const auto input_accessor = Accessor<const T, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = Accessor<T, 4, u32>(output, output_strides.as_safe<u32>());

        NOA_ASSERT(border_mode != BorderMode::REFLECT || window_size / 2 + 1 <= shape[3]);
        if (border_mode != BorderMode::REFLECT && border_mode != BorderMode::ZERO) {
            NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                      BorderMode::REFLECT, BorderMode::ZERO, border_mode);
        }

        switch (window_size) {
            case 3:
                return stream.enqueue(
                        "median_filter_1d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_1d_<T, BorderMode::REFLECT, 3> :
                        median_filter_1d_<T, BorderMode::ZERO, 3>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            case 5:
                return stream.enqueue(
                        "median_filter_1d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_1d_<T, BorderMode::REFLECT, 5> :
                        median_filter_1d_<T, BorderMode::ZERO, 5>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            case 7:
                return stream.enqueue(
                        "median_filter_1d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_1d_<T, BorderMode::REFLECT, 7> :
                        median_filter_1d_<T, BorderMode::ZERO, 7>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            case 9:
                return stream.enqueue(
                        "median_filter_1d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_1d_<T, BorderMode::REFLECT, 9> :
                        median_filter_1d_<T, BorderMode::ZERO, 9>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            case 11:
                return stream.enqueue(
                        "median_filter_1d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_1d_<T, BorderMode::REFLECT, 11> :
                        median_filter_1d_<T, BorderMode::ZERO, 11>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            case 13:
                return stream.enqueue(
                        "median_filter_1d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_1d_<T, BorderMode::REFLECT, 13> :
                        median_filter_1d_<T, BorderMode::ZERO, 13>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            case 15:
                return stream.enqueue(
                        "median_filter_1d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_1d_<T, BorderMode::REFLECT, 15> :
                        median_filter_1d_<T, BorderMode::ZERO, 15>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            case 17:
                return stream.enqueue(
                        "median_filter_1d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_1d_<T, BorderMode::REFLECT, 17> :
                        median_filter_1d_<T, BorderMode::ZERO, 17>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            case 19:
                return stream.enqueue(
                        "median_filter_1d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_1d_<T, BorderMode::REFLECT, 19> :
                        median_filter_1d_<T, BorderMode::ZERO, 19>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            case 21:
                return stream.enqueue(
                        "median_filter_1d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_1d_<T, BorderMode::REFLECT, 21> :
                        median_filter_1d_<T, BorderMode::ZERO, 21>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            default:
                NOA_THROW("Unsupported window size. It should be an odd number from 1 to 21, got {}", window_size);
        }
    }

    template<typename T, typename>
    void median_filter_2d(
            const T* input, Strides4<i64> input_strides,
            T* output, Strides4<i64> output_strides,
            Shape4<i64> shape, BorderMode border_mode, i64 window_size, Stream& stream
    ) {
        NOA_ASSERT(input != output && noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        if (window_size <= 1)
            return noa::cuda::memory::copy(input, input_strides, output, output_strides, shape, stream);

        const auto order_2d = noa::indexing::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (noa::any(order_2d != Vec2<i64>{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
        }

        const auto shape_2d = shape.filter(2, 3).as_safe<i32>();
        const u32 blocks_x = noa::math::divide_up(static_cast<u32>(shape_2d[1]), BLOCK_SIZE.x);
        const u32 blocks_y = noa::math::divide_up(static_cast<u32>(shape_2d[0]), BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const auto config = LaunchConfig{blocks, BLOCK_SIZE};

        const auto input_accessor = Accessor<const T, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = Accessor<T, 4, u32>(output, output_strides.as_safe<u32>());

        NOA_ASSERT(border_mode != BorderMode::REFLECT || noa::all(window_size / 2 + 1 <= shape_2d));
        if (border_mode != BorderMode::REFLECT && border_mode != BorderMode::ZERO) {
            NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                      BorderMode::REFLECT, BorderMode::ZERO, border_mode);
        }

        switch (window_size) {
            case 3:
                return stream.enqueue(
                        "median_filter_2d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_2d_<T, BorderMode::REFLECT, 3> :
                        median_filter_2d_<T, BorderMode::ZERO, 3>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            case 5:
                return stream.enqueue(
                        "median_filter_2d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_2d_<T, BorderMode::REFLECT, 5> :
                        median_filter_2d_<T, BorderMode::ZERO, 5>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            case 7:
                return stream.enqueue(
                        "median_filter_2d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_2d_<T, BorderMode::REFLECT, 7> :
                        median_filter_2d_<T, BorderMode::ZERO, 7>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            case 9:
                return stream.enqueue(
                        "median_filter_2d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_2d_<T, BorderMode::REFLECT, 9> :
                        median_filter_2d_<T, BorderMode::ZERO, 9>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            case 11:
                return stream.enqueue(
                        "median_filter_2d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_2d_<T, BorderMode::REFLECT, 11> :
                        median_filter_2d_<T, BorderMode::ZERO, 11>,
                        config, input_accessor, output_accessor, shape_2d, blocks_x);
            default:
                NOA_THROW("Unsupported window size. It should be an odd number from 1 to 11, got {}", window_size);
        }
    }

    template<typename T, typename>
    void median_filter_3d(
            const T* input, Strides4<i64> input_strides,
            T* output, Strides4<i64> output_strides,
            Shape4<i64> shape, BorderMode border_mode, i64 window_size, Stream& stream
    ) {
        NOA_ASSERT(input != output && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        if (window_size <= 1)
            return memory::copy(input, input_strides, output, output_strides, shape, stream);

        const auto order_3d = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
        if (noa::any(order_3d != Vec3<i64>{0, 1, 2})) {
            const auto order = (order_3d + 1).push_front(0);
            input_strides = input_strides.reorder(order);
            output_strides = output_strides.reorder(order);
            shape = shape.reorder(order);
        }

        const auto shape_3d = shape.pop_front().as_safe<i32>();
        const u32 blocks_x = noa::math::divide_up(static_cast<u32>(shape_3d[2]), BLOCK_SIZE.x);
        const u32 blocks_y = noa::math::divide_up(static_cast<u32>(shape_3d[1]), BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape_3d[0], shape[0]);
        const auto config = LaunchConfig{blocks, BLOCK_SIZE};

        const auto input_accessor = Accessor<const T, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = Accessor<T, 4, u32>(output, output_strides.as_safe<u32>());

        NOA_ASSERT(border_mode != BorderMode::REFLECT || noa::all(window_size / 2 + 1 <= shape_3d));
        if (border_mode != BorderMode::REFLECT && border_mode != BorderMode::ZERO) {
            NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                      BorderMode::REFLECT, BorderMode::ZERO, border_mode);
        }

        switch (window_size) {
            case 3:
                return stream.enqueue(
                        "median_filter_3d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_3d_<T, BorderMode::REFLECT, 3> :
                        median_filter_3d_<T, BorderMode::ZERO, 3>,
                        config, input_accessor, output_accessor, shape_3d, blocks_x);
            case 5:
                return stream.enqueue(
                        "median_filter_3d",
                        border_mode == BorderMode::REFLECT ?
                        median_filter_3d_<T, BorderMode::REFLECT, 5> :
                        median_filter_3d_<T, BorderMode::ZERO, 5>,
                        config, input_accessor, output_accessor, shape_3d, blocks_x);
            default:
                NOA_THROW("Unsupported window size. It should be an odd number from 1 to 11, got {}", window_size);
        }
    }

    #define NOA_INSTANTIATE_MEDFILT_(T)                                                                                                                 \
    template void median_filter_1d<T, void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, BorderMode, i64, Stream&);    \
    template void median_filter_2d<T, void>(const T*, Strides4<i64>, T*, Strides4<i64>, Shape4<i64>, BorderMode, i64, Stream&);                         \
    template void median_filter_3d<T, void>(const T*, Strides4<i64>, T*, Strides4<i64>, Shape4<i64>, BorderMode, i64, Stream&)

    NOA_INSTANTIATE_MEDFILT_(f16);
    NOA_INSTANTIATE_MEDFILT_(f32);
    NOA_INSTANTIATE_MEDFILT_(f64);
    NOA_INSTANTIATE_MEDFILT_(i32);
    NOA_INSTANTIATE_MEDFILT_(i64);
    NOA_INSTANTIATE_MEDFILT_(u32);
    NOA_INSTANTIATE_MEDFILT_(u64);
}
