#include <cub/device/device_scan.cuh>

#include "noa/common/Profiler.h"
#include "noa/common/Math.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Index.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"

#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/util/Block.cuh"
#include "noa/gpu/cuda/util/EwiseUnary.cuh"
#include "noa/gpu/cuda/util/EwiseBinary.cuh"

namespace {
    using namespace ::noa;

    constexpr uint ELEMENTS_PER_THREAD = 4;
    constexpr uint BLOCK_SIZE = 128;
    constexpr uint BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;

    constexpr dim3 ELEMENTS_PER_THREAD_2D(1, ELEMENTS_PER_THREAD);
    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD_2D.x,
                                      BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD_2D.y);

    template<typename T, typename E, typename I>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void extract1DContiguous_(const T* input, uint elements,
                              const uint* __restrict__ map, const uint* __restrict__ map_scan,
                              E* __restrict__ sequence_elements, I* __restrict__ sequence_indexes) {
        const uint base = BLOCK_WORK_SIZE * blockIdx.x + threadIdx.x;
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const uint gid = base + BLOCK_SIZE * i;
            if (gid < elements) {
                if (map[gid]) {
                    const uint pos_sequence = map_scan[gid] - 1; // inclusive sum will start at 1
                    if constexpr (!std::is_same_v<E, void>) {
                        sequence_elements[pos_sequence] = static_cast<E>(input[gid]);
                    } else {
                        (void) input;
                        (void) sequence_elements;
                    }
                    if constexpr (!std::is_same_v<I, void>)
                        sequence_indexes[pos_sequence] = static_cast<I>(gid);
                    else
                        (void) sequence_indexes;
                }
            }
        }
    }

    template<typename T, typename E, typename I>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void extractStride4D_(const T* input, uint4_t stride, uint4_t shape,
                          const uint* __restrict__ map, const uint* __restrict__ map_scan,
                          E* __restrict__ sequence_elements, I* __restrict__ sequence_indexes, uint blocks_x) {
        const uint4_t contiguous_stride = shape.stride();
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        map += at(gid[0], gid[1], contiguous_stride);
        map_scan += at(gid[0], gid[1], contiguous_stride);

        #pragma unroll
        for (int k = 0; k < ELEMENTS_PER_THREAD_2D.y; ++k) {
            #pragma unroll
            for (int l = 0; l < ELEMENTS_PER_THREAD_2D.x; ++l) {
                const uint ik = gid[2] + BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + BLOCK_SIZE_2D.x * l;
                if (ik < shape[2] && il < shape[3]) {
                    const uint offset = ik * contiguous_stride[2] + il * contiguous_stride[3];
                    if (map[offset]) {
                        const uint pos_sequence = map_scan[offset] - 1; // inclusive sum will start at 1
                        const uint pos_input = at(gid[0], gid[1], ik, il, stride);
                        if constexpr (!std::is_same_v<E, void>) {
                            sequence_elements[pos_sequence] = static_cast<E>(input[pos_input]);
                        } else {
                            (void) input;
                            (void) sequence_elements;
                        }
                        if constexpr (!std::is_same_v<I, void>)
                            sequence_indexes[pos_sequence] = static_cast<I>(pos_input);
                        else
                            (void) sequence_indexes;
                    }
                }
            }
        }
    }

    // TODO I'm sure there's a much better way of doing this but that's the naive approach I came up with.
    //      I checked PyTorch, but couldn't figure out how they know how many elements/indexes need to be extracted.
    //      One easy optimization is to merge in the same kernel the inclusive scan and the unary_op transformation
    //      using cub::BlockScan... Benchmark to compare with the CPU backend, because maybe transferring back and
    //      forth to the host is faster (it will use less memory that's for sure).
    template<typename E, typename I, typename T>
    std::tuple<E*, I*, size_t> extract_(const T* input, size4_t stride, size4_t shape, size_t elements,
                                        const uint* map, cuda::Stream& stream) {
        using namespace ::noa::cuda::memory;

        // Inclusive scan sum:
        size_t cub_tmp_bytes{};
        PtrDevice<uint> map_scan(elements, stream);
        NOA_THROW_IF(cub::DeviceScan::InclusiveSum(nullptr, cub_tmp_bytes, map,
                                                   map_scan.get(), elements, stream.id()));
        PtrDevice<std::byte> cub_tmp(cub_tmp_bytes, stream);
        NOA_THROW_IF(cub::DeviceScan::InclusiveSum(cub_tmp.get(), cub_tmp_bytes, map,
                                                   map_scan.get(), elements, stream.id()));
        cub_tmp.dispose();

        // Prepare the sequence(s):
        uint elements_to_extract{};
        copy(map_scan.get() + elements - 1, &elements_to_extract, 1, stream);
        stream.synchronize(); // we cannot use elements_to_extract before that point
        if (!elements_to_extract)
            return {nullptr, nullptr, 0};

        PtrDevice<E> sequence_elements;
        PtrDevice<I> sequence_indexes;
        if constexpr (!std::is_same_v<E, void>)
            sequence_elements.reset(elements_to_extract, stream);
        if constexpr (!std::is_same_v<I, void>)
            sequence_indexes.reset(elements_to_extract, stream);

        // Extract from map(_scan) to sequence(s):
        if (all(isContiguous(stride, shape))) {
            const uint blocks = noa::math::divideUp(static_cast<uint>(elements), BLOCK_WORK_SIZE);
            const cuda::LaunchConfig config{blocks, BLOCK_SIZE};
            stream.enqueue("memory::extract", extract1DContiguous_<T, E, I>, config,
                           input, elements, map, map_scan.get(), sequence_elements.get(), sequence_indexes.get());
        } else {
            const uint4_t i_shape{shape.get()};
            const uint blocks_x = noa::math::divideUp(i_shape[3], BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[2], BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, i_shape[1], i_shape[0]);
            const cuda::LaunchConfig config{blocks, BLOCK_SIZE_2D};
            stream.enqueue("memory::extract", extractStride4D_<T, E, I>, config,
                           input, uint4_t{stride}, i_shape, map, map_scan.get(),
                           sequence_elements.get(), sequence_indexes.get(), blocks_x);
        }

        // The sequences are attached to the input stream. They should be accessed and freed using that same stream.
        return {sequence_elements.release(), sequence_indexes.release(), static_cast<size_t>(elements_to_extract)};
    }

    template<typename E, typename I, typename T, int VEC_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void insert_(const E* sequence_values, const I* sequence_indexes,
                 size_t sequence_size, T* output) {
        const uint base = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < sequence_size) {
                    const I index = sequence_indexes[gid];
                    output[index] = static_cast<T>(sequence_values[gid]);
                }
            }
        } else {
            const uint remaining = sequence_size - base;
            sequence_values += base;
            sequence_indexes += base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint cid = BLOCK_SIZE * i + threadIdx.x;
                    if (cid < remaining) {
                        const I index = sequence_indexes[cid];
                        output[index] = static_cast<T>(sequence_values[cid]);
                    }
                }
            } else {
                E values[ELEMENTS_PER_THREAD];
                I indexes[ELEMENTS_PER_THREAD];
                using namespace noa::cuda::util::block;
                vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(sequence_values, values, threadIdx.x);
                vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(sequence_indexes, indexes, threadIdx.x);
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i)
                    output[indexes[i]] = static_cast<T>(values[i]);
            }
        }
    }
}

namespace noa::cuda::memory {
    template<typename E, typename I, typename T, typename UnaryOp>
    std::tuple<E*, I*, size_t> extract(const T* input, size4_t stride, size4_t shape,
                                       UnaryOp unary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if constexpr (std::is_same_v<E, void> && std::is_same_v<I, void>)
            return {nullptr, nullptr, 0};

        const size4_t contiguous_stride = shape.stride();
        const size_t elements = shape.elements();
        PtrDevice<uint> map(elements, stream);
        util::ewise::unary<true>("memory::extract", input, stride, map.get(), contiguous_stride,
                                 shape, stream, unary_op);
        return extract_<E, I>(input, stride, shape, elements, map.get(), stream);
    }

    #define INSTANTIATE_EXTRACT_UNARY_BASE_(T, E, I)  \
    template std::tuple<E*, I*, size_t> extract<E,I,T,::noa::math::logical_not_t>(const T*, size4_t, size4_t, ::noa::math::logical_not_t, Stream&); \

    #define INSTANTIATE_EXTRACT_UNARY_(T, E)            \
    INSTANTIATE_EXTRACT_UNARY_BASE_(T, E, uint32_t);    \
    INSTANTIATE_EXTRACT_UNARY_BASE_(T, E, uint64_t)

    INSTANTIATE_EXTRACT_UNARY_(int32_t, void);
    INSTANTIATE_EXTRACT_UNARY_(int64_t, void);
    INSTANTIATE_EXTRACT_UNARY_(uint32_t, void);
    INSTANTIATE_EXTRACT_UNARY_(uint64_t, void);
    INSTANTIATE_EXTRACT_UNARY_(half_t, void);
    INSTANTIATE_EXTRACT_UNARY_(float, void);
    INSTANTIATE_EXTRACT_UNARY_(double, void);

    INSTANTIATE_EXTRACT_UNARY_(int32_t, int32_t);
    INSTANTIATE_EXTRACT_UNARY_(int64_t, int64_t);
    INSTANTIATE_EXTRACT_UNARY_(uint32_t, uint32_t);
    INSTANTIATE_EXTRACT_UNARY_(uint64_t, uint64_t);
    INSTANTIATE_EXTRACT_UNARY_(half_t, half_t);
    INSTANTIATE_EXTRACT_UNARY_(float, float);
    INSTANTIATE_EXTRACT_UNARY_(double, double);

    template<typename E, typename I, typename T, typename U, typename BinaryOp, typename>
    std::tuple<E*, I*, size_t> extract(const T* input, size4_t stride, size4_t shape, U value,
                                       BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if constexpr (std::is_same_v<E, void> && std::is_same_v<I, void>)
            return {nullptr, nullptr, 0};

        const size4_t contiguous_stride = shape.stride();
        const size_t elements = shape.elements();
        PtrDevice<uint> map(elements, stream);
        util::ewise::binary<true>("memory::extract", input, stride, value,
                                  map.get(), contiguous_stride, shape, stream, binary_op);
        return extract_<E, I>(input, stride, shape, elements, map.get(), stream);
    }

    template<typename E, typename I, typename T, typename U, typename BinaryOp>
    std::tuple<E*, I*, size_t> extract(const T* input, size4_t stride, size4_t shape, const U* values,
                                       BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if constexpr (std::is_same_v<E, void> && std::is_same_v<I, void>)
            return {nullptr, nullptr, 0};

        memory::PtrDevice<U> buffer;
        values = util::ensureDeviceAccess(values, stream, buffer, shape[0]);

        const size4_t contiguous_stride = shape.stride();
        const size_t elements = shape.elements();
        PtrDevice<uint> map(elements, stream);
        util::ewise::binary<true>("memory::extract", input, stride, values,
                                  map.get(), contiguous_stride, shape, stream, binary_op);
        return extract_<E, I>(input, stride, shape, elements, map.get(), stream);
    }

    template<typename E, typename I, typename T, typename U, typename BinaryOp>
    std::tuple<E*, I*, size_t> extract(const T* input, size4_t input_stride,
                                       const U* array, size4_t array_stride,
                                       size4_t shape, BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if constexpr (std::is_same_v<E, void> && std::is_same_v<I, void>)
            return {nullptr, nullptr, 0};

        const size4_t contiguous_stride = shape.stride();
        const size_t elements = shape.elements();
        PtrDevice<uint> map(elements, stream);
        util::ewise::binary<true>("memory::extract", input, input_stride, array, array_stride,
                                  map.get(), contiguous_stride, shape, stream, binary_op);
        return extract_<E, I>(input, input_stride, shape, elements, map.get(), stream);
    }

    #define INSTANTIATE_EXTRACT_BINARY_BASE0_(T, E, I, BINARY)                                                          \
    template std::tuple<E*, I*, size_t> extract<E,I,T,T,BINARY,void>(const T*, size4_t, size4_t, T, BINARY, Stream&);   \
    template std::tuple<E*, I*, size_t> extract<E,I,T,T,BINARY>(const T*, size4_t, size4_t, const T*, BINARY, Stream&); \
    template std::tuple<E*, I*, size_t> extract<E,I,T,T,BINARY>(const T*, size4_t, const T*, size4_t, size4_t, BINARY, Stream&)

    #define INSTANTIATE_EXTRACT_BINARY_BASE2_(T, E, I)                  \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,E,I,::noa::math::equal_t);      \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,E,I,::noa::math::not_equal_t);  \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,E,I,::noa::math::less_t);       \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,E,I,::noa::math::less_equal_t); \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,E,I,::noa::math::greater_t);    \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,E,I,::noa::math::greater_equal_t)

    #define INSTANTIATE_EXTRACT_BINARY_(T, E)           \
    INSTANTIATE_EXTRACT_BINARY_BASE2_(T, E, uint32_t);  \
    INSTANTIATE_EXTRACT_BINARY_BASE2_(T, E, uint64_t)

    INSTANTIATE_EXTRACT_BINARY_(int32_t, void);
    INSTANTIATE_EXTRACT_BINARY_(int64_t, void);
    INSTANTIATE_EXTRACT_BINARY_(uint32_t, void);
    INSTANTIATE_EXTRACT_BINARY_(uint64_t, void);
    INSTANTIATE_EXTRACT_BINARY_(half_t, void);
    INSTANTIATE_EXTRACT_BINARY_(float, void);
    INSTANTIATE_EXTRACT_BINARY_(double, void);

    INSTANTIATE_EXTRACT_BINARY_(int32_t, int32_t);
    INSTANTIATE_EXTRACT_BINARY_(int64_t, int64_t);
    INSTANTIATE_EXTRACT_BINARY_(uint32_t, uint32_t);
    INSTANTIATE_EXTRACT_BINARY_(uint64_t, uint64_t);
    INSTANTIATE_EXTRACT_BINARY_(half_t, half_t);
    INSTANTIATE_EXTRACT_BINARY_(float, float);
    INSTANTIATE_EXTRACT_BINARY_(double, double);

    template<typename E, typename I, typename T>
    void insert(const E* sequence_values, const I* sequence_indexes, size_t sequence_size,
                T* output, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const uint blocks = noa::math::divideUp(static_cast<uint>(sequence_size), BLOCK_WORK_SIZE);
        const int vec_size = std::min(util::maxVectorCount(sequence_values), util::maxVectorCount(sequence_indexes));
        if (vec_size == 4) {
            return stream.enqueue("memory::insert", insert_<E, I, T, 4>, {blocks, BLOCK_SIZE},
                                  sequence_values, sequence_indexes, sequence_size, output);
        } else if (vec_size == 2) {
            return stream.enqueue("memory::insert", insert_<E, I, T, 2>, {blocks, BLOCK_SIZE},
                                  sequence_values, sequence_indexes, sequence_size, output);
        } else {
            return stream.enqueue("memory::insert", insert_<E, I, T, 1>, {blocks, BLOCK_SIZE},
                                  sequence_values, sequence_indexes, sequence_size, output);
        }
    }

    #define INSTANTIATE_INSERT_BASE_(T, I) \
    template void insert<T,I,T>(const T*, const I*, size_t, T*, Stream&)

    #define INSTANTIATE_INSERT_(T)          \
    INSTANTIATE_INSERT_BASE_(T, uint32_t);  \
    INSTANTIATE_INSERT_BASE_(T, uint64_t)

    INSTANTIATE_INSERT_(int32_t);
    INSTANTIATE_INSERT_(int64_t);
    INSTANTIATE_INSERT_(uint32_t);
    INSTANTIATE_INSERT_(uint64_t);
    INSTANTIATE_INSERT_(half_t);
    INSTANTIATE_INSERT_(float);
    INSTANTIATE_INSERT_(double);
}
