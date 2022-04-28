#include <cub/device/device_scan.cuh>

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

    template<typename T, typename I>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void extract1DContiguous_(const T* input, uint elements,
                              const uint* __restrict__ map, const uint* __restrict__ map_scan,
                              T* __restrict__ sequence_elements, I* __restrict__ sequence_indexes) {
        const uint base = BLOCK_WORK_SIZE * blockIdx.x + threadIdx.x;
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const uint gid = base + BLOCK_SIZE * i;
            if (gid < elements) {
                if (map[gid]) {
                    const uint pos_sequence = map_scan[gid] - 1; // inclusive sum will start at 1
                    if (sequence_elements)
                        sequence_elements[pos_sequence] = input[gid];
                    if (sequence_indexes)
                        sequence_indexes[pos_sequence] = static_cast<I>(gid);
                }
            }
        }
    }

    template<typename T, typename I>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void extractStride4D_(const T* input, uint4_t stride, uint4_t shape,
                          const uint* __restrict__ map, const uint* __restrict__ map_scan,
                          T* __restrict__ sequence_elements, I* __restrict__ sequence_indexes, uint blocks_x) {
        const uint4_t contiguous_stride = shape.stride();
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        map += indexing::at(gid[0], gid[1], contiguous_stride);
        map_scan += indexing::at(gid[0], gid[1], contiguous_stride);

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
                        const uint pos_input = indexing::at(gid[0], gid[1], ik, il, stride);
                        if (sequence_elements)
                            sequence_elements[pos_sequence] = input[pos_input];
                        if (sequence_indexes)
                            sequence_indexes[pos_sequence] = static_cast<I>(pos_input);
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
    template<typename T, typename I>
    cuda::memory::Extracted<T, I>
    extract_(const T* input, size4_t stride, size4_t shape, size_t elements,
             const uint* map, bool extract_values, bool extract_indexes, cuda::Stream& stream) {
        using namespace ::noa::cuda::memory;

        // Inclusive scan sum to get the number of elements to extract and
        // map_scan to know the order of the elements to extract.
        uint elements_to_extract{};
        PtrDevice<uint> map_scan(elements, stream);
        {
            size_t cub_tmp_bytes{};
            NOA_THROW_IF(cub::DeviceScan::InclusiveSum(nullptr, cub_tmp_bytes, map,
                                                       map_scan.get(), elements, stream.id()));
            PtrDevice<std::byte> cub_tmp(cub_tmp_bytes, stream);
            NOA_THROW_IF(cub::DeviceScan::InclusiveSum(cub_tmp.get(), cub_tmp_bytes, map,
                                                       map_scan.get(), elements, stream.id()));
            copy<uint>(map_scan.get() + elements - 1, &elements_to_extract, 1, stream);
            stream.synchronize(); // we cannot use elements_to_extract before that point
            if (!elements_to_extract)
                return {};
        }

        Extracted<T, I> extracted{
            extract_values ? PtrDevice<T>::alloc(elements_to_extract, stream) : nullptr,
            extract_indexes ? PtrDevice<I>::alloc(elements_to_extract, stream) : nullptr,
            elements_to_extract
        };

        // Compute the sequence(s).
        if (all(indexing::isContiguous(stride, shape))) {
            const uint blocks = noa::math::divideUp(static_cast<uint>(elements), BLOCK_WORK_SIZE);
            const cuda::LaunchConfig config{blocks, BLOCK_SIZE};
            stream.enqueue("memory::extract", extract1DContiguous_<T, I>, config,
                           input, elements, map, map_scan.get(),
                           extracted.values.get(), extracted.indexes.get());
        } else {
            const uint4_t i_shape{shape.get()};
            const uint blocks_x = noa::math::divideUp(i_shape[3], BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[2], BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, i_shape[1], i_shape[0]);
            const cuda::LaunchConfig config{blocks, BLOCK_SIZE_2D};
            stream.enqueue("memory::extract", extractStride4D_<T, I>, config,
                           input, uint4_t{stride}, i_shape, map, map_scan.get(),
                           extracted.values.get(), extracted.indexes.get(), blocks_x);
        }
        stream.attach(extracted.values, extracted.indexes);
        return extracted;
    }

    template<typename T, typename I, int VEC_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void insert_(const T* sequence_values, const I* sequence_indexes,
                 size_t sequence_size, T* output) {
        const uint base = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < sequence_size) {
                    const I index = sequence_indexes[gid];
                    output[index] = sequence_values[gid];
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
                        output[index] = sequence_values[cid];
                    }
                }
            } else {
                T values[ELEMENTS_PER_THREAD];
                I indexes[ELEMENTS_PER_THREAD];
                using namespace noa::cuda::util::block;
                vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(sequence_values, values, threadIdx.x);
                vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(sequence_indexes, indexes, threadIdx.x);
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i)
                    output[indexes[i]] = values[i];
            }
        }
    }
}

namespace noa::cuda::memory {
    template<typename value_t, typename index_t, typename T, typename U, typename UnaryOp, typename>
    Extracted<value_t, index_t> extract(const shared_t<T[]>& input, size4_t input_stride,
                                        const shared_t<U[]>& lhs, size4_t lhs_stride, size4_t shape,
                                        UnaryOp unary_op, bool extract_values, bool extract_indexes, Stream& stream) {
        if (!extract_values && !extract_indexes)
            return {};

        const size4_t contiguous_stride = shape.stride();
        const size_t elements = shape.elements();
        PtrDevice<uint> map{elements, stream};
        util::ewise::unary<true>("memory::extract",
                                 lhs.get(), lhs_stride,
                                 map.get(), contiguous_stride,
                                 shape, stream, unary_op);
        auto out = extract_<value_t, index_t>(input.get(), input_stride, shape, elements,
                                              map.get(), extract_values, extract_indexes, stream);
        stream.attach(input, lhs);
        return out;
    }

    #define INSTANTIATE_EXTRACT_UNARY_BASE_(T, I)                               \
    template Extracted<T, I> extract<T,I,T,T,::noa::math::logical_not_t,void>(  \
        const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, ::noa::math::logical_not_t, bool, bool, Stream&)

    #define INSTANTIATE_EXTRACT_UNARY_(T)           \
    INSTANTIATE_EXTRACT_UNARY_BASE_(T, uint32_t);   \
    INSTANTIATE_EXTRACT_UNARY_BASE_(T, uint64_t)

    INSTANTIATE_EXTRACT_UNARY_(int32_t);
    INSTANTIATE_EXTRACT_UNARY_(int64_t);
    INSTANTIATE_EXTRACT_UNARY_(uint32_t);
    INSTANTIATE_EXTRACT_UNARY_(uint64_t);
    INSTANTIATE_EXTRACT_UNARY_(half_t);
    INSTANTIATE_EXTRACT_UNARY_(float);
    INSTANTIATE_EXTRACT_UNARY_(double);


    template<typename value_t, typename index_t, typename T, typename U, typename V, typename BinaryOp, typename>
    Extracted<value_t, index_t> extract(const shared_t<T[]>& input, size4_t input_stride,
                                        const shared_t<U[]>& lhs, size4_t lhs_stride, V rhs, size4_t shape,
                                        BinaryOp binary_op, bool extract_values, bool extract_indexes, Stream& stream) {
        if (!extract_values && !extract_indexes)
            return {};

        const size4_t contiguous_stride = shape.stride();
        const size_t elements = shape.elements();
        PtrDevice<uint> map{elements, stream};
        util::ewise::binary<true>("memory::extract",
                                  lhs.get(), lhs_stride, rhs,
                                  map.get(), contiguous_stride,
                                  shape, stream, binary_op);
        auto out = extract_<value_t, index_t>(input.get(), input_stride, shape, elements,
                                              map.get(), extract_values, extract_indexes, stream);
        stream.attach(lhs);
        return out;
    }

    template<typename value_t, typename index_t, typename T, typename U, typename V, typename BinaryOp, typename>
    Extracted<value_t, index_t> extract(const shared_t<T[]>& input, size4_t input_stride,
                                        U lhs, const shared_t<V[]>& rhs, size4_t rhs_stride, size4_t shape,
                                        BinaryOp binary_op, bool extract_values, bool extract_indexes, Stream& stream) {
        if (!extract_values && !extract_indexes)
            return {};

        const size4_t contiguous_stride = shape.stride();
        const size_t elements = shape.elements();
        PtrDevice<uint> map{elements, stream};
        util::ewise::binary<true>("memory::extract",
                                  lhs, rhs.get(), rhs_stride,
                                  map.get(), contiguous_stride,
                                  shape, stream, binary_op);
        auto out = extract_<value_t, index_t>(input.get(), input_stride, shape, elements,
                                              map.get(), extract_values, extract_indexes, stream);
        stream.attach(rhs);
        return out;
    }

    template<typename value_t, typename index_t, typename T, typename U, typename V, typename BinaryOp, typename>
    Extracted<value_t, index_t> extract(const shared_t<T[]>& input, size4_t input_stride,
                                        const shared_t<U[]>& lhs, size4_t lhs_stride,
                                        const shared_t<V[]>& rhs, size4_t rhs_stride,
                                        size4_t shape, BinaryOp binary_op, bool extract_values, bool extract_indexes,
                                        Stream& stream) {
        if (!extract_values && !extract_indexes)
            return {};

        const size4_t contiguous_stride = shape.stride();
        const size_t elements = shape.elements();
        PtrDevice<uint> map{elements, stream};
        util::ewise::binary<true>("memory::extract",
                                  lhs.get(), lhs_stride,
                                  rhs.get(), rhs_stride,
                                  map.get(), contiguous_stride,
                                  shape, stream, binary_op);
        auto out = extract_<value_t, index_t>(input.get(), input_stride, shape, elements,
                                              map.get(), extract_values, extract_indexes, stream);
        stream.attach(lhs, rhs);
        return out;
    }

    #define INSTANTIATE_EXTRACT_BINARY_BASE0_(T, I, BINARY)                                                                                                         \
    template Extracted<T,I> extract<T,I,T,T,T,BINARY,void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, T, size4_t, BINARY, bool, bool, Stream&);  \
    template Extracted<T,I> extract<T,I,T,T,T,BINARY,void>(const shared_t<T[]>&, size4_t, T, const shared_t<T[]>&, size4_t, size4_t, BINARY, bool, bool, Stream&);  \
    template Extracted<T,I> extract<T,I,T,T,T,BINARY,void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, BINARY, bool, bool, Stream&)

    #define INSTANTIATE_EXTRACT_BINARY_BASE2_(T, I)                   \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,I,::noa::math::equal_t);      \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,I,::noa::math::not_equal_t);  \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,I,::noa::math::less_t);       \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,I,::noa::math::less_equal_t); \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,I,::noa::math::greater_t);    \
    INSTANTIATE_EXTRACT_BINARY_BASE0_(T,I,::noa::math::greater_equal_t)

    #define INSTANTIATE_EXTRACT_BINARY_(T)           \
    INSTANTIATE_EXTRACT_BINARY_BASE2_(T, uint32_t);  \
    INSTANTIATE_EXTRACT_BINARY_BASE2_(T, uint64_t)

    INSTANTIATE_EXTRACT_BINARY_(int32_t);
    INSTANTIATE_EXTRACT_BINARY_(int64_t);
    INSTANTIATE_EXTRACT_BINARY_(uint32_t);
    INSTANTIATE_EXTRACT_BINARY_(uint64_t);
    INSTANTIATE_EXTRACT_BINARY_(half_t);
    INSTANTIATE_EXTRACT_BINARY_(float);
    INSTANTIATE_EXTRACT_BINARY_(double);


    template<typename value_t, typename index_t, typename T, typename>
    void insert(const Extracted<value_t, index_t>& extracted, const shared_t<T[]>& output, Stream& stream) {
        const uint blocks = noa::math::divideUp(static_cast<uint>(extracted.count), BLOCK_WORK_SIZE);
        const int vec_size = std::min(util::maxVectorCount(extracted.values.get()),
                                      util::maxVectorCount(extracted.indexes.get()));
        if (vec_size == 4) {
            return stream.enqueue("memory::insert", insert_<value_t, index_t, 4>, {blocks, BLOCK_SIZE},
                                  extracted.values.get(), extracted.indexes.get(), extracted.count, output.get());
        } else if (vec_size == 2) {
            return stream.enqueue("memory::insert", insert_<value_t, index_t, 2>, {blocks, BLOCK_SIZE},
                                  extracted.values.get(), extracted.indexes.get(), extracted.count, output.get());
        } else {
            return stream.enqueue("memory::insert", insert_<value_t, index_t, 1>, {blocks, BLOCK_SIZE},
                                  extracted.values.get(), extracted.indexes.get(), extracted.count, output.get());
        }
        stream.attach(extracted.values, extracted.indexes, output);
    }

    #define INSTANTIATE_INSERT_BASE_(T, I) \
    template void insert<T,I,T,void>(const Extracted<T, I>&, const shared_t<T[]>&, Stream&)

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
