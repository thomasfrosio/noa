#include <cub/device/device_scan.cuh>

#include "noa/common/Math.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Index.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"

#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/EwiseUnary.cuh"
#include "noa/gpu/cuda/utils/EwiseBinary.cuh"

namespace {
    using namespace ::noa;

    constexpr uint32_t ELEMENTS_PER_THREAD = 4;
    constexpr uint32_t BLOCK_SIZE = 128;
    constexpr uint32_t BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;

    constexpr dim3 ELEMENTS_PER_THREAD_2D(1, ELEMENTS_PER_THREAD);
    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD_2D.x,
                                      BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD_2D.y);

    template<typename T, typename I>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void extract1DContiguous_(const T* __restrict__ input, uint32_t elements,
                              const uint32_t* __restrict__ map, const uint32_t* __restrict__ map_scan,
                              T* __restrict__ sequence_elements, I* __restrict__ sequence_offsets) {
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x + threadIdx.x;
        #pragma unroll
        for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const uint32_t gid = base + BLOCK_SIZE * i;
            if (gid < elements) {
                if (map[gid]) {
                    const uint32_t pos_sequence = map_scan[gid] - 1; // inclusive sum will start at 1
                    if (sequence_elements)
                        sequence_elements[pos_sequence] = input[gid];
                    if (sequence_offsets)
                        sequence_offsets[pos_sequence] = static_cast<I>(gid);
                }
            }
        }
    }

    template<typename T, typename I>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void extractStride4D_(const T* __restrict__ input, uint4_t strides, uint4_t shape,
                          const uint32_t* __restrict__ map, const uint32_t* __restrict__ map_scan,
                          T* __restrict__ sequence_elements, I* __restrict__ sequence_offsets, uint32_t blocks_x) {
        const uint4_t contiguous_strides = shape.strides();
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        map += indexing::at(gid[0], gid[1], contiguous_strides);
        map_scan += indexing::at(gid[0], gid[1], contiguous_strides);

        #pragma unroll
        for (int32_t k = 0; k < ELEMENTS_PER_THREAD_2D.y; ++k) {
            #pragma unroll
            for (int32_t l = 0; l < ELEMENTS_PER_THREAD_2D.x; ++l) {
                const uint32_t ik = gid[2] + BLOCK_SIZE_2D.y * k;
                const uint32_t il = gid[3] + BLOCK_SIZE_2D.x * l;
                if (ik < shape[2] && il < shape[3]) {
                    const uint32_t offset = ik * contiguous_strides[2] + il * contiguous_strides[3];
                    if (map[offset]) {
                        const uint32_t pos_sequence = map_scan[offset] - 1; // inclusive sum will start at 1
                        const uint32_t pos_input = indexing::at(gid[0], gid[1], ik, il, strides);
                        if (sequence_elements)
                            sequence_elements[pos_sequence] = input[pos_input];
                        if (sequence_offsets)
                            sequence_offsets[pos_sequence] = static_cast<I>(pos_input);
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
    extract_(const T* input, dim4_t strides, dim4_t shape, dim_t elements,
             const uint32_t* map, bool extract_values, bool extract_offsets, cuda::Stream& stream) {
        using namespace ::noa::cuda::memory;

        // Inclusive scan sum to get the number of elements to extract and
        // map_scan to know the order of the elements to extract.
        const auto int_elements = safe_cast<int32_t>(elements);
        uint32_t elements_to_extract{};
        PtrDevice<uint32_t> map_scan(elements, stream);
        {
            size_t cub_tmp_bytes{};
            NOA_THROW_IF(cub::DeviceScan::InclusiveSum(nullptr, cub_tmp_bytes, map,
                                                       map_scan.get(), int_elements, stream.id()));
            PtrDevice<std::byte> cub_tmp(cub_tmp_bytes, stream);
            NOA_THROW_IF(cub::DeviceScan::InclusiveSum(cub_tmp.get(), cub_tmp_bytes, map,
                                                       map_scan.get(), int_elements, stream.id()));
            copy(map_scan.get() + int_elements - 1, &elements_to_extract, 1, stream);
            stream.synchronize(); // we cannot use elements_to_extract before that point
            if (!elements_to_extract)
                return {};
        }

        Extracted<T, I> extracted{
            extract_values ? PtrDevice<T>::alloc(elements_to_extract, stream) : nullptr,
            extract_offsets ? PtrDevice<I>::alloc(elements_to_extract, stream) : nullptr,
            elements_to_extract
        };

        // Compute the sequence(s).
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        if (indexing::areContiguous(strides, shape)) {
            const uint32_t blocks = noa::math::divideUp(static_cast<uint32_t>(int_elements), BLOCK_WORK_SIZE);
            const cuda::LaunchConfig config{blocks, BLOCK_SIZE};
            stream.enqueue("memory::extract", extract1DContiguous_<T, I>, config,
                           input, elements, map, map_scan.get(),
                           extracted.values.get(), extracted.offsets.get());
        } else {
            const auto i_shape = safe_cast<uint4_t>(shape);
            const uint32_t blocks_x = noa::math::divideUp(i_shape[3], BLOCK_WORK_SIZE_2D.x);
            const uint32_t blocks_y = noa::math::divideUp(i_shape[2], BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, i_shape[1], i_shape[0]);
            const cuda::LaunchConfig config{blocks, BLOCK_SIZE_2D};
            stream.enqueue("memory::extract", extractStride4D_<T, I>, config,
                           input, safe_cast<uint4_t>(strides), i_shape, map, map_scan.get(),
                           extracted.values.get(), extracted.offsets.get(), blocks_x);
        }
        stream.attach(extracted.values, extracted.offsets);
        return extracted;
    }

    template<typename T, typename I, int32_t VEC_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void extract_(const T* input, T* output, const I* offsets, uint32_t elements) {
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint32_t gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements) {
                    const I offset = offsets[gid];
                    output[gid] = input[offset];
                }
            }
        } else {
            const uint32_t remaining = elements - base;
            offsets += base;
            output += base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint32_t cid = BLOCK_SIZE * i + threadIdx.x;
                    if (cid < remaining) {
                        const I offset = offsets[cid];
                        output[cid] = input[offset];
                    }
                }
            } else {
                I offsets_[ELEMENTS_PER_THREAD];
                T values[ELEMENTS_PER_THREAD];
                using namespace noa::cuda::utils::block;
                vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(offsets, offsets_, threadIdx.x);
                #pragma unroll
                for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i)
                    values[i] = input[offsets_[i]];
                vectorizedStore<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(values, output, threadIdx.x);
            }
        }
    }

    template<typename T, typename I, int32_t VEC_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void insert_(const T* sequence_values, const I* sequence_offsets, uint32_t sequence_size, T* output) {
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint32_t gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < sequence_size) {
                    const I index = sequence_offsets[gid];
                    output[index] = sequence_values[gid];
                }
            }
        } else {
            const uint32_t remaining = sequence_size - base;
            sequence_values += base;
            sequence_offsets += base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint32_t cid = BLOCK_SIZE * i + threadIdx.x;
                    if (cid < remaining) {
                        const I index = sequence_offsets[cid];
                        output[index] = sequence_values[cid];
                    }
                }
            } else {
                T values[ELEMENTS_PER_THREAD];
                I indexes[ELEMENTS_PER_THREAD];
                using namespace noa::cuda::utils::block;
                vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(sequence_values, values, threadIdx.x);
                vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(sequence_offsets, indexes, threadIdx.x);
                #pragma unroll
                for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i)
                    output[indexes[i]] = values[i];
            }
        }
    }
}

// TODO If input is contiguous AND lhs/rhs are equal to input AND the offsets are not extracted,
//      cud::DeviceSelect::If can be used instead.
namespace noa::cuda::memory {
    template<typename value_t, typename offset_t, typename T, typename U, typename UnaryOp, typename>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                         const shared_t<U[]>& lhs, dim4_t lhs_strides, dim4_t shape,
                                         UnaryOp unary_op, bool extract_values, bool extract_offsets, Stream& stream) {
        if (!extract_values && !extract_offsets)
            return {};

        const auto order = indexing::order(input_strides, shape);
        input_strides = indexing::reorder(input_strides, order);
        lhs_strides = indexing::reorder(lhs_strides, order);
        shape = indexing::reorder(shape, order);

        const dim4_t contiguous_strides = shape.strides();
        const dim_t elements = shape.elements();
        PtrDevice<uint32_t> map(elements, stream);
        utils::ewise::unary<true>(
                "memory::extract",
                lhs.get(), lhs_strides,
                map.get(), contiguous_strides,
                shape, false, stream, unary_op);
        auto out = extract_<value_t, offset_t>(
                input.get(), input_strides, shape, elements,
                map.get(), extract_values, extract_offsets, stream);
        stream.attach(input, lhs);
        return out;
    }

    #define INSTANTIATE_EXTRACT_UNARY_BASE_(T, I)                               \
    template Extracted<T, I> extract<T,I,T,T,::noa::math::logical_not_t,void>(  \
        const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, ::noa::math::logical_not_t, bool, bool, Stream&)

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


    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp, typename>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                         const shared_t<U[]>& lhs, dim4_t lhs_strides, V rhs, dim4_t shape,
                                         BinaryOp binary_op, bool extract_values, bool extract_offsets,
                                         Stream& stream) {
        if (!extract_values && !extract_offsets)
            return {};

        const auto order = indexing::order(input_strides, shape);
        input_strides = indexing::reorder(input_strides, order);
        lhs_strides = indexing::reorder(lhs_strides, order);
        shape = indexing::reorder(shape, order);

        const dim4_t contiguous_strides = shape.strides();
        const dim_t elements = shape.elements();
        PtrDevice<uint32_t> map(elements, stream);
        utils::ewise::binary<true>(
                "memory::extract",
                lhs.get(), lhs_strides, rhs,
                map.get(), contiguous_strides,
                shape, false, stream, binary_op);
        auto out = extract_<value_t, offset_t>(
                input.get(), input_strides, shape, elements,
                map.get(), extract_values, extract_offsets, stream);
        stream.attach(lhs);
        return out;
    }

    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp, typename>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                         U lhs, const shared_t<V[]>& rhs, dim4_t rhs_strides, dim4_t shape,
                                         BinaryOp binary_op, bool extract_values, bool extract_offsets,
                                         Stream& stream) {
        if (!extract_values && !extract_offsets)
            return {};

        const auto order = indexing::order(input_strides, shape);
        input_strides = indexing::reorder(input_strides, order);
        rhs_strides = indexing::reorder(rhs_strides, order);
        shape = indexing::reorder(shape, order);

        const dim4_t contiguous_strides = shape.strides();
        const dim_t elements = shape.elements();
        PtrDevice<uint32_t> map(elements, stream);
        utils::ewise::binary<true>(
                "memory::extract",
                lhs, rhs.get(), rhs_strides,
                map.get(), contiguous_strides,
                shape, false, stream, binary_op);
        auto out = extract_<value_t, offset_t>(
                input.get(), input_strides, shape, elements,
                map.get(), extract_values, extract_offsets, stream);
        stream.attach(rhs);
        return out;
    }

    template<typename value_t, typename offset_t, typename T, typename U, typename V, typename BinaryOp, typename>
    Extracted<value_t, offset_t> extract(const shared_t<T[]>& input, dim4_t input_strides,
                                         const shared_t<U[]>& lhs, dim4_t lhs_strides,
                                         const shared_t<V[]>& rhs, dim4_t rhs_strides,
                                         dim4_t shape, BinaryOp binary_op, bool extract_values, bool extract_offsets,
                                         Stream& stream) {
        if (!extract_values && !extract_offsets)
            return {};

        const auto order = indexing::order(input_strides, shape);
        input_strides = indexing::reorder(input_strides, order);
        lhs_strides = indexing::reorder(lhs_strides, order);
        rhs_strides = indexing::reorder(rhs_strides, order);
        shape = indexing::reorder(shape, order);

        const dim4_t contiguous_strides = shape.strides();
        const dim_t elements = shape.elements();
        PtrDevice<uint32_t> map(elements, stream);
        utils::ewise::binary<true>(
                "memory::extract",
                lhs.get(), lhs_strides,
                rhs.get(), rhs_strides,
                map.get(), contiguous_strides,
                shape, false, stream, binary_op);
        auto out = extract_<value_t, offset_t>(
                input.get(), input_strides, shape, elements,
                map.get(), extract_values, extract_offsets, stream);
        stream.attach(lhs, rhs);
        return out;
    }

    #define INSTANTIATE_EXTRACT_BINARY_BASE0_(T, I, BINARY)                                                                                                     \
    template Extracted<T,I> extract<T,I,T,T,T,BINARY,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, T, dim4_t, BINARY, bool, bool, Stream&); \
    template Extracted<T,I> extract<T,I,T,T,T,BINARY,void>(const shared_t<T[]>&, dim4_t, T, const shared_t<T[]>&, dim4_t, dim4_t, BINARY, bool, bool, Stream&); \
    template Extracted<T,I> extract<T,I,T,T,T,BINARY,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, BINARY, bool, bool, Stream&)

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


    template<typename T, typename U, typename V, typename>
    void extract(const shared_t<T[]>& input, const shared_t<U[]>& offsets,
                 const shared_t<V[]>& output, dim_t elements, Stream& stream) {
        const uint32_t blocks = noa::math::divideUp(safe_cast<uint32_t>(elements), BLOCK_WORK_SIZE);
        const int32_t vec_size = std::min(utils::maxVectorCount(output.get()),
                                          utils::maxVectorCount(offsets.get()));
        if (vec_size == 4) {
            stream.enqueue("memory::extract", extract_<T, U, 4>, {blocks, BLOCK_SIZE},
                           input.get(), output.get(), offsets.get(), elements);
        } else if (vec_size == 2) {
            stream.enqueue("memory::extract", extract_<T, U, 2>, {blocks, BLOCK_SIZE},
                           input.get(), output.get(), offsets.get(), elements);
        } else {
            stream.enqueue("memory::extract", extract_<T, U, 1>, {blocks, BLOCK_SIZE},
                           input.get(), output.get(), offsets.get(), elements);
        }
        stream.attach(input, offsets, output);
    }

    template<typename value_t, typename offset_t, typename T, typename>
    void insert(const Extracted<value_t, offset_t>& extracted, const shared_t<T[]>& output, Stream& stream) {
        const uint32_t blocks = noa::math::divideUp(static_cast<uint>(extracted.count), BLOCK_WORK_SIZE);
        const int32_t vec_size = std::min(utils::maxVectorCount(extracted.values.get()),
                                          utils::maxVectorCount(extracted.offsets.get()));
        if (vec_size == 4) {
            stream.enqueue("memory::insert", insert_<value_t, offset_t, 4>, {blocks, BLOCK_SIZE},
                           extracted.values.get(), extracted.offsets.get(), extracted.count, output.get());
        } else if (vec_size == 2) {
            stream.enqueue("memory::insert", insert_<value_t, offset_t, 2>, {blocks, BLOCK_SIZE},
                           extracted.values.get(), extracted.offsets.get(), extracted.count, output.get());
        } else {
            stream.enqueue("memory::insert", insert_<value_t, offset_t, 1>, {blocks, BLOCK_SIZE},
                           extracted.values.get(), extracted.offsets.get(), extracted.count, output.get());
        }
        stream.attach(extracted.values, extracted.offsets, output);
    }

    #define INSTANTIATE_INSERT_BASE_(T, I)                                                                                  \
    template void extract<T,I,T,void>(const shared_t<T[]>&, const shared_t<I[]>&, const shared_t<T[]>&, dim_t, Stream&);    \
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
