// Use cub to do the actual sorting. Thrust seems to have its own merge sort, but its main API does not allow you to
// choose between radix and merge sort. However, it seems that it always selects radix if the type and comparison
// operator is supported by the radix sort, i.e. base types and < or > comparison. In our case, that's all we need,
// so use the radix sort for everything...
#include <cub/device/device_radix_sort.cuh>

#include "noa/gpu/cuda/math/Sort.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Iota.h"

namespace {
    using namespace ::noa;

    template<typename T>
    cudaError cubRadixSortKeys_(void* temp_storage, size_t& temp_storage_bytes, cub::DoubleBuffer<T>& keys,
                                int32_t size, bool ascending, cuda::Stream& stream) {
        // half_t can be safely reinterpreted to CUDA's __half.
        using cubT = std::conditional_t<std::is_same_v<T, half_t>, __half, T>;

        if (ascending) {
            return cub::DeviceRadixSort::SortKeys(
                    temp_storage, temp_storage_bytes,
                    reinterpret_cast<cub::DoubleBuffer<cubT>&>(keys),
                    size, 0, static_cast<int>(sizeof(T) * 8),
                    stream.id());
        } else {
            return cub::DeviceRadixSort::SortKeysDescending(
                    temp_storage, temp_storage_bytes,
                    reinterpret_cast<cub::DoubleBuffer<cubT>&>(keys),
                    size, 0, static_cast<int>(sizeof(T) * 8),
                    stream.id());
        }
    }

    template<typename T, typename U>
    cudaError cubRadixSortPairs_(void* temp_storage, size_t& temp_storage_bytes,
                                 cub::DoubleBuffer<T>& keys, cub::DoubleBuffer<U>& values,
                                 int32_t size, bool ascending, cuda::Stream& stream) {
        // half_t can be safely reinterpreted to CUDA's __half.
        using cubT = std::conditional_t<std::is_same_v<T, half_t>, __half, T>;
        using cubU = std::conditional_t<std::is_same_v<U, half_t>, __half, U>;

        if (ascending) {
            return cub::DeviceRadixSort::SortPairs(
                    temp_storage, temp_storage_bytes,
                    reinterpret_cast<cub::DoubleBuffer<cubT>&>(keys),
                    reinterpret_cast<cub::DoubleBuffer<cubU>&>(values),
                    size, 0, static_cast<int>(sizeof(T) * 8),
                    stream.id());
        } else {
            return cub::DeviceRadixSort::SortPairsDescending(
                    temp_storage, temp_storage_bytes,
                    reinterpret_cast<cub::DoubleBuffer<cubT>&>(keys),
                    reinterpret_cast<cub::DoubleBuffer<cubU>&>(values),
                    size, 0, static_cast<int>(sizeof(T) * 8),
                    stream.id());
        }
    }

    // Sorts the third dimension of "values" using cub radix sort.
    // Works with non-contiguous strides. If dim is non-contiguous, allocates one row.
    // If there's a lot of rows to sort, sortBatched_ should be faster.
    template<typename T>
    void sortIterative_(const shared_t<T[]>& values, dim4_t strides, dim4_t shape,
                        int32_t dim, bool ascending, cuda::Stream& stream) {
        NOA_ASSERT(strides[dim] > 0); // nothing to sort if dim is broadcast

        const bool dim_is_contiguous = strides[dim] == 1;
        const auto dim_size = safe_cast<int32_t>(shape[dim]);
        const dim4_t dim_shape{1, 1, 1, shape[dim]};
        const dim_t dim_stride = strides[dim];
        const dim4_t dim_strides(dim_stride);

        // TODO Do one single allocation for buffer(s) and tmp storage. Problem is the alignment?

        // Prepare the alternate buffer.
        using unique_ptr_t = typename cuda::memory::PtrDevice<T>::alloc_unique_t;
        unique_ptr_t key_buffer;
        unique_ptr_t key_buffer_alt;
        if (dim_is_contiguous) {
            key_buffer = nullptr;
            key_buffer_alt = cuda::memory::PtrDevice<T>::alloc(dim_size, stream);
        } else {
            key_buffer = cuda::memory::PtrDevice<T>::alloc(dim_size, stream);
            key_buffer_alt = cuda::memory::PtrDevice<T>::alloc(dim_size, stream);
        }
        cub::DoubleBuffer<T> keys(key_buffer.get(), key_buffer_alt.get());

        // Allocates for the small tmp storage.
        size_t temp_storage_bytes;
        NOA_THROW_IF(cubRadixSortKeys_<T>(nullptr, temp_storage_bytes, keys, dim_size, ascending, stream));
        const auto temp_storage = cuda::memory::PtrDevice<byte_t>::alloc(temp_storage_bytes, stream);

        // Prepare the iterations.
        dim3_t iter_shape_;
        dim3_t iter_strides_;
        int32_t count = 0;
        for (int32_t i = 0; i < 4; ++i) {
            if (i != dim) {
                iter_shape_[count] = shape[i];
                iter_strides_[count] = strides[i];
                ++count;
            }
        }

        // Sort the axis.
        for (dim_t i = 0; i < iter_shape_[0]; ++i) {
            for (dim_t j = 0; j < iter_shape_[1]; ++j) {
                for (dim_t k = 0; k < iter_shape_[2]; ++k) {

                    const dim_t offset = indexing::at(i, j, k, iter_strides_);
                    T* values_iter = values.get() + offset;

                    // (Re)set the buffers.
                    keys.selector = 0;
                    if (dim_is_contiguous) {
                        keys.d_buffers[0] = values_iter;
                    } else {
                        cuda::memory::copy(values_iter, dim_strides,
                                           key_buffer.get(), dim_shape.strides(),
                                           dim_shape, stream);
                    }

                    NOA_THROW_IF(cubRadixSortKeys_<T>(temp_storage.get(), temp_storage_bytes, keys,
                                                      dim_size, ascending, stream));

                    if (dim_is_contiguous) {
                        if (keys.selector != 0) {
                            // Unfortunately, the results are in the alternate buffer,
                            // so copy it back to the original array.
                            cuda::memory::copy(key_buffer_alt.get(), dim_shape.strides(),
                                               values_iter /* or key_buffer */, dim_strides,
                                               dim_shape, stream);
                        }
                    } else {
                        cuda::memory::copy(keys.selector == 0 ? key_buffer.get() : key_buffer_alt.get(),
                                           dim_shape.strides(),
                                           values_iter, dim_strides,
                                           dim_shape, stream);
                    }
                }
            }
        }
        stream.attach(values);
    }

    // Sort any dimension [0..3] of the input array, in-place.
    // The array can have non-contiguous strides in any dimension.
    // Basically allocates x2 the shape...
    template<typename T>
    void sortBatched_(const shared_t<T[]>& values, dim4_t strides, dim4_t shape,
                      int32_t dim, bool ascending, cuda::Stream& stream) {
        const bool contiguous = indexing::areContiguous(strides, shape);
        const auto elements = safe_cast<int32_t>(shape.elements());

        // Prepare the keys.
        dim4_t tile = shape;
        tile[dim] = 1; // mark elements with their original axis.
        shared_t<uint[]> key_buffer = cuda::memory::PtrDevice<uint>::alloc(elements, stream);
        const auto key_buffer_alt = cuda::memory::PtrDevice<uint>::alloc(elements, stream);
        cuda::memory::iota(key_buffer, shape.strides(), shape, tile, stream);

        // Prepare the values.
        using unique_ptr_t = typename cuda::memory::PtrDevice<T>::alloc_unique_t;
        shared_t<T[]> val_buffer;
        unique_ptr_t val_buffer_alt;
        if (contiguous) {
            val_buffer = values;
            val_buffer_alt = cuda::memory::PtrDevice<T>::alloc(elements, stream);
        } else {
            val_buffer = cuda::memory::PtrDevice<T>::alloc(elements, stream);
            val_buffer_alt = cuda::memory::PtrDevice<T>::alloc(elements, stream);
            cuda::memory::copy(values.get(), strides, val_buffer.get(), shape.strides(), shape, stream);
        }

        // Gather them in the cub interface.
        cub::DoubleBuffer<uint> cub_keys(key_buffer.get(), key_buffer_alt.get());
        cub::DoubleBuffer<T> cub_vals(val_buffer.get(), val_buffer_alt.get());

        // Allocates for the small tmp storage.
        // The documentation says this should be a small value and is relative to the input size.
        size_t tmp_bytes0, tmp_bytes1;
        const cudaError err0 = cubRadixSortPairs_<uint, T>(
                nullptr, tmp_bytes0, cub_keys, cub_vals, elements, ascending, stream);
        const cudaError err1 = cubRadixSortPairs_<T, uint>(
                nullptr, tmp_bytes1, cub_vals, cub_keys, elements, ascending, stream);
        if (err0 != cudaSuccess || err1 != cudaSuccess) {
            NOA_THROW("Could not find temporary allocation size. 0:{}, 1:{}",
                      cudaGetErrorString(err0), cudaGetErrorString(err1));
        }

        tmp_bytes0 = std::max(tmp_bytes0, tmp_bytes1);
        const auto tmp = cuda::memory::PtrDevice<byte_t>::alloc(tmp_bytes0, stream);

        // Sort the entire array based on the values, but updates the original indexes.
        // It is important that the second sort is stable, which is the case with radix sort.
        NOA_THROW_IF((cubRadixSortPairs_<T, uint>(tmp.get(), tmp_bytes0, cub_vals, cub_keys, elements, ascending, stream)));
        NOA_THROW_IF((cubRadixSortPairs_<uint, T>(tmp.get(), tmp_bytes0, cub_keys, cub_vals, elements, true, stream)));

        // Then permute it back to the original order.
        // Find the permutation from "key_val" to "values":
        dim4_t input_shape(shape[dim]);
        int4_t permutation(3);
        int32_t count = 0;
        for (int32_t i = 0; i < 4; ++i) {
            if (i != dim) {
                input_shape[count] = shape[i];
                permutation[i] = count;
                ++count;
            }
        }

        // Permutation. Since we do 2 sorts, it seems that the result is always at the position 0. If "values" was
        // contiguous, it means the result is already in "values" but with a possible permutation. In this case,
        // we have to permute in the alternate buffer and then copy the result back to "values"...
        const dim4_t current_strides_permuted = indexing::reorder(input_shape.strides(), permutation);
        if (values.get() == cub_vals.Current() && !all(permutation == int4_t{0, 1, 2, 3})) {
            cuda::memory::copy(values.get(), current_strides_permuted,
                               val_buffer_alt.get(), shape.strides(),
                               shape, stream);
            cuda::memory::copy(val_buffer_alt.get(), shape.strides(), values.get(), strides, shape, stream);
        } else {
            cuda::memory::copy(cub_vals.selector == 0 ? val_buffer.get() : val_buffer_alt.get(),
                               current_strides_permuted,
                               values.get(), strides, shape, stream);
        }
        stream.attach(values);
    }
}

namespace noa::cuda::math {
    template<typename T, typename>
    void sort(const shared_t<T[]>& array, dim4_t strides, dim4_t shape, bool ascending, int32_t dim, Stream& stream) {
        NOA_ASSERT(array.get() != nullptr && all(shape > 0));

        // Allow dim = -1 to specify the first non-empty dimension in the rightmost order.
        if (dim == -1)
            dim = shape[3] > 1 ? 3 : shape[2] > 1 ? 2 : shape[1] > 1 ? 1 : 0;
        NOA_ASSERT(dim >= 0 && dim <= 3);

        if (strides[dim] == 0)
            return; // there's one value in the dimension to sort...

        // If there's not a lot of axes to sort, use the iterative version which uses less memory
        // and does a single sort per axis. Otherwise, use the batched version which uses more memory
        // but uses 2 sorts (1 being a stable sort), and a possible permutation, for the entire array.
        dim4_t shape_ = shape;
        shape_[dim] = 1;
        const dim_t iterations = shape_.elements();
        if (iterations < 10)
            sortIterative_(array, strides, shape, dim, ascending, stream);
        else
            sortBatched_(array, strides, shape, dim, ascending, stream);
    }

    #define NOA_INSTANTIATE_SORT_(T) \
    template void sort<T,void>(const shared_t<T[]>&, dim4_t, dim4_t, bool, int32_t, Stream&)

    NOA_INSTANTIATE_SORT_(int16_t);
    NOA_INSTANTIATE_SORT_(int32_t);
    NOA_INSTANTIATE_SORT_(int64_t);
    NOA_INSTANTIATE_SORT_(uint16_t);
    NOA_INSTANTIATE_SORT_(uint32_t);
    NOA_INSTANTIATE_SORT_(uint64_t);
    NOA_INSTANTIATE_SORT_(half_t);
    NOA_INSTANTIATE_SORT_(float);
    NOA_INSTANTIATE_SORT_(double);
}
