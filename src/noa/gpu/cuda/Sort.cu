// Use cub to do the actual sorting. Thrust seems to have its own merge sort, but its main API does not allow you to
// choose between radix and merge sort. However, it seems that it always selects radix if the type and comparison
// operator is supported by the radix sort, i.e. base types and < or > comparison. In our case, that's all we need,
// so use the radix sort for everything...
#include <cub/device/device_radix_sort.cuh>

#include "noa/gpu/cuda/Sort.hpp"
#include "noa/gpu/cuda/memory/AllocatorDevice.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/memory/Iota.hpp"

namespace {
    using namespace ::noa;

    template<typename T>
    cudaError cub_radix_sort_keys_(
            void* temp_storage, size_t& temp_storage_bytes,
            cub::DoubleBuffer<T>& keys,
            i32 size, bool ascending, noa::cuda::Stream& stream) {
        // f16 can be safely reinterpreted to CUDA's __half.
        using cubT = std::conditional_t<std::is_same_v<T, f16>, __half, T>;

        if (ascending) {
            return cub::DeviceRadixSort::SortKeys(
                    temp_storage, temp_storage_bytes,
                    reinterpret_cast<cub::DoubleBuffer<cubT>&>(keys),
                    size, 0, static_cast<i32>(sizeof(T) * 8),
                    stream.id());
        } else {
            return cub::DeviceRadixSort::SortKeysDescending(
                    temp_storage, temp_storage_bytes,
                    reinterpret_cast<cub::DoubleBuffer<cubT>&>(keys),
                    size, 0, static_cast<i32>(sizeof(T) * 8),
                    stream.id());
        }
    }

    template<typename T, typename U>
    cudaError cub_radix_sort_pairs_(
            void* temp_storage, size_t& temp_storage_bytes,
            cub::DoubleBuffer<T>& keys, cub::DoubleBuffer<U>& values,
            i32 size, bool ascending, noa::cuda::Stream& stream) {
        // f16 can be safely reinterpreted to CUDA's __half.
        using cubT = std::conditional_t<std::is_same_v<T, f16>, __half, T>;
        using cubU = std::conditional_t<std::is_same_v<U, f16>, __half, U>;

        if (ascending) {
            return cub::DeviceRadixSort::SortPairs(
                    temp_storage, temp_storage_bytes,
                    reinterpret_cast<cub::DoubleBuffer<cubT>&>(keys),
                    reinterpret_cast<cub::DoubleBuffer<cubU>&>(values),
                    size, 0, static_cast<i32>(sizeof(T) * 8),
                    stream.id());
        } else {
            return cub::DeviceRadixSort::SortPairsDescending(
                    temp_storage, temp_storage_bytes,
                    reinterpret_cast<cub::DoubleBuffer<cubT>&>(keys),
                    reinterpret_cast<cub::DoubleBuffer<cubU>&>(values),
                    size, 0, static_cast<i32>(sizeof(T) * 8),
                    stream.id());
        }
    }

    // Sorts the third dimension of "values" using cub radix sort.
    // Works with non-contiguous strides. If dim is non-contiguous, allocates one row.
    // If there's a lot of rows to sort, sort_batched_ should be faster.
    template<typename T>
    void sort_iterative_(
            T* values, const Strides4<i64>& strides, const Shape4<i64>& shape,
            i32 dim, bool ascending, noa::cuda::Stream& stream) {
        NOA_ASSERT(strides[dim] > 0); // nothing to sort if dim is broadcast

        const bool dim_is_contiguous = strides[dim] == 1;
        const auto dim_size = safe_cast<i32>(shape[dim]);
        const auto dim_shape = Shape4<i64>{1, 1, 1, shape[dim]};
        const auto dim_stride = strides[dim];
        const auto dim_strides = Strides4<i64>(dim_stride);

        // TODO Do one single allocation for buffer(s) and tmp storage. Problem is the alignment?

        // Prepare the alternate buffer.
        using unique_t = typename noa::cuda::memory::AllocatorDevice<T>::unique_type;
        unique_t key_buffer;
        unique_t key_buffer_alt;
        if (dim_is_contiguous) {
            key_buffer = nullptr;
            key_buffer_alt = noa::cuda::memory::AllocatorDevice<T>::allocate_async(dim_size, stream);
        } else {
            key_buffer = noa::cuda::memory::AllocatorDevice<T>::allocate_async(dim_size, stream);
            key_buffer_alt = noa::cuda::memory::AllocatorDevice<T>::allocate_async(dim_size, stream);
        }
        cub::DoubleBuffer<T> keys(key_buffer.get(), key_buffer_alt.get());

        // Allocates for the small tmp storage.
        using namespace ::noa::cuda; // error2string
        size_t temp_storage_bytes{};
        NOA_THROW_IF(cub_radix_sort_keys_<T>(nullptr, temp_storage_bytes, keys, dim_size, ascending, stream));
        const auto temp_storage = noa::cuda::memory::AllocatorDevice<Byte>::allocate_async(
                static_cast<i64>(temp_storage_bytes), stream);

        // Prepare the iterations.
        Shape3<i64> iter_shape;
        Strides3<i64> iter_strides;
        i32 count = 0;
        for (i32 i = 0; i < 4; ++i) {
            if (i != dim) {
                iter_shape[count] = shape[i];
                iter_strides[count] = strides[i];
                ++count;
            }
        }

        // Sort the axis.
        for (i64 i = 0; i < iter_shape[0]; ++i) {
            for (i64 j = 0; j < iter_shape[1]; ++j) {
                for (i64 k = 0; k < iter_shape[2]; ++k) {

                    const i64 offset = noa::indexing::at(i, j, k, iter_strides);
                    T* values_iter = values + offset;

                    // (Re)set the buffers.
                    keys.selector = 0;
                    if (dim_is_contiguous) {
                        keys.d_buffers[0] = values_iter;
                    } else {
                        noa::cuda::memory::copy(
                                values_iter, dim_strides,
                                key_buffer.get(), dim_shape.strides(),
                                dim_shape, stream);
                    }

                    NOA_THROW_IF(cub_radix_sort_keys_<T>(
                            temp_storage.get(), temp_storage_bytes, keys,
                            dim_size, ascending, stream));

                    if (dim_is_contiguous) {
                        if (keys.selector != 0) {
                            // Unfortunately, the results are in the alternate buffer,
                            // so copy it back to the original array.
                            noa::cuda::memory::copy(
                                    key_buffer_alt.get(), dim_shape.strides(),
                                    values_iter /* or key_buffer */, dim_strides,
                                    dim_shape, stream);
                        }
                    } else {
                        noa::cuda::memory::copy(
                                keys.selector == 0 ? key_buffer.get() : key_buffer_alt.get(),
                                dim_shape.strides(),
                                values_iter, dim_strides,
                                dim_shape, stream);
                    }
                }
            }
        }
    }

    // Sort any dimension [0..3] of the input array, in-place.
    // The array can have non-contiguous strides in any dimension.
    // Basically allocates x2 the shape...
    template<typename T>
    void sort_batched_(
            T* values, const Strides4<i64>& strides, const Shape4<i64>& shape,
            i32 dim, bool ascending, noa::cuda::Stream& stream) {
        const bool contiguous = noa::indexing::are_contiguous(strides, shape);
        const auto elements = safe_cast<i32>(shape.elements());

        // Prepare the keys.
        Vec4<i64> tile = shape.vec();
        tile[dim] = 1; // mark elements with their original axis.
        const auto key_buffer = noa::cuda::memory::AllocatorDevice<u32>::allocate_async(elements, stream);
        const auto key_buffer_alt = noa::cuda::memory::AllocatorDevice<u32>::allocate_async(elements, stream);
        noa::cuda::memory::iota(key_buffer.get(), shape.strides(), shape, tile, stream);

        // Prepare the values.
        using unique_t = typename noa::cuda::memory::AllocatorDevice<T>::unique_type;
        unique_t val_buffer;
        unique_t val_buffer_alt;
        T* val_ptr;
        if (contiguous) {
            val_ptr = values;
            val_buffer_alt = noa::cuda::memory::AllocatorDevice<T>::allocate_async(elements, stream);
        } else {
            val_buffer = noa::cuda::memory::AllocatorDevice<T>::allocate_async(elements, stream);
            val_ptr = val_buffer.get();
            val_buffer_alt = noa::cuda::memory::AllocatorDevice<T>::allocate_async(elements, stream);
            noa::cuda::memory::copy(values, strides, val_ptr, shape.strides(), shape, stream);
        }

        // Gather them in the cub interface.
        cub::DoubleBuffer<u32> cub_keys(key_buffer.get(), key_buffer_alt.get());
        cub::DoubleBuffer<T> cub_values(val_ptr, val_buffer_alt.get());

        // Allocates for the small tmp storage.
        // The documentation says this should be a small value and is relative to the input size.
        size_t tmp_bytes0{}, tmp_bytes1{};
        const cudaError err0 = cub_radix_sort_pairs_<u32, T>(
                nullptr, tmp_bytes0, cub_keys, cub_values, elements, ascending, stream);
        const cudaError err1 = cub_radix_sort_pairs_<T, u32>(
                nullptr, tmp_bytes1, cub_values, cub_keys, elements, ascending, stream);
        if (err0 != cudaSuccess || err1 != cudaSuccess) {
            NOA_THROW("Could not find temporary allocation size. 0:{}, 1:{}",
                      cudaGetErrorString(err0), cudaGetErrorString(err1));
        }

        tmp_bytes0 = std::max(tmp_bytes0, tmp_bytes1);
        const auto tmp = noa::cuda::memory::AllocatorDevice<Byte>::allocate_async(static_cast<i64>(tmp_bytes0), stream);

        // Sort the entire array based on the values, but updates the original indexes.
        // It is important that the second sort is stable, which is the case with radix sort.
        using namespace ::noa::cuda; // error2string
        NOA_THROW_IF((cub_radix_sort_pairs_<T, u32>(tmp.get(), tmp_bytes0, cub_values, cub_keys, elements, ascending, stream)));
        NOA_THROW_IF((cub_radix_sort_pairs_<u32, T>(tmp.get(), tmp_bytes0, cub_keys, cub_values, elements, true, stream)));

        // Then permute it back to the original order.
        // Find the permutation from "key_val" to "values":
        Shape4<i64> input_shape(shape[dim]);
        Vec4<i32> permutation(3);
        i32 count = 0;
        for (i32 i = 0; i < 4; ++i) {
            if (i != dim) {
                input_shape[count] = shape[i];
                permutation[i] = count;
                ++count;
            }
        }

        // Permutation. Since we do 2 sorts, it seems that the result is always at the position 0. If "values" was
        // contiguous, it means the result is already in "values" but with a possible permutation. In this case,
        // we have to permute in the alternate buffer and then copy the result back to "values"...
        const auto current_strides_permuted = noa::indexing::reorder(input_shape.strides(), permutation);
        if (values == cub_values.Current() && !noa::all(permutation == Vec4<i32>{0, 1, 2, 3})) {
            noa::cuda::memory::copy(
                    values, current_strides_permuted,
                    val_buffer_alt.get(), shape.strides(),
                    shape, stream);
            noa::cuda::memory::copy(
                    val_buffer_alt.get(), shape.strides(),
                    values, strides,
                    shape, stream);
        } else {
            noa::cuda::memory::copy(
                    cub_values.selector == 0 ? val_ptr : val_buffer_alt.get(),
                    current_strides_permuted,
                    values, strides, shape, stream);
        }
    }
}

namespace noa::cuda {
    template<typename T, typename>
    void sort(T* array, const Strides4<i64>& strides, const Shape4<i64>& shape,
              bool ascending, i32 dim, Stream& stream) {
        NOA_ASSERT(noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(array, stream.device());

        // Allow dim = -1 to specify the first non-empty dimension in the rightmost order.
        if (dim == -1)
            dim = shape[3] > 1 ? 3 : shape[2] > 1 ? 2 : shape[1] > 1 ? 1 : 0;
        NOA_ASSERT(dim >= 0 && dim <= 3);

        if (strides[dim] == 0)
            return; // there's only one value in the dimension to sort...

        // If there's not a lot of axes to sort, use the iterative version which uses less memory
        // and does a single sort per axis. Otherwise, use the batched version which uses more memory
        // but uses 2 sorts (1 being a stable sort), and a possible permutation, for the entire array.
        auto shape_4d = shape;
        shape_4d[dim] = 1;
        const auto iterations = shape_4d.elements();
        if (iterations < 10)
            sort_iterative_(array, strides, shape, dim, ascending, stream);
        else
            sort_batched_(array, strides, shape, dim, ascending, stream);
    }

    #define NOA_INSTANTIATE_SORT_(T) \
    template void sort<T,void>(T*, const Strides4<i64>&, const Shape4<i64>&, bool, i32, Stream&)

    NOA_INSTANTIATE_SORT_(i8);
    NOA_INSTANTIATE_SORT_(i16);
    NOA_INSTANTIATE_SORT_(i32);
    NOA_INSTANTIATE_SORT_(i64);
    NOA_INSTANTIATE_SORT_(u8);
    NOA_INSTANTIATE_SORT_(u16);
    NOA_INSTANTIATE_SORT_(u32);
    NOA_INSTANTIATE_SORT_(u64);
    NOA_INSTANTIATE_SORT_(f16);
    NOA_INSTANTIATE_SORT_(f32);
    NOA_INSTANTIATE_SORT_(f64);
}
