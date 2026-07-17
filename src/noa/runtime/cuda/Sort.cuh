#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/runtime/core/Iwise.hpp"
#include "noa/runtime/cpu/Iwise.hpp"
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Copy.cuh"
#include "noa/runtime/cuda/Iwise.cuh"
#include "noa/runtime/cuda/Stream.hpp"

// Use cub to do the actual sorting. Thrust seems to have its own merge sort, but its main API does not allow you to
// choose between radix and merge sort. However, it seems that it always selects radix if the type and comparison
// operator are supported by the radix sort, i.e. base types and < or > comparison. In our case, that's all we need,
// so use the radix sort for everything...
#include <cub/device/device_radix_sort.cuh>

namespace noa::cuda::details {
    template<typename T>
    cudaError cub_radix_sort_keys_(
        void* temp_storage, usize& temp_storage_bytes,
        cub::DoubleBuffer<T>& keys,
        i32 size, bool ascending, Stream& stream
    ) {
        // f16 can be reinterpreted to CUDA's __half.
        using cub_t = std::conditional_t<std::is_same_v<T, f16>, __half, T>;

        if (ascending) {
            return cub::DeviceRadixSort::SortKeys(
                temp_storage, temp_storage_bytes,
                reinterpret_cast<cub::DoubleBuffer<cub_t>&>(keys),
                size, 0, static_cast<i32>(sizeof(T) * 8),
                stream.id());
        } else {
            return cub::DeviceRadixSort::SortKeysDescending(
                temp_storage, temp_storage_bytes,
                reinterpret_cast<cub::DoubleBuffer<cub_t>&>(keys),
                size, 0, static_cast<i32>(sizeof(T) * 8),
                stream.id());
        }
    }

    template<typename T, typename U>
    cudaError cub_radix_sort_pairs_(
        void* temp_storage, usize& temp_storage_bytes,
        cub::DoubleBuffer<T>& keys, cub::DoubleBuffer<U>& values,
        i32 size, bool ascending, Stream& stream
    ) {
        // f16 can be reinterpreted to CUDA's __half.
        using cub_t = std::conditional_t<std::is_same_v<T, f16>, __half, T>;
        using cub_u = std::conditional_t<std::is_same_v<U, f16>, __half, U>;

        if (ascending) {
            return cub::DeviceRadixSort::SortPairs(
                temp_storage, temp_storage_bytes,
                reinterpret_cast<cub::DoubleBuffer<cub_t>&>(keys),
                reinterpret_cast<cub::DoubleBuffer<cub_u>&>(values),
                size, 0, static_cast<i32>(sizeof(T) * 8),
                stream.id());
        } else {
            return cub::DeviceRadixSort::SortPairsDescending(
                temp_storage, temp_storage_bytes,
                reinterpret_cast<cub::DoubleBuffer<cub_t>&>(keys),
                reinterpret_cast<cub::DoubleBuffer<cub_u>&>(values),
                size, 0, static_cast<i32>(sizeof(T) * 8),
                stream.id());
        }
    }

    // Sorts the third dimension of "values" using cub radix sort.
    // Works with non-contiguous strides. If dim is non-contiguous, allocates one row.
    // If there's a lot of rows to sort, sort_batched_ should be faster.
    template<typename T, usize N>
    void sort_iterative_(
        T* values, const Strides<isize, N>& strides, const Shape<isize, N>& shape,
        i32 dim, bool ascending, Stream& stream
    ) {
        const bool dim_is_contiguous = strides[dim] == 1;
        const auto dim_size = safe_cast<i32>(shape[dim]);
        const auto dim_shape = Shape{shape[dim]};
        const auto dim_strides = Strides{strides[dim]};

        // Prepare the alternate buffer.
        // TODO Do one single allocation for buffer(s) and tmp storage. Problem is the alignment?
        using unique_t = AllocatorDevice::allocate_type<T>;
        unique_t key_buffer;
        unique_t key_buffer_alt;
        if (dim_is_contiguous) {
            key_buffer = nullptr;
            key_buffer_alt = AllocatorDevice::allocate_async<T>(dim_size, stream);
        } else {
            key_buffer = AllocatorDevice::allocate_async<T>(dim_size, stream);
            key_buffer_alt = AllocatorDevice::allocate_async<T>(dim_size, stream);
        }
        auto keys = cub::DoubleBuffer<T>(key_buffer.get(), key_buffer_alt.get());

        // Allocates for the small tmp storage.
        usize temp_storage_bytes{};
        check(cub_radix_sort_keys_<T>(nullptr, temp_storage_bytes, keys, dim_size, ascending, stream));
        const auto temp_storage = AllocatorDevice::allocate_async<Byte>(
            static_cast<isize>(temp_storage_bytes), stream);

        // Prepare the iterations.
        Shape<isize, N - 1> iter_shape;
        Strides<isize, N - 1> iter_strides;
        i32 count = 0;
        for (usize i{}; i < N; ++i) {
            if (i != static_cast<usize>(dim)) {
                iter_shape[count] = shape[i];
                iter_strides[count] = strides[i];
                ++count;
            }
        }

        // Sort the axis.
        noa::cpu::iwise<noa::cpu::IwiseConfig<0>>(iter_shape, [&](const Vec<isize, N - 1>& indices) {
            T* values_iter = values + offset_at(iter_strides, indices);

            // (Re)set the buffers.
            keys.selector = 0;
            if (dim_is_contiguous) {
                keys.d_buffers[0] = values_iter;
            } else {
                copy(values_iter, dim_strides,
                     key_buffer.get(), Strides<isize, 1>{1},
                     dim_shape, stream);
            }

            check(cub_radix_sort_keys_<T>(
                temp_storage.get(), temp_storage_bytes, keys,
                dim_size, ascending, stream));

            if (dim_is_contiguous) {
                if (keys.selector != 0) {
                    // Unfortunately, the results are in the alternate buffer,
                    // so copy it back to the original array.
                    copy(key_buffer_alt.get(), Strides<isize, 1>{1},
                         values_iter /* or key_buffer */, dim_strides,
                         dim_shape, stream);
                }
            } else {
                copy(keys.selector == 0 ? key_buffer.get() : key_buffer_alt.get(),
                     Strides<isize, 1>{1},
                     values_iter, dim_strides,
                     dim_shape, stream);
            }
        });
    }

    // Sort any dimension [0..N-1] of the input array, in-place.
    // The array can have non-contiguous strides in any dimension.
    // Basically allocates 3 to 4 times the shape...
    template<typename T, usize N>
    void sort_batched_(
        T* values, const Strides<isize, N>& strides, const Shape<isize, N>& shape,
        i32 dim, bool ascending, Stream& stream
    ) {
        const bool is_contiguous = strides.is_contiguous(shape);
        const auto n_elements = safe_cast<i32>(shape.n_elements());
        const auto shape_i32 = shape.template as<i32>();

        // Prepare the keys.
        const auto key_buffer = AllocatorDevice::allocate_async<u32>(n_elements, stream);
        const auto key_buffer_alt = AllocatorDevice::allocate_async<u32>(n_elements, stream);
        auto tile = shape_i32.vec;
        tile[dim] = 1; // mark elements with their original line.
        iwise(shape_i32, nd::Iota(AccessorContiguous<u32, N, i32>(key_buffer.get(), shape_i32.strides()), shape_i32, tile), stream);

        // Prepare the values.
        using unique_t = AllocatorDevice::allocate_type<T>;
        unique_t val_buffer;
        unique_t val_buffer_alt;
        T* val_ptr;
        if (is_contiguous) {
            val_ptr = values;
            val_buffer_alt = AllocatorDevice::allocate_async<T>(n_elements, stream);
        } else {
            val_buffer = AllocatorDevice::allocate_async<T>(n_elements, stream);
            val_ptr = val_buffer.get();
            val_buffer_alt = AllocatorDevice::allocate_async<T>(n_elements, stream);
            copy(values, strides, val_ptr, shape.strides(), shape, stream);
        }

        // Gather them in the cub interface.
        auto cub_keys = cub::DoubleBuffer<u32>(key_buffer.get(), key_buffer_alt.get());
        auto cub_values = cub::DoubleBuffer<T>(val_ptr, val_buffer_alt.get());

        // Allocates for the small tmp storage.
        // The documentation says this should be a small value and is relative to the input size.
        usize tmp_bytes0{}, tmp_bytes1{};
        const cudaError err0 = cub_radix_sort_pairs_<u32, T>(
            nullptr, tmp_bytes0, cub_keys, cub_values, n_elements, ascending, stream);
        const cudaError err1 = cub_radix_sort_pairs_<T, u32>(
            nullptr, tmp_bytes1, cub_values, cub_keys, n_elements, ascending, stream);
        check(err0 == cudaSuccess and err1 == cudaSuccess,
              "Could not find temporary allocation size. 0=({}), 1=({})",
              error2string(err0), error2string(err1));

        tmp_bytes0 = std::max(tmp_bytes0, tmp_bytes1);
        const auto tmp = AllocatorDevice::allocate_async<Byte>(static_cast<isize>(tmp_bytes0), stream);

        // Sort the entire array based on the values, but updates the original indexes.
        // It is important that the second sort is stable, which is the case with radix sort.
        check(cub_radix_sort_pairs_<T, u32>(tmp.get(), tmp_bytes0, cub_values, cub_keys, n_elements, ascending, stream));
        check(cub_radix_sort_pairs_<u32, T>(tmp.get(), tmp_bytes0, cub_keys, cub_values, n_elements, true, stream));

        // Then permute it back to the original order.
        // Find the permutation from "key_val" to "values":
        auto input_shape = Shape<isize, N>::from_value(shape[dim]);
        auto permutation = Vec<i32, N>::from_value(N - 1);
        i32 count = 0;
        for (i32 i = 0; i < static_cast<i32>(N); ++i) {
            if (i != dim) {
                input_shape[count] = shape[i];
                permutation[i] = count;
                ++count;
            }
        }

        // Permutation. Since we do 2 sorts, it seems that the result is always at position 0. If "values" was
        // contiguous, it means the result is already in "values" but with a possible permutation. In this case,
        // we have to permute in the alternate buffer and then copy the result back to "values"...
        const auto current_strides_permuted = input_shape.strides().permute(permutation);
        if (values == cub_values.Current() and permutation != Vec<i32, N>::arange()) {
            copy(values, current_strides_permuted, val_buffer_alt.get(), shape.strides(), shape, stream);
            copy(val_buffer_alt.get(), shape.strides(), values, strides, shape, stream);
        } else {
            copy(cub_values.selector == 0 ? val_ptr : val_buffer_alt.get(), current_strides_permuted,
                 values, strides, shape, stream);
        }
    }
}

namespace noa::cuda {
    template<typename T, usize N>
    void sort(
        T* array, const Strides<isize, N>& strides, const Shape<isize, N>& shape,
        bool ascending, i32 dim, Stream& stream
    ) {
        if constexpr (N == 1) {
            details::sort_iterative_(array, Strides<isize, 2>{0, strides[0]}, Shape<isize, 2>{1, shape[0]}, 1, ascending, stream);
        } else {
            // If there's not a lot of lines to sort, use the iterative version which uses less memory
            // and does a single sort per line. Otherwise, use the batched version which uses more memory
            // but uses 2 sorts (1 being a stable sort), and a possible permutation, for the entire array.
            auto shape_nd = shape;
            shape_nd[dim] = 1;
            const auto n_iterations = shape_nd.n_elements();
            if (n_iterations < 10)
                details::sort_iterative_(array, strides, shape, dim, ascending, stream);
            else
                details::sort_batched_(array, strides, shape, dim, ascending, stream);
        }
    }
}
