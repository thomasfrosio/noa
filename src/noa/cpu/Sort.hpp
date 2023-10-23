#pragma once

#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"
#include "noa/cpu/AllocatorHeap.hpp"

#if defined(NOA_IS_OFFLINE)
#include <algorithm>

namespace noa::cpu::guts::sort {
    template<typename T, typename U>
    using KeyValPair = Pair<T, U>;

    // Ascending order.
    template<bool SORT_VALUES>
    struct IsLess {
        template<typename T, typename U>
        constexpr bool operator()(const KeyValPair<T, U> lhs, const KeyValPair<T, U> rhs) const noexcept {
            if constexpr (SORT_VALUES)
                return lhs.second < rhs.second;
            else
                return lhs.first < rhs.first;
        }
    };

    // Descending order.
    template<bool SORT_VALUES>
    struct IsGreater {
        template<typename T, typename U>
        constexpr bool operator()(const KeyValPair<T, U> lhs, const KeyValPair<T, U> rhs) const noexcept {
            if constexpr (SORT_VALUES)
                return lhs.second > rhs.second;
            else
                return lhs.first > rhs.first;
        }
    };

    // Maybe this could be nice in the main API? Probably too specialized.
    template<typename T>
    void iota_(
            T* input, const Strides4<i64>& strides, const Shape4<i64>& shape,
            const Shape4<i64>& tile, KeyValPair<u32, T>* output
    ) {
        const auto tile_strides = tile.strides();
        for (i64 i = 0; i < shape[0]; ++i) {
            for (i64 j = 0; j < shape[1]; ++j) {
                for (i64 k = 0; k < shape[2]; ++k) {
                    for (i64 l = 0; l < shape[3]; ++l, ++output) {
                        const i64 key = noa::offset_at(
                                i % tile[0], j % tile[1],
                                k % tile[2], l % tile[3],
                                tile_strides);
                        *output = {static_cast<u32>(key),
                                   input[noa::offset_at(i, j, k, l, strides)]};
                    }
                }
            }
        }
    }

    // This is like memory::permute(), but working with the pair as input.
    template<typename T>
    void permute_(
            const Pair<uint, T>* src, const Strides4<i64>& src_strides, const Shape4<i64>& src_shape,
            T* dst, const Strides4<i64>& dst_strides, const Vec4<i64>& permutation
    ) {
        const auto dst_shape = src_shape.reorder(permutation);
        const auto src_strides_permuted = src_strides.reorder(permutation);

        for (i64 i = 0; i < dst_shape[0]; ++i)
            for (i64 j = 0; j < dst_shape[1]; ++j)
                for (i64 k = 0; k < dst_shape[2]; ++k)
                    for (i64 l = 0; l < dst_shape[3]; ++l)
                        dst[noa::offset_at(i, j, k, l, dst_strides)] =
                                src[noa::offset_at(i, j, k, l, src_strides_permuted)].second;
    }

    // Sorts the third dimension of "values" using std::sort.
    // Works with non-contiguous strides. If row is non-contiguous, allocates one row.
    // If there's a lot of rows to sort, sort_batched_ may be faster at the cost of more memory allocated.
    template<typename T>
    void sort_iterative_(T* values, const Strides4<i64>& strides, const Shape4<i64>& shape, i32 dim, bool ascending) {
        NOA_ASSERT(strides[dim] > 0); // nothing to sort if dim is broadcast

        Strides3<i64> shape_;
        Strides3<i64> strides_;
        int count = 0;
        for (int i = 0; i < 4; ++i) {
            if (i != dim) {
                shape_[count] = shape[i];
                strides_[count] = strides[i];
                ++count;
            }
        }

        const bool dim_is_contiguous = strides[dim] == 1;
        const i64 dim_size = shape[dim];
        const i64 dim_stride = strides[dim];

        const auto buffer = AllocatorHeap<T>::allocate(dim_is_contiguous ? 0 : dim_size);
        for (i64 i = 0; i < shape_[0]; ++i) {
            for (i64 j = 0; j < shape_[1]; ++j) {
                for (i64 k = 0; k < shape_[2]; ++k) {

                    const i64 offset = noa::offset_at(i, j, k, strides_);
                    T* values_ptr = values + offset;

                    // If row is strided, copy in buffer...
                    if (!dim_is_contiguous) {
                        for (i64 l = 0; l < dim_size; ++l)
                            buffer.get()[l] = values_ptr[l * dim_stride];
                        values_ptr = buffer.get();
                    }

                    if (ascending)
                        std::sort(values_ptr, values_ptr + dim_size, noa::less_t{});
                    else
                        std::sort(values_ptr, values_ptr + dim_size, noa::greater_t{});

                    // ... and copy the sorted row back to the original array.
                    if (!dim_is_contiguous) {
                        for (i64 l = 0; l < dim_size; ++l, ++values_ptr)
                            values[offset + l * dim_stride] = values_ptr[l];
                    }
                }
            }
        }
    }

    // Sort any dimension [0..3] of the input array, in-place.
    // The array can have non-contiguous strides in any dimension.
    // Basically allocates x2 the shape...
    template<typename T>
    void sort_batched_(T* values, const Strides4<i64>& strides, const Shape4<i64>& shape, i32 dim, bool ascending) {
        using keypair_t = KeyValPair<u32, T>;
        auto tile = shape;
        tile[dim] = 1; // mark elements with their original axis.
        std::vector<keypair_t> key_val(static_cast<size_t>(shape.elements()));
        iota_(values, strides, shape, tile, key_val.data());

        // Sort the entire array based on the values, but update the original indexes.
        if (ascending)
            std::sort(key_val.begin(), key_val.end(), IsLess<true>{});
        else
            std::sort(key_val.begin(), key_val.end(), IsGreater<true>{});

        // Sort the entire array based on the original indexes.
        // The key here is the iota and the stable_sort. Let say the dim to sort is the rows.
        //  1) the iota has a tile of 1 in row dimension, so the elements in the same row have the same index.
        //  2) by sorting the indexes, we replace the elements in their original row, but because we use a stable
        //     sort, we also keep their relative position, thus preserving the order from the first sort.
        std::stable_sort(key_val.begin(), key_val.end(), IsLess<false>{});

        // Now the problem is that key_val is permuted, with the "dim" being the innermost dimension.
        // The relative order of the other dimensions are preserved. In any case, we need to permute
        // it back, so do this while transferring the values back to the original array.

        // Find the permutation from "key_val" to "values":
        // dim=3 -> {0,1,2,3} {0,1,2,3}
        // dim=2 -> {0,1,3,2} {1,2,3,0}
        // dim=1 -> {0,2,3,1} {2,3,0,1}
        // dim=0 -> {1,2,3,0} {3,0,1,2}
        auto input_shape = Shape4<i64>::filled_with(shape[dim]);
        auto permutation = Vec4<i64>::filled_with(3);
        i64 count = 0;
        for (i32 i = 0; i < 4; ++i) {
            if (i != dim) {
                input_shape[count] = shape[i];
                permutation[i] = count;
                ++count;
            }
        }
        permute_(key_val.data(), input_shape.strides(), input_shape, values, strides, permutation);
    }
}

namespace noa::cpu {
    // Sorts an array, in-place.
    template<typename T, typename = std::enable_if_t<nt::is_restricted_scalar_v<T>>>
    void sort(T* array, const Strides4<i64>& strides, const Shape4<i64>& shape, bool ascending, i32 dim) {
        NOA_ASSERT(array && all(shape > 0));

        // Allow dim = -1 to specify the first non-empty dimension in the rightmost order.
        if (dim == -1)
            dim = shape[3] > 1 ? 3 : shape[2] > 1 ? 2 : shape[1] > 1 ? 1 : 0;
        NOA_ASSERT(dim >= 0 && dim <= 3);

        if (strides[dim] == 0)
            return; // there's only one value in the dimension to sort...

        // If there's not a lot of axes to sort, use the iterative version which uses less memory
        // and does a single sort per axis. Otherwise, use the batched version which uses more memory
        // but uses 2 sorts (1 being a stable sort), and a permutation/copy, for the entire array.
        auto shape_4d = shape;
        shape_4d[dim] = 1;
        const auto iterations = shape_4d.elements();
        if (iterations < 100)
            guts::sort::sort_iterative_(array, strides, shape, dim, ascending);
        else
            guts::sort::sort_batched_(array, strides, shape, dim, ascending);
    }
}
#endif
