#pragma once

#include <algorithm>
#include "noa/runtime/core/Access.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/cpu/Allocators.hpp"
#include "noa/runtime/cpu/Iwise.hpp"

namespace noa::cpu::details {
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

    // Sorts the third dimension of "values" using std::sort.
    // Works with non-contiguous strides. If row is non-contiguous, allocates one row.
    // If there's a lot of rows to sort, sort_batched_ may be faster at the cost of more memory allocated.
    template<typename T, usize N>
    void sort_iterative_(T* values, const Strides<isize, N>& strides, const Shape<isize, N>& shape, i32 dim, bool ascending) {
        NOA_ASSERT(strides[dim] > 0); // nothing to sort if dim is broadcast

        Shape<isize, N - 1> shape_;
        Strides<isize, N - 1> strides_;
        i32 count{};
        for (i32 i{}; i < static_cast<i32>(N); ++i) {
            if (i != dim) {
                shape_[count] = shape[i];
                strides_[count] = strides[i];
                ++count;
            }
        }

        const bool dim_is_contiguous = strides[dim] == 1;
        const isize dim_size = shape[dim];
        const isize dim_stride = strides[dim];
        const auto buffer = AllocatorHeap::allocate<T>(dim_is_contiguous ? 0 : dim_size);
        iwise<IwiseConfig<0>>(shape_, [&](const Vec<isize, N - 1>& indices) {
            const auto offset = offset_at(strides_, indices);
            T* values_ptr = values + offset;

            // If row is strided, copy in buffer...
            if (not dim_is_contiguous) {
                for (isize l = 0; l < dim_size; ++l)
                    buffer.get()[l] = values_ptr[l * dim_stride];
                values_ptr = buffer.get();
            }

            if (ascending)
                std::sort(values_ptr, values_ptr + dim_size, std::less{});
            else
                std::sort(values_ptr, values_ptr + dim_size, std::greater{});

            // ... and copy the sorted row back to the original array.
            if (not dim_is_contiguous) {
                for (isize l{}; l < dim_size; ++l, ++values_ptr)
                    values[offset + l * dim_stride] = values_ptr[l];
            }
        });
    }

    // Sort any dimension [0..3] of the input array, in-place.
    // The array can have non-contiguous strides in any dimension.
    // Basically allocates x2 the shape...
    template<typename T, usize N>
    void sort_batched_(T* values, const Strides<isize, N>& strides, const Shape<isize, N>& shape, i32 dim, bool ascending) {
        using keypair_t = KeyValPair<u32, T>;
        auto tile = shape;
        tile[dim] = 1; // mark elements with their original axis.
        const auto tile_strides = tile.strides();
        auto key_val = std::vector<keypair_t>(static_cast<usize>(shape.n_elements()));
        iwise<IwiseConfig<0>>(shape, [&, output = key_val.data()](Vec<isize, N> indices) mutable {
            const isize key = offset_at(tile_strides, indices % tile.vec);
            *output = {
                static_cast<u32>(key),
                values[offset_at(strides, indices)]
            };
            ++output; // SAFETY assumes serial
        });

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
        // TODO check for N < 4
        // dim=3 -> {0,1,2,3} {0,1,2,3}
        // dim=2 -> {0,1,3,2} {1,2,3,0}
        // dim=1 -> {0,2,3,1} {2,3,0,1}
        // dim=0 -> {1,2,3,0} {3,0,1,2}
        auto input_shape = Shape<isize, N>::filled_with(shape[dim]);
        auto permutation = Vec<isize, N>::filled_with(N - 1);
        isize count{};
        for (i32 i{}; i < static_cast<i32>(N); ++i) {
            if (i != dim) {
                input_shape[count] = shape[i];
                permutation[i] = count;
                ++count;
            }
        }

        const auto src = key_val.data();
        const auto dst_shape = input_shape.permute(permutation);
        const auto src_strides_permuted = input_shape.strides().permute(permutation);
        iwise<IwiseConfig<0>>(dst_shape, [&](Vec<isize, N> indices) {
            values[offset_at(strides, indices)] = src[offset_at(src_strides_permuted, indices)].second;
        });
    }
}

namespace noa::cpu {
    template<typename T, usize N>
    void sort(T* array, const Strides<isize, N>& strides, const Shape<isize, N>& shape, bool ascending, i32 dim) {
        if constexpr (N == 1) {
            details::sort_iterative_(array, Strides<isize, 2>{0, strides}, Shape<isize, 2>{1, shape}, 1, ascending);
        } else {
            // If there are not a lot of axes to sort, use the iterative version which uses less memory
            // and does a single sort per axis. Otherwise, use the batched version which uses more memory
            // but uses 2 sorts (1 being a stable sort), and a permutation/copy, for the entire array.
            auto shape_ = shape;
            shape_[dim] = 1;
            const auto iterations = shape_.n_elements();
            if (iterations < 100)
                details::sort_iterative_(array, strides, shape, dim, ascending);
            else
                details::sort_batched_(array, strides, shape, dim, ascending);
        }
    }
}
