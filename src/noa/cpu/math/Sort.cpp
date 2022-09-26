#include <algorithm>

#include "noa/common/Functors.h"
#include "noa/cpu/math/Sort.h"
#include "noa/cpu/memory/PtrHost.h"

namespace {
    using namespace ::noa;

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
    void iota_(T* input, dim4_t strides, dim4_t shape, dim4_t tile, KeyValPair<uint, T>* output) {
        const dim4_t tile_strides = tile.strides();
        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {
                    for (dim_t l = 0; l < shape[3]; ++l, ++output) {
                        const dim_t key = indexing::at(i % tile[0],
                                                        j % tile[1],
                                                        k % tile[2],
                                                        l % tile[3],
                                                        tile_strides);
                        *output = {static_cast<uint>(key),
                                   input[indexing::at(i, j, k, l, strides)]};
                    }
                }
            }
        }
    }

    // This is like memory::permute(), but working with the pair as input.
    template<typename T>
    void permute_(const Pair<uint, T>* src, dim4_t src_strides, dim4_t src_shape,
                  T* dst, dim4_t dst_strides, int4_t permutation) {
        const dim4_t dst_shape = indexing::reorder(src_shape, permutation);
        const dim4_t src_strides_permuted = indexing::reorder(src_strides, permutation);

        for (dim_t i = 0; i < dst_shape[0]; ++i)
            for (dim_t j = 0; j < dst_shape[1]; ++j)
                for (dim_t k = 0; k < dst_shape[2]; ++k)
                    for (dim_t l = 0; l < dst_shape[3]; ++l)
                        dst[indexing::at(i, j, k, l, dst_strides)] =
                                src[indexing::at(i, j, k, l, src_strides_permuted)].second;
    }

    // Sorts the third dimension of "values" using std::sort.
    // Works with non-contiguous strides. If row is non-contiguous, allocates one row.
    // If there's a lot of rows to sort, sortBatched_ may be faster at the cost of more memory allocated.
    template<typename T>
    void sortIterative_(T* values, dim4_t strides, dim4_t shape, int dim, bool ascending) {
        NOA_ASSERT(strides[dim] > 0); // nothing to sort if dim is broadcast

        dim3_t shape_;
        dim3_t strides_;
        int count = 0;
        for (int i = 0; i < 4; ++i) {
            if (i != dim) {
                shape_[count] = shape[i];
                strides_[count] = strides[i];
                ++count;
            }
        }

        const bool dim_is_contiguous = strides[dim] == 1;
        const dim_t dim_size = shape[dim];
        const dim_t dim_stride = strides[dim];

        cpu::memory::PtrHost<T> buffer(dim_is_contiguous ? 0 : dim_size);
        for (dim_t i = 0; i < shape_[0]; ++i) {
            for (dim_t j = 0; j < shape_[1]; ++j) {
                for (dim_t k = 0; k < shape_[2]; ++k) {

                    const dim_t offset = indexing::at(i, j, k, strides_);
                    T* values_ = values + offset;

                    // If row is non-contiguous, copy in buffer...
                    if (!dim_is_contiguous) {
                        for (dim_t l = 0; l < dim_size; ++l)
                            buffer[l] = values_[l * dim_stride];
                        values_ = buffer.get();
                    }

                    if (ascending)
                        std::sort(values_, values_ + dim_size, math::less_t{});
                    else
                        std::sort(values_, values_ + dim_size, math::greater_t{});

                    // ... and copy the sorted row back to original array.
                    if (!dim_is_contiguous) {
                        for (dim_t l = 0; l < dim_size; ++l, ++values_)
                            values[offset + l * dim_stride] = values_[l];
                    }
                }
            }
        }
    }

    // Sort any dimension [0..3] of the input array, in-place.
    // The array can have non-contiguous strides in any dimension.
    // Basically allocates x2 the shape...
    template<typename T>
    void sortBatched_(T* values, dim4_t strides, dim4_t shape, int dim, bool ascending) {
        using keypair_t = KeyValPair<uint, T>;
        dim4_t tile = shape;
        tile[dim] = 1; // mark elements with their original axis.
        std::vector<keypair_t> key_val(shape.elements());
        iota_(values, strides, shape, tile, key_val.data());

        // Sort the entire array based on the values, but updates the original indexes.
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
        dim4_t input_shape(shape[dim]);
        int4_t permutation(3);
        int count = 0;
        for (int i = 0; i < 4; ++i) {
            if (i != dim) {
                input_shape[count] = shape[i];
                permutation[i] = count;
                ++count;
            }
        }
        permute_(key_val.data(), input_shape.strides(), input_shape, values, strides, permutation);
    }
}

namespace noa::cpu::math {
    template<typename T, typename>
    void sort(const shared_t<T[]>& array, dim4_t strides, dim4_t shape, bool ascending, int dim, Stream& stream) {
        // Allow dim = -1 to specify the first non-empty dimension in the rightmost order.
        if (dim == -1)
            dim = shape[3] > 1 ? 3 : shape[2] > 1 ? 2 : shape[1] > 1 ? 1 : 0;
        NOA_ASSERT(dim >= 0 && dim <= 3);

        if (strides[dim] == 0)
            return; // there's one value in the dimension to sort...

        // If there's not a lot of axes to sort, use the iterative version which uses less memory
        // and does a single sort per axis. Otherwise, use the batched version which uses more memory
        // but uses 2 sorts (1 being a stable sort), and a permutation/copy, for the entire array.
        stream.enqueue([=]() {
            dim4_t shape_ = shape;
            shape_[dim] = 1;
            const dim_t iterations = shape_.elements();
            if (iterations < 100)
                sortIterative_(array.get(), strides, shape, dim, ascending);
            else
                sortBatched_(array.get(), strides, shape, dim, ascending);
        });
    }

    #define NOA_INSTANTIATE_SORT_(T) \
    template void sort<T,void>(const shared_t<T[]>&, dim4_t, dim4_t, bool, int, Stream&)

    NOA_INSTANTIATE_SORT_(bool);
    NOA_INSTANTIATE_SORT_(int8_t);
    NOA_INSTANTIATE_SORT_(int16_t);
    NOA_INSTANTIATE_SORT_(int32_t);
    NOA_INSTANTIATE_SORT_(int64_t);
    NOA_INSTANTIATE_SORT_(uint8_t);
    NOA_INSTANTIATE_SORT_(uint16_t);
    NOA_INSTANTIATE_SORT_(uint32_t);
    NOA_INSTANTIATE_SORT_(uint64_t);
    NOA_INSTANTIATE_SORT_(half_t);
    NOA_INSTANTIATE_SORT_(float);
    NOA_INSTANTIATE_SORT_(double);
}
