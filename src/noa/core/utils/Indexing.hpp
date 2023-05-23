#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/types/Constants.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Mat.hpp"
#include "noa/core/utils/ClampCast.hpp"
#include "noa/core/utils/SafeCast.hpp"

namespace noa::indexing {
    // In Debug mode, assertions are added to check for negative indexes
    // and if the cast to the offset type is bound safe. However, it doesn't
    // check for integer overflow.

    /// Returns the memory offset corresponding to the given 4D indexes.
    /// \param i0,i1,i2,i3  Multi-dimensional indexes.
    /// \param strides      Strides associated with these indexes.
    template<typename I0, typename I1, typename I2, typename I3, typename Offset,
             typename std::enable_if_t<traits::are_int_v<I0, I1, I2, I3>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto at(
            I0 i0, I1 i1, I2 i2, I3 i3,
            const Strides4<Offset>& strides
    ) noexcept -> Offset {
        static_assert(sizeof(Offset) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        NOA_ASSERT(is_safe_cast<Offset>(i0) && is_safe_cast<Offset>(i1) &&
                   is_safe_cast<Offset>(i2) && is_safe_cast<Offset>(i3));

        return static_cast<Offset>(i0) * strides[0] +
               static_cast<Offset>(i1) * strides[1] +
               static_cast<Offset>(i2) * strides[2] +
               static_cast<Offset>(i3) * strides[3];
    }

    /// Returns the memory offset corresponding to the given 4D indexes.
    template<typename Index, typename Offset>
    [[nodiscard]] NOA_FHD constexpr auto at(const Vec4<Index>& index, const Strides4<Offset>& strides) noexcept {
        return at(index[0], index[1], index[2], index[3], strides);
    }

    /// Returns the memory offset corresponding to the given 3D indexes.
    /// \param i0,i1,i2     Multi-dimensional indexes.
    /// \param strides      Strides associated with these indexes. Only the first 3 values are used.
    template<typename I0, typename I1, typename I2, typename Offset, size_t N,
             typename std::enable_if_t<traits::are_int_v<I0, I1, I2> && (N >= 3), bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto at(
            I0 i0, I1 i1, I2 i2,
            const Strides<Offset, N>& strides)
    noexcept -> Offset {
        static_assert(sizeof(Offset) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        NOA_ASSERT(is_safe_cast<Offset>(i0) && is_safe_cast<Offset>(i1) && is_safe_cast<Offset>(i2));

        return static_cast<Offset>(i0) * strides[0] +
               static_cast<Offset>(i1) * strides[1] +
               static_cast<Offset>(i2) * strides[2];
    }

    /// Returns the memory offset corresponding to the given 3D indexes.
    template<typename Index, typename Offset, size_t N,
             typename std::enable_if_t<traits::is_int_v<Index> && (N >= 3), bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto at(const Vec3<Index>& index, const Strides<Offset, N>& strides) noexcept {
        return at(index[0], index[1], index[2], strides);
    }

    /// Returns the memory offset corresponding to the given 2D indexes.
    /// \param i0,i1        Multi-dimensional indexes.
    /// \param strides      Strides associated with these indexes. Only the first 2 values are used.
    template<typename I0, typename I1, typename Offset, size_t N,
            typename std::enable_if_t<traits::are_int_v<I0, I1> && (N >= 2), bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto at(
            I0 i0, I1 i1,
            const Strides<Offset, N>& strides)
    noexcept -> Offset {
        static_assert(sizeof(Offset) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        NOA_ASSERT(is_safe_cast<Offset>(i0) && is_safe_cast<Offset>(i1));

        return static_cast<Offset>(i0) * strides[0] +
               static_cast<Offset>(i1) * strides[1];
    }

    /// Returns the memory offset corresponding to the given 3D indexes.
    template<typename Index, typename Offset, size_t N,
            typename std::enable_if_t<traits::is_int_v<Index> && (N >= 2), bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto at(const Vec2<Index>& index, const Strides<Offset, N>& strides) noexcept {
        return at(index[0], index[1], strides);
    }

    /// Returns the memory offset corresponding to the given 1D indexes.
    /// \param i0           Index.
    /// \param strides      Strides associated with these indexes. Only the first value is used.
    template<typename Index, typename Strides,
             typename std::enable_if_t<
                     traits::is_int_v<Index> &&
                     (traits::is_stridesX_v<Strides> || traits::is_int_v<Strides>), bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto at(Index i0, Strides strides) noexcept {
        using offset_t = traits::value_type_t<Strides>;
        static_assert(sizeof(offset_t) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        NOA_ASSERT(is_safe_cast<offset_t>(i0));

        if constexpr (traits::is_int_v<Strides>) {
            return static_cast<offset_t>(i0) * strides;
        } else {
            return static_cast<offset_t>(i0) * strides[0];
        }
    }

    /// Returns the memory offset corresponding to the given 3D indexes.
    template<typename Index, typename Offset, size_t N,
             typename std::enable_if_t<traits::is_int_v<Index> && (N >= 1), bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto at(const Vec1<Index>& index, const Strides<Offset, N>& strides) noexcept {
        return at(index[0], strides);
    }

    /// If \p idx is out-of-bound, computes a valid index, i.e. [0, size-1], according to \p MODE.
    /// Otherwise, returns \p idx. \p size should be > 0.
    template<BorderMode MODE, typename SInt, typename = std::enable_if_t<std::is_signed_v<SInt>>>
    [[nodiscard]] NOA_IHD constexpr SInt at(SInt idx, SInt size) {
        static_assert(MODE == BorderMode::CLAMP || MODE == BorderMode::PERIODIC ||
                      MODE == BorderMode::MIRROR || MODE == BorderMode::REFLECT);
        NOA_ASSERT(size > 0);

        // a % b == a - b * (a / b) == a + b * (-a / b)
        // Having a < 0 is well-defined since C++11.
        if constexpr (MODE == BorderMode::CLAMP) {
            if (idx < 0)
                idx = 0;
            else if (idx >= size)
                idx = size - 1;
        } else if constexpr (MODE == BorderMode::PERIODIC) {
            // 0 1 2 3 0 1 2 3 0 1 2 3 |  0 1 2 3  | 0 1 2 3 0 1 2 3 0 1 2 3
            SInt rem = idx % size; // FIXME maybe enclose this, at the expense of two jumps?
            idx = rem < 0 ? rem + size : rem;
        } else if constexpr (MODE == BorderMode::MIRROR) {
            // 0 1 2 3 3 2 1 0 0 1 2 3 3 2 1 0 |  0 1 2 3  | 3 2 1 0 0 1 2 3 3 2 1 0
            if (idx < 0)
                idx = -idx - 1;
            if (idx >= size) {
                SInt period = 2 * size;
                idx %= period;
                if (idx >= size)
                    idx = period - idx - 1;
            }
        } else if constexpr (MODE == BorderMode::REFLECT) {
            // 0 1 2 3 2 1 0 1 2 3 2 1 |  0 1 2 3  | 2 1 0 1 2 3 2 1 0
            if (idx < 0)
                idx = -idx;
            if (idx >= size) {
                SInt period = 2 * size - 2;
                idx %= period;
                if (idx >= size)
                    idx = period - idx;
            }
        }
        return idx;
    }
}

namespace noa::indexing {
    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr bool is_rightmost(const Strides<T, N>& strides) {
        return strides.is_rightmost();
    }

    /// Whether \p strides and \p shape describe a contiguous array.
    /// \tparam ORDER 'C' for C-contiguous (row-major) or 'F' for F-contiguous (column-major).
    /// \note Empty dimensions are contiguous by definition since their strides are not used.
    ///       Broadcast dimensions are NOT contiguous.
    template<char ORDER = 'C', typename T>
    [[nodiscard]] NOA_FHD constexpr bool are_contiguous(const Strides4<T>& strides, const Shape4<T>& shape) {
        if (noa::any(shape == 0)) // guard against empty array
            return false;

        if constexpr (ORDER == 'c' || ORDER == 'C') {
            return (shape[0] == 1 || strides[0] == shape[3] * shape[2] * shape[1]) &&
                   (shape[1] == 1 || strides[1] == shape[3] * shape[2]) &&
                   (shape[2] == 1 || strides[2] == shape[3]) &&
                   (shape[3] == 1 || strides[3] == 1);

        } else if constexpr (ORDER == 'f' || ORDER == 'F') {
            return (shape[0] == 1 || strides[0] == shape[3] * shape[2] * shape[1]) &&
                   (shape[1] == 1 || strides[1] == shape[3] * shape[2]) &&
                   (shape[2] == 1 || strides[2] == 1) &&
                   (shape[3] == 1 || strides[3] == shape[2]);
        } else {
            static_assert(traits::always_false_v<T>);
        }
    }

    /// For each dimension, check if it is contiguous.
    /// \details If one wants to know in which dimension the contiguity is broken or if the contiguity is only
    ///          required in a particular dimension, this function can be useful. It supports broadcasting and
    ///          empty dimensions. It is also guarded against empty shapes.
    /// \tparam ORDER 'C' for C-contiguous (row-major) or 'F' for F-contiguous (column-major).
    /// \note If one want to check whether the array is contiguous or not, while all(is_contiguous(...)) is valid,
    ///       are_contiguous(...) is preferred simply because it is clearer and slightly more efficient.
    /// \note Broadcast dimensions are NOT contiguous. Only empty dimensions are treated as contiguous
    ///       regardless of their stride. Functions that require broadcast dimensions to be "contiguous"
    ///       should call effective_shape() first, to "cancel" the broadcasting and mark the dimension as empty.
    template<char ORDER = 'C', typename T>
    [[nodiscard]] NOA_IHD constexpr auto is_contiguous(Strides4<T> strides, const Shape4<T>& shape) {
        if (any(shape == 0)) // guard against empty array
            return Vec4<bool>{0, 0, 0, 0};

        strides *= Strides4<T>(shape != 1); // mark the stride of empty dimensions unusable

        if constexpr (ORDER == 'c' || ORDER == 'C') {
            // If dimension is broadcast or empty, we cannot use the stride
            // and need to use the corrected stride one dimension up.
            const auto corrected_stride_2 = strides[3] ? shape[3] * strides[3] : 1;
            const auto corrected_stride_1 = strides[2] ? shape[2] * strides[2] : corrected_stride_2;
            const auto corrected_stride_0 = strides[1] ? shape[1] * strides[1] : corrected_stride_1;

            // If dimension is empty, it is by definition contiguous and its stride does not matter.
            // Broadcast dimensions break the contiguity because the corrected_stride cannot be 0.
            // This is true for empty dimensions, but empty dimensions are contiguous by definition
            // and their strides do not matter, i.e. we skip the comparison.
            return Vec4<bool>{shape[0] == 1 || strides[0] == corrected_stride_0,
                              shape[1] == 1 || strides[1] == corrected_stride_1,
                              shape[2] == 1 || strides[2] == corrected_stride_2,
                              shape[3] == 1 || strides[3] == 1};

        } else if constexpr (ORDER == 'f' || ORDER == 'F') {
            const auto corrected_stride_3 = strides[2] ? shape[2] * strides[2] : 1;
            const auto corrected_stride_1 = strides[3] ? shape[3] * strides[3] : corrected_stride_3;
            const auto corrected_stride_0 = strides[1] ? shape[1] * strides[1] : corrected_stride_1;

            return Vec4<bool>{shape[0] == 1 || strides[0] == corrected_stride_0,
                              shape[1] == 1 || strides[1] == corrected_stride_1,
                              shape[2] == 1 || strides[2] == 1,
                              shape[3] == 1 || strides[3] == corrected_stride_3};

        } else {
            static_assert(traits::always_false_v<T>);
        }
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr bool is_vector(const Shape<T, 4>& shape, bool can_be_batched = false) {
        return shape.is_vector(can_be_batched);
    }

    template<typename T, size_t N, typename = std::enable_if_t<N <= 3>>
    [[nodiscard]] NOA_FHD constexpr bool is_vector(const Shape<T, N>& shape) {
        return shape.is_vector();
    }

    /// Returns the effective shape: if a dimension has a stride of 0, the effective size is 1 (empty dimension).
    template<typename T, typename U, size_t N>
    [[nodiscard]] NOA_IH constexpr auto effective_shape(Shape<T, N> shape, const Strides<U, N>& strides) noexcept {
        for (size_t i = 0; i < N; ++i)
            shape[i] = strides[i] ? shape[i] : 1;
        return shape;
    }

    /// Returns the index of the first non-empty dimension, excluding the batch dimension, going from left to right.
    /// If all dimensions are empty, the index of the width dimension is returned, ie 3.
    template<typename Index>
    [[nodiscard]] NOA_IHD constexpr Index non_empty_dhw_dimension(const Shape4<Index>& shape) noexcept {
        for (Index i = 1; i < 4; ++i)
            if (shape[i] > 1)
                return i;
        return 3;
    }

    /// Returns the order the dimensions should be sorted so that they are in the rightmost order.
    /// The input dimensions have the following indexes: {0, 1, 2, 3}.
    /// For instance, if \p strides is representing a F-contiguous order, this function returns {0, 1, 3, 2}.
    /// Empty dimensions are pushed to the left side (the outermost side) and their strides are ignored.
    /// This is mostly intended to find the fastest way through an array using nested loops in the rightmost order.
    /// \see indexing::reorder(...).
    template<typename Int, size_t N>
    [[nodiscard]] NOA_IHD constexpr auto order(Strides<Int, N> strides, const Shape<Int, N>& shape) {
        Vec<Int, N> order;
        for (size_t i = 0; i < N; ++i) {
            order[i] = static_cast<Int>(i);
            strides[i] = shape[i] <= 1 ? noa::math::Limits<Int>::max() : strides[i];
        }

        return noa::stable_sort(order, [&](Int a, Int b) {
            return strides[a] > strides[b];
        });
    }

    /// Returns the order the dimensions should be sorted so that empty dimensions are on the left.
    /// The input dimensions have the following indexes: {0, 1, 2, 3}.
    /// Coupled with indexing::reorder(), this effectively pushes all zeros and ones in \p shape to the left.
    /// The difference with indexing::order() is that this function does not change the order of the non-empty
    /// dimensions relative to each other. Note that the order of the empty dimensions is preserved.
    template<typename Int, size_t N>
    [[nodiscard]] NOA_HD constexpr auto squeeze(const Shape<Int, N>& shape) {
        Vec<Int, N> order{};
        int idx = N - 1;
        for (int i = idx; i >= 0; --i) {
            if (shape[i] > 1)
                order[idx--] = i;
            else
                order[idx - i] = i;
        }
        // The empty dimensions are entered in descending order.
        // To put them back to the original ascending order, we can do:
        if (idx) {
            if (N >= 2 && idx == 1)
                small_stable_sort<2>(order.data());
            if (N >= 3 && idx == 2)
                small_stable_sort<3>(order.data());
            else if (N >= 4 && idx == 3)
                small_stable_sort<4>(order.data());
        }
        return order;
    }

    /// Reorder (i.e. sort) \p vector according to the indexes in \p order.
    template<typename VecLike, typename Int, size_t N,
            typename std::enable_if_t<noa::traits::is_int_v<Int> &&
                                      (noa::traits::is_vecN_v<VecLike, N> ||
                                       noa::traits::is_shapeN_v<VecLike, N> ||
                                       noa::traits::is_stridesN_v<VecLike, N>), bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto reorder(VecLike vector, const Vec<Int, N>& order) {
        return vector.reorder(order);
    }

    /// Returns the reordered matrix according to the indexes in \p order.
    /// The columns are reordered, and then the rows. This can be useful to swap the axes of a matrix.
    /// \param[in] matrix   Square and (truncated) affine matrix to reorder.
    /// \param[in] order    Order of indexes. Should have the same number of elements as the matrices are rows.
    template<typename Matrix, typename Int, size_t N,
            typename std::enable_if_t<traits::is_matXX_v<Matrix> && Matrix::ROWS == N, bool> = true>
    [[nodiscard]] NOA_FHD constexpr Matrix reorder(const Matrix& matrix, const Vec<Int, N>& order) {
        Matrix reordered_matrix;
        for (size_t row = 0; row < N; ++row) {
            using row_t = typename Matrix::row_type;
            row_t reordered_row;
            for (size_t column = 0; column < N; ++column)
                reordered_row[column] = matrix[row][order[column]];
            reordered_matrix[order[row]] = reordered_row;
        }
        return reordered_matrix;
    }

    /// (Circular) shifts \p v by a given amount.
    /// If \p shift is positive, shifts to the right, otherwise, shifts to the left.
    template<typename VecLike, typename Int, size_t N,
            typename std::enable_if_t<noa::traits::is_int_v<Int> &&
                                      (noa::traits::is_vecN_v<VecLike, N> ||
                                       noa::traits::is_shapeN_v<VecLike, N> ||
                                       noa::traits::is_stridesN_v<VecLike, N>), bool> = true>
    [[nodiscard]] NOA_FHD constexpr auto circular_shift(VecLike vector, const Vec<Int, N>& order) {
        return vector.circular_shift(order);
    }

    /// Whether \p strides describes a column-major layout.
    /// Note that this functions assumes BDHW order, where H is the number of rows and W is the number of columns.
    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr bool is_column_major(const Strides<T, N>& strides) {
        constexpr size_t COL = N - 2;
        constexpr size_t ROW = N - 1;
        return strides[COL] <= strides[ROW];
    }

    /// Whether \p strides describes a column-major layout.
    /// This function effectively squeezes the shape before checking the order.
    /// Furthermore, strides of empty dimensions are ignored and are contiguous by definition.
    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr bool is_column_major(const Strides<T, N>& strides, const Shape<T, N>& shape) {
        int second{-1}, first{-1};
        for (int i = N - 1; i >= 0; --i) {
            if (shape[i] > 1) {
                if (first == -1)
                    first = i;
                else if (second == -1)
                    second = i;
            }
        }
        return second == -1 || first == -1 || strides[second] <= strides[first];
    }

    /// Whether \p strides describes a row-major layout.
    /// Note that this functions assumes BDHW order, where H is the number of rows and W is the number of columns.
    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr bool is_row_major(const Strides<T, N>& strides) {
        constexpr size_t COL = N - 2;
        constexpr size_t ROW = N - 1;
        return strides[COL] >= strides[ROW];
    }

    /// Whether \p strides describes a row-major layout.
    /// This function effectively squeezes the shape before checking the order.
    /// Furthermore, strides of empty dimensions are ignored and are contiguous by definition.
    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr bool is_row_major(const Strides<T, N>& strides, const Shape<T, N>& shape) {
        int second{-1}, first{-1};
        for (int i = N - 1; i >= 0; --i) {
            if (shape[i] > 1) {
                if (first == -1)
                    first = i;
                else if (second == -1)
                    second = i;
            }
        }
        return second == -1 || first == -1 || strides[second] >= strides[first];
    }

    /// Sets the input stride so that the input can be iterated as if it as the same size as the output.
    /// \param input_size           Size of the input. Should correspond to \p output_size or be 1.
    /// \param[out] input_stride    Input stride. If broadcast, it is set to 0.
    /// \param output_size          Size of the output.
    /// \return Whether the input and output size are compatible.
    template<typename Int>
    [[nodiscard]] NOA_IH constexpr bool broadcast(Int input_size, Int& input_stride, Int output_size) noexcept {
        if (input_size == 1 && output_size != 1)
            input_stride = 0; // broadcast this dimension
        else if (input_size != output_size)
            return false; // dimension sizes don't match
        return true;
    }

    /// Sets the input strides so that the input can be iterated as if it as the same shape as the output.
    /// \param input_shape          Shape of the input. Each dimension should correspond to \p output_shape or be 1.
    /// \param[out] input_strides   Input strides. Strides in dimensions that need to be broadcast are set to 0.
    /// \param output_shape         Shape of the output.
    /// \return Whether the input and output shape are compatible.
    template<typename T, size_t N>
    [[nodiscard]] NOA_IH constexpr bool broadcast(
            const Shape<T, N>& input_shape,
            Strides<T, N>& input_strides,
            const Shape<T, N>& output_shape
    ) noexcept {
        for (size_t i = 0; i < N; ++i) {
            if (input_shape[i] == 1 && output_shape[i] != 1)
                input_strides[i] = 0; // broadcast this dimension
            else if (input_shape[i] != output_shape[i])
                return false; // dimension sizes don't match
        }
        return true;
    }

    /// Computes the new strides of an array after reshaping.
    /// \param old_shape        Old shape. An empty shape (dimension of 0) returns false.
    /// \param old_strides      Old strides.
    /// \param new_shape        New shape.
    /// \param[out] new_strides New strides.
    /// \return Whether the input and output shape and strides are compatible.
    ///         If false, \p new_strides is left in an undefined state.
    /// \note Zero strides are allowed.
    template<typename T, size_t OldN, size_t NewN>
    [[nodiscard]] NOA_IH constexpr bool reshape(
            Shape<T, OldN> old_shape, Strides<T, OldN> old_strides,
            Shape<T, NewN> new_shape, Strides<T, NewN>& new_strides
    ) noexcept {
        // from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/TensorUtils.cpp
        if (!old_shape.elements())
            return false;

        auto view_d = static_cast<int64_t>(NewN) - 1;
        T chunk_base_strides = old_strides[OldN - 1];
        T tensor_numel = 1;
        T view_numel = 1;
        for (int64_t tensor_d = static_cast<int64_t>(OldN) - 1; tensor_d >= 0; --tensor_d) {
            tensor_numel *= old_shape[tensor_d];
            // if end of tensor size chunk, check view
            if ((tensor_d == 0) ||
                (old_shape[tensor_d - 1] != 1 && old_strides[tensor_d - 1] != tensor_numel * chunk_base_strides)) {
                while (view_d >= 0 && (view_numel < tensor_numel || new_shape[view_d] == 1)) {
                    new_strides[view_d] = view_numel * chunk_base_strides;
                    view_numel *= new_shape[view_d];
                    --view_d;
                }

                if (view_numel != tensor_numel)
                    return false;
                if (tensor_d > 0) {
                    chunk_base_strides = old_strides[tensor_d - 1];
                    tensor_numel = 1;
                    view_numel = 1;
                }
            }
        }
        return view_d == -1;
    }
}


namespace noa::indexing {
    template<typename Int>
    [[nodiscard]] auto extract_matmul_layout(
            const Strides4<Int>& lhs_strides, const Shape4<Int>& lhs_shape,
            const Strides4<Int>& rhs_strides, const Shape4<Int>& rhs_shape,
            const Strides4<Int>& output_strides, const Shape4<Int>& output_shape,
            bool lhs_transpose, bool rhs_transpose
    ) -> std::tuple<Shape3<Int>, Strides3<Int>, bool> {

        // First extract and check the shape: MxK @ KxN = MxN
        const auto m = lhs_shape[2 + lhs_transpose];
        const auto n = rhs_shape[3 - rhs_transpose];
        const auto k = lhs_shape[3 - lhs_transpose];
        NOA_CHECK(lhs_shape[1] == 1 && rhs_shape[1] == 1 && output_shape[1] == 1,
                  "Only 2D matrices are supported, but got shape lhs={}, rhs={} and output={}",
                  lhs_shape, rhs_shape, output_shape);
        NOA_CHECK(m == output_shape[2] && n == output_shape[3] &&
                  k == rhs_shape[2 + rhs_transpose],
                  "The matrix multiplication (MxK * KxN = MxN) is invalid. "
                  "Got shape lhs={}, rhs={} and output={}",
                  lhs_shape.filter(2, 3), rhs_shape.filter(2, 3), output_shape.filter(2, 3));

        const std::array strides{&lhs_strides, &rhs_strides, &output_strides};
        const std::array shapes{&lhs_shape, &rhs_shape, &output_shape};
        const auto is_vector = Vec3<bool>{
                noa::indexing::is_vector(lhs_shape, true),
                noa::indexing::is_vector(rhs_shape, true),
                noa::indexing::is_vector(output_shape, true)};
        const auto is_column_major = Vec3<bool>{
                noa::indexing::is_column_major(lhs_strides),
                noa::indexing::is_column_major(rhs_strides),
                noa::indexing::is_column_major(output_strides)};

        // Enforce common order and extract the pitch, aka secondmost stride
        bool are_column_major{true};
        bool is_order_found{false};
        Strides3<Int> secondmost_strides;
        for (size_t i = 0; i < 3; ++i) {
            if (!is_vector[i]) {
                const auto& stride = *strides[i];
                const auto& shape = *shapes[i];

                // OpenBLAS and cublas require:
                //  1) the matrices should be either all row major or all column major.
                //  2) the innermost stride should be 1, i.e. contiguous
                //  3) the secondmost stride should be >= than the innermost extent.

                NOA_CHECK(!is_order_found || are_column_major == is_column_major[i],
                          "All matrices should either be row-major or column-major");
                if (!is_order_found)
                    are_column_major = is_column_major[i];
                is_order_found = true;

                secondmost_strides[i] = stride[2 + are_column_major];
                NOA_CHECK(stride[3 - are_column_major] == 1 &&
                          secondmost_strides[i] >= shape[3 - are_column_major],
                          "The innermost dimension of the matrices (before the optional transposition) "
                          "should be contiguous and the second-most dimension cannot be broadcast. "
                          "Got shape={}, strides={}, layout={}",
                          shape, stride, are_column_major ? "column" : "row");
            }
        }

        // At this point we know the order, so set the vectors according to the chosen order.
        for (size_t i = 0; i < 3; ++i) {
            if (is_vector[i]) {
                const auto& stride = *strides[i];
                const auto& shape = *shapes[i];

                // For vectors, it is more difficult here, so for now enforce contiguity.
                NOA_CHECK(noa::indexing::are_contiguous(stride, shape),
                          "Only contiguous vectors are currently supported, but got shape={} and strides={}",
                          shape, stride);

                const bool is_column_vector = shape[2] >= shape[3];
                if (is_column_vector == are_column_major) {
                    secondmost_strides[i] = shape[3 - is_column_vector];
                } else {
                    secondmost_strides[i] = 1;
                }
            }
        }

        return {{m, n, k},
                secondmost_strides,
                are_column_major};
    }
}

namespace noa::indexing {
    /// Whether the range [lhs_start, lhs_end] overlaps with the range [rhs_start, rhs_end].
    [[nodiscard]] constexpr inline bool are_overlapped(
            std::uintptr_t lhs_start, std::uintptr_t lhs_end,
            std::uintptr_t rhs_start, std::uintptr_t rhs_end) noexcept {
        return lhs_start <= rhs_end && lhs_end >= rhs_start;
    }

    template<typename T, typename U, typename Integer>
    [[nodiscard]] constexpr auto are_overlapped(
            const T* lhs, const Integer lhs_size,
            const U* rhs, const Integer rhs_size
    ) noexcept -> bool {
        if (lhs_size == 0 || rhs_size == 0)
            return false;

        const auto lhs_start = reinterpret_cast<std::uintptr_t>(lhs);
        const auto rhs_start = reinterpret_cast<std::uintptr_t>(rhs);
        const auto lhs_end = reinterpret_cast<std::uintptr_t>(lhs + lhs_size);
        const auto rhs_end = reinterpret_cast<std::uintptr_t>(rhs + rhs_size);
        return are_overlapped(lhs_start, lhs_end, rhs_start, rhs_end);
    }

    template<typename T, typename U, typename V, size_t N>
    [[nodiscard]] constexpr auto are_overlapped(
            const T* lhs, const Strides<V, N>& lhs_strides, const Shape<V, N>& lhs_shape,
            const U* rhs, const Strides<V, N>& rhs_strides, const Shape<V, N>& rhs_shape
    ) noexcept -> bool {
        if (noa::any(lhs_shape == 0) || noa::any(rhs_shape == 0))
            return false;
        return are_overlapped(lhs, at((lhs_shape - 1).vec(), lhs_strides),
                              rhs, at((rhs_shape - 1).vec(), rhs_strides));
    }

    template<typename T, typename U, typename V, size_t N>
    [[nodiscard]] constexpr auto are_overlapped(
            const T* lhs, const Strides<V, N>& lhs_strides,
            const U* rhs, const Strides<V, N>& rhs_strides,
            const Shape<V, N>& shape
    ) noexcept -> bool {
        return are_overlapped(lhs, lhs_strides, shape, rhs, rhs_strides, shape);
    }

    /// Whether the array with this memory layout has elements pointing to the same memory.
    /// This is useful to guard against data-race when the array is passed as output.
    template<typename T, size_t N>
    [[nodiscard]] constexpr auto are_elements_unique(
            const Strides<T, N>& strides,
            const Shape<T, N>& shape
    ) noexcept -> bool {
        // FIXME Check that a dimension doesn't overlap with another dimension.
        //       For now, just check the for broadcasting.
        for (size_t i = 0; i < N; ++i)
            if (strides[i] == 0 && shape[i] > 1)
                return false;
        return true;
    }
}

namespace noa::indexing {
    /// Returns the 2D rightmost indexes corresponding to
    /// the given memory offset in a contiguous layout.
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Vec2<T> offset2index(T offset, T size) noexcept {
        NOA_ASSERT(size > 0);
        const auto i0 = offset / size;
        const auto i1 = offset - i0 * size;
        return {i0, i1};
    }

    /// Returns the 3D rightmost indexes corresponding to
    /// the given memory offset in a contiguous layout.
    /// \param offset   Linear memory offset.
    /// \param s0,s1    DH sizes.
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Vec3<T> offset2index(T offset, T s0, T s1) noexcept {
        NOA_ASSERT(s0 > 0 && s1 > 0);
        const auto i0 = offset / (s0 * s1);
        offset -= i0 * s0 * s1;
        const auto i1 = offset / s1;
        offset -= i1 * s1;
        return {i0, i1, offset};
    }

    /// Returns the 4D rightmost indexes corresponding to
    /// the given memory offset in a contiguous layout.
    /// \param offset   Linear memory offset.
    /// \param s0,s1,s2 DHW sizes.
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Vec4<T> offset2index(T offset, T s0, T s1, T s2) noexcept {
        NOA_ASSERT(s0 > 0 && s1 > 0 && s2 > 0);
        const auto i0 = offset / (s0 * s1 * s2);
        offset -= i0 * s0 * s1 * s2;
        const auto i1 = offset / (s1 * s2);
        offset -= i1 * s1 * s2;
        const auto i2 = offset / s2;
        offset -= i2 * s2;
        return {i0, i1, i2, offset};
    }

    /// Returns the multidimensional indexes corresponding to a memory \p offset, assuming BDHW C-contiguity.
    /// \param offset   Memory offset within the array.
    /// \param shape    Shape of the array.
    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr Vec<T, N> offset2index(T offset, Shape<T, N> shape) noexcept {
        if constexpr (N == 1) {
            return Vec<T, N>{offset};
        } else if constexpr (N == 2) {
            return offset2index(offset, shape[1]);
        } else if constexpr (N == 3) {
            return offset2index(offset, shape[1], shape[2]);
        } else {
            return offset2index(offset, shape[1], shape[2], shape[3]);
        }
    }

    /// Returns the multidimensional indexes corresponding to a memory \p offset.
    /// \details Given a memory layout (i.e. strides and shape), this function computes the ND logical indexes
    ///          pointing at the given memory \p offset. Broadcasting is not supported, so the strides should
    ///          be greater than 0. Otherwise, any ordering is supported.
    /// \param offset   Memory offset within the array.
    /// \param strides  Strides of the array.
    /// \param shape    Shape of the array.
    template<bool ASSUME_RIGHTMOST = false, typename T, size_t N>
    [[nodiscard]] NOA_IHD constexpr auto offset2index(
            T offset,
            const Strides<T, N>& strides,
            const Shape<T, N>& shape
    ) noexcept {
        NOA_ASSERT(noa::all(shape > 0));
        Vec<T, N> out{0};
        T remain = offset;

        if constexpr (ASSUME_RIGHTMOST) {
            for (size_t i = 0; i < N; ++i) {
                if (shape[i] > 1) { // if empty, ignore it.
                    NOA_ASSERT(strides[i] > 0);
                    out[i] = remain / strides[i]; // single-divide optimization should kick in
                    remain %= strides[i]; // or remain -= out[i] * stride
                }
            }
        } else {
            const auto rightmost_order = order(strides, shape);
            for (size_t i = 0; i < N; ++i) {
                const auto idx = rightmost_order[i];
                if (shape[idx] > 1) {
                    NOA_ASSERT(strides[idx] > 0);
                    out[idx] = remain / strides[idx];
                    remain %= strides[idx];
                }
            }
        }

        NOA_ASSERT(remain == 0);
        return out;
    }
}

namespace noa::indexing {
    /// Reinterprets (i.e. casts) a 4D array.
    /// \usage 1. Create an object with the original shape, strides and pointer of the array to reinterpret.\n
    ///        2. Call the as<New> method to reinterpret the Old array as a New array.
    ///           If sizeof(Old) == sizeof(New), then this is equivalent to calling reinterpret_cast<New*>
    ///           on the original Old pointer.\n
    ///        3. Get the new shape, stride, and pointer from the output of the as<New> method.\n
    /// \note Reinterpretation is not always possible/well-defined. Old and New types, as well as the original
    ///       shape/strides should be compatible, otherwise an error will be thrown. This is mostly to represent
    ///       any data type as a array of bytes, or to switch between complex and real floating-point numbers with
    ///       the same precision.
    template<typename Old, typename Index>
    struct Reinterpret {
    public:
        static_assert(std::is_integral_v<Index>);
        using old_type = Old;
        using index_type = Index;
        using shape_type = Shape4<index_type>;
        using strides_type = Strides4<index_type>;
        using vec_type = Vec4<index_type>;

    public:
        shape_type shape{};
        strides_type strides{};
        old_type* ptr{};

    public:
        constexpr Reinterpret(const Shape4<index_type>& a_shape,
                              const Strides4<index_type>& a_strides,
                              old_type* a_ptr) noexcept
                : shape(a_shape),
                  strides(a_strides),
                  ptr{a_ptr} {}

    public:
        template<typename New>
        [[nodiscard]] auto as() const {
            Reinterpret<New, index_type> out(shape, strides, reinterpret_cast<New*>(ptr));
            if constexpr (traits::is_almost_same_v<old_type, New>)
                return out;

            // The "downsize" and "upsize" branches expects the strides and shape to be in the rightmost order.
            const vec_type rightmost_order = order(out.strides, out.shape);

            if constexpr (sizeof(old_type) > sizeof(New)) { // downsize
                constexpr index_type ratio = sizeof(old_type) / sizeof(New);
                NOA_CHECK(strides[rightmost_order[3]] == 1,
                          "The stride of the innermost dimension must be 1 to view a {} as a {}",
                          string::human<old_type>(), string::human<New>());
                out.strides[rightmost_order[0]] *= ratio;
                out.strides[rightmost_order[1]] *= ratio;
                out.strides[rightmost_order[2]] *= ratio;
                out.strides[rightmost_order[3]] = 1;
                out.shape[rightmost_order[3]] *= ratio;

            } else if constexpr (sizeof(old_type) < sizeof(New)) { // upsize
                constexpr index_type ratio = sizeof(New) / sizeof(old_type);
                NOA_CHECK(out.shape[rightmost_order[3]] % ratio == 0,
                          "The size of the innermost dimension must be divisible by {} to view a {} as a {}",
                          ratio, string::human<old_type>(), string::human<New>());

                NOA_CHECK(!(reinterpret_cast<std::uintptr_t>(ptr) % alignof(New)),
                          "The memory offset should be at least aligned to {} bytes to be viewed as a {}, but got {}",
                          alignof(New), string::human<New>(), static_cast<const void*>(ptr));

                NOA_CHECK(out.strides[rightmost_order[3]] == 1,
                          "The stride of the innermost dimension must be 1 to view a {} as a {}",
                          string::human<old_type>(), string::human<New>());

                for (int i = 0; i < 3; ++i) {
                    NOA_CHECK(!(out.strides[i] % ratio), "The strides must be divisible by {} to view a {} as a {}",
                              ratio, string::human<old_type>(), string::human<New>());
                    out.strides[i] /= ratio;
                }
                out.strides[rightmost_order[3]] = 1;
                out.shape[rightmost_order[3]] /= ratio;
            }
            return out;
        }
    };
}

namespace noa::indexing {
    /// Ellipsis or "..." operator, which selects the full extent of the remaining outermost dimension(s).
    struct ellipsis_t {};

    /// Selects the entire the dimension.
    struct full_extent_t {};

    /// Slice operator.
    /// Negative indexes are valid and starts from the end like in python.
    /// Indexes will be clamped to the dimension size.
    /// The step must be non-zero positive (negative strides are not supported).
    struct slice_t {
        template<typename T = int64_t, typename U = int64_t, typename V = int64_t>
        constexpr explicit slice_t(
                T start_ = 0,
                U end_ = std::numeric_limits<int64_t>::max(),
                V step_ = V{1})
                : start(static_cast<int64_t>(start_)),
                  end(static_cast<int64_t>(end_)),
                  step(static_cast<int64_t>(step_)) {}

        int64_t start{};
        int64_t end{};
        int64_t step{};
    };

    /// Splits a [0, \p size) range into \p n slices of approximately equal length.
    /// This is useful when distributing work to \p n threads.
    [[nodiscard]] inline std::vector<slice_t> split_into_slices(size_t size, size_t n) {
        std::vector<slice_t> slices;
        slices.reserve(n);
        const int count = static_cast<int>(n);
        const int size_ = static_cast<int>(size);
        for (int i = 0; i < count; ++i) {
            const int k = size_ / count;
            const int m = size_ % count;
            const int slice_start = i * k + noa::math::min(i, m);
            const int slice_end = (i + 1) * k + noa::math::min(i + 1, m);
            slices.emplace_back(slice_start, slice_end);
        }
        return slices;
    }

    /// Utility to create indexing subregions.
    /// Dimensions can be extracted using either:
    /// -   A single index value: This is bound-checked. Negative values are allowed.
    /// -   full_extent_t: Selects the entire dimension.
    /// -   slice_t: Slice operator. Slices are clamped to the dimension size. Negative values are allowed.
    /// -   ellipsis_t: Fills all unspecified dimensions with full_extent_t.
    struct Subregion {
    public:
        using index_type = int64_t;
        using offset_type = int64_t;
        using shape_type = Shape4<index_type>;
        using strides_type = Strides4<offset_type>;

    public:
        shape_type shape;
        strides_type strides;
        offset_type offset{0};

    public: // Useful traits
        template<typename U>
        static constexpr bool is_indexer_v =
                std::bool_constant<noa::traits::is_int_v<U> ||
                                   noa::traits::is_almost_same_v<U, full_extent_t> ||
                                   noa::traits::is_almost_same_v<U, slice_t>>::value;
        template<typename... Ts> using are_indexer = noa::traits::bool_and<is_indexer_v<Ts>...>;
        template<typename... Ts> static constexpr bool are_indexer_v = are_indexer<Ts...>::value;

    public:
        constexpr Subregion() = default;

        template<typename T, typename U, typename V = int64_t>
        constexpr Subregion(const Shape4<T>& start_shape,
                            const Strides4<U>& start_strides,
                            V start_offset = V{0}) noexcept
                : shape(start_shape.template as_safe<index_type>()),
                  strides(start_strides.template as_safe<offset_type>()),
                  offset(start_offset) {}

    public: // Extraction methods
        template<typename A,
                 typename B = full_extent_t,
                 typename C = full_extent_t,
                 typename D = full_extent_t,
                 typename = std::enable_if_t<are_indexer_v<A, B, C, D>>>
        [[nodiscard]] constexpr Subregion extract(A&& i0, B&& i1 = {}, C&& i2 = {}, D&& i3 = {}) const {
            Subregion out{};
            extract_dim_(i0, 0, shape[0], strides[0], out.shape.data() + 0, out.strides.data() + 0, &out.offset);
            extract_dim_(i1, 1, shape[1], strides[1], out.shape.data() + 1, out.strides.data() + 1, &out.offset);
            extract_dim_(i2, 2, shape[2], strides[2], out.shape.data() + 2, out.strides.data() + 2, &out.offset);
            extract_dim_(i3, 3, shape[3], strides[3], out.shape.data() + 3, out.strides.data() + 3, &out.offset);
            return out;
        }

        [[nodiscard]] constexpr Subregion extract(ellipsis_t) const {
            return *this;
        }

        template<typename A, typename = std::enable_if_t<is_indexer_v<A>>>
        [[nodiscard]] constexpr Subregion extract(ellipsis_t, A&& i3) const {
            return this->extract(full_extent_t{}, full_extent_t{}, full_extent_t{}, i3);
        }

        template<typename A, typename B, typename = std::enable_if_t<are_indexer_v<A, B>>>
        [[nodiscard]] constexpr Subregion extract(ellipsis_t, A&& i2, B&& i3) const {
            return this->extract(full_extent_t{}, full_extent_t{}, i2, i3);
        }

        template<typename A, typename B, typename C, typename = std::enable_if_t<are_indexer_v<A, B, C>>>
        [[nodiscard]] constexpr Subregion extract(ellipsis_t, A&& i1, B&& i2, C&& i3) const {
            return this->extract(full_extent_t{}, i1, i2, i3);
        }

    private:
        // Compute the new size, strides and offset, for one dimension, given an indexing mode (integral, slice or full).
        template<typename IndexMode>
        static constexpr void extract_dim_(
                IndexMode idx_mode, int64_t dim,
                int64_t old_size, int64_t old_strides,
                int64_t* new_size, int64_t* new_strides,
                int64_t* new_offset) {

            if constexpr (traits::is_int_v<IndexMode>) {
                auto index = clamp_cast<int64_t>(idx_mode);
                NOA_CHECK(!(index < -old_size || index >= old_size),
                          "Index {} is out of range for a size of {} at dimension {}",
                          index, old_size, dim);

                if (index < 0)
                    index += old_size;
                *new_strides = old_strides; // or 0
                *new_size = 1;
                *new_offset += old_strides * index;

            } else if constexpr(std::is_same_v<indexing::full_extent_t, IndexMode>) {
                *new_strides = old_strides;
                *new_size = old_size;
                *new_offset += 0;
                (void) idx_mode;
                (void) dim;

            } else if constexpr(std::is_same_v<indexing::slice_t, IndexMode>) {
                NOA_CHECK(idx_mode.step > 0, "Slice step must be positive, got {}", idx_mode.step);

                if (idx_mode.start < 0)
                    idx_mode.start += old_size;
                if (idx_mode.end < 0)
                    idx_mode.end += old_size;

                idx_mode.start = noa::math::clamp(idx_mode.start, int64_t{0}, old_size);
                idx_mode.end = noa::math::clamp(idx_mode.end, idx_mode.start, old_size);

                *new_size = noa::math::divide_up(idx_mode.end - idx_mode.start, idx_mode.step);
                *new_strides = old_strides * idx_mode.step;
                *new_offset += idx_mode.start * old_strides;
                (void) dim;
            } else {
                static_assert(traits::always_false_v<IndexMode>);
            }
        }
    };
}
