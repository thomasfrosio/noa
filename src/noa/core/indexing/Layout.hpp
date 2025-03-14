#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Error.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa::indexing {
    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr bool is_rightmost(const Strides<T, N>& strides) noexcept {
        return strides.is_rightmost();
    }

    /// Whether \p strides and \p shape describe a contiguous array.
    /// \tparam ORDER 'C' for C-contiguous (row-major) or 'F' for F-contiguous (column-major).
    /// \note Empty dimensions are contiguous by definition since their strides are not used.
    ///       Broadcast dimensions are NOT contiguous.
    template<char ORDER = 'C', typename T>
    [[nodiscard]] NOA_HD constexpr bool are_contiguous(const Strides4<T>& strides, const Shape4<T>& shape) noexcept {
        if (shape.is_empty()) // guard against empty array
            return false;

        if constexpr (ORDER == 'c' or ORDER == 'C') {
            return (shape[0] == 1 or strides[0] == shape[3] * shape[2] * shape[1]) and
                   (shape[1] == 1 or strides[1] == shape[3] * shape[2]) and
                   (shape[2] == 1 or strides[2] == shape[3]) and
                   (shape[3] == 1 or strides[3] == 1);

        } else if constexpr (ORDER == 'f' or ORDER == 'F') {
            return (shape[0] == 1 or strides[0] == shape[3] * shape[2] * shape[1]) and
                   (shape[1] == 1 or strides[1] == shape[3] * shape[2]) and
                   (shape[2] == 1 or strides[2] == 1) and
                   (shape[3] == 1 or strides[3] == shape[2]);
        } else {
            static_assert(nt::always_false<T>);
        }
    }

    /// Checks whether all of the 4d accessors are contiguous.
    template<char ORDER = 'C', nt::tuple_of_accessor_or_empty T, nt::integer I>
    NOA_HD constexpr auto are_contiguous(
        const T& accessors,
        const Shape4<I>& shape
    ) noexcept -> bool {
        return accessors.all([&shape]<typename U>(const U& accessor) {
            if constexpr (nt::accessor_value<U>) {
                return true;
            } else {
                static_assert(U::SIZE == 4);
                return are_contiguous<ORDER>(accessor.strides_full(), shape);
            }
        });
    }

    /// For each dimension, check if it is contiguous.
    /// \details If one wants to know in which dimension the contiguity is broken or if the contiguity is only
    ///          required in a particular dimension, this function can be useful. It supports broadcasting and
    ///          empty dimensions. It is also guarded against empty shapes.
    /// \tparam ORDER 'C' for C-contiguous (row-major) or 'F' for F-contiguous (column-major).
    /// \note If one want to check whether the array is contiguous, while all(is_contiguous(...)) is valid,
    ///       are_contiguous(...) is preferred simply because it is clearer and slightly more efficient.
    /// \note Broadcast dimensions are NOT contiguous. Only empty dimensions are treated as contiguous
    ///       regardless of their stride. Functions that require broadcast dimensions to be "contiguous"
    ///       should call effective_shape() first, to "cancel" the broadcasting and mark the dimension as empty.
    template<char ORDER = 'C', typename T>
    [[nodiscard]] NOA_HD constexpr auto is_contiguous(Strides4<T> strides, const Shape4<T>& shape) noexcept {
        if (shape.is_empty()) // guard against empty array
            return Vec4<bool>{};

        strides *= Strides4<T>::from_vec(shape != 1); // mark the stride of empty dimensions unusable

        if constexpr (ORDER == 'c' or ORDER == 'C') {
            // If dimension is broadcast or empty, we cannot use the stride
            // and need to use the corrected stride one dimension up.
            const auto corrected_stride_2 = strides[3] ? shape[3] * strides[3] : 1;
            const auto corrected_stride_1 = strides[2] ? shape[2] * strides[2] : corrected_stride_2;
            const auto corrected_stride_0 = strides[1] ? shape[1] * strides[1] : corrected_stride_1;

            // If dimension is empty, it is by definition contiguous and its stride does not matter.
            // Broadcast dimensions break the contiguity because the corrected_stride cannot be 0.
            // This is true for empty dimensions, but empty dimensions are contiguous by definition
            // and their strides do not matter, i.e. we skip the comparison.
            return Vec{shape[0] == 1 or strides[0] == corrected_stride_0,
                       shape[1] == 1 or strides[1] == corrected_stride_1,
                       shape[2] == 1 or strides[2] == corrected_stride_2,
                       shape[3] == 1 or strides[3] == 1};

        } else if constexpr (ORDER == 'f' or ORDER == 'F') {
            const auto corrected_stride_3 = strides[2] ? shape[2] * strides[2] : 1;
            const auto corrected_stride_1 = strides[3] ? shape[3] * strides[3] : corrected_stride_3;
            const auto corrected_stride_0 = strides[1] ? shape[1] * strides[1] : corrected_stride_1;

            return Vec{shape[0] == 1 or strides[0] == corrected_stride_0,
                       shape[1] == 1 or strides[1] == corrected_stride_1,
                       shape[2] == 1 or strides[2] == 1,
                       shape[3] == 1 or strides[3] == corrected_stride_3};

        } else {
            static_assert(nt::always_false<T>);
        }
    }

    /// Checks whether all of the 4d accessors are contiguous.
    template<char ORDER = 'C', nt::tuple_of_accessor_or_empty T, nt::integer I>
    auto is_contiguous(
        const T& accessors,
        const Shape4<I>& shape
    ) noexcept -> Vec4<bool> {
        auto out = Vec4<bool>::from_value(true);
        accessors.for_each([&shape, &out]<typename U>(const U& accessor) {
            if constexpr (not nt::accessor_value<U>) {
                static_assert(U::SIZE == 4);
                out = out and is_contiguous<ORDER>(accessor.strides_full().template as<I>(), shape);
            }
        });
        return out;
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr bool is_vector(const Shape<T, 4>& shape, bool can_be_batched = false) noexcept {
        return shape.is_vector(can_be_batched);
    }

    template<typename T, size_t N> requires (N <= 3)
    [[nodiscard]] NOA_FHD constexpr bool is_vector(const Shape<T, N>& shape) noexcept {
        return shape.is_vector();
    }

    /// Returns the effective shape: if a dimension has a stride of 0, the effective size is 1 (empty dimension).
    template<typename T, typename U, size_t N>
    [[nodiscard]] NOA_FHD constexpr auto effective_shape(Shape<T, N> shape, const Strides<U, N>& strides) noexcept {
        for (size_t i{}; i < N; ++i)
            shape[i] = strides[i] ? shape[i] : 1;
        return shape;
    }

    /// Returns the index of the first non-empty dimension, excluding the batch dimension, going from left to right.
    /// If all dimensions are empty, the index of the width dimension is returned, ie 3.
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr auto non_empty_dhw_dimension(const Shape4<T>& shape) noexcept -> T {
        for (T i{1}; i < 4; ++i)
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
    template<typename T, size_t N>
    [[nodiscard]] NOA_HD constexpr auto order(Strides<T, N> strides, const Shape<T, N>& shape) noexcept {
        Vec<T, N> order;
        for (size_t i{}; i < N; ++i) {
            order[i] = static_cast<T>(i);
            strides[i] = shape[i] <= 1 ? std::numeric_limits<T>::max() : strides[i];
        }

        return stable_sort(order, [&](T a, T b) {
            return strides[a] > strides[b];
        });
    }

    /// Returns the order the dimensions should be sorted so that empty dimensions are on the left.
    /// The input dimensions have the following indexes: {0, 1, 2, 3}.
    /// Coupled with indexing::reorder(), this effectively pushes all zeros and ones in \p shape to the left.
    /// The difference with indexing::order() is that this function does not change the order of the non-empty
    /// dimensions relative to each other. Note that the order of the empty dimensions is preserved.
    template<typename T> requires (nt::vec_integer<T> or nt::shape_or_strides<T>)
    [[nodiscard]] NOA_HD constexpr auto squeeze_left(const T& shape) noexcept {
        using value_t = T::value_type;
        constexpr auto SIZE = static_cast<value_t>(T::SIZE);
        Vec<value_t, T::SIZE> order{};
        value_t index{};
        for (value_t i{}; i < SIZE; ++i) { // store empty dimensions
            if (shape[i] <= 1)
                order[index++] = i;
        }
        for (value_t i{}; i < SIZE; ++i) { // then valid dimensions
            if (shape[i] > 1)
                order[index++] = i;
        }
        return order;
    }

    template<typename T> requires (nt::vec_integer<T> or nt::shape_or_strides<T>)
    [[nodiscard]] NOA_HD constexpr auto squeeze_right(const T& shape) noexcept {
        using value_t = T::value_type;
        constexpr auto SIZE = static_cast<value_t>(T::SIZE);
        Vec<value_t, T::SIZE> order{};
        value_t index{};
        for (value_t i{}; i < SIZE; ++i) { // store valid dimensions
            if (shape[i] > 1)
                order[index++] = i;
        }
        for (value_t i{}; i < SIZE; ++i) { // then empty dimensions
            if (shape[i] <= 1)
                order[index++] = i;
        }
        return order;
    }

    /// Reorder (i.e. sort) \p vector according to the indexes in \p order.
    template<typename T, nt::integer I, size_t N> requires nt::vec_shape_or_strides_of_size<T, N>
    [[nodiscard]] NOA_FHD constexpr auto reorder(T vector, const Vec<I, N>& order) noexcept {
        return vector.reorder(order);
    }

    /// Returns the reordered matrix according to the indexes in \p order.
    /// The columns are reordered, and then the rows. This can be useful to swap the axes of a matrix.
    /// \param[in] matrix   Square and (truncated) affine matrix to reorder.
    /// \param[in] order    Order of indexes. Should have the same number of elements as the matrices are rows.
    template<nt::mat T, nt::integer I, size_t N> requires (T::ROWS == N)
    [[nodiscard]] NOA_HD constexpr T reorder(const T& matrix, const Vec<I, N>& order) noexcept {
        T reordered_matrix;
        for (size_t row{}; row < N; ++row) {
            typename T::row_type reordered_row{}; // no need to initialize, but g++ warn it may be uninitialized before use...
            for (size_t column{}; column < N; ++column)
                reordered_row[column] = matrix[row][order[column]];
            reordered_matrix[order[row]] = reordered_row;
        }
        return reordered_matrix;
    }

    /// (Circular) shifts \p v by a given amount.
    /// If \p shift is positive, shifts to the right, otherwise, shifts to the left.
    template<typename T, nt::integer I, size_t N> requires nt::vec_shape_or_strides_of_size<T, N>
    [[nodiscard]] NOA_FHD constexpr auto circular_shift(T vector, const Vec<I, N>& order) noexcept {
        return vector.circular_shift(order);
    }

    /// Whether \p strides describes a column-major layout.
    /// Note that this functions assumes BDHW order, where H is the number of rows and W is the number of columns.
    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr bool is_column_major(const Strides<T, N>& strides) noexcept {
        constexpr size_t COL = N - 2;
        constexpr size_t ROW = N - 1;
        return strides[COL] <= strides[ROW];
    }

    /// Whether \p strides describes a column-major layout.
    /// This function effectively squeezes the shape before checking the order.
    /// Furthermore, strides of empty dimensions are ignored and are contiguous by definition.
    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr bool is_column_major(
        const Strides<T, N>& strides,
        const Shape<T, N>& shape
    ) noexcept {
        i32 second{-1}, first{-1};
        for (i32 i = N - 1; i >= 0; --i) {
            if (shape[i] > 1) {
                if (first == -1)
                    first = i;
                else if (second == -1)
                    second = i;
            }
        }
        return second == -1 or first == -1 or strides[second] <= strides[first];
    }

    /// Whether \p strides describes a row-major layout.
    /// Note that this functions assumes BDHW order, where H is the number of rows and W is the number of columns.
    template<typename T, size_t N>
    [[nodiscard]] NOA_FHD constexpr bool is_row_major(const Strides<T, N>& strides) noexcept {
        constexpr size_t COL = N - 2;
        constexpr size_t ROW = N - 1;
        return strides[COL] >= strides[ROW];
    }

    /// Whether \p strides describes a row-major layout.
    /// This function effectively squeezes the shape before checking the order.
    /// Furthermore, strides of empty dimensions are ignored and are contiguous by definition.
    template<typename T, size_t N>
    [[nodiscard]] NOA_HD constexpr bool is_row_major(
        const Strides<T, N>& strides,
        const Shape<T, N>& shape
    ) noexcept {
        i32 second{-1}, first{-1};
        for (i32 i = N - 1; i >= 0; --i) {
            if (shape[i] > 1) {
                if (first == -1)
                    first = i;
                else if (second == -1)
                    second = i;
            }
        }
        return second == -1 or first == -1 or strides[second] >= strides[first];
    }

    /// Sets the input stride so that the input can be iterated as if it as the same size as the output.
    /// \param input_size           Size of the input. Should correspond to \p output_size or be 1.
    /// \param[out] input_stride    Input stride. If broadcast, it is set to 0.
    /// \param output_size          Size of the output.
    /// \return Whether the input and output size are compatible.
    template<nt::integer I>
    [[nodiscard]] NOA_FHD constexpr bool broadcast(I input_size, I& input_stride, I output_size) noexcept {
        if (input_size == 1 and output_size != 1)
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
    [[nodiscard]] NOA_FHD constexpr bool broadcast(
        const Shape<T, N>& input_shape,
        Strides<T, N>& input_strides,
        const Shape<T, N>& output_shape
    ) noexcept {
        for (size_t i{}; i < N; ++i) {
            if (input_shape[i] == 1 and output_shape[i] != 1)
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
    [[nodiscard]] constexpr bool reshape(
        const Shape<T, OldN>& old_shape,
        const Strides<T, OldN>& old_strides,
        const Shape<T, NewN>& new_shape,
        Strides<T, NewN>& new_strides
    ) noexcept {
        // from https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorUtils.cpp
        if (old_shape.is_empty())
            return false;

        auto view_d = static_cast<i64>(NewN) - 1;
        T chunk_base_strides = old_strides[OldN - 1];
        T tensor_numel = 1;
        T view_numel = 1;
        for (i64 tensor_d = static_cast<i64>(OldN) - 1; tensor_d >= 0; --tensor_d) {
            tensor_numel *= old_shape[tensor_d];
            // if end of tensor size chunk, check view
            if ((tensor_d == 0) or
                (old_shape[tensor_d - 1] != 1 and old_strides[tensor_d - 1] != tensor_numel * chunk_base_strides)) {
                while (view_d >= 0 and (view_numel < tensor_numel or new_shape[view_d] == 1)) {
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

    /// Tries to infer the size of a dimension with size -1, if it exists.
    /// Also checks that new shape is compatible with the number of elements.
    /// If the inference failed or if the inferred shape isn't correct, returns false.
    template<typename T, size_t N> requires (N > 0)
    [[nodiscard]] constexpr bool infer_size(Shape<T, N>& shape, T n_elements) noexcept {
        // Adapted from https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/InferSize.h
        T infer_dim{-1};
        T new_size{1};
        for (size_t dim{}; dim < N; ++dim) {
            if (shape[dim] == -1) {
                if (infer_dim != -1)
                    return false; // only one dimension can be inferred
                infer_dim = static_cast<T>(dim);
            } else if (shape[dim] >= 0) {
                new_size *= shape[dim];
            } else {
                return false; // invalid shape dimension
            }
        }

        // Only the number of elements matters. So non-inferred dimensions can have different sizes
        // as long as the number of elements is the same. If inference, find the integer multiple to
        // complete the shape.
        if (n_elements == new_size) {
            if (infer_dim != -1)
                shape[infer_dim] = 1; // the dimension asked for inference is empty
            return true;
        } else if (infer_dim != -1 and new_size > 0 and n_elements % new_size == 0) {
            shape[infer_dim] = n_elements / new_size;
            return true; // inferred
        } else {
            return false; // shape and n_elements don't match, or empty array
        }
    }

    /// Whether the range [lhs_start, lhs_end] overlaps with the range [rhs_start, rhs_end].
    [[nodiscard]] constexpr bool are_overlapped(
        std::uintptr_t lhs_start, std::uintptr_t lhs_end,
        std::uintptr_t rhs_start, std::uintptr_t rhs_end
    ) noexcept {
        return lhs_start <= rhs_end and lhs_end >= rhs_start;
    }

    template<typename T, typename U, nt::integer I>
    [[nodiscard]] constexpr auto are_overlapped(
        const T* lhs, const I lhs_size,
        const U* rhs, const I rhs_size
    ) noexcept -> bool {
        if (lhs_size == 0 or rhs_size == 0)
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
        if (lhs_shape.is_empty() or rhs_shape.is_empty())
            return false;
        return are_overlapped(lhs, offset_at((lhs_shape - 1).vec, lhs_strides),
                              rhs, offset_at((rhs_shape - 1).vec, rhs_strides));
    }

    template<typename T, typename U, typename V, size_t N>
    [[nodiscard]] constexpr auto are_overlapped(
        const T* lhs, const Strides<V, N>& lhs_strides,
        const U* rhs, const Strides<V, N>& rhs_strides,
        const Shape<V, N>& shape
    ) noexcept -> bool {
        return are_overlapped(lhs, lhs_strides, shape, rhs, rhs_strides, shape);
    }

    /// Whether an array with this memory layout has unique elements, i.e. elements do not point to the same memory.
    /// This is useful to guard against data-race when the array is passed as output.
    /// FIXME this is not perfect, as it return false when dimensions are intertwined
    template<typename T, size_t N> requires (N > 0)
    [[nodiscard]] constexpr auto are_elements_unique(
        Strides<T, N> strides,
        Shape<T, N> shape
    ) noexcept -> bool {
        auto rightmost_order = order(strides, shape);
        if (vany(NotEqual{}, rightmost_order, Vec<T, N>::arange())) {
            strides = strides.reorder(rightmost_order);
            shape = shape.reorder(rightmost_order);
        }

        for (size_t i{}; i < N - 1; ++i)
            if (shape[i] > 1 and strides[i] < shape[i + 1])
                return false;
        return shape[N - 1] <= 1 or strides[N - 1] >= 1;
    }

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
        check(lhs_shape[1] == 1 and rhs_shape[1] == 1 and output_shape[1] == 1,
              "Only 2d matrices are supported, but got shape lhs:shape={}, rhs:shape={} and output:shape={}",
              lhs_shape, rhs_shape, output_shape);
        check(m == output_shape[2] and
              n == output_shape[3] and
              k == rhs_shape[2 + rhs_transpose],
              "The matrix multiplication (MxK * KxN = MxN) is invalid. "
              "Got lhs:shape={}, rhs:shape={} and output:shape={}, lhs_transpose={}, rhs_transpose={}",
              lhs_shape.filter(2, 3), rhs_shape.filter(2, 3), output_shape.filter(2, 3),
              lhs_transpose, rhs_transpose);

        const std::array strides{&lhs_strides, &rhs_strides, &output_strides};
        const std::array shapes{&lhs_shape, &rhs_shape, &output_shape};
        const Vec is_vector{
            ni::is_vector(lhs_shape, true),
            ni::is_vector(rhs_shape, true),
            ni::is_vector(output_shape, true)
        };
        const Vec is_column_major{
            ni::is_column_major(lhs_strides),
            ni::is_column_major(rhs_strides),
            ni::is_column_major(output_strides)
        };

        // Enforce common order and extract the secondmost stride (lda, ldb, ldc).
        bool are_column_major{true};
        bool is_order_found{false};
        Strides3<Int> secondmost_strides;
        for (size_t i = 0; i < 3; ++i) {
            if (not is_vector[i]) {
                const auto& stride = *strides[i];
                const auto& shape = *shapes[i];

                // OpenBLAS and cublas require:
                //  1) the matrices should be either all row major or all column major.
                //  2) the innermost stride should be 1, i.e. contiguous
                //  3) the secondmost stride should be >= than the innermost extent.

                check(not is_order_found or are_column_major == is_column_major[i],
                      "All matrices should either be row-major or column-major");
                if (not is_order_found)
                    are_column_major = is_column_major[i];
                is_order_found = true;

                secondmost_strides[i] = stride[2 + are_column_major];
                check(stride[3 - are_column_major] == 1 and
                      secondmost_strides[i] >= shape[3 - are_column_major],
                      "The innermost dimension of the matrices (before the optional transposition) "
                      "should be contiguous and the second-most dimension cannot be broadcasted. "
                      "Got shape={}, strides={}, layout={}",
                      shape, stride, are_column_major ? "column" : "row");
            }
        }

        for (size_t i = 0; i < 3; ++i) {
            if (is_vector[i]) {
                const auto& stride = *strides[i];
                const auto& shape = *shapes[i];

                // For vectors, it is more difficult here, so for now enforce contiguity.
                check(are_contiguous(stride, shape),
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

    /// Reinterprets (i.e. casts) a ND array.
    /// \usage 1. Create an object with the original shape, strides and pointer of the array to reinterpret.\n
    ///        2. Call the as<New> method to reinterpret the Old array as a New array.
    ///           If sizeof(Old) == sizeof(New), then this is equivalent to calling reinterpret_cast<New*>
    ///           on the original Old pointer.\n
    ///        3. Get the new shape, stride, and pointer from the output of the as<New> method.\n
    /// \note Reinterpretation is not always possible/well-defined. Old and New types, as well as the original
    ///       shape/strides should be compatible, otherwise an error will be thrown. This is mostly to represent
    ///       any data type as a array of bytes, or to switch between complex and real floating-point numbers with
    ///       the same precision.
    template<size_t N, typename T, nt::integer I>
    struct ReinterpretLayout {
    public:
        using old_type = T;
        using index_type = I;
        using shape_type = Shape<index_type, N>;
        using strides_type = Strides<index_type, N>;
        using vec_type = Vec<index_type, N>;

    public:
        shape_type shape{};
        strides_type strides{};
        old_type* ptr{};

    public:
        constexpr ReinterpretLayout(
            const Shape<index_type, N>& a_shape,
            const Strides<index_type, N>& a_strides,
            old_type* a_ptr
        ) noexcept :
            shape{a_shape},
            strides{a_strides},
            ptr{a_ptr} {}

    public:
        template<typename New>
        [[nodiscard]] auto as() const {
            using return_t = ReinterpretLayout<N, New, index_type>;

            if constexpr (nt::is_almost_same_v<old_type, New> or
                          std::is_void_v<New> or
                          std::is_void_v<old_type> or
                          N == 0) {
                return return_t(shape, strides, static_cast<New*>(ptr));
            } else {
                auto out = return_t(shape, strides, reinterpret_cast<New*>(ptr));

                // The "downsize" and "upsize" branches expects the strides and shape to be in the rightmost order.
                const vec_type rightmost_order = order(out.strides, out.shape);

                if constexpr (sizeof(old_type) > sizeof(New)) { // downsize
                    constexpr index_type ratio = sizeof(old_type) / sizeof(New);
                    check(strides[rightmost_order[N - 1]] == 1,
                          "The stride of the innermost dimension must be 1 to view a {} as a {}",
                          ns::stringify<old_type>(), ns::stringify<New>());
                    NOA_NV_DIAG_SUPPRESS(186)
                    for (size_t i{}; i < N - 1; ++i)
                        out.strides[rightmost_order[i]] *= ratio;
                    NOA_NV_DIAG_DEFAULT(186)
                    out.strides[rightmost_order[N - 1]] = 1;
                    out.shape[rightmost_order[N - 1]] *= ratio;

                } else if constexpr (sizeof(old_type) < sizeof(New)) { // upsize
                    constexpr index_type ratio = sizeof(New) / sizeof(old_type);
                    check(out.shape[rightmost_order[N - 1]] % ratio == 0,
                          "The size of the innermost dimension must be divisible by {} to view a {} as a {}",
                          ratio, ns::stringify<old_type>(), ns::stringify<New>());

                    check(not (reinterpret_cast<std::uintptr_t>(ptr) % alignof(New)),
                          "The memory offset should be at least aligned to {} bytes to be viewed as a {}, but got {}",
                          alignof(New), ns::stringify<New>(), static_cast<const void*>(ptr));

                    check(out.strides[rightmost_order[N - 1]] == 1,
                          "The stride of the innermost dimension must be 1 to view a {} as a {}",
                          ns::stringify<old_type>(), ns::stringify<New>());

                    for (size_t i{}; i < N - 1; ++i) {
                        check(not (out.strides[i] % ratio),
                              "The strides must be divisible by {} to view a {} as a {}",
                              ratio, ns::stringify<old_type>(), ns::stringify<New>());
                        out.strides[i] /= ratio;
                    }
                    out.strides[rightmost_order[N - 1]] = 1;
                    out.shape[rightmost_order[N - 1]] /= ratio;
                }
                return out;
            }
        }
    };
}
