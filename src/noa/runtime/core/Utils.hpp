#pragma once

#include "noa/runtime/core/Shape.hpp"

namespace noa::details {
    /// Returns the index of the first non-empty dimension, excluding the batch dimension, going from left to right.
    /// If all dimensions are empty, the index of the width dimension is returned, ie 3.
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr auto non_empty_dhw_dimension(const Shape<T, 4>& shape) noexcept -> T {
        for (T i{1}; i < 4; ++i)
            if (shape[i] > 1)
                return i;
        return 3;
    }

    /// Computes the new strides of an array after reshaping.
    /// \param old_shape        Old shape. An empty shape (dimension of 0) returns false.
    /// \param old_strides      Old strides.
    /// \param new_shape        New shape.
    /// \param[out] new_strides New strides.
    /// \return Whether the input and output shape and strides are compatible.
    ///         If false, \p new_strides is left in an undefined state.
    /// \note Zero strides are allowed.
    template<typename T, usize OldN, usize NewN>
    [[nodiscard]] constexpr bool reshape(
        const Shape<T, OldN>& old_shape,
        const Strides<T, OldN>& old_strides,
        const Shape<T, NewN>& new_shape,
        Strides<T, NewN>& new_strides
    ) noexcept {
        // from https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorUtils.cpp
        if (old_shape.is_empty())
            return false;

        auto view_d = static_cast<isize>(NewN) - 1;
        T chunk_base_strides = old_strides[OldN - 1];
        T tensor_numel = 1;
        T view_numel = 1;
        for (isize tensor_d = static_cast<isize>(OldN) - 1; tensor_d >= 0; --tensor_d) {
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
    template<typename T, usize N> requires (N > 0)
    [[nodiscard]] constexpr bool infer_size(Shape<T, N>& shape, T n_elements) noexcept {
        // Adapted from https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/InferSize.h
        T infer_dim{-1};
        T new_size{1};
        for (usize dim{}; dim < N; ++dim) {
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

    template<nt::integer I, usize N, typename... T>
    NOA_FHD constexpr void permute_all(Vec<I, N> const& permutation, T&... vectors) {
        ((vectors = vectors.permute(permutation)), ...);
    }

    template<bool CHECK_ORDER = false, typename I, usize N, typename... T>
    NOA_FHD constexpr void permute_all_to_rightmost_order(
        Strides<I, N> strides,
        const Shape<I, N>& shape,
        T&... vectors
    ) {
        auto permutation = strides.rightmost_order(shape);
        if constexpr (CHECK_ORDER) {
            if (permutation != Vec<I, N>::arange())
                ((vectors = vectors.permute(permutation)), ...);
        } else {
            ((vectors = vectors.permute(permutation)), ...);
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

    template<typename T, typename U, typename V, usize N>
    [[nodiscard]] constexpr auto are_overlapped(
        const T* lhs, const Strides<V, N>& lhs_strides, const Shape<V, N>& lhs_shape,
        const U* rhs, const Strides<V, N>& rhs_strides, const Shape<V, N>& rhs_shape
    ) noexcept -> bool {
        if (lhs_shape.is_empty() or rhs_shape.is_empty())
            return false;
        return are_overlapped(lhs, offset_at((lhs_shape - 1).vec, lhs_strides),
                              rhs, offset_at((rhs_shape - 1).vec, rhs_strides));
    }

    template<typename T, typename U, typename V, usize N>
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
    template<typename T, usize N> requires (N > 0)
    [[nodiscard]] constexpr auto are_elements_unique(
        Strides<T, N> strides,
        Shape<T, N> shape
    ) noexcept -> bool {
        permute_all_to_rightmost_order<true>(strides, shape, strides, shape);

        for (usize i{}; i < N - 1; ++i)
            if (shape[i] > 1 and strides[i] < shape[i + 1])
                return false;
        return shape[N - 1] <= 1 or strides[N - 1] >= 1;
    }

    template<typename Int>
    [[nodiscard]] auto extract_matmul_layout(
        const Strides<Int, 4>& lhs_strides, const Shape<Int, 4>& lhs_shape,
        const Strides<Int, 4>& rhs_strides, const Shape<Int, 4>& rhs_shape,
        const Strides<Int, 4>& output_strides, const Shape<Int, 4>& output_shape,
        bool lhs_transpose, bool rhs_transpose
    ) -> std::tuple<Shape<Int, 3>, Strides<Int, 3>, bool> {

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
            lhs_shape.is_vector(true),
            rhs_shape.is_vector(true),
            output_shape.is_vector(true)
        };
        const Vec is_column_major{
            lhs_strides.is_column_major(),
            rhs_strides.is_column_major(),
            output_strides.is_column_major()
        };

        // Enforce common order and extract the secondmost stride (lda, ldb, ldc).
        bool are_column_major{true};
        bool is_order_found{false};
        Strides<Int, 3> secondmost_strides;
        for (usize i = 0; i < 3; ++i) {
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
                      "should be contiguous and the second-most dimension cannot be broadcast. "
                      "Got shape={}, strides={}, layout={}",
                      shape, stride, are_column_major ? "column" : "row");
            }
        }

        for (usize i = 0; i < 3; ++i) {
            if (is_vector[i]) {
                const auto& stride = *strides[i];
                const auto& shape = *shapes[i];

                // For vectors, it is more difficult here, so for now enforce contiguity.
                check(stride.is_contiguous(shape),
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
    template<usize N, typename T, nt::integer I>
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
                const vec_type rightmost_order = out.strides.rightmost_order(out.shape);

                if constexpr (sizeof(old_type) > sizeof(New)) { // downsize
                    constexpr index_type ratio = sizeof(old_type) / sizeof(New);
                    check(strides[rightmost_order[N - 1]] == 1,
                          "The stride of the innermost dimension must be 1 to view a {} as a {}",
                          nd::stringify<old_type>(), nd::stringify<New>());
                    NOA_NV_DIAG_SUPPRESS(186)
                    for (usize i{}; i < N - 1; ++i)
                        out.strides[rightmost_order[i]] *= ratio;
                    NOA_NV_DIAG_DEFAULT(186)
                    out.strides[rightmost_order[N - 1]] = 1;
                    out.shape[rightmost_order[N - 1]] *= ratio;

                } else if constexpr (sizeof(old_type) < sizeof(New)) { // upsize
                    constexpr index_type ratio = sizeof(New) / sizeof(old_type);
                    check(out.shape[rightmost_order[N - 1]] % ratio == 0,
                          "The size of the innermost dimension must be divisible by {} to view a {} as a {}",
                          ratio, nd::stringify<old_type>(), nd::stringify<New>());

                    check(not (reinterpret_cast<std::uintptr_t>(ptr) % alignof(New)),
                          "The memory offset should be at least aligned to {} bytes to be viewed as a {}, but got {}",
                          alignof(New), nd::stringify<New>(), static_cast<const void*>(ptr));

                    check(out.strides[rightmost_order[N - 1]] == 1,
                          "The stride of the innermost dimension must be 1 to view a {} as a {}",
                          nd::stringify<old_type>(), nd::stringify<New>());

                    for (usize i{}; i < N - 1; ++i) {
                        check(not (out.strides[i] % ratio),
                              "The strides must be divisible by {} to view a {} as a {}",
                              ratio, nd::stringify<old_type>(), nd::stringify<New>());
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
