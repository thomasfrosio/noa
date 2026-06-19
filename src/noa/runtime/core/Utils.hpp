#pragma once

#include "noa/runtime/core/Access.hpp"
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

    template<typename T>
    [[nodiscard]] auto extract_matmul_layout(
        const Strides<T, 2>& lhs_strides, const Shape<T, 2>& lhs_shape,
        const Strides<T, 2>& rhs_strides, const Shape<T, 2>& rhs_shape,
        const Strides<T, 2>& output_strides, const Shape<T, 2>& output_shape,
        bool lhs_transpose, bool rhs_transpose
    ) -> Tuple<Shape<T, 3>, Strides<T, 3>, bool> {

        // First, extract and check the shape: MxK @ KxN = MxN
        const auto m = lhs_shape[0 + lhs_transpose];
        const auto n = rhs_shape[1 - rhs_transpose];
        const auto k = lhs_shape[1 - lhs_transpose];
        check(m == output_shape[0] and
              n == output_shape[1] and
              k == rhs_shape[0 + rhs_transpose],
              "The matrix multiplication (MxK * KxN = MxN) is invalid. "
              "Got lhs:shape={}, rhs:shape={} and output:shape={}, lhs_transpose={}, rhs_transpose={}",
              lhs_shape, rhs_shape, output_shape, lhs_transpose, rhs_transpose);

        const auto strides = std::array{&lhs_strides, &rhs_strides, &output_strides};
        const auto shapes = std::array{&lhs_shape, &rhs_shape, &output_shape};
        const auto is_vector = Vec{
            lhs_shape.is_vector(),
            rhs_shape.is_vector(),
            output_shape.is_vector()
        };
        const auto is_column_major = Vec{
            lhs_strides.is_column_major(),
            rhs_strides.is_column_major(),
            output_strides.is_column_major()
        };

        // Enforce common order and extract the secondmost stride (lda, ldb, ldc).
        bool are_column_major{true};
        bool is_order_found{false};
        Strides<T, 3> secondmost_strides;
        for (usize i = 0; i < 3; ++i) {
            if (not is_vector[i]) {
                const auto& stride = *strides[i];
                const auto& shape = *shapes[i];

                // OpenBLAS and cublas require:
                //  1) the matrices should be either all row major or all column major.
                //  2) the innermost stride should be 1.
                //  3) the secondmost stride should be >= than the innermost extent.

                check(not is_order_found or are_column_major == is_column_major[i],
                      "All matrices should either be row-major or column-major");
                if (not is_order_found)
                    are_column_major = is_column_major[i];
                is_order_found = true;

                secondmost_strides[i] = stride[0 + are_column_major];
                check(stride[1 - are_column_major] == 1 and
                      secondmost_strides[i] >= shape[1 - are_column_major],
                      "The innermost dimension of the matrices (before the optional transposition) "
                      "should be contiguous and the second-most dimension cannot be broadcast, "
                      "but got shape={}, strides={}, layout={}",
                      shape, stride, are_column_major ? "column" : "row");
            }
        }

        for (usize i = 0; i < 3; ++i) {
            if (is_vector[i]) {
                const auto& stride = *strides[i];
                const auto& shape = *shapes[i];

                // For vectors, for now, enforce contiguity.
                check(stride.is_contiguous(shape),
                      "Only contiguous vectors are currently supported, but got shape={} and strides={}",
                      shape, stride);

                const bool is_column_vector = shape[0] >= shape[1];
                if (is_column_vector == are_column_major) {
                    secondmost_strides[i] = shape[1 - is_column_vector];
                } else {
                    secondmost_strides[i] = 1;
                }
            }
        }

        return noa::make_tuple(Shape<T, 3>{m, n, k}, secondmost_strides, are_column_major);
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
    template<typename T, usize N, nt::integer I, StridesTraits StridesTrait>
    struct ReinterpretLayout {
    public:
        using value_type = T;
        using index_type = I;
        using shape_type = Shape<index_type, N>;
        using strides_type = Strides<index_type, N>;
        using vec_type = Vec<index_type, N>;

        static constexpr StridesTraits STRIDES_TRAIT = StridesTrait;
        static constexpr bool IS_CONTIGUOUS = STRIDES_TRAIT == StridesTraits::CONTIGUOUS;
        static constexpr usize SIZE = N;
        static constexpr isize SSIZE = N;

    public:
        shape_type shape{};
        strides_type strides{};
        value_type* ptr{};

    public:
        constexpr ReinterpretLayout(
            value_type* a_ptr,
            const Shape<index_type, N>& a_shape,
            const Strides<index_type, N>& a_strides
        ) noexcept :
            shape{a_shape},
            strides{a_strides},
            ptr{a_ptr} {}

    public:
        template<typename NewT, usize NewN = N, nt::integer NewI = I, StridesTraits NewStridesTrait = StridesTrait>
        [[nodiscard]] constexpr auto as() const {
            const auto reinterpreted = as_<NewT>();
            if constexpr (STRIDES_TRAIT != StridesTraits::CONTIGUOUS and
                          NewStridesTrait == StridesTraits::CONTIGUOUS) {
                check(reinterpreted.strides[N - 1] == 1,
                      "Cannot convert a non-contiguous layout with a rightmost stride of {} to a contiguous layout",
                      reinterpreted.strides[N - 1]);
            }

            using output_t = ReinterpretLayout<NewT, NewN, NewI, NewStridesTrait>;
            if constexpr (NewN == N) {
                return output_t(
                    reinterpreted.ptr,
                    reinterpreted.shape.template as_safe<NewI>(),
                    reinterpreted.strides.template as_safe<NewI>());
            } else if constexpr (NewN > N) {
                // Add empty dimensions on the left.
                constexpr usize n_dimensions_to_add = NewN - N;
                auto new_truncated_shape = reinterpreted.shape.template as_safe<NewI>();
                auto new_truncated_strides = reinterpreted.strides.template as_safe<NewI>();
                auto new_leftmost_stride = new_truncated_strides[0] * new_truncated_shape[0];
                return output_t(
                    reinterpreted.ptr,
                    new_truncated_shape.template push_front<n_dimensions_to_add>(1),
                    new_truncated_strides.template push_front<n_dimensions_to_add>(new_leftmost_stride));
            } else {
                // Construct the new shape by stacking the outer dimensions together.
                constexpr usize OFFSET = N - NewN;
                auto new_shape = Shape<index_type, N>::filled_with(1);
                for (usize i{}; i < N; ++i)
                    new_shape[max(i, OFFSET)] *= reinterpreted.shape[i];
                // TODO replace with this:
                // auto new_shape = reinterpreted.shape;
                // for (usize i{}; i < OFFSET; ++i) {
                //     new_shape[i + 1] *= new_shape[i]; // i+1 exists
                //     new_shape[i] = 1;
                // }

                // Reshape.
                Strides<index_type, N> new_stride{};
                check(noa::reshape(reinterpreted.shape, reinterpreted.strides, new_shape, new_stride),
                      "An array of shape {} and strides {} cannot be reshaped to shape {}",
                      reinterpreted.shape, reinterpreted.strides, new_shape);

                // Then remove the outer empty dimensions.
                return output_t(
                    reinterpreted.ptr,
                    new_shape.template pop_front<OFFSET>().template as_safe<NewI>(),
                    new_stride.template pop_front<OFFSET>().template as_safe<NewI>());
            }
        }

    private:
        template<typename NewT>
        [[nodiscard]] constexpr auto as_() const {
            using return_t = ReinterpretLayout<NewT, N, index_type, STRIDES_TRAIT>;

            if constexpr (nt::is_almost_same_v<value_type, NewT> or std::is_void_v<NewT> or std::is_void_v<value_type> or N == 0) {
                return return_t(static_cast<NewT*>(ptr), shape, strides);
            } else {
                auto out = return_t(reinterpret_cast<NewT*>(ptr), shape, strides);

                // The "downsize" and "upsize" branches expects the strides and shape to be in the rightmost order.
                const vec_type rightmost_order = out.strides.rightmost_order(out.shape);

                if constexpr (sizeof(value_type) > sizeof(NewT)) { // downsize
                    constexpr index_type ratio = sizeof(value_type) / sizeof(NewT);
                    if constexpr (not IS_CONTIGUOUS) {
                        check(strides[rightmost_order[N - 1]] == 1,
                              "The stride of the innermost dimension must be 1 to view a {} as a {}",
                              nd::stringify<value_type>(), nd::stringify<NewT>());
                    }
                    NOA_NV_DIAG_SUPPRESS(186)
                    for (usize i{}; i < N - 1; ++i)
                        out.strides[rightmost_order[i]] *= ratio;
                    NOA_NV_DIAG_DEFAULT(186)
                    out.strides[rightmost_order[N - 1]] = 1;
                    out.shape[rightmost_order[N - 1]] *= ratio;

                } else if constexpr (sizeof(value_type) < sizeof(NewT)) { // upsize
                    constexpr index_type ratio = sizeof(NewT) / sizeof(value_type);
                    check(out.shape[rightmost_order[N - 1]] % ratio == 0,
                          "The size of the innermost dimension must be divisible by {} to view a {} as a {}",
                          ratio, nd::stringify<value_type>(), nd::stringify<NewT>());

                    check(not (reinterpret_cast<std::uintptr_t>(ptr) % alignof(NewT)),
                          "The memory offset should be at least aligned to {} bytes to be viewed as a {}, but got {}",
                          alignof(NewT), nd::stringify<NewT>(), static_cast<const void*>(ptr));

                    if constexpr (not IS_CONTIGUOUS) {
                        check(out.strides[rightmost_order[N - 1]] == 1,
                              "The stride of the innermost dimension must be 1 to view a {} as a {}",
                              nd::stringify<value_type>(), nd::stringify<NewT>());
                    }

                    for (usize i{}; i < N - 1; ++i) {
                        check(not (out.strides[i] % ratio),
                              "The strides must be divisible by {} to view a {} as a {}",
                              ratio, nd::stringify<value_type>(), nd::stringify<NewT>());
                        out.strides[i] /= ratio;
                    }
                    out.strides[rightmost_order[N - 1]] = 1;
                    out.shape[rightmost_order[N - 1]] /= ratio;
                }
                return out;
            }
        }
    };

    template<typename T, usize N, nt::integer I>
    using ReinterpretLayoutStrided = ReinterpretLayout<T, N, I, StridesTraits::STRIDED>;

    template<typename T, usize N, nt::integer I>
    using ReinterpretLayoutContiguous = ReinterpretLayout<T, N, I, StridesTraits::CONTIGUOUS>;

    template<typename T>
    struct BatchedParameter {
        static constexpr bool IS_BATCHED = nt::pointer<T> or nt::accessor_nd<T, 1> or nt::span_nd<T, 1>;
        using type = T;
        using value_type = std::conditional_t<IS_BATCHED, nt::value_type_t<T>, T>;

        constexpr auto& operator[](nt::integer auto i) const noexcept {
            if constexpr (IS_BATCHED)
                return value[i];
            else
                return value;
        }

        constexpr auto& operator[](nt::integer auto i) noexcept {
            if constexpr (IS_BATCHED)
                return value[i];
            else
                return value;
        }

        NOA_NO_UNIQUE_ADDRESS T value;
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_batched_parameter<nd::BatchedParameter<T>> : std::true_type {};
}
