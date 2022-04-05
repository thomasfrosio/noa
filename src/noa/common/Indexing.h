/// \file noa/common/types/Indexing.h
/// \brief Indexing utilities.
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/types/Int2.h"
#include "noa/common/types/Int3.h"
#include "noa/common/types/Int4.h"
#include "noa/common/types/ClampCast.h"

namespace noa::indexing {
    /// Returns the memory offset corresponding to the given 4D indexes.
    /// \param i0,i1,i2,i3  Multi-dimensional indexes.
    /// \param stride       Strides associated with these indexes.
    template<typename T, typename U, typename V, typename W, typename Z>
    NOA_FHD constexpr auto at(T i0, U i1, V i2, W i3, Int4<Z> stride) noexcept {
        static_assert(sizeof(Z) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        static_assert(std::is_integral_v<T> || std::is_integral_v<U> ||
                      std::is_integral_v<V> || std::is_integral_v<W>);
        return static_cast<Z>(i0) * stride[0] +
               static_cast<Z>(i1) * stride[1] +
               static_cast<Z>(i2) * stride[2] +
               static_cast<Z>(i3) * stride[3];
    }

    /// Returns the memory offset corresponding to the given 4D indexes.
    /// \param index    Multi-dimensional indexes.
    /// \param stride   Strides associated with these indexes.
    template<typename T, typename U>
    NOA_FHD constexpr auto at(Int4<T> index, Int4<U> stride) noexcept {
        return at(index[0], index[1], index[2], index[3], stride);
    }

    /// Returns the memory offset corresponding to the given 3D indexes.
    /// \tparam W       Int3 or Int4.
    /// \param i0,i1,i2 Multi-dimensional indexes.
    /// \param stride   Strides associated with these indexes. Only the first 3 values are used.
    template<typename T, typename U, typename V, typename W,
             typename = std::enable_if_t<noa::traits::is_int4_v<W> || noa::traits::is_int3_v<W>>>
    NOA_FHD constexpr auto at(T i0, U i1, V i2, W stride) noexcept {
        using value_t = noa::traits::value_type_t<W>;
        static_assert(sizeof(W) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        static_assert(std::is_integral_v<T> || std::is_integral_v<U> || std::is_integral_v<V>);
        return static_cast<value_t>(i0) * stride[0] +
               static_cast<value_t>(i1) * stride[1] +
               static_cast<value_t>(i2) * stride[2];
    }

    /// Returns the memory offset corresponding to the given 3D indexes.
    /// \tparam U       Int3 or Int4.
    /// \param index    Multi-dimensional indexes.
    /// \param stride   Strides associated with these indexes. Only the first 3 values are used.
    template<typename T, typename U,
             typename = std::enable_if_t<noa::traits::is_int4_v<U> || noa::traits::is_int3_v<U>>>
    NOA_FHD constexpr auto at(Int3<T> index, U stride) noexcept {
        return at(index[0], index[1], index[2], stride);
    }

    /// Returns the memory offset corresponding to the given 2D indexes.
    /// \tparam W       Int2, Int3, or Int4.
    /// \param i0,i1    Multi-dimensional indexes.
    /// \param stride   Strides associated with these indexes. Only the first 2 values are used.
    template<typename T, typename U, typename V,
             typename = std::enable_if_t<noa::traits::is_intX_v<V>>>
    NOA_FHD constexpr auto at(T i0, U i1, V stride) noexcept {
        using value_t = noa::traits::value_type_t<V>;
        static_assert(sizeof(value_t) >= 4, "don't compute memory offsets with less than 4 bytes values...");
        static_assert(std::is_integral_v<T> || std::is_integral_v<U>);
        return static_cast<value_t>(i0) * stride[0] +
               static_cast<value_t>(i1) * stride[1];
    }

    /// Returns the memory offset corresponding to the given 2D indexes.
    /// \tparam W       Int2, Int3, or Int4.
    /// \param i0,i1    Multi-dimensional indexes.
    /// \param stride   Strides associated with these indexes. Only the first 2 values are used.
    template<typename T, typename U,
             typename = std::enable_if_t<noa::traits::is_intX_v<U>>>
    NOA_FHD constexpr auto at(Int2<T> index, U stride) noexcept {
        return at(index[0], index[1], stride);
    }
}

namespace noa::indexing {
    /// Returns the 2D rightmost indexes corresponding to the given memory offset.
    /// \param offset   Linear memory offset.
    /// \param pitch    Pitch of the innermost dimension.
    /// \return         {index in the outermost dimension,
    ///                  index in the innermost dimension}.
    template<typename T>
    NOA_FHD constexpr Int2<T> indexes(T offset, T pitch) noexcept {
        const T i0 = offset / pitch;
        const T i1 = offset - i0 * pitch;
        return {i0, i1};
    }

    /// Returns the 3D rightmost indexes corresponding to the given memory offset.
    /// \param offset   Linear memory offset.
    /// \param p0       Pitch of the outermost dimension.
    /// \param p1       Pitch of the innermost dimension.
    /// \return         {index in the outermost dimension,
    ///                  index in the second-most dimension,
    ///                  index in the innermost dimension}.
    template<typename T>
    NOA_FHD constexpr Int3<T> indexes(T offset, T p0, T p1) noexcept {
        const T i0 = offset / (p0 * p1);
        const T tmp = offset - i0 * p0 * p1; // remove the offset to section
        const T i1 = tmp / p1;
        const T i2 = tmp - i1 * p1;
        return {i0, i1, i2};
    }
}

namespace noa::indexing {
    /// Whether or not the dimensions are contiguous.
    /// \param shape    Rightmost shape.
    /// \param stride   Rightmost stride.
    template<typename T>
    NOA_FHD auto isContiguous(Int4<T> stride, Int4<T> shape) {
        return bool4_t{(shape[1] * stride[1]) == stride[0],
                       (shape[2] * stride[2]) == stride[1],
                       (shape[3] * stride[3]) == stride[2],
                       stride[3] == 1};
    }

    /// Returns the rightmost order of the dimensions to have them in the most contiguous order.
    /// For instance, if stride is in the left-most order, this function returns {3,2,1,0}.
    template<typename T>
    NOA_FHD Int4<T> order(Int4<T> stride) {
        auto swap = [](T& a, T& b) {
            T tmp = a;
            a = b;
            b = tmp;
        };
        Int4<T> order{0, 1, 2, 3};
        if (stride[0] < stride[1])
            swap(order[0], order[1]);
        if (stride[2] < stride[3])
            swap(order[2], order[3]);
        if (stride[0] < stride[2])
            swap(order[0], order[2]);
        if (stride[1] < stride[3])
            swap(order[1], order[3]);
        if (stride[2] < stride[3])
            swap(order[2], order[3]);
        return order;
    }

    /// Reorder (i.e. sort) \p v according to the indexes in \p order.
    template<typename T, typename U>
    NOA_FHD Int4<T> reorder(Int4<T> v, Int4<U> order) {
        return {v[order[0]], v[order[1]], v[order[2]], v[order[3]]};
    }

    /// Sets the input stride so that the input can be iterated as if it as the same shape as the output.
    /// \param input_shape          Rightmost shape of the input. Should correspond to \p output_shape or be 1.
    /// \param[in,out] input_stride Rightmost input stride.
    ///                             Strides in dimensions that need to be broadcast are set to 0.
    /// \param output_shape         Rightmost shape of the output.
    /// \return Whether the input and output shape are compatible.
    template<typename T>
    NOA_FHD bool broadcast(Int4<T> input_shape, Int4<T>& input_stride, Int4<T> output_shape) noexcept {
        for (size_t i = 0; i < 4; ++i) {
            if (input_shape[i] == 1 && output_shape[i] != 1)
                input_stride[i] = 0; // broadcast this dimension
            else if (input_shape[i] != output_shape[i])
                return false; // dimension sizes don't match
        }
        return true;
    }
}

namespace noa::indexing {
    /// Reinterpret a 4D shape/stride.
    template<typename T, typename I = size_t>
    struct Reinterpret {
    public:
        static_assert(std::is_integral_v<I>);
        Int4<I> shape{};
        Int4<I> stride{};
        T* ptr{};

    public:
        template<typename U, typename = std::enable_if_t<std::is_integral_v<U>>>
        constexpr Reinterpret(Int4<U> a_shape, Int4<U> a_stride, T a_ptr) noexcept
                : shape{a_shape}, stride{a_stride}, ptr{a_ptr} {}

        template<typename V>
        Reinterpret<V> as() const {
            using origin_t = T;
            using new_t = V;
            Reinterpret<V> out{shape, stride, reinterpret_cast<V*>(ptr)};

            if constexpr (sizeof(origin_t) > sizeof(new_t)) { // downsize
                constexpr I ratio = sizeof(origin_t) / sizeof(new_t);
                NOA_CHECK(stride[3] == 1, "The stride of the innermost dimension must be 1 to view a {} as a {}",
                          string::human<origin_t>(), string::human<new_t>());
                out.stride[0] *= ratio;
                out.stride[1] *= ratio;
                out.stride[2] *= ratio;
                out.stride[3] = 1;
                out.shape[3] *= ratio;

            } else if constexpr (sizeof(origin_t) < sizeof(new_t)) { // upsize
                constexpr I ratio = sizeof(new_t) / sizeof(origin_t);
                static_assert(alignof(cdouble_t) == 16);
                NOA_CHECK(shape[3] % ratio == 0,
                          "The size of the innermost dimension must be divisible by {} to view a {} as a {}",
                          ratio, string::human<origin_t>(), string::human<new_t>());
                NOA_CHECK(ptr % alignof(new_t),
                          "The memory offset should at least be aligned to {} bytes to be viewed as a {}, but got {}",
                          alignof(new_t), string::human<new_t>(), static_cast<const void*>(ptr));

                NOA_CHECK(stride[3] == 1, "The stride of the innermost dimension must be 1 to view a {} as a {}",
                          string::human<origin_t>(), string::human<new_t>());
                for (int i = 0; i < 3; ++i) {
                    NOA_CHECK(stride[i] % ratio == 0, "The strides must be divisible by {} to view a {} as a {}",
                              ratio, string::human<origin_t>(), string::human<new_t>());
                    out.stride[i] /= ratio;
                }
                out.stride[3] = 1;
                out.shape[3] /= ratio;
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

    /// Slice operator. The start and end can be negative and out of bound. The step must be non-zero positive.
    struct slice_t {
        int64_t start{0};
        int64_t end{std::numeric_limits<int64_t>::max()};
        int64_t step{1};
    };

    /// Utility for indexing subregions.
    template<typename I>
    struct Subregion {
    public:
        static_assert(std::is_signed_v<I>);
        using dim_t = I;
        Int4<dim_t> shape{};
        Int4<dim_t> stride{};
        dim_t offset{};

    private:
        template<typename U>
        static constexpr bool is_indexer_v =
                std::bool_constant<noa::traits::is_int_v<U> ||
                                   noa::traits::is_same_v<U, indexing::full_extent_t> ||
                                   noa::traits::is_same_v<U, indexing::slice_t>>::value;

    public:
        constexpr Subregion() = default;

        template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
        constexpr Subregion(Int4<T> a_shape, Int4<T> a_stride, T a_offset = T{0}) noexcept
                : shape{a_shape}, stride{a_stride}, offset{static_cast<dim_t>(a_offset)} {}

        template<typename A,
                 typename B = indexing::full_extent_t,
                 typename C = indexing::full_extent_t,
                 typename D = indexing::full_extent_t,
                 typename = std::enable_if_t<is_indexer_v<A> && is_indexer_v<B> &&
                                             is_indexer_v<C> && is_indexer_v<D>>>
        constexpr Subregion operator()(A&& i0, B&& i1 = {}, C&& i2 = {}, D&& i3 = {}) const {
            Subregion out{};
            indexDim_(i0, 0, shape[0], stride[0], out.shape.get() + 0, out.stride.get() + 0, &out.offset);
            indexDim_(i1, 1, shape[1], stride[1], out.shape.get() + 1, out.stride.get() + 1, &out.offset);
            indexDim_(i2, 2, shape[2], stride[2], out.shape.get() + 2, out.stride.get() + 2, &out.offset);
            indexDim_(i3, 3, shape[3], stride[3], out.shape.get() + 3, out.stride.get() + 3, &out.offset);
            return out;
        }

        constexpr Subregion operator()(indexing::ellipsis_t) const {
            return *this;
        }

        template<typename A, typename = std::enable_if_t<is_indexer_v<A>>>
        constexpr Subregion operator()(indexing::ellipsis_t, A&& i3) const {
            return (*this)(indexing::full_extent_t{}, indexing::full_extent_t{}, indexing::full_extent_t{}, i3);
        }

        template<typename A, typename B,
                 typename = std::enable_if_t<is_indexer_v<A> && is_indexer_v<B>>>
        constexpr Subregion operator()(indexing::ellipsis_t, A&& i2, B&& i3) const {
            return (*this)(indexing::full_extent_t{}, indexing::full_extent_t{}, i2, i3);
        }

        template<typename A, typename B, typename C,
                 typename = std::enable_if_t<is_indexer_v<A> && is_indexer_v<B> && is_indexer_v<C>>>
        constexpr Subregion operator()(indexing::ellipsis_t, A&& i1, B&& i2, C&& i3) const {
            return (*this)(indexing::full_extent_t{}, i1, i2, i3);
        }

    private:
        // Compute the new size, stride and offset, for one dimension, given an indexing mode (integral, slice or full).
        template<typename IndexMode>
        static constexpr void indexDim_(IndexMode idx_mode, int dim, dim_t old_size, dim_t old_stride,
                                        dim_t* new_size, dim_t* new_stride, dim_t* new_offset) {
            if constexpr (traits::is_int_v<IndexMode>) {
                auto index = clamp_cast<int64_t>(idx_mode);
                NOA_CHECK(index < -old_size || index >= old_size,
                          "Index {} is out of range for a size of {} at dimension {}", index, old_size, dim);

                if (index < 0)
                    index += old_size;
                *new_stride = old_stride; // or 0
                *new_size = 1;
                *new_offset += old_stride * index;

            } else if constexpr(std::is_same_v<indexing::full_extent_t, IndexMode>) {
                *new_stride = old_stride;
                *new_size = old_size;
                *new_offset += 0;
                (void) idx_mode;

            } else if constexpr(std::is_same_v<indexing::slice_t, IndexMode>) {
                NOA_CHECK(idx_mode.step > 0, "Slice step must be positive, got {}", idx_mode.step);

                if (idx_mode.start < 0)
                    idx_mode.start += old_size;
                if (idx_mode.end < 0)
                    idx_mode.end += old_size;

                idx_mode.start = noa::math::clamp(idx_mode.start, int64_t{0}, old_size);
                idx_mode.end = noa::math::clamp(idx_mode.end, idx_mode.start, old_size);

                *new_size = noa::math::divideUp(idx_mode.end - idx_mode.start, idx_mode.step);
                *new_stride = old_stride * idx_mode.step;
                *new_offset += idx_mode.start * old_stride;
            } else {
                static_assert(traits::always_false_v<IndexMode>);
            }
        }
    };
}
