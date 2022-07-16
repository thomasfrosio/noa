/// \file noa/common/types/Indexing.h
/// \brief Indexing utilities.
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/types/Constants.h"
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

    /// Returns a valid index.
    /// If \p idx is out-of-bound, computes a valid index according to \p MODE. \p len should be > 0.
    template<BorderMode MODE, typename T, typename = std::enable_if_t<std::is_signed_v<T>>>
    NOA_IHD T at(T idx, T len) {
        static_assert(MODE == BORDER_CLAMP || MODE == BORDER_PERIODIC ||
                      MODE == BORDER_MIRROR || MODE == BORDER_REFLECT);
        // a % b == a - b * (a / b) == a + b * (-a / b)
        // Having a < 0 is well-defined since C++11.
        if constexpr (MODE == BORDER_CLAMP) {
            if (idx < 0)
                idx = 0;
            else if (idx >= len)
                idx = len - 1;
        } else if constexpr (MODE == BORDER_PERIODIC) {
            // 0 1 2 3 0 1 2 3 0 1 2 3 |  0 1 2 3  | 0 1 2 3 0 1 2 3 0 1 2 3
            T rem = idx % len; // FIXME maybe enclose this, at the expense of two jumps?
            idx = rem < 0 ? rem + len : rem;
        } else if constexpr (MODE == BORDER_MIRROR) {
            // 0 1 2 3 3 2 1 0 0 1 2 3 3 2 1 0 |  0 1 2 3  | 3 2 1 0 0 1 2 3 3 2 1 0
            if (idx < 0)
                idx = -idx - 1;
            if (idx >= len) {
                T period = 2 * len;
                idx %= period;
                if (idx >= len)
                    idx = period - idx - 1;
            }
        } else if constexpr (MODE == BORDER_REFLECT) {
            // 0 1 2 3 2 1 0 1 2 3 2 1 |  0 1 2 3  | 2 1 0 1 2 3 2 1 0
            if (idx < 0)
                idx = -idx;
            if (idx >= len) {
                T period = 2 * len - 2;
                idx %= period;
                if (idx >= len)
                    idx = period - idx;
            }
        }
        return idx;
    }
}

namespace noa::indexing {
    /// Returns the 2D rightmost indexes corresponding to the given memory offset.
    /// \param offset   Linear memory offset.
    /// \param pitch    Pitch of the innermost dimension.
    template<typename T>
    NOA_FHD constexpr Int2<T> indexes(T offset, T pitch) noexcept {
        const T i0 = offset / pitch;
        const T i1 = offset - i0 * pitch;
        return {i0, i1};
    }

    /// Returns the 3D rightmost indexes corresponding to the given memory offset.
    /// \param offset   Linear memory offset.
    /// \param p0,p1    Pitch of the second-most and innermost dimension.
    template<typename T>
    NOA_FHD constexpr Int3<T> indexes(T offset, T p0, T p1) noexcept {
        const T i0 = offset / (p0 * p1);
        const T tmp = offset - i0 * p0 * p1;
        const T i1 = tmp / p1;
        const T i2 = tmp - i1 * p1;
        return {i0, i1, i2};
    }

    /// Returns the 4D rightmost indexes corresponding to the given memory offset.
    /// \param offset   Linear memory offset.
    /// \param p0,p1,p2 Pitch of the third-most, second-most and innermost dimension.
    template<typename T>
    NOA_FHD constexpr Int4<T> indexes(T offset, T p0, T p1, T p2) noexcept {
        const T i0 = offset / (p0 * p1 * p2);
        const T tmp0 = offset - i0 * p0 * p1 * p2;
        const T i1 = tmp0 / (p1 * p2);
        const T tmp1 = offset - i1 * p1 * p2;
        const T i2 = tmp1 / p2;
        const T i3 = tmp1 - i2 * p2;
        return {i0, i1, i2, i3};
    }

    /// Returns the 4D rightmost indexes corresponding to the given memory offset.
    /// \param offset   Linear memory offset.
    /// \param pitch    Pitch of the third-most, second-most and innermost dimension.
    template<typename T>
    NOA_FHD constexpr Int4<T> indexes(T offset, Int3<T> pitch) noexcept {
        return {offset, pitch[0], pitch[1], pitch[2]};
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

    /// Whether or not the dimensions are contiguous.
    /// \param shape    Rightmost shape.
    /// \param stride   Rightmost stride.
    template<typename T>
    NOA_FHD auto isContiguous(Int3<T> stride, Int3<T> shape) {
        return bool3_t{(shape[1] * stride[1]) == stride[0],
                       (shape[2] * stride[2]) == stride[1],
                       stride[2] == 1};
    }

    /// Whether or not the dimensions are contiguous.
    /// \param shape    Rightmost shape.
    /// \param stride   Rightmost stride.
    template<typename T>
    NOA_FHD auto isContiguous(Int2<T> stride, Int2<T> shape) {
        return bool2_t{(shape[1] * stride[1]) == stride[0],
                       stride[1] == 1};
    }

    /// Whether or not a shape describe vector.
    /// \details A vector has one dimension with a size >= 1 and all of the other dimensions empty (i.e. size == 1).
    ///          By this definition, the shapes {1,1,1,1}, {5,1,1,1} and {1,1,1,5} are all vectors.
    ///          If \p can_be_batched is true, the shape can describe a batch of vectors. For instance {4,1,1,5} is
    ///          describing 4 row vectors with a length of 5.
    template<typename T>
    NOA_FHD bool isVector(Int4<T> shape, bool can_be_batched = false) {
        int non_empty_dimension = 0;
        for (int i = 0; i < 4; ++i) {
            if (shape[i] == 0)
                return false; // empty/invalid shape
            if (!(can_be_batched && i == 0) && shape[i] > 1)
                ++non_empty_dimension;
        }
        return non_empty_dimension <= 1;
    }

    /// Returns the order of \p stride, from the less (0) to the most (3) contiguous dimension.
    /// For instance, if \p stride is representing a column-major order, this function returns {0, 1, 3, 2}.
    template<typename T>
    NOA_FHD Int4<T> order(Int4<T> stride) {
        Int4<T> order{0, 1, 2, 3}; // row-major
        if (stride[0] < stride[1]) {
            std::swap(stride[0], stride[1]);
            std::swap(order[0], order[1]);
        }
        if (stride[2] < stride[3]) {
            std::swap(stride[2], stride[3]);
            std::swap(order[2], order[3]);
        }
        if (stride[0] < stride[2]) {
            std::swap(stride[0], stride[2]);
            std::swap(order[0], order[2]);
        }
        if (stride[1] < stride[3]) {
            std::swap(stride[1], stride[3]);
            std::swap(order[1], order[3]);
        }
        if (stride[1] < stride[2]) {
            std::swap(stride[2], stride[3]);
            std::swap(order[1], order[2]);
        }
        return order;
    }

    /// Reorder (i.e. sort) \p v according to the indexes in \p order.
    template<typename T, typename U>
    NOA_FHD Int4<T> reorder(Int4<T> v, Int4<U> order) {
        return {v[order[0]], v[order[1]], v[order[2]], v[order[3]]};
    }

    template<typename T, typename = std::enable_if_t<traits::is_intX_v<T>>>
    NOA_FHD bool isColMajor(T&& stride) {
        constexpr size_t N = std::decay_t<T>::COUNT;;
        constexpr size_t COL = N - 2;
        constexpr size_t ROW = N - 1;
        return stride[COL] <= stride[ROW];
    }

    template<typename T, typename = std::enable_if_t<traits::is_intX_v<T>>>
    NOA_FHD bool isColMajor(T&& stride, T&& shape) {
        int second{-1}, first{-1};
        for (int i = std::decay_t<T>::COUNT - 1; i >= 0; --i) {
            if (shape[i] > 1) {
                if (first == -1)
                    first = i;
                else if (second == -1)
                    second = i;
            }
        }
        return second == -1 || first == -1 || stride[second] <= stride[first];
    }

    template<typename T, typename = std::enable_if_t<traits::is_intX_v<T>>>
    NOA_FHD bool isRowMajor(T&& stride) {
        constexpr size_t N = std::decay_t<T>::COUNT;
        constexpr size_t COL = N - 2;
        constexpr size_t ROW = N - 1;
        return stride[COL] >= stride[ROW];
    }

    /// Sets the input stride so that the input can be iterated as if it as the same size as the output.
    /// \param input_size           Size of the input. Should correspond to \p output_size or be 1.
    /// \param[out] input_stride    Input stride. If broadcast, it is set to 0.
    /// \param output_size          Size of the output.
    /// \return Whether the input and output size are compatible.
    template<typename T>
    NOA_IH bool broadcast(T input_size, T& input_stride, T output_size) noexcept {
        if (input_size == 1 && output_size != 1)
            input_stride = 0; // broadcast this dimension
        else if (input_size != output_size)
            return false; // dimension sizes don't match
        return true;
    }

    /// Sets the input stride so that the input can be iterated as if it as the same shape as the output.
    /// \param input_shape          Rightmost shape of the input.
    ///                             Each dimension should correspond to \p output_shape or be 1.
    /// \param[out] input_stride    Rightmost input stride.
    ///                             Strides in dimensions that need to be broadcast are set to 0.
    /// \param output_shape         Rightmost shape of the output.
    /// \return Whether the input and output shape are compatible.
    template<typename T>
    NOA_IH bool broadcast(Int4<T> input_shape, Int4<T>& input_stride, Int4<T> output_shape) noexcept {
        for (size_t i = 0; i < 4; ++i) {
            if (input_shape[i] == 1 && output_shape[i] != 1)
                input_stride[i] = 0; // broadcast this dimension
            else if (input_shape[i] != output_shape[i])
                return false; // dimension sizes don't match
        }
        return true;
    }

    /// Returns the effective shape: if a dimension has a stride of 0, the effective size is 1 (empty dimension).
    template<typename T, typename = std::enable_if_t<traits::is_intX_v<T>>>
    NOA_IH T effectiveShape(T shape, T stride) noexcept {
        for (size_t i = 0; i < std::decay_t<T>::COUNT; ++i)
            shape[i] = stride[i] ? shape[i] : 1;
        return shape;
    }

    template<typename T, typename = std::enable_if_t<traits::is_intX_v<T>>>
    NOA_IH T effectiveStride(T shape, T stride) noexcept {
        for (size_t i = 0; i < std::decay_t<T>::COUNT; ++i)
            stride[i] = shape[i] > 1 ? stride[i] : 0;
        return stride;
    }

    /// Computes the new rightmost stride of an array after reshaping.
    /// \param old_shape        Old rightmost shape. An empty shape (dimension of 0) returns false.
    /// \param old_stride       Old rightmost stride.
    /// \param new_shape        New rightmost shape.
    /// \param[out] new_stride  New rightmost stride.
    /// \return Whether the input and output shape are compatible, i.e. whether they have the same number of elements.
    ///         If false, \p new_stride is left in an undefined state.
    template<typename T>
    NOA_IH bool reshape(Int4<T> old_shape, Int4<T> old_stride,
                        Int4<T> new_shape, Int4<T>& new_stride) noexcept {
        // see https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/TensorUtils.cpp
        if (!math::prod(old_shape))
            return false;

        int view_d = 3;
        T chunk_base_stride = old_stride[3];
        T tensor_numel = 1;
        T view_numel = 1;
        for (int tensor_d = 3; tensor_d >= 0; --tensor_d) {
            tensor_numel *= old_shape[tensor_d];
            // if end of tensor size chunk, check view
            if ((tensor_d == 0) ||
                (old_shape[tensor_d - 1] != 1 && old_stride[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
                while (view_d >= 0 && (view_numel < tensor_numel || new_shape[view_d] == 1)) {
                    new_stride[view_d] = view_numel * chunk_base_stride;
                    view_numel *= new_shape[view_d];
                    --view_d;
                }

                if (view_numel != tensor_numel)
                    return false;
                if (tensor_d > 0) {
                    chunk_base_stride = old_stride[tensor_d - 1];
                    tensor_numel = 1;
                    view_numel = 1;
                }
            }
        }
        if (view_d != -1)
            return false;
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
        constexpr Reinterpret(Int4<U> a_shape, Int4<U> a_stride, T* a_ptr) noexcept
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
                NOA_CHECK(!(reinterpret_cast<std::uintptr_t>(ptr) % alignof(new_t)),
                          "The memory offset should at least be aligned to {} bytes to be viewed as a {}, but got {}",
                          alignof(new_t), string::human<new_t>(), static_cast<const void*>(ptr));

                NOA_CHECK(stride[3] == 1, "The stride of the innermost dimension must be 1 to view a {} as a {}",
                          string::human<origin_t>(), string::human<new_t>());
                for (int i = 0; i < 3; ++i) {
                    NOA_CHECK(!(stride[i] % ratio), "The strides must be divisible by {} to view a {} as a {}",
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
        template<typename T = int64_t, typename U = int64_t, typename V = int64_t>
        constexpr explicit slice_t(T start_ = 0, U end_ = std::numeric_limits<int64_t>::max(), V step_ = V{1})
                : start(static_cast<int64_t>(start_)),
                  end(static_cast<int64_t>(end_)),
                  step(static_cast<int64_t>(step_)) {}

        int64_t start{};
        int64_t end{};
        int64_t step{};
    };

    /// Utility for indexing subregions.
    struct Subregion {
    private:
        template<typename U>
        static constexpr bool is_indexer_v =
                std::bool_constant<noa::traits::is_int_v<U> ||
                                   noa::traits::is_almost_same_v<U, indexing::full_extent_t> ||
                                   noa::traits::is_almost_same_v<U, indexing::slice_t>>::value;

    public:
        constexpr Subregion() = default;

        template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
        constexpr Subregion(Int4<T> a_shape, Int4<T> a_stride, T a_offset = T{0}) noexcept
                : m_shape{a_shape}, m_stride{a_stride}, m_offset{static_cast<size_t>(a_offset)} {}

        template<typename A,
                 typename B = indexing::full_extent_t,
                 typename C = indexing::full_extent_t,
                 typename D = indexing::full_extent_t,
                 typename = std::enable_if_t<is_indexer_v<A> && is_indexer_v<B> &&
                                             is_indexer_v<C> && is_indexer_v<D>>>
        constexpr Subregion operator()(A&& i0, B&& i1 = {}, C&& i2 = {}, D&& i3 = {}) const {
            Subregion out{};
            indexDim_(i0, 0, m_shape[0], m_stride[0], out.m_shape.get() + 0, out.m_stride.get() + 0, &out.m_offset);
            indexDim_(i1, 1, m_shape[1], m_stride[1], out.m_shape.get() + 1, out.m_stride.get() + 1, &out.m_offset);
            indexDim_(i2, 2, m_shape[2], m_stride[2], out.m_shape.get() + 2, out.m_stride.get() + 2, &out.m_offset);
            indexDim_(i3, 3, m_shape[3], m_stride[3], out.m_shape.get() + 3, out.m_stride.get() + 3, &out.m_offset);
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

    public:
        [[nodiscard]] constexpr size4_t shape() const noexcept { return size4_t{m_shape}; }
        [[nodiscard]] constexpr size4_t stride() const noexcept { return m_stride; }
        [[nodiscard]] constexpr size_t offset() const noexcept { return m_offset; }

    private:
        // Compute the new size, stride and offset, for one dimension, given an indexing mode (integral, slice or full).
        template<typename IndexMode>
        static constexpr void indexDim_(IndexMode idx_mode, int64_t dim,
                                        int64_t old_size, size_t old_stride,
                                        int64_t* new_size, size_t* new_stride,
                                        size_t* new_offset) {
            if constexpr (traits::is_int_v<IndexMode>) {
                auto index = clamp_cast<int64_t>(idx_mode);
                NOA_CHECK(!(index < -old_size || index >= old_size),
                          "Index {} is out of range for a size of {} at dimension {}", index, old_size, dim);

                if (index < 0)
                    index += old_size;
                *new_stride = old_stride; // or 0
                *new_size = 1;
                *new_offset += old_stride * static_cast<size_t>(index);

            } else if constexpr(std::is_same_v<indexing::full_extent_t, IndexMode>) {
                *new_stride = old_stride;
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

                *new_size = noa::math::divideUp(idx_mode.end - idx_mode.start, idx_mode.step);
                *new_stride = old_stride * static_cast<size_t>(idx_mode.step);
                *new_offset += static_cast<size_t>(idx_mode.start) * old_stride;
                (void) dim;
            } else {
                static_assert(traits::always_false_v<IndexMode>);
            }
        }

    private:
        long4_t m_shape{};
        size4_t m_stride{};
        size_t m_offset{};
    };
}
