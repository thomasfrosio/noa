#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/types/Int4.h"
#include "noa/common/Indexing.h"

namespace noa {
    /// View of a 4-dimensional array.
    /// \tparam T Type of the viewed memory region. Any (const-qualified) data type.
    /// \tparam I Type to hold sizes, strides and used for indexing. Any integral type.
    template<typename T, typename I = size_t>
    class View {
    public:
        using value_t = T;
        using dim_t = I;
        using dim4_t = Int4<dim_t>;
        using ptr_t = T*;
        using ref_t = T&;

    private:
        static_assert(!std::is_pointer_v<T>);
        static_assert(!std::is_reference_v<T>);
        static_assert(std::is_integral_v<I>);

        template<typename U>
        static constexpr bool is_indexable_v =
                std::bool_constant<noa::traits::is_int_v<U> ||
                                   noa::traits::is_almost_same_v<U, indexing::full_extent_t> ||
                                   noa::traits::is_almost_same_v<U, indexing::slice_t>>::value;

    public: // Constructors
        /// Creates an empty view.
        NOA_HD constexpr View() = default;

        /// Creates a contiguous view.
        NOA_HD constexpr View(T* data, Int4<I> shape)
                : m_shape(shape), m_stride(shape.stride()), m_ptr(data) {}

        /// Creates a (strided) view.
        NOA_HD constexpr View(T* data, Int4<I> shape, Int4<I> stride)
                : m_shape(shape), m_stride(stride), m_ptr(data) {}

        /// Creates a const view from an existing non-const view.
        template<typename U, typename = std::enable_if_t<std::is_const_v<T> && std::is_same_v<U, std::remove_const_t<T>>>>
        NOA_HD constexpr /* implicit */ View(const View<U>& view)
                : m_shape(view.shape()), m_stride(view.stride()), m_ptr(view.data()) {}

    public: // Getters
        [[nodiscard]] NOA_HD constexpr T* data() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr T* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr const dim4_t& shape() const noexcept { return m_shape; }
        [[nodiscard]] NOA_HD constexpr const dim4_t& stride() const noexcept { return m_stride; }
        [[nodiscard]] NOA_HD constexpr bool4_t contiguous() const noexcept { return indexing::isContiguous(m_stride, m_shape); }
        [[nodiscard]] NOA_HD constexpr bool empty() const noexcept { return !(m_ptr && m_shape.elements()); }

    public: // Data reinterpretation
        /// Reinterpret the managed array of \p T as an array of \p U.
        /// \note This is only well defined in cases where reinterpret_cast<U*>(T*) is well defined, for instance,
        ///       when \p U is a unsigned char (represent any data type as a array of bytes), or to switch between
        ///       complex and real floating-point numbers with the same precision.
        template<typename U, typename J = I>
        View<U, J> as() const {
            const auto out = indexing::Reinterpret<T, J>{m_shape, m_stride, get()}.template as<U>();
            return {out.ptr, out.shape, out.stride};
        }

        /// Reshapes the view.
        /// \param shape Rightmost shape. Must contain the same number of elements as the current shape.
        View reshape(Int4<I> shape) const {
            Int4<I> new_stride;
            if (!indexing::reshape(m_shape, m_stride, shape, new_stride))
                NOA_THROW("An view of shape {} cannot be reshaped to an view of shape {}", m_shape, shape);
            return {m_ptr, shape, new_stride};
        }

        /// Permutes the dimensions of the view.
        /// \param permutation  Rightmost permutation. Axes are numbered from 0 to 3, 3 being the innermost dimension.
        View permute(uint4_t permutation) const {
            return {m_ptr, indexing::reorder(m_shape, permutation), indexing::reorder(m_stride, permutation)};
        }

    public: // Setters
        NOA_HD constexpr void shape(size4_t shape) noexcept { m_shape = shape; }
        NOA_HD constexpr void stride(size4_t stride) noexcept { m_stride = stride; }
        NOA_HD constexpr void data(T* data) noexcept { m_ptr = data; }

    public: // Contiguous views
        [[nodiscard]] NOA_HD constexpr T* begin() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr T* end() const noexcept { return m_ptr + m_shape.elements(); }

        [[nodiscard]] NOA_HD constexpr T& front() const noexcept { return *begin(); }
        [[nodiscard]] NOA_HD constexpr T& back() const noexcept { return *(end() - 1); }

    public: // Loop-like indexing
        template<typename I0>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0) const noexcept {
            return m_ptr[static_cast<size_t>(i0) * m_stride[0]];
        }

        template<typename I0, typename I1>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0, I1 i1) const noexcept {
            return m_ptr[indexing::at(i0, i1, m_stride)];
        }

        template<typename I0, typename I1, typename I2>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0, I1 i1, I2 i2) const noexcept {
            return m_ptr[indexing::at(i0, i1, i2, m_stride)];
        }

        template<typename I0, typename I1, typename I2, typename I3>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
            return m_ptr[indexing::at(i0, i1, i2, i3, m_stride)];
        }

    public: // Subview
        template<typename A,
                 typename B = indexing::full_extent_t,
                 typename C = indexing::full_extent_t,
                 typename D = indexing::full_extent_t,
                 typename = std::enable_if_t<is_indexable_v<A> && is_indexable_v<B> &&
                                             is_indexable_v<C> && is_indexable_v<D>>>
        constexpr View subregion(A&& i0, B&& i1 = {}, C&& i2 = {}, D&& i3 = {}) const {
            const auto indexer = indexing::Subregion<int64_t>{long4_t{m_shape}, long4_t{m_stride}}(i0, i1, i2, i3);
            return {m_ptr + indexer.offset, size4_t{indexer.shape}, size4_t{indexer.stride}};
        }

        constexpr View subregion(indexing::ellipsis_t) const {
            return *this;
        }

        template<typename A, typename = std::enable_if_t<is_indexable_v<A>>>
        constexpr View subregion(indexing::ellipsis_t, A&& i3) const {
            return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, indexing::full_extent_t{}, i3);
        }

        template<typename A, typename B,
                 typename = std::enable_if_t<is_indexable_v<A> && is_indexable_v<B>>>
        constexpr View subregion(indexing::ellipsis_t, A&& i2, B&& i3) const {
            return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, i2, i3);
        }

        template<typename A, typename B, typename C,
                 typename = std::enable_if_t<is_indexable_v<A> && is_indexable_v<B> && is_indexable_v<C>>>
        constexpr View subregion(indexing::ellipsis_t, A&& i1, B&& i2, C&& i3) const {
            return subregion(indexing::full_extent_t{}, i1, i2, i3);
        }

    private:
        dim4_t m_shape;
        dim4_t m_stride;
        T* m_ptr{};
    };
}
