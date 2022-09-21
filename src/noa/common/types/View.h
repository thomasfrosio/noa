#pragma once
#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/types/Int4.h"
#include "noa/common/Indexing.h"

namespace noa {
    /// View over a 4-dimensional array.
    /// \tparam T Type of the viewed memory region. Any (const-qualified) data type.
    /// \tparam I Type to hold sizes, strides and used for indexing. Any integral type.
    template<typename T, typename I = size_t>
    class View {
    public:
        using value_t = T;
        using value_type = T;
        using ptr_t = T*;
        using ptr_type = T*;
        using ref_t = T&;
        using ref_type = T&;

        using dim_t = I;
        using dim4_t = Int4<dim_t>;

    private:
        static_assert(!std::is_pointer_v<T>);
        static_assert(!std::is_reference_v<T>);
        static_assert(std::is_integral_v<I>);

        template<typename U>
        static constexpr bool is_indexable_v =
                std::bool_constant<traits::is_int_v<U> ||
                                   traits::is_almost_same_v<U, indexing::full_extent_t> ||
                                   traits::is_almost_same_v<U, indexing::slice_t>>::value;

    public: // Constructors
        /// Creates an empty view.
        NOA_HD constexpr View() = default;

        /// Creates a contiguous view.
        NOA_HD constexpr View(T* data, const Int4<I>& shape)
                : m_shape(shape), m_strides(shape.strides()), m_ptr(data) {}

        /// Creates a (strided) view.
        NOA_HD constexpr View(T* data, const Int4<I>& shape, const Int4<I>& stride)
                : m_shape(shape), m_strides(stride), m_ptr(data) {}

        /// Creates a const view from an existing non-const view.
        template<typename U, typename = std::enable_if_t<std::is_const_v<T> && std::is_same_v<U, std::remove_const_t<T>>>>
        NOA_HD constexpr /* implicit */ View(const View<U>& view)
                : m_shape(view.shape()), m_strides(view.strides()), m_ptr(view.data()) {}

    public: // Getters
        /// Returns the pointer to the data.
        [[nodiscard]] NOA_HD constexpr T* get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr T* data() const noexcept { return m_ptr; }

        /// Returns the BDHW shape of the array.
        [[nodiscard]] NOA_HD constexpr const dim4_t& shape() const noexcept { return m_shape; }

        /// Returns the BDHW strides of the array.
        [[nodiscard]] NOA_HD constexpr const dim4_t& strides() const noexcept { return m_strides; }

        /// Returns the number of elements in the array.
        [[nodiscard]] NOA_HD constexpr dim_t elements() const noexcept { return m_shape.elements(); }
        [[nodiscard]] NOA_HD constexpr dim_t size() const noexcept { return elements(); }

        /// Whether the dimensions of the array are C or F contiguous.
        template<char ORDER = 'C'>
        [[nodiscard]] NOA_HD constexpr bool contiguous() const noexcept {
            return indexing::areContiguous<ORDER>(m_strides, m_shape);
        }

        /// Whether the view is empty. A View is empty if not initialized,
        /// or if the viewed data is null, or if one of its dimension is 0.
        [[nodiscard]] NOA_HD constexpr bool empty() const noexcept {
            return !m_ptr || any(m_shape == 0);
        }

    public: // Data reinterpretation
        /// Reinterpret the managed array of \p T as an array of \p U.
        /// \note This is only well defined in cases where reinterpret_cast<U*>(T*) is well defined, for instance,
        ///       when \p U is a unsigned char or std::byte (represent any data type as a array of bytes),
        ///       or to switch between complex and real floating-point numbers with the same precision.
        template<typename U, typename J = I>
        View<U, J> as() const {
            const auto out = indexing::Reinterpret<T, J>(m_shape, m_strides, get()).template as<U>();
            return {out.ptr, out.shape, out.strides};
        }

        /// Reshapes the view.
        /// \param shape New shape. Must contain the same number of elements as the current shape.
        View reshape(Int4<I> shape) const {
            Int4<I> new_stride;
            if (!indexing::reshape(m_shape, m_strides, shape, new_stride))
                NOA_THROW("A view of shape {} and stride {} cannot be reshaped to a view of shape {}",
                          m_shape, m_strides, shape);
            return {m_ptr, shape, new_stride};
        }

        /// Reshapes the array in a vector along a particular axis.
        /// Returns a row vector by default.
        View flat(int axis) const {
            Int4<I> output_shape(1);
            output_shape[axis] = m_shape.elements();
            return reshape(output_shape);
        }

        /// Permutes the dimensions of the view.
        /// \param permutation  Permutation with the axes numbered from 0 to 3.
        View permute(uint4_t permutation) const {
            return {m_ptr,
                    indexing::reorder(m_shape, dim4_t(permutation)),
                    indexing::reorder(m_strides, dim4_t(permutation))};
        }

    public: // Setters
        /// (Re)Sets the shape.
        NOA_HD constexpr void shape(size4_t shape) noexcept { m_shape = shape; }

        /// (Re)Sets the strides.
        NOA_HD constexpr void strides(size4_t strides) noexcept { m_strides = strides; }

        /// (Re)Sets the viewed data.
        NOA_HD constexpr void data(T* data) noexcept { m_ptr = data; }

    public: // Iteration on contiguous views
        [[nodiscard]] NOA_HD constexpr T* begin() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr T* end() const noexcept { return m_ptr + m_shape.elements(); }

        [[nodiscard]] NOA_HD constexpr T& front() const noexcept { return *begin(); }
        [[nodiscard]] NOA_HD constexpr T& back() const noexcept { return *(end() - 1); }

    public: // Loop-like indexing
        /// Returns a reference at the specified memory \p offset.
        template<typename I0, typename = std::enable_if_t<traits::is_int_v<I0>>>
        [[nodiscard]] NOA_HD constexpr T& operator[](I0 offset) const noexcept {
            NOA_ASSERT(!empty() && offset <= indexing::at(m_shape - 1, m_strides));
            return m_ptr[offset];
        }

        /// Returns a reference at the beginning of the specified batch index.
        template<typename I0>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 batch) const noexcept {
            NOA_ASSERT(!empty() && static_cast<dim_t>(batch) < m_shape[0]);
            return m_ptr[indexing::at(batch, m_strides[0])];
        }

        /// Returns a reference at the beginning of the specified batch and depth indexes.
        template<typename I0, typename I1>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 batch, I1 depth) const noexcept {
            NOA_ASSERT(!empty() &&
                       static_cast<dim_t>(batch) < m_shape[0] &&
                       static_cast<dim_t>(depth) < m_shape[1]);
            return m_ptr[indexing::at(batch, depth, m_strides)];
        }

        /// Returns a reference at the beginning of the specified batch, depth and height indexes.
        template<typename I0, typename I1, typename I2>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 batch, I1 depth, I2 height) const noexcept {
            NOA_ASSERT(!empty() &&
                       static_cast<dim_t>(batch) < m_shape[0] &&
                       static_cast<dim_t>(depth) < m_shape[1] &&
                       static_cast<dim_t>(height) < m_shape[2]);
            return m_ptr[indexing::at(batch, depth, height, m_strides)];
        }

        /// Returns a reference at the beginning of the specified batch, depth, height and width indexes.
        template<typename I0, typename I1, typename I2, typename I3>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 batch, I1 depth, I2 height, I3 width) const noexcept {
            NOA_ASSERT(!empty() &&
                       static_cast<dim_t>(batch) < m_shape[0] &&
                       static_cast<dim_t>(depth) < m_shape[1] &&
                       static_cast<dim_t>(height) < m_shape[2] &&
                       static_cast<dim_t>(width) < m_shape[3]);
            return m_ptr[indexing::at(batch, depth, height, width, m_strides)];
        }

    public: // Subregion
        template<typename A,
                 typename B = indexing::full_extent_t,
                 typename C = indexing::full_extent_t,
                 typename D = indexing::full_extent_t,
                 typename = std::enable_if_t<is_indexable_v<A> && is_indexable_v<B> &&
                                             is_indexable_v<C> && is_indexable_v<D>>>
        constexpr View subregion(A&& i0, B&& i1 = {}, C&& i2 = {}, D&& i3 = {}) const {
            const auto indexer = indexing::Subregion(m_shape, m_strides)(i0, i1, i2, i3);
            return {m_ptr + indexer.offset(), indexer.shape(), indexer.strides()};
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
        dim4_t m_strides;
        T* m_ptr{};
    };
}
