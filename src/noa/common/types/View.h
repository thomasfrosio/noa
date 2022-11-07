#pragma once

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Indexing.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Accessor.h"
#include "noa/common/types/Int4.h"
#include "noa/common/types/SafeCast.h"

namespace noa {
    /// View over a 4-dimensional array.
    /// \details This class is meant to provide a simple 4D shaped representation of some memory region.
    ///          It is similar to the Array class in that way, but is non-owning and the data type can
    ///          be const-qualified. It keeps track of a pointer and the size and stride of each dimension.
    ///          As such, indexing are bound-checked (in Debug builds).
    /// \note For a smaller type, which defines the dimensionality of the data statically
    ///       and doesn't keep track of the size of the dimensions, use Accessor.
    template<typename T, typename I>
    class View {
    public:
        static_assert(!std::is_pointer_v<T>);
        static_assert(!std::is_reference_v<T>);
        static_assert(std::is_integral_v<I>);

        using accessor_type = Accessor<T, 4, I>;
        using accessor_reference_type = AccessorReference<T, 4, I>;

        using pointer_type = typename accessor_type::pointer_type;
        using value_type = typename accessor_type::value_type;
        using index_type = typename accessor_type::index_type;
        using index4_type = Int4<index_type>;

        static constexpr index_type COUNT = accessor_type::COUNT;

    private:
        template<typename U>
        static constexpr bool is_indexable_v =
                std::bool_constant<traits::is_int_v<U> ||
                                   traits::is_almost_same_v<U, indexing::full_extent_t> ||
                                   traits::is_almost_same_v<U, indexing::slice_t>>::value;

    public: // Constructors
        /// Creates an empty view.
        NOA_HD constexpr View() = default;

        /// Creates a contiguous view.
        NOA_HD constexpr View(T* data, const index4_type& shape)
                : m_shape(shape), m_strides(shape.strides()), m_ptr(data) {}

        /// Creates a (strided) view.
        NOA_HD constexpr View(T* data, const index4_type& shape, const index4_type& stride)
                : m_shape(shape), m_strides(stride), m_ptr(data) {}

        /// Creates a const view from an existing non-const view.
        template<typename U, typename = std::enable_if_t<std::is_const_v<T> && std::is_same_v<U, std::remove_const_t<T>>>>
        NOA_HD constexpr /* implicit */ View(const View<U, I>& view)
                : m_shape(view.shape()), m_strides(view.strides()), m_ptr(view.data()) {}

    public: // Getters
        /// Returns the pointer to the data.
        [[nodiscard]] NOA_HD constexpr pointer_type get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr pointer_type data() const noexcept { return m_ptr; }

        /// Returns the BDHW shape of the array.
        [[nodiscard]] NOA_HD constexpr index4_type& shape() noexcept { return m_shape; }
        [[nodiscard]] NOA_HD constexpr const index4_type& shape() const noexcept { return m_shape; }

        /// Returns the BDHW strides of the array.
        [[nodiscard]] NOA_HD constexpr index4_type& strides() noexcept { return m_strides; }
        [[nodiscard]] NOA_HD constexpr const index4_type& strides() const noexcept { return m_strides; }

        /// Returns the number of elements in the array.
        [[nodiscard]] NOA_HD constexpr index_type elements() const noexcept { return m_shape.elements(); }
        [[nodiscard]] NOA_HD constexpr index_type size() const noexcept { return elements(); }

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
        /// Returns a 4D Accessor.
        [[nodiscard]] constexpr auto accessor() const {
            using output_t = Accessor<T, 4, index_type, AccessorTraits::DEFAULT>;
            return output_t(m_ptr, m_strides.get());
        }

        /// Returns an Accessor and its corresponding size/shape.
        /// \details While constructing the accessor, this function can also reinterpret the current view.
        ///          This is only well defined in cases where View::as<T0>() is well defined. If N < 4,
        ///          the outer-dimensions are stacked together.
        template<typename T0, int N, typename I0 = index_type,
                 AccessorTraits NEW_TRAITS = AccessorTraits::DEFAULT>
        [[nodiscard]] constexpr auto accessor() const {
            using output_shape_t = std::conditional_t<N == 4, Int4<I0>,
                                   std::conditional_t<N == 3, Int3<I0>,
                                   std::conditional_t<N == 2, Int2<I0>, I0>>>;
            using output_accessor_t = Accessor<T0, N, I0, NEW_TRAITS>;
            using output_t = std::pair<output_accessor_t, output_shape_t>;

            const auto reinterpreted = indexing::Reinterpret<value_type, I0>(
                    m_shape, m_strides, get()).template as<T0>();

            constexpr I0 STRIDES_OFFSET = 4 - N;
            if constexpr (N == 4) {
                return output_t{output_accessor_t(reinterpreted.ptr, reinterpreted.strides.get(STRIDES_OFFSET)),
                                output_shape_t(reinterpreted.shape.get(STRIDES_OFFSET))};
            } else {
                Int4<I0> new_shape(1);
                for (index_type i = 0; i < 4; ++i)
                    new_shape[math::max(i, STRIDES_OFFSET)] *= reinterpreted.shape[i];

                Int4<I0> new_stride;
                if (!indexing::reshape(reinterpreted.shape, reinterpreted.strides, new_shape, new_stride))
                    NOA_THROW("A view of shape {} and stride {} cannot be reshaped to a view of shape {}",
                              reinterpreted.shape, reinterpreted.strides, new_shape);

                output_shape_t output_shape;
                if constexpr (N == 1)
                    output_shape = new_shape[3];
                else
                    output_shape = output_shape_t(new_shape.get(STRIDES_OFFSET));

                return output_t{output_accessor_t(reinterpreted.ptr, new_stride.get(STRIDES_OFFSET)),
                                output_shape};
            }
        }

        /// Reinterpret the managed array of \p T as an array of \p T0.
        /// \note This is only well defined in cases where reinterpret_cast<T0*>(T*) is well defined, for instance,
        ///       when \p T0 is a unsigned char or std::byte to represent any data type as an array of bytes,
        ///       or to switch between complex and real floating-point numbers with the same precision.
        template<typename T0, typename I0 = I>
        View<T0, I0> as() const {
            const auto out = indexing::Reinterpret<T, I0>(m_shape, m_strides, get()).template as<T0>();
            return {out.ptr, out.shape, out.strides};
        }

        /// Reshapes the view.
        /// \param shape New shape. Must contain the same number of elements as the current shape.
        View reshape(const index4_type& shape) const {
            index4_type new_stride;
            if (!indexing::reshape(m_shape, m_strides, shape, new_stride))
                NOA_THROW("A view of shape {} and stride {} cannot be reshaped to a view of shape {}",
                          m_shape, m_strides, shape);
            return {m_ptr, shape, new_stride};
        }

        /// Reshapes the array in a vector along a particular axis.
        /// Returns a row vector by default.
        View flat(int axis) const {
            index4_type output_shape(1);
            output_shape[axis] = m_shape.elements();
            return reshape(output_shape);
        }

        /// Permutes the dimensions of the view.
        /// \param permutation  Permutation with the axes numbered from 0 to 3.
        View permute(const index4_type& permutation) const {
            return {m_ptr,
                    indexing::reorder(m_shape, permutation),
                    indexing::reorder(m_strides, permutation)};
        }

    public: // Setters
        /// (Re)Sets the shape.
        NOA_HD constexpr void shape(size4_t shape) noexcept { m_shape = shape; }

        /// (Re)Sets the strides.
        NOA_HD constexpr void strides(size4_t strides) noexcept { m_strides = strides; }

        /// (Re)Sets the viewed data.
        NOA_HD constexpr void data(T* data) noexcept { m_ptr = data; }

    public: // Loop-like indexing
        /// Returns a reference at the specified memory \p offset.
        template<typename I0, typename = std::enable_if_t<traits::is_int_v<I0>>>
        [[nodiscard]] NOA_HD constexpr T& operator[](I0 offset) const noexcept {
            NOA_ASSERT(!empty() && static_cast<index_type>(offset) <= indexing::at(m_shape - 1, m_strides));
            return m_ptr[offset];
        }

        /// Returns a reference at the beginning of the specified batch index.
        template<typename I0>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0) const noexcept {
            NOA_ASSERT(static_cast<index_type>(i0) < m_shape[0]);
            return accessor_reference_type(m_ptr, m_strides.get())(i0);
        }

        /// Returns a reference at the beginning of the specified batch and depth indexes.
        template<typename I0, typename I1>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0, I1 i1) const noexcept {
            NOA_ASSERT(static_cast<index_type>(i0) < m_shape[0] &&
                       static_cast<index_type>(i1) < m_shape[1]);
            return accessor_reference_type(m_ptr, m_strides.get())(i0, i1);
        }

        /// Returns a reference at the beginning of the specified batch, depth and height indexes.
        template<typename I0, typename I1, typename I2>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0, I1 i1, I2 i2) const noexcept {
            NOA_ASSERT(static_cast<index_type>(i0) < m_shape[0] &&
                       static_cast<index_type>(i1) < m_shape[1] &&
                       static_cast<index_type>(i2) < m_shape[2]);
            return accessor_reference_type(m_ptr, m_strides.get())(i0, i1, i2);
        }

        /// Returns a reference at the beginning of the specified batch, depth, height and width indexes.
        template<typename I0, typename I1, typename I2, typename I3>
        [[nodiscard]] NOA_HD constexpr T& operator()(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
            NOA_ASSERT(static_cast<index_type>(i0) < m_shape[0] &&
                       static_cast<index_type>(i1) < m_shape[1] &&
                       static_cast<index_type>(i2) < m_shape[2] &&
                       static_cast<index_type>(i3) < m_shape[3]);
            return accessor_reference_type(m_ptr, m_strides.get())(i0, i1, i2, i3);
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
        index4_type m_shape;
        index4_type m_strides;
        T* m_ptr{};
    };
}
