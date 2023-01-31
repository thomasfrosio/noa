#pragma once

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Indexing.h"
#include "noa/common/types/Accessor.h"
#include "noa/common/types/Vec.h"
#include "noa/common/types/Shape.h"

namespace noa {
    /// View over a 4-dimensional array.
    /// \details This class is meant to provide a simple 4D shaped representation of some memory region.
    ///          It keeps track of a pointer and the size and stride of each dimension.
    ///          As such, indexing is bound-checked (only in Debug builds).
    ///          It is similar to the Array class, but is non-owning and the data type can be const-qualified.
    template<typename T, typename I>
    class View {
    public:
        using shape_type = Shape<I, N>;
        using accessor_type = Accessor<T, N, I, PointerTrait, StridesTrait>;
        using pointer_type = typename accessor_type::pointer_type;
        using value_type = typename accessor_type::value_type;
        using index_type = typename accessor_type::index_type;
        using strides_type = typename accessor_type::strides_type;

        static constexpr index_type COUNT = N;
        static constexpr size_t SIZE = N;
        static constexpr int64_t SSIZE = N;
        static constexpr bool IS_RESTRICT = PointerTrait == PointerTraits::RESTRICT;
        static constexpr bool IS_CONTIGUOUS = StridesTrait == StridesTraits::CONTIGUOUS;

    public: // Constructors
        // Creates an empty view.
        NOA_HD constexpr View() = default;

        // Creates a view.
        NOA_HD constexpr View(
                T* data,
                const shape_type& shape,
                const Strides<index_type, SIZE>& strides)
                : m_accessor(data, strides), m_shape(shape) {}

        // Creates a contiguous view, thus assuming the innermost dimension has a stride of 1.
        template<typename Void,
                 typename = std::enable_if_t<
                         (SIZE > 1) && IS_CONTIGUOUS && std::is_void_v<Void>>>
        NOA_HD constexpr View(
                pointer_type pointer,
                const shape_type& shape,
                const Strides<index_type, SIZE - 1>& strides) noexcept
                : m_accessor(pointer, strides), m_shape(shape) {}

        // Creates a contiguous 1D view, thus assuming the stride is 1.
        template<typename Void,
                 typename = std::enable_if_t<
                         (SIZE == 1) && IS_CONTIGUOUS && std::is_void_v<Void>>>
        NOA_HD constexpr View(
                pointer_type pointer,
                const shape_type& shape) noexcept
                : m_accessor(pointer), m_shape(shape) {}

        // Creates a const view from an existing non-const view.
        template<typename U, typename = std::enable_if_t<details::is_mutable_value_type_v<U, value_type>>>
        NOA_HD constexpr /* implicit */ View(const View<U, N, I>& view)
                : m_accessor(view.data(), view.strides()), m_shape(view.shape()) {}

    public: // Getters
        [[nodiscard]] NOA_HD constexpr pointer_type get() const noexcept { return m_accessor.get(); }
        [[nodiscard]] NOA_HD constexpr pointer_type data() const noexcept { return m_accessor.data(); }
        [[nodiscard]] NOA_HD constexpr shape_type& shape() noexcept { return m_shape; }
        [[nodiscard]] NOA_HD constexpr const shape_type& shape() const noexcept { return m_shape; }
        [[nodiscard]] NOA_HD constexpr strides_type& strides() noexcept { return m_accessor.strides(); }
        [[nodiscard]] NOA_HD constexpr const strides_type& strides() const noexcept { return m_accessor.strides(); }
        [[nodiscard]] NOA_HD constexpr index_type elements() const noexcept { return m_shape.elements(); }
        [[nodiscard]] NOA_HD constexpr index_type size() const noexcept { return elements(); }

        template<char ORDER = 'C'>
        [[nodiscard]] NOA_HD constexpr bool are_contiguous() const noexcept {
            return noa::indexing::are_contiguous<ORDER>(m_accessor.strides(), m_shape);
        }

        template<char ORDER = 'C'>
        [[nodiscard]] NOA_HD constexpr bool is_contiguous() const noexcept {
            return noa::indexing::is_contiguous<ORDER>(m_accessor.strides(), m_shape);
        }

        /// Whether the view is empty. A View is empty if not initialized,
        /// or if the viewed data is null, or if one of its dimension is 0.
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept {
            return !m_ptr || noa::any(m_shape == 0);
        }

    public: // Data reinterpretation
        /// Returns a ND Accessor.
        [[nodiscard]] constexpr auto accessor() const {
            using output_t = Accessor<T, N, index_type, AccessorTraits::DEFAULT>;
            return output_t(m_ptr, m_strides);
        }

        /// Returns a new View<NewT, NewN, NewI, NewTraits>.
        /// \details While constructing the accessor, this function can also reinterpret the current view.
        ///          This is only well defined in cases where View::as<T0>() is well defined. If N < 4,
        ///          the outer-dimensions are stacked together.
        template<typename NewT, int NewN, typename NewI = index_type,
                AccessorTraits NewTraits = AccessorTraits::DEFAULT>
        [[nodiscard]] constexpr auto view() const {

        }

        /// Returns an Accessor and its corresponding size/shape.
        /// \details While constructing the accessor, this function can also reinterpret the current view.
        ///          This is only well defined in cases where View::as<T0>() is well defined. If N < 4,
        ///          the outer-dimensions are stacked together.
        template<typename T0, int NDIM, typename I0 = index_type,
                 AccessorTraits NEW_TRAITS = AccessorTraits::DEFAULT>
        [[nodiscard]] constexpr auto as() const {
            using output_shape_t = Vec<I0, NDIM>;
            using output_accessor_t = Accessor<T0, N, I0, NEW_TRAITS>;
            using output_t = std::pair<output_accessor_t, output_shape_t>;

            const auto reinterpreted = indexing::Reinterpret<value_type, I0>(
                    m_shape, m_strides, get()).template as<T0>();

            constexpr I0 STRIDES_OFFSET = 4 - N;
            if constexpr (N == 4) {
                return output_t{output_accessor_t(reinterpreted.ptr, reinterpreted.strides.get(STRIDES_OFFSET)),
                                output_shape_t(reinterpreted.shape.get(STRIDES_OFFSET))};
            } else {
                Vec4<I0> new_shape{1};
                for (index_type i = 0; i < 4; ++i)
                    new_shape[math::max(i, STRIDES_OFFSET)] *= reinterpreted.shape[i];

                Vec4<I0> new_stride{};
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
        template<typename NewT, typename NewI = index_type>
        View<NewT, N, NewI> as() const {
            const auto out = indexing::Reinterpret<T, NewI>(m_shape, m_strides, get()).template as<T0>();
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
                 typename = std::enable_if_t<indexing::Subregion::are_indexer_v<A, B, C, D>>>
        constexpr View subregion(A&& i0, B&& i1 = {}, C&& i2 = {}, D&& i3 = {}) const {
            const auto indexer = indexing::Subregion(m_shape, m_strides).extract(i0, i1, i2, i3);
            return {m_ptr + indexer.offset(), indexer.shape(), indexer.strides()};
        }

        constexpr View subregion(indexing::ellipsis_t) const {
            return *this;
        }

        template<typename A,
                 typename = std::enable_if_t<indexing::Subregion::is_indexer_v<A>>>
        constexpr View subregion(indexing::ellipsis_t, A&& i3) const {
            return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, indexing::full_extent_t{}, i3);
        }

        template<typename A, typename B,
                 typename = std::enable_if_t<indexing::Subregion::are_indexer_v<A, B>>>
        constexpr View subregion(indexing::ellipsis_t, A&& i2, B&& i3) const {
            return subregion(indexing::full_extent_t{}, indexing::full_extent_t{}, i2, i3);
        }

        template<typename A, typename B, typename C,
                 typename = std::enable_if_t<indexing::Subregion::are_indexer_v<A, B, C>>>
        constexpr View subregion(indexing::ellipsis_t, A&& i1, B&& i2, C&& i3) const {
            return subregion(indexing::full_extent_t{}, i1, i2, i3);
        }

    private:
        accessor_type m_accessor;
        shape_type m_shape{};
    };
}
