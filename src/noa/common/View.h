#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/types/IntX.h"
#include "noa/common/Offset.h"

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
}

namespace noa {
    /// View of a 4-dimensional array.
    /// \tparam T Any (const-qualified) data type.
    template<typename T>
    class View {
    private:
        static_assert(!std::is_pointer_v<T>);
        static_assert(!std::is_reference_v<T>);

        template<typename I>
        static constexpr bool is_indexer_v =
                std::bool_constant<noa::traits::is_int_v<I> ||
                                   noa::traits::is_same_v<I, indexing::full_extent_t> ||
                                   noa::traits::is_same_v<I, indexing::slice_t>>::value;

    public:
        using value_t = T;
        using ptr_t = T*;
        using ref_t = T&;

    public: // Constructors
        /// Creates an empty view.
        View() = default;

        /// Creates a (strided) view.
        template<typename I1, typename I2, typename = std::enable_if_t<traits::is_int4_v<I1> && traits::is_int4_v<I2>>>
        View(T* data, I1 stride, I2 shape) : m_stride(stride), m_shape(shape), m_data(data) {}

        /// Creates a contiguous view.
        template<typename I, typename = std::enable_if_t<traits::is_int4_v<I>>>
        View(T* data, I shape) : m_stride(shape.stride()), m_shape(shape), m_data(data) {}

        /// Creates a const view from an existing non-const view.
        template<typename U, typename = std::enable_if_t<std::is_const_v<T> && std::is_same_v<U, std::remove_const_t<T>>>>
        /* implicit */ View(const View<U>& view) : m_stride(view.stride()), m_shape(view.shape()), m_data(view.data()) {}

    public: // Getters
        [[nodiscard]] T* data() const noexcept { return m_data; }
        [[nodiscard]] T* get() const noexcept { return m_data; }
        [[nodiscard]] const size4_t& shape() const noexcept { return m_shape; }
        [[nodiscard]] const size4_t& stride() const noexcept { return m_stride; }
        [[nodiscard]] bool4_t contiguous() const noexcept { return isContiguous(m_stride, m_shape); }
        [[nodiscard]] bool empty() const noexcept { return !(m_data && m_shape.elements());}

    public: // Setters
        void data(T* data) noexcept { m_data = data; }
        void stride(size4_t stride) noexcept { m_stride = stride; }
        void shape(size4_t shape) noexcept { m_shape = shape; }

    public: // Contiguous views
        [[nodiscard]] T* begin() const noexcept { return m_data; }
        [[nodiscard]] T* end() const noexcept { return m_data + m_shape.elements(); }

        [[nodiscard]] T& front() const noexcept { return *begin(); }
        [[nodiscard]] T& back() const noexcept { return *(end() - 1); }

    public: // Loop-like indexing
        template<typename I>
        [[nodiscard]] constexpr T& operator()(I i0) const noexcept {
            return m_data[static_cast<size_t>(i0) * m_stride[0]];
        }

        template<typename I, typename J>
        [[nodiscard]] constexpr T& operator()(I i0, J i1) const noexcept {
            return m_data[at(i0, i1, m_stride)];
        }

        template<typename I, typename J, typename K>
        [[nodiscard]] constexpr T& operator()(I i0, J i1, K i2) const noexcept {
            return m_data[at(i0, i1, i2, m_stride)];
        }

        template<typename I, typename J, typename K, typename L>
        [[nodiscard]] constexpr T& operator()(I i0, J i1, K i2, L i3) const noexcept {
            return m_data[at(i0, i1, i2, i3, m_stride)];
        }

    public: // Subview
        template<typename A,
                 typename B = indexing::full_extent_t,
                 typename C = indexing::full_extent_t,
                 typename D = indexing::full_extent_t,
                 typename = std::enable_if_t<is_indexer_v<A> && is_indexer_v<B> &&
                                             is_indexer_v<C> && is_indexer_v<D>>>
        constexpr View subview(A&& i0, B&& i1 = {}, C&& i2 = {}, D&& i3 = {}) const {
            int64_t offset{};
            long4_t new_shape;
            long4_t new_stride;

            indexDim_(i0, 0, &offset, new_stride.get() + 0, new_shape.get() + 0);
            indexDim_(i1, 1, &offset, new_stride.get() + 1, new_shape.get() + 1);
            indexDim_(i2, 2, &offset, new_stride.get() + 2, new_shape.get() + 2);
            indexDim_(i3, 3, &offset, new_stride.get() + 3, new_shape.get() + 3);

            return {m_data + offset, new_stride, new_shape};
        }

        constexpr View subview(indexing::ellipsis_t) const {
            return *this;
        }

        template<typename A, typename = std::enable_if_t<is_indexer_v<A>>>
        constexpr View subview(indexing::ellipsis_t, A&& i3) const {
            return subview(indexing::full_extent_t{}, indexing::full_extent_t{}, indexing::full_extent_t{}, i3);
        }

        template<typename A, typename B,
                 typename = std::enable_if_t<is_indexer_v<A> && is_indexer_v<B>>>
        constexpr View subview(indexing::ellipsis_t, A&& i2, B&& i3) const {
            return subview(indexing::full_extent_t{}, indexing::full_extent_t{}, i2, i3);
        }

        template<typename A, typename B, typename C,
                 typename = std::enable_if_t<is_indexer_v<A> && is_indexer_v<B> && is_indexer_v<C>>>
        constexpr View subview(indexing::ellipsis_t, A&& i1, B&& i2, C&& i3) const {
            return subview(indexing::full_extent_t{}, i1, i2, i3);
        }

        template<typename A, typename B, typename C, typename D,
                 typename = std::enable_if_t<noa::traits::is_int_v<A> && noa::traits::is_int_v<B> &&
                                             noa::traits::is_int_v<C> && noa::traits::is_int_v<D>>>
        constexpr View reorder(A a, B b, C c, D d) const noexcept {
            NOA_ASSERT(a >= 0 && a < 3 && b >= 0 && b < 3 && c >= 0 && c < 3 && d >= 0 && d < 3);
            return {m_data,
                    size4_t{m_stride[a], m_stride[b], m_stride[c], m_stride[d]},
                    size4_t{m_shape[a], m_shape[b], m_shape[c], m_shape[d]}};
        }

    private:
        template<typename I>
        constexpr void indexDim_(I idx_mode, int dim, int64_t* new_offset, int64_t* new_stride, int64_t* new_size) const {
            const auto old_size = static_cast<int64_t>(m_shape[dim]);
            const auto old_stride = static_cast<int64_t>(m_stride[dim]);

            if constexpr (traits::is_int_v<I>) {
                auto index = clamp_cast<int64_t>(idx_mode);

                if (index < -old_size || index >= old_size)
                    NOA_THROW("Index {} is out of range for a size of {} at dimension {}",
                              index, old_size, dim);

                if (index < 0)
                    index += old_size;
                *new_stride = old_stride; // or 0
                *new_size = 1;
                *new_offset += old_stride * index;

            } else if constexpr(std::is_same_v<indexing::full_extent_t, I>) {
                *new_stride = old_stride;
                *new_size = old_size;
                *new_offset += 0;
                (void) idx_mode;

            } else if constexpr(std::is_same_v<indexing::slice_t, I>) {
                NOA_CHECK(idx_mode.step > 0, "Slice step must be positive, got {}", idx_mode.step);

                if (idx_mode.start < 0)
                    idx_mode.start += old_size;
                if (idx_mode.end < 0)
                    idx_mode.end += old_size;

                idx_mode.start = noa::math::clamp(idx_mode.start, int64_t{0}, old_size);
                idx_mode.end = noa::math::clamp(idx_mode.end, idx_mode.start, old_size);

                *new_offset += idx_mode.start * old_stride;
                *new_size = noa::math::divideUp(idx_mode.end - idx_mode.start, idx_mode.step);
                *new_stride = old_stride * idx_mode.step;
            } else {
                static_assert(traits::always_false_v<I>);
            }
        }

    private:
        size4_t m_stride;
        size4_t m_shape;
        T* m_data{};
    };
}
