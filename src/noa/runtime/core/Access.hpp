#pragma once

#include "noa/runtime/core/Border.hpp"
#include "noa/runtime/core/Shape.hpp"

namespace noa::inline types {
    enum class PointerTraits { DEFAULT, RESTRICT }; // TODO ATOMIC?
    enum class StridesTraits { STRIDED, CONTIGUOUS };
}

namespace noa {
    /// Returns the memory offset corresponding to the given index and stride.
    /// \note The common_type is used for the multiplication and is also returned.
    /// \note This is UB if the result of the multiplication is out-of-range and if the common_type is signed.
    template<nt::integer T, nt::integer U>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(T stride, U index) noexcept {
        using common_t = std::common_type_t<T, U>;
        return static_cast<common_t>(index) * static_cast<common_t>(stride);
    }

    /// Returns the memory offset corresponding to the given indices.
    /// \param strides  Strides associated with these indices.
    /// \param indices  Multi-dimensional indices.
    /// \note If the resulting offset is used for pointer arithmetic, prefer to use the safer offset_pointer.
    template<nt::integer T, usize N, usize A, nt::integer... I> requires (N >= sizeof...(I))
    [[nodiscard]] NOA_FHD constexpr auto offset_at(const Strides<T, N, A>& strides, I... indices) noexcept {
        return [&strides]<usize... J>(std::index_sequence<J...>, auto&... indices_) {
            return (noa::offset_at(strides[J], indices_) + ...);
        }(std::make_index_sequence<sizeof...(I)>{}, indices...); // nvcc bug - capture indices fails
    }

    /// Returns the memory offset corresponding to the given indices.
    /// \param strides  Strides associated with these indices.
    /// \param indices  Multi-dimensional indices.
    /// \note If the resulting offset is used for pointer arithmetic, prefer to use the safer offset_pointer.
    template<nt::integer T, usize N0, usize A0, nt::integer U, usize N1, usize A1> requires (N0 >= N1)
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
        const Strides<T, N0, A0>& strides,
        const Vec<U, N1, A1>& indices
    ) noexcept {
        return [&]<usize... I>(std::index_sequence<I...>) {
            return (noa::offset_at(strides[I], indices[I]) + ...);
        }(std::make_index_sequence<N1>{});
    }

    template<nt::integer... I, nt::indexable_nd<sizeof...(I)> T>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
        const T& indexable,
        I... indices
    ) noexcept {
        return [&indexable]<usize... J>(std::index_sequence<J...>, auto&... indices_) {
            typename T::index_type offset{};
            ((offset += noa::offset_at(indexable.template stride<J>(), indices_)), ...);
            return offset;
        }(std::make_index_sequence<sizeof...(indices)>{}, indices...);
    }

    template<nt::integer I, usize N, usize A, nt::indexable_nd<N> T>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
        const T& indexable,
        const Vec<I, N, A>& indices
    ) noexcept {
        return [&]<usize... J>(std::index_sequence<J...>) {
            typename T::index_type offset{};
            ((offset += noa::offset_at(indexable.template stride<J>(), indices[J])), ...);
            return offset;
        }(std::make_index_sequence<N>{});
    }

    /// Returns the memory offset corresponding to the given indices.
    /// \param indexable    Indexable object, providing the strides. See indexable.
    /// \param pointer      Pointer to offset.
    /// \param indices      Multi-dimensional indices.
    /// \note If the resulting pointer is to be dereferenced, prefer to use the safer read or read_at.
    template<nt::integer... I, nt::indexable_nd<sizeof...(I)> T, nt::pointer P>
    [[nodiscard]] NOA_FHD constexpr auto offset_pointer(
        const T& indexable,
        P pointer,
        I... indices
    ) noexcept -> P {
        return [&]<usize... J>(std::index_sequence<J...>, auto&... indices_) {
            ((pointer += noa::offset_at(indexable.template stride<J>(), indices_)), ...);
            return pointer;
        }(std::make_index_sequence<sizeof...(I)>{}, indices...);
    }

    template<nt::integer I, usize N, usize A, nt::indexable_nd<N> T, nt::pointer P>
    [[nodiscard]] NOA_FHD constexpr auto offset_pointer(
        const T& indexable,
        P pointer,
        const Vec<I, N, A>& indices
    ) noexcept -> P {
        return [&]<usize... J>(std::index_sequence<J...>) {
            ((pointer += noa::offset_at(indexable.template stride<J>(), indices[J])), ...);
            return pointer;
        }(std::make_index_sequence<N>{});
    }

    template<typename T, typename... I> requires nt::indexable<T, I...>
    NOA_HD constexpr auto offset_inplace(T& indexable, const I&... indices) noexcept -> T& {
        if constexpr (requires { indexable.is_empty(); }) {
            NOA_ASSERT(not indexable.is_empty());
        }
        indexable.reset_pointer(noa::offset_pointer(indexable, indexable.get(), indices...));
        return indexable;
    }

    /// Reads the element at the given indices.
    template<typename T, typename... I> requires nt::indexable<T, I...>
    [[nodiscard]] NOA_FHD constexpr auto read(const T& indexable, const I&... indices) -> auto& {
        if constexpr (requires { indexable.is_empty(); }) {
            NOA_ASSERT(not indexable.is_empty());
        }
        if constexpr (requires { indexable.shape(); })
            bounds_check(indexable.shape(), indices...);

        return *noa::offset_pointer(indexable, indexable.get(), indices...);
    }

    /// Reads the element at the given indices, enforcing bounds check.
    template<typename T, typename... I> requires nt::indexable<T, I...>
    [[nodiscard]] NOA_FHD constexpr auto read_at(const T& indexable, const I&... indices) -> auto& {
        if constexpr (requires { indexable.is_empty(); })
            check(not indexable.is_empty());
        if constexpr (requires { indexable.shape(); })
            bounds_check<true>(indexable.shape(), indices...);

        return *noa::offset_pointer(indexable, indexable.get(), indices...);
    }
}

namespace noa::details {
    /// CRTP-type adding indexing related member functions to the type T.
    /// Both Accessor and Span uses this type to index pointers and read values.
    template<typename T, usize N>
    struct Indexer {
        template<typename... I> requires nt::offset_indexing<N, I...>
        NOA_HD constexpr auto offset_inplace(const I&... indices) noexcept -> T& {
            return noa::offset_inplace(static_cast<T&>(*this), indices...);
        }

        template<nt::pointer P, typename... I> requires nt::offset_indexing<N, I...>
        [[nodiscard]] NOA_HD constexpr auto offset_pointer(P pointer, const I&... indices) const noexcept -> P {
            return noa::offset_pointer(static_cast<const T&>(*this), pointer, indices...);
        }

        template<typename... I> requires nt::offset_indexing<N, I...>
        [[nodiscard]] NOA_HD constexpr auto offset_at(const I&... indices) const noexcept {
            return noa::offset_at(static_cast<const T&>(*this), indices...);
        }

        template<typename... I> requires nt::iwise_indexing<N, I...>
        [[nodiscard]] NOA_HD constexpr auto operator()(const I&... indices) const noexcept -> auto& {
            return noa::read(static_cast<const T&>(*this), indices...);
        }

        template<typename... I> requires nt::iwise_indexing<N, I...>
        [[nodiscard]] NOA_HD constexpr auto at(const I&... indices) const -> auto& {
            return noa::read_at(static_cast<const T&>(*this), indices...);
        }
    };
}

namespace noa {
    /// If \p idx is out-of-bound, computes a valid index, i.e. [0, size-1], according to \p MODE.
    /// Otherwise, returns \p idx. \p size should be > 0.
    template<Border MODE, nt::sinteger T>
    [[nodiscard]] NOA_HD constexpr auto index_at(T idx, T size) noexcept -> T {
        static_assert(MODE == Border::CLAMP or MODE == Border::PERIODIC or
                      MODE == Border::MIRROR or MODE == Border::REFLECT);
        NOA_ASSERT(size > 0);

        // a % b == a - b * (a / b) == a + b * (-a / b)
        // Having a < 0 is well-defined since C++11.
        if constexpr (MODE == Border::CLAMP) {
            if (idx < 0)
                idx = 0;
            else if (idx >= size)
                idx = size - 1;
        } else if constexpr (MODE == Border::PERIODIC) {
            // 0 1 2 3 0 1 2 3 0 1 2 3 |  0 1 2 3  | 0 1 2 3 0 1 2 3 0 1 2 3
            T rem = idx % size;
            idx = rem < 0 ? rem + size : rem;
        } else if constexpr (MODE == Border::MIRROR) {
            // 0 1 2 3 3 2 1 0 0 1 2 3 3 2 1 0 |  0 1 2 3  | 3 2 1 0 0 1 2 3 3 2 1 0
            if (idx < 0)
                idx = -idx - 1;
            if (idx >= size) {
                T period = 2 * size;
                idx %= period;
                if (idx >= size)
                    idx = period - idx - 1;
            }
        } else if constexpr (MODE == Border::REFLECT) {
            // 0 1 2 3 2 1 0 1 2 3 2 1 |  0 1 2 3  | 2 1 0 1 2 3 2 1 0
            if (idx < 0)
                idx = -idx;
            if (idx >= size) {
                T period = 2 * size - 2;
                idx %= period;
                if (idx >= size)
                    idx = period - idx;
            }
        }
        return idx;
    }

    template<Border MODE, nt::sinteger T, usize N, usize A0, usize A1>
    [[nodiscard]] NOA_HD constexpr auto index_at(
        const Vec<T, N, A0>& indices,
        const Shape<T, N, A1>& shape
    ) noexcept {
        Vec<T, N, A0> out;
        for (usize i{}; i < N; ++i)
            out[i] = index_at<MODE>(indices[i], shape[i]);
        return out;
    }

    /// Returns the 2d rightmost indexes corresponding to
    /// the given memory offset in a contiguous layout.
    template<nt::integer T>
    [[nodiscard]] NOA_HD constexpr auto offset2index(T offset, T size) noexcept -> Vec<T, 2> {
        NOA_ASSERT(size > 0);
        const auto i0 = offset / size;
        const auto i1 = offset - i0 * size;
        return {i0, i1};
    }

    /// Returns the 3d rightmost indexes corresponding to
    /// the given memory offset in a contiguous layout.
    /// \param offset   Linear memory offset.
    /// \param s0,s1    DH sizes.
    template<nt::integer T>
    [[nodiscard]] NOA_HD constexpr auto offset2index(T offset, T s0, T s1) noexcept -> Vec<T, 3> {
        NOA_ASSERT(s0 > 0 and s1 > 0);
        const auto i0 = offset / (s0 * s1);
        offset -= i0 * s0 * s1;
        const auto i1 = offset / s1;
        offset -= i1 * s1;
        return {i0, i1, offset};
    }

    /// Returns the 4D rightmost indexes corresponding to
    /// the given memory offset in a contiguous layout.
    /// \param offset   Linear memory offset.
    /// \param s0,s1,s2 DHW sizes.
    template<nt::integer T>
    [[nodiscard]] NOA_HD constexpr auto offset2index(T offset, T s0, T s1, T s2) noexcept -> Vec<T, 4> {
        NOA_ASSERT(s0 > 0 and s1 > 0 and s2 > 0);
        const auto i0 = offset / (s0 * s1 * s2);
        offset -= i0 * s0 * s1 * s2;
        const auto i1 = offset / (s1 * s2);
        offset -= i1 * s1 * s2;
        const auto i2 = offset / s2;
        offset -= i2 * s2;
        return {i0, i1, i2, offset};
    }

    /// Returns the multidimensional indices corresponding to a memory \p offset, assuming BDHW C-contiguity.
    /// \param offset   Memory offset within the array.
    /// \param shape    Shape of the array.
    template<nt::integer T, usize N>
    [[nodiscard]] NOA_FHD constexpr auto offset2index(
        std::type_identity_t<T> offset,
        const Shape<T, N>& shape
    ) noexcept -> Vec<T, N> {
        if constexpr (N == 1) {
            return Vec<T, N>{offset};
        } else if constexpr (N == 2) {
            return offset2index(offset, shape[1]);
        } else if constexpr (N == 3) {
            return offset2index(offset, shape[1], shape[2]);
        } else {
            return offset2index(offset, shape[1], shape[2], shape[3]);
        }
    }

    /// Returns the multidimensional indices corresponding to a memory \p offset.
    /// \details Given a memory layout (i.e. strides and shape), this function computes the nd logical indices
    ///          pointing at the given memory \p offset. Broadcasting is not supported, so the strides should
    ///          be greater than 0. Otherwise, any ordering is supported.
    /// \param offset   Memory offset within the array.
    /// \param strides  Strides of the array.
    /// \param shape    Shape of the array.
    template<bool ASSUME_RIGHTMOST = false, nt::integer T, usize N>
    [[nodiscard]] NOA_HD constexpr auto offset2index(
        std::type_identity_t<T> offset,
        const Strides<T, N>& strides,
        const Shape<T, N>& shape
    ) noexcept -> Vec<T, N> {
        NOA_ASSERT(not shape.is_empty());
        Vec<T, N> out{};
        T remain = offset;

        if constexpr (ASSUME_RIGHTMOST) {
            for (usize i{}; i < N; ++i) {
                if (shape[i] > 1) { // if empty, ignore it.
                    NOA_ASSERT(strides[i] > 0);
                    out[i] = remain / strides[i]; // single-divide optimization should kick in
                    remain %= strides[i]; // or remain -= out[i] * stride
                }
            }
        } else {
            const auto rightmost_order = strides.rightmost_order(shape);
            for (usize i{}; i < N; ++i) {
                const auto idx = rightmost_order[i];
                if (shape[idx] > 1) {
                    NOA_ASSERT(strides[idx] > 0);
                    out[idx] = remain / strides[idx];
                    remain %= strides[idx];
                }
            }
        }

        NOA_ASSERT(remain == 0);
        return out;
    }
}
