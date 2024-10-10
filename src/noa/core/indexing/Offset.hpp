#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Enums.hpp" // Border
#include "noa/core/indexing/Layout.hpp" // order
#include "noa/core/types/Shape.hpp"
#include "noa/core/utils/SafeCast.hpp"

namespace noa::inline types {
    enum class PointerTraits { DEFAULT, RESTRICT }; // TODO ATOMIC?
    enum class StridesTraits { STRIDED, CONTIGUOUS };
}

namespace noa::indexing {
    /// Returns the memory offset corresponding to the given indices.
    /// \param strides  Strides associated with these indices.
    /// \param indices  Multi-dimensional indices.
    template<nt::integer T, size_t N, size_t A, nt::integer... I> requires (N >= sizeof...(I))
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
        const Strides<T, N, A>& strides,
        I... indices
    ) noexcept -> T {
        NOA_ASSERT((is_safe_cast<T>(indices) and ...));
        return [&strides]<size_t... J>(std::index_sequence<J...>, auto&... indices_) {
            return ((static_cast<T>(indices_) * strides[J]) + ...);
        }(std::make_index_sequence<sizeof...(I)>{}, indices...); // nvcc bug - capture indices fails
    }

    /// Returns the memory offset corresponding to the given indices.
    /// \param strides  Strides associated with these indices.
    /// \param indices  Multi-dimensional indices.
    template<nt::integer T, size_t N0, size_t A0, nt::integer U, size_t N1, size_t A1> requires (N0 >= N1)
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
        const Strides<T, N0, A0>& strides,
        const Vec<U, N1, A1>& indices
    ) noexcept -> T {
        NOA_ASSERT((is_safe_cast<Vec<T, N1>>(indices)));
        return [&]<size_t... I>(std::index_sequence<I...>) {
            return ((static_cast<T>(indices[I]) * strides[I]) + ...);
        }(std::make_index_sequence<N1>{});
    }

    /// Returns the memory offset corresponding to the given 1D indices.
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
        nt::integer auto stride,
        nt::integer auto index
    ) noexcept {
        using int_t = decltype(stride);
        NOA_ASSERT(is_safe_cast<int_t>(index));
        return static_cast<int_t>(index) * stride;
    }

    template<nt::integer... I, nt::indexer_compatible<sizeof...(I)> T>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
        const T& indexable,
        I... indices
    ) noexcept {
        return [&indexable]<size_t... J>(std::index_sequence<J...>, auto&... indices_) {
            typename T::index_type offset{};
            ((offset += ni::offset_at(indexable.template stride<J>(), indices_)), ...);
            return offset;
        }(std::make_index_sequence<sizeof...(indices)>{}, indices...);
    }

    template<nt::integer I, size_t N, size_t A, nt::indexer_compatible<N> T>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
        const T& indexable,
        const Vec<I, N, A>& indexes
    ) noexcept {
        return [&]<size_t... J>(std::index_sequence<J...>) {
            typename T::index_type offset{};
            ((offset += ni::offset_at(indexable.template stride<J>(), indexes[J])), ...);
            return offset;
        }(std::make_index_sequence<N>{});
    }

    template<nt::integer... I, nt::indexer_compatible<sizeof...(I)> T, nt::pointer P>
    [[nodiscard]] NOA_FHD constexpr P offset_pointer(
        const T& indexable,
        P pointer,
        I... indices
    ) noexcept {
        return [&]<size_t... J>(std::index_sequence<J...>, auto&... indices_) {
            ((pointer += ni::offset_at(indexable.template stride<J>(), indices_)), ...);
            return pointer;
        }(std::make_index_sequence<sizeof...(I)>{}, indices...);
    }

    template<nt::integer I, size_t N, size_t A, nt::indexer_compatible<N> T, nt::pointer P>
    [[nodiscard]] NOA_FHD constexpr P offset_pointer(
        const T& indexable,
        P pointer,
        const Vec<I, N, A>& indices
    ) noexcept {
        return [&]<size_t... J>(std::index_sequence<J...>) {
            ((pointer += ni::offset_at(indexable.template stride<J>(), indices[J])), ...);
            return pointer;
        }(std::make_index_sequence<N>{});
    }

    /// CRTP-type adding indexing related member functions to the type T.
    template<typename T, size_t N>
    struct Indexer {
        template<typename... I> requires nt::offset_indexing<N, I...>
        NOA_HD constexpr auto& offset_inplace(const I&... indices) noexcept {
            auto& self = static_cast<T&>(*this);
            NOA_ASSERT(not self.is_empty());
            self.reset_pointer(ni::offset_pointer(self, self.get(), indices...));
            return self;
        }

        template<nt::pointer P, typename... I> requires nt::offset_indexing<N, I...>
        [[nodiscard]] NOA_HD constexpr P offset_pointer(P pointer, const I&... indices) const noexcept {
            auto& self = static_cast<const T&>(*this);
            return ni::offset_pointer(self, pointer, indices...);
        }

        template<typename... I> requires nt::offset_indexing<N, I...>
        [[nodiscard]] NOA_HD constexpr auto offset_at(const I&... indices) const noexcept {
            auto& self = static_cast<const T&>(*this);
            return ni::offset_at(self, indices...);
        }

        template<typename... I> requires nt::iwise_general_indexing<N, I...>
        [[nodiscard]] NOA_HD constexpr auto& operator()(const I&... indices) const noexcept {
            auto& self = static_cast<const T&>(*this);
            NOA_ASSERT(not self.is_empty()); // TODO Check is_inbound if self.shape() is valid?
            return *ni::offset_pointer(self, self.get(), indices...);
        }
    };
}

namespace noa::indexing {
    /// If \p idx is out-of-bound, computes a valid index, i.e. [0, size-1], according to \p MODE.
    /// Otherwise, returns \p idx. \p size should be > 0.
    template<Border MODE, nt::sinteger T>
    [[nodiscard]] NOA_HD constexpr T index_at(T idx, T size) noexcept {
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

    template<Border MODE, nt::sinteger T, size_t N, size_t A0, size_t A1>
    [[nodiscard]] NOA_HD constexpr auto index_at(
        const Vec<T, N, A0>& indices,
        const Shape<T, N, A1>& shape
    ) noexcept {
        Vec<T, N, A0> out;
        for (size_t i{}; i < N; ++i)
            out[i] = index_at<MODE>(indices[i], shape[i]);
        return out;
    }

    /// Returns the 2d rightmost indexes corresponding to
    /// the given memory offset in a contiguous layout.
    template<nt::integer T>
    [[nodiscard]] NOA_HD constexpr auto offset2index(T offset, T size) noexcept -> Vec2<T> {
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
    [[nodiscard]] NOA_HD constexpr auto offset2index(T offset, T s0, T s1) noexcept -> Vec3<T> {
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
    [[nodiscard]] NOA_HD constexpr auto offset2index(T offset, T s0, T s1, T s2) noexcept -> Vec4<T> {
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
    template<nt::integer T, size_t N>
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
    template<bool ASSUME_RIGHTMOST = false, nt::integer T, size_t N>
    [[nodiscard]] NOA_HD constexpr auto offset2index(
        std::type_identity_t<T> offset,
        const Strides<T, N>& strides,
        const Shape<T, N>& shape
    ) noexcept -> Vec<T, N> {
        NOA_ASSERT(not shape.is_empty());
        Vec<T, N> out{};
        T remain = offset;

        if constexpr (ASSUME_RIGHTMOST) {
            for (size_t i{}; i < N; ++i) {
                if (shape[i] > 1) { // if empty, ignore it.
                    NOA_ASSERT(strides[i] > 0);
                    out[i] = remain / strides[i]; // single-divide optimization should kick in
                    remain %= strides[i]; // or remain -= out[i] * stride
                }
            }
        } else {
            const auto rightmost_order = order(strides, shape);
            for (size_t i{}; i < N; ++i) {
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

    /// Whether the indices are in-bound, i.e. 0 <= indices < shape.
    template<nt::integer T, size_t N0, size_t N1, size_t A0, size_t A1> requires (N1 <= N0)
    [[nodiscard]] NOA_FHD constexpr bool is_inbound(
        const Shape<T, N0, A0>& shape,
        const Vec<T, N1, A1>& indices
    ) noexcept {
        if constexpr (nt::sinteger<T>) {
            for (size_t i{}; i < N1; ++i)
                if (indices[i] < T{} or indices[i] >= shape[i])
                    return false;
        } else {
            for (size_t i{}; i < N1; ++i)
                if (indices[i] >= shape[i])
                    return false;
        }
        return true;
    }

    /// Whether the indices are in-bound, i.e. 0 <= indices < shape.
    template<nt::integer T, size_t N, size_t A, nt::same_as<T>... U> requires (sizeof...(U) <= N)
    [[nodiscard]] NOA_FHD constexpr bool is_inbound(
        const Shape<T, N, A>& shape,
        const U&... indices
    ) noexcept {
        if constexpr (nt::sinteger<T>) {
            return [&shape]<size_t... I>(std::index_sequence<I...>, auto&... indices_) {
                return ((indices_ >= T{} or indices_ < shape[I]) and ...);
            }(std::make_index_sequence<sizeof...(U)>{}, indices...);
        } else {
            return [&shape]<size_t... I>(std::index_sequence<I...>, auto&... indices_) {
                return ((indices_ < shape[I]) and ...);
            }(std::make_index_sequence<sizeof...(U)>{}, indices...);
        }
    }
}
