#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/Error.hpp"

#ifdef NOA_HAS_GCC_STATIC_BOUNDS_CHECKING
namespace noa::indexing::details {
    [[gnu::error("out-of-bounds sequence access detected")]]
    void static_bounds_check_failed(); // not defined

    // https://tristanbrindle.com/posts/compile-time-bounds-checking-in-flux
    // The compilation either fails or the function gets optimized away!
    template<typename T>
    constexpr void static_bounds_check(T size, T index) {
        if (__builtin_constant_p(index)) {
            if constexpr (nt::sinteger<T>)
                if (index < T{})
                    static_bounds_check_failed();

            if (__builtin_constant_p(size))
                if (index >= size)
                    static_bounds_check_failed();
        }
    }
}
#endif

namespace noa::inline types {
    template<typename, usize, usize> class Vec;
    template<typename, usize, usize> class Shape;
}

namespace noa::indexing {
    /// Checks whether the index is out-of-bound.
    /// In release and if ENFORCE==false, this function gets either optimized away or fails to compile.
    /// If size and index are known at compile time (more likely to be true in release mode), the compiler
    /// can raise an error if the index is out-of-bound.
    template<bool ENFORCE = false, nt::integer T, nt::integer U>
    NOA_FHD constexpr void bounds_check(T size, U index) {
        using common_t = std::common_type_t<T, T>;
        #ifdef NOA_HAS_GCC_STATIC_BOUNDS_CHECKING
        details::static_bounds_check(static_cast<common_t>(size), static_cast<common_t>(index));
        #endif

        if constexpr (ENFORCE) {
            if constexpr (nt::sinteger<U>) {
                check(index >= U{} and static_cast<common_t>(index) < static_cast<common_t>(size));
            } else {
                check(static_cast<common_t>(index) < static_cast<common_t>(size));
            }
        } else {
            if constexpr (nt::sinteger<U>) {
                NOA_ASSERT(index >= U{} and static_cast<common_t>(index) < static_cast<common_t>(size));
            } else {
                NOA_ASSERT(static_cast<common_t>(index) < static_cast<common_t>(size));
            }
        }
    }

    template<bool ENFORCE = false, nt::integer T, nt::integer U, usize N0, usize N1, usize A0, usize A1> requires (N1 <= N0)
    NOA_FHD constexpr void bounds_check(const Shape<T, N0, A0>& shape, const Vec<U, N1, A1>& indices) {
        for (usize i{}; i < N1; ++i)
            bounds_check<ENFORCE>(shape[i], indices[i]);
    }

    template<bool ENFORCE = false, nt::integer T, nt::integer... U, usize N, usize A> requires (sizeof...(U) <= N)
    NOA_FHD constexpr void bounds_check(const Shape<T, N, A>& shape, U... indices) {
        [&shape]<usize... I>(std::index_sequence<I...>, auto... indices_) {
            (bounds_check<ENFORCE>(shape[I], indices_), ...);
        }(std::make_index_sequence<sizeof...(U)>{}, indices...); // nvcc bug
    }

    /// Whether the indices are in-bound, i.e., 0 <= indices < shape.
    template<nt::integer T, usize N0, usize N1, usize A0, usize A1> requires (N1 <= N0)
    [[nodiscard]] NOA_FHD constexpr bool is_inbound(
        const Shape<T, N0, A0>& shape,
        const Vec<T, N1, A1>& indices
    ) noexcept {
        if constexpr (nt::sinteger<T>) {
            return indices >= 0 and indices < shape.vec;
        } else {
            return indices < shape.vec;
        }
    }

    /// Whether the indices are in-bound, i.e., 0 <= indices < shape.
    template<nt::integer T, usize N, usize A, nt::same_as<T>... U> requires (sizeof...(U) <= N)
    [[nodiscard]] NOA_FHD constexpr bool is_inbound(
        const Shape<T, N, A>& shape,
        const U&... indices
    ) noexcept {
        if constexpr (nt::sinteger<T>) {
            return Vec{indices...} >= 0 and Vec{indices...} < shape.vec;
        } else {
            return Vec{indices...} < shape.vec;
        }
    }
}
