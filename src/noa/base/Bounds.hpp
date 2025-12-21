#pragma once

#include "noa/base/Config.hpp"
#include "noa/base/Traits.hpp"
#include "noa/base/Error.hpp"

#ifdef NOA_HAS_GCC_STATIC_BOUNDS_CHECKING
namespace noa::details {
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

namespace noa {
    /// Checks whether the index is out-of-bound.
    /// In release and if ENFORCE==false, this function gets either optimized away or fails to compile.
    /// If size and index are known at compile time (more likely to be true in release mode), the compiler
    /// can raise an error if the index is out-of-bound.
    template<bool ENFORCE = false, nt::integer T, nt::integer U>
    NOA_FHD constexpr void bounds_check(T size, U index) {
        using common_t = std::common_type_t<T, T>;
        #ifdef NOA_HAS_GCC_STATIC_BOUNDS_CHECKING
        nd::static_bounds_check(static_cast<common_t>(size), static_cast<common_t>(index));
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

    /// Whether the index is in-bound, i.e., 0 <= index < size.
    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr bool is_inbound(const T size, const T index) noexcept {
        if constexpr (nt::sinteger<T>) {
            return index >= 0 and index < size;
        } else {
            return index < size;
        }
    }
}
