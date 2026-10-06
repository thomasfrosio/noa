#pragma once

#include <utility>

#include "noa/base/Config.hpp"
#include "noa/base/Traits.hpp"

// C++20 backport
namespace noa {
    template<typename T>
    NOA_FHD constexpr auto to_underlying(T value) noexcept {
        return static_cast<std::underlying_type_t<T>>(value);
    }
}

// C++23 backport
namespace noa {
    template<class T, class U>
    [[nodiscard]] constexpr auto&& forward_like(U&& x) noexcept {
        constexpr bool is_adding_const = std::is_const_v<std::remove_reference_t<T>>;
        if constexpr (std::is_lvalue_reference_v<T&&>) {
            if constexpr (is_adding_const)
                return std::as_const(x);
            else
                return static_cast<U&>(x);
        } else {
            if constexpr (is_adding_const)
                return std::move(std::as_const(x));
            else
                return std::move(x);
        }
    }
}

// Static for each
namespace noa {
    template<typename Integer, Integer... I, typename Op, typename... Args>
    void static_for_each(std::integer_sequence<Integer, I...>, Op&& op, Args&&... args) {
        (op.template operator()<I>(args...), ...);
    }

    template<usize N, typename Integer = usize, typename Op, typename... Args>
    void static_for_each(Op&& op, Args&&... args) {
        static_for_each(std::make_integer_sequence<Integer, N>{}, std::forward<Op>(op), args...);
    }
}

namespace noa {
    // Simple constexpr bit reference.
    template<nt::uinteger T>
    struct BitReference {
    private:
        T* m_bits;
        usize m_offset;

    public:
        static constexpr auto is_set(const T* bits, usize offset) noexcept -> bool {
            return (*bits & (T{1} << offset)) != 0;
        }

        constexpr explicit BitReference(T* bits, usize offset) : m_bits(bits), m_offset(offset) {}

        constexpr /*implicit*/ operator bool() const noexcept {
            return is_set(m_bits, m_offset);
        }

        constexpr auto is_set() const noexcept -> bool {
            return is_set(m_bits, m_offset);
        }

        constexpr void set(bool value) noexcept {
            using mask_t = std::conditional_t<(sizeof(T) <= 4), u32, T>;
            const auto mask = mask_t{1} << m_offset;
            auto result = static_cast<mask_t>(*m_bits);
            if (value)
                result |= mask; // set bit
            else
                result &= ~mask; // clear bit
            *m_bits = static_cast<T>(result);
        }

        constexpr auto operator=(bool value) noexcept -> BitReference& {
            set(value);
            return *this;
        }
    };
}
