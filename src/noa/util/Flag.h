/**
 * @file Flag.h
 * @brief Converts scoped enums into type-safe flags (i.e. bitset like).
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

#include "noa/API.h"


namespace Noa::Traits {
    template<typename E>
    using is_scoped_enum = std::integral_constant<bool,
            std::is_enum_v<E> && !std::is_convertible_v<E, int>>;

    template<typename E>
    inline constexpr bool is_scoped_enum_v = is_scoped_enum<E>::value;
}


namespace Noa {
    /**
     * Templated class that converts scoped enumerators into zero-overhead bitsets.
     * @tparam Enum     Scoped enum (enum class). Its fields should define bitmasks.
     *                  The operator bool() evaluates whether or not all bits are 0.
     * @note            Zero-overhead: https://godbolt.org/z/7G9qao
     */
    template<typename Enum, typename = std::enable_if_t<Traits::is_scoped_enum_v<Enum>>>
    class NOA_API Flag {
        using int_t = std::underlying_type_t<Enum>;
        int_t m_bitset{0};

    public:
        inline constexpr Flag() = default;

        /** Implicitly converting enum to Flag<enum>. */
        inline constexpr Flag(Enum v) noexcept : m_bitset(static_cast<int_t>(v)) {}

        inline constexpr void operator&=(Flag<Enum> rhs) noexcept {
            m_bitset &= rhs.m_bitset;
        }

        inline constexpr void operator^=(Flag<Enum> rhs) noexcept {
            m_bitset ^= rhs.m_bitset;
        }

        inline constexpr void operator|=(Flag<Enum> rhs) noexcept {
            m_bitset |= rhs.m_bitset;
        }

        [[nodiscard]] inline constexpr Flag<Enum> operator&(Flag<Enum> rhs) const noexcept {
            return static_cast<Enum>(m_bitset & rhs.m_bitset);
        }

        [[nodiscard]] inline constexpr Flag<Enum> operator|(Flag<Enum> rhs) const noexcept {
            return static_cast<Enum>(m_bitset | rhs.m_bitset);
        }

        [[nodiscard]] inline constexpr Flag<Enum> operator^(Flag<Enum> rhs) const noexcept {
            return static_cast<Enum>(m_bitset ^ rhs.m_bitset);
        }

        [[nodiscard]] inline constexpr Flag<Enum> operator~() const noexcept {
            return static_cast<Enum>(~m_bitset);
        }

        [[nodiscard]] inline constexpr explicit operator bool() const noexcept {
            return m_bitset;
        }

        [[nodiscard]] inline constexpr bool operator==(Flag<Enum> rhs) const noexcept {
            return m_bitset == rhs.m_bitset;
        }

        [[nodiscard]] inline constexpr bool operator&&(Flag<Enum> rhs) const noexcept {
            return m_bitset && rhs.m_bitset;
        }

        [[nodiscard]] inline constexpr bool operator||(Flag<Enum> rhs) const noexcept {
            return m_bitset || rhs.m_bitset;
        }

        [[nodiscard]] std::string toString() const {
            std::bitset<sizeof(int_t)> bitset;
            return bitset.to_string();
        }
    };

    template<typename Enum, typename = std::enable_if_t<Traits::is_scoped_enum_v<Enum>>>
    NOA_API inline constexpr Flag<Enum> operator|(Enum lhs, Enum rhs) {
        Flag<Enum> out(lhs);
        out |= rhs;
        return out;
    }
}


template<typename T>
struct fmt::formatter<::Noa::Flag<T>> : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const ::Noa::Flag<T>& a, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(a.toString(), ctx);
    }
};

template<typename Enum>
inline std::ostream& operator<<(std::ostream& os, const ::Noa::Flag<Enum>& err) {
    os << err.toString();
    return os;
}
