/**
 * @file Flag.h
 * @brief Converts scoped enums into type-safe flags (i.e. bitset like).
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

#include <spdlog/fmt/fmt.h>

#include <ostream>
#include <string>

#include "noa/Define.h"
#include "noa/util/string/Format.h"
#include "noa/util/traits/BaseTypes.h"

namespace Noa::Traits {
    template<typename Enum, typename = void>
    struct is_flag_enum : std::false_type {};

    template<typename Enum>
    struct is_flag_enum<Enum, decltype(static_cast<void>(Enum::_flag_size_))> : is_scoped_enum_v<Enum> {};

    // A scoped enum with the field _flag_size_
    template<typename T> constexpr bool is_flag_enum_v = is_flag_enum<T>::value;
}

namespace Noa {
    /**
     * Templated class that converts scoped enumerators into zero-overhead bitsets.
     * @tparam Enum     Scoped enum (enum class). Its fields should define bitmasks and it should
     *                  have a field named @c _flag_size_ with the number of available bits as value.
     *                  The operator bool() evaluates whether or not all bits are 0.
     * @note            Zero-overhead: https://godbolt.org/z/7G9qao
     */
    template<typename Enum, typename = std::enable_if_t<Traits::is_flag_enum_v<Enum>>>
    class Flag {
        using int_t = std::underlying_type_t<Enum>;
        int_t m_bitset{0};

    public:
        /** Creates a flag with all bits turned off. */
        NOA_FHD constexpr Flag() = default;

        /** Implicitly converts Enum to Flag<Enum>. */
        NOA_FHD constexpr Flag(Enum v) noexcept: m_bitset(static_cast<int_t>(v)) {}

        NOA_FHD constexpr void operator&=(Flag<Enum> rhs) noexcept { m_bitset &= rhs.m_bitset; }
        NOA_FHD constexpr void operator^=(Flag<Enum> rhs) noexcept { m_bitset ^= rhs.m_bitset; }
        NOA_FHD constexpr void operator|=(Flag<Enum> rhs) noexcept { m_bitset |= rhs.m_bitset; }

        [[nodiscard]] NOA_FHD constexpr Flag<Enum> operator&(Flag<Enum> rhs) const noexcept {
            return static_cast<Enum>(m_bitset & rhs.m_bitset);
        }

        [[nodiscard]] NOA_FHD constexpr Flag<Enum> operator|(Flag<Enum> rhs) const noexcept {
            return static_cast<Enum>(m_bitset | rhs.m_bitset);
        }

        [[nodiscard]] NOA_FHD constexpr Flag<Enum> operator^(Flag<Enum> rhs) const noexcept {
            return static_cast<Enum>(m_bitset ^ rhs.m_bitset);
        }

        [[nodiscard]] NOA_FHD constexpr Flag<Enum> operator~() const noexcept { return static_cast<Enum>(~m_bitset); }

        [[nodiscard]] NOA_FHD constexpr explicit operator bool() const noexcept { return m_bitset; }

        [[nodiscard]] NOA_FHD constexpr bool operator==(Flag<Enum> rhs) const noexcept {
            return m_bitset == rhs.m_bitset;
        }

        [[nodiscard]] NOA_FHD constexpr bool operator&&(Flag<Enum> rhs) const noexcept {
            return m_bitset && rhs.m_bitset;
        }

        [[nodiscard]] NOA_FHD constexpr bool operator||(Flag<Enum> rhs) const noexcept {
            return m_bitset || rhs.m_bitset;
        }

        [[nodiscard]] NOA_FHD static constexpr size_t size() noexcept {
            return static_cast<std::size_t>(Enum::_flag_size_);
        }

        [[nodiscard]] NOA_HOST std::string toString() const {
            return String::format("{:#b}", m_bitset);
        }
    };

    template<typename Enum, typename = std::enable_if_t<Traits::is_flag_enum_v<Enum>>>
    NOA_IHD constexpr Flag<Enum> operator|(Enum lhs, Enum rhs) {
        Flag<Enum> out(lhs);
        out |= rhs;
        return out;
    }
}

template<typename Enum>
struct fmt::formatter<Noa::Flag<Enum>> : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const Noa::Flag<Enum>& a, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(a.toString(), ctx);
    }
};

template<typename Enum>
inline std::ostream& operator<<(std::ostream& os, const Noa::Flag<Enum>& err) {
    os << err.toString();
    return os;
}
