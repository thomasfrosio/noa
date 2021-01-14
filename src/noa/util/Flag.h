/**
 * @file Flag.h
 * @brief Converts scoped enums into type-safe flags (i.e. bitset like).
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

#include <spdlog/fmt/fmt.h>

#include <bitset>
#include <ostream>
#include <string>

#include "noa/API.h"
#include "noa/util/Constants.h"
#include "noa/util/string/Format.h"
#include "noa/util/traits/BaseTypes.h"

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
        inline constexpr Flag(Enum v) noexcept: m_bitset(static_cast<int_t>(v)) {}

        inline constexpr void operator&=(Flag<Enum> rhs) noexcept { m_bitset &= rhs.m_bitset; }
        inline constexpr void operator^=(Flag<Enum> rhs) noexcept { m_bitset ^= rhs.m_bitset; }
        inline constexpr void operator|=(Flag<Enum> rhs) noexcept { m_bitset |= rhs.m_bitset; }

        [[nodiscard]] inline constexpr Flag<Enum> operator&(Flag<Enum> rhs) const noexcept {
            return static_cast<Enum>(m_bitset & rhs.m_bitset);
        }

        [[nodiscard]] inline constexpr Flag<Enum> operator|(Flag<Enum> rhs) const noexcept {
            return static_cast<Enum>(m_bitset | rhs.m_bitset);
        }

        [[nodiscard]] inline constexpr Flag<Enum> operator^(Flag<Enum> rhs) const noexcept {
            return static_cast<Enum>(m_bitset ^ rhs.m_bitset);
        }

        [[nodiscard]] inline constexpr Flag<Enum> operator~() const noexcept { return static_cast<Enum>(~m_bitset); }

        [[nodiscard]] inline constexpr explicit operator bool() const noexcept { return m_bitset; }

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

    /**
     * Flag<Errno> is NOT a bitset and should not be used as such. Hence this template specialization.
     * Only one error number can be used at the same time (like ERRNO).
     */
    template<>
    class NOA_API Flag<Errno> {
        using int_t = std::underlying_type_t<Errno>;
        int_t err{0}; // Errno::good by default

    public:
        inline constexpr Flag() = default;

        /** Implicitly converting Errno to Flag<Errno>. */
        inline constexpr Flag(Errno v) noexcept: err(static_cast<int_t>(v)) {}

        [[nodiscard]] inline constexpr explicit operator bool() const noexcept { return err; }
        [[nodiscard]] inline constexpr bool operator==(Flag<Errno> rhs) const noexcept { return err == rhs.err; }
        [[nodiscard]] inline constexpr bool operator&&(Flag<Errno> rhs) const noexcept { return err && rhs.err; }
        [[nodiscard]] inline constexpr bool operator||(Flag<Errno> rhs) const noexcept { return err || rhs.err; }

        /** Update if and only if there is no error. */
        inline constexpr Flag<Errno> update(Flag<Errno> candidate) noexcept {
            if (candidate && !err)
                err = candidate.err;
            return *this;
        }

        /** Converts the error number in a human-readable string. */
        [[nodiscard]] std::string toString() const noexcept {
            switch (static_cast<Errno>(err)) {
                case Errno::fail: {
                    return "Errno::fail";
                }
                case Errno::invalid_argument: {
                    return "Errno::invalid_argument";
                }
                case Errno::invalid_size: {
                    return "Errno::invalid_size";
                }
                case Errno::invalid_data: {
                    return "Errno::invalid_data";
                }
                case Errno::invalid_state: {
                    return "Errno::invalid_state";
                }
                case Errno::out_of_range: {
                    return "Errno::out_of_range";
                }
                case Errno::not_supported: {
                    return "Errno::not_supported";
                }
                case Errno::fail_close: {
                    return "Errno::fail_close";
                }
                case Errno::fail_open: {
                    return "Errno::fail_open";
                }
                case Errno::fail_read: {
                    return "Errno::fail_read";
                }
                case Errno::fail_write: {
                    return "Errno::fail_write";
                }
                case Errno::out_of_memory: {
                    return "Errno::out_of_memory";
                }
                case Errno::fail_os: {
                    return "Errno::fail_os";
                }
                default: {
                    return String::format("{}:{}:{}: DEV: Errno \"{}\" was not added",
                                          __FILE__, __FUNCTION__, __LINE__);
                }
            }
        }
    };
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
