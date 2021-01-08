#pragma once

#include <cstdint>
#include <string>
#include <bitset>

#include <spdlog/spdlog.h>

#include "noa/API.h"
#include "noa/util/Flag.h"


namespace Noa {
    /**
     * Error numbers used throughout the @a Noa namespace.
     * Errno should evaluate to @c false if no errors (@c Errno::good), and @c true for errors.
     */
    NOA_API using errno_t = uint32_t;

    struct NOA_API Errno {
        static constexpr errno_t good{0U}; // this one should not change !
        static constexpr errno_t fail{1U};

        static constexpr errno_t invalid_argument{2U};
        static constexpr errno_t invalid_size{3U};
        static constexpr errno_t invalid_data{4U};
        static constexpr errno_t invalid_state{5U};
        static constexpr errno_t out_of_range{6U};
        static constexpr errno_t not_supported{7U};

        // I/O and streams
        static constexpr errno_t fail_close{10U};
        static constexpr errno_t fail_open{11U};
        static constexpr errno_t fail_read{12U};
        static constexpr errno_t fail_write{13U};

        // OS
        static constexpr errno_t out_of_memory{20U};
        static constexpr errno_t fail_os{21U};


        /** Updates the error number if it is Errno::good, otherwise keeps its current value. */
        static inline constexpr errno_t set(errno_t& current, errno_t candidate) noexcept {
            if (candidate && !current)
                current = candidate;
            return current;
        }


        /** Returns a string describing the meaning of the error number. */
        static std::string toString(errno_t err) noexcept {
            switch (err) {
                case Errno::fail:
                    return "Errno::fail";
                case Errno::invalid_argument:
                    return "Errno::invalid_argument";
                case Errno::invalid_size:
                    return "Errno::invalid_size";
                case Errno::invalid_data:
                    return "Errno::invalid_data";
                case Errno::invalid_state:
                    return "Errno::invalid_state";
                case Errno::out_of_range:
                    return "Errno::out_of_range";
                case Errno::not_supported:
                    return "Errno::not_supported";
                case Errno::fail_close:
                    return "Errno::fail_close";
                case Errno::fail_open:
                    return "Errno::fail_open";
                case Errno::fail_read:
                    return "Errno::fail_read";
                case Errno::fail_write:
                    return "Errno::fail_write";
                case Errno::out_of_memory:
                    return "Errno::out_of_memory";
                case Errno::fail_os:
                    return "Errno::fail_os";
                default:
                    return "DEV: Errno::toString() not defined";
            }
        }
    };

    enum class Errno1 {
        good = 0, // this one should not change !

        fail,
        invalid_argument,
        invalid_size,
        invalid_data,
        invalid_state,
        out_of_range,
        not_supported,

        // I/O and streams
        fail_close,
        fail_open,
        fail_read,
        fail_write,

        // OS
        out_of_memory,
        fail_os,
    };

    template<>
    class Flag<Errno1> {
        using int_t = std::underlying_type_t<Errno1>;
        int_t m_bitset{0};

    public:
        inline constexpr Flag() = default;

        /** Implicitly converting enum to Flag<enum>. */
        inline constexpr Flag(Errno1 v) noexcept : m_bitset(static_cast<int_t>(v)) {}


        [[nodiscard]] inline constexpr explicit operator bool() const noexcept {
            return m_bitset;
        }

        [[nodiscard]] inline constexpr bool operator==(Flag<Errno1> rhs) const noexcept {
            return m_bitset == rhs.m_bitset;
        }

        [[nodiscard]] inline constexpr bool operator&&(Flag<Errno1> rhs) const noexcept {
            return m_bitset && rhs.m_bitset;
        }

        [[nodiscard]] inline constexpr bool operator||(Flag<Errno1> rhs) const noexcept {
            return m_bitset || rhs.m_bitset;
        }

        inline constexpr Flag<Errno1> update(Flag<Errno1> candidate) noexcept {
            if (candidate && !m_bitset)
                m_bitset = candidate.m_bitset;
            return *this;
        }

        [[nodiscard]] std::string toString() const noexcept {
            switch (m_bitset) {
                case Errno::fail:
                    return "Errno::fail";
                case Errno::invalid_argument:
                    return "Errno::invalid_argument";
                case Errno::invalid_size:
                    return "Errno::invalid_size";
                case Errno::invalid_data:
                    return "Errno::invalid_data";
                case Errno::invalid_state:
                    return "Errno::invalid_state";
                case Errno::out_of_range:
                    return "Errno::out_of_range";
                case Errno::not_supported:
                    return "Errno::not_supported";
                case Errno::fail_close:
                    return "Errno::fail_close";
                case Errno::fail_open:
                    return "Errno::fail_open";
                case Errno::fail_read:
                    return "Errno::fail_read";
                case Errno::fail_write:
                    return "Errno::fail_write";
                case Errno::out_of_memory:
                    return "Errno::out_of_memory";
                case Errno::fail_os:
                    return "Errno::fail_os";
                default:
                    return fmt::format("{}:{}:{}: DEV: Errno \"{}\" undefined",
                                       __FILE__, __FUNCTION__, __LINE__);
            }
        }
    };
}


template<typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_same_v<::Noa::Flag<::Noa::Errno1>, T>, char>>
        : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const T& a, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(a.toString(), ctx);
    }
};


inline std::ostream& operator<<(std::ostream& os, const ::Noa::Flag<::Noa::Errno1>& err) {
    os << err.toString();
    return os;
}
