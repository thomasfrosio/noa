/**
 * @file noa/Errno.h
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

#include <spdlog/fmt/fmt.h>
#include <ostream>
#include <type_traits>

namespace Noa {
    /** Type safe error number. */
    class Errno {
    private:
        // Keeps things type safe...
        enum class Errno_t {
            good = 0, // this one should not change !
            fail,
            fail_close, fail_os, fail_open, fail_read, fail_write, fail_seek,
            invalid_argument, invalid_size, invalid_data, invalid_state,
            out_of_range, not_supported, not_implemented, out_of_memory,
            dtype_not_supported, dtype_invalid, dtype_complex, dtype_real
        };

        using int_t = std::underlying_type_t<Errno_t>;
        int_t err{0}; // Default to Errno::good

    public:
        // Makes enum public so that it can be used Errno::good
        static constexpr Errno_t good = Errno_t::good;              // No error.
        static constexpr Errno_t fail = Errno_t::fail;              // An unknown error occurred.

        static constexpr Errno_t fail_os = Errno_t::fail_os;        // An OS related error occurred.

        static constexpr Errno_t fail_close = Errno_t::fail_close;  // The file stream could not be closed.
        static constexpr Errno_t fail_open = Errno_t::fail_open;    // The file stream could not be opened.
        static constexpr Errno_t fail_read = Errno_t::fail_read;    // A read operation on the file stream failed.
        static constexpr Errno_t fail_write = Errno_t::fail_write;  // A write operation on the file stream failed.
        static constexpr Errno_t fail_seek = Errno_t::fail_seek;    // A seek operation on the file stream failed.

        static constexpr Errno_t invalid_argument = Errno_t::invalid_argument;  // Failed due to an invalid argument.
        static constexpr Errno_t invalid_size = Errno_t::invalid_size;          // Failed due to an invalid size.
        static constexpr Errno_t invalid_data = Errno_t::invalid_data;          // Failed due to invalid data.
        static constexpr Errno_t invalid_state = Errno_t::invalid_state;        // Failed due to an invalid state.

        static constexpr Errno_t out_of_range = Errno_t::out_of_range;          // A value was out of range
        static constexpr Errno_t out_of_memory = Errno_t::out_of_memory;        // Ran out of memory.
        static constexpr Errno_t not_supported = Errno_t::not_supported;        // Option or data is not supported.
        static constexpr Errno_t not_implemented = Errno_t::not_implemented;    // Code path not implemented.

        static constexpr Errno_t dtype_not_supported = Errno_t::dtype_not_supported;    // Data type not supported.
        static constexpr Errno_t dtype_invalid = Errno_t::dtype_invalid;                // Data type not recognized.
        static constexpr Errno_t dtype_complex = Errno_t::dtype_complex;                // Real data was expected, got complex.
        static constexpr Errno_t dtype_real = Errno_t::dtype_real;                      // Complex data was expected, got real.

    public:
        /** Initialize to Errno::good. */
        inline constexpr Errno() = default;

        /** Implicitly converting Errno_t to Errno. */
        inline constexpr Errno(Errno_t v) noexcept: err(static_cast<int_t>(v)) {}

        [[nodiscard]] inline constexpr explicit operator bool() const noexcept { return err; }
        [[nodiscard]] inline constexpr bool operator==(Errno rhs) const noexcept { return err == rhs.err; }
        [[nodiscard]] inline constexpr bool operator&&(Errno rhs) const noexcept { return err && rhs.err; }
        [[nodiscard]] inline constexpr bool operator||(Errno rhs) const noexcept { return err || rhs.err; }

        /** Update if there's no underlying error number already set. */
        inline constexpr Errno update(Errno candidate) noexcept {
            if (candidate && !err)
                err = candidate.err;
            return *this;
        }

        /** Converts the underlying error number to a string. */
        [[nodiscard]] inline constexpr const char* toString() const noexcept {
            switch (static_cast<Errno_t>(err)) {
                case Errno::good:
                    return "Errno::good";
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
                case Errno::dtype_not_supported:
                    return "Errno::dtype_not_supported";
                case Errno::dtype_invalid:
                    return "Errno::dtype_invalid";
                case Errno::dtype_complex:
                    return "Errno::dtype_complex";
                case Errno::dtype_real:
                    return "Errno::dtype_real";
                default:
                    return "Errno::?";
            }
        }
    };

    [[nodiscard]] inline constexpr const char* toString(Errno err) noexcept {
        return err.toString();
    }
}

template<>
struct fmt::formatter<Noa::Errno> : fmt::formatter<std::string_view> {
    template<typename FormatCtx>
    auto format(Noa::Errno err, FormatCtx& ctx) {
        return fmt::formatter<std::string_view>::format(Noa::toString(err), ctx);
    }
};

inline std::ostream& operator<<(std::ostream& os, Noa::Errno err) {
    os << Noa::toString(err);
    return os;
}
