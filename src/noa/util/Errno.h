#pragma once

#include "noa/API.h"
#include <cstdint>
#include <string>


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
}
