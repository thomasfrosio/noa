#pragma once

#include "noa/API.h"


namespace Noa {
    /**
     * Error numbers used throughout the @a Noa namespace.
     * Errno should evaluate to @c false if no errors (@c Errno::good), and @c true for errors.
     * Code should first check whether or not there's an error before looking for specific errors.
     */
    struct NOA_API Errno {
        static constexpr errno_t good{0U}; // this one should not change !
        static constexpr errno_t fail{1U};

        static constexpr errno_t invalid_argument{2U};
        static constexpr errno_t invalid_size{3U};
        static constexpr errno_t invalid_data{4U};
        static constexpr errno_t out_of_range{5U};
        static constexpr errno_t not_supported{6U};

        // Streams
        static constexpr errno_t fail_close{10U};
        static constexpr errno_t fail_open{11U};
        static constexpr errno_t fail_read{12U};
        static constexpr errno_t fail_write{13U};

        // OS
        static constexpr errno_t out_of_memory{20U};
        static constexpr errno_t fail_os{21U};


        /** Returns a string describing the meaning of the error number. */
        static std::string toString(errno_t err) noexcept {
            if (err == Errno::fail)
                return "generic failure";
            else if (err == Errno::invalid_argument)
                return "invalid argument";
            else if (err == Errno::invalid_size)
                return "invalid size";
            else if (err == Errno::invalid_data)
                return "invalid data";
            else if (err == Errno::out_of_range)
                return "out of range";
            else if (err == Errno::not_supported)
                return "not supported";

            else if (err == Errno::fail_close)
                return "fail to close the stream";
            else if (err == Errno::fail_open)
                return "fail to open the stream";
            else if (err == Errno::fail_read)
                return "fail to read from the stream";
            else if (err == Errno::fail_write)
                return "fail to write to the stream";

            else if (err == Errno::out_of_memory)
                return "out of memory";
            else if (err == Errno::fail_os)
                return "generic OS failure";

            else
                return "unknown error number";
        }
    };
}
