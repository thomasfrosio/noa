/**
 * @file Base.h
 * @brief Contain the minimum files to include for the core.
 * @author Thomas - ffyr2w
 * @date 20 Jul 2020
 */
#pragma once

// Streams:
#include <iostream>
#include <fstream>
#include <string>
#include <string_view>

// Containers:
#include <map>
#include <unordered_map>
#include <vector>
#include <array>
#include <tuple>

// Others:
#include <cctype>
#include <cstring>  // std::strerror
#include <cerrno>   // errno
#include <cmath>

#include <filesystem>
#include <thread>
#include <utility>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <complex>

// NOA_API and NOA_VERSION*
#include "noa/API.h"
#include "noa/Version.h"


namespace Noa {
    /** Some useful types */
    namespace fs = std::filesystem;
    NOA_API using iolayout_t = uint16_t;
    NOA_API using errno_t = uint8_t;

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

        // I/O
        static constexpr errno_t fail_close{10U};
        static constexpr errno_t fail_open{11U};
        static constexpr errno_t fail_read{12U};
        static constexpr errno_t fail_write{13U};

        // OS
        static constexpr errno_t out_of_memory{20U};
        static constexpr errno_t fail_os{21U};
    };
}

// NOA base:
#include "noa/util/Log.h"
#include "noa/util/Exception.h"

#if NOA_DEBUG
#define NOA_CORE_DEBUG(...) ::Noa::Log::get()->debug(__VA_ARGS__)
#else
#define NOA_CORE_DEBUG
#endif

#define NOA_LOG_TRACE(...) Noa::Log::get()->trace(__VA_ARGS__)
#define NOA_LOG_INFO(...)  Noa::Log::get()->info(__VA_ARGS__)
#define NOA_LOG_WARN(...)  Noa::Log::get()->warn(__VA_ARGS__)
#define NOA_LOG_ERROR(...) throw Noa::Error(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define NOA_LOG_ERROR_FUNC(func, ...) throw Noa::Error(__FILE__, func, __LINE__, __VA_ARGS__)
