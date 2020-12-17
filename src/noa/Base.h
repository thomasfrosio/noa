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
}

// NOA base:
#include "noa/util/Errno.h"
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
