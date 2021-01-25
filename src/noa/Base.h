/**
 * @file Base.h
 * @brief Contain the minimum files to include for the core.
 * @author Thomas - ffyr2w
 * @date 20 Jul 2020
 */
#pragma once

// These headers are mostly included for the precompiled header.
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
#include <cstdint>
#include <cctype>
#include <cstring>  // std::strerror
#include <cerrno>   // errno
#include <cmath>

#include <exception>
#include <filesystem>
#include <thread>
#include <utility>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <complex>
#include <bitset>

// NOA_API and NOA_VERSION*
//#include "noa/API.h"
#include "noa/Version.h"

// Basic utilities:
#include "noa/util/Types.h"
#include "noa/util/Errno.h"
#include "noa/util/Log.h"
#include "noa/util/Exception.h"
