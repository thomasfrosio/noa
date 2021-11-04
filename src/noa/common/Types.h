/// \file noa/common/Types.h
/// \brief The basic types used by noa.
/// \author Thomas - ffyr2w
/// \date 11/01/2021

#pragma once

#include <cstdint>
#include <cstddef>
#include <climits>
#include <type_traits>

// Assume Posix and/or Windows, both of which guarantee CHAR_BIT == 8.
// The rest should fine for all modern hardware.
static_assert(CHAR_BIT == 8);
static_assert(sizeof(short) == 2);
static_assert(sizeof(int) == 4);
static_assert(sizeof(float) == 4);
static_assert(std::is_same_v<int8_t, signed char>);
static_assert(std::is_same_v<uint8_t, unsigned char>);
static_assert(std::is_same_v<int16_t, signed short>);
static_assert(std::is_same_v<uint16_t, unsigned short>);
static_assert(std::is_same_v<int32_t, signed int>);
static_assert(std::is_same_v<uint32_t, unsigned int>);

#include "noa/common/types/Index.h"
#include "noa/common/types/IntX.h"
#include "noa/common/types/FloatX.h"
#include "noa/common/types/MatX.h"
#include "noa/common/types/Complex.h"
#include "noa/common/types/Stats.h"
#include "noa/common/types/Constants.h"
#include "noa/common/types/ClampCast.h"

#include <ios>
#include <filesystem>
namespace noa {
    namespace fs = std::filesystem;
    using path_t = fs::path;
}
