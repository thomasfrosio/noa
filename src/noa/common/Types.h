/// \file noa/common/Types.h
/// \brief The basic types.
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

#include "noa/common/types/Bool2.h"
#include "noa/common/types/Bool3.h"
#include "noa/common/types/Bool4.h"
#include "noa/common/types/Float2.h"
#include "noa/common/types/Float3.h"
#include "noa/common/types/Float4.h"
#include "noa/common/types/Int2.h"
#include "noa/common/types/Int3.h"
#include "noa/common/types/Int4.h"
#include "noa/common/types/Mat22.h"
#include "noa/common/types/Mat23.h"
#include "noa/common/types/Mat33.h"
#include "noa/common/types/Mat34.h"
#include "noa/common/types/Mat44.h"

#include "noa/common/types/ClampCast.h"
#include "noa/common/types/Complex.h"
#include "noa/common/types/Constants.h"
#include "noa/common/types/Half.h"
#include "noa/common/types/View.h"

#include <ios>
#include <filesystem>

namespace noa {
    namespace fs = std::filesystem;
    using path_t = fs::path;

    template<typename T>
    using shared_t = std::shared_ptr<T>;

    template<typename T, typename D = std::default_delete<T>>
    using unique_t = std::unique_ptr<T, D>;
}
