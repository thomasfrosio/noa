#pragma once

#include <cstdint>
#include <cstddef>
#include <climits>
#include <type_traits>

// Assume POSIX and/or Windows, both of which guarantee CHAR_BIT == 8.
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

#include "noa/common/types/Accessor.h"
#include "noa/common/types/Complex.h"
#include "noa/common/types/Constants.h"
#include "noa/common/types/Functors.h"
#include "noa/common/types/Half.h"
#include "noa/common/types/Mat.h"
#include "noa/common/types/Shape.h"
#include "noa/common/types/Vec.h"

#include "noa/common/utils/Any.h"
#include "noa/common/utils/ClampCast.h"
#include "noa/common/utils/Irange.h"
#include "noa/common/utils/Pair.h"
#include "noa/common/utils/SafeCast.h"
#include "noa/common/utils/Sort.h"

#include <ios>
#include <filesystem>

namespace noa {
    namespace fs = std::filesystem;
    using Path = fs::path;
    using Byte = std::byte;

    using u8 = uint8_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;

    using i8 = int8_t;
    using i16 = int16_t;
    using i32 = int32_t;
    using i64 = int64_t;

    using f16 = Half;
    using f32 = float;
    using f64 = double;
    static_assert(sizeof(f16) == 2);
    static_assert(sizeof(f32) == 4);
    static_assert(sizeof(f64) == 8);

    using c16 = Complex<Half>;
    using c32 = Complex<float>;
    using c64 = Complex<double>;
    static_assert(sizeof(c16) == sizeof(f16) * 2);
    static_assert(sizeof(c32) == sizeof(f32) * 2);
    static_assert(sizeof(c64) == sizeof(f64) * 2);
    static_assert(alignof(c16) == 4);
    static_assert(alignof(c32) == 8);
    static_assert(alignof(c64) == 16);

    template<typename T>
    using Shared = std::shared_ptr<T>;

    template<typename T, typename D = std::default_delete<T>>
    using Unique = std::unique_ptr<T, D>;
}
