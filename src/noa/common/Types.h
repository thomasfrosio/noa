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

#include "noa/common/types/Sizes.h"     // defines size2_t, size3_t and size4_t
#include "noa/common/types/IntX.h"      // defines intX_t, uintX_t, longX_t and ulongX_t
#include "noa/common/types/FloatX.h"    // defines floatX_t and doubleX_t
#include "noa/common/types/MatX.h"      // defines floatXX_t and doubleXX_t
#include "noa/common/types/Complex.h"   // defines cfloat_t and cdouble_t
#include "noa/common/types/Stats.h"     // defines Stats<T>
#include "noa/common/types/Constants.h" // defines some enums
#include "noa/common/types/ClampCast.h" // defines clamp_cast<T>()

// Fixed-size integers:
//      When the minimum required size is enough, there's no need to use fixed-size integers.
//      When the size should be fixed (e.g. serialization of data), then they become very useful.
//      The only issue, in most cases, is that (u)int64_t is aliased to either (u)long or (u)long long.
//      As such, the library will always be prepared (with one exceptions) for both (u)long AND (u)long long, that
//      includes type traits, template explicit instantiations, (de)serialization, etc. Whether users use the "real"
//      types or the fixed-size aliases, it should not matter. As mentioned above, there's one exception: (u)longX_t.
//      These static vectors are aliases of IntX<(u)int64_t>, ensuring to use the "default" 8-bytes integer elements,
//      regardless of the platform.
//      Note that for simplicity and safety, when 8-bytes integers are required, we do recommend using the fixed-
//      size integers, i.e. int64_t or uint64_t, as opposed to (u)long or (u)long long.
//
// Complex:
//      Since the CUDA backend might be included in the build, and for simplicity, the library never uses std::complex.
//      Instead, it includes a complex type that can be used interchangeably across all backends and can be
//      reinterpreted to std::complex or cuComplex/cuDoubleComplex. It is a simple struct of two floats or two doubles.

#include <ios>
#include <filesystem>
namespace noa {
    namespace fs = std::filesystem;
    using path_t = fs::path;
}
