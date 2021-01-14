/**
 * @file Constants.h
 * @brief Define some constants...
 * @author Thomas - ffyr2w
 * @date 11/01/2021
 */
#pragma once

#include <cstdint>

#include "noa/API.h"

namespace Noa {
    namespace fs = std::filesystem;

    /** Error numbers. Often used as Flag<Errno>. */
    NOA_API enum class Errno {
        good = 0, // this one should not change !
        fail, invalid_argument, invalid_size, invalid_data, invalid_state, out_of_range, not_supported,
        fail_close, fail_open, fail_read, fail_write, /* I/O and streams */
        out_of_memory, fail_os, /* OS */
    };

    /** Memory resource used by Pointer and PointerByte. */
    NOA_API enum class Resource : uint8_t {
        host, pinned, device
    };

    /** To which intent the pointer should be used. Mostly used as Flag<Intent>. */
    NOA_API enum class Intent : uint8_t {
        read = 0x01,
        write = 0x02,
    };

    /** Specifies the type of the data, allowing to correctly reinterpret or convert serialized data. */
    NOA_API enum class DataType {
        byte, ubyte, int16, uint16, int32, uint32, float32
    };
}
