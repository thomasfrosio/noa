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

        fail,
        invalid_argument,
        invalid_size,
        invalid_data,
        invalid_state,
        out_of_range,
        not_supported,

        // I/O and streams
        fail_close,
        fail_open,
        fail_read,
        fail_write,

        // OS
        out_of_memory,
        fail_os,
    };

    /** Memory resource used by Pointer and PointerByte. */
    NOA_API enum class Resource : uint8_t {
        host, pinned, device
    };

    /** To which intent the file/pointer should be used. Often used as Flag<Intent>. */
    NOA_API enum class Intent : uint8_t {
        read = 0x01,
        write = 0x02,
        trunc = 0x04,
        app = 0x08,
        ate = 0x08,
        bin = 0x10
    };

    /** Supported image file formats. */
    NOA_API enum class FileFormat {
        MRC, TIFF, EER, EM, DM, RAW
    };

    /**
     * Specifies the layout (i.e. the format) of the data, allowing us to correctly reinterpret the data.
     * @details Reading/writing from/to a sequence of bytes is done as follows:
     *          `char* <-(1)-> DataType* <-(2)-> float*` where (1) is a reinterpret_cast and (2) is a
     *          static_cast. Going from left to right is a read and from right to left is a write.
     */
    NOA_API enum class DataType {
        byte, ubyte, int16, uint16, int32, uint32, float32
    };
}
