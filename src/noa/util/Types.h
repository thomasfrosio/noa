/**
 * @file Constants.h
 * @brief Define some constants...
 * @author Thomas - ffyr2w
 * @date 11/01/2021
 */
#pragma once

#include <cstdint>
#include <filesystem>

#include "noa/util/Complex.h"

namespace Noa {
    using cfloat = Complex<float>;
    using cdouble = Complex<double>;

    namespace fs = std::filesystem;

    /** To which intent the pointer should be used. Mostly used as Flag<Intent>. */
    enum class Intent : uint8_t {
        read = 0x01,
        write = 0x02,
        _flag_size_ = 2
    };

    /** Specifies the type of the data, allowing to correctly reinterpret or convert serialized data. */
    enum class DataType {
        byte, ubyte, int16, uint16, int32, uint32, float32
    };
}
