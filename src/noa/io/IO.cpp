#include "noa/io/IO.h"

using namespace Noa;

std::ios_base::openmode IO::toIOSBase(uint openmode) {
    std::ios_base::openmode mode{};
    if (openmode & OpenMode::READ)
        mode |= std::ios::in;
    if (openmode & OpenMode::WRITE)
        mode |= std::ios::out;
    if (openmode & OpenMode::BINARY)
        mode |= std::ios::binary;
    if (openmode & OpenMode::TRUNC)
        mode |= std::ios::trunc;
    if (openmode & OpenMode::APP)
        mode |= std::ios::app;
    if (openmode & OpenMode::ATE)
        mode |= std::ios::ate;
    return mode;
}

const char* IO::toString(DataType dtype) {
    switch (dtype) {
        case DataType::BYTE:
            return "BYTE";
        case DataType::UBYTE:
            return "UBYTE";
        case DataType::INT16:
            return "INT16";
        case DataType::UINT16:
            return "UINT16";
        case DataType::INT32:
            return "INT32";
        case DataType::UINT32:
            return "UINT32";
        case DataType::FLOAT32:
            return "FLOAT32";
        case DataType::CINT16:
            return "CINT16";
        case DataType::CFLOAT32:
            return "CFLOAT32";
        default:
            NOA_THROW("DEV: missing code path");
    }
}

void IO::swapEndian(char* ptr, size_t elements, size_t bytes_per_elements) {
    if (bytes_per_elements == 2) {
        IO::swapEndian<2>(ptr, elements);
    } else if (bytes_per_elements == 4) {
        IO::swapEndian<4>(ptr, elements);
    } else if (bytes_per_elements == 8) {
        IO::swapEndian<8>(ptr, elements);
    } else if (bytes_per_elements != 1)
        NOA_THROW("bytes per elements should be 1, 2, 4 or 8, got {}", bytes_per_elements);
}

void IO::toFloat(const char* input, float* output, DataType dtype, size_t elements) {
    if (dtype == DataType::BYTE) {
        signed char tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            std::memcpy(&tmp, &input[idx], 1);
            output[idx] = static_cast<float>(tmp);
        }
    } else if (dtype == DataType::UBYTE) {
        unsigned char tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            std::memcpy(&tmp, &input[idx], 1);
            output[idx] = static_cast<float>(tmp);
        }
    } else if (dtype == DataType::INT16) {
        int16_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            std::memcpy(&tmp, &input[idx * 2], 2);
            output[idx] = static_cast<float>(tmp);
        }
    } else if (dtype == DataType::UINT16) {
        uint16_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            std::memcpy(&tmp, &input[idx * 2], 2);
            output[idx] = static_cast<float>(tmp);
        }
    } else if (dtype == DataType::INT32) {
        int32_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            std::memcpy(&tmp, &input[idx * 4], 4);
            output[idx] = static_cast<float>(tmp);
        }
    } else if (dtype == DataType::UINT32) {
        uint32_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            std::memcpy(&tmp, &input[idx * 4], 4);
            output[idx] = static_cast<float>(tmp);
        }
    } else if (dtype == DataType::FLOAT32) {
        std::memcpy(output, input, elements * 4);
    } else {
        NOA_THROW("Expecting a real dtype, got {}. Use IO::toComplexDataType instead", toString(dtype));
    }
}

void IO::toDataType(const float* input, char* output, DataType dtype, size_t elements) {
    if (dtype == DataType::BYTE) {
        signed char tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            tmp = static_cast<signed char>(input[idx]);
            std::memcpy(&output[idx], &tmp, 1);
        }
    } else if (dtype == DataType::UBYTE) {
        unsigned char tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            tmp = static_cast<unsigned char>(input[idx]);
            std::memcpy(&output[idx], &tmp, 1);
        }
    } else if (dtype == DataType::INT16) {
        int16_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            tmp = static_cast<int16_t>(input[idx]);
            std::memcpy(&output[idx * 2], &tmp, 2);
        }
    } else if (dtype == DataType::UINT16) {
        uint16_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            tmp = static_cast<uint16_t>(input[idx]);
            std::memcpy(&output[idx * 2], &tmp, 2);
        }
    } else if (dtype == DataType::INT32) {
        int32_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            tmp = static_cast<int32_t>(input[idx]);
            std::memcpy(&output[idx * 4], &tmp, 4);
        }
    } else if (dtype == DataType::UINT32) {
        uint32_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            tmp = static_cast<uint32_t>(input[idx]);
            std::memcpy(&output[idx * 4], &tmp, 4);
        }
    } else if (dtype == DataType::FLOAT32) {
        std::memcpy(output, input, elements * 4);
    } else {
        NOA_THROW("Expecting a real dtype, got {}. Use IO::toComplexDataType instead", toString(dtype));
    }
}
