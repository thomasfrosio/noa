#include "noa/util/IO.h"

using namespace Noa;

std::string IO::toString(DataType dtype) {
    if (dtype == DataType::byte)
        return "char";
    else if (dtype == DataType::ubyte)
        return "uchar";
    else if (dtype == DataType::int16)
        return "int16";
    else if (dtype == DataType::uint16)
        return "uint16";
    else if (dtype == DataType::int32)
        return "int32";
    else if (dtype == DataType::uint32)
        return "uint32";
    else if (dtype == DataType::float32)
        return "float32";
    else
        NOA_ERROR("DEV: one of the data type is not implemented");
}

Errno IO::swapEndian(char* ptr, size_t elements, size_t bytes_per_elements) {
    if (bytes_per_elements == 2)
        swapEndian<2>(ptr, elements);
    else if (bytes_per_elements == 4) {
        swapEndian<4>(ptr, elements);
    } else if (bytes_per_elements != 1)
        return Errno::invalid_argument;
    return Errno::good;
}

void IO::toFloat(const char* input, float* output, DataType dtype, size_t elements) {
    if (dtype == DataType::byte) {
        signed char tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            std::memcpy(&tmp, &input[idx], 1);
            output[idx] = static_cast<float>(tmp);
        }
    } else if (dtype == DataType::ubyte) {
        unsigned char tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            std::memcpy(&tmp, &input[idx], 1);
            output[idx] = static_cast<float>(tmp);
        }
    } else if (dtype == DataType::int16) {
        int16_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            std::memcpy(&tmp, &input[idx * 2], 2);
            output[idx] = static_cast<float>(tmp);
        }
    } else if (dtype == DataType::uint16) {
        uint16_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            std::memcpy(&tmp, &input[idx * 2], 2);
            output[idx] = static_cast<float>(tmp);
        }
    } else if (dtype == DataType::int32) {
        int32_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            std::memcpy(&tmp, &input[idx * 4], 4);
            output[idx] = static_cast<float>(tmp);
        }
    } else if (dtype == DataType::uint32) {
        uint32_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            std::memcpy(&tmp, &input[idx * 4], 4);
            output[idx] = static_cast<float>(tmp);
        }
    } else if (dtype == DataType::float32) {
        std::memcpy(output, input, elements * 4);

    } else
        NOA_ERROR("DEV: one of the data type is not implemented");
}

void IO::toDataType(const float* input, char* output, DataType dtype, size_t elements) {
    if (dtype == DataType::byte) {
        signed char tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            tmp = static_cast<signed char>(input[idx]);
            std::memcpy(&output[idx], &tmp, 1);
        }
    } else if (dtype == DataType::ubyte) {
        unsigned char tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            tmp = static_cast<unsigned char>(input[idx]);
            std::memcpy(&output[idx], &tmp, 1);
        }
    } else if (dtype == DataType::int16) {
        int16_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            tmp = static_cast<int16_t>(input[idx]);
            std::memcpy(&output[idx * 2], &tmp, 2);
        }
    } else if (dtype == DataType::uint16) {
        uint16_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            tmp = static_cast<uint16_t>(input[idx]);
            std::memcpy(&output[idx * 2], &tmp, 2);
        }
    } else if (dtype == DataType::int32) {
        int32_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            tmp = static_cast<int32_t>(input[idx]);
            std::memcpy(&output[idx * 4], &tmp, 4);
        }
    } else if (dtype == DataType::uint32) {
        uint32_t tmp;
        for (size_t idx{0}; idx < elements; ++idx) {
            tmp = static_cast<uint32_t>(input[idx]);
            std::memcpy(&output[idx * 4], &tmp, 4);
        }
    } else if (dtype == DataType::float32) {
        std::memcpy(output, input, elements * 4);

    } else
        NOA_ERROR("DEV: one of the data type is not implemented");
}
