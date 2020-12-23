#include "noa/util/IO.h"

using namespace Noa;



std::string IO::toString(DataType dtype) noexcept {
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
        return "unknown data type";
}


errno_t IO::swapEndian(char* ptr, size_t elements, size_t bytes_per_elements) {
    if (bytes_per_elements == 2)
        for (size_t i{0}; i < elements * bytes_per_elements; i += bytes_per_elements)
            reverse<2>(ptr + i);
    else if (bytes_per_elements == 4) {
        for (size_t i{0}; i < elements * bytes_per_elements; i += bytes_per_elements)
            reverse<4>(ptr + i);
    } else if (bytes_per_elements != 1)
        return Errno::invalid_argument;
    return Errno::good;
}


void IO::toFloat(const char* input, float* output, DataType dtype, size_t elements) {
    if (dtype == DataType::byte) {
        auto tmp = reinterpret_cast<const signed char*>(input);
        for (size_t idx{0}; idx < elements; ++idx)
            output[idx] = static_cast<float>(tmp[idx]);  // or *output++ = static_cast<float>(*tmp++);

    } else if (dtype == DataType::ubyte) {
        auto tmp = reinterpret_cast<const unsigned char*>(input);
        for (size_t idx{0}; idx < elements; ++idx)
            output[idx] = static_cast<float>(tmp[idx]);

    } else if (dtype == DataType::int16) {
        auto tmp = reinterpret_cast<const int16_t*>(input);
        for (size_t idx{0}; idx < elements; ++idx)
            output[idx] = static_cast<float>(tmp[idx]);

    } else if (dtype == DataType::uint16) {
        auto tmp = reinterpret_cast<const uint16_t*>(input);
        for (size_t idx{0}; idx < elements; ++idx)
            output[idx] = static_cast<float>(tmp[idx]);

    } else if (dtype == DataType::int32) {
        auto tmp = reinterpret_cast<const int32_t*>(input);
        for (size_t idx{0}; idx < elements; ++idx)
            output[idx] = static_cast<float>(tmp[idx]);

    } else if (dtype == DataType::uint32) {
        auto tmp = reinterpret_cast<const uint32_t*>(input);
        for (size_t idx{0}; idx < elements; ++idx)
            output[idx] = static_cast<float>(tmp[idx]);

    } else if (dtype == DataType::float32) {
        std::memcpy(output, input, elements * bytesPerElement(DataType::float32));

    } else
        NOA_LOG_ERROR("DEV: one of the data type is not implemented");
}


void IO::toDataType(const float* input, char* output, DataType dtype, size_t elements) {
    if (dtype == DataType::byte) {
        auto tmp = reinterpret_cast<signed char*>(output);
        for (size_t idx{0}; idx < elements; ++idx)
            tmp[idx] = static_cast<signed char>(input[idx]);

    } else if (dtype == DataType::ubyte) {
        auto tmp = reinterpret_cast<unsigned char*>(output);
        for (size_t idx{0}; idx < elements; ++idx)
            tmp[idx] = static_cast<unsigned char>(input[idx]);

    } else if (dtype == DataType::int16) {
        auto tmp = reinterpret_cast<int16_t*>(output);
        for (size_t idx{0}; idx < elements; ++idx)
            tmp[idx] = static_cast<int16_t>(input[idx]);

    } else if (dtype == DataType::uint16) {
        auto tmp = reinterpret_cast<uint16_t*>(output);
        for (size_t idx{0}; idx < elements; ++idx)
            tmp[idx] = static_cast<uint16_t>(input[idx]);

    } else if (dtype == DataType::int32) {
        auto tmp = reinterpret_cast<int32_t*>(output);
        for (size_t idx{0}; idx < elements; ++idx)
            tmp[idx] = static_cast<int32_t>(input[idx]);

    } else if (dtype == DataType::uint32) {
        auto tmp = reinterpret_cast<uint32_t*>(output);
        for (size_t idx{0}; idx < elements; ++idx)
            tmp[idx] = static_cast<uint32_t>(input[idx]);

    } else if (dtype == DataType::float32) {
        std::memcpy(output, input, elements * bytesPerElement(DataType::float32));

    } else
        NOA_LOG_ERROR("DEV: one of the data type is not implemented");
}
