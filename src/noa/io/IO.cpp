#include "noa/io/IO.h"

namespace noa::io {
    std::ios_base::openmode toIOSBase(uint openmode) {
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

    std::ostream& operator<<(std::ostream& os, DataType dtype) {
        switch (dtype) {
            case DataType::BYTE:
                os << "BYTE";
                break;
            case DataType::UBYTE:
                os << "UBYTE";
                break;
            case DataType::INT16:
                os << "INT16";
                break;
            case DataType::UINT16:
                os << "UINT16";
                break;
            case DataType::INT32:
                os << "INT32";
                break;
            case DataType::UINT32:
                os << "UINT32";
                break;
            case DataType::FLOAT32:
                os << "FLOAT32";
                break;
            case DataType::CINT16:
                os << "CINT16";
                break;
            case DataType::CFLOAT32:
                os << "CFLOAT32";
                break;
            default:
                NOA_THROW("DEV: missing code path");
        }
        return os;
    }

    void swapEndian(char* ptr, size_t elements, size_t bytes_per_elements) {
        if (bytes_per_elements == 2) {
            swapEndian<2>(ptr, elements);
        } else if (bytes_per_elements == 4) {
            swapEndian<4>(ptr, elements);
        } else if (bytes_per_elements == 8) {
            swapEndian<8>(ptr, elements);
        } else if (bytes_per_elements != 1)
            NOA_THROW("Bytes per elements should be 1, 2, 4 or 8, got {}", bytes_per_elements);
    }

    void toFloat(const char* input, float* output, DataType dtype, size_t elements) {
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
            NOA_THROW("Expecting a real dtype, got {}. Use io::toComplexDataType instead", dtype);
        }
    }

    void toDataType(const float* input, char* output, DataType dtype, size_t elements) {
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
            NOA_THROW("Expecting a real dtype, got {}. Use io::toComplexDataType instead", dtype);
        }
    }
}
