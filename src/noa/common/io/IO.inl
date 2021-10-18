#ifndef NOA_IO_INL_
#error "This file should not be included by anything other than IO.h"
#endif

#include <algorithm> // std::reverse
#include <ostream>

#include "noa/common/traits/BaseTypes.h"
#include "noa/common/Types.h"

namespace noa::io::details {
    template<size_t BYTES_IN_ELEMENTS>
    inline void reverse(char* element) noexcept {
        std::reverse(element, element + BYTES_IN_ELEMENTS);
    }

    template<size_t BYTES_PER_ELEMENTS>
    inline void swapEndian(char* ptr, size_t elements) noexcept {
        for (size_t i{0}; i < elements * BYTES_PER_ELEMENTS; i += BYTES_PER_ELEMENTS)
            reverse<BYTES_PER_ELEMENTS>(ptr + i);
    }
}

namespace noa::io {
    std::ostream& operator<<(std::ostream& os, Format format) {
        switch (format) {
            case Format::FORMAT_UNKNOWN:
                return os << "FORMAT_UNKNOWN";
            case Format::MRC:
                return os << "MRC";
            case Format::TIFF:
                return os << "TIFF";
            case Format::EER:
                return os << "EER";
            case Format::JPEG:
                return os << "JPEG";
            case Format::PNG:
                return os << "PNG";
        }
        return os;
    }

    constexpr std::ios_base::openmode toIOSBase(open_mode_t open_mode) noexcept {
        std::ios_base::openmode mode{};
        if (open_mode & OpenMode::READ)
            mode |= std::ios::in;
        if (open_mode & OpenMode::WRITE)
            mode |= std::ios::out;
        if (open_mode & OpenMode::BINARY)
            mode |= std::ios::binary;
        if (open_mode & OpenMode::TRUNC)
            mode |= std::ios::trunc;
        if (open_mode & OpenMode::APP)
            mode |= std::ios::app;
        if (open_mode & OpenMode::ATE)
            mode |= std::ios::ate;
        return mode;
    }

    template<typename T>
    constexpr DataType getDataType() noexcept {
        if constexpr (noa::traits::is_same_v<T, int8_t> ||
                      (noa::traits::is_same_v<T, char> && noa::traits::is_sint_v<char>)) {
            return DataType::INT8;
        } else if constexpr (noa::traits::is_same_v<T, uint8_t> ||
                             (noa::traits::is_same_v<T, char> && noa::traits::is_uint_v<char>)) {
            return DataType::UINT8;
        } else if constexpr (noa::traits::is_same_v<T, int16_t>) {
            return DataType::INT16;
        } else if constexpr (noa::traits::is_same_v<T, uint16_t>) {
            return DataType::UINT16;
        } else if constexpr ((noa::traits::is_same_v<T, long> && sizeof(long) == 4) ||
                             noa::traits::is_same_v<T, int32_t>) {
            return DataType::INT32;
        } else if constexpr ((noa::traits::is_same_v<T, long> && sizeof(unsigned long) == 4) ||
                             noa::traits::is_same_v<T, uint32_t>) {
            return DataType::UINT32;
        } else if constexpr ((noa::traits::is_same_v<T, long> && sizeof(long) == 8) ||
                             noa::traits::is_same_v<T, long long>) {
            return DataType::INT64;
        } else if constexpr ((noa::traits::is_same_v<T, unsigned long> && sizeof(unsigned long) == 8) ||
                             noa::traits::is_same_v<T, unsigned long long>) {
            return DataType::UINT64;
        } else if constexpr (noa::traits::is_same_v<T, float>) {
            return DataType::FLOAT32;
        } else if constexpr (noa::traits::is_same_v<T, double>) {
            return DataType::FLOAT64;
        } else if constexpr (noa::traits::is_same_v<T, cfloat_t>) {
            return DataType::CFLOAT32;
        } else if constexpr (noa::traits::is_same_v<T, cdouble_t>) {
            return DataType::CFLOAT64;
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    template<typename T>
    constexpr void getDataTypeMinMax(DataType data_type, T* min, T* max) noexcept {
        if constexpr (noa::traits::is_scalar_v<T>) {
            switch (data_type) {
                case DataType::UINT4:
                    *min = 0;
                    *max = 15;
                    break;
                case DataType::INT8:
                    *min = clamp_cast<T>(math::Limits<int8_t>::min());
                    *max = clamp_cast<T>(math::Limits<int8_t>::max());
                    break;
                case DataType::UINT8:
                    *min = clamp_cast<T>(math::Limits<uint8_t>::min());
                    *max = clamp_cast<T>(math::Limits<uint8_t>::max());
                    break;
                case DataType::INT16:
                    *min = clamp_cast<T>(math::Limits<int16_t>::min());
                    *max = clamp_cast<T>(math::Limits<int16_t>::max());
                    break;
                case DataType::UINT16:
                    *min = clamp_cast<T>(math::Limits<uint16_t>::min());
                    *max = clamp_cast<T>(math::Limits<uint16_t>::max());
                    break;
                case DataType::INT32:
                    *min = clamp_cast<T>(math::Limits<int32_t>::min());
                    *max = clamp_cast<T>(math::Limits<int32_t>::max());
                    break;
                case DataType::UINT32:
                    *min = clamp_cast<T>(math::Limits<uint32_t>::min());
                    *max = clamp_cast<T>(math::Limits<uint32_t>::max());
                    break;
                case DataType::INT64:
                    *min = clamp_cast<T>(math::Limits<int64_t>::min());
                    *max = clamp_cast<T>(math::Limits<int64_t>::max());
                    break;
                case DataType::UINT64:
                    *min = clamp_cast<T>(math::Limits<uint64_t>::min());
                    *max = clamp_cast<T>(math::Limits<uint64_t>::max());
                    break;
                case DataType::FLOAT32:
                    *min = clamp_cast<T>(math::Limits<float>::lowest());
                    *max = clamp_cast<T>(math::Limits<float>::max());
                    break;
                case DataType::FLOAT64:
                    *min = clamp_cast<T>(math::Limits<double>::lowest());
                    *max = clamp_cast<T>(math::Limits<double>::max());
                    break;
                case DataType::CINT16:
                    *min = clamp_cast<T>(math::Limits<int16_t>::min());
                    *max = clamp_cast<T>(math::Limits<int16_t>::max());
                    break;
                case DataType::CFLOAT32:
                    *min = clamp_cast<T>(math::Limits<float>::lowest());
                    *max = clamp_cast<T>(math::Limits<float>::max());
                    break;
                case DataType::CFLOAT64:
                    *min = clamp_cast<T>(math::Limits<double>::lowest());
                    *max = clamp_cast<T>(math::Limits<double>::max());
                    break;
                default:
                    break; // TODO add FLOAT16 and CFLOAT16
            }
        } else if constexpr (noa::traits::is_complex_v<T>) {
            getDataTypeMinMax(data_type, &min->real, &max->real);
            min->imag = min->real;
            max->imag = max->real;
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    bool isBigEndian() noexcept {
        int16_t number = 1;
        return *reinterpret_cast<char*>(&number) == 0; // char[0] == 0
    }

    std::ostream& operator<<(std::ostream& os, DataType data_type) {
        switch (data_type) {
            case DataType::DATA_UNKNOWN:
                return os << "UNKNOWN";
            case DataType::UINT4:
                return os << "UINT4";
            case DataType::INT8:
                return os << "INT8";
            case DataType::UINT8:
                return os << "UINT8";
            case DataType::INT16:
                return os << "INT16";
            case DataType::UINT16:
                return os << "UINT16";
            case DataType::INT32:
                return os << "INT32";
            case DataType::UINT32:
                return os << "UINT32";
            case DataType::INT64:
                return os << "INT64";
            case DataType::UINT64:
                return os << "UINT64";
            case DataType::FLOAT16:
                return os << "FLOAT16";
            case DataType::FLOAT32:
                return os << "FLOAT32";
            case DataType::FLOAT64:
                return os << "FLOAT64";
            case DataType::CINT16:
                return os << "CINT16";
            case DataType::CFLOAT16:
                return os << "CFLOAT16";
            case DataType::CFLOAT32:
                return os << "CFLOAT32";
            case DataType::CFLOAT64:
                return os << "CFLOAT64";
        }
        return os;
    }

    void swapEndian(char* ptr, size_t elements, size_t bytes_per_elements) noexcept {
        if (bytes_per_elements == 2) {
            details::swapEndian<2>(ptr, elements);
        } else if (bytes_per_elements == 4) {
            details::swapEndian<4>(ptr, elements);
        } else if (bytes_per_elements == 8) {
            details::swapEndian<8>(ptr, elements);
        }
    }

    template<typename T>
    void swapEndian(T* ptr, size_t elements) noexcept {
        swapEndian(reinterpret_cast<char*>(ptr), elements, sizeof(T));
    }

    size_t getSerializedSize(DataType data_type, size_t elements, size_t elements_per_row) noexcept {
        switch (data_type) {
            case DataType::UINT4: {
                if (elements_per_row == 0 || !(elements_per_row % 2)) {
                    return elements / 2;
                } else {
                    NOA_ASSERT(!(elements % elements_per_row)); // otherwise, last partial row is ignored
                    size_t rows = elements / elements_per_row;
                    size_t bytes_per_row = (elements_per_row + 1) / 2;
                    return bytes_per_row * rows;
                }
            }
            case DataType::INT8:
            case DataType::UINT8:
                return elements;
            case DataType::INT16:
            case DataType::UINT16:
            case DataType::FLOAT16:
                return elements * 2;
            case DataType::INT32:
            case DataType::UINT32:
            case DataType::FLOAT32:
            case DataType::CINT16:
            case DataType::CFLOAT16:
                return elements * 4;
            case DataType::INT64:
            case DataType::UINT64:
            case DataType::FLOAT64:
            case DataType::CFLOAT32:
                return elements * 8;
            case DataType::CFLOAT64:
                return elements * 16;
            case DataType::DATA_UNKNOWN:
                return 0;
        }
        return 0;
    }
}
