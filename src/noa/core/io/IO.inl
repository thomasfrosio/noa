#ifndef NOA_IO_INL_
#error "This is a private header"
#endif

#include <algorithm> // std::reverse
#include <ostream>

#include "noa/core/Types.hpp"

namespace noa::io::details {
    template<i64 BYTES_IN_ELEMENTS>
    inline void reverse(Byte* element) noexcept {
        std::reverse(element, element + BYTES_IN_ELEMENTS);
    }

    template<i64 BYTES_PER_ELEMENTS>
    inline void swap_endian(Byte* ptr, i64 elements) noexcept {
        for (i64 i{0}; i < elements * BYTES_PER_ELEMENTS; i += BYTES_PER_ELEMENTS)
            reverse<BYTES_PER_ELEMENTS>(ptr + i);
    }
}

namespace noa::io {
    std::ostream& operator<<(std::ostream& os, Format format) {
        switch (format) {
            case Format::UNKNOWN:
                return os << "Format::UNKNOWN";
            case Format::MRC:
                return os << "Format::MRC";
            case Format::TIFF:
                return os << "Format::TIFF";
            case Format::EER:
                return os << "Format::EER";
            case Format::JPEG:
                return os << "Format::JPEG";
            case Format::PNG:
                return os << "Format::PNG";
        }
        return os;
    }

    std::ostream& operator<<(std::ostream& os, OpenModeStream open_mode) {
        // If any other than the first 6 bits are set, this is an invalid mode.
        if (!is_valid_open_mode(open_mode.mode)) {
            os << "OpenMode::UNKNOWN";
            return os;
        }

        struct Modes { OpenMode mode{}; const char* string{}; };
        constexpr std::array<Modes, 6> MODES{
                Modes{OpenMode::READ, "READ"},
                Modes{OpenMode::WRITE, "WRITE"},
                Modes{OpenMode::BINARY, "BINARY"},
                Modes{OpenMode::TRUNC, "TRUNC"},
                Modes{OpenMode::APP, "APP"},
                Modes{OpenMode::ATE, "ATE"}
        };

        bool add{false};
        os << "OpenMode::(";
        for (size_t i = 0; i < 6; ++i) {
            if (open_mode.mode & MODES[i].mode) {
                if (add)
                    os << '|';
                os << MODES[i].string;
                add = true;
            }
        }
        os << ')';
        return os;
    }

    constexpr bool is_valid_open_mode(open_mode_t open_mode) noexcept {
        constexpr open_mode_t MASK = 0xFFFFFFC0;
        return !(open_mode & MASK);
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

    template<typename ToSerialize>
    constexpr DataType dtype() noexcept {
        if constexpr (noa::traits::is_almost_same_v<ToSerialize, i8>) {
            return DataType::I8;
        } else if constexpr (noa::traits::is_almost_same_v<ToSerialize, u8>) {
            return DataType::U8;
        } else if constexpr (noa::traits::is_almost_same_v<ToSerialize, i16>) {
            return DataType::I16;
        } else if constexpr (noa::traits::is_almost_same_v<ToSerialize, u16>) {
            return DataType::U16;
        } else if constexpr (noa::traits::is_almost_same_v<ToSerialize, i32>) {
            return DataType::I32;
        } else if constexpr (noa::traits::is_almost_same_v<ToSerialize, u32>) {
            return DataType::U32;
        } else if constexpr (noa::traits::is_almost_same_v<ToSerialize, i64>) {
            return DataType::I64;
        } else if constexpr (noa::traits::is_almost_same_v<ToSerialize, u64>) {
            return DataType::U64;
        } else if constexpr (noa::traits::is_almost_same_v<ToSerialize, f16>) {
            return DataType::F16;
        } else if constexpr (noa::traits::is_almost_same_v<ToSerialize, f32>) {
            return DataType::F32;
        } else if constexpr (noa::traits::is_almost_same_v<ToSerialize, f64>) {
            return DataType::F64;
        } else if constexpr (noa::traits::is_almost_same_v<ToSerialize, c16>) {
            return DataType::C16;
        } else if constexpr (noa::traits::is_almost_same_v<ToSerialize, c32>) {
            return DataType::C32;
        } else if constexpr (noa::traits::is_almost_same_v<ToSerialize, c64>) {
            return DataType::C64;
        } else {
            static_assert(noa::traits::always_false_v<ToSerialize>);
        }
    }

    template<typename Numeric>
    constexpr auto type_range(DataType data_type) noexcept -> std::pair<Numeric, Numeric> {
        if constexpr (noa::traits::is_scalar_v<Numeric>) {
            switch (data_type) {
                case DataType::U4:
                    return {Numeric{0}, Numeric{15}};
                case DataType::I8:
                    return {clamp_cast<Numeric>(noa::math::Limits<i8>::min()),
                            clamp_cast<Numeric>(noa::math::Limits<i8>::max())};
                case DataType::U8:
                    return {clamp_cast<Numeric>(noa::math::Limits<u8>::min()),
                            clamp_cast<Numeric>(noa::math::Limits<u8>::max())};
                case DataType::I16:
                    return {clamp_cast<Numeric>(noa::math::Limits<i16>::min()),
                            clamp_cast<Numeric>(noa::math::Limits<i16>::max())};
                case DataType::U16:
                    return {clamp_cast<Numeric>(noa::math::Limits<u16>::min()),
                            clamp_cast<Numeric>(noa::math::Limits<u16>::max())};
                case DataType::I32:
                    return {clamp_cast<Numeric>(noa::math::Limits<i32>::min()),
                            clamp_cast<Numeric>(noa::math::Limits<i32>::max())};
                case DataType::U32:
                    return {clamp_cast<Numeric>(noa::math::Limits<u32>::min()),
                            clamp_cast<Numeric>(noa::math::Limits<u32>::max())};
                case DataType::I64:
                    return {clamp_cast<Numeric>(noa::math::Limits<i64>::min()),
                            clamp_cast<Numeric>(noa::math::Limits<i64>::max())};
                case DataType::U64:
                    return {clamp_cast<Numeric>(noa::math::Limits<u64>::min()),
                            clamp_cast<Numeric>(noa::math::Limits<u64>::max())};
                case DataType::CI16:
                    return {clamp_cast<Numeric>(noa::math::Limits<i16>::min()),
                            clamp_cast<Numeric>(noa::math::Limits<i16>::max())};
                case DataType::F16:
                case DataType::C16:
                    return {clamp_cast<Numeric>(noa::math::Limits<f16>::lowest()),
                            clamp_cast<Numeric>(noa::math::Limits<f16>::max())};
                case DataType::F32:
                case DataType::C32:
                    return {clamp_cast<Numeric>(noa::math::Limits<f32>::lowest()),
                            clamp_cast<Numeric>(noa::math::Limits<f32>::max())};
                case DataType::F64:
                case DataType::C64:
                    return {clamp_cast<Numeric>(noa::math::Limits<f64>::lowest()),
                            clamp_cast<Numeric>(noa::math::Limits<f64>::max())};
                default:
                    break;
            }
        } else if constexpr (noa::traits::is_complex_v<Numeric>) {
            using real_t = noa::traits::value_type_t<Numeric>;
            auto[min, max] = type_range<real_t>(data_type);
            return {Numeric{min, min}, Numeric{max, max}};
        } else {
            static_assert(noa::traits::always_false_v<Numeric>);
        }
        return {}; // unreachable
    }

    bool is_big_endian() noexcept {
        i16 number = 1;
        return *reinterpret_cast<unsigned char*>(&number) == 0; // char[0] == 0
    }

    std::ostream& operator<<(std::ostream& os, DataType data_type) {
        switch (data_type) {
            case DataType::UNKNOWN:
                return os << "DataType::UNKNOWN";
            case DataType::U4:
                return os << "DataType::U4";
            case DataType::I8:
                return os << "DataType::I8";
            case DataType::U8:
                return os << "DataType::U8";
            case DataType::I16:
                return os << "DataType::I16";
            case DataType::U16:
                return os << "DataType::U16";
            case DataType::I32:
                return os << "DataType::I32";
            case DataType::U32:
                return os << "DataType::U32";
            case DataType::I64:
                return os << "DataType::I64";
            case DataType::U64:
                return os << "DataType::U64";
            case DataType::F16:
                return os << "DataType::F16";
            case DataType::F32:
                return os << "DataType::F32";
            case DataType::F64:
                return os << "DataType::F64";
            case DataType::CI16:
                return os << "DataType::CI16";
            case DataType::C16:
                return os << "DataType::C16";
            case DataType::C32:
                return os << "DataType::C32";
            case DataType::C64:
                return os << "DataType::C64";
        }
        return os;
    }

    void swap_endian(Byte* ptr, i64 elements, i64 bytes_per_elements) noexcept {
        if (bytes_per_elements == 2) {
            details::swap_endian<2>(ptr, elements);
        } else if (bytes_per_elements == 4) {
            details::swap_endian<4>(ptr, elements);
        } else if (bytes_per_elements == 8) {
            details::swap_endian<8>(ptr, elements);
        }
    }

    template<typename T>
    void swap_endian(T* ptr, i64 elements) noexcept {
        swap_endian(reinterpret_cast<Byte*>(ptr), elements, sizeof(T));
    }

    i64 serialized_size(DataType data_type, i64 elements, i64 elements_per_row) noexcept {
        switch (data_type) {
            case DataType::U4: {
                if (elements_per_row == 0 || !(elements_per_row % 2)) {
                    return elements / 2;
                } else {
                    NOA_ASSERT(!(elements % elements_per_row)); // otherwise, last partial row is ignored
                    const auto rows = elements / elements_per_row;
                    const auto bytes_per_row = (elements_per_row + 1) / 2;
                    return bytes_per_row * rows;
                }
            }
            case DataType::I8:
            case DataType::U8:
                return elements;
            case DataType::I16:
            case DataType::U16:
            case DataType::F16:
                return elements * 2;
            case DataType::I32:
            case DataType::U32:
            case DataType::F32:
            case DataType::CI16:
            case DataType::C16:
                return elements * 4;
            case DataType::I64:
            case DataType::U64:
            case DataType::F64:
            case DataType::C32:
                return elements * 8;
            case DataType::C64:
                return elements * 16;
            case DataType::UNKNOWN:
                return 0;
        }
        return 0;
    }
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator (operator<<))
namespace fmt {
    template<> struct formatter<noa::io::Format> : ostream_formatter {};
    template<> struct formatter<noa::io::OpenModeStream> : ostream_formatter {};
    template<> struct formatter<noa::io::DataType> : ostream_formatter {};
}
