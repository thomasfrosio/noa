#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include <algorithm> // std::reverse
#include <filesystem>
#include <ios>
#include <ostream>
#include "noa/core/utils/Irange.hpp"
#include "noa/core/types/Half.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Shape.hpp"

namespace noa {
    namespace fs = std::filesystem;
    using Path = fs::path;
}

namespace noa::io::guts {
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

/// Enumerators and bit masks related to IO.
namespace noa::io {
    enum class Format {
        UNKNOWN,
        MRC,
        TIFF,
        EER,
        JPEG,
        PNG
    };

    struct OpenMode {
        bool read{};
        bool write{};
        bool truncate{};
        bool binary{};
        bool append{};
        bool at_the_end{};
    };

    /// Data type used for (de)serialization.
    enum class DataType {
        UNKNOWN,
        I8,
        U8,
        I16,
        U16,
        I32,
        U32,
        I64,
        U64,
        F16,
        F32,
        F64,
        C16,
        C32,
        C64,
        U4, // not "real" type
        CI16 // not "real" type
    };

    /// Returns the range that \T values, about to be converted to \p data_type, should be in.
    /// \details (De)Serialization functions can clamp the values to fit the destination types. However, if
    ///          one wants to clamp the values beforehand, this function becomes really useful. It computes
    ///          the lowest and maximum value that the \p data_type can hold and clamps them to type \p T.
    /// \tparam T           A restricted numeric.
    ///                     If complex, real and imaginary parts are set with the same value.
    /// \param DataType     Data type. If DataType::UNKNOWN, return 0.
    /// \return             Minimum and maximum \p T values in the range of \p data_type.
    template<typename Numeric>
    constexpr auto type_range(DataType data_type) noexcept -> std::pair<Numeric, Numeric> {
        if constexpr (nt::is_scalar_v<Numeric>) {
            switch (data_type) {
                case DataType::U4:
                    return {Numeric{0}, Numeric{15}};
                case DataType::I8:
                    return {clamp_cast<Numeric>(std::numeric_limits<i8>::min()),
                            clamp_cast<Numeric>(std::numeric_limits<i8>::max())};
                case DataType::U8:
                    return {clamp_cast<Numeric>(std::numeric_limits<u8>::min()),
                            clamp_cast<Numeric>(std::numeric_limits<u8>::max())};
                case DataType::I16:
                    return {clamp_cast<Numeric>(std::numeric_limits<i16>::min()),
                            clamp_cast<Numeric>(std::numeric_limits<i16>::max())};
                case DataType::U16:
                    return {clamp_cast<Numeric>(std::numeric_limits<u16>::min()),
                            clamp_cast<Numeric>(std::numeric_limits<u16>::max())};
                case DataType::I32:
                    return {clamp_cast<Numeric>(std::numeric_limits<i32>::min()),
                            clamp_cast<Numeric>(std::numeric_limits<i32>::max())};
                case DataType::U32:
                    return {clamp_cast<Numeric>(std::numeric_limits<u32>::min()),
                            clamp_cast<Numeric>(std::numeric_limits<u32>::max())};
                case DataType::I64:
                    return {clamp_cast<Numeric>(std::numeric_limits<i64>::min()),
                            clamp_cast<Numeric>(std::numeric_limits<i64>::max())};
                case DataType::U64:
                    return {clamp_cast<Numeric>(std::numeric_limits<u64>::min()),
                            clamp_cast<Numeric>(std::numeric_limits<u64>::max())};
                case DataType::CI16:
                    return {clamp_cast<Numeric>(std::numeric_limits<i16>::min()),
                            clamp_cast<Numeric>(std::numeric_limits<i16>::max())};
                case DataType::F16:
                case DataType::C16:
                    return {clamp_cast<Numeric>(std::numeric_limits<f16>::lowest()),
                            clamp_cast<Numeric>(std::numeric_limits<f16>::max())};
                case DataType::F32:
                case DataType::C32:
                    return {clamp_cast<Numeric>(std::numeric_limits<f32>::lowest()),
                            clamp_cast<Numeric>(std::numeric_limits<f32>::max())};
                case DataType::F64:
                case DataType::C64:
                    return {clamp_cast<Numeric>(std::numeric_limits<f64>::lowest()),
                            clamp_cast<Numeric>(std::numeric_limits<f64>::max())};
                default:
                    break;
            }
        } else if constexpr (nt::is_complex_v<Numeric>) {
            using real_t = nt::value_type_t<Numeric>;
            auto[min, max] = type_range<real_t>(data_type);
            return {Numeric{min, min}, Numeric{max, max}};
        } else {
            static_assert(nt::always_false_v<Numeric>);
        }
        return {}; // unreachable
    }

    /// Whether this code was compiled for big-endian.
    inline bool is_big_endian() noexcept {
        i16 number = 1;
        return *reinterpret_cast<unsigned char*>(&number) == 0; // char[0] == 0
    }

    /// Changes the endianness of the elements in an array, in-place.
    /// \param[in] ptr              Array of bytes to swap. Should contain at least (elements * bytes_per_element).
    /// \param elements             How many elements to swap.
    /// \param bytes_per_element    Size, in bytes, of one element. If not 2, 4, or 8, do nothing.
    inline void swap_endian(Byte* ptr, i64 elements, i64 bytes_per_elements) noexcept {
        if (bytes_per_elements == 2) {
            guts::swap_endian<2>(ptr, elements);
        } else if (bytes_per_elements == 4) {
            guts::swap_endian<4>(ptr, elements);
        } else if (bytes_per_elements == 8) {
            guts::swap_endian<8>(ptr, elements);
        }
    }

    /// Changes the endianness of the elements in an array, in-place.
    /// \param[in] ptr  Array of bytes to swap.
    /// \param elements How many elements to swap.
    template<typename T>
    inline void swap_endian(T* ptr, i64 elements) noexcept {
        swap_endian(reinterpret_cast<Byte*>(ptr), elements, sizeof(T));
    }

    /// Returns the number of bytes necessary to hold a number of \p elements formatted with a given \p data_type.
    /// \param data_type        Data type used for serialization. If DataType::UNKNOWN, returns 0.
    /// \param elements         Number of elements.
    /// \param elements_per_row Number of \p T elements per row.
    ///                         Only used when \p data_type is U4.
    ///                         This is to account for the half-byte padding at the end of odd rows with U4.
    ///                         If 0, the number of elements per row is assumed to be even.
    ///                         Otherwise, \p elements should be a multiple of \p elements_per_row.
    inline i64 serialized_size(DataType data_type, i64 elements, i64 elements_per_row = 0) noexcept {
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

namespace noa::io {
    /// Converts the values in \p input, according to the desired \p data_type,
    /// and saves the converted values into \p output. Values are saved in the BDHW order.
    /// \tparam Input       Any restricted numeric. If \p Input is complex, \p input is reinterpreted to
    ///                     the corresponding real type array, requiring its innermost dimension to
    ///                     be contiguous.
    /// \param[in] input    Values to serialize.
    /// \param strides      BDHW strides of \p input.
    /// \param shape        BDHW shape of \p input.
    /// \param[out] output  Array containing the serialized values.
    ///                     See serialized_size() to know how many bytes will be written into this array.
    /// \param data_type    Desired data type of the serialized values.
    ///                     If it describes a complex value, \p Input should be complex.
    ///                     If it describes a scalar, \p Input should be a scalar.
    ///                     If it is U4, \p input should be C-contiguous.
    /// \param clamp        Whether the input values should be clamped to the range of the desired data type.
    ///                     If false, out-of-range values are undefined.
    /// \param swap_endian  Whether the endianness of the serialized data should be swapped.
    template<typename Input, typename = std::enable_if_t<nt::is_numeric_v<Input>>>
    void serialize(const Input* input, const Strides4<i64>& strides, const Shape4<i64>& shape,
                   Byte* output, DataType data_type,
                   bool clamp = false, bool swap_endian = false);

    /// Overload taking the input type as a data type.
    /// \note \p strides and \p shape are in number of \p input_data_type elements.
    void serialize(const void* input, DataType input_data_type,
                   const Strides4<i64>& strides, const Shape4<i64>& shape,
                   Byte* output, DataType output_data_type,
                   bool clamp = false, bool swap_endian = false);

    /// Overload taking the output as an ostream.
    /// \param[out] output  Output stream to write into. The current position is used as starting point.
    ///                     See serialized_size to know how many bytes will be written into this stream.
    /// \throws Exception   If the stream fails to write data, an exception is thrown. Note that the stream is
    ///                     reset to its good state before the exception is thrown.
    template<typename Input>
    void serialize(const Input* input, const Strides4<i64>& strides, const Shape4<i64>& shape,
                   std::ostream& output, DataType data_type,
                   bool clamp = false, bool swap_endian = false);

    /// Overload taking the input type as a data type and the output as an ostream.
    /// \note \p strides and \p shape are in number of \p input_data_type elements.
    void serialize(const void* input, DataType input_data_type,
                   const Strides4<i64>& strides, const Shape4<i64>& shape,
                   std::ostream& output, DataType data_type,
                   bool clamp = false, bool swap_endian = false);
}

namespace noa::io {
    /// Deserializes the \p input array according to its \p data_type, converts the values to \p T,
    /// and save them in the \p output array. Values are saved in the rightmost order.
    /// \tparam T           Any data type. If \p T is complex, \p output is reinterpreted to the corresponding
    ///                     real type array, requiring its innermost dimension to be contiguous.
    /// \param[in] input    Array containing the values to deserialize.
    ///                     See serialized_size to know how many bytes will be read from this array.
    /// \param data_type    Data type of the serialized values.
    ///                     If it describes a complex value, \p T should be complex.
    ///                     If it describes a scalar, \p T should be a scalar.
    ///                     If it is U4, \p output should be C-contiguous.
    /// \param[out] output  Values to serialize.
    /// \param strides      BDHW strides of \p output.
    /// \param shape        BDHW shape of \p output.
    /// \param clamp        Whether the deserialized values should be clamped to the \p T range.
    ///                     If false, out-of-range values are undefined.
    /// \param swap_endian  Whether the endianness of the serialized data should be swapped.
    template<typename T>
    void deserialize(const Byte* input, DataType data_type,
                     T* output, const Strides4<i64>& strides, const Shape4<i64>& shape,
                     bool clamp = false, bool swap_endian = false);

    /// Overload taking the output type as a data type.
    /// \note \p strides and \p shape are in number of \p output_data_type elements.
    void deserialize(const Byte* input, DataType input_data_type,
                     void* output, DataType output_data_type,
                     const Strides4<i64>& strides, const Shape4<i64>& shape,
                     bool clamp = false, bool swap_endian = false);

    /// Overload taking the input as an istream.
    /// \param[in] input    Input stream to read from. The current position is used as starting point.
    ///                     See serialized_size to know how many bytes will be read from this stream.
    /// \throws Exception   If the stream fails to read data, an exception is thrown. Note that the stream is
    ///                     reset to its good state before the exception is thrown.
    template<typename T>
    void deserialize(std::istream& input, DataType data_type,
                     T* output, const Strides4<i64>& strides, const Shape4<i64>& shape,
                     bool clamp = false, bool swap_endian = false);

    /// Overload taking the input as an istream and the output type as a data type.
    /// \note \p strides and \p shape are in number of \p output_data_type elements.
    void deserialize(std::istream& input, DataType input_data_type,
                     void* output, DataType output_data_type,
                     const Strides4<i64>& strides, const Shape4<i64>& shape,
                     bool clamp = false, bool swap_endian = false);
}

namespace noa::io {
    inline std::ostream& operator<<(std::ostream& os, Format format) {
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

    inline std::ostream& operator<<(std::ostream& os, OpenMode mode) {
        std::array flags{mode.read, mode.write, mode.truncate, mode.binary, mode.append, mode.at_the_end};
        std::array names{"read", "write", "truncate", "binary", "append", "at_the_end"};

        bool add{false};
        os << "OpenMode{";
        for (auto i: noa::irange<size_t>(6)) {
            if (flags[i]) {
                if (add)
                    os << '-';
                os << names[i];
                add = true;
            }
        }
        os << '}';
        return os;
    }

    /// Switches from an OpenMode to a \c std::ios_base::openmode flag.
    inline constexpr std::ios_base::openmode to_ios_base(OpenMode open_mode) noexcept {
        std::ios_base::openmode mode{};
        if (open_mode.read)
            mode |= std::ios::in;
        if (open_mode.write)
            mode |= std::ios::out;
        if (open_mode.binary)
            mode |= std::ios::binary;
        if (open_mode.truncate)
            mode |= std::ios::trunc;
        if (open_mode.append)
            mode |= std::ios::app;
        if (open_mode.at_the_end)
            mode |= std::ios::ate;
        return mode;
    }

    inline std::ostream& operator<<(std::ostream& os, DataType data_type) {
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

    /// Returns the DataType corresponding to the type \p T.
    /// \tparam T Any data type.
    template<typename T>
    constexpr DataType dtype() noexcept {
        if constexpr (nt::is_almost_same_v<T, i8>) {
            return DataType::I8;
        } else if constexpr (nt::is_almost_same_v<T, u8>) {
            return DataType::U8;
        } else if constexpr (nt::is_almost_same_v<T, i16>) {
            return DataType::I16;
        } else if constexpr (nt::is_almost_same_v<T, u16>) {
            return DataType::U16;
        } else if constexpr (nt::is_almost_same_v<T, i32>) {
            return DataType::I32;
        } else if constexpr (nt::is_almost_same_v<T, u32>) {
            return DataType::U32;
        } else if constexpr (nt::is_almost_same_v<T, i64>) {
            return DataType::I64;
        } else if constexpr (nt::is_almost_same_v<T, u64>) {
            return DataType::U64;
        } else if constexpr (nt::is_almost_same_v<T, f16>) {
            return DataType::F16;
        } else if constexpr (nt::is_almost_same_v<T, f32>) {
            return DataType::F32;
        } else if constexpr (nt::is_almost_same_v<T, f64>) {
            return DataType::F64;
        } else if constexpr (nt::is_almost_same_v<T, c16>) {
            return DataType::C16;
        } else if constexpr (nt::is_almost_same_v<T, c32>) {
            return DataType::C32;
        } else if constexpr (nt::is_almost_same_v<T, c64>) {
            return DataType::C64;
        } else {
            static_assert(nt::always_false_v<T>);
        }
    }
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator)
namespace fmt {
    template<> struct formatter<noa::io::Format> : ostream_formatter {};
    template<> struct formatter<noa::io::OpenMode> : ostream_formatter {};
    template<> struct formatter<noa::io::DataType> : ostream_formatter {};
}
#endif
