#pragma once

#include <algorithm> // std::reverse
#include <filesystem>
#include <ios>
#include <ostream>

#include "noa/core/utils/Irange.hpp"
#include "noa/core/types/Half.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Span.hpp"

namespace noa {
    namespace fs = std::filesystem;
    inline namespace types {
        using Path = fs::path;
    }
}

namespace noa::io::guts {
    template<i64 BYTES_IN_ELEMENTS>
    void reverse(std::byte* element) noexcept {
        std::reverse(element, element + BYTES_IN_ELEMENTS);
    }

    template<i64 BYTES_PER_ELEMENTS>
    void swap_endian(std::byte* ptr, i64 elements) noexcept {
        for (i64 i{}; i < elements * BYTES_PER_ELEMENTS; i += BYTES_PER_ELEMENTS)
            reverse<BYTES_PER_ELEMENTS>(ptr + i);
    }
}

namespace noa::io {
    /// Controls how files should be opened.
    struct Open {
        bool read{};
        bool write{};
        bool truncate{};
        bool binary{};
        bool append{};
        bool at_the_end{};

        /// Converts to the std::ios_base::openmode flag.
        [[nodiscard]] constexpr auto to_ios_base() const noexcept {
            std::ios_base::openmode mode{};
            if (read)
                mode |= std::ios::in;
            if (write)
                mode |= std::ios::out;
            if (binary)
                mode |= std::ios::binary;
            if (truncate)
                mode |= std::ios::trunc;
            if (append)
                mode |= std::ios::app;
            if (at_the_end)
                mode |= std::ios::ate;
            return mode;
        }
    };

    struct Encoding {
    public:
        enum class Format {
            UNKNOWN = 0,
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
            U4,
            CI16
        };
        using enum Format;

    public:
        /// Format used for the encoding.
        Format format{};

        /// Whether the values should be clamped to the value range of the destination type.
        /// If false, out-of-range values are undefined.
        bool clamp{};

        /// Whether the endianness of the serialized data should be swapped.
        bool endian_swap{};

    public:
        /// Returns the range that \T values, about to be serialized with the current format, should be in.
        /// \details (De)Serialization functions can clamp the values to fit the destination types. However, if
        ///          one wants to clamp the values beforehand, this function becomes really useful. It computes
        ///          the lowest and maximum value that the format can hold and clamps them to type \p T.
        /// \return Minimum and maximum \p T values in the range of the current format.
        template<typename T>
        [[nodiscard]] constexpr auto value_range() const -> Pair<T, T> {
            if constexpr (nt::scalar<T>) {
                switch (format) {
                    case U4:
                        return {T{0}, T{15}};
                    case I8:
                        return {clamp_cast<T>(std::numeric_limits<i8>::min()),
                                clamp_cast<T>(std::numeric_limits<i8>::max())};
                    case U8:
                        return {clamp_cast<T>(std::numeric_limits<u8>::min()),
                                clamp_cast<T>(std::numeric_limits<u8>::max())};
                    case I16:
                        return {clamp_cast<T>(std::numeric_limits<i16>::min()),
                                clamp_cast<T>(std::numeric_limits<i16>::max())};
                    case U16:
                        return {clamp_cast<T>(std::numeric_limits<u16>::min()),
                                clamp_cast<T>(std::numeric_limits<u16>::max())};
                    case I32:
                        return {clamp_cast<T>(std::numeric_limits<i32>::min()),
                                clamp_cast<T>(std::numeric_limits<i32>::max())};
                    case U32:
                        return {clamp_cast<T>(std::numeric_limits<u32>::min()),
                                clamp_cast<T>(std::numeric_limits<u32>::max())};
                    case I64:
                        return {clamp_cast<T>(std::numeric_limits<i64>::min()),
                                clamp_cast<T>(std::numeric_limits<i64>::max())};
                    case U64:
                        return {clamp_cast<T>(std::numeric_limits<u64>::min()),
                                clamp_cast<T>(std::numeric_limits<u64>::max())};
                    case CI16:
                        return {clamp_cast<T>(std::numeric_limits<i16>::min()),
                                clamp_cast<T>(std::numeric_limits<i16>::max())};
                    case F16:
                    case C16:
                        return {clamp_cast<T>(std::numeric_limits<f16>::lowest()),
                                clamp_cast<T>(std::numeric_limits<f16>::max())};
                    case F32:
                    case C32:
                        return {clamp_cast<T>(std::numeric_limits<f32>::lowest()),
                                clamp_cast<T>(std::numeric_limits<f32>::max())};
                    case F64:
                    case C64:
                        return {clamp_cast<T>(std::numeric_limits<f64>::lowest()),
                                clamp_cast<T>(std::numeric_limits<f64>::max())};
                    case UNKNOWN:
                        panic("Encoding format is not set");
                }
                return {}; // unreachable
            } else if constexpr (nt::complex<T>) {
                using real_t = nt::value_type_t<T>;
                auto[min, max] = value_range<real_t>();
                return {T{min, min}, T{max, max}};
            } else {
                static_assert(nt::always_false<T>);
            }
        }

        [[nodiscard]] static constexpr auto encoded_size(
            Format format,
            i64 n_elements,
            i64 n_elements_per_row = 0
        ) noexcept -> i64 {
            switch (format) {
                case U4: {
                    if (n_elements_per_row == 0 or is_even(n_elements_per_row))
                        return n_elements / 2;

                    check(is_multiple_of(n_elements, n_elements_per_row),
                          "The number of elements is not compatible with the size of the rows");
                    const auto rows = n_elements / n_elements_per_row;
                    const auto bytes_per_row = (n_elements_per_row + 1) / 2;
                    return bytes_per_row * rows;
                }
                case I8:
                case U8:
                    return n_elements;
                case I16:
                case U16:
                case F16:
                    return n_elements * 2;
                case I32:
                case U32:
                case F32:
                case CI16:
                case C16:
                    return n_elements * 4;
                case I64:
                case U64:
                case F64:
                case C32:
                    return n_elements * 8;
                case C64:
                    return n_elements * 16;
                case UNKNOWN:
                    return 0;
            }
            return 0; // unreachable
        }

        /// Returns the number of bytes necessary to hold \p n_elements with the current format.
        /// \param n_elements           Number of elements.
        /// \param n_elements_per_row   Number of elements per row.
        ///                             Only used with the U4 format.
        ///                             This is to account for the half-byte padding at the end of odd rows with U4.
        ///                             If 0, the number of elements per row is assumed to be even.
        ///                             Otherwise, \p n_elements should be a multiple of \p n_elements_per_row.
        [[nodiscard]] constexpr auto encoded_size(
            i64 n_elements,
            i64 n_elements_per_row = 0
        ) const noexcept -> i64 {
            return encoded_size(format, n_elements, n_elements_per_row);
        }

        /// Returns the encoding format corresponding to the type \p T.
        template<typename T>
        static constexpr auto to_format() noexcept -> Format {
            if constexpr (nt::almost_same_as<T, i8>) {
                return I8;
            } else if constexpr (nt::almost_same_as<T, u8>) {
                return U8;
            } else if constexpr (nt::almost_same_as<T, i16>) {
                return I16;
            } else if constexpr (nt::almost_same_as<T, u16>) {
                return U16;
            } else if constexpr (nt::almost_same_as<T, i32>) {
                return I32;
            } else if constexpr (nt::almost_same_as<T, u32>) {
                return U32;
            } else if constexpr (nt::almost_same_as<T, i64>) {
                return I64;
            } else if constexpr (nt::almost_same_as<T, u64>) {
                return U64;
            } else if constexpr (nt::almost_same_as<T, f16>) {
                return F16;
            } else if constexpr (nt::almost_same_as<T, f32>) {
                return F32;
            } else if constexpr (nt::almost_same_as<T, f64>) {
                return F64;
            } else if constexpr (nt::almost_same_as<T, c16>) {
                return C16;
            } else if constexpr (nt::almost_same_as<T, c32>) {
                return C32;
            } else if constexpr (nt::almost_same_as<T, c64>) {
                return C64;
            } else {
                static_assert(nt::always_false<T>);
            }
        }
    };

    /// Whether this code was compiled for big-endian.
    constexpr bool is_big_endian() noexcept {
        return std::endian::native == std::endian::big;
    }

    /// Changes the endianness of the elements in an array, in-place.
    /// \param[in] ptr              Array of bytes to swap. Should contain at least (elements * bytes_per_element).
    /// \param n_elements           How many elements to swap.
    /// \param bytes_per_elements   Size, in bytes, of one element. If not 2, 4, or 8, do nothing.
    constexpr void swap_endian(std::byte* ptr, i64 n_elements, i64 bytes_per_elements) noexcept {
        if (bytes_per_elements == 2) {
            guts::swap_endian<2>(ptr, n_elements);
        } else if (bytes_per_elements == 4) {
            guts::swap_endian<4>(ptr, n_elements);
        } else if (bytes_per_elements == 8) {
            guts::swap_endian<8>(ptr, n_elements);
        }
    }

    /// Changes the endianness of the elements in an array, in-place.
    /// \param[in] ptr      Array of bytes to swap.
    /// \param n_elements   How many elements to swap.
    template<typename T>
    void swap_endian(T* ptr, i64 n_elements) noexcept {
        swap_endian(reinterpret_cast<std::byte*>(ptr), n_elements, sizeof(T));
    }
}

namespace noa::io {
    /// Converts the values in \p input, according to the desired \p encoding,
    /// and saves the converted values into \p output. Values are saved in the rightmost order.
    /// \tparam T           If \p T is complex, \p input is reinterpreted to the corresponding real type array,
    ///                     requiring its innermost dimension to be contiguous.
    /// \param[in] input    Values to serialize.
    /// \param[out] output  Array containing the serialized values.
    ///                     See Encoding::encoded_size() to know how many bytes will be written into this array.
    /// \param encoding     Desired encoding to apply to the input values before saving them into the output.
    template<nt::numeric T>
    void serialize(const Span<const T, 4>& input, const SpanContiguous<std::byte, 1>& output, Encoding encoding);

    /// Overload taking the output as an ostream.
    /// \param[out] output  Output stream to write into. The current position is used as a starting point.
    ///                     See Encoding::encoded_size() to know how many bytes will be written into this stream.
    /// \throws Exception   If the stream fails to write data, an exception is thrown. Note that the stream is
    ///                     reset to its good state before the exception is thrown.
    template<nt::numeric T>
    void serialize(const Span<const T, 4>& input, std::ostream& output, Encoding encoding);

    /// Deserializes the \p input, according to its \p encoding, converts the decoded values to
    /// the output value type, and save them in the \p output. Values are saved in the rightmost order.
    /// \tparam T           If \p T is complex, \p input is reinterpreted to the corresponding real type array,
    ///                     requiring its innermost dimension to be contiguous.
    /// \param[in] input    Values to deserialize.
    ///                     See Encoding::encoded_size() to know how many bytes will be read from this array.
    /// \param encoding     Encoding used to serialize the input values.
    /// \param[out] output  Deserialize values.
    template<nt::numeric T>
    void deserialize(const SpanContiguous<const std::byte, 1>& input, Encoding encoding, const Span<T, 4>& output);

    /// Overload taking the input as an istream.
    /// \param[in] input    Input stream to read from. The current position is used as a starting point.
    ///                     See Encoding::encoded_size() to know how many bytes will be read from this stream.
    /// \throws Exception   If the stream fails to read data, an exception is thrown. Note that the stream is
    ///                     reset to its good state before the exception is thrown.
    template<nt::numeric T>
    void deserialize(std::istream& input, Encoding encoding, const Span<T, 4>& output);
}

namespace noa::io {
    auto operator<<(std::ostream& os, Open mode) -> std::ostream&;
    auto operator<<(std::ostream& os, Encoding::Format data_type) -> std::ostream&;
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator)
namespace fmt {
    template<> struct formatter<noa::io::Open> : ostream_formatter {};
    template<> struct formatter<noa::io::Encoding::Format> : ostream_formatter {};
}
