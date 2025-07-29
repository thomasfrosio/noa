#pragma once

#include "noa/core/io/IO.hpp"
#include "noa/core/types/Span.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Half.hpp"

namespace noa::io {
    /// Compression scheme.
    /// TODO This is a very rudimentary support of compression schemes.
    ///      We currently only support this in BasicImageFile,
    ///      and the compression is done through the libtiff library.
    enum class Compression {
        NONE = 0, UNKNOWN = 1,
        LZW, DEFLATE
    };

    struct Encoding {
    public:
        enum class Type {
            UNKNOWN = 0,
            I8, U8, I16, U16, I32, U32, I64, U64, F16, F32, F64, C16, C32, C64,
            U4, CI16
        };
        using enum Type;

    public: // member variables
        /// Data type used for the encoding.
        Type dtype{};

        /// Whether the values should be clamped to the value range of the destination type.
        /// If false, out-of-range values are undefined.
        bool clamp{};

        /// Whether the endianness of the serialized data should be swapped.
        bool endian_swap{};

    public: // static function
        /// Returns the number of bytes necessary to hold a given number of elements with the current dtype.
        [[nodiscard]] static constexpr auto encoded_size(Type dtype, i64 n_elements) -> i64 {
            switch (dtype) {
                case U4: {
                    check(is_even(n_elements), "u4 encoding requires an even number of elements");
                    return n_elements / 2;
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

        /// Returns the data type corresponding to the type \p T.
        template<typename T>
        static constexpr auto to_dtype() noexcept -> Type {
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

    public: // member functions
        /// Returns the range that \T values, about to be serialized with the current dtype, should be in.
        /// \details (De)Serialization functions can clamp the values to fit the destination types. However, if
        ///          one wants to clamp the values beforehand, this function becomes really useful. It computes
        ///          the lowest and maximum value that the dtype can hold and clamps them to type \p T.
        /// \return Minimum and maximum \p T values in the range of the current dtype.
        template<typename T>
        [[nodiscard]] constexpr auto value_range() const -> Pair<T, T> {
            if constexpr (nt::scalar<T>) {
                switch (dtype) {
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
                        panic("Data type is not set");
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

        [[nodiscard]] constexpr auto encoded_size(i64 n_elements) const noexcept -> i64 {
            return encoded_size(dtype, n_elements);
        }
    };

    auto operator<<(std::ostream& os, Encoding::Type dtype) -> std::ostream&;
    auto operator<<(std::ostream& os, Encoding encoding) -> std::ostream&;
}

namespace fmt {
    template<> struct formatter<noa::io::Encoding::Type> : ostream_formatter {};
    template<> struct formatter<noa::io::Encoding> : ostream_formatter {};
}

namespace noa::io {
    /// Encodes the values in the input array into the output array. Values are saved in the BDHW order.
    /// \tparam T           If complex, the input is reinterpreted to the corresponding real type array,
    ///                     requiring its innermost dimension to be contiguous.
    /// \param[in] input    Values to encode.
    /// \param[out] output  Array where the encoded values are saved.
    ///                     See Encoding::encoded_size() to know how many bytes will be written into this array.
    /// \param encoding     Desired encoding to apply to the input values before saving them into the output.
    /// \param n_threads    Maximum number of OpenMP threads to use.
    ///                     Internally, this is clamped to one thread per 8M elements.
    template<nt::numeric T>
    void encode(
        const Span<const T, 4>& input,
        SpanContiguous<std::byte, 1> output,
        Encoding encoding,
        i32 n_threads = 1
    );

    /// Encodes the values in the input array into the file.
    /// The encoding starts at the current cursor position of the file.
    /// This is otherwise similar to the overload taking an array of bytes.
    template<nt::numeric T>
    void encode(
        const Span<const T, 4>& input,
        std::FILE* output,
        Encoding encoding,
        i32 n_threads = 1
    );

    /// Decodes the values in the input array into the output array. Values are saved in the BDHW order.
    /// \tparam T           If complex, the input is reinterpreted to the corresponding real type array,
    ///                     requiring its innermost dimension to be contiguous.
    /// \param[in] input    Values to decode.
    ///                     See Encoding::encoded_size() to know how many bytes will be read from this array.
    /// \param encoding     Encoding used to encode the input values.
    /// \param[out] output  Array where the decoded values are saved.
    /// \param n_threads    Maximum number of OpenMP threads to use.
    ///                     Internally, this is clamped to one thread per 8M elements.
    template<nt::numeric T>
    void decode(
        SpanContiguous<const std::byte, 1> input,
        Encoding encoding,
        const Span<T, 4>& output,
        i32 n_threads = 1
    );

    /// Decodes the values in the file into the output array.
    /// The decoding starts at the current cursor position of the file.
    /// This is otherwise similar to the overload taking an array of bytes.
    template<nt::numeric T>
    void decode(
        std::FILE* input,
        Encoding encoding,
        const Span<T, 4>& output,
        i32 n_threads = 1
    );
}
