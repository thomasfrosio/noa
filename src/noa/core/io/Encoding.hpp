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

    struct DataType {
        enum class Enum {
            UNKNOWN = 0,
            I8, U8, I16, U16, I32, U32, I64, U64,
            F16, F32, F64,
            C16, C32, C64,
            U4, CI16
        } value{};

    public: // behave like an enum class
        using enum Enum;
        constexpr DataType() = default;
        constexpr /*implicit*/ DataType(Enum value_) noexcept : value(value_) {}
        constexpr /*implicit*/ operator Enum() const noexcept { return value; }

        /// Implicit constructor from string literal.
        template<usize N>
        constexpr /*implicit*/ DataType(const char (& name)[N]) {
            std::string_view name_(name);
            if (name_ == "i8" or name_ == "I8")
                value = I8;
            else if (name_ == "u8" or name_ == "U8")
                value = U8;
            else if (name_ == "i16" or name_ == "I16")
                value = I16;
            else if (name_ == "u16" or name_ == "U16")
                value = U16;
            else if (name_ == "i32" or name_ == "I32")
                value = I32;
            else if (name_ == "u32" or name_ == "U32")
                value = U32;
            else if (name_ == "iz" or name_ == "I64")
                value = I64;
            else if (name_ == "u64" or name_ == "U64")
                value = U64;

            else if (name_ == "f16" or name_ == "F16")
                value = F16;
            else if (name_ == "f32" or name_ == "F32")
                value = F32;
            else if (name_ == "f64" or name_ == "F64")
                value = F64;

            else if (name_ == "c16" or name_ == "C16")
                value = C16;
            else if (name_ == "c32" or name_ == "C32")
                value = C32;
            else if (name_ == "c64" or name_ == "C64")
                value = C64;

            else if (name_ == "u4" or name_ == "U4")
                value = U4;
            else if (name_ == "ci16" or name_ == "CI16")
                value = CI16;

            else if (name_ == "unknown" or name_ == "UNKNOWN")
                value = UNKNOWN;

            else
                // If it is a constant expression, this creates a compile time error because throwing
                // an exception at compile time is not allowed. At runtime, it throws the exception.
                panic("invalid dtype");
        }

    public:
        /// Returns the number of bytes necessary to hold a given number of elements with the current dtype.
        [[nodiscard]] static constexpr auto n_bytes(DataType dtype, isize n_elements) -> isize {
            switch (dtype) {
                case U4: {
                    check(is_even(n_elements), "dtype=u4 requires an even number of elements");
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

        [[nodiscard]] constexpr auto n_bytes(isize n_elements) const noexcept -> isize {
            return n_bytes(value, n_elements);
        }

        /// Returns the data type corresponding to the type \p T.
        template<typename T>
        static constexpr auto from_type() noexcept -> DataType {
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
            } else if constexpr (nt::almost_same_as<T, isize>) {
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

        /// Returns the minimum and maximum \p T values in the range of the current dtype.
        /// \details (De)Serialization functions can clamp the values to fit the destination types. However, if
        ///          one wants to clamp the values beforehand, this function becomes really useful. It computes
        ///          the lowest and maximum value that the dtype can hold and clamps them to type \p T.
        template<typename T>
        [[nodiscard]] constexpr auto value_range() const -> Pair<T, T> {
            if constexpr (nt::scalar<T>) {
                switch (value) {
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
    };

    auto operator<<(std::ostream& os, DataType::Enum dtype) -> std::ostream&;
    auto operator<<(std::ostream& os, DataType encoding) -> std::ostream&;
}

namespace fmt {
    template<> struct formatter<noa::io::DataType::Enum> : ostream_formatter {};
    template<> struct formatter<noa::io::DataType> : ostream_formatter {};
}

namespace noa::io {
    struct EncodeOptions {
        /// Whether the values should be clamped to the value range of the destination type.
        /// If false, out-of-range values are undefined.
        bool clamp{};

        /// Whether the endianness of the serialized data should be swapped.
        bool endian_swap{};

        /// Maximum number of OpenMP threads to use.
        /// Internally, this is clamped to one thread per 8M elements.
        i32 n_threads{1};
    };
    using DecodeOptions = EncodeOptions;

    /// Encodes the values in the input array into the output array. Values are saved in the BDHW order.
    /// \tparam T           If complex, the input is reinterpreted to the corresponding real type array,
    ///                     requiring its innermost dimension to be contiguous.
    /// \param[in] input    Values to encode.
    /// \param[out] output  Array where the encoded values are saved. It is not allowed to overlap with the input
    ///                     and should be at least of size output_dtype.n_bytes(input.n_elements()).
    /// \param output_dtype Output data type.
    /// \param options      Encoding options.
    template<nt::numeric T>
    void encode(
        const Span<const T, 4>& input,
        const SpanContiguous<std::byte, 1>& output,
        const DataType& output_dtype,
        const EncodeOptions& options = {}
    );

    /// Encodes the values in the input array into the file.
    /// The input type is specified by the input data type.
    /// This is otherwise similar to the overload taking an array of numeric types.
    void encode(
        const SpanContiguous<const std::byte, 1>& input,
        const DataType& input_dtype,
        const SpanContiguous<std::byte, 1>& output,
        const DataType& output_dtype,
        const EncodeOptions& options = {}
    );

    /// Encodes the values in the input array into the output array.
    /// The encoding starts at the current cursor position of the file.
    /// This is otherwise similar to the overload taking an array of bytes.
    template<nt::numeric T>
    void encode(
        const Span<const T, 4>& input,
        std::FILE* output,
        const DataType& output_dtype,
        const EncodeOptions& options = {}
    );

    /// Encodes the values in the input array into the file.
    /// The encoding starts at the current cursor position of the file.
    /// The input type is specified by the input data type.
    /// This is otherwise similar to the overload taking an array of numeric types.
    void encode(
        const SpanContiguous<const std::byte, 1>& input,
        const DataType& input_dtype,
        std::FILE* output,
        const DataType& output_dtype,
        const EncodeOptions& options = {}
    );
}

namespace noa::io {
    /// Decodes the values in the input array into the output array. Values are saved in the BDHW order.
    /// \tparam T           If complex, the input is reinterpreted to the corresponding real type array,
    ///                     requiring its innermost dimension to be contiguous.
    /// \param[in] input    Bytes to decode. It is not allowed to overlap with the output
    ///                     and should be at least of size input_dtype.n_bytes(output.n_elements()).
    /// \param input_dtype  Data type of the input.
    /// \param[out] output  Array where the decoded values are saved.
    /// \param options      Decoding options
    template<nt::numeric T>
    void decode(
        const SpanContiguous<const std::byte, 1>& input,
        const DataType& input_dtype,
        const Span<T, 4>& output,
        const DecodeOptions& options = {}
    );

    /// Decodes the values in the input array into the output array.
    /// The output type is specified by the output data type.
    /// This is otherwise similar to the overload taking an array of numeric types.
    void decode(
        const SpanContiguous<const std::byte, 1>& input,
        const DataType& input_dtype,
        const SpanContiguous<std::byte, 1>& output,
        const DataType& output_dtype,
        const DecodeOptions& options = {}
    );

    /// Decodes the values in the file into the output array.
    /// The decoding starts at the current cursor position of the file.
    /// This is otherwise similar to the overload taking an array of bytes.
    template<nt::numeric T>
    void decode(
        std::FILE* input,
        const DataType& input_dtype,
        const Span<T, 4>& output,
        const DecodeOptions& options = {}
    );

    /// Decodes the values in the file into the output array.
    /// The decoding starts at the current cursor position of the file.
    /// The output type is specified by the output data type.
    /// This is otherwise similar to the overload taking an array of numeric types.
    void decode(
        std::FILE* input,
        const DataType& input_dtype,
        const SpanContiguous<std::byte, 1>& output,
        const DataType& output_dtype,
        const DecodeOptions& options = {}
    );
}
