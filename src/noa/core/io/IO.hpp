/// \file noa/core/IO.h
/// \brief IO namespace and I/O related functions.
/// \author Thomas - ffyr2w
/// \date 31 Oct 2020

#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"

/// Enumerators and bit masks related to IO.
namespace noa::io {
    enum Format {
        FORMAT_UNKNOWN,
        MRC,
        TIFF,
        EER,
        JPEG,
        PNG
    };
    inline std::ostream& operator<<(std::ostream& os, Format format);

    /// Bit masks to control file openings.
    using open_mode_t = uint32_t;
    enum OpenMode : open_mode_t {
        READ = 1U << 0,
        WRITE = 1U << 1,
        TRUNC = 1U << 2,
        BINARY = 1U << 3,
        APP = 1U << 4,
        ATE = 1U << 5
    };
    struct OpenModeStream { open_mode_t mode; };
    inline std::ostream& operator<<(std::ostream& os, OpenModeStream open_mode);

    inline constexpr bool isValidOpenMode(open_mode_t open_mode) noexcept;

    /// Switches from an OpenMode to a \c std::ios_base::openmode flag.
    inline constexpr std::ios_base::openmode toIOSBase(open_mode_t open_mode) noexcept;

    /// Data type used for (de)serialization.
    enum DataType {
        DTYPE_UNKNOWN,
        INT8,
        UINT8,
        INT16,
        UINT16,
        INT32,
        UINT32,
        INT64,
        UINT64,
        FLOAT16,
        FLOAT32,
        FLOAT64,
        CFLOAT16,
        CFLOAT32,
        CFLOAT64,

        UINT4, // not "real" type
        CINT16 // not "real" type
    };
    inline std::ostream& operator<<(std::ostream& os, DataType data_type);

    /// Returns the DataType corresponding to the type \p T.
    /// \tparam T Any data type.
    template<typename T>
    inline constexpr DataType dtype() noexcept;

    /// Returns the range that \T values, about to be converted to \p data_type, should be in.
    /// \details (De)Serialization functions can clamp the values to fit the destination types. However, if
    ///          one wants to clamp the values beforehand, this function becomes really useful. It computes
    ///          the lowest and maximum value that the \p data_type can hold and clamps them to type \p T.
    /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
    ///                     If complex, real and imaginary parts are set with the same value.
    /// \param DataType     Data type. If DTYPE_UNKNOWN, return 0.
    /// \return             Minimum and maximum \p T values in the range of \p data_type.
    template<typename T>
    inline constexpr std::pair<T, T> typeMinMax(DataType data_type) noexcept;

    /// Whether this code was compiled for big-endian.
    inline bool isBigEndian() noexcept;

    /// Changes the endianness of the elements in an array, in-place.
    /// \param[in] ptr              Array of bytes to swap. Should contain at least (elements * bytes_per_element).
    /// \param elements             How many elements to swap.
    /// \param bytes_per_element    Size, in bytes, of one element. If not 2, 4, or 8, do nothing.
    inline void swapEndian(byte_t* ptr, size_t elements, size_t bytes_per_elements) noexcept;

    /// Changes the endianness of the elements in an array, in-place.
    /// \param[in] ptr  Array of bytes to swap.
    /// \param elements How many elements to swap.
    template<typename T>
    inline void swapEndian(T* ptr, size_t elements) noexcept;

    /// Returns the number of bytes necessary to hold a number of \p elements formatted with a given \p data_type.
    /// \param data_type        Data type used for serialization. If DTYPE_UNKNOWN, returns 0.
    /// \param elements         Number of elements.
    /// \param elements_per_row Number of \p T elements per row.
    ///                         Only used when \p data_type is UINT4.
    ///                         This is to account for the half-byte padding at the end of odd rows with UINT4.
    ///                         If 0, the number of elements per row is assumed to be even.
    ///                         Otherwise, \p elements should be a multiple of \p elements_per_row.
    inline size_t serializedSize(DataType data_type, size_t elements, size_t elements_per_row = 0) noexcept;
}

namespace noa::io {
    /// Converts the values in \p input, according to the desired \p data_type,
    /// and saves the converted values into \p output. Values are saved in the rightmost order.
    /// \tparam T           Any data type. If \p T is complex, \p input is reinterpreted to the corresponding
    ///                     real type array, requiring its innermost dimension to be contiguous.
    /// \param[in] input    On the \b host. Values to serialize.
    /// \param strides      BDHW strides of \p input.
    /// \param shape        BDHW shape of \p input.
    /// \param[out] output  On the \b host. Array containing the serialized values.
    ///                     See serializedSize to know how many bytes will be written into this array.
    /// \param data_type    Desired data type of the serialized values.
    ///                     If it describes a complex value, \p T should be complex.
    ///                     If it describes a scalar, \p T should be a scalar.
    ///                     If it is UINT4, \p input should be C-contiguous.
    /// \param clamp        Whether the input values should be clamped to the range of the desired data type.
    ///                     If false, out-of-range values are undefined.
    /// \param swap_endian  Whether the endianness of the serialized data should be swapped.
    template<typename T>
    void serialize(const T* input, size4_t strides, size4_t shape,
                   byte_t* output, DataType data_type,
                   bool clamp = false, bool swap_endian = false);

    /// Overload taking the input type as a data type.
    /// \note \p strides and \p shape are in number of \p input_data_type elements.
    void serialize(const void* input, DataType input_data_type, size4_t strides, size4_t shape,
                   byte_t* output, DataType output_data_type,
                   bool clamp = false, bool swap_endian = false);

    /// Overload taking the output as an ostream.
    /// \param[out] output  Output stream to write into. The current position is used as starting point.
    ///                     See serializedSize to know how many bytes will be written into this stream.
    /// \throws Exception   If the stream fails to write data, an exception is thrown. Note that the stream is
    ///                     reset to its good state before the exception is thrown.
    template<typename T>
    void serialize(const T* input, size4_t strides, size4_t shape,
                   std::ostream& output, DataType data_type,
                   bool clamp = false, bool swap_endian = false);

    /// Overload taking the input type as a data type and the output as an ostream.
    /// \note \p strides and \p shape are in number of \p input_data_type elements.
    void serialize(const void* input, DataType input_data_type, size4_t strides, size4_t shape,
                   std::ostream& output, DataType data_type,
                   bool clamp = false, bool swap_endian = false);
}

namespace noa::io {
    /// Deserializes the \p input array according to its \p data_type, converts the values to \p T,
    /// and save them in the \p output array. Values are saved in the rightmost order.
    /// \tparam T           Any data type. If \p T is complex, \p output is reinterpreted to the corresponding
    ///                     real type array, requiring its innermost dimension to be contiguous.
    /// \param[in] input    On the \b host. Array containing the values to deserialize.
    ///                     See serializedSize to know how many bytes will be read from this array.
    /// \param data_type    Data type of the serialized values.
    ///                     If it describes a complex value, \p T should be complex.
    ///                     If it describes a scalar, \p T should be a scalar.
    ///                     If it is UINT4, \p output should be C-contiguous.
    /// \param[out] output  On the \b host. Values to serialize.
    /// \param strides      BDHW strides of \p output.
    /// \param shape        BDHW shape of \p output.
    /// \param clamp        Whether the deserialized values should be clamped to the \p T range.
    ///                     If false, out-of-range values are undefined.
    /// \param swap_endian  Whether the endianness of the serialized data should be swapped.
    template<typename T>
    void deserialize(const byte_t* input, DataType data_type,
                     T* output, size4_t strides, size4_t shape,
                     bool clamp = false, bool swap_endian = false);

    /// Overload taking the output type as a data type.
    /// \note \p strides and \p shape are in number of \p output_data_type elements.
    void deserialize(const byte_t* input, DataType input_data_type,
                     void* output, DataType output_data_type, size4_t strides, size4_t shape,
                     bool clamp = false, bool swap_endian = false);

    /// Overload taking the input as an istream.
    /// \param[in] input    Input stream to read from. The current position is used as starting point.
    ///                     See serializedSize to know how many bytes will be read from this stream.
    /// \throws Exception   If the stream fails to read data, an exception is thrown. Note that the stream is
    ///                     reset to its good state before the exception is thrown.
    template<typename T>
    void deserialize(std::istream& input, DataType data_type,
                     T* output, size4_t strides, size4_t shape,
                     bool clamp = false, bool swap_endian = false);

    /// Overload taking the input as an istream and the output type as a data type.
    /// \note \p strides and \p shape are in number of \p output_data_type elements.
    void deserialize(std::istream& input, DataType input_data_type,
                     void* output, DataType output_data_type, size4_t strides, size4_t shape,
                     bool clamp = false, bool swap_endian = false);
}

#define NOA_IO_INL_
#include "noa/core/io/IO.inl"
#undef NOA_IO_INL_
