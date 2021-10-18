/// \file noa/common/IO.h
/// \brief IO namespace and I/O related functions.
/// \author Thomas - ffyr2w
/// \date 31 Oct 2020

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

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
    NOA_IH std::ostream& operator<<(std::ostream& os, Format format);

    /// Bit masks to control file openings.
    using open_mode_t = uint; // just to be a bit clearer about the input type
    enum OpenMode : open_mode_t {
        READ = 1U << 0,
        WRITE = 1U << 1,
        TRUNC = 1U << 2,
        BINARY = 1U << 3,
        APP = 1U << 4,
        ATE = 1U << 5
    };

    /// Switches from an OpenMode to a \c std::ios_base::openmode flag.
    NOA_IH constexpr std::ios_base::openmode toIOSBase(open_mode_t open_mode) noexcept;

    /// Data type used for (de)serialization.
    enum DataType {
        DATA_UNKNOWN,
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
    NOA_IH std::ostream& operator<<(std::ostream& os, DataType data_type);

    /// Returns the DataType corresponding to the type \p T.
    /// \tparam T (u|s)char, (u)short, (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    template<typename T>
    NOA_IH constexpr DataType getDataType() noexcept;

    /// Returns the range that \T values, about to be converted to \p data_type, should be in.
    /// \details (De)Serialization functions can clamp the values to fit the destination types. However, if
    ///          one wants to clamp the values beforehand, this function becomes really useful. It computes
    ///          the lowest and maximum value that the \p data_type can hold and clamps them to type \p T.
    /// \tparam T           Any data type (integer, floating-point, complex). See traits::is_data.
    ///                     If complex, real and imaginary parts are set with the same value.
    /// \param DataType     Data type. If DATA_UNKNOWN, do nothing.
    /// \param[out] min     Minimum \p T value in the range of \p data_type.
    /// \param[out] max     Maximum \p T value in the range of \p data_type.
    template<typename T>
    NOA_IH constexpr void getDataTypeMinMax(DataType data_type, T* min, T* max) noexcept;

    /// Whether this code was compiled for big-endian.
    NOA_IH bool isBigEndian() noexcept;

    /// Changes the endianness of the elements in an array, in-place.
    /// \param[in] ptr              Array of bytes to swap. Should contain at least (elements * bytes_per_element).
    /// \param elements             How many elements to swap.
    /// \param bytes_per_element    Size, in bytes, of one element. If not 2, 4, or 8, do nothing.
    NOA_IH void swapEndian(char* ptr, size_t elements, size_t bytes_per_elements) noexcept;

    template<typename T>
    NOA_IH void swapEndian(T* ptr, size_t elements) noexcept;

    /// Returns the number of bytes necessary to hold a number of \p elements formatted with a given \p data_type.
    /// \param data_type            Data type used for serialization. If DATA_UNKNOWN, returns 0.
    /// \param elements             Total number of elements.
    /// \param elements_per_row     Number of \p T elements per row.
    ///                             Only used when \p data_type is UINT4.
    ///                             This is to account for the half-byte padding at the end of odd rows with UINT4.
    ///                             If 0, the number of elements per row is assumed to be even.
    ///                             Otherwise, \p elements should be a multiple of \p elements_per_row.
    NOA_IH size_t getSerializedSize(DataType data_type, size_t elements, size_t elements_per_row = 0) noexcept;

    /// Converts the values in \p input, according to the desired \p data_type, and saves the converted
    /// values into the \p output array.
    /// \tparam T               (u|s)char, (u)short, (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Values to serialize.
    /// \param[out] output      On the \b host. Array containing the serialized values.
    ///                         See getSerializedSize to know how many bytes will be written in this array.
    /// \param data_type        Desired data type of the serialized values.
    ///                         If it is complex, \p T should be complex. If it is a scalar, \p T should be a scalar.
    /// \param elements         Total number of \p T elements to serialize.
    /// \param clamp            Whether the input values should be clamped to the range of the desired data type.
    ///                         If false, out-of-range values are undefined.
    /// \param swap_endian      Whether the endianness of the serialized data should be swapped.
    /// \param elements_per_row Number of \p T elements per row. See getSerializedSize for more details.
    template<typename T>
    NOA_HOST void serialize(const T* input, char* output, DataType data_type,
                            size_t elements, bool clamp = false, bool swap_endian = false, size_t elements_per_row = 0);

    /// Overload taking the input type as a data type.
    NOA_HOST void serialize(const void* input, DataType input_data_type, char* output, DataType output_data_type,
                            size_t elements, bool clamp = false, bool swap_endian = false, size_t elements_per_row = 0);

    /// Overload taking the output as an ostream.
    /// \param[out] output  Output stream to write into. The current position is used as starting point.
    ///                     See getSerializedSize to know how many bytes will be written into this stream.
    /// \throws Exception   If the stream fails to write data, an exception is thrown. Note that the stream is
    ///                     reset to its good state before the exception is thrown.
    template<typename T>
    NOA_HOST void serialize(const T* input, std::ostream& output, DataType data_type,
                            size_t elements, bool clamp = false, bool swap_endian = false, size_t elements_per_row = 0);

    /// Overload taking the input type as a data type and the output as an ostream.
    NOA_HOST void serialize(const void* input, DataType input_data_type, std::ostream& output, DataType data_type,
                            size_t elements, bool clamp = false, bool swap_endian = false, size_t elements_per_row = 0);

    /// Deserializes the \p input array according to its \p data_type, converts the values to \p T,
    /// and save them in the \p output array.
    /// \tparam T               Type that can be initiated from \p data_type.
    /// \param[in] input        On the \b host. Array to deserialize.
    ///                         See getSerializedSize() to know how many bytes will be read from this array.
    /// \param data_type        Data type of the serialized values.
    /// \param[out] output      On the \b host. Output array containing the deserialized values.
    /// \param elements         Total number of serialized elements to deserialize.
    /// \param elements_per_row Number of \p T elements per row. See getSerializedSize for more details.
    template<typename T>
    NOA_HOST void deserialize(const char* input, DataType data_type, T* output,
                              size_t elements, bool clamp = false, size_t elements_per_row = 0);

    /// Overload taking the output type as a data type.
    NOA_HOST void deserialize(const char* input, DataType input_data_type, void* output, DataType output_data_type,
                              size_t elements, bool clamp = false, size_t elements_per_row = 0);

    /// Overload taking the input as an istream.
    /// \param[in] input    Input stream to read from. The current position is used as starting point.
    ///                     See getSerializedSize to know how many bytes will be read from this stream.
    /// \param swap_endian  Whether the endianness of the serialized data should be swapped before conversion.
    /// \throws Exception   If the stream fails to read data, an exception is thrown. Note that the stream is
    ///                     reset to its good state before the exception is thrown.
    template<typename T>
    NOA_HOST void deserialize(std::istream& input, DataType data_type, T* output,
                              size_t elements, bool clamp = false,
                              bool swap_endian = false, size_t elements_per_row = 0);

    /// Overload taking the input as an istream and the output type as a data type.
    NOA_HOST void deserialize(std::istream& input, DataType input_data_type, void* output, DataType output_data_type,
                              size_t elements, bool clamp = false,
                              bool swap_endian = false, size_t elements_per_row = 0);
}

#define NOA_IO_INL_
#include "noa/common/io/IO.inl"
#undef NOA_IO_INL_
