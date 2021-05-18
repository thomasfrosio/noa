/**
 * @file IO.h
 * @brief IO namespace and I/O related functions.
 * @author Thomas - ffyr2w
 * @date 31/10/2020
 */
#pragma once

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <utility>      // std::swap
#include <algorithm>    // std::min
#include <fstream>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Types.h"

#define IO_BYTES_BATCH 16777216

/// Enumerators and bit masks related to IO.
namespace Noa::IO {
    /// Bit masks to control file openings.
    enum OpenMode : uint {
        READ = 1U << 0,
        WRITE = 1U << 1,
        TRUNC = 1U << 2,
        BINARY = 1U << 3,
        APP = 1U << 4,
        ATE = 1U << 5
    };

    /// Specifies the type of the data, allowing to correctly (de)serialized data.
    enum class DataType { BYTE, UBYTE, INT16, UINT16, INT32, UINT32, FLOAT32, CINT16, CFLOAT32 };
}

/// Gathers a bunch of file I/O related functions.
namespace Noa::IO {
    /// Switches an OpenMode to an @c ios_base flag.
    NOA_HOST std::ios_base::openmode toIOSBase(uint openmode);

    /// Whether or not @a dtype describe a complex data type.
    NOA_IH constexpr bool isComplex(DataType dtype) noexcept {
        return dtype == DataType::CFLOAT32 || dtype == DataType::CINT16;
    }

    /// Converts the data type into a string for logging. */
    NOA_HOST std::ostream& operator<<(std::ostream& os, DataType layout);

    /// Returns the number of bytes of one element with a given layout. Returns 0 if the layout is not recognized.
    NOA_IH constexpr size_t bytesPerElement(DataType dtype) noexcept {
        switch (dtype) {
            case DataType::BYTE:
            case DataType::UBYTE:
                return 1;
            case DataType::INT16:
            case DataType::UINT16:
                return 2;
            case DataType::FLOAT32:
            case DataType::INT32:
            case DataType::UINT32:
            case DataType::CINT16:
                return 4;
            case DataType::CFLOAT32:
                return 8;
            default:
                NOA_THROW("DEV: missing code path, got {}", dtype);
        }
    }

    /**
     * Reverses the bytes of an element.
     * @note    Knowing the number of bytes at compile time allows clang to optimize the
     *          entire function to a @c bswap. gcc is less good at it, but it is still much
     *          better than the runtime option.
     */
    template<size_t BYTES_PER_ELEMENTS>
    NOA_IH void reverse(char* element) {
        for (size_t byte{0}; byte < BYTES_PER_ELEMENTS / 2; ++byte)
            std::swap(element[byte], element[BYTES_PER_ELEMENTS - byte - 1]);
    }

    /// Changes the endianness of the elements in an array, in place.
    template<size_t BYTES_PER_ELEMENTS>
    NOA_IH void swapEndian(char* ptr, size_t elements) {
        for (size_t i{0}; i < elements * BYTES_PER_ELEMENTS; i += BYTES_PER_ELEMENTS)
            reverse<BYTES_PER_ELEMENTS>(ptr + i);
    }

    /**
     * Changes the endianness of the elements in an array, in place.
     * @param[in] ptr               Array of bytes to swap. Should contain at least (elements * bytes_per_element).
     * @param elements              How many elements to swap.
     * @param bytes_per_element     Size, in bytes, of one element.
     */
    NOA_HOST void swapEndian(char* ptr, size_t elements, size_t bytes_per_elements);

    /**
     * Converts an array of a given data type, i.e. @a input, to an array of floats, i.e. @a output.
     * @param[in] input     Source. Should contain at least @c n bytes, where @c n is @a elements * bytesPerElements(dtype).
     * @param[out] output   Destination. Should contain at least @c n bytes, where @c n is @a elements * 4.
     * @param dtype         Data type of @a input. Should not be @c CINT16 or @c CFLOAT32.
     * @param elements      Number of elements (i.e. floats) to process.
     *
     * @note 30/12/20 - TF: The previous implementation, similar to toDataType() and based on type
     *                      punning with reinterpret_cast, was undefined behavior. This new
     *                      implementation complies to the standard and produces identical code
     *                      on -O2 with GCC and Clang. https://godbolt.org/z/fPxv7v
     */
    NOA_HOST void toFloat(const char* input, float* output, DataType dtype, size_t elements);

    /// Overload for complex dtype. @see toFloat.
    NOA_IH void toComplexFloat(const char* input, cfloat_t* output, DataType dtype, size_t elements) {
        if (dtype == DataType::CFLOAT32)
            toFloat(input, reinterpret_cast<float*>(output), DataType::FLOAT32, elements * 2);
        else if (dtype == DataType::CINT16)
            toFloat(input, reinterpret_cast<float*>(output), DataType::INT16, elements * 2);
        else
            NOA_THROW("Expecting a complex dtype ({} or {}), got {}", DataType::CFLOAT32, DataType::CINT16, dtype);
    }

    /**
     * Converts an array of floats, i.e. @a input, to an array of a given real data type, i.e. @a output.
     * @param[in] input     Source. Should contain at least @c n bytes, where @c n is @a elements * 4.
     * @param[out] output   Destination. Should contain at least @c n bytes, where @c n is @a elements * bytesPerElements(dtype).
     * @param dtype         Data type of @a input. Should not be @c CINT16 or @c CFLOAT32.
     * @param elements      Number of elements (i.e. floats) to process.
     */
    NOA_HOST void toDataType(const float* input, char* output, DataType dtype, size_t elements);

    /// Overload for complex dtype. @see toDataType.
    NOA_IH void toComplexDataType(const cfloat_t* input, char* output, DataType dtype, size_t elements) {
        if (dtype == DataType::CFLOAT32)
            toDataType(reinterpret_cast<const float*>(input), output, DataType::FLOAT32, elements * 2);
        else if (dtype == DataType::CINT16)
            toDataType(reinterpret_cast<const float*>(input), output, DataType::INT16, elements * 2);
        else
            NOA_THROW("Expecting a complex dtype ({} or {}), got {}", DataType::CFLOAT32, DataType::CINT16, dtype);
    }

    /**
     * Reads @a elements floats from @a fs into @a output.
     * @tparam BYTES_BATCH  Number of bytes per batch. See @a batch.
     * @param[in] fs        File stream to read from. Should be opened. The current position is used as starting point.
     * @param[out] output   Destination. Should contain at least @c n bytes, where @c n is @a elements * 4.
     * @param elements      How many floats should be read from @c fs.
     * @param dtype         DataType of the data to read. It will be casted to float.
     * @param batch         Whether or not the data should be loaded by batches.
     *                      It requires less total memory but can be slower.
     * @param swap_bytes    Whether or not the bytes of each element should be swapped BEFORE float
     *                      conversion, effectively changing the endianness of the input data.
     * @throw Exception     If failed to read from @a fs or if the eof was passed.
     *                      If failed to allocate enough memory to load the file.
     */
    template<size_t BYTES_BATCH = IO_BYTES_BATCH>
    NOA_HOST void readFloat(std::fstream& fs, float* output, size_t elements,
                            DataType dtype, bool batch = true, bool swap_bytes = false) {
        static_assert(!(BYTES_BATCH % 16), "batch should be a multiple of 16 bytes <=> 128 bits");

        size_t bytes_per_element = bytesPerElement(dtype);

        // Shortcut.
        if (dtype == DataType::FLOAT32) {
            fs.read(reinterpret_cast<char*>(output), static_cast<std::streamsize>(elements * bytes_per_element));
            if (fs.fail())
                NOA_THROW("File stream error. Failed while reading (dtype:{}, elements:{}, batch:{})",
                          dtype, elements, batch);
            else if (swap_bytes)
                swapEndian(reinterpret_cast<char*>(output), elements, bytes_per_element);
            return;
        }

        // All in or by batches.
        size_t bytes_remain = elements * bytes_per_element;
        size_t bytes_buffer = batch && bytes_remain > BYTES_BATCH ? BYTES_BATCH : bytes_remain;
        std::unique_ptr<char[]> buffer(new(std::nothrow)
        char[bytes_buffer]);
        if (!buffer)
            NOA_THROW("Allocation failed. Requiring {} bytes (dtype:{})", bytes_buffer, dtype);

        // Read until there's nothing left.
        for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
            bytes_buffer = std::min(bytes_remain, bytes_buffer);
            size_t elements_buffer = bytes_buffer / bytes_per_element;

            fs.read(buffer.get(), static_cast<std::streamsize>(bytes_buffer));
            if (fs.fail()) {
                NOA_THROW("File stream error. Failed while reading (dtype:{}, elements:{}, batch:{})",
                          dtype, elements, batch);
            } else if (swap_bytes)
                swapEndian(buffer.get(), elements_buffer, bytes_per_element);

            toFloat(buffer.get(), output, dtype, elements_buffer);
            output += elements_buffer;
        }
    }

    /// Overload for complex dtype. @see readFloat.
    template<size_t BYTES_BATCH = IO_BYTES_BATCH>
    NOA_IH void readComplexFloat(std::fstream& fs, cfloat_t* output, size_t elements,
                                 DataType dtype, bool batch = true, bool swap_bytes = false) {
        if (dtype == DataType::CFLOAT32)
            readFloat<BYTES_BATCH>(fs, reinterpret_cast<float*>(output), elements * 2,
                                   DataType::FLOAT32, batch, swap_bytes);
        else if (dtype == DataType::CINT16)
            readFloat<BYTES_BATCH>(fs, reinterpret_cast<float*>(output), elements * 2,
                                   DataType::INT16, batch, swap_bytes);
        else
            NOA_THROW("Expecting a complex dtype ({} or {}), got {}", DataType::CFLOAT32, DataType::CINT16, dtype);
    }

    /**
     * Writes @a elements floats from @a input into @a fs.
     * @tparam BYTES_BATCH  Number of bytes per batch. See @a batch.
     * @param[out] input    Source. Should contain at least @c n bytes, where @c n is @a elements * 4.
     * @param[in] fs        File stream to write into. Should be open. The current position is used as starting point.
     * @param elements      How many floats should be read from @a input and written into @a fs.
     * @param dtype         DataType of the data to write. @a input will be casted to this type. See toDataType().
     * @param batch         Whether or not the data should be written by batches.
     *                      This requires less total memory but can be slower.
     * @param swap_bytes    Whether or not the bytes of each element should be swapped AFTER conversion
     *                      to the desired data type, effectively changing the endianness of the input data.
     * @throw Exception     If failed to write to @a fs or if the eof was passed.
     *                      If failed to allocate enough memory to load the file.
     */
    template<size_t BYTES_BATCH = IO_BYTES_BATCH>
    NOA_HOST void writeFloat(const float* input, std::fstream& fs, size_t elements,
                             DataType dtype, bool batch = true, bool swap_endian = false) {
        static_assert(!(BYTES_BATCH % 16), "batch should be a multiple of 16 bytes <=> 128 bits");

        size_t bytes_per_element = bytesPerElement(dtype);

        // Shortcut if the dtype is FLOAT32.
        if (!swap_endian && dtype == DataType::FLOAT32) {
            fs.write(reinterpret_cast<const char*>(input),
                     static_cast<std::streamsize>(elements * bytes_per_element));
            if (fs.fail())
                NOA_THROW("File stream error. Failed while writing (dtype:{}, elements:{}, batch:{})",
                          dtype, elements, batch);
            return;
        }

        // Read all in or by batches of ~17MB.
        size_t bytes_remain = elements * bytes_per_element;
        size_t bytes_buffer = batch && bytes_remain > BYTES_BATCH ? BYTES_BATCH : bytes_remain;
        std::unique_ptr<char[]> buffer(new(std::nothrow)
        char[bytes_buffer]);
        if (!buffer)
            NOA_THROW("Allocation failed. Requiring {} bytes (dtype:{})", bytes_buffer, dtype);

        // Read until there's nothing left.
        for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
            bytes_buffer = std::min(bytes_remain, bytes_buffer);
            size_t elements_buffer = bytes_buffer / bytes_per_element;

            // Cast the dtype to floats.
            toDataType(input, buffer.get(), dtype, elements_buffer);
            if (swap_endian)
                swapEndian(buffer.get(), elements_buffer, bytes_per_element);

            fs.write(buffer.get(), static_cast<std::streamsize>(bytes_buffer));
            if (fs.fail()) {
                NOA_THROW("File stream error. Failed while writing (dtype:{}, elements:{}, batch:{})",
                          dtype, elements, batch);
            }
            input += elements_buffer;
        }
    }

    /// Overload for complex dtype. @see writeFloat.
    template<size_t BYTES_BATCH = IO_BYTES_BATCH>
    NOA_IH void writeComplexFloat(const cfloat_t* input, std::fstream& fs, size_t elements,
                                  DataType dtype, bool batch = true, bool swap_endian = false) {
        if (dtype == DataType::CFLOAT32)
            writeFloat<BYTES_BATCH>(reinterpret_cast<const float*>(input), fs, elements * 2,
                                    DataType::FLOAT32, batch, swap_endian);
        else if (dtype == DataType::CINT16)
            writeFloat<BYTES_BATCH>(reinterpret_cast<const float*>(input), fs, elements * 2,
                                    DataType::INT16, batch, swap_endian);
        else
            NOA_THROW("Expecting a complex dtype ({} or {}), got {}", DataType::CFLOAT32, DataType::CINT16, dtype);
    }
}

#undef IO_BYTES_BATCH
