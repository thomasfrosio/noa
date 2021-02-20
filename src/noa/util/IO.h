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
#include "noa/Errno.h"
#include "noa/Types.h"

#define BYTES_BATCH 1<<24

/** Gathers a bunch of file I/O related functions. */
namespace Noa::IO {
    /** Specifies the type of the data, allowing to correctly (de)serialized data. */
    enum class DataType {
        byte, ubyte, int16, uint16, int32, uint32, float32, cint16, cfloat32
    };

    inline constexpr bool isComplex(DataType dtype) noexcept {
        if (dtype == DataType::cfloat32 || dtype == DataType::cint16)
            return true;
        else
            return false;
    }

    /** Returns the number of bytes of one element with a given layout. Returns 0 if the layout is not recognized. */
    inline constexpr size_t bytesPerElement(DataType dtype) noexcept {
        if (dtype == DataType::byte || dtype == DataType::ubyte)
            return 1u;
        else if (dtype == DataType::int16 || dtype == DataType::uint16)
            return 2u;
        else if (dtype == DataType::float32 || dtype == DataType::int32 ||
                 dtype == DataType::uint32 || dtype == DataType::cint16)
            return 4u;
        else if (dtype == DataType::cfloat32)
            return 8u;
        else
            return 0u;
    }

    /** Convert the data type into a string for logging. */
    const char* toString(DataType layout);

    /**
     * Reverses the bytes of an element.
     * @note    Knowing the number of bytes at compile time allows clang to optimize the
     *          entire function to a @c bswap. gcc is less good at it, but it is still much
     *          better than the runtime option.
     */
    template<size_t bytes_per_elements>
    inline void reverse(char* element) {
        for (size_t byte{0}; byte < bytes_per_elements / 2; ++byte)
            std::swap(element[byte], element[bytes_per_elements - byte - 1]);
    }

    /** Changes the endianness of the elements in an array, in place. */
    template<size_t bytes_per_elements>
    inline void swapEndian(char* ptr, size_t elements) {
        for (size_t i{0}; i < elements * bytes_per_elements; i += bytes_per_elements)
            reverse<bytes_per_elements>(ptr + i);
    }

    /**
     * Changes the endianness of the elements in an array, in place.
     * @param[in] ptr                   Array of bytes to swap. Should contain at least (elements * bytes_per_element).
     * @param[in] elements              How many elements to swap.
     * @param[in] bytes_per_element     Size, in bytes, of one element.
     */
    Errno swapEndian(char* ptr, size_t elements, size_t bytes_per_elements);

    /**
     * Converts an array of a given data type, i.e. @a input, to an array of floats, i.e. @a output.
     * @param[in] input     Source. Should contain at least @c n bytes, where @c n is @a elements * bytesPerElements(dtype).
     * @param[out] output   Destination. Should contain at least @c n bytes, where @c n is @a elements * 4.
     * @param dtype         Data type of @a input. Should not be @c cint16 or @c cfloat32.
     * @param[in] elements  Number of elements (i.e. floats) to process.
     * @return              Errno::dtype_complex, if @a dtype is @c cint16 or @c cfloat32.
     *                      Errno::good, otherwise.
     *
     * @note 30/12/20 - TF: The previous implementation, similar to toDataType() and based on type
     *                      punning with reinterpret_cast, was undefined behavior. This new
     *                      implementation complies to the standard and produces identical code
     *                      on -O2 with GCC and Clang. https://godbolt.org/z/fPxv7v
     */
    Errno toFloat(const char* input, float* output, DataType dtype, size_t elements);

    /**
     * Converts an array of a given data type, i.e. @a input, to an array of complex floats, i.e. @a output.
     * @param[in] input     Source. Should contain at least @a elements * bytesPerElements(dtype) bytes.
     * @param[out] output   Destination. Should contain at least @a elements * 8 bytes.
     * @param dtype         Complex data type of @a input. Should be either @c cint16 or @c cfloat32.
     * @param[in] elements  Number of complex floats (one complex float being 2 floats) to process.
     * @return              Errno::dtype_real, if @a dtype is anything else other than @c cint16 or @c cfloat32.
     *                      Errno::good, otherwise.
     */
    inline Errno toComplexFloat(const char* input, cfloat_t* output, DataType dtype, size_t elements) {
        if (dtype == DataType::cfloat32)
            return toFloat(input, reinterpret_cast<float*>(output), DataType::float32, elements * 2);
        else if (dtype == DataType::cint16)
            return toFloat(input, reinterpret_cast<float*>(output), DataType::int16, elements * 2);
        return Errno::dtype_real;
    }

    /**
     * Converts an array of floats, i.e. @a input, to an array of a given real data type, i.e. @a output.
     * @param[in] input     Source. Should contain at least @c n bytes, where @c n is @a elements * 4.
     * @param[out] output   Destination. Should contain at least @c n bytes, where @c n is @a elements * bytesPerElements(dtype).
     * @param dtype         Data type of @a input.
     * @param[in] elements  Number of elements (i.e. floats) to process.
     * @return Errno        Errno::dtype_complex, if @a dtype is not a complex data type.
     *                      Errno::good, otherwise.
     */
    Errno toDataType(const float* input, char* output, DataType dtype, size_t elements);

    /**
     * Converts an array of complex floats, i.e. @a input, to an array of a given complex data type, i.e. @a output.
     * @param[in] input     Source. Should contain at least @a elements * 8 bytes.
     * @param[out] output   Destination. Should contain at least @a elements * bytesPerElements(dtype) bytes.
     * @param dtype         Complex data type of @a input. Should be either @c cint16 or @c cfloat32.
     * @param[in] elements  Number of elements (i.e. complex floats) to process.
     */
    inline Errno toComplexDataType(const cfloat_t* input, char* output, DataType dtype, size_t elements) {
        if (dtype == DataType::cfloat32)
            return toDataType(reinterpret_cast<const float*>(input), output, DataType::float32, elements * 2);
        else if (dtype == DataType::cint16)
            return toDataType(reinterpret_cast<const float*>(input), output, DataType::int16, elements * 2);
        return Errno::dtype_real;
    }

    /**
     * Reads @a elements floats from @a fs into @a out.
     * @tparam bytes_batch      Number of bytes per batch. See @a batch.
     * @param[in] fs            File stream to read from. Should be opened. The current position is used as starting point.
     * @param[out] output       Destination. Should contain at least @c n bytes, where @c n is @a elements * 4.
     * @param[in] elements      How many floats should be read from @c fs.
     * @param[in] dtype         DataType of the data to read. It will be casted to float. See toFloat().
     * @param[in] batch         Whether or not the data should be loaded by batches.
     *                          It requires less total memory but can be slower.
     * @param[in] swap_bytes    Whether or not the bytes of each element should be swapped BEFORE float
     *                          conversion, effectively changing the endianness of the input data.
     * @return                  @c Errno::fail_read, if failed to read from @a fs or if the eof was passed.
     *                          @c Errno::out_of_memory, if failed to allocate enough memory to load the file.
     *                          @c Errno::good, otherwise.
     */
    template<size_t bytes_batch = BYTES_BATCH>
    Errno readFloat(std::fstream& fs, float* output, size_t elements,
                    DataType dtype, bool batch = true, bool swap_bytes = false) {
        static_assert(!(bytes_batch % 16), "batch should be a multiple of 16 bytes <=> 128 bits");

        size_t bytes_per_element = bytesPerElement(dtype);
        if (!bytes_per_element)
            return Errno::invalid_argument;

        // Shortcut if the dtype is float32.
        if (dtype == DataType::float32) {
            fs.read(reinterpret_cast<char*>(output),
                    static_cast<std::streamsize>(elements * bytes_per_element));
            if (fs.fail())
                return Errno::fail_read;
            else if (swap_bytes)
                swapEndian(reinterpret_cast<char*>(output), elements, bytes_per_element);
            return Errno::good;
        }

        // All in or by batches.
        size_t bytes_remain = elements * bytes_per_element;
        size_t bytes_buffer = batch && bytes_remain > bytes_batch ? bytes_batch : bytes_remain;
        std::unique_ptr<char[]> buffer(new(std::nothrow) char[bytes_buffer]);
        if (!buffer)
            return Errno::out_of_memory;

        // Read until there's nothing left.
        Errno err;
        for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
            bytes_buffer = std::min(bytes_remain, bytes_buffer);
            size_t elements_buffer = bytes_buffer / bytes_per_element;

            fs.read(buffer.get(), static_cast<std::streamsize>(bytes_buffer));
            if (fs.fail()) {
                err = Errno::fail_read;
                break;
            } else if (swap_bytes)
                swapEndian(buffer.get(), elements_buffer, bytes_per_element);

            err = toFloat(buffer.get(), output, dtype, elements_buffer);
            if (err)
                break;
            output += elements_buffer;
        }
        return err;
    }

    /**
     * Reads @a elements complex floats from @a fs into @a out.
     * @tparam bytes_batch      Number of bytes per batch. See @a batch.
     * @param[in] fs            File stream to read from. Should be opened. The current position is used as starting point.
     * @param[out] output       Destination. Should contain at least @a elements * 8 bytes.
     * @param[in] elements      How many complex floats should be read from @c fs.
     * @param[in] dtype         DataType of the complex data to read. Should be @c cint16 or @c cfloat32.
     * @param[in] batch         See readFloat().
     * @param[in] swap_bytes    See readFloat().
     * @return Errno            @c Errno::dtype_real, if @a dtype is a real data type.
     *                          Any Errno from readFloat().
     */
    template<size_t bytes_batch = BYTES_BATCH>
    inline Errno readComplexFloat(std::fstream& fs, cfloat_t* output, size_t elements,
                                  DataType dtype, bool batch = true, bool swap_bytes = false) {
        if (dtype == DataType::cfloat32)
            return readFloat<bytes_batch>(fs, reinterpret_cast<float*>(output), elements * 2,
                                          DataType::float32, batch, swap_bytes);
        else if (dtype == DataType::cint16)
            return readFloat<bytes_batch>(fs, reinterpret_cast<float*>(output), elements * 2,
                                          DataType::int16, batch, swap_bytes);
        return Errno::dtype_real;
    }

    /**
     * Writes @a elements floats from @a in into @a fs.
     * @tparam bytes_batch      Number of bytes per batch. See @a batch.
     * @param[out] input        Source. Should contain at least @c n bytes, where @c n is @a elements * 4.
     * @param[in] fs            File stream to write into. Should be open. The current position is used as starting point.
     * @param[in] elements      How many floats should be read from @a input and written into @a fs.
     * @param[in] dtype         DataType of the data to write. @a input will be casted to this type. See toDataType().
     * @param[in] batch         Whether or not the data should be written by batches.
     *                          This requires less total memory but can be slower.
     * @param[in] swap_bytes    Whether or not the bytes of each element should be swapped AFTER conversion
     *                          to the desired data type, effectively changing the endianness of the input data.
     * @return                  @c Errno::fail_write, if failed to write into @a fs or if the eof was passed.
     *                          @c Errno::out_of_memory, if failed to allocate enough memory.
     *                          @c Errno::otherwise, otherwise.
     */
    template<size_t bytes_batch = BYTES_BATCH>
    Errno writeFloat(const float* input, std::fstream& fs, size_t elements,
                     DataType dtype, bool batch = true, bool swap_endian = false) {
        static_assert(!(bytes_batch % 16), "batch should be a multiple of 16 bytes <=> 128 bits");

        size_t bytes_per_element = bytesPerElement(dtype);
        if (!bytes_per_element)
            return Errno::invalid_argument;

        // Shortcut if the dtype is float32.
        if (!swap_endian && dtype == DataType::float32) {
            fs.write(reinterpret_cast<const char*>(input),
                     static_cast<std::streamsize>(elements * bytes_per_element));
            if (fs.fail())
                return Errno::fail_write;
            return Errno::good;
        }

        // Read all in or by batches of ~17MB.
        size_t bytes_remain = elements * bytes_per_element;
        size_t bytes_buffer = batch && bytes_remain > bytes_batch ? bytes_batch : bytes_remain;
        std::unique_ptr<char[]> buffer(new(std::nothrow) char[bytes_buffer]);
        if (!buffer)
            return Errno::out_of_memory;

        // Read until there's nothing left.
        Errno err;
        for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
            bytes_buffer = std::min(bytes_remain, bytes_buffer);
            size_t elements_buffer = bytes_buffer / bytes_per_element;

            // Cast the dtype to floats.
            err = toDataType(input, buffer.get(), dtype, elements_buffer);
            if (err)
                break;
            if (swap_endian)
                swapEndian(buffer.get(), elements_buffer, bytes_per_element);

            fs.write(buffer.get(), static_cast<std::streamsize>(bytes_buffer));
            if (fs.fail()) {
                err = Errno::fail_write;
                break;
            }
            input += elements_buffer;
        }
        return err;
    }

    template<size_t bytes_batch = BYTES_BATCH>
    inline Errno writeComplexFloat(const cfloat_t* input, std::fstream& fs, size_t elements,
                            DataType dtype, bool batch = true, bool swap_endian = false) {
        if (dtype == DataType::cfloat32)
            return writeFloat<bytes_batch>(reinterpret_cast<const float*>(input), fs, elements * 2,
                                           DataType::float32, batch, swap_endian);
        else if (dtype == DataType::cint16)
            return writeFloat<bytes_batch>(reinterpret_cast<const float*>(input), fs, elements * 2,
                                           DataType::int16, batch, swap_endian);
        return Errno::dtype_real;
    }
}

#undef BYTES_BATCH
