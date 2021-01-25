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
#include <utility>  // std::swap

#include "noa/Base.h"

#define BYTES_BATCH 1<<24

/** Gathers a bunch of file I/O related functions. */
namespace Noa::IO {
    /** Returns the number of bytes of one element with a given layout. Returns 0 if the layout is not recognized. */
    NOA_API inline constexpr size_t bytesPerElement(DataType layout) noexcept {
        if (layout == DataType::byte || layout == DataType::ubyte)
            return 1;
        else if (layout == DataType::int16 || layout == DataType::uint16)
            return 2;
        else if (layout == DataType::float32 || layout == DataType::int32 ||
                 layout == DataType::uint32)
            return 4;
        else
            return 0;
    }

    /** Convert the data type into a string for logging. */
    NOA_API std::string toString(DataType layout);

    /**
     * Reverses the bytes of an element.
     * @note    Knowing the number of bytes at compile time allows clang to optimize the
     *          entire function to a @c bswap. gcc is less good at it, but it is still much
     *          better than the runtime option.
     */
    template<size_t bytes_per_elements>
    NOA_API inline void reverse(char* element) {
        for (size_t byte{0}; byte < bytes_per_elements / 2; ++byte)
            std::swap(element[byte], element[bytes_per_elements - byte - 1]);
    }

    /** Changes the endianness of the elements in an array, in place. */
    template<size_t bytes_per_elements>
    NOA_API inline void swapEndian(char* ptr, size_t elements) {
        for (size_t i{0}; i < elements * bytes_per_elements; i += bytes_per_elements)
            reverse<bytes_per_elements>(ptr + i);
    }

    /**
     * Changes the endianness of the elements in an array, in place.
     * @param[in] ptr                   Array of bytes to swap. Should contain at least (elements * bytes_per_element).
     * @param[in] elements              How many elements to swap.
     * @param[in] bytes_per_element     Size, in bytes, of one element.
     */
    NOA_API Errno swapEndian(char* ptr, size_t elements, size_t bytes_per_elements);

    /**
     * Converts an array of a given data type, i.e. @a input, to an array of float, i.e. @a output.
     * @param[in] input     Source. Should contain at least @c n bytes, where @c n is @a elements * bytesPerElements(dtype).
     * @param[out] output   Destination. Should contain at least @c n bytes, where @c n is @a elements * 4.
     * @param dtype         Data type of @a input.
     * @param[in] elements  Number of elements (i.e. floats) to process.
     *
     * @note 30/12/20 - TF: The previous implementation, similar to toDataType() and based on type
     *                      punning with reinterpret_cast, was undefined behavior. This new
     *                      implementation complies to the standard and produces identical code
     *                      on -O2 with GCC and Clang. https://godbolt.org/z/fPxv7v
     */
    NOA_API void toFloat(const char* input, float* output, DataType dtype, size_t elements);

    /**
     * Converts an array of float, i.e. @a input, to an array of a given data type, i.e. @a output.
     * @param[in] input     Source. Should contain at least @c n bytes, where @c n is @a elements * 4.
     * @param[out] output   Destination. Should contain at least @c n bytes, where @c n is @a elements * bytesPerElements(dtype).
     * @param dtype         Data type of @a input.
     * @param[in] elements  Number of elements (i.e. floats) to process.
     */
    NOA_API void toDataType(const float* input, char* output, DataType dtype, size_t elements);

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
    NOA_API Errno readFloat(std::fstream& fs, float* output, size_t elements,
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
        auto* buffer = new(std::nothrow) char[bytes_buffer];
        if (!buffer)
            return Errno::out_of_memory;

        // Read until there's nothing left.
        Errno err;
        for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
            bytes_buffer = std::min(bytes_remain, bytes_buffer);
            size_t elements_buffer = bytes_buffer / bytes_per_element;

            fs.read(buffer, static_cast<std::streamsize>(bytes_buffer));
            if (fs.fail()) {
                err = Errno::fail_read;
                break;
            } else if (swap_bytes)
                swapEndian(buffer, elements_buffer, bytes_per_element);

            toFloat(buffer, output, dtype, elements_buffer);
            output += elements_buffer;
        }
        delete[] buffer;
        return err;
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
    NOA_API Errno writeFloat(const float* input, std::fstream& fs, size_t elements,
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
        auto* buffer = new(std::nothrow) char[bytes_buffer];
        if (!buffer)
            return Errno::out_of_memory;

        // Read until there's nothing left.
        Errno err;
        for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
            bytes_buffer = std::min(bytes_remain, bytes_buffer);
            size_t elements_buffer = bytes_buffer / bytes_per_element;

            // Cast the dtype to floats.
            toDataType(input, buffer, dtype, elements_buffer);
            if (swap_endian)
                swapEndian(buffer, elements_buffer, bytes_per_element);

            fs.write(buffer, static_cast<std::streamsize>(bytes_buffer));
            if (fs.fail()) {
                err = Errno::fail_write;
                break;
            }
            input += elements_buffer;
        }
        delete[] buffer;
        return err;
    }
}

#undef BYTES_BATCH
