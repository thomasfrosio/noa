/**
 * @file Cast.h
 * @brief IO namespace and file I/O related functions.
 * @author Thomas - ffyr2w
 * @date 31/10/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/Traits.h"

#define RC(type, var) reinterpret_cast<type>(var)
#define SC(type, var) static_cast<type>(var)


/** Gathers a bunch of file I/O related functions. */
namespace Noa::IO {
    /**
     * Specifies the layout (i.e. the format) of the data, allowing us to correctly reinterpret the data.
     * @details Reading/writing from/to a sequence of bytes is done as follows:
     *          `char* <-(1)-> Layout* <-(2)-> float*` where (1) is a reinterpret_cast and (2) is a
     *          static_cast. Going from left to right is a read and from right to left is a write.
     */
    NOA_API enum class Layout {
        unset, byte, ubyte, int16, uint16, int32, uint32, float32
    };


    /** Returns the number of bytes of one element with a given layout. Returns 0 if the layout is not recognized. */
    NOA_API inline size_t bytesPerElement(Layout layout) noexcept {
        if (layout == Layout::byte || layout == Layout::ubyte)
            return 1;
        else if (layout == Layout::int16 || layout == Layout::uint16)
            return 2;
        else if (layout == Layout::float32 || layout == Layout::int32 || layout == Layout::uint32)
            return 4;
        else
            return 0;
    }


    NOA_API inline std::string toString(Layout layout) noexcept {
        if (layout == Layout::byte)
            return "char";
        else if (layout == Layout::ubyte)
            return "uchar";
        else if (layout == Layout::int16)
            return "int16";
        else if (layout == Layout::uint16)
            return "uint16";
        else if (layout == Layout::int32)
            return "int32";
        else if (layout == Layout::uint32)
            return "uint32";
        else if (layout == Layout::float32)
            return "float32";
        else if (layout == Layout::unset)
            return "unset layout";
        else
            return "unknown layout";
    }


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


    /**
     * Changes the endianness of the elements in an array, in place.
     * @param[in] ptr                   Array of bytes to swap. Should contain at least (elements * bytes_per_element).
     * @param[in] elements              How many elements to swap.
     * @param[in] bytes_per_element     Size, in bytes, of one element.
     */
    NOA_API inline errno_t swapEndian(char* ptr, size_t elements, size_t bytes_per_elements) {
        if (bytes_per_elements == 2)
            for (size_t i{0}; i < elements * bytes_per_elements; i += bytes_per_elements)
                reverse<2>(ptr + i);
        else if (bytes_per_elements == 4) {
            for (size_t i{0}; i < elements * bytes_per_elements; i += bytes_per_elements)
                reverse<4>(ptr + i);
        } else if (bytes_per_elements != 1)
            return Errno::invalid_argument;
        return Errno::good;
    }


    /**
     * Reads @a elements floats from @a fs into @a out.
     * @param[in] fs            File stream. Should be opened. The current position is used as starting point.
     * @param[out] ptr_out      The array of floats to write in. Size should be greater or equal than @a elements.
     * @param[in] elements      How many floats should be read from @c fs.
     * @param[in] layout        Layout of the data to read. It will be casted to float. See @c IO::Layout.
     * @param[in] use_buffer    Whether or not the data should be loaded by batches of ~17MB.
     *                          It requires less total memory but can be slower.
     * @param[in] swap_bytes    Whether or not the bytes of each element should be swapped BEFORE float
     *                          conversion, effectively changing the endianness of the input data.
     * @return                  @c Errno::fail_read, if failed to read from @a fs or if the eof was passed.
     *                          @c Errno::out_of_memory, if failed to allocate enough memory to load the file.
     *                          @c Errno::invalid_argument, if the layout is not supported nor recognized.
     */
    template<size_t bytes_batch = 1 << 24>
    NOA_API errno_t readFloat(std::fstream& fs, float* ptr_out, size_t elements,
                              Layout layout, bool use_buffer = true, bool swap_bytes = false) {
        static_assert(!(bytes_batch % 16), "batch should be a multiple of 16 bytes <=> 128 bits");

        size_t bytes_per_element = bytesPerElement(layout);
        if (!bytes_per_element)
            return Errno::invalid_argument;

        // Shortcut if the layout is float32.
        if (layout == Layout::float32) {
            fs.read(RC(char*, ptr_out), SC(std::streamsize, elements * bytes_per_element));
            if (fs.fail())
                return Errno::fail_read;
            else if (swap_bytes)
                swapEndian(RC(char*, ptr_out), elements, bytes_per_element);
            return Errno::good;
        }

        // All in or by batches.
        size_t bytes_remain = elements * bytes_per_element;
        size_t bytes_buffer = use_buffer && bytes_remain > bytes_batch ? bytes_batch : bytes_remain;
        auto* ptr_buffer = new(std::nothrow) char[bytes_buffer];
        if (!ptr_buffer)
            return Errno::out_of_memory;

        // Read until there's nothing left.
        errno_t err{Errno::good};
        for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
            bytes_buffer = std::min(bytes_remain, bytes_buffer);
            size_t elements_buffer = bytes_buffer / bytes_per_element;

            fs.read(ptr_buffer, SC(std::streamsize, bytes_buffer));
            if (fs.fail()) {
                err = Errno::fail_read;
                break;
            } else if (swap_bytes)
                swapEndian(ptr_buffer, elements_buffer, bytes_per_element);

            // Cast the layout to floats.
            if (layout == Layout::byte) {
                auto tmp = RC(signed char*, ptr_buffer);
                for (size_t idx{0}; idx < elements_buffer; ++idx)
                    ptr_out[idx] = SC(float, tmp[idx]);  // or *ptr_out = SC(float, *tmp++);

            } else if (layout == Layout::ubyte) {
                auto tmp = RC(unsigned char*, ptr_buffer);
                for (size_t idx{0}; idx < elements_buffer; ++idx)
                    ptr_out[idx] = SC(float, tmp[idx]);

            } else if (layout == Layout::int16) {
                auto tmp = RC(int16_t*, ptr_buffer);
                for (size_t idx{0}; idx < elements_buffer; ++idx)
                    ptr_out[idx] = SC(float, tmp[idx]);

            } else if (layout == Layout::uint16) {
                auto tmp = RC(uint16_t*, ptr_buffer);
                for (size_t idx{0}; idx < elements_buffer; ++idx)
                    ptr_out[idx] = SC(float, tmp[idx]);

            } else if (layout == Layout::int32) {
                auto tmp = RC(int32_t*, ptr_buffer);
                for (size_t idx{0}; idx < elements_buffer; ++idx)
                    ptr_out[idx] = SC(float, tmp[idx]);

            } else if (layout == Layout::uint32) {
                auto tmp = RC(uint32_t*, ptr_buffer);
                for (size_t idx{0}; idx < elements_buffer; ++idx)
                    ptr_out[idx] = SC(float, tmp[idx]);

            } else {
                err = Errno::invalid_argument;
                break;
            }
            ptr_out += elements_buffer;
        }
        delete[] ptr_buffer;
        return err;
    }


    /**
     * Writes @a elements floats from @a in into @a fs.
     * @param[in] fs            File stream. Should be open. The current position is used as starting point.
     * @param[out] in           The array of floats to read from. Size should be greater or equal than @a elements.
     * @param[in] elements      How many floats should be read from @a in and written into @c fs.
     * @param[in] flags         The desired layout to save the data (i.e. @a in) into. See @c IO::Layout.
     * @param[in] use_buffer    Whether or not the data should be written by batches of ~17MB.
     *                          This requires less total memory but can be slower.
     * @return                  @c Errno::fail_write, if failed to read from @a fs or if the eof was passed.
     *                          @c Errno::out_of_memory, if failed to allocate enough memory to load the file.
     *                          @c Errno::invalid_argument, if the layout is not supported nor recognized.
     */
    template<size_t bytes_batch = 1 << 24>
    NOA_API errno_t writeFloat(std::fstream& fs, float* ptr_in, size_t elements,
                               Layout layout, bool use_buffer = true) {
        static_assert(!(bytes_batch % 16), "batch should be a multiple of 16 bytes <=> 128 bits");

        size_t bytes_per_element = bytesPerElement(layout);
        if (!bytes_per_element)
            return Errno::invalid_argument;

        // Shortcut if the layout is float32.
        if (layout == Layout::float32) {
            fs.write(RC(char*, ptr_in), SC(std::streamsize, elements * bytes_per_element));
            if (fs.fail())
                return Errno::fail_write;
            return Errno::good;
        }

        // Read all in or by batches of ~17MB.
        size_t bytes_remain = elements * bytes_per_element;
        size_t bytes_buffer = use_buffer && bytes_remain > bytes_batch ? bytes_batch : bytes_remain;
        auto* buffer = new(std::nothrow) char[bytes_buffer];
        if (!buffer)
            return Errno::out_of_memory;

        // Read until there's nothing left.
        errno_t err{Errno::good};
        for (; bytes_remain > 0; bytes_remain -= bytes_buffer) {
            bytes_buffer = std::min(bytes_remain, bytes_buffer);
            size_t elements_buffer = bytes_buffer / bytes_per_element;

            // Cast the layout to floats.
            if (layout == Layout::byte) {
                auto tmp = RC(signed char*, buffer);
                for (size_t idx{0}; idx < elements_buffer; ++idx)
                    tmp[idx] = SC(signed char, ptr_in[idx]);

            } else if (layout == Layout::ubyte) {
                auto tmp = RC(unsigned char*, buffer);
                for (size_t idx{0}; idx < elements_buffer; ++idx)
                    tmp[idx] = SC(unsigned char, ptr_in[idx]);

            } else if (layout == Layout::int16) {
                auto tmp = RC(int16_t*, buffer);
                for (size_t idx{0}; idx < elements_buffer; ++idx)
                    tmp[idx] = SC(int16_t, ptr_in[idx]);

            } else if (layout == Layout::uint16) {
                auto tmp = RC(uint16_t*, buffer);
                for (size_t idx{0}; idx < elements_buffer; ++idx)
                    tmp[idx] = SC(uint16_t, ptr_in[idx]);

            } else if (layout == Layout::int32) {
                auto tmp = RC(int32_t*, buffer);
                for (size_t idx{0}; idx < elements_buffer; ++idx)
                    tmp[idx] = SC(int32_t, ptr_in[idx]);

            } else if (layout == Layout::uint32) {
                auto tmp = RC(uint32_t*, buffer);
                for (size_t idx{0}; idx < elements_buffer; ++idx)
                    tmp[idx] = SC(uint32_t, ptr_in[idx]);

            } else {
                err = Errno::invalid_argument;
                break;
            }

            fs.write(buffer, SC(std::streamsize, bytes_buffer));
            if (fs.fail()) {
                err = Errno::fail_write;
                break;
            }
            ptr_in += elements_buffer;
        }
        delete[] buffer;
        return err;
    }
}

#undef RC
#undef SC
