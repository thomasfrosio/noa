/**
 * @file Cast.h
 * @brief IO namespace and file I/O related functions.
 * @author Thomas - ffyr2w
 * @date 31/10/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/structures/Vectors.h"


/** Gathers a bunch of file I/O related functions. */
namespace Noa::IO {
    /**
     * Bit masks used by the functions in @c Noa::IO to specify the layout type.
     * @warning In most functions, one and only one layout is excepted and specifying more than
     *          one layout is undefined behavior, except specified otherwise.
     *
     * @details Reading/writing from/to a sequence of bytes is done as follows:
     *          `char* <-(1)-> layout* <-(2)-> float*` where (1) is a reinterpret_cast and (2) is a
     *          static_cast. Going from left to right is a read and from right to left is a write.
     */
    struct NOA_API Layout {
        static constexpr iolayout_t byte{0x0001u};        // 0x0 00000001
        static constexpr iolayout_t ubyte{0x0002u};       // 0x0 00000010
        static constexpr iolayout_t int16{0x0004u};       // 0x0 00000100
        static constexpr iolayout_t uint16{0x0008u};      // 0x0 00001000
        static constexpr iolayout_t int32{0x0010u};       // 0x0 00010000
        static constexpr iolayout_t uint32{0x0020u};      // 0x0 00100000
        static constexpr iolayout_t float32{0x0040u};     // 0x0 01000000
//        static constexpr iolayout_t complex32{0x0080u};      // 0x0 10000000
//        static constexpr iolayout_t complex64{0x0100u};      // 00000001 0x0

        static std::string toString(iolayout_t layout) {
            if (layout & Layout::byte)
                return "char";
            else if (layout & Layout::ubyte)
                return "unsigned char";
            else if (layout & Layout::int16)
                return "int16";
            else if (layout & Layout::uint16)
                return "unsigned int16";
            else if (layout & Layout::int32)
                return "int32";
            else if (layout & Layout::uint32)
                return "unsigned int32";
            else if (layout & Layout::float32)
                return "float32";
            else
                return "unknown layout";
        }
    };


    /**
     * How many bytes (char) does an element represent given a specific layout?
     * @param[in] layout    See @c IO::Flags. Only layouts are used.
     * @return              The number of bytes per element; 0 if the layout is not recognized.
     */
    inline size_t bytesPerElement(iolayout_t layout) noexcept {
        if (layout & Layout::byte || layout & Layout::ubyte)
            return 1;
        else if (layout & Layout::int16 || layout & Layout::uint16)
            return 2;
        else if (layout & Layout::float32 || layout & Layout::int32 || layout & Layout::uint32)
            return 4;
        else
            return 0;
    }


    /**
     * Swap @a count elements of @a size bytes from @a array, in place.
     * @param[in] array     Array of bytes to swap.
     * @param[in] count     How many elements (of @a size bytes) to swap.
     * @param[in] size      Size, in bytes, of one element. For instance, one float is 4 bytes.
     */
    inline void swap(char* array, size_t count, size_t size) {
        for (size_t element{0}; element < count; ++element)
            for (size_t byte{0}; byte < size / 2; ++byte)
                std::swap(array[byte], array[size - byte - 1]);
    }


    /**
     * Reads @a elements floats from @a fs into @a out.
     * @param[in] fs            File stream. Should be opened. The current position is used as starting point.
     * @param[out] out          The array of floats to write in. Size should be greater or equal than @a elements.
     * @param[in] elements      How many floats should be read from @c fs.
     * @param[in] layout        Layout of the data to read. It will be casted to float. See @c IO::Layout.
     * @param[in] swap_bytes    Whether or not the bytes of each element are swapped before float
     *                          conversion, effectively changing the endianness of the data.
     * @param[in] use_buffer    Whether or not the data is loaded by batches of ~17MB. This requires
     *                          less total memory but can be slower.
     * @return                  @c Errno::fail_read if failed to read from @a fs or if the eof was passed.
     *                          @c Errno::out_of_memory if failed to allocate enough memory to load the file.
     *                          @c Errno::invalid_argument if the layout is not supported nor recognized.
     */
    errno_t readFloat(std::fstream& fs, float* out, size_t elements,
                      iolayout_t layout, bool swap_bytes = false, bool use_buffer = true);


    /**
     * Writes @a elements floats from @a in into @a fs.
     * @param[in] fs            File stream. Should be open. The current position is used as starting point.
     * @param[out] in           The array of floats to read from. Size should be greater or equal than @a elements.
     * @param[in] elements      How many floats should be read from @a in and written into @c fs.
     * @param[in] flags         The desired layout to save the data (i.e. @a in) into. See @c IO::Layout.
     * @param[in] swap_bytes    Whether or not the bytes of each element are swapped before conversion
     *                          to the desired layout, effectively changing the endianness of the data.
     * @param[in] use_buffer    Whether or not the data is written by batches of ~17MB. This requires
     *                          less total memory but can be slower.
     * @return                  @c Errno::fail_write if failed to read from @a fs or if the eof was passed.
     *                          @c Errno::out_of_memory if failed to allocate enough memory to load the file.
     *                          @c Errno::invalid_argument if the layout is not supported nor recognized.
     */
    errno_t writeFloat(std::fstream& fs, float* in, size_t floats,
                       iolayout_t layout, bool swap_bytes = false, bool use_buffer = true);
}
