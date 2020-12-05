/**
 * @file Header.h
 * @brief Header abstract class. Headers handle the metadata of image files.
 * @author Thomas - ffyr2w
 * @date 31/10/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/IO.h"


namespace Noa::Header {
    /**
     * Base header.
     * @note    The children should populate and keep up to date the members of this class, as the
     *          @c OS functions will use it to understand the data layout.
     * @note    The headers are intended to be stored as a public member variable of the ImageFile template class.
     */
    class Header {
    protected:
        iolayout_t m_layout{0u};
        bool m_is_big_endian{false};

    public:
        Header() = default;
        virtual ~Header() = default;

        /** Retrieves the IO layout, specifying the data layout. */
        [[nodiscard]] inline iolayout_t getLayout() const { return m_layout; }

        /** Sets the IO layout and updates whatever value it corresponds in the header file. */
        virtual errno_t setLayout(iolayout_t layout) = 0;

        /** Gets the endianness. False: little endian. True: big endian. */
        [[nodiscard]] inline bool isBigEndian() const { return m_is_big_endian; }

        [[nodiscard]] inline bool isSwapRequired() const {
            return isBigEndian() != OS::isBigEndian();
        }

        /** Sets the endianness and updates whatever value it corresponds in the header file. */
        virtual void setEndianness(bool big_endian) = 0;

        /** Gets the position, is bytes, where the data starts, relative to the beginning of the file */
        [[nodiscard]] virtual size_t getOffset() const = 0;

        /**
         * Reads the header from a file.
         * @param[in] fstream   File stream to read from. Position is reset at 0. Should be opened.
         * @return              @c Errno::fail_read, if the function wasn't able to read @a fstream.
         *                      @c Errno::invalid_data, if the header is not recognized.
         *                      @c Errno::not_supported, if the data is not supported.
         *                      @c Errno::good, otherwise.
         * @warning The position at which the stream is left does not necessarily correspond to the
         *          beginning of the data. Use getOffset().
         */
        virtual errno_t read(std::fstream& fstream) = 0;

        /**
         * Writes the header into a file.
         * @param[in] fstream   File stream to write into. Position is reset at 0. Should be opened.
         * @return              @c Errno::fail_write, if the function wasn't able to write into @a fstream.
         *                      @c Errno::good, otherwise.
         */
        virtual errno_t write(std::fstream& fstream) = 0;

        /** (Re)sets the metadata to the default values, i.e. create a new header. */
        virtual void reset() = 0;

        /** Gets the (x, y, z) size dimensions of the image. */
        [[nodiscard]] virtual Int3<size_t> getShape() const = 0;

        /** Sets the (x, y, z) size dimensions of the image. */
        virtual errno_t setShape(Int3<size_t>) = 0;

        /** Gets the (x, y, z) pixel size of the image. */
        [[nodiscard]] virtual Float3<float> getPixelSize() const = 0;

        /** Sets the (x, y, z) pixel size of the image. */
        virtual errno_t setPixelSize(Float3<float>) = 0;

        /** Prints the header. If @a brief is true, it should only print the shape and pixel size. */
        [[nodiscard]] virtual std::string toString(bool brief) const = 0;
    };
}
