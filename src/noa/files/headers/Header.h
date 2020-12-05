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
     * This class is mainly an abstract, but it also stores the @c IO::Layout.
     * @note    The children should populate and keep up to date this flag, as the @c OS functions
     *          will use it to understand the data layout.
     * @note    The headers are intended to be stored as a public member variable of the ImageFile
     *          template class.
     *
     * @details To properly interact with the ImageFile and the user, a header should have the
     *          following public functions:
     *  -# Access header:           read() and write() functions, to read (and initialize) the header
     *                              from a file stream and write the header into a file stream.
     *  -# Understand the data:     getIO() and getOffset() functions. They are used by the ImageFile
     *                              to know everything there is to know about the data (e.g. layout,
     *                              endianness, offset, etc.).
     *  -# Understand the metadata: reset(), print(), getShape(), setShape(), getPixelSize() and
     *                              setPixelSize(). These are meant to interact and modify the meta
     *                              data of the file.
     */
    class Header {
    protected:
        iolayout_t m_io_layout{0u};
        iolayout_t m_io_option{0u};

    public:
        /** Constructor. Headers should have a constructor with no arguments. */
        Header() = default;

        /** Destructor. */
        virtual ~Header() = default;

        /** Retrieve the IO layout, specifying the data layout and options. */
        [[nodiscard]] inline iolayout_t getLayout() const { return m_io_layout; }

        /** Retrieve the IO options, specifying the data options. */
        [[nodiscard]] inline iolayout_t getOption() const { return m_io_layout; }

        /** Get the position, is bytes, where the data starts, relative to the beginning of the file */
        [[nodiscard]] virtual size_t getOffset() const = 0;


        /**
         *
         * @param layout
         * @return
         */
        virtual errno_t setLayout(iolayout_t layout) = 0;


        /**
         * Read the header from @a fstream.
         * @see ImageFile::open()
         * @param[in] fstream   File stream to read from. Position is reset at 0.
         * @return              @c Errno::fail_read if the function wasn't able to read @a fstream.
         *                      @c Errno::invalid_data if the header is not recognized.
         *                      @c Errno::not_supported if the data is not supported.
         *                      @c Errno::good (0) otherwise.
         * @warning The position at which the stream is left does not necessarily correspond to the
         *          beginning of the data. Use getOffset().
         */
        virtual errno_t read(std::fstream& fstream) = 0;

        /**
         * Write the header into @a fstream.
         * @param[in] fstream   File stream to write into. Position is reset at 0.
         * @return              @c Errno::fail_write if the function wasn't able to write into @a fstream.
         *                      @c Errno::good (0) otherwise.
         */
        virtual errno_t write(std::fstream& fstream) = 0;

        /** (Re)set the metadata to the default values, i.e. create a new header. */
        virtual void reset() = 0;

        /** Get the x, y, and z size dimensions of the image. */
        [[nodiscard]] virtual Int3<size_t> getShape() const = 0;

        /** Set the x, y, and z size dimensions of the image. */
        virtual errno_t setShape(Int3<size_t>) = 0;

        /** Get the x, y, and z pixel size of the image. */
        [[nodiscard]] virtual Float3<float> getPixelSize() const = 0;

        /** Set the x, y, and z pixel size of the image. */
        virtual errno_t setPixelSize(Float3<float>) = 0;

        /** Print the header. If @a brief is true, it should only print the shape and pixel size. */
        virtual void print(bool brief) const = 0;
    };
}
