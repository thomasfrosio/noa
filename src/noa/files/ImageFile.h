/**
 * @file ImageFile.h
 * @brief MRC file class.
 * @author Thomas - ffyr2w
 * @date 24/10/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/files/File.h"

// Supported headers:
#include "noa/files/headers/MRCHeader.h"


namespace Noa {
    /**
     * An uniform API to handle image files.
     * This templated class holds a header and is an intermediary between the header and the IO functions.
     * To add support for a new file format, adding a new header for this format should be enough.
     */
    template<typename H, typename = std::enable_if<std::is_base_of_v<H, Header>>>
    class NOA_API ImageFile : public File {
    public:
        H header{};

    public:
        /** Initializes the underlying file stream, header and path with their default constructor. */
        explicit ImageFile() : File() {}


        /**
         * Stores @a path and initializes the underlying stream. The file is not opened.
         * @tparam T    A valid path (or convertible to a path), by lvalue or rvalue.
         * @param path  Filename to copy or move in the current instance.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit ImageFile(T&& path) : File(std::forward<T>(path)) {}


        /** Resets the path and opens the associated file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline errno_t open(T&& path, std::ios_base::openmode mode, bool long_wait = false) {
            m_path = std::forward<T>(path);
            return open(mode, long_wait);
        }


        /**
         * Opens and associates the stored file to the underlying file stream and read the file's metadata.
         * @param[in] mode      Any opening mode (in|out|trunc|app|ate|binary). Binary is automatically added.
         * @param[in] long_wait Wait for the file to exist for 10*30s, otherwise wait for 5*10ms.
         * @return              Any @c Errno from File::open() and Header::read().
         */
        inline errno_t open(std::ios_base::openmode mode, bool long_wait = false) {
            mode |= std::ios::binary;
            if (errno_t err = File::open(m_path, *m_fstream, mode, long_wait))
                return err;
            return header.read(*m_fstream);
        }


        /**
         * Writes the header into the file and closes the underlying file stream.
         * @return  Any @c Errno from File::close() and Header::write().
         */
        inline errno_t close() {
            if (errno_t err = header.write(*m_fstream))
                return err;
            return File::close(*m_fstream);
        }


        /**
         * Reads the entire file into @a out. The ordering is (x=1, y=2, z=3).
         * @param out   Output array. Should be at least equal to this->shape.elements().
         * @return      Any @c Errno returned by IO::readFloat().
         */
        errno_t read(float* out) {
            if (!isOpen())
                return Errno::fail_read;

            m_fstream->seekg(static_cast<long>(header.getOffset()));
            return IO::readFloat(*m_fstream, out, header.getShape().prod(),
                                 header.getLayout(), header.isSwapped());
        }


        /**
         * Reads slices (z sections) from the file data and stores the data into @a out.
         * @note Slices refer to whatever dimension is in the last order, but since the only
         *       data ordering allowed is (x=1, y=2, z=3), slices are effectively z sections.
         *
         * @param out       Output float array. It should be large enough to contain the desired
         *                  data, that is `shape.x * shape.y * z_count`.
         * @param z_pos     Slice to start reading.
         * @param z_count   Number of slices to read.
         * @return          Any @c Errno returned by IO::readFloat().
         */
        errno_t readSlice(float* out, size_t z_pos, size_t z_count) {
            if (!isOpen())
                return Errno::fail_read;

            iolayout_t io_layout = header.getLayout();
            Int3<size_t> shape = header.getShape();
            size_t elements_per_slice = shape.x * shape.y;
            size_t elements_to_read = elements_per_slice * z_count;
            size_t bytes_per_slice = elements_per_slice * IO::bytesPerElement(io_layout);

            m_fstream->seekg(static_cast<long>(header.getOffset() + z_pos * bytes_per_slice));
            return IO::readFloat(*m_fstream, out, elements_to_read,
                                 io_layout, header.isSwapped());
        }


        /**
         * Writes the entire file. The ordering is expected to be (x=1, y=2, z=3).
         * @param out   Output array. Should be at least equal to this->shape.elements().
         * @return      Any @c Errno returned by IO::writeFloat().
         */
        errno_t write(float* out) {
            if (!isOpen())
                return Errno::fail_write;

            m_fstream->seekg(static_cast<long>(header.getOffset()));
            return IO::writeFloat(*m_fstream, out, header.getShape().prod(),
                                  header.getLayout(), header.isSwapped());
        }


        /**
         * Writes slices (z sections) from @a out into the file.
         * @param out       Output float array.
         * @param z_pos     Slice to start writing.
         * @param z_count   Number of slices to write.
         * @return          Any @c Errno returned by IO::writeFloat().
         */
        errno_t writeSlice(float* out, size_t z_pos, size_t z_count) {
            if (!isOpen())
                return Errno::fail_read;

            iolayout_t io_layout = header.getLayout();
            Int3<size_t> shape = header.getShape();
            size_t elements_per_slice = shape.x * shape.y;
            size_t elements_to_read = elements_per_slice * z_count;
            size_t bytes_per_slice = elements_per_slice * IO::bytesPerElement(io_layout);

            m_fstream->seekg(static_cast<long>(header.getOffset() + z_pos * bytes_per_slice));
            return IO::writeFloat(*m_fstream, out, elements_to_read,
                                  io_layout, header.isSwapped());
        }
    };
}
