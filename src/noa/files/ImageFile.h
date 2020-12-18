/**
 * @file ImageFile.h
 * @brief ImageFile class, an uniform API for all image file types.
 * @author Thomas - ffyr2w
 * @date 24/10/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/files/File.h"
#include "noa/files/headers/Header.h"


namespace Noa {
    /** Holds a header and is an intermediary between the path, the header and the IO. */
    class NOA_API ImageFile : public File {
    public:
        Header header{};

    public:
        /**
         * Initializes the underlying file stream, but keeps the path and file header empty.
         * @note @a m_state can be set to @c Errno::good or Errno::no_handle.
         */
        inline explicit ImageFile() : File() { m_state = Errno::no_handle; }


        /**
         * Initializes the underlying file stream and file header, but keeps the path empty.
         * @note @a m_state can be set to @c Errno::good or Errno::no_handle.
         */
        inline explicit ImageFile(Header::Type type) : File(), header(type) { checkHandle_(); }


        /**
         * Stores @a path and initializes the underlying file stream and file header.
         * The file is NOT opened. The type of file header is deduced from the @a path extension.
         * @note @a m_state can be set to @c Errno::good or Errno::no_handle.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline explicit ImageFile(T&& path)
                : File(std::forward<T>(path)), header(m_path) { checkHandle_(); }


        /**
         * Stores @a path and initializes the underlying file stream and file header.
         * The file is NOT opened. The type of file header is NOT deduced from the @a path extension.
         * @note @a m_state can be set to @c Errno::good or Errno::no_handle.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline ImageFile(T&& path, Header::Type type)
                : File(std::forward<T>(path)), header(type) { checkHandle_(); }


        /**
         * Stores @a path and initializes the underlying file stream and file header.
         * The file is opened. The type of file header is deduced from the @a path extension.
         * @note @a m_state can be set to @c Errno::no_handle or any returned @c Errno of File::open();
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline ImageFile(T&& path, std::ios_base::openmode mode, bool long_wait = false)
                : File(std::forward<T>(path), mode, long_wait), header(m_path) { checkHandle_(); }


        /**
         * Stores @a path and initializes the underlying file stream and file header.
         * The file is opened. The type of file header is NOT deduced from the @a path extension.
         * @note @a m_state can be set to @c Errno::no_handle or any returned @c Errno of File::open();
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline ImageFile(T&& path, Header::Type type,
                         std::ios_base::openmode mode, bool long_wait = false)
                : File(std::forward<T>(path), mode, long_wait), header(type) { checkHandle_(); }


        /**
         * Opens and associates the stored file to the underlying file stream with File::open()
         * and read the file's metadata.
         * @note    The open mode is automatically switched to std::ios::binary mode.
         * @note    @a m_state can be set to any returned @c Errno from File::open() and Header::read().
         */
        inline errno_t open(std::ios_base::openmode mode, bool long_wait = false) {
            if (m_state)
                return m_state;
            m_state = File::open(m_path, *m_fstream, mode | std::ios::binary, long_wait);
            if (!m_state)
                m_state = header.read(*m_fstream);
            return m_state;
        }


        /**
         * Resets the path and (re)opens the file associated of this new path.
         * @note @a m_state can be set to any returned @c Errno from File::open() and Header::read().
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline errno_t open(T&& path, std::ios_base::openmode mode, bool long_wait = false) {
            m_path = std::forward<T>(path);
            return open(mode, long_wait);
        }


        /**
         * Writes the header into the file and closes the underlying file stream.
         * @note @a m_state can be is set to any returned @c Errno from File::close() and Header::write().
         */
        inline errno_t close() {
            if (m_state)
                return m_state;
            m_state = header.write(*m_fstream);
            if (!m_state)
                m_state = File::close(*m_fstream);
            return m_state;
        }


        /**
         * Reads the entire data into @a out. The ordering is (x=1, y=2, z=3).
         * @param ptr_out   Output array. Should be at least equal to @c shape.prod().
         * @note            @a m_state can be set to any returned @c Errno from IO::readFloat().
         */
        errno_t readAll(float* ptr_out);


        /**
         * Reads slices (z sections) from the file data and stores the data into @a out.
         * @note Slices refer to whatever dimension is in the last order, but since the only
         *       data ordering allowed is (x=1, y=2, z=3), slices are effectively z sections.
         *
         * @param ptr_out   Output float array. It should be large enough to contain the desired
         *                  data, that is `shape.prodSlice() * z_count`.
         * @param z_pos     Slice to start reading from.
         * @param z_count   Number of slices to read.
         * @note            @a m_state can be set to any returned @c Errno from IO::readFloat().
         */
        errno_t readSlice(float* ptr_out, size_t z_pos, size_t z_count);


        /**
         * Writes the entire file. The ordering is expected to be (x=1, y=2, z=3).
         * @param out   Output array. Should be at least equal to shape.prod().
         * @note        @a m_state can be set to any returned @c Errno from IO::writeFloat().
         */
        errno_t writeAll(float* out);


        /**
         * Writes slices (z sections) from @a out into the file.
         * @param out       Output float array.
         * @param z_pos     Slice to start writing.
         * @param z_count   Number of slices to write.
         * @note            @a m_state can be set to any returned @c Errno from IO::writeFloat().
         */
        errno_t writeSlice(float* out, size_t z_pos, size_t z_count);

    private:
        inline void checkHandle_() {
            if (!header.isHandled_())
                setState_(Errno::no_handle);
        }
    };
}
