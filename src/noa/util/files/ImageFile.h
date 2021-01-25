/**
 * @file ImageFile.h
 * @brief ImageFile class. The interface to specialized image files.
 * @author Thomas - ffyr2w
 * @date 19 Dec 2020
 */
#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <ios>
#include <filesystem>
#include <string>

#include "noa/util/Types.h"
#include "noa/util/Errno.h"
#include "noa/util/OS.h"
#include "noa/util/IntX.h"
#include "noa/util/FloatX.h"

namespace Noa {
    /**
     * The ImageFile tries to offer an uniform API to work with all image files
     * (e.g. @a MRCFile, @a TIFFile, @a EERFile, etc.).
     */
    class ImageFile {
    protected:
        using openmode_t = std::ios_base::openmode;

    public:
        ImageFile() = default;
        virtual ~ImageFile() = default;

        /**
         * Returns an ImageFile with a path, but the file is NOT opened.
         * @return  One of the derived ImageFile or nullptr if the extension is not recognized.
         * @warning Before opening the file, check that the returned pointer is valid.
         *
         * @note    @c new(std::nothrow) could be used to prevent a potential bad_alloc, but in this case
         *          returning a nullptr could also be because the extension is not recognized...
         *
         * @note    The previous implementation was using an opaque pointer, but this is much simpler and probably
         *          safer since it is obvious that we are dealing with a unique_ptr and that it is necessary to check
         *          whether or not it is a nullptr before using it.
         */
        [[nodiscard]] static std::unique_ptr<ImageFile> get(const std::string& extension);

        [[nodiscard]] inline static std::unique_ptr<ImageFile> get(const fs::path& extension) {
            return get(extension.string());
        }

        // Below are the functions that derived classes should override.
        //  ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
    public:
        /**
         * (Re)Opens the file.
         * @param[in] mode      Should be one or a combination of the following:
         *                      @c in:              Read.           File should exists.
         *                      @c in|out:          Read & Write.   File should exists.     Backup copy.
         *                      @c out, out|trunc:  Write.          Overwrite the file.     Backup move.
         *                      @c in|out|trunc:    Read & Write.   Overwrite the file.     Backup move.
         * @param[in] wait      Wait for the file to exist for 10*3s, otherwise wait for 5*10ms.
         * @return              Any of the following error number:
         *                      @c Errno::invalid_state, if the image file type is not recognized.
         *                      @c Errno::fail_close, if failed to close the file before starting.
         *                      @c Errno::fail_open, if failed to open the file.
         *                      @c Errno::fail_os, if an underlying OS error was raised.
         *                      @c Errno::fail_read, if the
         *                      @c Errno::not_supported, if the file format is not supported.
         *                      @c Errno::invalid_data, if the file is not recognized.
         *                      @c Errno::good, otherwise.
         *
         * @note                Internally, the @c std::ios::binary is always considered on. On the
         *                      other hand, @c std::ios::app and @c std::ios::ate are always
         *                      considered off. Changing any of these bits has no effect.
         */
        virtual Errno open(openmode_t mode, bool wait) = 0;

        /** Resets the path and opens the file. */
        virtual Errno open(const fs::path&, openmode_t, bool) = 0;

        /** Resets the path and opens the file. */
        virtual Errno open(fs::path&&, openmode_t, bool) = 0;

        /** Whether or not the file is open. */
        [[nodiscard]] virtual bool isOpen() const = 0;

        /** Closes the file. For some file format, there can be write operation to save buffered data. */
        virtual Errno close() = 0;

        /** Gets the path. */
        [[nodiscard]] virtual const fs::path* path() const noexcept = 0;

        [[nodiscard]] virtual Int3 <size_t> getShape() const = 0;
        virtual Errno setShape(Int3 <size_t>) = 0;

        [[nodiscard]] virtual Float3<float> getPixelSize() const = 0;
        virtual Errno setPixelSize(Float3<float>) = 0;

        [[nodiscard]] virtual std::string toString(bool) const = 0;

        /**
        * Reads the entire data into @a out. The ordering is (x=1, y=2, z=3).
        * @param ptr_out   Output array. Should be at least equal to @c shape.prod().
        * @note            @a m_state can be set to any returned @c Errno from IO::readFloat().
        */
        virtual Errno readAll(float*) = 0;

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
        virtual Errno readSlice(float*, size_t, size_t) = 0;

        /** Sets the data type for all future writing operation. */
        virtual Errno setDataType(DataType) = 0;

        /**
         * Writes the entire file. The ordering is expected to be (x=1, y=2, z=3).
         * @param out   Output array. Should be at least equal to shape.prod().
         * @note        @a m_state can be set to any returned @c Errno from IO::writeFloat().
         */
        virtual Errno writeAll(const float*) = 0;

        /**
         * Writes slices (z sections) from @a out into the file.
         * @param out       Output float array.
         * @param z_pos     Slice to start writing.
         * @param z_count   Number of slices to write.
         * @note            @a m_state can be set to any returned @c Errno from IO::writeFloat().
         */
        virtual Errno writeSlice(const float*, size_t, size_t) = 0;
    };
}
