/**
 * @file AbstractImageFile.h
 * @brief AbstractImageFile class. The interface to specialized image files.
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

#include "noa/API.h"
#include "noa/util/Constants.h"
#include "noa/util/Flag.h"
#include "noa/util/OS.h"
#include "noa/util/IntX.h"
#include "noa/util/FloatX.h"

namespace Noa {
    /**
     * The AbstractImageFile tries to offer an uniform API to work with all image files (e.g. @a MRCFile, @a TIFFiles,
     * @a EERFile, etc.). This works even if the image file type is unknown at compile time, using ImageFile::get().
     */
    class NOA_API AbstractImageFile {
    protected:
        using openmode_t = std::ios_base::openmode;
        fs::path m_path{};
        Noa::Flag<Errno> m_state{};

    public:
        AbstractImageFile() = default;
        virtual ~AbstractImageFile() = default;

        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit AbstractImageFile(T&& path) : m_path(std::forward<T>(path)) {}

        /** Whether or not the file exists. */
        inline bool exists() noexcept { return !m_state && OS::existsFile(m_path, m_state); }

        /** Gets the file size. Returns 0 if is fails. */
        inline size_t size() noexcept { return !m_state ? OS::size(m_path, m_state) : 0U; }

        /** Gets the path. */
        [[nodiscard]] inline const fs::path* path() const noexcept { return &m_path; }

        [[nodiscard]] inline Noa::Flag<Errno> state() const { return m_state; }
        inline void clear() { m_state = Errno::good; }

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
        virtual Noa::Flag<Errno> open(openmode_t, bool) = 0;

        /** Resets the path and opens the file. */
        virtual Noa::Flag<Errno> open(const fs::path&, openmode_t, bool) = 0;

        /** Resets the path and opens the file. */
        virtual Noa::Flag<Errno> open(fs::path&&, openmode_t, bool) = 0;

        /** Whether or not the file is open. */
        [[nodiscard]] virtual bool isOpen() const = 0;

        /** Closes the file. For some file format, there can be write operation to save buffered data. */
        virtual Noa::Flag<Errno> close() = 0;

        [[nodiscard]] virtual explicit operator bool() const noexcept = 0;

        [[nodiscard]] virtual Noa::Int3<size_t> getShape() const = 0;
        virtual Noa::Flag<Errno> setShape(Noa::Int3<size_t>) = 0;

        [[nodiscard]] virtual Noa::Float3<float> getPixelSize() const = 0;
        virtual Noa::Flag<Errno> setPixelSize(Noa::Float3<float>) = 0;

        [[nodiscard]] virtual std::string toString(bool) const = 0;

        /**
        * Reads the entire data into @a out. The ordering is (x=1, y=2, z=3).
        * @param ptr_out   Output array. Should be at least equal to @c shape.prod().
        * @note            @a m_state can be set to any returned @c Errno from IO::readFloat().
        */
        virtual Noa::Flag<Errno> readAll(float*) = 0;

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
        virtual Noa::Flag<Errno> readSlice(float*, size_t, size_t) = 0;

        /** Sets the data type for all future writing operation. */
        virtual Noa::Flag<Errno> setDataType(DataType) = 0;

        /**
         * Writes the entire file. The ordering is expected to be (x=1, y=2, z=3).
         * @param out   Output array. Should be at least equal to shape.prod().
         * @note        @a m_state can be set to any returned @c Errno from IO::writeFloat().
         */
        virtual Noa::Flag<Errno> writeAll(float*) = 0;

        /**
         * Writes slices (z sections) from @a out into the file.
         * @param out       Output float array.
         * @param z_pos     Slice to start writing.
         * @param z_count   Number of slices to write.
         * @note            @a m_state can be set to any returned @c Errno from IO::writeFloat().
         */
        virtual Noa::Flag<Errno> writeSlice(float*, size_t, size_t) = 0;
    };
}
