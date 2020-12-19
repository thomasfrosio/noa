/**
 * @file ImageFile.h
 * @brief ImageFile class, an uniform API for all image file types.
 * @author Thomas - ffyr2w
 * @date 24/10/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/String.h"
#include "noa/files/AbstractImageFile.h"
#include "noa/files/MRCFile.h"


namespace Noa {
    /**
     * The ImageFile offers a uniform API to work with all image files, that is @a AbstractImageFile
     * classes (e.g. @a MRCFile, @a TIFFiles, @a EERFile, etc.). This works even if the image file
     * type is unknown at compile time.
     * @note It is movable, but not copyable.
     * */
    class NOA_API ImageFile {
    private:
        using openmode_t = std::ios_base::openmode;
        std::unique_ptr<AbstractImageFile> m_handle{nullptr}; // The underlying image file.

    public:
        /** Creates an empty, not-initialized, instance. */
        inline explicit ImageFile() = default;


        /** Creates an initialized instance, but linked to no paths. */
        inline explicit ImageFile(IO::FileFormat type) { setHandle_(type); }


        /** Creates an initialized instance with a path, but the file is NOT opened. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline ImageFile(T&& path, IO::FileFormat type) { setHandle_(path, type); }


        /** Creates an initialized instance with a path and opens the file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline ImageFile(T&& path, IO::FileFormat type, openmode_t mode, bool long_wait = false) {
            setHandle_(path, type);
            open(mode, long_wait);
        }


        /**
         * Creates an initialized instance with a path, but the file is NOT opened.
         * The type of image file is deduced from the @a path extension.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline explicit ImageFile(T&& path) { setHandle_(path); }


        /**
         * Creates an initialized instance with a path and opens the file.
         * The type of image file is deduced from the @a path extension.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline ImageFile(T&& path, openmode_t mode, bool long_wait = false) {
            setHandle_(path);
            open(mode, long_wait);
        }


        /**
         * (Re)Opens the file.
         * @param[in] mode      Should be one or a combination of the following:
         *                      @c in:              Opens in reading mode. File should exists.
         *                      @c in|out:          Opens in reading and writing mode. File should exists. Backup copy.
         *                      @c out, out|trunc:  Opens in writing mode. Overwrite the file. Backup move.
         *                      @c in|out|trunc:    Opens in reading and writing mode. Overwrite the file. Backup move.
         * @param[in] wait      Wait for the file to exist for 10*3s, otherwise wait for 5*10ms.
         * @return              Any of the following error number:
         *                      @c Errno::no_handle, if the image file type is not recognized.
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
        inline errno_t open(openmode_t mode, bool wait = false) const {
            return m_handle ? m_handle->open(mode, wait) : Errno::no_handle;
        }


        /** Resets the path and opens the file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline errno_t open(T&& path, openmode_t mode, bool wait = false) const {
            return m_handle ? m_handle->open(path, mode, wait) : Errno::no_handle;
        }


        /**
         * Writes the header into the file and closes the underlying file stream.
         * @note @a m_state can be is set to any returned @c Errno from File::close() and Header::write().
         */
        inline errno_t close() const { return m_handle ? m_handle->close() : Errno::no_handle; }

        /** Whether or not the file exists. */
        inline bool exists() const noexcept { return m_handle && m_handle->exists(); }

        /** Gets the file size. Returns 0 if is fails. */
        inline size_t size() const noexcept { return m_handle ? m_handle->size() : 0U; }

        /** Gets the path. */
        [[nodiscard]] inline const fs::path* path() const noexcept {
            return m_handle ? m_handle->path() : nullptr;
        }


        [[nodiscard]] inline errno_t state() const {
            return m_handle ? m_handle->state() : Errno::no_handle;
        }

        inline void resetState() const { if (m_handle) m_handle->resetState(); }

        [[nodiscard]] inline Int3<size_t> getShape() const {
            return m_handle ? m_handle->getShape() : Int3<size_t>{};
        }

        errno_t setShape(Int3<size_t> shape) const {
            return m_handle ? m_handle->setShape(shape) : Errno::no_handle;
        }

        [[nodiscard]] Float3<float> getPixelSize() const {
            return m_handle ? m_handle->getPixelSize() : Float3<float>{};
        }

        errno_t setPixelSize(Float3<float> pixel_size) const {
            return m_handle ? m_handle->setPixelSize(pixel_size) : Errno::no_handle;
        }

        [[nodiscard]] std::string toString(bool brief) const {
            return m_handle ? m_handle->toString(brief) : std::string{};
        }

        [[nodiscard]] bool isOpen() const { return m_handle && m_handle->isOpen(); }


        /**
         * Reads the entire data into @a out. The ordering is (x=1, y=2, z=3).
         * @param ptr_out   Output array. Should be at least equal to @c shape.prod().
         * @note            @a m_state can be set to any returned @c Errno from IO::readFloat().
         */
        inline errno_t readAll(float* data) const {
            return m_handle ? m_handle->readAll(data) : Errno::no_handle;
        }


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
        inline errno_t readSlice(float* data, size_t z_pos, size_t z_count) const {
            return m_handle ? m_handle->readSlice(data, z_pos, z_count) : Errno::no_handle;
        }


        /**
         * Writes the entire file. The ordering is expected to be (x=1, y=2, z=3).
         * @param out   Output array. Should be at least equal to shape.prod().
         * @note        @a m_state can be set to any returned @c Errno from IO::writeFloat().
         */
        inline errno_t writeAll(float* data) const {
            return m_handle ? m_handle->writeAll(data) : Errno::no_handle;
        }


        /**
         * Writes slices (z sections) from @a out into the file.
         * @param out       Output float array.
         * @param z_pos     Slice to start writing.
         * @param z_count   Number of slices to write.
         * @note            @a m_state can be set to any returned @c Errno from IO::writeFloat().
         */
        inline errno_t writeSlice(float* data, size_t z_pos, size_t z_count) const {
            return m_handle ? m_handle->writeSlice(data, z_pos, z_count) : Errno::no_handle;
        }


        inline errno_t setStatistics(float min, float max, float mean, float rms) {
            if (auto mrcfile = dynamic_cast<MRCFile*>(m_handle.get()))
                return mrcfile->setStatistics(min, max, mean, rms);
            else
                return Errno::not_supported;
        }

    private:
        /** Sets the desired handle (i.e. the opaque pointer). */
        inline void setHandle_(IO::FileFormat type) {
            if (type == IO::FileFormat::MRC)
                m_handle = std::make_unique<MRCFile>();
        }


        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline void setHandle_(T&& path, IO::FileFormat type) {
            if (type == IO::FileFormat::MRC)
                m_handle = std::make_unique<MRCFile>(std::forward<T>(path));
        }


        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline void setHandle_(T&& path) {
            std::string extension = String::toLower(path.extension().string());
            if (extension == ".mrc" || extension == ".st" ||
                extension == ".rec" || extension == ".mrcs")
                setHandle_(std::forward<T>(path), IO::FileFormat::MRC);
            else if (extension == ".tif" || extension == ".tiff")
                setHandle_(std::forward<T>(path), IO::FileFormat::TIFF);
            else if (extension == ".eer")
                setHandle_(std::forward<T>(path), IO::FileFormat::EER);
        }
    };
}
