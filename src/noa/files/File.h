/**
 * @file File.h
 * @brief Base class for files.
 * @author Thomas - ffyr2w
 * @date 28/10/2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/OS.h"


namespace Noa {
    /**
     * Base class for all file stream based class.
     * It is not copyable, but it is movable.
     */
    class NOA_API File {
    protected:
        fs::path m_path{};
        std::unique_ptr<std::fstream> m_fstream{nullptr};

    public:
        /**
         * Sets the path and initializes the stream. The file is not opened.
         * @tparam T        A valid path (or convertible to std::filesystem::path).
         * @param[in] path  Filename to copy or move into the current instance.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        explicit File(T&& path)
                : m_path(std::forward<T>(path)),
                  m_fstream(std::make_unique<std::fstream>()) {}


        /**
         * Sets and opens the associated file. The file is not read.
         * @tparam T            Same as default constructor.
         * @param[in] path      Same as default constructor.
         * @param[in] mode      See File::open().
         * @param[in] long_wait See File::open().
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        explicit File(T&& path, std::ios_base::openmode mode, bool long_wait = false)
                : m_path(std::forward<T>(path)),
                  m_fstream(std::make_unique<std::fstream>()) {
            File::open(m_path, *m_fstream, mode, long_wait);
        }


        /**
         * Opens and associates the file in @a path with @a fstream.
         * @tparam T            Same as default constructor.
         * @tparam S            A file stream, one of @c std::(i|o)fstream.
         * @param[in] path      Same as default constructor.
         * @param[out] fstream  File stream, opened or closed, to associate with @c path.
         * @param[in] mode      Any of the @c std::ios_base::openmode.
         *                      in: Open ifstream.
         *                      out: Open ofstream.
         *                      trunc: Discard the contents of the streams (i.e. overwrite).
         *                      binary: Disable text conversions.
         *                      ate: ofstream and ifstream seek the end of the file after opening.
         *                      app: ofstream seeks the end of the file before each writing.
         * @param[in] long_wait Wait for the file to exist for 10*3s, otherwise wait for 5*10ms.
         * @return              @c Errno::fail_close, if failed to close @a fstream before starting.
         *                      @c Errno::fail_open, if failed to open @a fstream.
         *                      @c Errno::fail_os, if an underlying OS error was raised.
         *                      @c Errno::good, otherwise.
         *
         * @note                If the file is opened in writing mode (@a mode & out), a backup is saved
         *                      before opening the file. If the file is overwritten (@a mode & trunc)
         *                      the file is moved, otherwise it is copied.
         */
        template<typename S, typename = std::enable_if_t<std::is_same_v<S, std::ifstream> ||
                                                         std::is_same_v<S, std::ofstream> ||
                                                         std::is_same_v<S, std::fstream>>>
        static errno_t open(const fs::path& path, S& fstream, std::ios_base::openmode mode,
                            bool long_wait = false) {
            if (close(fstream))
                return Errno::fail_close;

            uint8_t iterations = long_wait ? 10 : 5;
            size_t time_to_wait = long_wait ? 3000 : 10;

            if constexpr (!std::is_same_v<S, std::ifstream>) {
                if (mode & std::ios::out &&
                    (OS::mkdir(path) || OS::backup(path, mode ^ std::ios::trunc)))
                    return Errno::fail_os;
            }
            for (uint8_t it{0}; it < iterations; ++it) {
                fstream.open(path.c_str(), mode);
                if (fstream)
                    return Errno::good;
                std::this_thread::sleep_for(std::chrono::milliseconds(time_to_wait));
            }
            return Errno::fail_open;
        }


        /**
         * Close @a fstream if it is opened, otherwise don't do anything.
         * @return @c Errno::fail_close, if @a fstream failed to close.
         *         @c Errno::good, otherwise.
         */
        template<typename S, typename = std::enable_if_t<std::is_same_v<S, std::ifstream> ||
                                                         std::is_same_v<S, std::ofstream> ||
                                                         std::is_same_v<S, std::fstream>>>
        static inline errno_t close(S& fstream) {
            if (!fstream.is_open())
                return Errno::good;
            fstream.close();
            return fstream.fail() ? Errno::fail_close : Errno::good;
        }


        /** Whether or not @a m_path points to a regular file or a symlink. */
        inline bool exist(errno_t& err) const noexcept { return OS::existsFile(m_path, err); }


        /** Get the size (in bytes) of the file at @a m_path. Symlinks are followed. */
        inline size_t size(errno_t& err) const noexcept { return OS::size(m_path, err); }


        [[nodiscard]] inline bool bad() const noexcept { return m_fstream->bad(); }
        [[nodiscard]] inline bool eof() const noexcept { return m_fstream->eof(); }
        [[nodiscard]] inline bool fail() const noexcept { return m_fstream->fail(); }
        [[nodiscard]] inline bool isOpen() const noexcept { return m_fstream->is_open(); }
        [[nodiscard]] explicit operator bool() const noexcept { return !m_fstream->fail(); }
        [[nodiscard]] bool operator!() const noexcept { return m_fstream->fail(); }
    };
}
