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
     *
     * It is voluntarily brief and does not have virtual functions. Differences between files
     * can be quite significant in their way of opening/closing reading/writing.
     */
    class NOA_API File {
    protected:
        fs::path m_path{};
        std::unique_ptr<std::fstream> m_fstream{nullptr};

    public:
        /**
         * Set @a m_path to @a path and initialize the stream @a m_fstream. The file is not opened.
         * @tparam T        A valid path (or convertible to std::filesystem::path) by lvalue or rvalue.
         * @param[in] path  Filename to copy or move in the current instance.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        explicit File(T&& path)
                : m_path(std::forward<T>(path)),
                  m_fstream(std::make_unique<std::fstream>()) {}


        /**
         * Set @a m_path to @a path, open and associate it with the file stream @a m_fstream.
         * @tparam T            A valid path, by lvalue or rvalue.
         * @param[in] path      Filename to store in the current instance.
         * @param[in] mode      Any of the open mode (in|out|trunc|app|ate|binary).
         * @param[in] long_wait Wait for the file to exist for 10*30s, otherwise wait for 5*10ms.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        explicit File(T&& path, std::ios_base::openmode mode, bool long_wait = false)
                : m_path(std::forward<T>(path)),
                  m_fstream(std::make_unique<std::fstream>()) {
            File::open(m_path, *m_fstream, mode, long_wait);
        }


        /**
         * Open and associate the file in @a path with the file stream buffer of @a fstream.
         * @tparam T            A valid path, by lvalue or rvalue.
         * @tparam S            A file stream, one of @c std::(i|o)fstream.
         * @param[in] path      Path pointing at the filename to open.
         * @param[out] fstream  File stream, opened or closed, to associate with @c path.
         * @param[in] mode      Any of the @c std::ios_base::openmode.
         *                      in: Open ifstream. Operations on the ofstream will be ignored.
         *                      out: Open ofstream. Operations on the ifstream will be ignored.
         *                      binary: Disable text conversions.
         *                      ate: ofstream and ifstream seek the end of the file after opening.
         *                      app: ofstream seeks the end of the file before each writing.
         * @param[in] long_wait Wait for the file to exist for 10*30s, otherwise wait for 5*10ms.
         * @return              @c Errno::fail_close, if failed to close @a fstream before starting.
         *                      @c Errno::fail_open, if failed to open @a fstream.
         *                      @c Errno::fail_os, if an underlying OS API was raised.
         *                      @c Errno::good (0), otherwise.
         */
        template<typename S, typename = std::enable_if_t<std::is_same_v<S, std::ifstream> ||
                                                         std::is_same_v<S, std::ofstream> ||
                                                         std::is_same_v<S, std::fstream>>>
        static uint8_t open(const fs::path& path,
                            S& fstream,
                            std::ios_base::openmode mode,
                            bool long_wait = false) {
            if (close(fstream))
                return Errno::fail_close;

            uint8_t iterations = long_wait ? 10 : 5;
            size_t time_to_wait = long_wait ? 3000 : 10;

            if (OS::backup(path))
                return Errno::fail_os;
            if constexpr (!std::is_same_v<S, std::ifstream>) {
                if (mode & std::ios::out && OS::mkdir(path))
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
         *         @c Errno::good (0), otherwise.
         */
        template<typename S, typename = std::enable_if_t<std::is_same_v<S, std::ifstream> ||
                                                         std::is_same_v<S, std::ofstream> ||
                                                         std::is_same_v<S, std::fstream>>>
        static inline uint8_t close(S& fstream) {
            if (!fstream.is_open())
                return Errno::good;
            fstream.close();
            return fstream.fail() ? Errno::fail_close : Errno::good;
        }


        /** Whether or not @a m_path points to a regular file or a symlink. */
        inline bool exists(uint8_t& err) const noexcept {
            return OS::exists(m_path, err);
        }


        /**
         * Get the size of the file at @a m_path. Symlinks are followed.
         * @param[in] err   @c Errno to use. @c Errno::fail_os upon failure.
         * @return          The size of the file in bytes.
         * @throw ErrorCore If the file (or target) doesn't exist.
         */
        inline size_t size(uint8_t& err) const noexcept {
            return OS::size(m_path, err);
        }


        /** Whether or not the badbit of @a m_fstream is turned on. */
        [[nodiscard]] inline bool bad() const noexcept { return m_fstream->bad(); }


        /** Whether or not the eof of @a m_fstream is turned on. */
        [[nodiscard]] inline bool eof() const noexcept { return m_fstream->eof(); }


        /** Whether or not the failbit of @a m_fstream is turned on. */
        [[nodiscard]] inline bool fail() const noexcept { return m_fstream->fail(); }


        /** Whether or not the underlying stream is open. */
        [[nodiscard]] inline bool isOpen() const noexcept { return m_fstream->is_open(); }


        /** Whether or not the failbit of @a m_fstream is turned off. */
        [[nodiscard]] explicit operator bool() const noexcept { return !m_fstream->fail(); }


        /** Whether or not the failbit of @a m_fstream is turned on. */
        [[nodiscard]] bool operator!() const noexcept { return m_fstream->fail(); }
    };
}
