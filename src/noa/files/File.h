/**
 * @file File.h
 * @brief
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
    class File {
    protected:
        fs::path m_path{};
        std::unique_ptr<std::fstream> m_fstream{nullptr};

    public:
        /**
         * Set @a m_path to @a path and initialize the stream @a m_fstream. The file is not opened.
         * @tparam T    A valid path (or convertible to std::filesystem::path) by lvalue or rvalue.
         * @param path  Filename to copy or move in the current instance.
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

    public:
        /**
         * Open and associate the file in @a path with the file stream buffer of @c fs.
         * @tparam T            A valid path, by lvalue or rvalue.
         * @tparam S            A file stream, one of @c std::(i|o)fstream.
         * @param[in] path      Path pointing at the filename to open.
         * @param[out] fs       File stream, opened or closed, to associate with @c path.
         * @param[in] mode      Any of the @c std::ios_base::openmode.
         *                      in: Open ifstream. Operations on the ofstream will be ignored.
         *                      out: Open ofstream. Operations on the ifstream will be ignored.
         *                      binary: Disable text conversions.
         *                      ate: ofstream and ifstream seek the end of the file after opening.
         *                      app: ofstream seeks the end of the file before each writing.
         * @param[in] long_wait Wait for the file to exist for 10*30s, otherwise wait for 5*10ms.
         * @return              Errno::fail_close, if failed to close @c fs before starting.
         *                      Errno::fail_open, if failed to open @c fs.
         *                      Errno::fail_os, if an underlying OS API was raised.
         *                      Errno::good (0), otherwise.
         */
        template<typename S,
                typename = std::enable_if_t<std::is_same_v<S, std::ifstream> ||
                                            std::is_same_v<S, std::ofstream> ||
                                            std::is_same_v<S, std::fstream>>>
        static uint8_t open(const fs::path& path,
                            S& fs,
                            std::ios_base::openmode mode,
                            bool long_wait = false) {
            if (!close(fs)) {
                return Errno::fail_close;
            }
            uint8_t iterations = long_wait ? 10 : 5;
            size_t time_to_wait = long_wait ? 3000 : 10;

            if (OS::backup(path))
                return Errno::fail_os;
            if constexpr (!std::is_same_v<S, std::ifstream>) {
                if (mode & std::ios::out && OS::mkdir(path))
                    return Errno::fail_os;
            }
            for (uint8_t it{0}; it < iterations; ++it) {
                fs.open(path.c_str(), mode);
                if (fs)
                    return Errno::good;
                std::this_thread::sleep_for(std::chrono::milliseconds(time_to_wait));
            }
            return Errno::fail_open;
        }


        /**
         * Close @a fstream if it is opened, otherwise don't do anything.
         * @return  Whether or not the stream was closed.
         */
        template<typename S, typename = std::enable_if_t<std::is_same_v<S, std::ifstream> ||
                                                         std::is_same_v<S, std::ofstream> ||
                                                         std::is_same_v<S, std::fstream>>>
        static inline bool close(S& fstream) noexcept {
            if (!fstream.is_open())
                return true;
            fstream.close();
            return !fstream.fail();
        }


        /** Whether or not @a m_path points to a file. */
        static inline bool exists(const fs::path& file, uint8_t& err) noexcept {
            try {
                auto status = fs::status(file);
                return fs::is_regular_file(status) || fs::is_symlink(status);
            } catch (std::exception& e) {
                err = Errno::fail_os;
                return false;
            }
        }


        /** Whether or not @a m_path points to a file. */
        inline bool exists(uint8_t& err) const noexcept {
            return File::exists(m_path, err);
        }


        /**
         * Get the size of the file at @a m_path. Symlinks are followed.
         * @return          The size of the file in bytes.
         * @throw ErrorCore If the file (or target) doesn't exist.
         */
        static inline size_t size(const fs::path& file, uint8_t& err) noexcept {
            return OS::size(file, err);
        }

        inline bool size(uint8_t& err) const noexcept {
            return OS::size(m_path, err);
        }
    };

}



