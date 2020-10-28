/**
 * @file TextFile.h
 * @brief Text file class.
 * @author Thomas - ffyr2w
 * @date 9 Oct 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/OS.h"


namespace Noa {

    /**
     * Basic text file, which handles a file stream.
     * It is not copyable, but it is movable.
     */
    class NOA_API TextFile {
    protected:
        std::filesystem::path m_path{};
        std::unique_ptr<std::fstream> m_fstream{nullptr};

    public:
        /**
         * Set @a m_path to @a path and initialize the stream @a m_fstream. The file is not opened.
         * @tparam T    A valid path (or convertible to std::filesystem::path) by lvalue or rvalue.
         * @param path  Filename to copy or move in the current instance.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit TextFile(T&& path)
                : m_path(std::forward<T>(path)), m_fstream(std::make_unique<std::fstream>()) {}


        /**
         * Set @a m_path to @a path, open and associate it with the file stream @a m_fstream.
         * @tparam T            A valid path, by lvalue or rvalue.
         * @param[in] path      Filename to store in the current instance.
         * @param[in] mode      Any of the open mode (in|out|trunc|app|ate|binary).
         * @param[in] long_wait Wait for the file to exist for 10*30s, otherwise wait for 5*10ms.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit TextFile(T&& path, std::ios_base::openmode mode, bool long_wait = false)
                : m_path(std::forward<T>(path)), m_fstream(std::make_unique<std::fstream>()) {
            reopen(mode, long_wait);
        }


        /**
         * Reset @a m_path to @a path, open and associate it with the file stream @a m_fstream.
         * @tparam T            A valid path, by lvalue or rvalue.
         * @param[in] path      Filename to store in the current instance.
         * @param[in] mode      Any of the open mode (in|out|trunc|app|ate|binary).
         * @param[in] long_wait Wait for the file to exist for 10*3s, otherwise wait for 5*10ms.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline void open(T&& path, std::ios_base::openmode mode, bool long_wait = false) {
            m_path = std::forward<T>(path);
            reopen(mode, long_wait);
        }


        /**
         * Close the file and reopen it.
         * @param[in] mode      Any of the open mode (in|out|trunc|app|ate|binary).
         * @param[in] long_wait Wait for the file to exist for 10*30s, otherwise wait for 5*10ms.
         */
        inline void reopen(std::ios_base::openmode mode, bool long_wait = false) {
            open(m_path, *m_fstream, mode, long_wait);
        }


        /** Close @a m_fstream if it is open, otherwise don't do anything. */
        inline void close() {
            if (!close(*m_fstream)) {
                NOA_CORE_ERROR("\"{}\": error while closing to file. {}",
                               m_path.c_str(), std::strerror(errno));
            }
        }


        /**
         * Write a formatted string into the ofstream.
         * @tparam Args     Anything accepted by @c fmt::format()
         * @param[in] args  C-string and|or variable(s) used to compute the formatted string.
         *
         * @note            This function depends on the ofstream position. If std::ios::app,
         *                  the position is set to the end of the file at every call.
         */
        template<typename... Args>
        void write(Args&& ... args) {
            std::string message = fmt::format(std::forward<Args>(args)...);
            m_fstream->write(message.c_str(), static_cast<std::streamsize>(message.size()));
            if (m_fstream->fail()) {
                if (!m_fstream->is_open()) {
                    NOA_CORE_ERROR("\"{}\": file is not open. Open it with open() or reopen()",
                                   m_path.c_str());
                } else {
                    NOA_CORE_ERROR("\"{}\": error while writing to file. {}",
                                   m_path.c_str(), std::strerror(errno));
                }
            }
        }


        /**
         * Get the next line of the ifstream.
         * @param[in] line  Buffer into which the line will be stored. It is erased before starting.
         * @return          A temporary reference of the istream. Since its operator() is evaluating
         *                  istream.fail(), this is meant to be used in a @c while condition. If
         *                  evaluates to false, it means the line could not be read, either because
         *                  the stream is @c bad() or because it passed the end of line.
         *
         * @example Read a file line per line.
         * @code
         * TextFile file("some_file.txt");
         * std::string line;
         * while(file.getLine(line)) {
         *     // do something with the line
         * }
         * if (file.bad())
         *     // error while reading the file
         * else
         *     // everything is OK, the end of the file was reached without error.
         * @endcode
         */
        inline std::istream& getLine(std::string& line) { return std::getline(*m_fstream, line); }


        /** Whether or not the badbit of @a m_fstream is turned on. */
        [[nodiscard]] inline bool bad() const noexcept { return m_fstream->bad(); }


        /** Whether or not the eof of @a m_fstream is turned on. */
        [[nodiscard]] inline bool eof() const noexcept { return m_fstream->eof(); }


        /** Whether or not the failbit of @a m_fstream is turned on. */
        [[nodiscard]] inline bool fail() const noexcept { return m_fstream->fail(); }


        /** Whether or not the failbit of @a m_fstream is turned off. */
        [[nodiscard]] explicit operator bool() const noexcept { return !m_fstream->fail(); }


        /** Whether or not the failbit of @a m_fstream is turned on. */
        [[nodiscard]] bool operator!() const noexcept { return m_fstream->fail(); }


        /** Closes the stream and deletes the content of @a m_path. */
        inline void remove() {
            close();
            ::Noa::OS::remove(m_path);
        }


        /**
         * Close the stream, rename the file at @a m_path and set @a m_path to @a to.
         * The file is not reopened.
         * @param[in] to    Desired name or location.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline void rename(T&& to) {
            close();
            ::Noa::OS::rename(m_path, to);
            m_path = std::forward<T>(to);
        }


        /**
         * Rename the file on the fly. The file is renamed and the stream is reopened.
         * @param[in] to        Desired name or location.
         * @param[in] mode      Any of the open mode (in|out|trunc|app|ate|binary).
         * @param[in] long_wait Wait for the file to exist for 10*30s, otherwise wait for 5*10ms.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline void rename(T&& to, std::ios_base::openmode mode, bool long_wait = false) {
            close();
            ::Noa::OS::rename(m_path, to);
            m_path = std::forward<T>(to);
            reopen(mode, long_wait);
        }


        /**
         * Load the entire file into a @c std::string.
         * @return  String containing the whole content of @a m_path.
         * @note    The ifstream is rewound before reading.
         */
        std::string toString();


        /**
         * Get a reference of @a m_fstream.
         * @warning This should be safe and the class should be able to handle whatever changes are
         *          done outside the class. One thing that is possible but not really meant to be
         *          changed is the exception level of the stream. If you activate some exceptions,
         *          make sure you know what you are doing, specially when activating @c eofbit.
         *
         * @note @c std::fstream doesn't throw exceptions by default but keeps track of a few flags
         *          reporting on the situation. Here is more information on how to check for them.
         *          - @c goodbit: its value, 0, indicates the absence of any error flag. If 1,
         *                        all input and output operations have no effect.
         *                        See @c std::fstream::good().
         *          - @c eofbit:  Is set when there an attempt to read past the end of an input sequence.
         *                        When reaching the last character, the stream is still in good state,
         *                        but any subsequent extraction will be considered an attempt to read
         *                        past the end - `eofbit` is set to 1. The other situation is when
         *                        the reading doesn't happen character-wise and we reach the eof.
         *                        See @c std::fstream::eof().
         *          - @c failbit: Is set when a read or write operation fails. For example, in the
         *                        first example of `eofbit`, `failbit` is also set since we fail to
         *                        read, but in the second example it is not set since the int or string
         *                        was extracted. `failbit` is also set if the file couldn't be open.
         *                        See @c std::fstream::fail() or @c std::fstream::operator!().
         *          - @c badbit:  Is set when a problem with the underlying stream buffer happens. This
         *                        can happen from memory shortage or because the underlying stream
         *                        buffer throws an exception.
         *                        See @c std::fstream::bad().
         */
        [[nodiscard]] inline std::fstream& fstream() noexcept { return *m_fstream; }


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
         */
        template<typename S,
                typename = std::enable_if_t<std::is_same_v<S, std::ifstream> ||
                                            std::is_same_v<S, std::ofstream> ||
                                            std::is_same_v<S, std::fstream>>>
        static void open(const std::filesystem::path& path,
                         S& fs,
                         std::ios_base::openmode mode,
                         bool long_wait = false) {
            if (!close(fs)) {
                NOA_CORE_ERROR("\"{}\": error while closing the file. {}",
                               path.c_str(), std::strerror(errno));
            }
            size_t iterations = long_wait ? 10 : 5;
            size_t time_to_wait = long_wait ? 3000 : 10;

            for (size_t it{0}; it < iterations; ++it) {
                if constexpr (!std::is_same_v<S, std::ifstream>) {
                    // If only reading mode, the file should be there - no need to create it.
                    if (mode & std::ios::out && path.has_parent_path())
                        std::filesystem::create_directories(path.parent_path());
                }
                fs.open(path.c_str(), mode);
                if (fs)
                    return;
                std::this_thread::sleep_for(std::chrono::milliseconds(time_to_wait));
            }
            NOA_CORE_ERROR("\"{}\": error while opening the file. {}",
                           path.c_str(), std::strerror(errno));
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
    };
}


