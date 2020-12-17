/**
 * @file TextFile.h
 * @brief Text file class.
 * @author Thomas - ffyr2w
 * @date 9 Oct 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/files/File.h"


namespace Noa {
    class NOA_API TextFile : public File {
    public:
        /** Initializes the underlying file stream. */
        explicit TextFile() : File() {}


        /** Initializes the path and underlying file stream. The file isn't opened. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit TextFile(T&& path) : File(std::forward<T>(path)) {}


        /** Sets and opens the associated file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit TextFile(T&& path, std::ios_base::openmode mode, bool long_wait = false)
                : File(std::forward<T>(path)) {
            open(mode, long_wait);
        }


        /** Resets the path and opens the associated file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline errno_t open(T&& path, std::ios_base::openmode mode, bool long_wait = false) {
            m_path = std::forward<T>(path);
            return open(mode, long_wait);
        }


        /** Closes the stream and reopens it with the current path. */
        inline errno_t open(std::ios_base::openmode mode, bool long_wait = false) {
            return File::open(m_path, *m_fstream, mode, long_wait);
        }


        /** Closes the stream if it is opened, otherwise don't do anything. */
        inline errno_t close() { return File::close(*m_fstream); }


        /**
         * (Formats and) writes a string.
         * @tparam Args     Anything accepted by @c fmt::format()
         * @param[in] args  C-string and|or variable(s) used to compute the formatted string.
         *
         * @note            This function depends on the current ofstream position. If std::ios::app,
         *                  the position is set to the end of the file at every call.
         */
        template<typename... Args>
        void write(Args&& ... args) {
            std::string message = fmt::format(std::forward<Args>(args)...);
            m_fstream->write(message.c_str(), static_cast<std::streamsize>(message.size()));
        }


        /**
         * Gets the next line of the ifstream.
         * @param[in] line  Buffer into which the line will be stored. It is erased before starting.
         * @return          A temporary reference of the istream. With its operator() evaluating to
         *                  istream.fail(), this is meant to be used in a @c while condition. If
         *                  evaluates to false, it means the line could not be read, either because
         *                  the stream is @c bad() or because it reached the end of line.
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


        /** Closes the stream and deletes the file. */
        inline errno_t remove() {
            if (close() || OS::remove(m_path))
                return Errno::fail;
            return Errno::good;
        }


        /** Closes the stream and renames the file. The file is not reopened. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline errno_t rename(T&& to) {
            if (close() || OS::move(m_path, to))
                return Errno::fail;
            m_path = std::forward<T>(to);
            return Errno::good;
        }


        /** Closes, renames and reopens the file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline errno_t rename(T&& to, std::ios_base::openmode mode, bool long_wait = false) {
            if (rename(std::forward<T>(to)))
                return Errno::fail;
            return open(mode, long_wait);
        }


        /**
         * Loads the entire file into a @c std::string.
         * @return  String containing the whole content of @a m_path.
         * @note    The ifstream is rewound before reading.
         */
        std::string toString(errno_t& err);


        /**
         * Gets a reference of the underlying file stream.
         * @warning This should be safe and the class should be able to handle whatever changes are
         *          done outside the class. One thing that is possible but not really meant to be
         *          changed is the exception level of the stream. If you activate some exceptions,
         *          make sure you know what you are doing, specially when activating @c eofbit.
         *
         * @note @c std::fstream doesn't throw exceptions by default but keeps track of a few flags
         *       reporting on the situation. Here is more information on their meaning.
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
    };
}
