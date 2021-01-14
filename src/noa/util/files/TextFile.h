/**
 * @file TextFile.h
 * @brief Text file class.
 * @author Thomas - ffyr2w
 * @date 9 Oct 2020
 */
#pragma once

#include <cstddef>
#include <ios>          // std::streamsize
#include <fstream>
#include <filesystem>
#include <memory>
#include <utility>
#include <type_traits>
#include <string>

#include "noa/API.h"
#include "noa/util/Constants.h"
#include "noa/util/Flag.h"
#include "noa/util/OS.h"
#include "noa/util/string/Format.h"

namespace Noa {
    /** Base class for all text files. It is not copyable, but it is movable. */
    template<typename Stream = std::fstream,
             typename = std::enable_if_t<std::is_same_v<Stream, std::ifstream> ||
                                         std::is_same_v<Stream, std::ofstream> ||
                                         std::is_same_v<Stream, std::fstream>>>
    class NOA_API TextFile {
    private:
        using openmode_t = std::ios_base::openmode;
        fs::path m_path{};
        Stream m_fstream{};
        Flag<Errno> m_state{};

    public:
        /** Initializes the underlying file stream. */
        explicit TextFile() = default;

        /** Initializes the path and underlying file stream. The file isn't opened. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        explicit TextFile(T&& path) : m_path(std::forward<T>(path)) {}

        /** Sets and opens the associated file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        explicit TextFile(T&& path, openmode_t mode, bool long_wait = false) : m_path(std::forward<T>(path)) {
            m_state = open(mode, long_wait);
        }

        /** Resets the path and opens the associated file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        inline Flag<Errno> open(T&& path, openmode_t mode, bool long_wait = false) {
            m_path = std::forward<T>(path);
            return open(mode, long_wait);
        }

        /** Writes a string(_view) to the file. */
        template<typename T, typename = std::enable_if_t<Traits::is_string_v<T>>>
        inline void write(T&& string) {
            m_fstream.write(string.data(), static_cast<std::streamsize>(string.size()));
        }

        /**
         * (Formats and) writes a string(_view).
         * @tparam Args     Anything accepted by @c fmt::format()
         * @param[in] args  C-string and|or variable(s) used to compute the formatted string.
         */
        template<typename... Args>
        void write(Args&& ... args) { write(fmt::format(std::forward<Args>(args)...)); }

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
        inline std::istream& getLine(std::string& line) { return std::getline(m_fstream, line); }

        /**
         * Loads the entire file into a string.
         * @return  String containing the whole content of the file.
         * @note    The ifstream is rewound before reading.
         */
        std::string toString() {
            std::string buffer;

            m_fstream.seekg(0, std::ios::end);
            std::streampos size = m_fstream.tellg();
            if (size < 1) {
                if (size != 0)
                    m_state.update(Errno::fail_read);
                return buffer;
            }
            try {
                buffer.resize(static_cast<size_t>(size));
            } catch (std::length_error& e) {
                m_state.update(Errno::out_of_memory);
            }
            if (m_state)
                return buffer;

            m_fstream.seekg(0);
            m_fstream.read(buffer.data(), size);
            if (m_fstream.fail())
                m_state.update(Errno::fail_read);
            return buffer;
        }

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
        [[nodiscard]] inline Stream& fstream() noexcept { return m_fstream; }

        /** Whether or not @a m_path points to a regular file or a symlink pointing to a regular file. */
        inline bool exists() noexcept { return !m_state && OS::existsFile(m_path, m_state); }

        /** Gets the size (in bytes) of the file at @a m_path. Symlinks are followed. */
        inline size_t size() noexcept { return !m_state ? OS::size(m_path, m_state) : 0U; }

        [[nodiscard]] inline const fs::path& path() const noexcept { return m_path; }

        [[nodiscard]] inline bool bad() const noexcept { return m_fstream.bad(); }
        [[nodiscard]] inline bool eof() const noexcept { return m_fstream.eof(); }
        [[nodiscard]] inline bool fail() const noexcept { return m_fstream.fail(); }
        [[nodiscard]] inline bool isOpen() const noexcept { return m_fstream.is_open(); }

        [[nodiscard]] inline Flag<Errno> state() const { return m_state; }

        inline void clear() {
            m_state = Errno::good;
            m_fstream.clear();
        }

        /** Whether or not the instance is in a "good" state. */
        [[nodiscard]] inline explicit operator bool() const noexcept {
            return !m_state && !m_fstream.fail();
        }

        /**
         * Opens the file.
         * @param[in] mode      Any of the @c openmode_t. See below.
         * @param[in] long_wait Wait for the file to exist for 10*3s, otherwise wait for 5*10ms.
         * @return              @c Errno::fail_close, if failed to close the file before starting.
         *                      @c Errno::fail_open, if failed to open the file.
         *                      @c Errno::fail_os, if an underlying OS error was raised.
         *                      @c Errno::good, otherwise.
         *
         * @note @a mode should have at least one of the following bit combination on:
         *  - 1) @c in:                 Read.         File should exists.
         *  - 2) @c in|out:             Read & Write. File should exists.    Backup copy.
         *  - 3) @c out, out|trunc:     Write.        Overwrite the file.    Backup move.
         *  - 4) @c in|out|trunc:       Read & Write. Overwrite the file.    Backup move.
         *  - 5) @c app, out|app:       Write.        Append or create file. Backup copy. Seek to eof before each write.
         *  - 6) @c in|app, in|out|app: Read & Write. Append or create file. Backup copy. Seek to eof before each write.
         *
         * @note Additionally, @c ate and/or @c binary can be turned on:
         *  - @c ate:    @c ofstream and @c ifstream seek the end of the file after opening.
         *  - @c binary: Disable text conversions.
         *
         * @warning As shown above, specifying @c trunc and @c app is undefined.
         */
        Flag<Errno> open(openmode_t mode, bool long_wait = false) {
            if (close())
                return m_state;

            int iterations = long_wait ? 10 : 5;
            size_t time_to_wait = long_wait ? 3000 : 10;

            if constexpr (!std::is_same_v<Stream, std::ifstream>) {
                if (mode & std::ios::out || mode & std::ios::app) /* all except case 1 */ {
                    bool exists = OS::existsFile(m_path, m_state);
                    bool overwrite = mode & std::ios::trunc || !(mode & std::ios::in); // case 3|4
                    if (exists)
                        m_state.update(OS::backup(m_path, overwrite));
                    else if (overwrite || mode & std::ios::app) /* all except case 2 */
                        m_state.update(OS::mkdir(m_path.parent_path()));
                    if (m_state)
                        return m_state;
                }
            }
            for (int it{0}; it < iterations; ++it) {
                m_fstream.open(m_path, mode);
                if (m_fstream.is_open())
                    return m_state;
                std::this_thread::sleep_for(std::chrono::milliseconds(time_to_wait));
            }
            m_state = Errno::fail_open; // m_state is necessarily Errno::good at this stage.
            return m_state;
        }

        /** Closes the stream if it is opened, otherwise don't do anything. */
        inline Flag<Errno> close() {
            if (m_fstream.is_open()) {
                m_fstream.close();
                if (m_fstream.fail())
                    m_state.update(Errno::fail_close);
            }
            return m_state;
        }
    };
}
