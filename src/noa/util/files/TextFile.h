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
#include <thread>
#include <string>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Types.h"
#include "noa/util/OS.h"
#include "noa/util/string/Format.h"

namespace Noa {
    /** Base class for all text files. It is not copyable, but it is movable. */
    template<typename Stream = std::fstream,
             typename = std::enable_if_t<std::is_same_v<Stream, std::ifstream> ||
                                         std::is_same_v<Stream, std::ofstream> ||
                                         std::is_same_v<Stream, std::fstream>>>
    class TextFile {
    private:
        fs::path m_path{};
        Stream m_fstream{};

    public:
        /** Initializes the underlying file stream. */
        explicit TextFile() = default;

        /** Initializes the path and underlying file stream. The file isn't opened. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        explicit TextFile(const fs::path& path) : m_path(std::forward<T>(path)) {}

        /** Sets and opens the associated file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        explicit TextFile(T&& path, openmode_t mode, bool long_wait = false) : m_path(std::forward<T>(path)) {
            open(mode, long_wait);
        }

        /** Resets the path and opens the associated file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        void open(T&& path, openmode_t mode, bool long_wait = false) {
            m_path = std::forward<T>(path);
            open(mode, long_wait);
        }

        /** Writes a string(_view) to the file. */
        template<typename T, typename = std::enable_if_t<Traits::is_string_v < T>>>
        inline void write(T&& string) {
            m_fstream.write(string.data(), static_cast<std::streamsize>(string.size()));
            if (m_fstream.fail()) {
                if (m_fstream.is_open())
                    NOA_THROW("File: \"{}\". File stream error while writing", m_path.filename());
                NOA_THROW("File: \"{}\". File stream error. Tyring to write to closed file", m_path.filename());
            }
        }

        /**
         * (Formats and) writes a string(_view).
         * @tparam Args     Anything accepted by @c fmt::format()
         * @param[in] args  C-string and|or variable(s) used to compute the formatted string.
         */
        template<typename... Args>
        void write(Args&& ... args) { write(String::format(std::forward<Args>(args)...)); }

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
            if (!size)
                return buffer;
            else if (size == -1)
                NOA_THROW("File: \"{}\". File stream error. Could not get the input position indicator",
                          m_path.filename());

            try {
                buffer.resize(static_cast<size_t>(size));
            } catch (std::length_error& e) {
                NOA_THROW("File: \"{}\". Passed the maximum permitted size while try to load file. Got {} bytes",
                          m_path.filename(), size);
            }

            m_fstream.seekg(0);
            m_fstream.read(buffer.data(), size);
            if (m_fstream.fail())
                NOA_THROW("File: \"{}\". File stream error. Could not read the entire file", m_path.filename());
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
        inline bool exists() { return OS::existsFile(m_path); }

        /** Gets the size (in bytes) of the file at @a m_path. Symlinks are followed. */
        inline size_t size() { return OS::size(m_path); }

        [[nodiscard]] inline const fs::path& path() const noexcept { return m_path; }
        [[nodiscard]] inline bool bad() const noexcept { return m_fstream.bad(); }
        [[nodiscard]] inline bool eof() const noexcept { return m_fstream.eof(); }
        [[nodiscard]] inline bool fail() const noexcept { return m_fstream.fail(); }
        [[nodiscard]] inline bool isOpen() const noexcept { return m_fstream.is_open(); }

        inline void clear() { m_fstream.clear(); }

        /** Whether or not the instance is in a "good" state. */
        [[nodiscard]] inline explicit operator bool() const noexcept { return !m_fstream.fail(); }

        /**
         * Opens the file.
         * @param mode          Any of the @c openmode_t. See below.
         * @param long_wait     Wait for the file to exist for 10*3s, otherwise wait for 5*10ms.
         * @throw Exception     If failed to close the file before starting.
         *                      If failed to open the file.
         *                      If an underlying OS error was raised.
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
        void open(openmode_t mode, bool long_wait = false) {
            close();

            int iterations = long_wait ? 10 : 5;
            size_t time_to_wait = long_wait ? 3000 : 10;

            if constexpr (!std::is_same_v<Stream, std::ifstream>) {
                if (mode & std::ios::out || mode & std::ios::app) /* all except case 1 */ {
                    bool overwrite = mode & std::ios::trunc || !(mode & std::ios::in); // case 3|4
                    try {
                        bool exists = OS::existsFile(m_path);
                        if (exists)
                            OS::backup(m_path, overwrite);
                        else if (overwrite || mode & std::ios::app) /* all except case 2 */
                            OS::mkdir(m_path.parent_path());
                    } catch (...) {
                        NOA_THROW("File: \"{}\". OS failure", m_path.filename());
                    }
                }
            }
            for (int it{0}; it < iterations; ++it) {
                m_fstream.open(m_path, mode);
                if (m_fstream.is_open())
                    return;
                std::this_thread::sleep_for(std::chrono::milliseconds(time_to_wait));
            }
            NOA_THROW("File: \"{}\". Failed to open, even after {} iterations interspaced by {} milliseconds",
                      m_path.filename(), iterations, time_to_wait);
        }

        /** Closes the stream if it is opened, otherwise don't do anything. */
        inline void close() {
            if (m_fstream.is_open()) {
                m_fstream.close();
                if (m_fstream.fail())
                    NOA_THROW("File: \"{}\". File stream error", m_path.filename());
            }
        }
    };
}
