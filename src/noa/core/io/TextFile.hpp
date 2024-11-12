#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Error.hpp"
#include "noa/core/io/IO.hpp"
#include "noa/core/io/OS.hpp"
#include "noa/core/Traits.hpp"

#ifdef NOA_IS_OFFLINE
#include <ios> // std::streamsize
#include <fstream>
#include <memory>
#include <type_traits>
#include <thread>
#include <string>

namespace noa::io {
    /// Read from and write to text files.
    /// It is not copyable, but it is movable.
    template<typename Stream = std::fstream>
    requires nt::is_any_v<Stream, std::ifstream, std::ofstream, std::fstream>
    class TextFile {
    public:
        /// Creates an empty file. Use .open() to open a new file.
        TextFile() = default;

        /// Opens the associated file. See .open() for more details.
        TextFile(Path path, Open mode) : m_path(std::move(path)) {
            open_(mode);
        }

        /// (Re)Opens the file.
        /// \param filename Path of the file to open.
        /// \param mode Open mode. Should be one of the following combination:
        ///                 1) read                             File should exists.
        ///                 2) read-write                       File should exists.    Backup copy.
        ///                 3) write, write-truncate            Overwrite the file.    Backup move.
        ///                 4) read-write-truncate              Overwrite the file.    Backup move.
        ///                 5) append, write-append             Append or create file. Backup copy. Append at each write.
        ///                 6) read-append, read-write-append   Append or create file. Backup copy. Append at each write.
        ///             Additionally, at_the_end and/or binary can be turned on:
        ///             - at_the_end: the stream go to the end of the file after opening.
        ///             - binary: Disable text conversions.
        ///
        /// \throws Exception   If any of the following cases:
        ///         - If failed to close the file before starting.
        ///         - If failed to open the file.
        ///         - If an underlying OS error was raised.
        void open(Path path, Open mode) {
            m_path = std::move(path);
            open_(mode);
        }

        /// Closes the stream if it is opened, otherwise do nothing.
        void close() {
            if (m_fstream.is_open()) {
                m_fstream.close();
                if (m_fstream.fail() and not m_fstream.eof())
                    panic("File: {}. File stream error", m_path);
            }
        }

        /// Writes a string to the file.
        void write(std::string_view string) {
            m_fstream.write(string.data(), static_cast<std::streamsize>(string.size()));
            if (m_fstream.fail()) {
                if (m_fstream.is_open())
                    panic("File: {}. File stream error while writing", m_path);
                panic("File: {}. File stream error. File is closed file", m_path);
            }
        }

        /// Gets the next line of the file.
        /// \param[out] line Buffer into which the line will be stored. It is erased before starting.
        /// \return Whether the line was successfully read. If not, the stream failed or reached
        ///         the end of the file. Use bad() to check if an error occurred while reading the file.
        ///
        /// \example Read a file line per line.
        /// \code
        /// TextFile file("some_file.txt");
        /// std::string line;
        /// while(file.next_line(line)) {
        ///     // do something with the line
        /// }
        /// if (file.bad())
        ///     // error while reading the file
        /// // file.eof() == true; everything is OK, the end of the file was reached without error.
        /// \endcode
        bool next_line(std::string& line) {
            return static_cast<bool>(std::getline(m_fstream, line));
        }

        /// Gets the next line of the file. If an error occurs, throw an exception.
        /// \param[out] line Buffer into which the line will be stored. It is erased before starting.
        /// \example Read a file line per line.
        /// \code
        /// TextFile file("some_file.txt");
        /// std::string line;
        /// while(file.next_line_or_throw(line)) {
        ///     // do something with the line
        /// }
        /// // file.eof() == true; end of file reached successfully
        /// \endcode
        bool next_line_or_throw(std::string& line) {
            bool success = next_line(line);
            if (not success and not this->eof())
                panic("File: {}. Failed to read a line", m_path);
            return success;
        }

        /// Reads the entire file.
        std::string read_all() {
            std::string buffer;

            // FIXME use file_size(m_path) instead?
            m_fstream.seekg(0, std::ios::end);
            const std::streampos size = m_fstream.tellg();
            if (!size)
                return buffer;
            else if (size < 0)
                panic("File: {}. File stream error. Could not get the input position indicator", m_path);

            try {
                buffer.resize(static_cast<size_t>(size));
            } catch (std::length_error& e) {
                panic("File: {}. Passed the maximum permitted size while try to load file. Got {} bytes",
                      m_path, static_cast<std::streamoff>(size));
            }

            m_fstream.seekg(0);
            m_fstream.read(buffer.data(), size);
            if (m_fstream.fail())
                panic("File: {}. File stream error. Could not read the entire file", m_path);
            return buffer;
        }

        /// Gets a reference of the underlying file stream.
        /// \note   This should be safe and the class should be able to handle whatever changes are
        ///         done outside the class. One thing that is possible but not really meant to be
        ///         changed is the exception level of the stream. If you activate some exceptions,
        ///         make sure you know what you are doing, specially when activating \c eofbit.
        ///
        /// \note \c std::fstream doesn't throw exceptions by default but keeps track of a few flags
        ///       reporting on the situation. Here is more information on their meaning.
        ///          - \c goodbit: its value, 0, indicates the absence of any error flag. If 1,
        ///                        all input and output operations have no effect.
        ///                        See \c std::fstream::good().
        ///          - \c eofbit:  Is set when there an attempt to read past the end of an input sequence.
        ///                        When reaching the last character, the stream is still in good state,
        ///                        but any subsequent extraction will be considered an attempt to read
        ///                        past the end - `eofbit` is set to 1. The other situation is when
        ///                        the reading doesn't happen character-wise and we reach the eof.
        ///                        See \c std::fstream::eof().
        ///          - \c failbit: Is set when a read or write operation fails. For example, in the
        ///                        first example of `eofbit`, `failbit` is also set since we fail to
        ///                        read, but in the second example it is not set since the int or string
        ///                        was extracted. `failbit` is also set if the file couldn't be open.
        ///                        See \c std::fstream::fail() or \c std::fstream::operator!().
        ///          - \c badbit:  Is set when a problem with the underlying stream buffer happens. This
        ///                        can happen from memory shortage or because the underlying stream
        ///                        buffer throws an exception.
        ///                        See \c std::fstream::bad().
        [[nodiscard]] Stream& fstream() noexcept { return m_fstream; }

        /// Gets the size (in bytes) of the file. Symlinks are followed.
        [[nodiscard]] i64 size() {
            m_fstream.flush();
            return noa::io::file_size(m_path);
        }

        [[nodiscard]] const Path& path() const noexcept { return m_path; }
        [[nodiscard]] bool bad() const noexcept { return m_fstream.bad(); }
        [[nodiscard]] bool eof() const noexcept { return m_fstream.eof(); }
        [[nodiscard]] bool fail() const noexcept { return m_fstream.fail(); }
        [[nodiscard]] bool is_open() const noexcept { return m_fstream.is_open(); }
        void clear_flags() { m_fstream.clear(); }

        /// Whether the underlying file stream is in a "good" state.
        [[nodiscard]] explicit operator bool() const noexcept { return not m_fstream.fail(); }

    private:
        void open_(Open mode) {
            close();

            if constexpr (not std::is_same_v<Stream, std::ifstream>) {
                if (mode.write or mode.append) /* all except case 1 */ {
                    const bool overwrite = mode.truncate or not (mode.read or mode.append); // case 3|4
                    const bool exists = is_file(m_path);
                    try {
                        if (exists)
                            backup(m_path, overwrite);
                        else if (overwrite or mode.append) /* all except case 2 */
                            mkdir(m_path.parent_path());
                    } catch (...) {
                        panic("File: {}. {}. Could not open the file because of an OS failure", m_path, mode);
                    }
                }
            }
            const bool read_only = not (mode.write or mode.truncate or mode.append);
            const bool read_write_only = mode.write and mode.read and not (mode.truncate or mode.append);
            if constexpr (std::is_same_v<Stream, std::ifstream>)
                check(read_only, "File: {}. {} is not allowed for read-only TextFile", m_path, mode);

            for (i32 it{}; it < 5; ++it) {
                m_fstream.open(m_path, mode.to_ios_base());
                if (m_fstream.is_open())
                    return;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            m_fstream.clear();

            if (read_only or read_write_only) // case 1|2
                check(is_file(m_path), "File: {}. {}. Trying to open a file that does not exist", m_path, mode);
            panic("File: {}. {}. Failed to open the file. Check the permissions for that directory", m_path, mode);
        }

    private:
        Stream m_fstream{};
        Path m_path{};
    };

    using InputTextFile = TextFile<std::ifstream>;
    using OutputTextFile = TextFile<std::ofstream>;

    /// Reads the entire text file.
    inline std::string read_text(const Path& path) {
        InputTextFile text_file(path, Open{.read=true});
        return text_file.read_all();
    }

    /// Saves the entire text file.
    inline void save_text(std::string_view string, const Path& path) {
        OutputTextFile text_file(path, Open{.write=true});
        text_file.write(string);
    }
}
#endif
