#pragma once

#include <ios> // std::streamsize
#include <fstream>
#include <type_traits>
#include <thread>
#include <string>

#include "noa/core/Error.hpp"
#include "noa/core/io/IO.hpp"
#include "noa/core/Traits.hpp"

namespace noa::io {
    /// Read from and write to text files.
    /// It is not copyable, but it is movable.
    template<typename Stream = std::fstream>
    requires nt::is_any_v<Stream, std::ifstream, std::ofstream, std::fstream>
    class TextFile {
    public:
        /// Creates an empty file. Use .open() to open a new file.
        TextFile() = default;

        /// Opens the file. See .open() for more details.
        TextFile(Path path, Open mode) : m_path(std::move(path)) {
            open_(mode);
        }

        /// (Re)Opens the file.
        /// \param path     Path of the file to open.
        /// \param mode     Open mode. All opening modes are supported.
        /// \throws If any of the following cases:
        ///         - If failed to close the file before starting.
        ///         - If failed to open the file.
        ///         - If an underlying OS error was raised.
        void open(Path path, Open mode) {
            m_path = std::move(path);
            open_(mode);
        }

        /// Closes the file if it is open, otherwise do nothing.
        void close() {
            if (m_fstream.is_open()) {
                m_fstream.close();
                if (m_fstream.fail() and not m_fstream.eof())
                    panic("File: {}. File stream error", m_path);
            }
        }

        /// Writes a string at the end of the file.
        void write(std::string_view string) {
            // m_fstream.seekp(0, std::ios_base::end);
            m_fstream.write(string.data(), static_cast<std::streamsize>(string.size()));
            check(not m_fstream.fail(), "Could not write to file {}", m_path);
        }

        /// Gets the next line of the file.
        /// \param[out] line Buffer into which the line will be stored. It is erased before starting.
        /// \return Whether the line was successfully read. If not, the stream failed or reached
        ///         the end of the file. Use bad() to check if an error occurred while reading the file.
        ///
        /// \example
        /// \code
        /// // Read a file line per line.
        /// TextFile file("some_file.txt");
        /// std::string line;
        /// while(file.next_line(line)) {
        ///     // do something with the line
        /// }
        /// if (file.bad())
        ///     // error while reading the file
        /// // file.eof() == true; everything is OK, the end of the file was reached without error.
        /// \endcode
        auto next_line(std::string& line) -> bool {
            return static_cast<bool>(std::getline(m_fstream, line));
        }

        /// Gets the next line of the file. If an error occurs, throw an exception.
        /// \param[out] line Buffer into which the line will be stored. It is erased before starting.
        /// \example
        /// \code
        /// // Read a file line per line.
        /// TextFile file("some_file.txt");
        /// std::string line;
        /// while(file.next_line_or_throw(line)) {
        ///     // do something with the line
        /// }
        /// // file.eof() == true; end of file reached successfully
        /// \endcode
        auto next_line_or_throw(std::string& line) -> bool {
            bool success = next_line(line);
            if (not success and not this->eof())
                panic("File: {}. Failed to read a line", m_path);
            return success;
        }

        /// Reads the entire file.
        auto read_all() -> std::string {
            std::string buffer;

            // FIXME use file_size(m_path) instead?
            m_fstream.seekg(0, std::ios::end);
            const std::streampos size = m_fstream.tellg();
            check(size >= 0, "File: {}. File stream error. Could not get the input position indicator", m_path);
            if (not size)
                return buffer;

            try {
                buffer.resize(static_cast<size_t>(size));
            } catch (std::length_error& e) {
                panic("File: {}. Passed the maximum permitted size while try to load file. Got {} bytes",
                      m_path, static_cast<std::streamoff>(size));
            }

            m_fstream.seekg(0);
            m_fstream.read(buffer.data(), size);
            check(not m_fstream.fail(), "File: {}. File stream error. Could not read the entire file", m_path);
            return buffer;
        }

        [[nodiscard]] auto ssize() -> i64 {
            m_fstream.flush();
            return noa::io::file_size(m_path);
        }
        [[nodiscard]] auto size() -> u64 {
            return static_cast<u64>(ssize());
        }

        [[nodiscard]] auto path() const noexcept -> const Path& { return m_path; }
        [[nodiscard]] auto fstream() noexcept -> Stream& { return m_fstream; }
        [[nodiscard]] auto bad() const noexcept -> bool { return m_fstream.bad(); }
        [[nodiscard]] auto eof() const noexcept -> bool { return m_fstream.eof(); }
        [[nodiscard]] auto fail() const noexcept -> bool { return m_fstream.fail(); }
        [[nodiscard]] auto is_open() const noexcept -> bool { return m_fstream.is_open(); }
        void clear_flags() { m_fstream.clear(); }

    private:
        void open_(Open mode) {
            close();
            check(mode.is_valid(), "Invalid open mode");

            if constexpr (not std::is_same_v<Stream, std::ifstream>) {
                if (mode.write) {
                    const bool overwrite = mode.truncate or not (mode.read or mode.append); // case 3 and 5
                    const bool exists = is_file(m_path);
                    try {
                        if (exists and mode.backup)
                            backup(m_path, overwrite);
                        else if (overwrite or mode.append)
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

            for (i32 it{}; it < 3; ++it) {
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
    inline auto read_text(const Path& path) -> std::string {
        InputTextFile text_file(path, Open{.read = true});
        return text_file.read_all();
    }

    /// Saves the entire text file.
    inline void write_text(std::string_view string, const Path& path) {
        OutputTextFile text_file(path, Open{.write = true});
        text_file.write(string);
    }
}

namespace noa {
    using noa::io::read_text;
    using noa::io::write_text;
}
