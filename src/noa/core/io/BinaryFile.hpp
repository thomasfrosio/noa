#pragma once

#include <filesystem>
#include <fstream>
#include "noa/core/Traits.hpp"
#include "noa/core/Error.hpp"
#include "noa/core/io/OS.hpp"
#include "noa/core/io/IO.hpp"
#include "noa/core/utils/Strings.hpp"

namespace noa::io {
    struct BinaryFileOptions {
        /// Whether the file should be deleted after closing.
        bool close_delete{};

        /// Whether the file should be placed in the current working directory.
        /// Otherwise, temporary_directory() is used.
        bool default_to_cwd{};
    };

    /// Binary file. This is also meant to be used as a temporary file.
    ///     - the data is not formatted on reads and writes.
    ///     - the filename and path can be generated automatically.
    ///     - the file can be automatically deleted after closing.
    class BinaryFile {
    public:
        /// Generate an unused filename.
        static auto generate_filename(bool use_cwd = false) -> Path {
            Path out;
            if (use_cwd)
                out = fs::current_path() / "";
            else
                out = temporary_directory() / "";
            while (true) {
                const i32 tag = 10000 + std::rand() / (99999 / (99999 - 10000 + 1) + 1); // 5 random digits
                out.replace_filename(fmt::format("tmp_{}.bin", tag));
                if (not is_file(out))
                    break;
            }
            return out;
        }

    public:
        /// Creates an empty instance. Use open() to open a file.
        BinaryFile() = default;

        /// Stores the path and opens the file. \see open() for more details.
        BinaryFile(Path path, Open open_mode, BinaryFileOptions options = {}) :
            m_path(std::move(path)),
            m_delete(options.close_delete)
        {
            open_(open_mode);
        }

        /// Generates a temporary filename and opens the file.
        explicit BinaryFile(Open open_mode, BinaryFileOptions options = {}) :
            m_path(generate_filename(options.default_to_cwd)),
            m_delete(options.close_delete)
        {
            open_(open_mode);
        }

        /// Close the currently opened file (if any) and opens a new file.
        /// \param path         Filename of the file to open.
        /// \param open_mode    Should be one or a combination of the following ("binary" is always considered on):
        ///                     \c read:                    File should exists.
        ///                     \c read-write:              File should exists.     Backup copy.
        ///                     \c write, write-truncate:   Overwrite the file.     Backup move.
        ///                     \c read-write-truncate:     Overwrite the file.     Backup move.
        /// \param options      Only .close_delete is used.
        ///
        /// \throws Exception   If any of the following cases:
        ///         - If the file does not exist and \p open_mode is set to read or read-write.
        ///         - If the permissions do not match the \p open_mode.
        ///         - If failed to close the file before starting (if any).
        ///         - If an underlying OS error was raised.
        void open(Path path, Open open_mode, BinaryFileOptions options = {}) {
            close();
            m_path = std::move(path);
            m_delete = options.close_delete;
            open_(open_mode);
        }

        /// Close the currently opened file (if any) and opens a new file with a automatically generated path/name.
        void open(Open open_mode, BinaryFileOptions options = {}) {
            close();
            m_path = generate_filename(options.default_to_cwd);
            m_delete = options.close_delete;
            open_(open_mode);
        }

        template<typename T>
        void read(SpanContiguous<T, 1> output, i64 offset = 0) {
            m_fstream.seekg(offset);
            if (m_fstream.fail())
                panic("File: {}. Could not seek to the desired offset ({} bytes)", m_path, offset);
            const i64 bytes = output.n_elements() * static_cast<i64>(sizeof(T));
            m_fstream.read(reinterpret_cast<char*>(output.get()), bytes);
            if (m_fstream.fail())
                panic("File stream error. Failed while reading {} bytes from {}", bytes, m_path);
        }

        template<typename T>
        void write(SpanContiguous<const T, 1> input, i64 offset = 0) {
            m_fstream.seekp(offset);
            if (m_fstream.fail())
                panic("File: {}. Could not seek to the desired offset ({} bytes)", m_path, offset);
            const i64 bytes = input.n_elements() * static_cast<i64>(sizeof(T));
            m_fstream.write(reinterpret_cast<const char*>(input.get()), bytes);
            if (m_fstream.fail())
                panic("File stream error. Failed while writing {} bytes from {}", bytes, m_path);
        }

        void close() {
            if (not is_open())
                return;

            m_fstream.close();
            if (m_fstream.fail())
                panic("File: {}. File stream error. Could not close the file", m_path);
            if (m_delete)
                noa::io::remove(m_path);
        }

        bool exists() {
            m_fstream.flush();
            return noa::io::is_file(m_path);
        }

        i64 size() {
            m_fstream.flush();
            return noa::io::file_size(m_path);
        }

        [[nodiscard]] std::fstream& fstream() noexcept { return m_fstream; }
        [[nodiscard]] const fs::path& path() const noexcept { return m_path; }
        [[nodiscard]] bool is_open() const noexcept { return m_fstream.is_open(); }

        ~BinaryFile() noexcept(false) {
            try {
                close();
            } catch (...) {
                if (!std::uncaught_exceptions()) {
                    std::rethrow_exception(std::current_exception());
                }
            }
        }

    private:
        void open_(Open open_mode, const std::source_location& location = std::source_location::current());

    private:
        std::fstream m_fstream{};
        Path m_path{};
        bool m_delete{false};
    };
}
