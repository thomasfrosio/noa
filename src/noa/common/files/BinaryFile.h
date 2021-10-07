/// \file noa/common/files/BinaryFile.h
/// \brief BinaryFile class.
/// \author Thomas - ffyr2w
/// \date 18 Aug 2020

#pragma once

#include <type_traits>
#include <filesystem>
#include <utility>

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/common/string/Format.h"
#include "noa/common/OS.h"

namespace noa {
    /// Binary file. This is also meant to be used as a temporary file.
    ///     - the data is not formatted on reads and writes.
    ///     - the filename and path can be generated automatically.
    ///     - the file can be automatically deleted after closing.
    class BinaryFile {
    private:
        std::fstream m_fstream{};
        path_t m_path{};
        bool m_delete{false};

    public:
        /// Creates an empty instance. Use open() to open a file.
        BinaryFile() = default;

        /// Stores the path. Use open() to open the file.
        NOA_HOST explicit BinaryFile(path_t path) : m_path(std::move(path)) {}

        /// Generates a temporary filename and opens the file.
        NOA_HOST explicit BinaryFile(uint open_mode, bool close_delete = false)
                : m_path(generateFilename_()), m_delete(close_delete) {
            open_(open_mode);
        }

        /// Stores the path and opens the file. \see open() for more details.
        NOA_HOST BinaryFile(path_t path, uint open_mode, bool close_delete = false)
                : m_path(std::move(path)), m_delete(close_delete) {
            open_(open_mode);
        }

        /// (Re)Opens the file.
        /// \param path         Filename of the file to open.
        /// \param open_mode    io::OpenMode bit mask. Should be one or a combination of the following:
        ///                     \c READ:                File should exists.
        ///                     \c READ|WRITE:          File should exists.     Backup copy.
        ///                     \c WRITE, WRITE|TRUNC:  Overwrite the file.     Backup move.
        ///                     \c READ|WRITE|TRUNC:    Overwrite the file.     Backup move.
        /// \param close_delete Whether or not the file should be deleted after closing.
        ///
        /// \throws Exception   If any of the following cases:
        ///         - If the file does not exist and \p open_mode is \c io::READ or \c io::READ|io::WRITE.
        ///         - If the permissions do not match the \p open_mode.
        ///         - If failed to close the file before starting (if any).
        ///         - If an underlying OS error was raised.
        ///
        /// \note   Internally, the \c io::BINARY flag is always considered on.
        NOA_HOST void open(path_t path, uint open_mode, bool close_delete = false) {
            m_path = std::move(path);
            m_delete = close_delete;
            open_(open_mode);
        }

        /// (Re)Opens the file.
        /// \note If the path is not set, a temporary filename is created.
        ///       In this case, \p open_mode should have the io::WRITE flag turned on.
        NOA_HOST void open(uint open_mode, bool close_delete = false) {
            if (m_path.empty())
                m_path = generateFilename_();
            m_delete = close_delete;
            open_(open_mode);
        }

        template<typename T>
        NOA_HOST void read(T* output, size_t offset, size_t elements) {
            m_fstream.seekg(static_cast<std::streamoff>(offset));
            if (m_fstream.fail())
                NOA_THROW("File: {}. Could not seek to the desired offset ({} bytes)", m_path, offset);
            auto bytes = static_cast<std::streamsize>(elements * sizeof(T));
            m_fstream.read(reinterpret_cast<char*>(output), bytes);
            if (m_fstream.fail())
                NOA_THROW("File stream error. Failed while reading {} bytes from {}", bytes, m_path);
        }

        template<typename T>
        NOA_HOST void write(T* output, size_t offset, size_t elements) {
            m_fstream.seekp(static_cast<std::streamoff>(offset));
            if (m_fstream.fail())
                NOA_THROW("File: {}. Could not seek to the desired offset ({} bytes)", m_path, offset);
            auto bytes = static_cast<std::streamsize>(elements * sizeof(T));
            m_fstream.write(reinterpret_cast<char*>(output), bytes);
            if (m_fstream.fail())
                NOA_THROW("File stream error. Failed while writing {} bytes from {}", bytes, m_path);
        }

        void close() {
            m_fstream.close();
            if (m_fstream.fail())
                NOA_THROW("File: {}. File stream error. Could not close the file", m_path);
            if (m_delete)
                os::remove(m_path);
        }

        NOA_HOST void flush() { m_fstream.flush(); }
        NOA_HOST bool exists() { return os::existsFile(m_path); }
        NOA_HOST size_t size() { return os::size(m_path); }
        NOA_HOST void clear() { m_fstream.clear(); }

        [[nodiscard]] NOA_HOST std::fstream& fstream() noexcept { return m_fstream; }
        [[nodiscard]] NOA_HOST const fs::path& path() const noexcept { return m_path; }
        [[nodiscard]] NOA_HOST bool bad() const noexcept { return m_fstream.bad(); }
        [[nodiscard]] NOA_HOST bool eof() const noexcept { return m_fstream.eof(); }
        [[nodiscard]] NOA_HOST bool fail() const noexcept { return m_fstream.fail(); }
        [[nodiscard]] NOA_HOST bool isOpen() const noexcept { return m_fstream.is_open(); }
        [[nodiscard]] NOA_HOST explicit operator bool() const noexcept { return !m_fstream.fail(); }

        NOA_HOST ~BinaryFile() { close(); }

    private:
        // Generate an unused filename.
        static NOA_HOST path_t generateFilename_() {
            path_t out(os::tempDirectory() / "");
            while (true) {
                int tag = 10000 + std::rand() / (99999 / (99999 - 10000 + 1) + 1); // 5 random digits
                out.replace_filename(string::format("tmp_{}.bin", tag));
                if (!os::existsFile(out))
                    break;
            }
            return out;
        }

        void open_(uint open_mode);
    };
}
