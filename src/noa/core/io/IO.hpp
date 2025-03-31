#pragma once

#include <algorithm> // std::reverse
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <ios>
#include <ostream>

#include "noa/core/Error.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/utils/ClampCast.hpp"
#include "noa/core/utils/Irange.hpp"

namespace noa {
    namespace fs = std::filesystem;
    inline namespace types {
        using Path = fs::path;
    }
}

namespace noa::io {
    /// Controls how files should be opened.
    /// \details There are six opening modes:
    ///     1) read:                  r     Readable.           The file should exist.          No backup.
    ///     2) read-write:            r+    Readable-Writable.  The file should exist.          Backup copy.
    ///     3) read-write-truncate:   w+    Readable-Writable.  Create or overwrite the file.   Backup move.
    ///     4) read-write-append:     a+    Readable-Writable.  Create or append the file.      Backup copy.
    ///     5) write(-truncate):      w     Writable.           Create or overwrite the file.   Backup move.
    ///     6) write-append:          a     Writable.           Create or append the file.      Backup copy.
    struct Open {
        bool read{};
        bool write{};
        bool truncate{};
        bool append{};

        /// Whether a backup of existing files should be created.
        /// Backups are saved (whether by copying or moving the existing file) with the '~' postfix.
        bool backup{true};

        /// Whether the open mode is valid.
        [[nodiscard]] constexpr auto is_valid() const noexcept -> bool {
            if (not read and not write)
                return false;
            if (read and not write and (truncate or append)) // truncate and append are not allowed in read-only mode
                return false;
            if (truncate and append) // mutually exclusive
                return false;
            return true;
        }

        /// Converts to the std::ios_base::openmode flag.
        [[nodiscard]] constexpr auto to_ios_base() const noexcept {
            std::ios_base::openmode mode{};
            if (read)
                mode |= std::ios::in;
            if (write)
                mode |= std::ios::out;
            if (truncate)
                mode |= std::ios::trunc;
            if (append)
                mode |= std::ios::app;
            return mode;
        }

        static auto from_stdio(std::string_view mode, bool backup = true) -> Open {
            mode = noa::string::trim(mode);
            if (mode == "r")
                return {.read = true, .backup = backup};
            if (mode == "r+")
                return {.read = true, .write = true, .backup = backup};
            if (mode == "w+")
                return {.read = true, .write = true, .truncate = true, .backup = backup};
            if (mode == "a+")
                return {.read = true, .write = true, .append = true, .backup = backup};
            if (mode == "w")
                return {.write = true, .backup = backup};
            if (mode == "a")
                return {.write = true, .append = true, .backup = backup};

            panic("invalid stdio open mode: {}", mode);
        }
    };

    /// Whether this code was compiled for big-endian.
    constexpr bool is_big_endian() noexcept {
        return std::endian::native == std::endian::big;
    }

    template<nt::numeric T>
    auto swap_endian(T value) noexcept -> T {
        auto* ptr = reinterpret_cast<std::byte*>(&value);
        std::reverse(ptr, ptr + sizeof(T));
        return value;
    }

    template<size_t SIZEOF>
    auto swap_endian(void* value, i64 n_elements) noexcept {
        auto* ptr = static_cast<std::byte*>(value);
        for (i64 i{}; i < n_elements; ++i) {
            std::reverse(ptr, ptr + SIZEOF);
        }
    }
}

namespace noa::io {
    inline auto operator<<(std::ostream& os, Open mode) -> std::ostream& {
        std::array flags{mode.read, mode.write, mode.truncate, mode.append};
        std::array names{"read", "write", "truncate", "append"};

        bool add{};
        os << "Open{";
        for (auto i: irange(flags.size())) {
            if (flags[i]) {
                if (add)
                    os << '|';
                os << names[i];
                add = true;
            }
        }
        os << '}';
        return os;
    }
}

namespace fmt {
    template<> struct formatter<noa::io::Open> : ostream_formatter {};
}

// Gathers a bunch of OS/filesystem related functions.
// These are just thin wrappers that throw upon failure.
namespace noa::io {
    namespace fs = std::filesystem;

    /// Whether or not the path points to an existing regular file. Symlinks are followed.
    inline bool is_file(const fs::path& path) {
        try {
            return fs::is_regular_file(path);
        } catch (const fs::filesystem_error& e) {
            panic_runtime(e.what());
        } catch (const std::exception& e) {
            panic("File: {}. {}", path, e.what());
        }
    }

    /// Whether or not the file-status describes an existing regular file. Symlinks are followed.
    inline bool is_file(const fs::file_status& file_status) noexcept {
        return fs::is_regular_file(file_status);
    }

    /// Whether or not the path points to an existing directory.
    inline bool is_directory(const fs::path& path) {
        try {
            return fs::is_directory(path);
        } catch (const fs::filesystem_error& e) {
            panic_runtime(e.what());
        } catch (const std::exception& e) {
            panic("Path: {}. {}", path, e.what());
        }
    }

    /// Whether or not the file-status describes an existing directory.
    inline bool is_directory(const fs::file_status& file_status) noexcept {
        return fs::is_directory(file_status);
    }

    /// Whether or not the path points to an existing file or directory. Symlinks are followed.
    inline bool is_file_or_directory(const fs::path& path) {
        try {
            return fs::exists(path);
        } catch (const fs::filesystem_error& e) {
            panic_runtime(e.what());
        } catch (const std::exception& e) {
            panic("File: {}. {}", path, e.what());
        }
    }

    /// Whether or not the file-status describes an existing file or directory. Symlinks are followed.
    inline bool is_file_or_directory(const fs::file_status& file_status) noexcept {
        return fs::exists(file_status);
    }

    /// Gets the size, in bytes, of a regular file. Symlinks are followed.
    /// \note The result of attempting to determine the size of a directory (as well as any other
    ///       file that is not a regular file or a symlink) is implementation-defined.
    inline auto file_size(const fs::path& path) -> int64_t {
        try {
            return clamp_cast<int64_t>(fs::file_size(path));
        } catch (const fs::filesystem_error& e) {
            panic_runtime(e.what());
        } catch (const std::exception& e) {
            panic("File: {}. {}", path, e.what());
        }
    }

    /// \param path         File or empty directory. If it doesn't exist, do nothing.
    /// \throw Exception    If the file or empty directory could not be removed or if the directory was not empty.
    /// \note Symlinks are removed but not their targets.
    inline void remove(const fs::path& path) {
        try {
            fs::remove(path);
        } catch (const fs::filesystem_error& e) {
            panic_runtime(e.what());
        } catch (const std::exception& e) {
            panic("File: {}. {}", path, e.what());
        }
    }

    /// \param path If it is a file, this is similar to remove()
    ///             If it is a directory, it removes it and all its content.
    ///             If it does not exist, do nothing.
    /// \throw Exception    If the file or directory could not be removed.
    /// \note Symlinks are remove but not their targets.
    inline void remove_all(const fs::path& path) {
        try {
            fs::remove_all(path);
        } catch (const fs::filesystem_error& e) {
            panic_runtime(e.what());
        } catch (const std::exception& e) {
            panic("File: {}. {}", path, e.what());
        }
    }

    /// Changes the name or location of a file or directory.
    /// \param from     File or directory to rename.
    /// \param to       Desired name or location.
    /// \throw If \a from could not be moved. One of the reasons my be:
    ///          - \a to ends with . or ..
    ///          - \a to names a non-existing directory ending with a directory separator.
    ///          - \a from is a directory which is an ancestor of \a to.
    ///          - If \a from is a file, \a to should be a file (or non-existing file).
    ///          - If \a from is a directory, \a to should be non-existing directory. On POSIX, \a to can
    ///            be an empty directory, but on other systems, it can fail.
    ///
    /// \note    Symlinks are not followed: if \a from is a symlink, it is itself renamed,
    ///          not its target. If \a to is an existing symlink, it is itself erased, not its target.
    inline void move(const fs::path& from, const fs::path& to) {
        try {
            fs::rename(from, to);
        } catch (const fs::filesystem_error& e) {
            panic_runtime(e.what());
        } catch (const std::exception& e) {
            panic("File: {} to {}. {}", from, to, e.what());
        }
    }

    /// Copies a single regular file (symlinks are followed).
    /// \param from     Existing file to copy.
    /// \param to       Desired name or location. Non-existing file or a regular file different than \a from.
    /// \param options  The behavior is undefined if there is more than one option.
    ///                 When the file \a to already exists:
    ///                 \c fs::copy_options::none:                  Throw an error.
    ///                 \c fs::copy_options::skip_existing:         Keep the existing file, without reporting an error.
    ///                 \c fs::copy_options::overwrite_existing:    Replace the existing file.
    ///                 \c fs::copy_options::update_existing:       Replace the existing file only if it is older
    ///                                                             than the file being copied.
    /// \return         Whether or not the file was copied.
    ///
    /// \throw If \a from is not a regular file.
    ///        If \a from and \a to are equivalent.
    ///        If \a option is empty or if \a to exists and options == fs::copy_options::none.
    inline bool copy_file(
        const fs::path& from,
        const fs::path& to,
        const fs::copy_options options = fs::copy_options::overwrite_existing
    ) {
        try {
            return fs::copy_file(from, to, options);
        } catch (const fs::filesystem_error& e) {
            panic_runtime(e.what());
        } catch (const std::exception& e) {
            panic("File: {} to {}. {}", from, to, e.what());
        }
    }

    /// Copies a symbolic link, effectively reading the symlink and creating a new symlink of the target.
    /// \param from     Path to a symbolic link to copy. Can resolve to a file or directory.
    /// \param to       Destination path of the new symlink.
    /// \throw          If the symlink could not be copied.
    inline void copy_symlink(const fs::path& from, const fs::path& to) {
        try {
            fs::copy_symlink(from, to);
        } catch (const fs::filesystem_error& e) {
            panic_runtime(e.what());
        } catch (const std::exception& e) {
            panic("File: {} to {}. {}", from, to, e.what());
        }
    }

    /// Copies files and directories
    /// \param from     Existing regular file, symlink or directory to copy.
    /// \param to       Desired name or location. Non-existing, regular file, symlink or directory, different than \a from.
    /// \param options  The behavior is undefined if there is more than one option of the same group.
    ///  Group 1:
    ///   ├─ \c fs::copy_options::none:                 Skip subdirectories.
    ///   └─ \c fs::copy_options::recursive:            Recursively copy subdirectories and their content.
    ///  Group 2:
    ///   ├─ \c fs::copy_options::none:                 Symlinks are followed.
    ///   ├─ \c fs::copy_options::copy_symlinks:        Symlinks are not followed.
    ///   └─ \c fs::copy_options::skip_symlinks:        Ignore symlinks.
    ///  Group 3:
    ///   ├─ \c fs::copy_options::none:                 Copy file content.
    ///   ├─ \c fs::copy_options::directories_only:     Copy the directory structure, but do not copy any non-directory files.
    ///   ├─ \c fs::copy_options::create_symlinks:      Instead of creating copies of files, create symlinks pointing to the originals.
    ///   │                                             Note: \a from must be an absolute path unless \a to is in the current directory.
    ///   │                                             Note: \a from must be a file.
    ///   └─ \c fs::copy_options::create_hard_links:    Instead of creating copies of files, create hardlinks.
    ///  Group 4:
    ///   └─ Any of the options from copy_file().
    ///
    /// \throw If \a from or \a to is not a regular file, symlink or directory.
    ///        If \a from is equivalent to \a to or if \a from does not exist.
    ///        If \a from is a directory but \a to is a regular file.
    ///        If \a from is a directory and \c create_symlinks is on.
    ///
    /// \note If \a from is a regular file and \a to is a directory, it copies \a from into \a to.
    /// \note If \a from and \a to are directories, it copies the content of \a from into \a to.
    /// \note To copy a single file, use copy_file().
    inline void copy(
        const fs::path& from,
        const fs::path& to,
        const fs::copy_options options =
            fs::copy_options::recursive |
            fs::copy_options::copy_symlinks |
            fs::copy_options::overwrite_existing
    ) {
        try {
            fs::copy(from, to, options);
        } catch (const fs::filesystem_error& e) {
            panic_runtime(e.what());
        } catch (const std::exception& e) {
            panic("File: {} to {}. {}", from, to, e.what());
        }
    }

    /// \param from         Path of the file to backup. The backup is suffixed with '~'.
    /// \param overwrite    Whether or not it should perform a backup move or backup copy.
    /// \warning With backup moves, symlinks are moved, whereas backup copies follow
    ///          the symlinks and copy the targets. This is usually the expected behavior.
    inline void backup(const fs::path& from, bool overwrite = false) {
        try {
            const fs::path to(from.string() + '~');
            if (overwrite)
                noa::io::move(from, to);
            else
                noa::io::copy_file(from, to);
        } catch (...) {
            panic("File: {}. Could not backup the file", from);
        }
    }

    /// \param path         Path of the directory or directories to create.
    /// \throw Exception    If the directory or directories could not be created.
    /// \note               Existing directories are tolerated and do not generate errors.
    inline void mkdir(const fs::path& path) {
        if (path.empty())
            return;
        try {
            fs::create_directories(path);
        } catch (const fs::filesystem_error& e) {
            panic_runtime(e.what());
        } catch (const std::exception& e) {
            panic("File: {}. {}", path, e.what());
        }
    }

    /// Returns the directory location suitable for temporary files.
    inline auto temporary_directory() -> fs::path {
        try {
            return fs::temp_directory_path();
        } catch (const fs::filesystem_error& e) {
            panic_runtime(e.what());
        } catch (const std::exception& e) {
            panic_runtime(e.what());
        }
    }
}
