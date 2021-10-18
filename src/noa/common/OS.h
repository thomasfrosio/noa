/// \file noa/common/OS.h
/// \brief OS namespace and some file system related functions
/// \author Thomas - ffyr2w
/// \date 9 Oct 2020

#pragma once

#include <filesystem>
#include <cstddef>
#include <cstdint>

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Types.h"

/// Gathers a bunch of OS/filesystem related functions.
/// All functions throws upon failure.
///
/// \note   It is useful to remember that with filesystem::status, symlinks ARE followed.
///         As such, functions using filesystem::status work on the targets, not the links.
///         Therefore, there's no need to explicitly check for and read symlinks.
namespace noa::os {
    using copy_opt = fs::copy_options;

    /// Whether or not \a path points to an existing regular file. Symlinks are followed.
    NOA_IH bool existsFile(const fs::path& path) {
        try {
            return fs::is_regular_file(path);
        } catch (const fs::filesystem_error& e) {
            NOA_THROW(e.what());
        } catch (const std::exception& e) {
            NOA_THROW("File: {}. {}", path, e.what());
        }
    }

    /// Whether or not \a file_status describes an existing regular file. Symlinks are followed.
    NOA_IH bool existsFile(const fs::file_status& file_status) noexcept {
        return fs::is_regular_file(file_status);
    }

    /// Whether or not \a path points to an existing file or directory. Symlinks are followed.
    NOA_IH bool exists(const fs::path& path) {
        try {
            return fs::exists(path);
        } catch (const fs::filesystem_error& e) {
            NOA_THROW(e.what());
        } catch (const std::exception& e) {
            NOA_THROW("File: {}. {}", path, e.what());
        }
    }

    /// Whether or not \a file_status describes an existing file or directory. Symlinks are followed.
    NOA_IH bool exists(const fs::file_status& file_status) noexcept {
        return fs::exists(file_status);
    }

    /// Gets the size, in bytes, of a regular file. Symlinks are followed.
    /// \note The result of attempting to determine the size of a directory (as well as any other
    ///       file that is not a regular file or a symlink) is implementation-defined.
    NOA_IH size_t size(const fs::path& path) {
        try {
            return fs::file_size(path);
        } catch (const fs::filesystem_error& e) {
            NOA_THROW(e.what());
        } catch (const std::exception& e) {
            NOA_THROW("File: {}. {}", path, e.what());
        }
    }

    /// \param path  File or empty directory. If it doesn't exist, do nothing.
    /// \throw If the file or empty directory could not be removed or if the directory was not empty.
    /// \note Symlinks are removed but not their targets.
    NOA_IH void remove(const fs::path& path) {
        try {
            fs::remove(path);
        } catch (const fs::filesystem_error& e) {
            NOA_THROW(e.what());
        } catch (const std::exception& e) {
            NOA_THROW("File: {}. {}", path, e.what());
        }
    }

    /// \param path If it is a file, this is similar to os::remove()
    ///             If it is a directory, it removes it and all its content.
    ///             If it does not exist, do nothing.
    /// \throw If the file or directory could not be removed.
    /// \note Symlinks are remove but not their targets.
    NOA_IH void removeAll(const fs::path& path) {
        try {
            fs::remove_all(path);
        } catch (const fs::filesystem_error& e) {
            NOA_THROW(e.what());
        } catch (const std::exception& e) {
            NOA_THROW("File: {}. {}", path, e.what());
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
    NOA_IH void move(const fs::path& from, const fs::path& to) {
        try {
            fs::rename(from, to);
        } catch (const fs::filesystem_error& e) {
            NOA_THROW(e.what());
        } catch (const std::exception& e) {
            NOA_THROW("File: {} to {}. {}", from, to, e.what());
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
    NOA_IH bool copyFile(const fs::path& from, const fs::path& to,
                         const copy_opt options = copy_opt::overwrite_existing) {
        try {
            return fs::copy_file(from, to, options);
        } catch (const fs::filesystem_error& e) {
            NOA_THROW(e.what());
        } catch (const std::exception& e) {
            NOA_THROW("File: {} to {}. {}", from, to, e.what());
        }
    }

    /// Copies a symbolic link, effectively reading the symlink and creating a new symlink of the target.
    /// \param from     Path to a symbolic link to copy. Can resolve to a file or directory.
    /// \param to       Destination path of the new symlink.
    /// \throw          If the symlink could not be copied.
    NOA_IH void copySymlink(const fs::path& from, const fs::path& to) {
        try {
            fs::copy_symlink(from, to);
        } catch (const fs::filesystem_error& e) {
            NOA_THROW(e.what());
        } catch (const std::exception& e) {
            NOA_THROW("File: {} to {}. {}", from, to, e.what());
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
    ///   └─ Any of the options from copyFile().
    ///
    /// \throw If \a from or \a to is not a regular file, symlink or directory.
    ///        If \a from is equivalent to \a to or if \a from does not exist.
    ///        If \a from is a directory but \a to is a regular file.
    ///        If \a from is a directory and \c create_symlinks is on.
    ///
    /// \note If \a from is a regular file and \a to is a directory, it copies \a from into \a to.
    /// \note If \a from and \a to are directories, it copies the content of \a from into \a to.
    /// \note To copy a single file, use copyFile().
    NOA_IH void copy(const fs::path& from,
                     const fs::path& to,
                     const copy_opt options = copy_opt::recursive |
                                              copy_opt::copy_symlinks |
                                              copy_opt::overwrite_existing) {
        try {
            fs::copy(from, to, options);
        } catch (const fs::filesystem_error& e) {
            NOA_THROW(e.what());
        } catch (const std::exception& e) {
            NOA_THROW("File: {} to {}. {}", from, to, e.what());
        }
    }

    /// \param from         Path of the file to backup. The backup is suffixed with '~'.
    /// \param overwrite    Whether or not it should perform a backup move or backup copy.
    ///
    /// \throw   If the backup did not succeed.
    /// \warning With backup moves, symlinks are moved, whereas backup copies follow
    ///          the symlinks and copy the targets. This is usually the expected behavior.
    NOA_IH void backup(const fs::path& from, bool overwrite = false) {
        try {
            fs::path to(from.string() + '~');
            if (overwrite)
                os::move(from, to);
            else
                os::copyFile(from, to);
        } catch (...) {
            NOA_THROW("File: {}. Could not backup the file", from);
        }
    }

    /// \param path     Path of the directory or directories to create.
    /// \throw          If the directory or directories could not be created.
    /// \note           Existing directories are tolerated and do not generate errors.
    NOA_IH void mkdir(const fs::path& path) {
        if (path.empty())
            return;
        try {
            fs::create_directories(path);
        } catch (const fs::filesystem_error& e) {
            NOA_THROW(e.what());
        } catch (const std::exception& e) {
            NOA_THROW("File: {}. {}", e.what());
        }
    }

    /// Returns the directory location suitable for temporary files.
    NOA_IH path_t tempDirectory() {
        try {
            return fs::temp_directory_path();
        } catch (const fs::filesystem_error& e) {
            NOA_THROW(e.what());
        } catch (const std::exception& e) {
            NOA_THROW(e.what());
        }
    }
}
