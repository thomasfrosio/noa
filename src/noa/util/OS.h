/**
 * @file OS.h
 * @brief OS namespace and some file system related functions
 * @author Thomas - ffyr2w
 * @date 9 Oct 2020
 */
#pragma once

#include <filesystem>
#include <type_traits>
#include <exception>
#include <cstddef>
#include <cstdint>

#include "noa/API.h"
#include "noa/util/Constants.h" // Errno
#include "noa/util/Flag.h"

/**
 * Gathers a bunch of OS/filesystem related functions.
 * All functions are @c noexcept and use @c Errno::fail_os to report failure.
 *
 * @note:   It is useful to remember that with filesystem::status, symlinks ARE followed.
 *          As such, functions using filesystem::status work on the targets, not the links.
 *          Therefore, there's no need to explicitly check for and read symlinks.
 */
namespace Noa::OS {
    using copy_opt = fs::copy_options;

    /** Whether or not @a path points to an existing regular file. Symlinks are followed. */
    template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
    NOA_API inline bool existsFile(T&& path, Noa::Flag<Noa::Errno>& err) noexcept {
        try {
            return fs::is_regular_file(path);
        } catch (const std::exception&) {
            err = Errno::fail_os;
            return false;
        }
    }

    /** Whether or not @a file_status describes an existing regular file. Symlinks are followed. */
    NOA_API inline bool existsFile(const fs::file_status& file_status, Noa::Flag<Noa::Errno>& err) noexcept {
        try {
            return fs::is_regular_file(file_status);
        } catch (const std::exception&) {
            err = Errno::fail_os;
            return false;
        }
    }

    /** Whether or not @a path points to an existing file or directory. Symlinks are followed. */
    template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
    NOA_API inline bool exists(T&& path, Noa::Flag<Noa::Errno>& err) noexcept {
        try {
            return fs::exists(path);
        } catch (const std::exception&) {
            err = Errno::fail_os;
            return false;
        }
    }

    /** Whether or not @a file_status describes an existing file or directory. Symlinks are followed. */
    NOA_API inline bool exists(const fs::file_status& file_status, Noa::Flag<Noa::Errno>& err) noexcept {
        try {
            return fs::exists(file_status);
        } catch (const std::exception&) {
            err = Errno::fail_os;
            return false;
        }
    }

    /**
     * Gets the size, in bytes, of a regular file. Symlinks are followed.
     * @warning The result of attempting to determine the size of a directory (as well as any other
     *          file that is not a regular file or a symlink) is implementation-defined.
     */
    NOA_API inline size_t size(const fs::path& path, Noa::Flag<Noa::Errno>& err) noexcept {
        try {
            return fs::file_size(path);
        } catch (const std::exception&) {
            err = Errno::fail_os;
            return 0U;
        }
    }

    /**
     * @param[in] path  File or empty directory. If it doesn't exist, do nothing.
     * @return          @c Errno::fail_os, if the file or directory was not removed or if the directory was not empty.
     *                  @c Errno::good, otherwise.
     * @note            Symlinks are removed but not their targets.
     */
    NOA_API inline Noa::Flag<Noa::Errno> remove(const fs::path& path) noexcept {
        try {
            fs::remove(path);
            return Errno::good;
        } catch (const std::exception&) {
            return Errno::fail_os;
        }
    }

    /**
     * @param[in] path  If it is a file, this is similar to OS::remove()
     *                  If it is a directory, it removes it and all its content.
     * @return          @c Errno::fail_os if the file or directory was not removed.
     *                  @c Errno::good otherwise.
     * @note            Symlinks are remove but not their targets.
     */
    NOA_API inline Noa::Flag<Noa::Errno> removeAll(const fs::path& path) noexcept {
        try {
            fs::remove_all(path);
            return Errno::good;
        } catch (const std::exception&) {
            return Errno::fail_os;
        }
    }

    /**
     * Changes the name or location of a file or directory.
     * @param[in] from  File or directory to rename.
     * @param[in] to    Desired name or location.
     * @return          @c Errno::fail_os if the file was not moved.
     *                  @c Errno::good otherwise.
     * @note    Symlinks are not followed: if @a from is a symlink, it is itself renamed,
     *          not its target. If @a to is an existing symlink, it is itself erased, not its target.
     * @note    If @a from is a file, @a to should be a file (or non-existing file).
     *          If @a from is a directory, @a to should be non-existing directory. On POSIX, @a to can
     *          be an empty directory, but on other systems, it can fail: https://en.cppreference.com/w/cpp/filesystem/rename
     * @warning Will fail if:   @a to ends with . or ..
     *                          @a to names a non-existing directory ending with a directory separator.
     *                          @a from is a directory which is an ancestor of @a to.
     */
    NOA_API inline Noa::Flag<Noa::Errno> move(const fs::path& from, const fs::path& to) noexcept {
        try {
            fs::rename(from, to);
            return Errno::good;
        } catch (const std::exception&) {
            return Errno::fail_os;
        }
    }

    /**
     * Copies a single regular file (symlinks are followed).
     * @param[in] from      Existing file to copy.
     * @param[in] to        Desired name or location. Non-existing file or a regular file different than @a from.
     * @param[in] options   The behavior is undefined if there is more than one option.
     *                      When the file @a to already exists:
     *                      @c fs::copy_options::none:                  Report an error.
     *                      @c fs::copy_options::skip_existing:         Keep the existing file, without reporting an error.
     *                      @c fs::copy_options::overwrite_existing:    Replace the existing file.
     *                      @c fs::copy_options::update_existing:       Replace the existing file only if it is older than the file being copied.
     * @return              @c Errno::fail_os, if the file was not copied.
     *                      @c Errno::good, otherwise.
     */
    NOA_API inline Noa::Flag<Noa::Errno> copyFile(const fs::path& from,
                                                  const fs::path& to,
                                                  const copy_opt options = copy_opt::overwrite_existing) noexcept {
        try {
            if (fs::copy_file(from, to, options))
                return Errno::good;
            return Errno::fail_os;
        } catch (const std::exception&) {
            return Errno::fail_os;
        }
    }

    /**
     * Copies a symbolic link, effectively reading the symlink and creating a new symlink of the target.
     * @param[in] from  Path to a symbolic link to copy. Can resolve to a file or directory.
     * @param[in] to    Destination path of the new symlink.
     * @return          @c Errno::fail_os, if the symlink was not copied.
     *                  @c Errno::good, otherwise.
     */
    NOA_API inline Noa::Flag<Noa::Errno> copySymlink(const fs::path& from, const fs::path& to) noexcept {
        try {
            fs::copy_symlink(from, to);
            return Errno::good;
        } catch (const std::exception&) {
            return Errno::fail_os;
        }
    }

    /**
     * Copies files and directories
     * @param[in] from      Existing regular file, symlink or directory to copy.
     * @param[in] to        Desired name or location. Non-existing, regular file, symlink or directory, different than @a from.
     * @param[in] options   The behavior is undefined if there is more than one option of the same group.
     *  Group 1:
     *   ├─ @c fs::copy_options::none:                  Skip subdirectories.
     *   └─ @c fs::copy_options::recursive:             Recursively copy subdirectories and their content.
     *  Group 2:
     *   ├─ @c fs::copy_options::none:                  Symlinks are followed.
     *   ├─ @c fs::copy_options::copy_symlinks:         Symlinks are not followed.
     *   └─ @c fs::copy_options::skip_symlinks:         Ignore symlinks.
     *  Group 3:
     *   ├─ @c fs::copy_options::none:                  Copy file content.
     *   ├─ @c fs::copy_options::directories_only:      Copy the directory structure, but do not copy any non-directory files.
     *   ├─ @c fs::copy_options::create_symlinks:       Instead of creating copies of files, create symlinks pointing to the originals.
     *   │                                              Note: @a from must be an absolute path unless @a to is in the current directory.
     *   │                                              Note: @a from must be a file.
     *   └─ @c fs::copy_options::create_hard_links:     Instead of creating copies of files, create hardlinks.
     *  Group 4:
     *   └─ Any of the options from copyFile().
     *
     * @return              @c Errno::fail_os:  If @a from or @a to is not a regular file, symlink or directory.
     *                                          If @a from is equivalent to @a to or if @a from does not exist.
     *                                          If @a from is a directory but @a to is a regular file.
     *                                          If @a from is a directory and @c create_symlinks is on.
     *                      @c Errno::good, otherwise.
     *
     * @note If @a from is a regular file and @a to is a directory, it copies @a from into @a to.
     * @note If @a from and @a to are directories, it copies the content of @a from into @a to.
     * @note To copy a single file, use copyFile().
     */
    NOA_API inline Noa::Flag<Noa::Errno> copy(const fs::path& from,
                                              const fs::path& to,
                                              const copy_opt options = copy_opt::recursive |
                                                                       copy_opt::copy_symlinks |
                                                                       copy_opt::overwrite_existing) noexcept {
        try {
            fs::copy(from, to, options);
            return Errno::good;
        } catch (const std::exception&) {
            return Errno::fail_os;
        }
    }

    /**
     * @param[in] from      Path of the file to backup. The backup is suffixed with '~'.
     * @param[in] overwrite Whether or not it should perform a backup move or backup copy.
     * @return              @c Errno::fail_os, if the backup did not succeed.
     *                      @c Errno::good, otherwise.
     * @warning             With backup moves, symlinks are moved, whereas backup copies follow
     *                      the symlinks and copy the targets. This is usually the expected behavior.
     */
    NOA_API inline Noa::Flag<Noa::Errno> backup(const fs::path& from, bool overwrite = false) noexcept {
        try {
            fs::path to = from.string() + '~';
            return overwrite ? OS::move(from, to) : OS::copyFile(from, to);
        } catch (const std::exception& e) {
            return Errno::fail_os;
        }
    }

    /**
     * @param[in] path  Path of the directory or directories to create.
     * @return          @c Errno::fail_os if the directory or directories were not created.
     *                  @c Errno::good otherwise.
     * @note            Existing directories are tolerated and do not generate errors.
     */
    NOA_API inline Noa::Flag<Noa::Errno> mkdir(const fs::path& path) noexcept {
        if (path.empty())
            return Errno::good;
        try {
            fs::create_directories(path);
            return Errno::good;
        } catch (const std::exception&) {
            return Errno::fail_os;
        }
    }

    /**
     * @return  Whether or not the machine running this code is big-endian.
     * @note Logic: int16_t is made of 2 bytes, int16_t = 1, or 0x0001 in hexadecimal, is:
     * little-endian: 00000001 00000000 or 0x01 0x00 -> to char* -> char[0] == 1, char[1] == 0
     * big-endian   : 00000000 00000001 or 0x00 0x01 -> to char* -> char[0] == 0, char[1] == 1
     */
    NOA_API inline bool isBigEndian() noexcept {
        int16_t number = 1;
        return *reinterpret_cast<char*>(&number) == 0; // char[0] == 0
    }
}
