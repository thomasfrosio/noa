/**
 * @file OS.h
 * @brief OS namespace and some file system related functions
 * @author Thomas - ffyr2w
 * @date 9 Oct 2020
 */
#pragma once

#include "noa/Base.h"


/**
 * Gathers a bunch of OS/filesystem related functions.
 * Functions are all @c noexcept and use @c Errno::fail_os error number to report failure.
 */
namespace Noa::OS {
    /**
     * @param[in] path      Path pointing at the file to check. Symlinks are followed.
     * @param[in] status    File status. It should corresponds to @a path.
     * @param[out] err      @c Errno to use. Is set to @c Errno::fail_os upon failure.
     * @return              The size of the file, in bytes. @c 0 if it fails.
     */
    inline size_t size(const fs::path& path, const fs::file_status& status, errno_t& err) noexcept {
        try {
            if (fs::is_symlink(status))
                return fs::file_size(fs::read_symlink(path));
            else
                return fs::file_size(path);
        } catch (std::exception& e) {
            err = Errno::fail_os;
            return 0;
        }
    }


    /** Overload without the file status */
    inline size_t size(const fs::path& path, errno_t& err) noexcept {
        try {
            return OS::size(path, fs::status(path), err);
        } catch (std::exception& e) {
            err = Errno::fail_os;
            return 0;
        }
    }


    /**
     * @param[in] path  File or empty directory. If it doesn't exist, do nothing.
     * @return          @c Errno::fail_os if the file or directory was not be removed
     *                  or if the directory was not empty.
     * @note            Symlinks are remove but not their targets.
     */
    inline errno_t remove(const fs::path& path) noexcept {
        try {
            fs::remove(path);
            return Errno::good;
        } catch (std::exception& e) {
            return Errno::fail_os;
        }
    }


    /**
     * @note            Symlinks are remove but not their targets.
     * @param[in] path  If it is a file, this is similar to OS::remove()
     *                  If it is a directory, it removes it and all its content.
     * @return          @c Errno::fail_os if the file or directory was not removed.
     *                  @c Errno::good otherwise.
     */
    inline errno_t removeDirectory(const fs::path& path) noexcept {
        try {
            fs::remove_all(path);
            return Errno::good;
        } catch (std::exception& e) {
            return Errno::fail_os;
        }
    }


    /**
     * Change the name or location of a file.
     * @note Symlinks are not followed: if @a from is a symlink, it is itself renamed,
     *       not its target. If @a to is an existing symlink, it is itself moved, not its target.
     * @param[in] from  File to rename.
     * @param[in] to    Desired name or location.
     * @return          @c Errno::fail_os if the file was not moved.
     *                  @c Errno::good otherwise.
     */
    inline errno_t move(const fs::path& from, const fs::path& to) noexcept {
        try {
            fs::rename(from, to);
            return Errno::good;
        } catch (std::exception& e) {
            return Errno::fail_os;
        }
    }


    /**
     * Copy the name or location of a file.
     * @note Symlinks are followed; their targets is copied, not the symlink.
     * @param[in] from  File to rename.
     * @param[in] to    Desired name or location.
     * @return          @c Errno::fail_os if the file was not copied.
     *                  @c Errno::good otherwise.
     */
    inline errno_t copy(const fs::path& from, const fs::path& to) noexcept {
        try {
            if (fs::copy_file(from, to, fs::copy_options::overwrite_existing))
                return Errno::good;
            return Errno::fail_os;
        } catch (std::exception& e) {
            return Errno::fail_os;
        }
    }


    /**
     * @param[in] from      Path of the file to backup.
     * @param[in] status    File status of @a from.
     * @param[in] copy      Whether or not the file should be copied or moved.
     * @return              @c Errno::fail_os if the file was renamed
     *                      @c Errno::good otherwise.
     */
    inline errno_t backup(const fs::path& from,
                          const fs::file_status& status,
                          bool copy) noexcept {
        if (fs::is_regular_file(status) || fs::is_symlink(status)) {
            try {
                fs::path to = from.string() + "~";
                if (copy)
                    return OS::copy(from, to);
                else
                    return OS::move(from, to);
            } catch (std::exception& e) {
                return Errno::fail_os;
            }
        }
        return Errno::good;
    }


    /** Overload without the file status */
    inline errno_t backup(const fs::path& from, bool copy) noexcept {
        try {
            return OS::backup(from, fs::status(from), copy);
        } catch (std::exception& e) {
            return Errno::fail_os;
        }
    }


    /**
     * @param[in] path  Path of the directories to create.
     * @return          @c Errno::fail_os if the directory or directories were not created.
     *                  @c Errno::good otherwise.
     */
    inline errno_t mkdir(const fs::path& path) noexcept {
        try {
            if (path.has_parent_path())
                fs::create_directories(path.parent_path());
            return Errno::good;
        } catch (std::exception& e) {
            return Errno::fail_os;
        }
    }


    /**
     * @param[in]   File to check.
     * @param[out]  @c Errno to use. Will be set to @c Errno::fail_os if an error is thrown.
     * @return      Whether or not @a file points to a file.
     */
    static inline bool exist(const fs::path& file, errno_t& err) noexcept {
        try {
            auto status = fs::status(file);
            return fs::is_regular_file(status) || fs::is_symlink(status);
        } catch (std::exception& e) {
            err = Errno::fail_os;
            return false;
        }
    }


    /**
     * @return  Whether or not the machine running this code is big-endian.
     * @note Logic: int16_t is made of 2 bytes, int16_t = 1, or 0x0001 in hexadecimal, is:
     * little-endian: 00000001 00000000 or 0x01 0x00 -> to char* -> char[0] == 1, char[1] == 0
     * big-endian   : 00000000 00000001 or 0x00 0x01 -> to char* -> char[0] == 0, char[1] == 1
     */
    inline bool isBigEndian() noexcept {
        int16_t number = 1;
        return *reinterpret_cast<char*>(&number) == 0; // char[0] == 0
    }
}
