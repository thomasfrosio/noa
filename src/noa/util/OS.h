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
     * Get the size of the file at @a path.
     * @param[in] path      Path pointing at the file to check. Symlinks are followed.
     * @param[in] status    File status. It should corresponds to @a path.
     * @param[out] err      @c Errno to use. @c Errno::fail_os upon failure.
     * @return              The size of the file, in bytes. @c 0 if it fails.
     */
    inline size_t size(const std::filesystem::path& path,
                       const std::filesystem::file_status& status,
                       uint8_t& err) noexcept {
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


    /**
     * Get the size of the file at @a path.
     * @param[in] path      Path pointing at the file to check. Symlinks are followed.
     * @param[out] err      @c Errno to use. @c Errno::fail_os if the file (or target)
     *                      doesn't exist or upon error.
     * @return              The size of the file, in bytes. @c 0 if it fails.
     */
    inline size_t size(const std::filesystem::path& path, uint8_t& err) noexcept {
        try {
            return OS::size(path, std::filesystem::status(path), err);
        } catch (std::exception& e) {
            err = Errno::fail_os;
            return 0;
        }
    }


    /**
     * Deletes the content of @a path, whether it is a file or an empty directory.
     * @note Symlinks are remove but not their targets.
     * @note If the file or directory doesn't exist, do nothing.
     * @return  @c Errno::fail_os if the file or directory was not be removed
     *          or if the directory was not empty.
     */
    inline uint8_t remove(const std::filesystem::path& path) noexcept {
        try {
            std::filesystem::remove(path);
            return Errno::good;
        } catch (std::exception& e) {
            return Errno::fail_os;
        }
    }


    /**
     * Delete the contents of @a path.
     * @note            Symlinks are remove but not their targets.
     * @param[in] path  If it is a file, this is similar to OS::remove()
     *                  If it is a directory, removes it and all its content.
     * @return          @c Errno::fail_os if the file or directory was not removed.
     *                  @c Errno::good otherwise.
     */
    inline uint8_t removeDirectory(const std::filesystem::path& path) noexcept {
        try {
            std::filesystem::remove_all(path);
            return Errno::good;
        } catch (std::exception& e) {
            return Errno::fail_os;
        }
    }


    /**
     * Change the name or location of a file.
     * @note Symlinks are not followed: if @a from is a symlink, it is itself renamed,
     *       not its target. If @a to is an existing symlink, it is itself erased, not its target.
     * @param[in] from  File to rename.
     * @param[in] to    Desired name or location.
     * @return          @c Errno::fail_os if the file was not moved.
     *                  @c Errno::good otherwise.
     */
    inline uint8_t move(const std::filesystem::path& from,
                        const std::filesystem::path& to) noexcept {
        try {
            std::filesystem::rename(from, to);
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
    inline uint8_t copy(const std::filesystem::path& from,
                        const std::filesystem::path& to) noexcept {
        try {
            if (std::filesystem::copy_file(from, to, fs::copy_options::overwrite_existing))
                return Errno::good;
            return Errno::fail_os;
        } catch (std::exception& e) {
            return Errno::fail_os;
        }
    }


    inline uint8_t mkdir(const std::filesystem::path& path) noexcept {
        try {
            if (path.has_parent_path())
                std::filesystem::create_directories(path.parent_path());
            return Errno::good;
        } catch (std::exception& e) {
            return Errno::fail_os;
        }
    }


    /**
     * @param[in]   File to check.
     * @param[out]  @c Errno to use. Will be set to @c Errno::fail_os if an error is thrown.
     *              Unchanged otherwise.
     * @return      Whether or not @a file points to a file.
     *              Set it to @c false if an error is thrown.
     */
    static inline bool exists(const fs::path& file, uint8_t& err) noexcept {
        try {
            auto status = fs::status(file);
            return fs::is_regular_file(status) || fs::is_symlink(status);
        } catch (std::exception& e) {
            err = Errno::fail_os;
            return false;
        }
    }


    inline uint8_t backup(const std::filesystem::path& from,
                          const std::filesystem::file_status& status) noexcept {
        if (fs::is_regular_file(status) || fs::is_symlink(status)) {
            try {
                fs::path to = from.string() + "~";
                return OS::copy(from, to);
            } catch (std::exception& e) {
                return Errno::fail_os;
            }
        }
        return Errno::good;
    }


    /**
     * Backup the file at @a path.
     * If the file exists, append its filename with @c '~'.
     * If the file does not exist, do nothing.
     */
    inline uint8_t backup(const std::filesystem::path& from) noexcept {
        try {
            return OS::backup(from, std::filesystem::status(from));
        } catch (std::exception& e) {
            return Errno::fail_os;
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
