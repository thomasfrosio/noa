/**
 * @file File.h
 * @brief File base class.
 * @author Thomas - ffyr2w
 * @date 9 Oct 2020
 */
#pragma once

#include "noa/Base.h"


namespace Noa::File {

    /** Base class to handle path related functions. */
    class NOA_API File {
    protected:
        std::filesystem::path m_path{};

    public:
        /**
         * Default constructor.
         * @tparam T    Convertible to std::filesystem::path.
         * @param str   Filename or more generic path, which will be copied or moved in @c m_path.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit File(T&& path) noexcept: m_path(std::forward<T>(path)) {}


        /**
         * Get the size of the file at @c path.
         * @param[in] path  Path pointing at the file to check. If it doesn't exist, throws an error.
         * @return          The size of the file, in bytes.
         */
        static inline size_t size(const std::filesystem::path& path) {
            try {
                return std::filesystem::file_size(path);
            } catch (std::exception& e) {
                NOA_CORE_ERROR("\"{}\": error while trying to get the file size.\n"
                               "OS: {}", path.c_str(), e.what());
            }
        }


        /**
         * Get the size of the file stored in the current instance (i.e. @c m_path).
         * @return  the size of the file in bytes or throws an error if the file doesn't exist.
         */
        [[nodiscard]] inline size_t size() const {
            return size(m_path);
        }


        /**
         * @param[in] path  File to check. Symlinks are followed.
         * @return          Whether or not the file @c path exists.
         */
        static inline bool exist(const std::filesystem::path& path) {
            try {
                return std::filesystem::exists(path);
            } catch (std::exception& e) {
                NOA_CORE_ERROR("\"{}\": error while checking if the file exists.\n"
                               "OS: {}", path.c_str(), e.what());
            }
        }


        /** @return Whether or not the file @c m_path exists. */
        [[nodiscard]] inline bool exist() const {
            return exist(m_path);
        }


        /**
         * Deletes the content of @c path, whether it is a file or empty directory.
         * @note Symlinks are remove but not their targets.
         * @note If the file or directory doesn't exist, do nothing.
         */
        static inline void remove(const std::filesystem::path& path) {
            try {
                std::filesystem::remove(path);
            } catch (std::exception& e) {
                NOA_CORE_ERROR("\"{}\": error while trying to remove the file or directory.\n"
                               "OS: {}", path.c_str(), e.what());
            }
        }


        /** Deletes the content of @c m_path, whether it is a file or empty directory. */
        inline void remove() const {
            remove(m_path);
        }


        /**
         * Delete the contents of @c path.
         * @param[in] path  If it is a file, this is similar to @c remove()
         *                  If it is a directory, removes it and all its content.
         * @note            Symlinks are remove but not their targets.
         */
        static inline void removeDirectory(const std::filesystem::path& path) {
            try {
                std::filesystem::remove_all(path);
            } catch (std::exception& e) {
                NOA_CORE_ERROR("\"{}\": error while trying to remove the file.\n"
                               "OS: {}", path.c_str(), e.what());
            }
        }


        /**
         * Change the name or location of a file.
         * @param[in] from  File to rename.
         * @param[in] to    Desired name or location.
         */
        static inline void rename(const std::filesystem::path& from,
                                  const std::filesystem::path& to) {
            try {
                std::filesystem::rename(from, to);
            } catch (std::exception& e) {
                NOA_CORE_ERROR("\"{}\": error while trying to rename the file to {}.\n"
                               "OS: {}", from.c_str(), to.c_str(), e.what());
            }
        }


        /**
         * Change the name or location of @c m_path.
         * @param[in] to    Desired name or location.
         */
        inline void rename(const std::filesystem::path& to) const {
            rename(m_path, to);
        }
    };
}
