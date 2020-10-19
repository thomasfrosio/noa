/**
 * @file File.h
 * @brief File abstract.
 * @author Thomas - ffyr2w
 * @date 9 Oct 2020
 */
#pragma once

#include <utility>

#include "noa/Base.h"
#include "noa/utils/Traits.h"


namespace Noa::File {

    /**
     *
     */
    class NOA_API File {
    protected:
        std::filesystem::path m_path{};

    public:
        /**
         * Default constructor.
         * @tparam T    Convertible to std::filesystem::path (or std::string...).
         * @param str   Filename or more generic path, which will be stored in @c m_path.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit File(T&& path) noexcept: m_path(std::forward<T>(path)) {}


        /**
         * Get the size of the file @c path.
         * @param[in] path  Path pointing at the file to check. If it doesn't exist, report an error.
         * @return          The size of the file in bytes.
         */
        static inline size_t size(const std::filesystem::path& path, uint8_t& err) noexcept {
            std::error_code error_code;
            size_t size = std::filesystem::file_size(path, error_code);
            if (error_code)
                err = Errno::fail;
            return size;
        }


        /**
         * Get the size of the file @c m_path.
         * @return  the size of the file in bytes or report an error if it doesn't exist.
         */
        [[nodiscard]] inline size_t size() const {
            try {
                return std::filesystem::file_size(m_path);
            } catch (std::exception& e) {
                NOA_CORE_ERROR("\"{}\": error while trying to get the file size.\n"
                               "OS: {}", m_path.c_str(), e.what());
            }
        }


        /**
         * @param[in] path  File to check. Symlink are followed.
         * @return          Whether or not the file @c path exists.
         */
        static inline bool exist(const std::filesystem::path& path, uint8_t& err) noexcept {
            std::error_code error_code;
            bool exist = std::filesystem::exists(path, error_code);
            if (error_code)
                err = Errno::fail;
            return exist;
        }


        /**
         * File to check for existence. Symlink are followed.
         * @return Whether or not the file @c m_path exists.
         */
        [[nodiscard]] inline bool exist() const {
            try {
                return std::filesystem::exists(m_path);
            } catch (std::exception& e) {
                NOA_CORE_ERROR("\"{}\": error while checking if the file exists.\n"
                               "OS: {}", m_path.c_str(), e.what());
            }
        }


        /**
         * Delete the content of @c path, whether it is a file or empty directory.
         * @note Symlinks are remove but not their targets.
         * @note If the file or directory doesn't exist, do nothing.
         */
        static inline void remove(const std::filesystem::path& path, uint8_t& err) noexcept {
            std::error_code error_code;
            std::filesystem::remove(path, error_code);
            if (error_code)
                err = Errno::fail;
        }


        /**
         * Delete the content of @c m_path, whether it is a file or empty directory.
         * @note Symlinks are remove but not their targets.
         * @note If the file or directory doesn't exist, do nothing.
         */
        inline void remove() const {
            try {
                std::filesystem::remove(m_path);
            } catch (std::exception& e) {
                NOA_CORE_ERROR("\"{}\": error while trying to remove the file.\n"
                               "OS: {}", m_path.c_str(), e.what());
            }
        }


        /**
         * Delete the contents of @c path.
         * @param[in] path  If it is a file, this is similar to @c remove()
         *                  If it is a directory, remove it and all its content.
         * @note            Symlinks are remove but not their targets.
         */
        static inline void removeDirectory(const std::filesystem::path& path,
                                           uint8_t& err) noexcept {
            std::error_code error_code;
            std::filesystem::remove_all(path, error_code);
            if (error_code)
                err = Errno::fail;
        }


        /**
         * Change the name or location of a file.
         * @param[in] from  File to look rename.
         * @param[in] to    Desired name or location.
         */
        static inline void rename(const std::filesystem::path& from,
                                  const std::filesystem::path& to,
                                  uint8_t& err) noexcept {
            std::error_code error_code;
            std::filesystem::rename(from, to, error_code);
            if (error_code)
                err = Errno::fail;
        }


        /**
         * Change the name or location of @c m_path.
         * @param[in] to    Desired name or location.
         */
        inline void rename(const std::filesystem::path& to) const {
            try {
                std::filesystem::rename(m_path, to);
            } catch (std::exception& e) {
                NOA_CORE_ERROR("\"{}\": error while trying to rename the file to {}.\n"
                               "OS: {}", m_path.c_str(), to.c_str(), e.what());
            }
        }
    };
}
