/**
 * @file File.h
 * @brief File abstract.
 * @author Thomas - ffyr2w
 * @date 9 Oct 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/utils/Traits.h"


namespace Noa::File {

    class NOA_API File {
    protected:
        std::filesystem::path m_path{};

    public:
        /**
         * Default constructor.
         * @tparam T    Convertible to std::filesystem::path (or std::string...).
         * @param str   Filename or more generic path, which will be stored in @c m_path.
         */
        template<typename T,
                typename = std::enable_if_t<std::is_constructible_v<std::filesystem::path, T>>>
        explicit File(T&& str) noexcept : m_path(std::forward<T>(str)) {}


        /**
         * Get the size of the file @c path.
         * @param[in] path  Path pointing at the file to check. If it doesn't exist, report an error.
         * @return          The size of the file in bytes.
         */
        static inline size_t size(const std::filesystem::path& path) {
            std::error_code error_code;
            size_t size = std::filesystem::file_size(path, error_code);
            checkError(error_code);
            return size;
        }


        /**
         * Get the size of the file @c m_path.
         * @return  the size of the file in bytes or report an error if it doesn't exist.
         */
        [[nodiscard]] inline size_t size() const {
            std::error_code error_code;
            size_t size = std::filesystem::file_size(m_path, error_code);
            checkError(error_code);
            return size;
        }


        /**
         * @param[in] path  File to check. Symlink are followed.
         * @return          Whether or not the file @c path exists.
         */
        static inline bool exist(const std::filesystem::path& path) {
            std::error_code error_code;
            bool exist = std::filesystem::remove(path, error_code);
            checkError(error_code);
            return exist;
        }


        /**
         * File to check for existence. Symlink are followed.
         * @return Whether or not the file @c m_path exists.
         */
        [[nodiscard]] inline bool exist() const {
            std::error_code error_code;
            bool exist = std::filesystem::exists(m_path, error_code);
            checkError(error_code);
            return exist;
        }


        /**
         * Delete the content of @c path, whether it is a file or empty directory.
         * @note Symlinks are remove but not their targets.
         * @note If the file or directory doesn't exist, do nothing.
         */
        static inline void remove(const std::filesystem::path& path) {
            std::error_code error_code;
            std::filesystem::remove(path, error_code);
            checkError(error_code);
        }


        /**
         * Delete the content of @c m_path, whether it is a file or empty directory.
         * @note Symlinks are remove but not their targets.
         * @note If the file or directory doesn't exist, do nothing.
         */
        inline void remove() const {
            std::error_code error_code;
            std::filesystem::remove(m_path, error_code);
            checkError(error_code);
        }


        /**
         * Delete the contents of @c path.
         * @param[in] path  If it is a file, this is similar to @c remove()
         *                  If it is a directory, remove it and all its content.
         * @note            Symlinks are remove but not their targets.
         */
        static inline void removeAll(const std::filesystem::path& path) {
            std::error_code error_code;
            std::filesystem::remove_all(path, error_code);
            checkError(error_code);
        }

        /**
         * Delete the contents of @c m_path.
         * If @c m_path is a file, this is similar to @c remove().
         * If @c m_path is a directory, remove it and all its content.
         * @note Symlinks are remove but not their targets.
         */
        inline void removeAll() const {
            std::error_code error_code;
            std::filesystem::remove_all(m_path, error_code);
            checkError(error_code);
        }


    private:
        static inline void checkError(const std::error_code& error_code) {
            if (error_code) {
                NOA_CORE_ERROR("error: {}", error_code.message());
            }
        }
    };
}
