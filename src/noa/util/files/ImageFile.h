/**
 * @file ImageFile.h
 * @brief ImageFile::get, to access the correct AbstractImageFile at runtime.
 * @author Thomas - ffyr2w
 * @date 24/10/2020
 */
#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <ios>
#include <filesystem>
#include <string>

#include "noa/API.h"
#include "noa/util/string/Format.h"
#include "noa/util/files/AbstractImageFile.h"
#include "noa/util/files/MRCFile.h"

namespace Noa {
    /**
     * @note    The previous implementation was using an opaque pointer, but this is much simpler and probably
     *          safer since it is obvious that we are dealing with a unique_ptr and that it is necessary to check
     *          whether or not it is a nullptr before using it.
     */
    class NOA_API ImageFile {
        using openmode_t = std::ios_base::openmode;
    public:
        /**
         * Creates an initialized instance with a path, but the file is NOT opened.
         * @return  One of the derived AbstractImageFile or nullptr if the extension is not recognized.
         * @warning Before using the file, check that the returned ptr is valid.
         *
         * @note    @c new(std::nothrow) could be used to prevent a potential bad_alloc, but in this case
         *          returning a nullptr could also be because the extension is not recognized...
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        [[nodiscard]] static std::unique_ptr<AbstractImageFile> get(T&& path) {
            std::string extension = String::toLower(path.extension().string());
            if (extension == ".mrc" || extension == ".st" || extension == ".rec" || extension == ".mrcs")
                return std::make_unique<MRCFile>(std::forward<T>(path));
                // else if (extension == ".tif" || extension == ".tiff")
                //    return std::make_unique<TIFFFile>(std::forward<T>(path));
                // else if (extension == ".eer")
                //    return std::make_unique<EERFile>(std::forward<T>(path));
            else
                return nullptr;
        }

        /**
         * Creates an initialized instance with a path and opens the file.
         * @return  One of the derived AbstractImageFile or nullptr if the extension is not recognized.
         * @see     AbstractImageFile::open() for more details.
         * @warning Before using the file, check that the returned ptr is valid and the file is open.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, fs::path>>>
        [[nodiscard]] static std::unique_ptr<AbstractImageFile> get(T&& path, openmode_t mode, bool long_wait = false) {
            std::string extension = String::toLower(path.extension().string());
            if (extension == ".mrc" || extension == ".st" || extension == ".rec" || extension == ".mrcs")
                return std::make_unique<MRCFile>(std::forward<T>(path), mode, long_wait);
                // else if (extension == ".tif" || extension == ".tiff")
                //    return std::make_unique<TIFFFile>(std::forward<T>(path), mode, long_wait);
                // else if (extension == ".eer")
                //    return std::make_unique<EERFile>(std::forward<T>(path), mode, long_wait);
            else
                return nullptr;
        }
    };
}
