/**
 * @file Text.h
 * @brief Text file class.
 * @author Thomas - ffyr2w
 * @date 9 Oct 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/files/File.h"


namespace Noa::File {

    /**
     * Basic text file, which handles a file stream.
     * It is not copyable, but it is movable.
     */
    class NOA_API Text : public File {
    protected:
        std::unique_ptr<std::fstream> m_fstream{nullptr};

    public:
        /**
         * Set @c m_path to @c path and initialize the stream @c m_fstream. The file is not opened.
         * @tparam T    A valid path, by lvalue or rvalue.
         * @param path  Filename to store in the current instance.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit Text(T&& path)
                : File(std::forward<T>(path)), m_fstream(std::make_unique<std::fstream>()) {}


        /**
         * Set @c m_path to @c path, open and associate it with the file stream @c m_fstream.
         * @tparam T            A valid path, by lvalue or rvalue.
         * @param[in] path      Filename to store in the current instance.
         * @param[in] mode      Any of the open mode (in|out|trunc|app|ate|binary).
         * @param[in] long_wait Wait for the file to exist for 10*30s, otherwise wait for 5*10ms.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit Text(T&& path, std::ios_base::openmode mode, bool long_wait = false)
                : File(std::forward<T>(path)), m_fstream(std::make_unique<std::fstream>()) {
            reopen(mode, long_wait);
        }


        /**
         * Reset @c m_path to @c path, open and associate it with the file stream @c m_fstream.
         * @tparam T            A valid path, by lvalue or rvalue.
         * @param[in] path      Filename to store in the current instance.
         * @param[in] mode      Any of the open mode (in|out|trunc|app|ate|binary).
         * @param[in] long_wait Wait for the file to exist for 10*3s, otherwise wait for 5*10ms.
         */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline void open(T&& path, std::ios_base::openmode mode, bool long_wait = false) {
            m_path = std::forward<T>(path);
            reopen(mode, long_wait);
        }


        /**
         * Close the file and reopen it.
         * @param[in] mode      Any of the open mode (in|out|trunc|app|ate|binary).
         * @param[in] long_wait Wait for the file to exist for 10*30s, otherwise wait for 5*10ms.
         */
        inline void reopen(std::ios_base::openmode mode, bool long_wait = false) {
            ::Noa::File::Text::open(m_path, *m_fstream, mode, long_wait);
        }


        /**
         * Close @c fstream if it is open, otherwise don't do anything.
         * @note @c fstream::close() will set @c failbit if the stream is not open.
         */
        inline void close() {
            if (!close(*m_fstream)) {
                NOA_CORE_ERROR("error detected while closing the file \"{}\": {}",
                               m_path.c_str(), std::strerror(errno));
            }
        }


        /**
         * Open and associate the file in @c path with the file stream buffer of @c fs.
         * @tparam T            A valid path, by lvalue or rvalue.
         * @tparam S            A file stream, one of @c std::(i|o)fstream.
         * @param[in] path      Path pointing at the filename to open.
         * @param[out] fs       File stream, opened or closed, to associate with @c path.
         * @param[in] mode      Any of the @std::ios_base::openmode.
         *                      in: Open ifstream. Operations on the ofstream will be ignored.
         *                      out: Open ofstream. Operations on the ifstream will be ignored.
         *                      binary: Disable text conversions.
         *                      ate: ofstream and ifstream seek the end of the file after opening.
         *                      app: ofstream seeks the end of the file before each writing.
         * @param[in] long_wait Wait for the file to exist for 10*30s, otherwise wait for 5*10ms.
         */
        template<typename S,
                typename = std::enable_if_t<std::is_same_v<S, std::ifstream> ||
                                            std::is_same_v<S, std::ofstream> ||
                                            std::is_same_v<S, std::fstream>>>
        static void open(const std::filesystem::path& path,
                         S& fs,
                         std::ios_base::openmode mode,
                         bool long_wait = false) {
            if (!Text::close(fs)) {
                NOA_CORE_ERROR("\"{}\": error while closing the file. {}",
                               path.c_str(), std::strerror(errno));
            }
            size_t iterations = long_wait ? 10 : 5;
            size_t time_to_wait = long_wait ? 3000 : 10;

            for (size_t it{0}; it < iterations; ++it) {
                if constexpr (!std::is_same_v<S, std::ifstream>) {
                    // If only reading mode, the file should be there - no need to create it.
                    if (mode & std::ios::out && path.has_parent_path())
                        std::filesystem::create_directories(path.parent_path());
                }
                fs.open(path.c_str(), mode);
                if (fs)
                    return;
                std::this_thread::sleep_for(std::chrono::milliseconds(time_to_wait));
            }
            NOA_CORE_ERROR("\"{}\": error while opening the file. {}",
                           path.c_str(), std::strerror(errno));
        }


        /**
         * Close @c fstream if it is open, otherwise don't do anything.
         * @return  Whether or not the file was closed.
         */
        template<typename S, typename = std::enable_if_t<std::is_same_v<S, std::ifstream> ||
                                                         std::is_same_v<S, std::ofstream> ||
                                                         std::is_same_v<S, std::fstream>>>
        static inline bool close(S& fstream) noexcept {
            if (!fstream.is_open())
                return true;
            fstream.close();
            return !fstream.fail();
        }


        /**
         * Write a formatted string into the ofstream.
         * @tparam Args     Anything accepted by @c fmt::format()
         * @param[in] args  C-string and|or variable(s) used to compute the formatted string.
         *
         * @note            This function depends on the ofstream position. If std::ios::app,
         *                  the position is set to the end of the file at every call.
         */
        template<typename... Args>
        void write(Args&& ... args) {
            std::string message = fmt::format(std::forward<Args>(args)...);
            m_fstream->write(message.c_str(), static_cast<std::streamsize>(message.size()));
            if (m_fstream->fail()) {
                if (!m_fstream->is_open()) {
                    NOA_CORE_ERROR("\"{}\": file is not open. Open it with open() or reopen()",
                                   m_path.c_str());
                } else {
                    NOA_CORE_ERROR("\"{}\": error while writing to file. {}",
                                   m_path.c_str(), std::strerror(errno));
                }
            }
        }


        /**
         * Load the entire file into a @c std::string.
         * @return  String containing the whole content of @c m_path.
         * @note    The ifstream is rewind before reading, so @c std::ios::ate has no effect
         *          on this function.
         */
        std::string toString();


        /**
         * Get a reference of the @c fstream.
         * @warning This should be safe and the class should be able to handle whatever changes are
         *          done outside the class. One thing that is possible but not really meant to be
         *          changed is the exception level of the stream. If you activate some exceptions,
         *          make sure you know what you are doing, specially when activating @c eofbit.
         *
         * @note @c std::fstream doesn't throw exceptions by default but keeps track of a few flags
         *          reporting on the situation. Here is more information on how to check for them.
         *          - @c goodbit: its value, 0, indicates the absence of any error flag. If 1,
         *                        all input and output operations have no effect.
         *                        See @c std::fstream::good().
         *          - @c eofbit:  Is set when there an attempt to read past the end of an input sequence.
         *                        When reaching the last character, the stream is still in good state,
         *                        but any subsequent extraction will be considered an attempt to read
         *                        past the end - `eofbit` is set to 1. The other situation is when
         *                        the reading doesn't happen character-wise and we reach the eof.
         *                        See @c std::fstream::eof().
         *          - @c failbit: Is set when a read or write operation fails. For example, in the
         *                        first example of `eofbit`, `failbit` is also set since we fail to
         *                        read, but in the second example it is not set since the int or string
         *                        was extracted. `failbit` is also set if the file couldn't be open.
         *                        See @c std::fstream::fail() or @c std::fstream::operator!().
         *          - @c badbit:  Is set when a problem with the underlying stream buffer happens. This
         *                        can happen from memory shortage or because the underlying stream
         *                        buffer throws an exception.
         *                        See @c std::fstream::bad().
         */
        [[nodiscard]] inline std::fstream& fstream() const noexcept {
            return *m_fstream;
        }
    };
}


