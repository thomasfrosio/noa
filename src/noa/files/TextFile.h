/**
 * @file TextFile.h
 * @brief Text file class.
 * @author Thomas - ffyr2w
 * @date 9 Oct 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/OS.h"


namespace Noa {
    /**
     * Base class for all text files.
     * It is not copyable, but it is movable.
     * @note    Instances of this class keep track of a @c Errno, referred to as @a state.
     *          Member functions can modify this state and usually returns it to let the caller
     *          knows about the current state of the class. Member functions will only modify this
     *          state if it is in a "good" state.
     */
    class NOA_API TextFile {
    private:
        fs::path m_path{};
        std::unique_ptr<std::fstream> m_fstream;
        errno_t m_state{Errno::good};

    public:
        /** Initializes the underlying file stream. */
        explicit TextFile() : m_fstream(std::make_unique<std::fstream>()) {}


        /** Initializes the path and underlying file stream. The file isn't opened. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit TextFile(T&& path)
                : m_path(std::forward<T>(path)), m_fstream(std::make_unique<std::fstream>()) {}


        /** Sets and opens the associated file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit TextFile(T&& path, std::ios_base::openmode mode, bool long_wait = false)
                : m_path(std::forward<T>(path)), m_fstream(std::make_unique<std::fstream>()) {
            m_state = open(m_path, *m_fstream, mode, long_wait);
        }


        /** Resets the path and opens the associated file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline errno_t open(T&& path, std::ios_base::openmode mode, bool long_wait = false) {
            m_path = std::forward<T>(path);
            return open(mode, long_wait);
        }


        /** Closes the stream and reopens it with the current path. */
        inline errno_t open(std::ios_base::openmode mode, bool long_wait = false) {
            setState_(open(m_path, *m_fstream, mode, long_wait));
            return m_state;
        }


        /** Closes the stream if it is opened, otherwise don't do anything. */
        inline errno_t close() {
            setState_(close(*m_fstream));
            return m_state;
        }


        /** Writes a string(_view) to the file. */
        template<typename T, typename = std::enable_if_t<Traits::is_string_v<T>>>
        void write(T&& string) {
            m_fstream->write(string.data(), static_cast<std::streamsize>(string.size()));
        }


        /**
         * (Formats and) writes a string.
         * @tparam Args     Anything accepted by @c fmt::format()
         * @param[in] args  C-string and|or variable(s) used to compute the formatted string.
         *
         * @note            This function depends on the current ofstream position. If std::ios::app,
         *                  the position is set to the end of the file at every call.
         */
        template<typename... Args>
        void write(Args&& ... args) {
            std::string message = fmt::format(std::forward<Args>(args)...);
            m_fstream->write(message.c_str(), static_cast<std::streamsize>(message.size()));
        }


        /**
         * Gets the next line of the ifstream.
         * @param[in] line  Buffer into which the line will be stored. It is erased before starting.
         * @return          A temporary reference of the istream. With its operator() evaluating to
         *                  istream.fail(), this is meant to be used in a @c while condition. If
         *                  evaluates to false, it means the line could not be read, either because
         *                  the stream is @c bad() or because it reached the end of line.
         *
         * @example Read a file line per line.
         * @code
         * TextFile file("some_file.txt");
         * std::string line;
         * while(file.getLine(line)) {
         *     // do something with the line
         * }
         * if (file.bad())
         *     // error while reading the file
         * else
         *     // everything is OK, the end of the file was reached without error.
         * @endcode
         */
        inline std::istream& getLine(std::string& line) { return std::getline(*m_fstream, line); }


        /** Closes the stream and deletes the file. */
        inline errno_t remove() {
            if (close() || OS::remove(m_path))
                setState_(Errno::fail);
            return m_state;
        }


        /** Closes the stream and renames the file. The file is not reopened. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline errno_t rename(T&& to) {
            if (close() || OS::move(m_path, to))
                setState_(Errno::fail);
            m_path = std::forward<T>(to);
            return m_state;
        }


        /** Closes, renames and reopens the file. */
        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        inline errno_t rename(T&& to, std::ios_base::openmode mode, bool long_wait = false) {
            if (rename(std::forward<T>(to)))
                return m_state;
            return open(mode, long_wait);
        }


        /**
         * Loads the entire file into a @c std::string.
         * @return  String containing the whole content of @a m_path.
         * @note    The ifstream is rewound before reading.
         */
        std::string toString();


        /**
         * Gets a reference of the underlying file stream.
         * @warning This should be safe and the class should be able to handle whatever changes are
         *          done outside the class. One thing that is possible but not really meant to be
         *          changed is the exception level of the stream. If you activate some exceptions,
         *          make sure you know what you are doing, specially when activating @c eofbit.
         *
         * @note @c std::fstream doesn't throw exceptions by default but keeps track of a few flags
         *       reporting on the situation. Here is more information on their meaning.
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
        [[nodiscard]] inline std::fstream& fstream() noexcept { return *m_fstream; }

        /** Whether or not @a m_path points to a regular file or a symlink pointing to a regular file. */
        inline bool exists() noexcept { return !m_state && OS::existsFile(m_path, m_state); }

        /** Gets the size (in bytes) of the file at @a m_path. Symlinks are followed. */
        inline size_t size() noexcept { return !m_state ? OS::size(m_path, m_state) : 0U; }

        [[nodiscard]] inline const fs::path& path() const noexcept { return m_path; }

        [[nodiscard]] inline bool bad() const noexcept { return m_fstream->bad(); }
        [[nodiscard]] inline bool eof() const noexcept { return m_fstream->eof(); }
        [[nodiscard]] inline bool fail() const noexcept { return m_fstream->fail(); }
        [[nodiscard]] inline bool isOpen() const noexcept { return m_fstream->is_open(); }

        [[nodiscard]] inline errno_t getState() const { return m_state; }
        inline void resetState() { m_state = Errno::good; }

        /** Whether or not the instance is in a "good" state. Checks for Errno and file stream state. */
        [[nodiscard]] inline explicit operator bool() const noexcept {
            return !m_state && !m_fstream->fail();
        }

        /** Whether or not the instance is in a "fail" state. Checks for Errno and file stream state. */
        [[nodiscard]] inline bool operator!() const noexcept {
            return m_state || m_fstream->fail();
        }

        /**
         * Opens and associates the file in @a path with @a fstream.
         * @tparam T            Same as default constructor.
         * @tparam S            A file stream, one of @c std::(i|o)fstream.
         * @param[in] path      Same as default constructor.
         * @param[out] fstream  File stream, opened or closed, to associate with @c path.
         * @param[in] mode      Any of the @c std::ios_base::openmode.
         *                      in: Opens the ifstream.
         *                      out: Opens the ofstream.
         *                      trunc: Discard the contents of the streams (i.e. overwrite).
         *                      binary: Disable text conversions.
         *                      ate: ofstream and ifstream seek the end of the file after opening.
         *                      app: ofstream seeks the end of the file before each writing.
         * @param[in] long_wait Wait for the file to exist for 10*3s, otherwise wait for 5*10ms.
         * @return              @c Errno::fail_close, if failed to close @a fstream before starting.
         *                      @c Errno::fail_open, if failed to open @a fstream.
         *                      @c Errno::fail_os, if an underlying OS error was raised.
         *                      @c Errno::good, otherwise.
         *
         * @note                If the file is opened in writing mode (@a mode & out), a backup is
         *                      saved before opening the file. If the file is overwritten
         *                      (@a mode & trunc), the file is moved, otherwise it is copied.
         */
        template<typename S, typename = std::enable_if_t<std::is_same_v<S, std::ifstream> ||
                                                         std::is_same_v<S, std::ofstream> ||
                                                         std::is_same_v<S, std::fstream>>>
        errno_t open(const fs::path& path, S& fstream, std::ios_base::openmode mode,
                     bool long_wait = false) {
            if (close(fstream))
                return Errno::fail_close;

            uint32_t iterations = long_wait ? 10 : 5;
            size_t time_to_wait = long_wait ? 3000 : 10;

            if constexpr (!std::is_same_v<S, std::ifstream>) {
                if (mode & std::ios::out) {
                    errno_t err{Errno::good};
                    bool exists = OS::existsFile(path, err);
                    if (exists)
                        err = OS::backup(path, !(mode & std::ios::trunc));
                    else
                        err = OS::mkdir(path.parent_path());
                    if (err)
                        return err;
                }
            }
            for (uint32_t it{0}; it < iterations; ++it) {
                fstream.open(path.c_str(), mode);
                if (fstream)
                    return Errno::good;
                std::this_thread::sleep_for(std::chrono::milliseconds(time_to_wait));
            }
            return Errno::fail_open;
        }


        /**
         * Close @a fstream if it is opened, otherwise don't do anything.
         * @return @c Errno::fail_close, if @a fstream failed to close.
         *         @c Errno::good, otherwise.
         */
        template<typename S, typename = std::enable_if_t<std::is_same_v<S, std::ifstream> ||
                                                         std::is_same_v<S, std::ofstream> ||
                                                         std::is_same_v<S, std::fstream>>>
        inline errno_t close(S& fstream) {
            if (!fstream.is_open())
                return Errno::good;
            fstream.close();
            return fstream.fail() ? Errno::fail_close : Errno::good;
        }

    private:
        /** Updates the state only if it is Errno::good. Otherwise, do nothing. */
        inline void setState_(errno_t err) {
            if (err && !m_state)
                m_state = err;
        }
    };
}
