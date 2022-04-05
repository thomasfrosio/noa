/// \file noa/common/files/TextFile.h
/// \brief Text file template class.
/// \author Thomas - ffyr2w
/// \date 9 Oct 2020

#pragma once

#include <cstddef>
#include <ios> // std::streamsize
#include <fstream>
#include <memory>
#include <utility>
#include <type_traits>
#include <thread>
#include <string>

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/OS.h"
#include "noa/common/Types.h"
#include "noa/common/io/IO.h"

namespace noa::io {
    /// Base class for all text files. It is not copyable, but it is movable.
    template<typename Stream = std::fstream>
    class TextFile {
    public:
        /// Initializes the underlying file stream.
        TextFile() = default;

        /// Sets and opens the associated file. See open() for more details.
        TextFile(path_t path, open_mode_t mode);

        /// (Re)Opens the file.
        /// \param filename     Path of the file to open.
        /// \param mode         Open mode. Should be one of the following combination:
        ///                     1) READ                     File should exists.
        ///                     2) READ|WRITE               File should exists.    Backup copy.
        ///                     3) WRITE, WRITE|TRUNC       Overwrite the file.    Backup move.
        ///                     4) READ|WRITE|TRUNC         Overwrite the file.    Backup move.
        ///                     5) APP, WRITE|APP           Append or create file. Backup copy. Append at each write.
        ///                     6) READ|APP, READ|WRITE|APP Append or create file. Backup copy. Append at each write.
        /// \throws Exception   If any of the following cases:
        ///         - If failed to close the file before starting.
        ///         - If failed to open the file.
        ///         - If an underlying OS error was raised.
        ///
        /// \note Additionally, APP and/or BINARY can be turned on:
        ///         - ATE: the stream go to the end of the file after opening.
        ///         - BINARY: Disable text conversions.
        /// \note Specifying TRUNC and APP is undefined.
        void open(path_t path, open_mode_t mode);

        /// Closes the stream if it is opened, otherwise don't do anything.
        void close();

        /// Writes a string or string_view to the file.
        void write(std::string_view string);

        /// Gets the next line of the ifstream.
        /// \param[in] line Buffer into which the line will be stored. It is erased before starting.
        /// \return         A temporary reference of the istream. Its operator bool() evaluates to
        ///                 istream.fail(). If false, it means the line could not be read, either
        ///                 because the stream is failed or because it reached the end of the file.
        ///
        /// \example Read a file line per line.
        /// \code
        /// TextFile file("some_file.txt");
        /// std::string line;
        /// while(file.getLine(line)) {
        ///     // do something with the line
        /// }
        /// if (file.bad())
        ///     // error while reading the file
        /// else
        ///     // file.eof() == true; everything is OK, the end of the file was reached without error.
        /// \endcode
        std::istream& getLine(std::string& line);

        /// Reads the entire file into a string.
        /// \return String containing the whole content of the file.
        /// \note   The ifstream is rewound before reading.
        std::string readAll();

        /// Gets a reference of the underlying file stream.
        /// \note   This should be safe and the class should be able to handle whatever changes are
        ///         done outside the class. One thing that is possible but not really meant to be
        ///         changed is the exception level of the stream. If you activate some exceptions,
        ///         make sure you know what you are doing, specially when activating \c eofbit.
        ///
        /// \note \c std::fstream doesn't throw exceptions by default but keeps track of a few flags
        ///       reporting on the situation. Here is more information on their meaning.
        ///          - \c goodbit: its value, 0, indicates the absence of any error flag. If 1,
        ///                        all input and output operations have no effect.
        ///                        See \c std::fstream::good().
        ///          - \c eofbit:  Is set when there an attempt to read past the end of an input sequence.
        ///                        When reaching the last character, the stream is still in good state,
        ///                        but any subsequent extraction will be considered an attempt to read
        ///                        past the end - `eofbit` is set to 1. The other situation is when
        ///                        the reading doesn't happen character-wise and we reach the eof.
        ///                        See \c std::fstream::eof().
        ///          - \c failbit: Is set when a read or write operation fails. For example, in the
        ///                        first example of `eofbit`, `failbit` is also set since we fail to
        ///                        read, but in the second example it is not set since the int or string
        ///                        was extracted. `failbit` is also set if the file couldn't be open.
        ///                        See \c std::fstream::fail() or \c std::fstream::operator!().
        ///          - \c badbit:  Is set when a problem with the underlying stream buffer happens. This
        ///                        can happen from memory shortage or because the underlying stream
        ///                        buffer throws an exception.
        ///                        See \c std::fstream::bad().
        [[nodiscard]] Stream& fstream() noexcept { return m_fstream; }

        /// Whether or not \a m_path points to a regular file or a symlink pointing to a regular file.
        bool exists() { return os::existsFile(m_path); }

        /// Gets the size (in bytes) of the file at \a m_path. Symlinks are followed.
        size_t size() { return os::size(m_path); }

        [[nodiscard]] const fs::path& path() const noexcept { return m_path; }
        [[nodiscard]] bool bad() const noexcept { return m_fstream.bad(); }
        [[nodiscard]] bool eof() const noexcept { return m_fstream.eof(); }
        [[nodiscard]] bool fail() const noexcept { return m_fstream.fail(); }
        [[nodiscard]] bool isOpen() const noexcept { return m_fstream.is_open(); }
        void clear() { m_fstream.clear(); }

        /// Whether or not the instance is in a "good" state.
        [[nodiscard]] explicit operator bool() const noexcept { return !m_fstream.fail(); }

    private:
        static_assert(std::is_same_v<Stream, std::ifstream> ||
                      std::is_same_v<Stream, std::ofstream> ||
                      std::is_same_v<Stream, std::fstream>);
        path_t m_path{};
        Stream m_fstream{};

    private:
        void open_(open_mode_t);
    };
}

#define NOA_TEXTFILE_INL_
#include "noa/common/io/TextFile.inl"
#undef NOA_TEXTFILE_INL_
