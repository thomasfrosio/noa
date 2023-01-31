#ifndef NOA_TEXTFILE_INL_
#error "This file should not be included by anything other than noa/core/io/TextFile.hpp"
#endif

namespace noa::io {
    /// Sets and opens the associated file.
    template<typename Stream>
    TextFile<Stream>::TextFile(path_t path, open_mode_t mode) : m_path(std::move(path)) {
        open_(mode);
    }

    template<typename Stream>
    void TextFile<Stream>::open(path_t path, open_mode_t mode) {
        m_path = std::move(path);
        open_(mode);
    }

    template<typename Stream>
    void TextFile<Stream>::write(std::string_view string) {
        m_fstream.write(string.data(), static_cast<std::streamsize>(string.size()));
        if (m_fstream.fail()) {
            if (m_fstream.is_open())
                NOA_THROW("File: {}. File stream error while writing", m_path);
            NOA_THROW("File: {}. File stream error. File is closed file", m_path);
        }
    }

    template<typename Stream>
    std::istream& TextFile<Stream>::getLine(std::string& line) {
        return std::getline(m_fstream, line);
    }

    template<typename Stream>
    std::string TextFile<Stream>::readAll() {
        std::string buffer;

        m_fstream.seekg(0, std::ios::end);
        std::streampos size = m_fstream.tellg();
        if (!size)
            return buffer;
        else if (size == -1)
            NOA_THROW("File: {}. File stream error. Could not get the input position indicator", m_path);

        try {
            buffer.resize(static_cast<size_t>(size));
        } catch (std::length_error& e) {
            NOA_THROW("File: {}. Passed the maximum permitted size while try to load file. Got {} bytes",
                      m_path, size);
        }

        m_fstream.seekg(0);
        m_fstream.read(buffer.data(), size);
        if (m_fstream.fail())
            NOA_THROW("File: {}. File stream error. Could not read the entire file", m_path);
        return buffer;
    }

    template<typename Stream>
    void TextFile<Stream>::close() {
        if (m_fstream.is_open()) {
            m_fstream.close();
            if (m_fstream.fail())
                NOA_THROW("File: {}. File stream error", m_path);
        }
    }

    template<typename Stream>
    void TextFile<Stream>::open_(open_mode_t mode) {
        close();

        NOA_CHECK(isValidOpenMode(mode), "File: {}. Invalid open mode", m_path);
        if constexpr (!std::is_same_v<Stream, std::ifstream>) {
            if (mode & io::WRITE || mode & io::APP) /* all except case 1 */ {
                bool overwrite = mode & io::TRUNC || !(mode & (io::READ | io::APP)); // case 3|4
                bool exists = os::existsFile(m_path);
                try {
                    if (exists)
                        os::backup(m_path, overwrite);
                    else if (overwrite || mode & io::APP) /* all except case 2 */
                        os::mkdir(m_path.parent_path());
                } catch (...) {
                    NOA_THROW("File: {}. Mode: {}. Could not open the file because of an OS failure. {}",
                              m_path, OpenModeStream{mode});
                }
            }
        }
        const bool read_only = !(mode & (io::WRITE | io::TRUNC | io::APP));
        const bool read_write_only = mode & io::WRITE && mode & io::READ && !(mode & (io::TRUNC | io::APP));
        if constexpr (std::is_same_v<Stream, std::ifstream>) {
            NOA_CHECK(read_only,
                      "File: {}. Mode {} is not allowed for read-only TextFile",
                      m_path, OpenModeStream{mode});
        }

        for (int it{0}; it < 5; ++it) {
            m_fstream.open(m_path, io::toIOSBase(mode));
            if (m_fstream.is_open())
                return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        m_fstream.clear();

        if (read_only || read_write_only) { // case 1|2
            NOA_CHECK(os::existsFile(m_path),
                      "File: {}. Mode: {}. Trying to open a file that does not exist",
                      m_path, OpenModeStream{mode});
        }
        NOA_THROW("File: {}. Mode: {}. Failed to open the file. Check the permissions for that directory",
                  m_path, OpenModeStream{mode});
    }
}
