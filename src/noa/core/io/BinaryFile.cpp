#include "noa/core/io/IO.hpp"
#include "noa/core/io/BinaryFile.hpp"

namespace noa::io {
    void BinaryFile::open_(open_mode_t open_mode, const std::source_location& location) {
        close();

        check(is_valid_open_mode(open_mode), "File: {}. Invalid open mode", m_path);
        const bool overwrite = open_mode & io::TRUNC || !(open_mode & (io::READ | io::APP));
        bool exists;
        try {
            exists = is_file(m_path);
            if (open_mode & io::WRITE) {
                if (exists)
                    backup(m_path, overwrite);
                else if (overwrite)
                    mkdir(m_path.parent_path());
            }
        } catch (...) {
            panic_at_location(
                    location, "File: {}. Mode: {}. Could not open the file because of an OS failure",
                    m_path, OpenModeStream{open_mode});
        }

        open_mode |= io::BINARY;
        for (i32 it = 0; it < 5; ++it) {
            m_fstream.open(m_path, io::to_ios_base(open_mode));
            if (m_fstream.is_open())
                return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        m_fstream.clear();

        if (open_mode & io::READ && !overwrite && !exists) {
            panic_at_location(
                    location, "File: {}. Mode: {}. Failed to open the file. The file does not exist",
                    m_path, OpenModeStream{open_mode});
        }
        panic_at_location(
                location, "File: {}. Mode: {}. Failed to open the file. Check the permissions for that directory",
                m_path, OpenModeStream{open_mode});
    }
}
