#include "noa/core/io/IO.hpp"
#include "noa/core/io/BinaryFile.hpp"

namespace noa::io {
    void BinaryFile::open_(OpenMode mode, const std::source_location& location) {
        close();

        const bool overwrite = mode.truncate or not (mode.read or mode.append);
        bool exists;
        try {
            exists = is_file(m_path);
            if (mode.write) {
                if (exists)
                    backup(m_path, overwrite);
                else if (overwrite)
                    mkdir(m_path.parent_path());
            }
        } catch (...) {
            panic_at_location(location, "File: {}. {}. Could not open the file because of an OS failure", m_path, mode);
        }

        mode.binary = true;
        for (i32 it = 0; it < 5; ++it) {
            m_fstream.open(m_path, io::to_ios_base(mode));
            if (m_fstream.is_open())
                return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        m_fstream.clear();

        if (mode.read and not overwrite and not exists)
            panic_at_location(location, "File: {}. {}. Failed to open the file. The file does not exist", m_path, mode);
        panic_at_location(location, "File: {}. {}. Failed to open the file. Check the permissions", m_path, mode);
    }
}
