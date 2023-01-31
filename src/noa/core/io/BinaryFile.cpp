#include "noa/core/io/IO.hpp"
#include "noa/core/io/BinaryFile.hpp"

namespace noa::io {
    void BinaryFile::open_(open_mode_t open_mode) {
        close();

        NOA_CHECK(isValidOpenMode(open_mode), "File: {}. Invalid open mode", m_path);
        bool overwrite = open_mode & io::TRUNC || !(open_mode & (io::READ | io::APP));
        bool exists;
        try {
            exists = os::existsFile(m_path);
            if (open_mode & io::WRITE) {
                if (exists)
                    os::backup(m_path, overwrite);
                else if (overwrite)
                    os::mkdir(m_path.parent_path());
            }
        } catch (...) {
            NOA_THROW_FUNC("open", "File: {}. Mode: {}. Could not open the file because of an OS failure. {}",
                           m_path, OpenModeStream{open_mode});
        }

        open_mode |= io::BINARY;
        for (int it = 0; it < 5; ++it) {
            m_fstream.open(m_path, io::toIOSBase(open_mode));
            if (m_fstream.is_open())
                return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        m_fstream.clear();

        if (open_mode & io::READ && !overwrite && !exists) {
            NOA_THROW_FUNC("open", "File: {}. Mode: {}. Failed to open the file. The file does not exist",
                           m_path, OpenModeStream{open_mode});
        }
        NOA_THROW_FUNC("open", "File: {}. Mode: {}. Failed to open the file. Check the permissions for that directory",
                       m_path, OpenModeStream{open_mode});
    }
}
