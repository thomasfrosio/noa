#include "noa/common/io/IO.h"
#include "noa/common/io/BinaryFile.h"

namespace noa {
    void BinaryFile::open_(uint open_mode) {
        close();

        bool overwrite = open_mode & io::TRUNC || !(open_mode & io::READ);
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
            NOA_THROW("File: {}. OS failure when trying to open the file", m_path);
        }

        open_mode |= io::BINARY;
        for (int it = 0; it < 5; ++it) {
            m_fstream.open(m_path, io::toIOSBase(open_mode));
            if (m_fstream.is_open())
                return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        NOA_THROW("File: {}. Failed to open the file", m_path);
    }
}
