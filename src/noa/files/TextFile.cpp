#include "noa/files/TextFile.h"


std::string Noa::TextFile::toString() {
    std::string buffer;
    try {
        buffer.resize(size());
    } catch (std::length_error& e) {
        NOA_CORE_ERROR("error while allocating the string buffer. The size of the file ({} Bytes) "
                       "is larger than the maximum size allowed ({} Bytes)",
                       size(), buffer.max_size());
    }

    m_fstream->seekg(0);
    m_fstream->read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (m_fstream->fail()) {
        if (!m_fstream->is_open()) {
            NOA_CORE_ERROR("\"{}\": file is not open. Open it with open() or reopen()",
                           m_path.c_str());
        } else {
            NOA_CORE_ERROR("\"{}\": error detected while reading the file. {}",
                           m_path.c_str(), std::strerror(errno));
        }
    }
    return buffer;
}
