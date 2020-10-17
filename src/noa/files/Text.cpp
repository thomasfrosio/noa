#include "noa/files/Text.h"


std::string Noa::File::Text::toString() {
    std::string buffer;
    try {
        buffer.reserve(size());
    } catch (std::length_error& e) {
        NOA_CORE_ERROR("error while allocating the string buffer. "
                       "The size of the file ({}) is larger than the maximum size allowed ({})",
                       size(), buffer.max_size());
    }

    m_file->seekg(0);
    m_file->read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (m_file->fail()) {
        if (!m_file->is_open()) {
            NOA_CORE_ERROR("\"{}\": file isn't open. Open it with open() or reopen()", m_path);
        } else {
            NOA_CORE_ERROR("\"{}\": error detected while reading the file: {}",
                           m_path, std::strerror(errno));
        }
    }
    return buffer;
}
