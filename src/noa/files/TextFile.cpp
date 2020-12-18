#include "noa/files/TextFile.h"


std::string Noa::TextFile::toString() {
    std::string buffer;
    try {
        buffer.resize(size());
    } catch (std::length_error& e) {
        setState_(Errno::out_of_memory);
    }
    if (m_state)
        return buffer;

    m_fstream->seekg(0);
    m_fstream->read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (m_fstream->fail())
        setState_(Errno::fail_read);
    return buffer;
}
