#include "noa/files/TextFile.h"


std::string Noa::TextFile::toString(Noa::errno_t& err) {
    std::string buffer;
    try {
        buffer.resize(size(err));
    } catch (std::length_error& e) {
        err = Errno::out_of_memory;
    }
    if (err)
        return buffer;

    m_fstream->seekg(0);
    m_fstream->read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (m_fstream->fail())
        err = Errno::fail;
    return buffer;
}
