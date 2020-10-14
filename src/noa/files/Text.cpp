#include "noa/files/Text.h"


void Noa::File::Text::open(std::ios_base::openmode mode, bool long_wait)  {
    if (m_path.empty()) {
        NOA_CORE_ERROR("the path of this instance isn't set - there's nothing to open");
    }
    close();

    // Trigger long_wait if wished.
    size_t iterations = long_wait ? 10 : 5;
    size_t time_to_wait = long_wait ? 3000 : 10;

    for (size_t it{0}; it < iterations; ++it) {
        // If only reading mode, the file should be there - no need to create it.
        if (mode & std::ios::out)
            std::filesystem::create_directories(m_path.parent_path());
        m_file->open(m_path, mode);
        if (m_file->fail())
            return;
        std::this_thread::sleep_for(std::chrono::milliseconds(time_to_wait));
    }
    NOA_CORE_ERROR("error while opening the file \"{}\": {}", m_path, std::strerror(errno));
}


std::string Noa::File::Text::load() {
    try {
        std::string buffer(size(), '\0');
        m_file->seekg(0);
        m_file->read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        if (m_file->fail()) {
            NOA_CORE_ERROR("error detected while reading the file \"{}\": {}",
                           m_path, std::strerror(errno));
        }
        return buffer;
    } catch (std::length_error& e) {
        NOA_CORE_ERROR("error while allocating the string buffer. "
                       "The size of the file ({}) is larger than the maximum size allowed ({})",
                       size(), std::string{}.max_size());
    }
}
