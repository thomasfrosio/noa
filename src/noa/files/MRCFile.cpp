#include "noa/files/MRCFile.h"


Noa::errno_t Noa::MRCFile::readAll(float* data) {
    if (m_state)
        return m_state;
    m_fstream->seekg(getOffset_());
    if (m_fstream)
        m_state = IO::readFloat(*m_fstream, data, getShape().prod(),
                                m_data_type, true, m_is_endian_swapped);
    return m_state;
}


Noa::errno_t Noa::MRCFile::readSlice(float* data, size_t z_pos, size_t z_count) {
    if (m_state)
        return m_state;

    Int3<size_t> shape = getShape();
    size_t elements_per_slice = shape.prodSlice();
    size_t elements_to_read = elements_per_slice * z_count;
    size_t bytes_per_slice = elements_per_slice * IO::bytesPerElement(m_data_type);

    m_fstream->seekg(getOffset_() + static_cast<long>(z_pos * bytes_per_slice));
    if (m_fstream)
        m_state = IO::readFloat(*m_fstream, data, elements_to_read,
                                m_data_type, true, m_is_endian_swapped);
    return m_state;
}


Noa::errno_t Noa::MRCFile::setDataType(IO::DataType data_type) {
    if (data_type == IO::DataType::byte) {
        *m_header.mode = 0;
    } else if (data_type == IO::DataType::float32) {
        *m_header.mode = 2;
    } else if (data_type == IO::DataType::int16) {
        *m_header.mode = 1;
    } else if (data_type == IO::DataType::uint16) {
        *m_header.mode = 6;
    } else if (data_type == IO::DataType::ubyte) {
        *m_header.mode = 1;
        if (*m_header.imod_stamp == 1146047817)
            *m_header.imod_flags &= 1;
    } else if (data_type == IO::DataType::int32 || data_type == IO::DataType::uint32) {
        return Errno::set(m_state, Errno::not_supported);
    } else {
        return Errno::set(m_state, Errno::invalid_data);
    }
    m_data_type = data_type;
    return m_state;
}


Noa::errno_t Noa::MRCFile::writeAll(float* data) {
    if (m_state)
        return m_state;

    m_fstream->seekg(getOffset_());
    if (m_fstream)
        m_state = IO::writeFloat(data, *m_fstream, getShape().prod(), m_data_type,
                                 true, m_is_endian_swapped);
    return m_state;
}


Noa::errno_t Noa::MRCFile::writeSlice(float* data, size_t z_pos, size_t z_count) {
    if (m_state)
        return m_state;

    Int3<size_t> shape = getShape();
    size_t elements_per_slice = shape.prodSlice();
    size_t elements_to_read = elements_per_slice * z_count;
    size_t bytes_per_slice = elements_per_slice * IO::bytesPerElement(m_data_type);

    m_fstream->seekg(getOffset_() + static_cast<long>(z_pos * bytes_per_slice));
    if (m_fstream)
        m_state = IO::writeFloat(data, *m_fstream, elements_to_read, m_data_type,
                                 true, m_is_endian_swapped);
    return m_state;
}


std::string Noa::MRCFile::toString(bool brief) const {
    if (brief)
        return fmt::format("Shape: {}; Pixel size: {}",
                           m_header.shape->toString(), getPixelSize().toString());

    std::string labels(800, '\0');
    for (int i{0}; i < 10; ++i) {
        labels += fmt::format("Label {}: {}\n", i + 1,
                              std::string_view(m_header.labels + 80 * i, 80));
    }

    return fmt::format("Shape (columns, rows, sections): {}\n"
                       "Pixel size (columns, rows, sections): {}\n"
                       "MRC data mode: {} ({})\n"
                       "Bit depth: {}\n"
                       "Extended headers: {} bytes\n"
                       "{}",
                       m_header.shape->toString(), getPixelSize().toString(),
                       *m_header.mode, IO::toString(m_data_type),
                       IO::bytesPerElement(m_data_type) * 8,
                       *m_header.extended_bytes_nb, labels);
}


void Noa::MRCFile::syncHeader_() {
    // C-style cast, reinterpret_cast and memcpy should all
    // be optimized into the same assembly code here.

    char* buffer = m_buffer.get();
    m_header.shape = reinterpret_cast<Int3<int>*>(buffer);
    m_header.mode = reinterpret_cast<int*>(buffer + 12);
    m_header.shape_sub = reinterpret_cast<Int3<int>*>(buffer + 16);
    m_header.shape_grid = reinterpret_cast<Int3<int>*>(buffer + 28);
    m_header.shape_cell = reinterpret_cast<Float3<float>*>(buffer + 40);
    m_header.angles = reinterpret_cast<Float3<float>*>(buffer + 52);
    m_header.map_order = reinterpret_cast<Int3<int>*>(buffer + 64);

    m_header.min = reinterpret_cast<float*>(buffer + 76);
    m_header.max = reinterpret_cast<float*>(buffer + 80);
    m_header.mean = reinterpret_cast<float*>(buffer + 84);

    m_header.space_group = reinterpret_cast<int*>(buffer + 88);
    m_header.extended_bytes_nb = reinterpret_cast<int*>(buffer + 92);
    m_header.extra00 = buffer + 96;
    m_header.extended_type = buffer + 104;
    m_header.nversion = reinterpret_cast<int*>(buffer + 108);

    m_header.imod_stamp = reinterpret_cast<int*>(buffer + 152);
    m_header.imod_flags = reinterpret_cast<int*>(buffer + 156);

    m_header.origin = reinterpret_cast<Float3<float>*>(buffer + 196);
    m_header.cmap = buffer + 208;
    m_header.stamp = buffer + 212;
    m_header.rms = reinterpret_cast<float*>(buffer + 216);

    m_header.nb_labels = reinterpret_cast<int*>(buffer + 220);
    m_header.labels = buffer + 224;
}


Noa::errno_t Noa::MRCFile::open_(std::ios_base::openmode mode, bool wait) {
    if (close())
        return m_state;

    uint32_t iterations = wait ? 10 : 5;
    size_t time_to_wait = wait ? 3000 : 10;

    bool exists = OS::existsFile(m_path, m_state);
    bool overwrite = mode & std::ios::trunc || !(mode & std::ios::in);
    if (mode & std::ios::out) {
        if (exists) {
            Errno::set(m_state, OS::backup(m_path, overwrite));
        } else if (overwrite) {
            Errno::set(m_state, OS::mkdir(m_path.parent_path()));
            initHeader_();
        }
        if (m_state)
            return m_state;
    }

    m_open_mode = mode | std::ios::binary;
    m_open_mode &= ~(std::ios::app | std::ios::ate);

    for (uint32_t it{0}; it < iterations; ++it) {
        m_fstream->open(m_path.c_str(), m_open_mode);
        if (m_fstream) {
            if (exists && !overwrite)
                readHeader_();
            return m_state;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(time_to_wait));
    }
    return Errno::set(m_state, Errno::fail_open);
}


Noa::errno_t Noa::MRCFile::close_() {
    if (!m_fstream->is_open())
        return m_state;

    if (m_open_mode & std::ios::out)
        writeHeader_();
    m_fstream->close();
    if (m_fstream->fail())
        Errno::set(m_state, Errno::fail_close);
    return m_state;
}


Noa::errno_t Noa::MRCFile::validate_() {
    if (*m_header.shape < 0 && *m_header.shape_cell < 0 && *m_header.shape_grid < 0)
        return Errno::set(m_state, Errno::invalid_data);

    // DataType.
    if (*m_header.mode == 0) {
        m_data_type = IO::DataType::byte;
        if (*m_header.imod_stamp == 1146047817 && *m_header.imod_flags & 1)
            m_data_type = IO::DataType::ubyte;
    } else if (*m_header.mode != 2) {
        m_data_type = IO::DataType::float32;
    } else if (*m_header.mode == 1) {
        m_data_type = IO::DataType::int16;
    } else if (*m_header.mode == 6) {
        m_data_type = IO::DataType::uint16;
    } else if (*m_header.mode == 16 || *m_header.mode == 101 ||
               *m_header.mode == 3 || *m_header.mode == 4) {
        return Errno::set(m_state, Errno::not_supported);
    } else {
        return Errno::set(m_state, Errno::invalid_data);
    }

    // Map order: x=1, y=2, z=3 is the only supported order atm.
    if (*m_header.map_order != Int3<int>(1, 2, 3)) {
        if (*m_header.map_order < 1 || *m_header.map_order > 3 || m_header.map_order->sum() != 6)
            return Errno::set(m_state, Errno::invalid_data);
        else
            return Errno::set(m_state, Errno::not_supported);
    }

    if (*m_header.extended_bytes_nb < 0)
        return Errno::set(m_state, Errno::invalid_data);

    // Endianness.
    if (m_header.stamp[0] == 68 && m_header.stamp[1] == 65 &&
        m_header.stamp[2] == 0 && m_header.stamp[3] == 0) /* little endian */
        m_is_endian_swapped = OS::isBigEndian();
    else if (m_header.stamp[0] == 17 && m_header.stamp[1] == 17 &&
             m_header.stamp[2] == 0 && m_header.stamp[3] == 0) /* big endian */
        m_is_endian_swapped = !OS::isBigEndian();
    else
        return Errno::set(m_state, Errno::invalid_data);
    return m_state;
}


void Noa::MRCFile::initHeader_() {
    *m_header.shape = 0;
    setDataType(IO::DataType::float32);
    *m_header.shape_sub = 0;
    *m_header.shape_grid = 1;
    *m_header.shape_cell = 1.f;
    *m_header.angles = 90.f;
    *m_header.map_order = {1, 2, 3};
    *m_header.min = 0;
    *m_header.max = -1;
    *m_header.mean = -2;
    *m_header.space_group = 1;
    *m_header.extended_bytes_nb = 0;
    for (int i = 0; i < 8; ++i)
        m_buffer.get()[96 + i] = 0;
    m_header.extended_type[0] = 'S';
    m_header.extended_type[1] = 'E';
    m_header.extended_type[2] = 'R';
    m_header.extended_type[3] = 'I';
    *m_header.nversion = 20140;
    for (int i = 0; i < 88; ++i)
        m_header.extra00[i] = 0;
    *m_header.origin = 0.f;
    m_header.cmap[0] = 'M';
    m_header.cmap[1] = 'A';
    m_header.cmap[2] = 'P';
    m_header.cmap[3] = ' ';
    setEndianness_();
    *m_header.rms = 0;
    *m_header.nb_labels = 0;
    for (int i = 0; i < 800; ++i)
        m_header.labels[i] = ' ';
}


void Noa::MRCFile::setEndianness_() {
    if (OS::isBigEndian()) {
        m_header.stamp[0] = 17;
        m_header.stamp[1] = 17;
        m_header.stamp[2] = 0;
        m_header.stamp[3] = 0;
    } else {
        m_header.stamp[0] = 68;
        m_header.stamp[1] = 65;
        m_header.stamp[2] = 0;
        m_header.stamp[3] = 0;
    }
    m_is_endian_swapped = false;
}
