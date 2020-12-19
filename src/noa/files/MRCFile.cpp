#include "noa/files/MRCFile.h"

#define RC(type, var) reinterpret_cast<type>(var)


void Noa::MRCFile::syncHeader_() {
    // C-style cast, reinterpret_cast and memcpy should all
    // be optimized into the same assembly code here.
    char* header = m_header.get();

    m_shape = RC(Int3<int>*, header);
    m_mode = RC(int*, header + 12);
    m_shape_sub = RC(Int3<int>*, header + 16);
    m_shape_grid = RC(Int3<int>*, header + 28);
    m_shape_cell = RC(Float3<float>*, header + 40);
    m_angles = RC(Float3<float>*, header + 52);
    m_map_order = RC(Int3<int>*, header + 64);

    m_min = RC(float*, header + 76);
    m_max = RC(float*, header + 80);
    m_mean = RC(float*, header + 84);

    m_space_group = RC(int*, header + 88);
    m_extended_bytes_nb = RC(int*, header + 92);
    m_extra00 = header + 96;
    m_extended_type = header + 104;
    m_nversion = RC(int*, header + 108);

    m_imod_stamp = RC(int*, header + 152);
    m_imod_flags = RC(int*, header + 156);

    m_origin = RC(Float3<float>*, header + 196);
    m_cmap = header + 208;
    m_stamp = header + 212;
    m_rms = RC(float*, header + 216);

    m_nb_labels = RC(int*, header + 220);
    m_labels = header + 224;
}


Noa::errno_t Noa::MRCFile::validate_() {
    if (*m_shape < 0 && *m_shape_cell < 0 && *m_shape_grid < 0)
        return Errno::set(m_state, Errno::invalid_data);

    // DataType.
    if (*m_mode == 0) {
        m_data_type = IO::DataType::byte;
        if (*m_imod_stamp == 1146047817 && *m_imod_flags & 1)
            m_data_type = IO::DataType::ubyte;
    } else if (*m_mode != 2) {
        m_data_type = IO::DataType::float32;
    } else if (*m_mode == 1) {
        m_data_type = IO::DataType::int16;
    } else if (*m_mode == 6) {
        m_data_type = IO::DataType::uint16;
    } else if (*m_mode == 16 || *m_mode == 101 ||
               *m_mode == 3 || *m_mode == 4) {
        return Errno::set(m_state, Errno::not_supported);
    } else {
        return Errno::set(m_state, Errno::invalid_data);
    }

    // Map order: x=1, y=2, z=3 is the only supported order atm.
    if (*m_map_order != Int3<int>(1, 2, 3)) {
        if (*m_map_order < 1 || *m_map_order > 3 || m_map_order->sum() != 6)
            return Errno::set(m_state, Errno::invalid_data);
        else
            return Errno::set(m_state, Errno::not_supported);
    }

    if (*m_extended_bytes_nb < 0)
        return Errno::set(m_state, Errno::invalid_data);

    // Endianness.
    char* header = m_header.get();
    if (header[212] == 68 && header[213] == 65 &&
        header[214] == 0 && header[215] == 0) /* little endian */
        m_is_endian_swapped = OS::isBigEndian();
    else if (header[212] == 17 && header[213] == 17 &&
             header[214] == 0 && header[215] == 0) /* big endian */
        m_is_endian_swapped = !OS::isBigEndian();
    else
        return Errno::set(m_state, Errno::invalid_data);
    return m_state;
}


Noa::errno_t Noa::MRCFile::setDataType_(Noa::IO::DataType data_type) {
    if (data_type == IO::DataType::byte) {
        *m_mode = 0;
    } else if (data_type == IO::DataType::float32) {
        *m_mode = 2;
    } else if (data_type == IO::DataType::int16) {
        *m_mode = 1;
    } else if (data_type == IO::DataType::uint16) {
        *m_mode = 6;
    } else if (data_type == IO::DataType::ubyte) {
        *m_mode = 1;
        if (*m_imod_stamp == 1146047817)
            *m_imod_flags &= 1;
    } else if (data_type == IO::DataType::int32 || data_type == IO::DataType::uint32) {
        return Errno::not_supported;
    } else {
        return Errno::invalid_data;
    }
    m_data_type = data_type;
    return Errno::good;
}


void Noa::MRCFile::initHeader_() {
    *m_shape = 0;
    *m_mode = 2;
    *m_shape_sub = 0;
    *m_shape_grid = 1;
    *m_shape_cell = 1.f;
    *m_angles = 90.f;
    *m_map_order = {1, 2, 3};
    *m_min = 0;
    *m_max = -1;
    *m_mean = -2;
    *m_space_group = 1;
    *m_extended_bytes_nb = 0;
    for (int i = 0; i < 8; ++i)
        m_header.get()[96 + i] = 0;
    m_extended_type[0] = 'S';
    m_extended_type[1] = 'E';
    m_extended_type[2] = 'R';
    m_extended_type[3] = 'I';
    *m_nversion = 20140;
    for (int i = 0; i < 88; ++i)
        m_extra00[i] = 0;
    *m_origin = 0.f;
    m_cmap[0] = 'M';
    m_cmap[1] = 'A';
    m_cmap[2] = 'P';
    m_cmap[3] = ' ';
    setEndianness_();
    *m_rms = 0;
    *m_nb_labels = 0;
    for (int i = 0; i < 800; ++i)
        m_labels[i] = ' ';
}


void Noa::MRCFile::setEndianness_() {
    char* header = m_header.get();
    if (OS::isBigEndian()) {
        header[212] = 17;
        header[213] = 17;
        header[214] = 0;
        header[215] = 0;
    } else {
        header[212] = 68;
        header[213] = 65;
        header[214] = 0;
        header[215] = 0;
    }
    m_is_endian_swapped = false;
}


std::string Noa::MRCFile::toString(bool brief) const {
    if (brief)
        return fmt::format("Shape: {}; Pixel size: {}",
                           m_shape->toString(), getPixelSize().toString());

    std::string labels(800, '\0');
    for (int i{0}; i < 10; ++i) {
        labels += fmt::format("Label {}: {}\n", i + 1, std::string_view(m_labels + 80 * i, 80));
    }

    return fmt::format("Shape (columns, rows, sections): {}\n"
                       "Pixel size (columns, rows, sections): {}\n"
                       "MRC data mode: {} ({})\n"
                       "Bit depth: {}\n"
                       "Extended headers: {} bytes\n"
                       "{}",
                       m_shape->toString(), getPixelSize().toString(),
                       *m_mode, IO::toString(m_data_type),
                       IO::bytesPerElement(m_data_type) * 8,
                       *m_extended_bytes_nb, labels);
}


Noa::errno_t Noa::MRCFile::open(std::ios_base::openmode mode, bool wait) {
    if (close())
        return m_state;

    uint32_t iterations = wait ? 10 : 5;
    size_t time_to_wait = wait ? 3000 : 10;

    bool exists = OS::existsFile(m_path, m_state);
    bool overwrite = mode & std::ios::trunc || !(mode & std::ios::in);
    if (mode & std::ios::out) {
        if (exists) {
            m_state = OS::backup(m_path, !overwrite);
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
    m_state = Errno::fail_open;
    return m_state;
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


Noa::errno_t Noa::MRCFile::writeAll(float* data) {
    if (m_state)
        return m_state;

    m_fstream->seekg(getOffset_());
    if (m_fstream)
        m_state = IO::writeFloat(*m_fstream, data, getShape().prod(), m_data_type);
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
        m_state = IO::writeFloat(*m_fstream, data, elements_to_read, m_data_type);
    return m_state;
}


#undef RC
