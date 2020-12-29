#include "noa/files/MRCFile.h"


Noa::errno_t Noa::MRCFile::readAll(float* data) {
    if (m_state)
        return m_state;
    m_fstream->seekg(getOffset_());
    if (m_fstream)
        m_state = IO::readFloat(*m_fstream, data, getShape().prod(),
                                m_header.data_type, true, m_header.is_endian_swapped);
    return m_state;
}


Noa::errno_t Noa::MRCFile::readSlice(float* data, size_t z_pos, size_t z_count) {
    if (m_state)
        return m_state;

    Int3<size_t> shape = getShape();
    size_t elements_per_slice = shape.prodSlice();
    size_t elements_to_read = elements_per_slice * z_count;
    size_t bytes_per_slice = elements_per_slice * IO::bytesPerElement(m_header.data_type);

    m_fstream->seekg(getOffset_() + static_cast<long>(z_pos * bytes_per_slice));
    if (m_fstream)
        m_state = IO::readFloat(*m_fstream, data, elements_to_read,
                                m_header.data_type, true, m_header.is_endian_swapped);
    return m_state;
}


Noa::errno_t Noa::MRCFile::writeAll(float* data) {
    if (m_state)
        return m_state;

    m_fstream->seekg(getOffset_());
    if (m_fstream)
        m_state = IO::writeFloat(data, *m_fstream, getShape().prod(), m_header.data_type,
                                 true, m_header.is_endian_swapped);
    return m_state;
}


Noa::errno_t Noa::MRCFile::writeSlice(float* data, size_t z_pos, size_t z_count) {
    if (m_state)
        return m_state;

    Int3<size_t> shape = getShape();
    size_t elements_per_slice = shape.prodSlice();
    size_t elements_to_read = elements_per_slice * z_count;
    size_t bytes_per_slice = elements_per_slice * IO::bytesPerElement(m_header.data_type);

    m_fstream->seekg(getOffset_() + static_cast<long>(z_pos * bytes_per_slice));
    if (m_fstream)
        m_state = IO::writeFloat(data, *m_fstream, elements_to_read, m_header.data_type,
                                 true, m_header.is_endian_swapped);
    return m_state;
}


Noa::errno_t Noa::MRCFile::setDataType(IO::DataType data_type) {
    // Check for the specific case of setting the data type in read or non-overwriting mode.
    if (!(m_open_mode & std::ios::trunc || !(m_open_mode & std::ios::in)))
        return Errno::set(m_state, Errno::not_supported);

    if (data_type == IO::DataType::float32 ||
        data_type == IO::DataType::byte || data_type == IO::DataType::ubyte ||
        data_type == IO::DataType::int16 || data_type == IO::DataType::uint16)
        m_header.data_type = data_type;
    else
        Errno::set(m_state, Errno::not_supported);
    return m_state;
}


std::string Noa::MRCFile::toString(bool brief) const {
    if (brief)
        return fmt::format("Shape: {}; Pixel size: {}", m_header.shape, m_header.pixel_size);

    std::string labels;
    for (int i{0}; i < m_header.nb_labels; ++i) {
        labels += fmt::format("Label {}: {}\n", i,
                              std::string_view(m_header.buffer.get() + 224 + 80 * i, 80));
    }

    return fmt::format("Shape (columns, rows, sections): {}\n"
                       "Pixel size (columns, rows, sections): {}\n"
                       "Data type: {}\n"
                       "Bit depth: {}\n"
                       "Extended headers: {} bytes\n"
                       "{}",
                       m_header.shape,
                       m_header.pixel_size,
                       IO::toString(m_header.data_type),
                       IO::bytesPerElement(m_header.data_type) * 8,
                       m_header.extended_bytes_nb, labels);
}


Noa::errno_t Noa::MRCFile::open_(std::ios_base::openmode mode, bool wait) {
    if (close())
        return m_state;

    uint32_t iterations = wait ? 10 : 5;
    size_t time_to_wait = wait ? 3000 : 10;

    bool exists = OS::existsFile(m_path, m_state);
    bool overwrite = mode & std::ios::trunc || !(mode & std::ios::in);
    if (mode & std::ios::out) {
        if (exists)
            Errno::set(m_state, OS::backup(m_path, overwrite));
        else if (overwrite)
            Errno::set(m_state, OS::mkdir(m_path.parent_path()));
        if (m_state)
            return m_state;
    }

    m_open_mode = mode | std::ios::binary;
    m_open_mode &= ~(std::ios::app | std::ios::ate);

    for (uint32_t it{0}; it < iterations; ++it) {
        m_fstream->open(m_path, m_open_mode);
        if (m_fstream->is_open()) {
            if (exists && !overwrite)
                readHeader_();
            else
                initHeader_();
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


void Noa::MRCFile::initHeader_() const {
    char* buffer = m_header.buffer.get();
    std::memset(buffer, 0, 1024); // Set everything to 0.

    // Set the unused flags which do not default to 0.
    // The used bytes will be set before closing the file.
    float angles[3] = {90, 90, 90};
    std::memcpy(buffer + 52, angles, 12);

    int32_t order[3] = {1, 2, 3};
    std::memcpy(buffer + 64, order, 12);

    buffer[104] = 'S';
    buffer[105] = 'E';
    buffer[106] = 'R';
    buffer[107] = 'I';

    buffer[208] = 'C';
    buffer[209] = 'M';
    buffer[210] = 'A';
    buffer[211] = ' ';

    // With new data, the endianness is always set to the endianness of the CPU.
    if (OS::isBigEndian()) {
        buffer[212] = 68;
        buffer[213] = 65;
        buffer[214] = 0;
        buffer[215] = 0;
    } else {
        buffer[212] = 17;
        buffer[213] = 17;
        buffer[214] = 0;
        buffer[215] = 0;
    }
}


Noa::errno_t Noa::MRCFile::readHeader_() {
    char* buffer = m_header.buffer.get();
    m_fstream->seekg(0);
    m_fstream->read(buffer, 1024);
    if (m_fstream->fail())
        return Errno::set(m_state, Errno::fail_read);
    else if (m_state)
        return m_state;

    // Set the header variables according to what was in the file.
    int32_t mode, imod_stamp, imod_flags, space_group;
    int32_t grid_size[3], order[3];
    float cell_size[3];
    char stamp[4];

    std::memcpy(&m_header.shape.x, buffer + 0, 4);
    std::memcpy(&m_header.shape.y, buffer + 4, 4);
    std::memcpy(&m_header.shape.z, buffer + 8, 4);
    std::memcpy(&mode, buffer + 12, 4);
    // 16-24: sub-volume (nxstart, nystart, nzstart).
    std::memcpy(&grid_size, buffer + 28, 12);
    std::memcpy(&cell_size, buffer + 40, 12);
    // 52-64: alpha, beta, gamma.
    std::memcpy(&order, buffer + 64, 12);
    std::memcpy(&m_header.min, buffer + 76, 4);
    std::memcpy(&m_header.max, buffer + 80, 4);
    std::memcpy(&m_header.mean, buffer + 84, 4);
    std::memcpy(&space_group, buffer + 88, 4);
    std::memcpy(&m_header.extended_bytes_nb, buffer + 92, 4);
    // 96-98: creatid, extra data, extType, nversion, extra data, nint, nreal, extra data.
    std::memcpy(&imod_stamp, buffer + 152, 4);
    std::memcpy(&imod_flags, buffer + 156, 4);
    // 160-208: idtype, lens, nd1, nd2, vd1, vd2, tiltangles, origin(x,y,z).
    // 208-212: cmap.
    std::memcpy(&stamp, buffer + 212, 4);
    std::memcpy(&m_header.rms, buffer + 216, 4);
    std::memcpy(&m_header.nb_labels, buffer + 220, 4);
    //224-1024: labels.

    // Pixel size.
    m_header.pixel_size = Float3<float>(cell_size) / Float3<float>(m_header.shape);
    if (m_header.shape < 1 || m_header.pixel_size <= 0 || m_header.extended_bytes_nb < 0) {
        m_state = Errno::invalid_data;
        return m_state;
    } else if (grid_size[0] != m_header.shape.x ||
               grid_size[1] != m_header.shape.y ||
               grid_size[2] != m_header.shape.z) {
        m_state = Errno::not_supported;
        return m_state;
    }

    // Data type.
    if (mode == 0) {
        if (imod_stamp == 1146047817 && imod_flags & 1)
            m_header.data_type = IO::DataType::ubyte;
        else
            m_header.data_type = IO::DataType::byte;
    } else if (mode == 2) {
        m_header.data_type = IO::DataType::float32;
    } else if (mode == 1) {
        m_header.data_type = IO::DataType::int16;
    } else if (mode == 6) {
        m_header.data_type = IO::DataType::uint16;
    } else if (mode == 16 || mode == 101 || mode == 3 || mode == 4) {
        m_state = Errno::not_supported;
        return m_state;
    } else {
        m_state = Errno::invalid_data;
        return m_state;
    }

    // Map order: x=1, y=2, z=3 is the only supported order.
    Int3<int32_t> tmp_order(order);
    if (tmp_order != Int3<int>(1, 2, 3)) {
        if (tmp_order < 1 || tmp_order > 3 || tmp_order.sum() != 6) {
            m_state = Errno::invalid_data;
            return m_state;
        }
    } else {
        m_state = Errno::not_supported;
        return m_state;
    }

    // Space group.
    if (space_group != 0 && space_group != 1) {
        if (space_group == 401) {
            m_state = Errno::not_supported;
            return m_state;
        }
        m_state = Errno::invalid_argument;
        return m_state;
    }

    // Endianness.
    if (stamp[0] == 68 && stamp[1] == 65 && stamp[2] == 0 && stamp[3] == 0) /* little */
        m_header.is_endian_swapped = OS::isBigEndian();
    else if (stamp[0] == 17 && stamp[1] == 17 && stamp[2] == 0 && stamp[3] == 0) /* big */
        m_header.is_endian_swapped = !OS::isBigEndian();
    else
        m_state = Errno::invalid_data;
    return m_state;
}


void Noa::MRCFile::writeHeader_() {
    char* buffer = m_header.buffer.get();

    // Data type.
    int32_t mode{}, imod_stamp{0}, imod_flags{0};
    if (m_header.data_type == IO::DataType::float32)
        mode = 2;
    else if (m_header.data_type == IO::DataType::byte)
        mode = 0;
    else if (m_header.data_type == IO::DataType::int16)
        mode = 1;
    else if (m_header.data_type == IO::DataType::uint16)
        mode = 6;
    else if (m_header.data_type == IO::DataType::ubyte) {
        mode = 0;
        imod_stamp = 1146047817;
        imod_flags &= 1;
    }

    // Pixel size.
    Float3<float> cell_size(m_header.shape);
    cell_size *= m_header.pixel_size;

    // Updating the buffer.
    std::memcpy(buffer + 0, &m_header.shape.x, 4);
    std::memcpy(buffer + 4, &m_header.shape.y, 4);
    std::memcpy(buffer + 8, &m_header.shape.z, 4);
    std::memcpy(buffer + 12, &mode, 4);
    // 16-24: sub-volume (nxstart, nystart, nzstart) -> 0 or unchanged.
    std::memcpy(buffer + 28, &m_header.shape.x, 4);
    std::memcpy(buffer + 32, &m_header.shape.y, 4);
    std::memcpy(buffer + 36, &m_header.shape.z, 4);
    std::memcpy(buffer + 40, &cell_size.x, 4);
    std::memcpy(buffer + 44, &cell_size.y, 4);
    std::memcpy(buffer + 48, &cell_size.z, 4);
    // 52-64: alpha, beta, gamma -> 90,90,90 or unchanged.
    // 64-76: mapc, mapr, maps -> 1,2,3 (anything else is not supported).
    std::memcpy(&m_header.min, buffer + 76, 4);
    std::memcpy(&m_header.max, buffer + 80, 4);
    std::memcpy(&m_header.mean, buffer + 84, 4);
    // 88-92: space group -> 0 or unchanged.
    std::memcpy(&m_header.extended_bytes_nb, buffer + 92, 4); // 0 or unchanged.
    // 96-98: creatid -> 0 or unchanged.
    // 98-104: extra data -> 0 or unchanged.
    // 104-108: extType -> "SERI" or unchanged.
    // 108-112: nversion -> 0 or unchanged.
    // 112-128: extra data -> 0 or unchanged.
    // 128-132: nint, nreal -> 0 or unchanged.
    // 132-152: extra data -> 0 or unchanged.
    std::memcpy(&imod_stamp, buffer + 152, 4);
    std::memcpy(&imod_flags, buffer + 156, 4);
    // 160-208: idtype, lens, nd1, nd2, vd1, vd2, tiltangles, origin(x,y,z) -> 0 or unchanged.
    // 208-212: cmap -> "MAP " or unchanged.
    // 212-216: stamp -> [68,65,0,0] or [17,17,0,0], or unchanged.
    std::memcpy(&m_header.rms, buffer + 216, 4);
    std::memcpy(&m_header.nb_labels, buffer + 220, 4); // 0 or unchanged.
    //224-1024: labels -> 0 or unchanged.

    // Write the buffer.
    m_fstream->seekp(0);
    m_fstream->write(buffer, 1024);
    if (m_fstream->fail())
        Errno::set(m_state, Errno::fail_write);
}
