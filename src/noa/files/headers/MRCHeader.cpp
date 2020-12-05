#include "noa/files/headers/MRCHeader.h"


void Noa::Header::MRCHeader::link_() {
    // C-style cast, reinterpret_cast and memcpy should all
    // be optimized into the same assembly code here.
    m_shape = reinterpret_cast<Int3<int>*>(m_data);
    m_mode = reinterpret_cast<int*>(m_data + 12);
    m_shape_sub = reinterpret_cast<Int3<int>*>(m_data + 16);
    m_shape_grid = reinterpret_cast<Int3<int>*>(m_data + 28);
    m_shape_cell = reinterpret_cast<Float3<float>*>(m_data + 40);
    m_angles = reinterpret_cast<Float3<float>*>(m_data + 52);
    m_map_order = reinterpret_cast<Int3<int>*>(m_data + 64);

    m_min = reinterpret_cast<float*>(m_data + 76);
    m_max = reinterpret_cast<float*>(m_data + 80);
    m_mean = reinterpret_cast<float*>(m_data + 84);

    m_space_group = reinterpret_cast<int*>(m_data + 88);
    m_extended_bytes_nb = reinterpret_cast<int*>(m_data + 92);
    m_extra00 = m_data + 96;
    m_extended_type = m_data + 104;
    m_nversion = reinterpret_cast<int*>(m_data + 108);

    m_imod_stamp = reinterpret_cast<int*>(m_data + 152);
    m_imod_flags = reinterpret_cast<int*>(m_data + 156);

    m_origin = reinterpret_cast<Float3<float>*>(m_data + 196);
    m_cmap = m_data + 208;
    m_stamp = m_data + 212;
    m_rms = reinterpret_cast<float*>(m_data + 216);

    m_nb_labels = reinterpret_cast<int*>(m_data + 220);
    m_labels = m_data + 224;
}


Noa::errno_t Noa::Header::MRCHeader::validate_() {
    if (*m_shape < 0 && *m_shape_cell < 0 && *m_shape_grid < 0)
        return Errno::invalid_data;

    // Layout.
    if (*m_mode == 0) {
        m_layout |= IO::Layout::byte;
        if (*m_imod_stamp == 1146047817 && *m_imod_flags & 1)
            m_layout |= IO::Layout::ubyte;
    } else if (*m_mode != 2) {
        m_layout |= IO::Layout::float32;
    } else if (*m_mode == 1) {
        m_layout |= IO::Layout::int16;
    } else if (*m_mode == 6) {
        m_layout |= IO::Layout::uint16;
    } else if (*m_mode == 16 || *m_mode == 101 ||
               *m_mode == 3 || *m_mode == 4) {
        return Errno::not_supported;
    } else {
        return Errno::invalid_data;
    }

    // Map order: x=1, y=2, z=3 is the only supported order atm.
    if (m_map_order->x == 1) {
        if (m_map_order->y == 3 && m_map_order->z == 2)
            return Errno::not_supported;
        else if (m_map_order->y != 2 || m_map_order->z != 3)
            return Errno::invalid_data;
    } else if (m_map_order->x == 2) {
        if ((m_map_order->y == 1 && m_map_order->z == 3) ||
            (m_map_order->y == 3 && m_map_order->z == 1))
            return Errno::not_supported;
        else
            return Errno::invalid_data;
    } else if (m_map_order->x == 3) {
        if ((m_map_order->y == 1 && m_map_order->z == 2) ||
            (m_map_order->y == 2 && m_map_order->z == 1))
            return Errno::not_supported;
        else
            return Errno::invalid_data;
    } else {
        return Errno::invalid_data;
    }

    if (*m_extended_bytes_nb < 0)
        return Errno::invalid_data;

    // Endianness.
    if (m_data[212] == 68 && m_data[213] == 65 && m_data[214] == 0 && m_data[215] == 0)
        m_is_big_endian = false;
    else if (m_data[212] == 17 && m_data[213] == 17 && m_data[214] == 0 && m_data[215] == 0)
        m_is_big_endian = true;
    else
        return Errno::invalid_data;

    return Errno::good;
}


void Noa::Header::MRCHeader::reset() {
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
        m_data[96 + i] = 0;
    m_extended_type[0] = 'S';
    m_extended_type[1] = 'E';
    m_extended_type[2] = 'R';
    m_extended_type[3] = 'I';
    *m_nversion = 20140;
    for (int i = 0; i < 88; ++i)
        m_extra00[i] = 0;
    *m_origin = 0;
    m_cmap[0] = 'M';
    m_cmap[1] = 'A';
    m_cmap[2] = 'P';
    m_cmap[3] = ' ';
    setEndianness(OS::isBigEndian());
    *m_rms = 0;
    *m_nb_labels = 0;
    for (int i = 0; i < 800; ++i)
        m_labels[i] = ' ';
}


Noa::errno_t Noa::Header::MRCHeader::setLayout(Noa::iolayout_t layout) {
    if (layout & IO::Layout::byte) {
        *m_mode = 0;
    } else if (layout & IO::Layout::float32) {
        *m_mode = 2;
    } else if (layout & IO::Layout::int16) {
        *m_mode = 1;
    } else if (layout & IO::Layout::uint16) {
        *m_mode = 6;
    } else if (layout & IO::Layout::ubyte) {
        *m_mode = 1;
        if (*m_imod_stamp == 1146047817)
            *m_imod_flags &= 1;
    } else if (layout & IO::Layout::int32 || layout & IO::Layout::uint32) {
        return Errno::not_supported;
    } else {
        return Errno::invalid_data;
    }
    m_layout = layout;
    return Errno::good;
}


void Noa::Header::MRCHeader::setEndianness(bool big_endian) {
    if (big_endian) {
        m_data[212] = 68;
        m_data[213] = 65;
        m_data[214] = 0;
        m_data[215] = 0;
        m_is_big_endian = false;
    } else {
        m_data[212] = 17;
        m_data[213] = 17;
        m_data[214] = 0;
        m_data[215] = 0;
        m_is_big_endian = true;
    }
}


std::string Noa::Header::MRCHeader::toString(bool brief) const {
    if (brief)
        return fmt::format("Shape: {}; Pixel size: {}",
                           m_shape->toString(), getPixelSize().toString());

    std::string labels(800, '\0');
    for (uint8_t i{0}; i < 10; ++i) {
        labels += fmt::format("Label {}: {}\n", i + 1, std::string_view(m_labels + 80 * i, 80));
    }

    return fmt::format("Shape (columns, rows, sections): {}\n"
                       "Pixel size (columns, rows, sections): {}\n"
                       "MRC data mode: {} ({})\n"
                       "Bit depth: {}\n"
                       "Extended headers: {} bytes\n"
                       "{}",
                       m_shape->toString(), getPixelSize().toString(),
                       *m_mode, IO::Layout::toString(m_layout), IO::bytesPerElement(m_layout) * 8,
                       *m_extended_bytes_nb, labels);
}
