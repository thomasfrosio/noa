#include "noa/files/headers/MRCHeader.h"


void Noa::Header::MRCHeader::link_() {
    // C-style cast, reinterpret_cast and memcpy should all
    // be optimized into the same assembly code here.
    shape = reinterpret_cast<Int3<int>*>(m_data);
    mode = reinterpret_cast<int*>(m_data + 12);
    shape_sub = reinterpret_cast<Int3<int>*>(m_data + 16);
    shape_grid = reinterpret_cast<Int3<int>*>(m_data + 28);
    shape_cell = reinterpret_cast<Float3<float>*>(m_data + 40);
    angles = reinterpret_cast<Float3<float>*>(m_data + 52);
    map_order = reinterpret_cast<Int3<int>*>(m_data + 64);

    min = reinterpret_cast<float*>(m_data + 76);
    max = reinterpret_cast<float*>(m_data + 80);
    mean = reinterpret_cast<float*>(m_data + 84);

    space_group = reinterpret_cast<int*>(m_data + 88);
    extended_bytes_nb = reinterpret_cast<int*>(m_data + 92);
    extra00 = m_data + 96;
    extended_type = m_data + 104;
    nversion = reinterpret_cast<int*>(m_data + 108);

    imod_stamp = reinterpret_cast<int*>(m_data + 152);
    imod_flags = reinterpret_cast<int*>(m_data + 156);

    origin = reinterpret_cast<Float3<float>*>(m_data + 196);
    cmap = m_data + 208;
    stamp = m_data + 212;
    rms = reinterpret_cast<float*>(m_data + 216);

    nb_labels = reinterpret_cast<int*>(m_data + 220);
    labels = m_data + 224;
}


Noa::errno_t Noa::Header::MRCHeader::validate_() {
    if (*shape < 0 && *shape_cell < 0 && *shape_grid < 0)
        return Errno::invalid_data;

    // Type.
    if (*mode == 0) {
        m_io_layout |= IO::Layout::byte;
        if (*imod_stamp == 1146047817 && *imod_flags & 1)
            m_io_layout |= IO::Layout::ubyte;
    } else if (*mode != 2) {
        m_io_layout |= IO::Layout::float32;
    } else if (*mode == 1) {
        m_io_layout |= IO::Layout::int16;
    } else if (*mode == 6) {
        m_io_layout |= IO::Layout::uint16;
    } else if (*mode == 16 || *mode == 101 ||
               *mode == 3 || *mode == 4) {
        return Errno::not_supported;
    } else {
        return Errno::invalid_data;
    }

    // Map order
    if (map_order->x == 1) {
        if (map_order->y == 3 && map_order->z == 2)
            return Errno::not_supported;
        else if (map_order->y != 2 || map_order->z != 3) /* x=1,y=2,z=3 is the only supported order */
            return Errno::invalid_data;
    } else if (map_order->x == 2) {
        if ((map_order->y == 1 && map_order->z == 3) || (map_order->y == 3 && map_order->z == 1))
            return Errno::not_supported;
        else
            return Errno::invalid_data;
    } else if (map_order->x == 3) {
        if ((map_order->y == 1 && map_order->z == 2) || (map_order->y == 2 && map_order->z == 1))
            return Errno::not_supported;
        else
            return Errno::invalid_data;
    } else {
        return Errno::invalid_data;
    }

    if (*extended_bytes_nb < 0)
        return Errno::invalid_data;

    // Endianness.
    if (((m_data[212] == 68 && m_data[213] == 65 &&
          m_data[214] == 0 && m_data[215] == 0) && !OS::isBigEndian()) ||
        ((m_data[212] == 17 && m_data[213] == 17 &&
          m_data[214] == 0 && m_data[215] == 0) && OS::isBigEndian())) {
        m_io_option |= IO::Option::swap_bytes;
    } else {
        return Errno::invalid_data;
    }
    return Errno::good;
}


void Noa::Header::MRCHeader::reset() {
    *shape = 0;
    *mode = 2;
    *shape_sub = 0;
    *shape_grid = 1;
    *shape_cell = 1.f;
    *angles = 90.f;
    *map_order = {1, 2, 3};
    *min = 0;
    *max = -1;
    *mean = -2;
    *space_group = 1;
    *extended_bytes_nb = 0;
    for (int i = 0; i < 8; ++i)
        m_data[96 + i] = 0;
    extended_type[0] = 'S';
    extended_type[1] = 'E';
    extended_type[2] = 'R';
    extended_type[3] = 'I';
    *nversion = 20140;
    for (int i = 0; i < 88; ++i)
        extra00[i] = 0;
    *origin = 0;
    cmap[0] = 'M';
    cmap[1] = 'A';
    cmap[2] = 'P';
    cmap[3] = ' ';
    setEndianness_();
    *rms = 0;
    *nb_labels = 0;
    for (int i = 0; i < 800; ++i)
        labels[i] = ' ';
}


void Noa::Header::MRCHeader::setEndianness_() {
    if (OS::isBigEndian()) {
        m_data[212] = 68;
        m_data[213] = 65;
        m_data[214] = 0;
        m_data[215] = 0;
    } else {
        m_data[212] = 17;
        m_data[213] = 17;
        m_data[214] = 0;
        m_data[215] = 0;
    }
}


Noa::errno_t Noa::Header::MRCHeader::setLayout(Noa::ioflag_t layout)  {
    if (layout & IO::Layout::byte) {
        *mode = 0;
    } else if (layout & IO::Layout::float32) {
        *mode = 2;
    } else if (layout & IO::Layout::int16) {
        *mode = 1;
    } else if (layout & IO::Layout::uint16) {
        *mode = 6;
    } else if (layout & IO::Layout::ubyte) {
        *mode = 1;
        if (*imod_stamp == 1146047817)
            *imod_flags &= 1;
    } else if (layout & IO::Layout::int32 || layout & IO::Layout::uint32) {
        return Errno::not_supported;
    } else {
        return Errno::invalid_data;
    }
    m_io_layout = layout;
    return Errno::good;
}
