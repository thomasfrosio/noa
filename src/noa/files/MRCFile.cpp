#include "MRCFile.h"


void Noa::MRCHeader::setDefault() {
    *size = 0;
    *mode = Type::float32;
    *size_sub = 0;
    *size_grid = 1;
    *size_cell = 1.f;
    *angles = 90.f;
    *map_order = {1, 2, 3};
    *min = 0;
    *max = -1;
    *mean = -2;
    *space_group = 1;
    *extended_bytes_nb = 0;
    for (int i = 0; i < 8; ++i)
        m_header[96 + i] = 0;
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
    setStamp();
    *rms = 0;
    *nb_labels = 0;
    for (int i = 0; i < 800; ++i)
        labels[i] = ' ';
}


void Noa::MRCHeader::setStamp() {
    if (OS::isBigEndian()) {
        m_header[212] = 68;
        m_header[213] = 65;
        m_header[214] = 0;
        m_header[215] = 0;
    } else {
        m_header[212] = 17;
        m_header[213] = 17;
        m_header[214] = 0;
        m_header[215] = 0;
    }
}


void Noa::MRCHeader::link_() {
    // C-style cast, reinterpret_cast and memcpy should all
    // be optimized into the same assembly code here.
    size = reinterpret_cast<Int3<int>*>(m_header);
    mode = reinterpret_cast<int*>(m_header + 12);
    size_sub = reinterpret_cast<Int3<int>*>(m_header + 16);
    size_grid = reinterpret_cast<Int3<int>*>(m_header + 28);
    size_cell = reinterpret_cast<Float3<float>*>(m_header + 40);
    angles = reinterpret_cast<Float3<float>*>(m_header + 52);
    map_order = reinterpret_cast<Int3<int>*>(m_header + 64);

    min = reinterpret_cast<float*>(m_header + 76);
    max = reinterpret_cast<float*>(m_header + 80);
    mean = reinterpret_cast<float*>(m_header + 84);

    space_group = reinterpret_cast<int*>(m_header + 88);
    extended_bytes_nb = reinterpret_cast<int*>(m_header + 92);
    extra00 = reinterpret_cast<char*>(m_header + 96);
    extended_type = reinterpret_cast<char*>(m_header + 104);
    nversion = reinterpret_cast<int*>(m_header + 108);

    imod_stamp = reinterpret_cast<int*>(m_header + 152);
    imod_flags = reinterpret_cast<int*>(m_header + 156);

    origin = reinterpret_cast<Float3<float>*>(m_header + 196);
    cmap = reinterpret_cast<char*>(m_header + 208);
    stamp = reinterpret_cast<char*>(m_header + 212);
    rms = reinterpret_cast<float*>(m_header + 216);

    nb_labels = reinterpret_cast<int*>(m_header + 220);
    labels = reinterpret_cast<char*>(m_header + 224);
}
