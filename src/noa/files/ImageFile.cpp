#include "noa/files/ImageFile.h"


Noa::errno_t Noa::ImageFile::readAll(float* data) {
    if (m_state)
        return m_state;
    m_fstream->seekg(static_cast<long>(header.getOffset()));
    if (m_fstream)
        m_state = IO::readFloat(*m_fstream, data, header.getShape().prod(),
                                header.getLayout(), true, header.isSwapped());
    return m_state;
}


Noa::errno_t Noa::ImageFile::readSlice(float* data, size_t z_pos, size_t z_count) {
    if (m_state)
        return m_state;

    IO::Layout io_layout = header.getLayout();
    Int3<size_t> shape = header.getShape();
    size_t elements_per_slice = shape.prodSlice();
    size_t elements_to_read = elements_per_slice * z_count;
    size_t bytes_per_slice = elements_per_slice * IO::bytesPerElement(io_layout);

    m_fstream->seekg(static_cast<long>(header.getOffset() + z_pos * bytes_per_slice));
    if (m_fstream)
        m_state = IO::readFloat(*m_fstream, data, elements_to_read,
                                io_layout, true, header.isSwapped());
    return m_state;
}


Noa::errno_t Noa::ImageFile::writeAll(float* data) {
    if (m_state)
        return m_state;

    m_fstream->seekg(static_cast<long>(header.getOffset()));
    if (m_fstream)
        m_state = IO::writeFloat(*m_fstream, data, header.getShape().prod(), header.getLayout());
    return m_state;
}


Noa::errno_t Noa::ImageFile::writeSlice(float* data, size_t z_pos, size_t z_count) {
    if (m_state)
        return m_state;

    IO::Layout io_layout = header.getLayout();
    Int3<size_t> shape = header.getShape();
    size_t elements_per_slice = shape.prodSlice();
    size_t elements_to_read = elements_per_slice * z_count;
    size_t bytes_per_slice = elements_per_slice * IO::bytesPerElement(io_layout);

    m_fstream->seekg(static_cast<long>(header.getOffset() + z_pos * bytes_per_slice));
    if (m_fstream)
        m_state = IO::writeFloat(*m_fstream, data, elements_to_read, io_layout);
    return m_state;
}
