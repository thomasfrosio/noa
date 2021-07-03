#include <cstring>  // memcpy, memset
#include <thread>   // std::this_thread::sleep_for
#include <chrono>   // std::chrono::milliseconds

#include "noa/common/Math.h"
#include "noa/common/OS.h"
#include "noa/common/Profiler.h"
#include "noa/common/files/MRCFile.h"
#include "noa/common/string/Format.h"

using namespace ::noa;

void MRCFile::readAll(float* to_write) {
    NOA_PROFILE_FUNCTION();
    m_fstream.seekg(getOffset_());
    if (m_fstream.fail())
        NOA_THROW("File: {}. Could not seek to the desired offset ({})", m_path, getOffset_());
    io::readFloat(m_fstream, to_write, getElements(getShape()),
                  m_header.data_type, true, m_header.is_endian_swapped);
}

void MRCFile::readAll(cfloat_t* to_write) {
    NOA_PROFILE_FUNCTION();
    m_fstream.seekg(getOffset_());
    if (m_fstream.fail())
        NOA_THROW("File: {}. Could not seek to the desired offset ({})", m_path, getOffset_());
    io::readComplexFloat(m_fstream, to_write, getElements(getShape()),
                         m_header.data_type, true, m_header.is_endian_swapped);
}

void MRCFile::readSlice(float* to_write, size_t z_pos, size_t z_count) {
    NOA_PROFILE_FUNCTION();
    size_t elements_per_slice = getElementsSlice(getShape());
    size_t elements_to_read = elements_per_slice * z_count;
    size_t bytes_per_slice = elements_per_slice * io::bytesPerElement(m_header.data_type);

    long offset = getOffset_() + static_cast<long>(z_pos * bytes_per_slice);
    m_fstream.seekg(offset);
    if (m_fstream.fail())
        NOA_THROW("File: {}. Could not seek to the desired offset ({})", m_path, offset);
    io::readFloat(m_fstream, to_write, elements_to_read, m_header.data_type, true, m_header.is_endian_swapped);
}

void MRCFile::readSlice(cfloat_t* to_write, size_t z_pos, size_t z_count) {
    NOA_PROFILE_FUNCTION();
    size_t elements_per_slice = getElementsSlice(getShape());
    size_t elements_to_read = elements_per_slice * z_count;
    size_t bytes_per_slice = elements_per_slice * io::bytesPerElement(m_header.data_type);

    long offset = getOffset_() + static_cast<long>(z_pos * bytes_per_slice);
    m_fstream.seekg(offset);
    if (m_fstream.fail())
        NOA_THROW("File: {}. Could not seek to the desired offset ({})", m_path, offset);
    io::readComplexFloat(m_fstream, to_write, elements_to_read, m_header.data_type, true, m_header.is_endian_swapped);
}

void MRCFile::writeAll(const float* to_read) {
    NOA_PROFILE_FUNCTION();
    m_fstream.seekp(getOffset_());
    if (m_fstream.fail())
        NOA_THROW("File: {}. Could not seek to the desired offset ({})", m_path, getOffset_());
    io::writeFloat(to_read, m_fstream, getElements(getShape()),
                   m_header.data_type, true, m_header.is_endian_swapped);
}

void MRCFile::writeAll(const cfloat_t* to_read) {
    NOA_PROFILE_FUNCTION();
    m_fstream.seekp(getOffset_());
    if (m_fstream.fail())
        NOA_THROW("File: {}. Could not seek to the desired offset ({})", m_path, getOffset_());
    io::writeComplexFloat(to_read, m_fstream, getElements(getShape()),
                          m_header.data_type, true, m_header.is_endian_swapped);
}

void MRCFile::writeSlice(const float* to_read, size_t z_pos, size_t z_count) {
    NOA_PROFILE_FUNCTION();
    size_t elements_per_slice = getElementsSlice(getShape());
    size_t elements_to_read = elements_per_slice * z_count;
    size_t bytes_per_slice = elements_per_slice * io::bytesPerElement(m_header.data_type);

    long offset = getOffset_() + static_cast<long>(z_pos * bytes_per_slice);
    m_fstream.seekp(offset);
    if (m_fstream.fail())
        NOA_THROW("File: {}. Could not seek to the desired offset ({})", m_path, offset);
    io::writeFloat(to_read, m_fstream, elements_to_read, m_header.data_type, true, m_header.is_endian_swapped);
}

void MRCFile::writeSlice(const cfloat_t* to_read, size_t z_pos, size_t z_count) {
    NOA_PROFILE_FUNCTION();
    size_t elements_per_slice = getElementsSlice(getShape());
    size_t elements_to_read = elements_per_slice * z_count;
    size_t bytes_per_slice = elements_per_slice * io::bytesPerElement(m_header.data_type);

    long offset = getOffset_() + static_cast<long>(z_pos * bytes_per_slice);
    m_fstream.seekp(offset);
    if (m_fstream.fail())
        NOA_THROW("File: {}. Could not seek to the desired offset ({})", m_path, offset);
    io::writeComplexFloat(to_read, m_fstream, elements_to_read, m_header.data_type, true, m_header.is_endian_swapped);
}

void MRCFile::setDataType(io::DataType data_type) {
    // In reading mode, changing the data type will have no effect, so let it pass.
    if (m_open_mode & io::READ && m_open_mode & io::WRITE && m_open_mode & io::BINARY)
        NOA_THROW("File: {}. Cannot change the data type in non-overwriting mode", m_path);

    switch (data_type) {
        case io::DataType::FLOAT32:
        case io::DataType::BYTE:
        case io::DataType::UBYTE:
        case io::DataType::INT16:
        case io::DataType::UINT16:
        case io::DataType::CFLOAT32:
        case io::DataType::CINT16:
            m_header.data_type = data_type;
            break;
        default:
            NOA_THROW("File: {}. Cannot change the data type. "
                      "Should be FLOAT32, CFLOAT32, INT16, UINT16, CINT16, BYTE or UBYTE, got {}",
                      m_path, data_type);
    }
}

std::string MRCFile::describe(bool brief) const {
    if (brief)
        return string::format("Shape: {}; Pixel size: {}", m_header.shape, m_header.pixel_size);

    return string::format("Format: MRC File\n"
                          "Shape (columns, rows, sections): {}\n"
                          "Pixel size (columns, rows, sections): {}\n"
                          "Data type: {}\n"
                          "Labels: {}\n"
                          "Extended headers: {} bytes",
                          m_header.shape,
                          m_header.pixel_size,
                          m_header.data_type,
                          m_header.nb_labels,
                          m_header.extended_bytes_nb);
}

void MRCFile::open_(uint open_mode) {
    NOA_PROFILE_FUNCTION();
    close();

    bool overwrite = open_mode & io::TRUNC || !(open_mode & io::READ);
    bool exists;
    try {
        exists = os::existsFile(m_path);
        if (open_mode & io::WRITE) {
            if (exists)
                os::backup(m_path, overwrite);
            else if (overwrite)
                os::mkdir(m_path.parent_path());
        }
    } catch (...) {
        NOA_THROW("File: {}. OS failure when trying to open the file", m_path);
    }

    m_open_mode = open_mode | io::BINARY;
    m_open_mode &= ~(io::APP | io::ATE);

    for (uint32_t it{0}; it < 5; ++it) {
        m_fstream.open(m_path, io::toIOSBase(m_open_mode));
        if (m_fstream.is_open()) {
            if (exists && !overwrite) /* case 1 or 2 */
                readHeader_();
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    NOA_THROW("File: {}. Failed to open the file", m_path);
}

void MRCFile::readHeader_() {
    NOA_PROFILE_FUNCTION();
    char buffer[1024];
    m_fstream.seekg(0);
    m_fstream.read(buffer, 1024);
    if (m_fstream.fail())
        NOA_THROW("File: {}. File stream error. Could not read the header", m_path);

    // Endianness.
    // Most software use 68-65, but the CCPEM standard is using 68-68... ?
    char stamp[4];
    std::memcpy(&stamp, buffer + 212, 4);
    if ((stamp[0] == 68 && stamp[1] == 65 && stamp[2] == 0 && stamp[3] == 0) ||
        (stamp[0] == 68 && stamp[1] == 68 && stamp[2] == 0 && stamp[3] == 0)) /* little */
        m_header.is_endian_swapped = os::isBigEndian();
    else if (stamp[0] == 17 && stamp[1] == 17 && stamp[2] == 0 && stamp[3] == 0) /* big */
        m_header.is_endian_swapped = !os::isBigEndian();
    else
        NOA_THROW("File: {}. Invalid data. Endianness was not recognized."
                  "Should be [68,65,0,0], [68,68,0,0] or [17,17,0,0], got [{},{},{},{}]",
                  m_path, stamp[0], stamp[1], stamp[2], stamp[3]);

    // If data is swapped, some parts of the buffer need to be swapped back.
    if (m_header.is_endian_swapped)
        swapHeader_(buffer);

    // Read & Write mode: save the buffer.
    if (m_open_mode & io::WRITE) {
        if (!m_header.buffer)
            m_header.buffer = std::make_unique<char[]>(1024);
        std::memcpy(m_header.buffer.get(), buffer, 1024);
    }

    // Set the header variables according to what was in the file.
    int32_t mode, imod_stamp, imod_flags, space_group;
    int32_t grid_size[3], order[3];
    float cell_size[3];

    // TODO: gcc -03 gives a -Wstringop-overflow warning, saying that 8 bytes are copied? Likely false positive.
    //       Could be replaced by std::memcpy(&m_header.shape, buffer, 12); but it is assuming no padding, which is
    //       fine since Int3<int32_t> is just a struct with 3 ints...
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
    // 212-216: stamp.
    std::memcpy(&m_header.stddev, buffer + 216, 4);
    std::memcpy(&m_header.nb_labels, buffer + 220, 4);
    //224-1024: labels.

    // Pixel size.
    m_header.pixel_size = float3_t(cell_size) / float3_t(m_header.shape);
    if (any(m_header.shape < 1)) {
        NOA_THROW("File: {}. Invalid data. Shape should be greater than zero, got {}",
                  m_path, m_header.shape, m_header.pixel_size, m_header.extended_bytes_nb);
    } else if (grid_size[0] != m_header.shape.x ||
               grid_size[1] != m_header.shape.y ||
               grid_size[2] != m_header.shape.z) {
        NOA_THROW("File: {}. Invalid data. Grid size should be equal to the shape (nx, ny, nz), "
                  "got grid:({},{},{}), shape:{}", m_path, grid_size[0], grid_size[1], grid_size[2], m_header.shape);
    } else if (any(m_header.pixel_size < 0.f)) {
        NOA_THROW("File: {}. Invalid data. Pixel size should not be negative, got {}",
                  m_path, m_header.pixel_size);
    } else if (m_header.extended_bytes_nb < 0) {
        NOA_THROW("File: {}. Invalid data. Extended header size should be positive, got {}",
                  m_path, m_header.extended_bytes_nb);
    }

    // Data type.
    if (mode == 0) {
        if (imod_stamp == 1146047817 && imod_flags & 1)
            m_header.data_type = io::DataType::UBYTE;
        else
            m_header.data_type = io::DataType::BYTE;
    } else if (mode == 2) {
        m_header.data_type = io::DataType::FLOAT32;
    } else if (mode == 1) {
        m_header.data_type = io::DataType::INT16;
    } else if (mode == 6) {
        m_header.data_type = io::DataType::UINT16;
    } else if (mode == 4) {
        m_header.data_type = io::DataType::CFLOAT32;
    } else if (mode == 3) {
        m_header.data_type = io::DataType::CINT16;
    } else if (mode == 16 || mode == 101) {
        NOA_THROW("File: {}. MRC mode {} is not supported", m_path, mode);
    } else {
        NOA_THROW("File: {}. Invalid data. MRC mode not recognized, got {}", m_path, mode);
    }

    // Map order: x=1, y=2, z=3 is the only supported order.
    int3_t tmp_order(order);
    if (all(tmp_order != int3_t(1, 2, 3))) {
        if (any(tmp_order < 1) || any(tmp_order > 3) || math::sum(tmp_order) != 6)
            NOA_THROW("File: {}. Invalid data. Map order should be (1,2,3), got {}", m_path, tmp_order);
        NOA_THROW("File: {}. Map order {} is not supported. Only (1,2,3) is supported", m_path, tmp_order);
    }

    // Space group.
    if (space_group != 0 && space_group != 1) {
        if (space_group == 401)
            NOA_THROW("File: {}. Space group 401 is not supported. Should be 0 or 1", m_path);
        NOA_THROW("File: {}. Invalid data. Space group should be 0 or 1, got {}", m_path, space_group);
    }
}

void MRCFile::close_() {
    NOA_PROFILE_FUNCTION();
    if (!m_fstream.is_open())
        return;

    // Writing mode: the header should be updated before closing the file.
    if (m_open_mode & io::WRITE) {
        // Writing & reading mode: the instance didn't create the file,
        // the header was saved by readHeader_().
        if (m_open_mode & io::READ) {
            writeHeader_(m_header.buffer.get());
        } else {
            char buffer[1024];
            defaultHeader_(buffer);
            writeHeader_(buffer);
        }
    }
    m_fstream.close();
    if (m_fstream.fail())
        NOA_THROW("File: {}. File stream error. Could not close the file", m_path);
}

void MRCFile::defaultHeader_(char* buffer) {
    NOA_PROFILE_FUNCTION();
    std::memset(buffer, 0, 1024); // Set everything to 0.

    // Set the unused flags which do not default to 0.
    // The used bytes will be set before closing the file by writeHeader_().
    float angles[3] = {90, 90, 90};
    std::memcpy(buffer + 52, angles, 12);

    int32_t order[3] = {1, 2, 3};
    std::memcpy(buffer + 64, order, 12);

    buffer[104] = 'S';
    buffer[105] = 'E';
    buffer[106] = 'R';
    buffer[107] = 'I';

    buffer[208] = 'M';
    buffer[209] = 'A';
    buffer[210] = 'P';
    buffer[211] = ' ';

    // With new data, the endianness is always set to the endianness of the CPU.
    if (os::isBigEndian()) {
        buffer[212] = 17;
        buffer[213] = 17;
        buffer[214] = 0;
        buffer[215] = 0;
    } else {
        buffer[212] = 68;
        buffer[213] = 65;
        buffer[214] = 0;
        buffer[215] = 0;
    }
}

void MRCFile::writeHeader_(char* buffer) {
    NOA_PROFILE_FUNCTION();
    // Data type.
    int32_t mode{}, imod_stamp{0}, imod_flags{0};
    switch (m_header.data_type) {
        case io::DataType::FLOAT32:
            mode = 2;
            break;
        case io::DataType::BYTE:
            mode = 0;
            break;
        case io::DataType::INT16:
            mode = 1;
            break;
        case io::DataType::UINT16:
            mode = 6;
            break;
        case io::DataType::CFLOAT32:
            mode = 4;
            break;
        case io::DataType::CINT16:
            mode = 3;
            break;
        case io::DataType::UBYTE:
            mode = 0;
            imod_stamp = 1146047817;
            imod_flags &= 1;
            break;
        default:
            NOA_THROW("The data type is not supported. Got {}", m_header.data_type);
    }

    // Pixel size.
    float3_t cell_size(m_header.shape);
    cell_size *= m_header.pixel_size; // can be 0.

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
    std::memcpy(buffer + 76, &m_header.min, 4);
    std::memcpy(buffer + 80, &m_header.max, 4);
    std::memcpy(buffer + 84, &m_header.mean, 4);
    // 88-92: space group -> 0 or unchanged.
    std::memcpy(buffer + 92, &m_header.extended_bytes_nb, 4); // 0 or unchanged.
    // 96-98: creatid -> 0 or unchanged.
    // 98-104: extra data -> 0 or unchanged.
    // 104-108: extType -> "SERI" or unchanged.
    // 108-112: nversion -> 0 or unchanged.
    // 112-128: extra data -> 0 or unchanged.
    // 128-132: nint, nreal -> 0 or unchanged.
    // 132-152: extra data -> 0 or unchanged.
    std::memcpy(buffer + 152, &imod_stamp, 4);
    std::memcpy(buffer + 156, &imod_flags, 4);
    // 160-208: idtype, lens, nd1, nd2, vd1, vd2, tiltangles, origin(x,y,z) -> 0 or unchanged.
    // 208-212: cmap -> "MAP " or unchanged.
    // 212-216: stamp -> [68,65,0,0] or [17,17,0,0], or unchanged.
    std::memcpy(buffer + 216, &m_header.stddev, 4);
    std::memcpy(buffer + 220, &m_header.nb_labels, 4); // 0 or unchanged.
    //224-1024: labels -> 0 or unchanged.

    // Swap back the header to its original endianness.
    if (m_header.is_endian_swapped)
        swapHeader_(buffer);

    // Write the buffer.
    m_fstream.seekp(0);
    m_fstream.write(buffer, 1024);
    if (m_fstream.fail())
        NOA_THROW("File: {}. File stream error. Could not write the header before closing the file", m_path);
}
