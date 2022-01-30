#include <cstring>  // memcpy, memset
#include <thread>   // std::this_thread::sleep_for
#include <chrono>   // std::chrono::milliseconds

#include "noa/Session.h"
#include "noa/common/OS.h"
#include "noa/common/io/header/MRCHeader.h"

namespace noa::io::details {
    void MRCHeader::setShape(size4_t new_shape) {
        if (m_open_mode & OpenMode::READ) {
            if (m_open_mode & OpenMode::WRITE) {
                Session::logger.warn("MRCHeader: changing the shape of the data in "
                                     "read|write mode might corrupt the file");
            } else {
                NOA_THROW("Trying to change the shape of the data in read mode is not allowed. "
                          "Hint: to fix the header of a file, open it in read|write mode");
            }
        }
        m_header.shape = Int4<int32_t>(new_shape).flip();
    }

    void MRCHeader::setDataType(io::DataType data_type) {
        if (m_open_mode & io::READ) {
            if (m_open_mode & OpenMode::WRITE) {
                Session::logger.warn("MRCHeader: changing the data type of the file in "
                                     "read|write mode might corrupt the file");
            } else {
                NOA_THROW("Trying to change the data type of the file in read mode is not allowed. "
                          "Hint: to fix the header of a file, open it in read|write mode");
            }
        }
        switch (data_type) {
            case DataType::UINT4:
            case DataType::INT8:
            case DataType::UINT8:
            case DataType::INT16:
            case DataType::UINT16:
            case DataType::FLOAT16:
            case DataType::FLOAT32:
            case DataType::CFLOAT32:
            case DataType::CINT16:
                m_header.data_type = data_type;
                break;
            default:
                NOA_THROW("Data type {} is not supported", data_type);
        }
    }

    void MRCHeader::setPixelSize(float3_t new_pixel_size) {
        if (m_open_mode & io::READ) {
            if (m_open_mode & OpenMode::WRITE) {
                Session::logger.warn("MRCHeader: changing the pixel size of the file in "
                                     "read|write mode might corrupt the file");
            } else {
                NOA_THROW("Trying to change the pixel size of the file in read mode is not allowed. "
                          "Hint: to fix the header of a file, open it in read|write mode");
            }
        }
        if (all(new_pixel_size >= 0))
            m_header.pixel_size = new_pixel_size.flip();
        else
            NOA_THROW("The pixel size should be positive, got {}", new_pixel_size);
    }

    std::string MRCHeader::infoString(bool brief) const noexcept {
        if (brief)
            return string::format("Shape: {}; Pixel size: {:.3f}", m_header.shape.flip(), m_header.pixel_size.flip());

        return string::format("Format: MRC File\n"
                              "Shape (batches, sections, rows, columns): {}\n"
                              "Pixel size (sections, rows, columns): {:.3f}\n"
                              "Data type: {}\n"
                              "Labels: {}\n"
                              "Extended headers: {} bytes",
                              m_header.shape.flip(),
                              m_header.pixel_size.flip(),
                              m_header.data_type,
                              m_header.nb_labels,
                              m_header.extended_bytes_nb);
    }

    void MRCHeader::open_(const path_t& filename, open_mode_t open_mode) {
        close_();

        bool overwrite = open_mode & io::TRUNC || !(open_mode & io::READ);
        bool exists;
        try {
            exists = os::existsFile(filename);
            if (open_mode & io::WRITE) {
                if (exists)
                    os::backup(filename, overwrite);
                else if (overwrite)
                    os::mkdir(filename.parent_path());
            }
        } catch (...) {
            NOA_THROW("OS failure when trying to open the file");
        }

        m_open_mode = open_mode | io::BINARY;
        m_open_mode &= ~(io::APP | io::ATE);

        for (uint32_t it{0}; it < 5; ++it) {
            m_fstream.open(filename, io::toIOSBase(m_open_mode));
            if (m_fstream.is_open()) {
                if (exists && !overwrite) /* case 1 or 2 */
                    readHeader_();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        m_fstream.clear();
        NOA_THROW("Failed to open the file");
    }

    void MRCHeader::readHeader_() {
        char buffer[1024];
        m_fstream.seekg(0);
        m_fstream.read(buffer, 1024);
        if (m_fstream.fail()) {
            m_fstream.clear();
            NOA_THROW("File stream error. Could not read the header");
        }

        // Endianness.
        // Some software use 68-65, but the CCPEM standard is using 68-68... ?
        char stamp[4];
        std::memcpy(&stamp, buffer + 212, 4);
        if ((stamp[0] == 68 && stamp[1] == 65 && stamp[2] == 0 && stamp[3] == 0) ||
            (stamp[0] == 68 && stamp[1] == 68 && stamp[2] == 0 && stamp[3] == 0)) /* little */
            m_header.is_endian_swapped = isBigEndian();
        else if (stamp[0] == 17 && stamp[1] == 17 && stamp[2] == 0 && stamp[3] == 0) /* big */
            m_header.is_endian_swapped = !isBigEndian();
        else
            NOA_THROW("Invalid data. Endianness was not recognized."
                      "Should be [68,65,0,0], [68,68,0,0] or [17,17,0,0], got [{},{},{},{}]",
                      stamp[0], stamp[1], stamp[2], stamp[3]);

        // If data is swapped, some parts of the buffer need to be swapped back.
        if (m_header.is_endian_swapped)
            swapHeader_(buffer);

        // Read & Write mode: save the buffer.
        if (m_open_mode & OpenMode::WRITE) {
            if (!m_header.buffer)
                m_header.buffer = std::make_unique<char[]>(1024);
            std::memcpy(m_header.buffer.get(), buffer, 1024);
        }

        // Get the header from file.
        int32_t mode, imod_stamp, imod_flags, space_group;
        int3_t shape, grid_size, order;
        float3_t cell_size;
        {
            std::memcpy(shape.get(), buffer, 12);
            std::memcpy(&mode, buffer + 12, 4);
            // 16-24: sub-volume (nxstart, nystart, nzstart).
            std::memcpy(grid_size.get(), buffer + 28, 12);
            std::memcpy(cell_size.get(), buffer + 40, 12);
            // 52-64: alpha, beta, gamma.
            std::memcpy(order.get(), buffer + 64, 12);
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
            std::memcpy(&m_header.std, buffer + 216, 4);
            std::memcpy(&m_header.nb_labels, buffer + 220, 4);
            //224-1024: labels.
        }

        // Set the 4D shape correctly
        const int ndim = shape.flip().ndim();
        if (any(shape < 1))
            NOA_THROW("Invalid data. Logical shape should be greater than zero, got nx,ny,nz:{}", shape);
        if (ndim <= 2) {
            // While not supported by some packages, sometimes it can be useful.
            if (any(grid_size != shape)) {
                NOA_THROW("1D or 2D data detected. The logical shape should be equal to the grid size. "
                          "Got nx,ny,nz:{}, mx,my,mz:{}", shape, grid_size);
            }
            m_header.shape = {shape[0], shape[1], 1, 1};
        } else { // ndim == 3
            if (space_group == 0) { // stack of images
                m_header.shape = {shape[0], shape[1], 1, shape[2]};
                // We should check grid_size[2] == 1, but some packages ignore this (why??), so do nothing for now.
                if (shape[0] != grid_size[0] || shape[1] != grid_size[1]) {
                    NOA_THROW("2D stack of images detected (ndim=3, group=0). The innermost two dimensions of the "
                              "logical shape and the grid size should be equal. Got nx,ny,nz:{}, mx,my,mz:{}",
                              shape, grid_size);
                }
            } else if (space_group == 1) { // volume
                if (any(grid_size != shape)) {
                    NOA_THROW("3D volume detected (ndim=3, group=1). The logical shape should be equal to the "
                              "grid size. Got nx,ny,nz:{}, mx,my,mz:{}", shape, grid_size);
                }
                m_header.shape = {shape[0], shape[1], shape[2], 1};
            } else if (space_group >= 401 && space_group <= 630) { // stack of volume
                // grid_size[2] = secs per vol, shape[2] = total sections
                if (shape[2] % grid_size[2]) {
                    NOA_THROW("3D volume stack detected. The total sections (nz:{}) should be divisible by the "
                              "number of sections per volume (mz:{})", shape[2], grid_size[2]);
                } else if (shape[0] != grid_size[0] || shape[1] != grid_size[1]) {
                    NOA_THROW("3D volume stack detected. The first two dimensions of logical shape and the grid size "
                              "should be equal. Got nx,ny,nz:{}, mx,my,mz:{}", shape, grid_size);
                }
                m_header.shape = {shape[0], shape[1], grid_size[2], shape[2] / grid_size[2]};
            } else {
                NOA_THROW("Data shape is not recognized. Got nx,ny,nz:{}, mx,my,mz:{}, group:",
                          shape, grid_size, space_group);
            }
        }

        // Set the pixel size:
        m_header.pixel_size = cell_size / float3_t(grid_size);
        if (any(m_header.pixel_size < 0))
            NOA_THROW("Invalid data. Pixel size should not be negative, got {}", m_header.pixel_size.flip());

        if (m_header.extended_bytes_nb < 0)
            NOA_THROW("Invalid data. Extended header size should be positive, got {}", m_header.extended_bytes_nb);

        // Convert mode to data type:
        switch (mode) {
            case 0:
                if (imod_stamp == 1146047817 && imod_flags & 1)
                    m_header.data_type = DataType::UINT8;
                else
                    m_header.data_type = DataType::INT8;
                break;
            case 1:
                m_header.data_type = DataType::INT16;
                break;
            case 2:
                m_header.data_type = DataType::FLOAT32;
                break;
            case 3:
                m_header.data_type = DataType::CINT16;
                break;
            case 4:
                m_header.data_type = DataType::CFLOAT32;
                break;
            case 6:
                m_header.data_type = DataType::UINT16;
                break;
            case 12:
                m_header.data_type = DataType::FLOAT16;
                break;
            case 16:
                NOA_THROW("MRC mode 16 is not yet supported");
            case 101:
                m_header.data_type = DataType::UINT4;
                break;
            default:
                NOA_THROW("Invalid data. MRC mode not recognized, got {}", mode);
        }

        // Map order: enforce row-major ordering, i.e. x=1, y=2, z=3.
        // TODO Add more orders, but how? Simplest is to compute and return strides.
        if (all(order != int3_t{1, 2, 3})) {
            if (any(order < 1) || any(order > 3) || math::sum(order) != 6)
                NOA_THROW("Invalid data. Map order should be (1,2,3), got {}", order);
            NOA_THROW("Map order {} is not supported. Only (1,2,3) is supported", order);
        }
    }

    void MRCHeader::close_() {
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
        if (m_fstream.fail()) {
            m_fstream.clear();
            NOA_THROW("File stream error. Could not close the file");
        }
    }

    void MRCHeader::defaultHeader_(char* buffer) {
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
        if (isBigEndian()) {
            buffer[212] = 17; // 16 * 1 + 1
            buffer[213] = 17;
        } else {
            buffer[212] = 68; // 16 * 4 + 4
            buffer[213] = 68;
        }
        buffer[214] = 0;
        buffer[215] = 0;
    }

    void MRCHeader::writeHeader_(char* buffer) {
        // Data type.
        int32_t mode{}, imod_stamp{0}, imod_flags{0};
        switch (m_header.data_type) {
            case DataType::UINT8:
                mode = 0;
                imod_stamp = 1146047817;
                imod_flags &= 1;
                break;
            case DataType::INT8:
                mode = 0;
                break;
            case DataType::INT16:
                mode = 1;
                break;
            case DataType::FLOAT32:
                mode = 2;
                break;
            case DataType::CINT16:
                mode = 3;
                break;
            case DataType::CFLOAT32:
                mode = 4;
                break;
            case DataType::UINT16:
                mode = 6;
                break;
            case DataType::FLOAT16:
                mode = 12;
                break;
            case DataType::UINT4:
                mode = 101;
                break;
            default:
                NOA_THROW("The data type {} is not supported", m_header.data_type);
        }

        int space_group;
        int3_t shape, grid_size;
        float3_t cell_size;
        const int ndim = m_header.shape.flip().ndim();
        if (ndim <= 3) { // 1D, 2D image, or 3D volume
            shape = grid_size = int3_t{m_header.shape.get()};
            space_group = ndim == 3 ? 1 : 0;
        } else { // ndim == 4
            if (m_header.shape[2] == 1) { // treat as stack of 2D images
                shape[0] = grid_size[0] = m_header.shape[0];
                shape[1] = grid_size[1] = m_header.shape[1];
                shape[2] = m_header.shape[3];
                grid_size[2] = 1;
                space_group = 0;
            } else { // treat as stack of volume
                shape[0] = grid_size[0] = m_header.shape[0];
                shape[1] = grid_size[1] = m_header.shape[1];
                shape[2] = m_header.shape[2] * m_header.shape[3]; // total sections
                grid_size[2] = m_header.shape[2]; // sections per volume
                space_group = 401;
            }
        }
        cell_size = float3_t{grid_size} * m_header.pixel_size;

        // Updating the buffer.
        std::memcpy(buffer + 0, shape.get(), 12);
        std::memcpy(buffer + 12, &mode, 4);
        // 16-24: sub-volume (nxstart, nystart, nzstart) -> 0 or unchanged.
        std::memcpy(buffer + 28, grid_size.get(), 12);
        std::memcpy(buffer + 40, cell_size.get(), 12);
        // 52-64: alpha, beta, gamma -> 90,90,90 or unchanged.
        // 64-76: mapc, mapr, maps -> 1,2,3 (anything else is not supported).
        std::memcpy(buffer + 76, &m_header.min, 4);
        std::memcpy(buffer + 80, &m_header.max, 4);
        std::memcpy(buffer + 84, &m_header.mean, 4);
        {
            int tmp; // if it is a volume stack, don't overwrite.
            std::memcpy(&tmp, buffer + 88, 4);
            if (!(tmp > 401 && space_group == 401))
                std::memcpy(buffer + 88, &space_group, 4);
        }
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
        std::memcpy(buffer + 216, &m_header.std, 4);
        std::memcpy(buffer + 220, &m_header.nb_labels, 4); // 0 or unchanged.
        //224-1024: labels -> 0 or unchanged.

        // Swap back the header to its original endianness.
        if (m_header.is_endian_swapped)
            swapHeader_(buffer);

        // Write the buffer.
        m_fstream.seekp(0);
        m_fstream.write(buffer, 1024);
        if (m_fstream.fail()) {
            m_fstream.clear();
            NOA_THROW("File stream error. Could not write the header before closing the file");
        }
    }

    void MRCHeader::read(void* output, DataType data_type, size_t start, size_t end, bool clamp) {
        NOA_ASSERT(end >= start);
        const size_t bytes_to_read = getSerializedSize(m_header.data_type, start);
        const auto offset = offset_() + static_cast<long>(bytes_to_read);
        m_fstream.seekg(offset);
        if (m_fstream.fail()) {
            m_fstream.clear();
            NOA_THROW("Could not seek to the desired offset ({})", offset);
        }
        deserialize(m_fstream, m_header.data_type, output, data_type, end - start, clamp,
                    m_header.is_endian_swapped, static_cast<size_t>(m_header.shape[0]));
    }

    void MRCHeader::readLine(void* output, DataType data_type, size_t start, size_t end, bool clamp) {
        NOA_ASSERT(end >= start);
        const auto elements_per_line = static_cast<size_t>(m_header.shape[0]);
        const size_t bytes_per_line = getSerializedSize(m_header.data_type, elements_per_line);
        const auto offset = offset_() + static_cast<long>(start * bytes_per_line);
        m_fstream.seekg(offset);
        if (m_fstream.fail()) {
            m_fstream.clear();
            NOA_THROW("Could not seek to the desired offset ({})", offset);
        }
        deserialize(m_fstream, m_header.data_type, output, data_type, (end - start) * elements_per_line, clamp,
                    m_header.is_endian_swapped, static_cast<size_t>(m_header.shape[0]));
    }

    [[maybe_unused]]
    void MRCHeader::readShape(void*, DataType, size4_t, size4_t, bool) {
        NOA_THROW("Function is currently not supported");
    }

    void MRCHeader::readSlice(void* output, DataType data_type, size_t start, size_t end, bool clamp) {
        NOA_ASSERT(end >= start);
        const size2_t slice{m_header.shape.get()};
        const size_t elements_per_slice = slice.elements();
        const size_t elements = elements_per_slice * (end - start);
        const size_t bytes_per_slice = getSerializedSize(m_header.data_type, elements_per_slice);
        const long offset = offset_() + static_cast<long>(start * bytes_per_slice);
        m_fstream.seekg(offset);
        if (m_fstream.fail()) {
            m_fstream.clear();
            NOA_THROW("Could not seek to the desired offset ({})", offset);
        }
        deserialize(m_fstream, m_header.data_type, output, data_type, elements, clamp,
                    m_header.is_endian_swapped, slice[0]);
    }

    void MRCHeader::readAll(void* output, DataType data_type, bool clamp) {
        m_fstream.seekg(offset_());
        if (m_fstream.fail()) {
            m_fstream.clear();
            NOA_THROW("Could not seek to the desired offset ({})", offset_());
        }
        deserialize(m_fstream, m_header.data_type, output, data_type, size4_t{m_header.shape}.elements(), clamp,
                    m_header.is_endian_swapped, static_cast<size_t>(m_header.shape[0]));
    }

    void MRCHeader::write(const void* input, DataType data_type, size_t start, size_t end, bool clamp) {
        NOA_ASSERT(end >= start);
        const size_t bytes_to_read = getSerializedSize(m_header.data_type, start);
        const auto offset = offset_() + static_cast<long>(bytes_to_read);
        m_fstream.seekp(offset);
        if (m_fstream.fail()) {
            m_fstream.clear();
            NOA_THROW("Could not seek to the desired offset ({})", offset);
        }
        serialize(input, data_type, m_fstream, m_header.data_type, end - start, clamp,
                  m_header.is_endian_swapped, static_cast<size_t>(m_header.shape[0]));
    }

    void MRCHeader::writeLine(const void* input, DataType data_type, size_t start, size_t end, bool clamp) {
        NOA_ASSERT(end >= start);
        auto elements_per_line = static_cast<size_t>(m_header.shape[0]);
        size_t bytes_per_line = getSerializedSize(m_header.data_type, elements_per_line);
        auto offset = offset_() + static_cast<long>(start * bytes_per_line);
        m_fstream.seekp(offset);
        if (m_fstream.fail()) {
            m_fstream.clear();
            NOA_THROW("Could not seek to the desired offset ({})", offset);
        }
        serialize(input, data_type, m_fstream, m_header.data_type, (end - start) * bytes_per_line, clamp,
                  m_header.is_endian_swapped, static_cast<size_t>(m_header.shape[0]));
    }

    [[maybe_unused]]
    void MRCHeader::writeShape(const void*, DataType, size4_t, size4_t, bool) {
        NOA_THROW("Function is currently not supported");
    }

    void MRCHeader::writeSlice(const void* input, DataType data_type, size_t start, size_t end, bool clamp) {
        NOA_ASSERT(end >= start);
        const size2_t slice{m_header.shape.get()};
        const size_t elements_per_slice = slice.elements();
        const size_t elements = elements_per_slice * (end - start);
        const size_t bytes_per_slice = getSerializedSize(m_header.data_type, elements_per_slice);
        const long offset = offset_() + static_cast<long>(start * bytes_per_slice);
        m_fstream.seekp(offset);
        if (m_fstream.fail()) {
            m_fstream.clear();
            NOA_THROW("Could not seek to the desired offset ({})", offset);
        }
        serialize(input, data_type, m_fstream, m_header.data_type, elements, clamp,
                  m_header.is_endian_swapped, slice[0]);
    }

    void MRCHeader::writeAll(const void* input, DataType data_type, bool clamp) {
        m_fstream.seekp(offset_());
        if (m_fstream.fail()) {
            m_fstream.clear();
            NOA_THROW("Could not seek to the desired offset ({})", offset_());
        }
        serialize(input, data_type, m_fstream, m_header.data_type, size4_t{m_header.shape}.elements(), clamp,
                  m_header.is_endian_swapped, static_cast<size_t>(m_header.shape[0]));
    }
}
