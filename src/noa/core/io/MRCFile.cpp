#include <cstring>  // memcpy, memset
#include <thread>   // std::this_thread::sleep_for
#include <chrono>   // std::chrono::milliseconds

#include "noa/core/io/OS.hpp"
#include "noa/core/io/MRCFile.hpp"

namespace noa::io {
    void MrcFile::open_(const Path& filename, Open open_mode, const std::source_location& location) {
        close_();

        const bool overwrite = open_mode.truncate or not open_mode.read;
        bool exists;
        try {
            exists = is_file(filename);
            if (open_mode.write) {
                if (exists)
                    backup(filename, overwrite);
                else if (overwrite)
                    mkdir(filename.parent_path());
            }
        } catch (...) {
            panic_at_location(
                    location, "File: {}. {}. Could not open the file because of an OS failure",
                    filename, open_mode);
        }

        m_open_mode = open_mode;
        m_open_mode.binary = true;
        m_open_mode.append = false;
        m_open_mode.at_the_end = false;

        for (u32 it{}; it < 5; ++it) {
            m_fstream.open(filename, m_open_mode.to_ios_base());
            if (m_fstream.is_open()) {
                if (exists and not overwrite) // case 1 or 2
                    read_header_(filename);
                m_filename = filename;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        m_fstream.clear();

        if (open_mode.read and not overwrite and not exists) {
            panic_at_location(location, "File: {}. {}. Failed to open the file. The file does not exist",
                              filename, open_mode);
        }
        panic_at_location(
                location, "File: {}. {}. Failed to open the file. Check the permissions for that directory",
                filename, open_mode);
    }

    void MrcFile::read_header_(const Path& filename) {
        Byte buffer[1024];
        m_fstream.seekg(0);
        m_fstream.read(reinterpret_cast<char*>(buffer), 1024);
        if (m_fstream.fail()) {
            m_fstream.clear();
            panic("File: {}. File stream error. Could not read the header", filename);
        }

        // Endianness.
        // Some software use 68-65, but the CCPEM standard is using 68-68...
        char stamp[4];
        std::memcpy(&stamp, buffer + 212, 4);
        if ((stamp[0] == 68 and stamp[1] == 65 and stamp[2] == 0 and stamp[3] == 0) or
            (stamp[0] == 68 and stamp[1] == 68 and stamp[2] == 0 and stamp[3] == 0)) { /* little */
            m_header.is_endian_swapped = is_big_endian();
        } else if (stamp[0] == 17 and stamp[1] == 17 and stamp[2] == 0 and stamp[3] == 0) {/* big */
            m_header.is_endian_swapped = not is_big_endian();
        } else {
            panic("File: {}. Invalid data. Endianness was not recognized."
                  "Should be [68,65,0,0], [68,68,0,0] or [17,17,0,0], got [{},{},{},{}]",
                  filename, stamp[0], stamp[1], stamp[2], stamp[3]);
        }

        // If data is swapped, some parts of the buffer need to be swapped as well.
        if (m_header.is_endian_swapped)
            swap_header_(buffer);

        // Read & Write mode: we'll need to update the header when closing the file, so save the buffer.
        // The endianness will be swapped back to the original endianness of the file.
        if (m_open_mode.write) {
            if (not m_header.buffer)
                m_header.buffer = std::make_unique<Byte[]>(1024);
            std::memcpy(m_header.buffer.get(), buffer, 1024);
        }

        // Get the header from file.
        i32 mode, imod_stamp, imod_flags, space_group;
        Shape3<i32> shape, grid_size;
        Vec3<i32> order;
        Vec3<f32> cell_size;
        {
            std::memcpy(shape.data(), buffer, 12);
            std::memcpy(&mode, buffer + 12, 4);
            // 16-24: sub-volume (nxstart, nystart, nzstart).
            std::memcpy(grid_size.data(), buffer + 28, 12);
            std::memcpy(cell_size.data(), buffer + 40, 12);
            // 52-64: alpha, beta, gamma.
            std::memcpy(order.data(), buffer + 64, 12);
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

        // Set the BDHW shape:
        const auto ndim = shape.flip().ndim();
        check(all(shape > 0),
              "File: {}. Invalid data. Logical shape should be "
              "greater than zero, got nx,ny,nz:{}", filename, shape);

        if (ndim <= 2) {
            check(all(grid_size == shape),
                  "File: {}. 1d or 2d data detected. "
                  "The logical shape should be equal to the grid size. "
                  "Got nx,ny,nz:{}, mx,my,mz:{}", filename, shape, grid_size);
            m_header.shape = {1, 1, shape[1], shape[0]};
        } else { // ndim == 3
            if (space_group == 0) { // stack of 2D images
                // FIXME We should check grid_size[2] == 1, but some packages ignore this, so do nothing for now.
                check(shape[0] == grid_size[0] and shape[1] == grid_size[1],
                      "File: {}. 2d stack of images detected (ndim=3, group=0). "
                      "The two innermost dimensions of the logical shape and "
                      "the grid size should be equal. Got nx,ny,nz:{}, mx,my,mz:{}",
                      filename, shape, grid_size);
                m_header.shape = {shape[2], 1, shape[1], shape[0]};
            } else if (space_group == 1) { // volume
                check(all(grid_size == shape),
                      "File: {}. 3d volume detected (ndim=3, group=1). "
                      "The logical shape should be equal to the grid size. "
                      "Got nx,ny,nz:{}, mx,my,mz:{}", filename, shape, grid_size);
                m_header.shape = {1, shape[2], shape[1], shape[0]};
            } else if (space_group >= 401 and space_group <= 630) { // stack of volume
                // grid_size[2] = sections per vol, shape[2] = total number of sections
                check(is_multiple_of(shape[2], grid_size[2]),
                      "File: {}. Stack of 3d volumes detected. "
                      "The total sections (nz:{}) should be divisible "
                      "by the number of sections per volume (mz:{})",
                      filename, shape[2], grid_size[2]);
                check(shape[0] == grid_size[0] and shape[1] == grid_size[1],
                      "File: {}. Stack of 3d volumes detected. "
                      "The first two dimensions of the shape and the grid size "
                      "should be equal. Got nx,ny,nz:{}, mx,my,mz:{}",
                      filename, shape, grid_size);
                m_header.shape = {shape[2] / grid_size[2], grid_size[2], shape[1], shape[0]};
            } else {
                panic("File: {}. Data shape is not recognized. "
                      "Got nx,ny,nz:{}, mx,my,mz:{}, group:",
                      filename, shape, grid_size, space_group);
            }
        }

        // Set the pixel size:
        m_header.pixel_size = cell_size / grid_size.vec.as<f32>();
        m_header.pixel_size = m_header.pixel_size.flip();
        check(all(m_header.pixel_size >= 0),
              "File: {}. Invalid data. Pixel size should not be negative, got {}",
              filename, m_header.pixel_size);
        check(m_header.extended_bytes_nb >= 0,
              "File: {}. Invalid data. Extended header size should be positive, got {}",
              filename, m_header.extended_bytes_nb);

        // Convert mode to encoding format:
        switch (mode) {
            case 0:
                if (imod_stamp == 1146047817 and imod_flags & 1)
                    m_header.encoding_format = Encoding::Format::U8;
                else
                    m_header.encoding_format = Encoding::Format::I8;
                break;
            case 1:
                m_header.encoding_format = Encoding::Format::I16;
                break;
            case 2:
                m_header.encoding_format = Encoding::Format::F32;
                break;
            case 3:
                m_header.encoding_format = Encoding::Format::CI16;
                break;
            case 4:
                m_header.encoding_format = Encoding::Format::C32;
                break;
            case 6:
                m_header.encoding_format = Encoding::Format::U16;
                break;
            case 12:
                m_header.encoding_format = Encoding::Format::F16;
                break;
            case 16:
                panic("File: {}. MRC mode 16 is not currently supported", filename);
            case 101:
                m_header.encoding_format = Encoding::Format::U4;
                break;
            default:
                panic("File: {}. Invalid data. MRC mode not recognized, got {}", filename, mode);
        }

        // Map order: enforce row-major ordering, i.e. x=1, y=2, z=3.
        if (any(order != Vec3<i32>{1, 2, 3})) {
            if (any(order < 1) or any(order > 3) or sum(order) != 6)
                panic("File: {}. Invalid data. Map order should be (1,2,3), got {}", filename, order);
            panic("File: {}. Map order {} is not supported. Only (1,2,3) is supported", filename, order);
        }
    }

    void MrcFile::close_() {
        if (not is_open())
            return;

        // Writing mode: the header should be updated before closing the file.
        if (m_open_mode.write) {
            // Writing & reading mode: the instance didn't create the file,
            // the header was saved by read_header_().
            if (m_open_mode.read) {
                write_header_(m_header.buffer.get());
            } else {
                Byte buffer[1024];
                default_header_(buffer);
                write_header_(buffer);
            }
        }
        m_fstream.close();
        if (m_fstream.fail() and not m_fstream.eof()) {
            m_fstream.clear();
            panic("File stream error. Could not close the file");
        }
        m_filename.clear();
    }

    void MrcFile::default_header_(Byte* buffer) {
        std::memset(buffer, 0, 1024); // Set everything to 0.
        auto* buffer_ptr = reinterpret_cast<char*>(buffer);

        // Set the unused flags which do not default to 0.
        // The used bytes will be set before closing the file by writeHeader_().
        f32 angles[3] = {90, 90, 90};
        std::memcpy(buffer + 52, angles, 12);

        i32 order[3] = {1, 2, 3};
        std::memcpy(buffer + 64, order, 12);

        buffer_ptr[104] = 'S';
        buffer_ptr[105] = 'E';
        buffer_ptr[106] = 'R';
        buffer_ptr[107] = 'I';

        buffer_ptr[208] = 'M';
        buffer_ptr[209] = 'A';
        buffer_ptr[210] = 'P';
        buffer_ptr[211] = ' ';

        // With new data, the endianness is always set to the endianness of the CPU.
        if (is_big_endian()) {
            buffer_ptr[212] = 17; // 16 * 1 + 1
            buffer_ptr[213] = 17;
        } else {
            buffer_ptr[212] = 68; // 16 * 4 + 4
            buffer_ptr[213] = 68;
        }
        buffer_ptr[214] = 0;
        buffer_ptr[215] = 0;
    }

    void MrcFile::write_header_(Byte* buffer) {
        // Data type.
        i32 mode{}, imod_stamp{}, imod_flags{};
        switch (m_header.encoding_format) {
            case Encoding::Format::U8:
                mode = 0;
                imod_stamp = 1146047817;
                imod_flags &= 1;
                break;
            case Encoding::Format::I8:
                mode = 0;
                break;
            case Encoding::Format::I16:
                mode = 1;
                break;
            case Encoding::Format::F32:
                mode = 2;
                break;
            case Encoding::Format::CI16:
                mode = 3;
                break;
            case Encoding::Format::C32:
                mode = 4;
                break;
            case Encoding::Format::U16:
                mode = 6;
                break;
            case Encoding::Format::F16:
                mode = 12;
                break;
            case Encoding::Format::U4:
                mode = 101;
                break;
            default:
                panic("File: {}. {} is not supported", m_filename, m_header.encoding_format);
        }

        i32 space_group{};
        Shape3<i32> shape, grid_size;
        Vec3<f32> cell_size;
        auto bdhw_shape = m_header.shape.as<i32>(); // can be empty if nothing was written...
        if (not bdhw_shape.is_batched()) { // 1d, 2d image, or 3d volume
            shape = grid_size = bdhw_shape.pop_front().flip();
            space_group =  bdhw_shape.ndim() == 3 ? 1 : 0;
        } else { // ndim == 4
            if (bdhw_shape[1] == 1) { // treat as stack of 2d images
                shape[0] = grid_size[0] = bdhw_shape[3];
                shape[1] = grid_size[1] = bdhw_shape[2];
                shape[2] = bdhw_shape[0];
                grid_size[2] = 1;
                space_group = 0;
            } else { // treat as stack of volume
                shape[0] = grid_size[0] = bdhw_shape[3];
                shape[1] = grid_size[1] = bdhw_shape[2];
                shape[2] = bdhw_shape[1] * bdhw_shape[0]; // total sections
                grid_size[2] = bdhw_shape[1]; // sections per volume
                space_group = 401;
            }
        }
        cell_size = grid_size.vec.as<f32>() * m_header.pixel_size.flip();

        // Updating the buffer.
        std::memcpy(buffer + 0, shape.data(), 12);
        std::memcpy(buffer + 12, &mode, 4);
        // 16-24: sub-volume (nxstart, nystart, nzstart) -> 0 or unchanged.
        std::memcpy(buffer + 28, grid_size.data(), 12);
        std::memcpy(buffer + 40, cell_size.data(), 12);
        // 52-64: alpha, beta, gamma -> 90,90,90 or unchanged.
        // 64-76: mapc, mapr, maps -> 1,2,3 (anything else is not supported).
        std::memcpy(buffer + 76, &m_header.min, 4);
        std::memcpy(buffer + 80, &m_header.max, 4);
        std::memcpy(buffer + 84, &m_header.mean, 4);
        {
            i32 tmp{}; // if it is a volume stack, don't overwrite.
            std::memcpy(&tmp, buffer + 88, 4);
            if (tmp <= 401 or space_group != 401)
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
            swap_header_(buffer);

        // Write the buffer.
        m_fstream.seekp(0);
        m_fstream.write(reinterpret_cast<char*>(buffer), 1024);
        if (m_fstream.fail()) {
            m_fstream.clear();
            panic("File: {}. File stream error. Could not write the header before closing the file", m_filename);
        }
    }
}
