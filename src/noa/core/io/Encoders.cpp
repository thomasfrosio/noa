#include <cstring>  // memcpy

#include "noa/core/io/Encoders.hpp"
#include "noa/core/io/IO.hpp"

namespace noa::io {
    auto EncoderMrc::read_header(
        std::FILE* file
    ) -> Tuple<Shape<i64, 4>, Vec<f64, 3>, Encoding::Type> {
        std::byte buffer[1024];
        check(std::fseek(file, 0, SEEK_SET) == 0, "Failed to seek {}", std::strerror(errno));
        check(std::fread(buffer, 1, 1024, file) == 1024, "Failed to read the header (1024 bytes)");

        // Endianness.
        char stamp[4];
        std::memcpy(&stamp, buffer + 212, 4);
        // Some software use 68-65, but the CCPEM standard is using 68-68...
        if ((stamp[0] == 68 and stamp[1] == 65 and stamp[2] == 0 and stamp[3] == 0) or
            (stamp[0] == 68 and stamp[1] == 68 and stamp[2] == 0 and stamp[3] == 0)) { /* little */
            m_is_endian_swapped = is_big_endian();
        } else if (stamp[0] == 17 and stamp[1] == 17 and stamp[2] == 0 and stamp[3] == 0) {/* big */
            m_is_endian_swapped = not is_big_endian();
        } else {
            panic("Invalid data. Endianness was not recognized. "
                  "Should be [68,65,0,0], [68,68,0,0] or [17,17,0,0], got [{},{},{},{}]",
                  stamp[0], stamp[1], stamp[2], stamp[3]);
        }
        if (m_is_endian_swapped) {
            /// Swap the endianness of the header.
            /// The buffer is at least the first 224 bytes of the MRC header.
            /// All used flags are swapped. Some unused flags are left unchanged.
            std::reverse(buffer, buffer + 24);
            swap_endian<4>(buffer, 24); // from 0 (nx) to 96 (next, included).
            swap_endian<4>(buffer + 152, 2); // imodStamp, imodFlags
            swap_endian<4>(buffer + 216, 2); // rms, nlabl
        }

        // Get the header.
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
            // std::memcpy(&min, buffer + 76, 4);
            // std::memcpy(&max, buffer + 80, 4);
            // std::memcpy(&mean, buffer + 84, 4);
            std::memcpy(&space_group, buffer + 88, 4);
            std::memcpy(&m_extended_bytes_nb, buffer + 92, 4);
            // 96-98: creatid, extra data, extType, nversion, extra data, nint, nreal, extra data.
            std::memcpy(&imod_stamp, buffer + 152, 4);
            std::memcpy(&imod_flags, buffer + 156, 4);
            // 160-208: idtype, lens, nd1, nd2, vd1, vd2, tiltangles, origin(x,y,z).
            // 208-212: cmap.
            // 212-216: stamp.
            // std::memcpy(&stddev, buffer + 216, 4);
            // std::memcpy(&m_nb_labels, buffer + 220, 4);
            //224-1024: labels.
        }

        // Set the BDHW shape:
        const auto ndim = shape.flip().ndim();
        check(all(shape > 0), "Invalid data. Logical shape should be greater than zero, got nx,ny,nz:{}", shape);

        if (ndim <= 2) {
            check(all(grid_size == shape),
                  "1d or 2d data detected. The logical shape should be equal to the grid size. "
                  "Got nx,ny,nz:{}, mx,my,mz:{}", shape, grid_size);
            m_shape = {1, 1, shape[1], shape[0]};
        } else { // ndim == 3
            if (space_group == 0) { // stack of 2D images
                // FIXME We should check grid_size[2] == 1, but some packages ignore this, so do nothing for now.
                check(shape[0] == grid_size[0] and shape[1] == grid_size[1],
                      "2d stack of images detected (ndim=3, group=0). "
                      "The two innermost dimensions of the logical shape and "
                      "the grid size should be equal. Got nx,ny,nz:{}, mx,my,mz:{}",
                      shape, grid_size);
                m_shape = {shape[2], 1, shape[1], shape[0]};
            } else if (space_group == 1) { // volume
                check(all(grid_size == shape),
                      "3d volume detected (ndim=3, group=1). "
                      "The logical shape should be equal to the grid size. "
                      "Got nx,ny,nz:{}, mx,my,mz:{}", shape, grid_size);
                m_shape = {1, shape[2], shape[1], shape[0]};
            } else if (space_group >= 401 and space_group <= 630) { // stack of volume
                // grid_size[2] = sections per vol, shape[2] = total number of sections
                check(is_multiple_of(shape[2], grid_size[2]),
                      "Stack of 3d volumes detected. "
                      "The total sections (nz:{}) should be divisible "
                      "by the number of sections per volume (mz:{})",
                      shape[2], grid_size[2]);
                check(shape[0] == grid_size[0] and shape[1] == grid_size[1],
                      "Stack of 3d volumes detected. "
                      "The first two dimensions of the shape and the grid size "
                      "should be equal. Got nx,ny,nz:{}, mx,my,mz:{}",
                      shape, grid_size);
                m_shape = {shape[2] / grid_size[2], grid_size[2], shape[1], shape[0]};
            } else {
                panic("Data shape is not recognized. "
                      "Got nx,ny,nz:{}, mx,my,mz:{}, group:",
                      shape, grid_size, space_group);
            }
        }

        // Set the pixel size:
        m_spacing = cell_size / grid_size.vec.as<f32>();
        m_spacing = m_spacing.flip();
        check(all(m_spacing >= 0), "Invalid data. Pixel size should not be negative, got {}", m_spacing);
        check(m_extended_bytes_nb >= 0, "Invalid data. Extended header size should be positive, got {}", m_extended_bytes_nb);

        // Convert mode to encoding format:
        switch (mode) {
            case 0: {
                if (imod_stamp == 1146047817 and imod_flags & 1)
                    m_dtype = Encoding::U8;
                else
                    m_dtype = Encoding::I8;
                break;
            }
            case 1: m_dtype = Encoding::I16; break;
            case 2: m_dtype = Encoding::F32; break;
            case 3: m_dtype = Encoding::CI16; break;
            case 4: m_dtype = Encoding::C32; break;
            case 6: m_dtype = Encoding::U16; break;
            case 12: m_dtype = Encoding::F16; break;
            case 16:
                panic("MRC mode 16 is not currently supported");
            case 101: {
                check(is_even(m_shape[3]),
                      "Mode 101 (u4) is only supported for shapes with even width, but got shape={}",
                      m_shape);
                m_dtype = Encoding::U4;
                break;
            }
            default:
                panic("Invalid data. MRC mode not recognized, got {}", mode);
        }

        // Map order: enforce row-major ordering, i.e. x=1, y=2, z=3.
        if (any(order != Vec{1, 2, 3})) {
            if (any(order < 1) or any(order > 3) or sum(order) != 6)
                panic("Invalid data. Map order should be (1,2,3), got {}", order);
            panic("Map order {} is not supported. Only (1,2,3) is supported", order);
        }

        return {m_shape, m_spacing.as<f64>(), m_dtype};
    }

    void EncoderMrc::write_header(
        std::FILE* file,
        const Shape<i64, 4>& shape,
        const Vec<f64, 3>& spacing,
        Encoding::Type dtype
    ) {
        // Sets the member variables.
        m_shape = shape;
        m_spacing = spacing.as<f32>();
        m_dtype = dtype;

        // Data type.
        i32 mode{}, imod_stamp{}, imod_flags{};
        switch (m_dtype) {
            case Encoding::U8: {
                mode = 0;
                imod_stamp = 1146047817;
                imod_flags &= 1;
                break;
            }
            case Encoding::I8:      mode = 0;   break;
            case Encoding::I16:     mode = 1;   break;
            case Encoding::F32:     mode = 2;   break;
            case Encoding::CI16:    mode = 3;   break;
            case Encoding::C32:     mode = 4;   break;
            case Encoding::U16:     mode = 6;   break;
            case Encoding::F16:     mode = 12;  break;
            case Encoding::U4: {
                check(is_even(m_shape[3]),
                      "Mode 101 (u4) is only supported for shapes with even width, but got shape={}",
                      m_shape);
                mode = 101;
                break;
            }
            default:
                panic("Data type {} is not supported", m_dtype);
        }

        // Shape/spacing.
        i32 space_group{};
        Shape3<i32> whd_shape, grid_size;
        Vec3<f32> cell_size;
        auto bdhw_shape = m_shape.as<i32>(); // can be empty if nothing was written...
        if (not bdhw_shape.is_batched()) { // 1d, 2d image, or 3d volume
            whd_shape = grid_size = bdhw_shape.pop_front().flip();
            space_group =  bdhw_shape.ndim() == 3 ? 1 : 0;
        } else { // ndim == 4
            if (bdhw_shape[1] == 1) { // treat as stack of 2d images
                whd_shape[0] = grid_size[0] = bdhw_shape[3];
                whd_shape[1] = grid_size[1] = bdhw_shape[2];
                whd_shape[2] = bdhw_shape[0];
                grid_size[2] = 1;
                space_group = 0;
            } else { // treat as stack of volume
                whd_shape[0] = grid_size[0] = bdhw_shape[3];
                whd_shape[1] = grid_size[1] = bdhw_shape[2];
                whd_shape[2] = bdhw_shape[1] * bdhw_shape[0]; // total sections
                grid_size[2] = bdhw_shape[1]; // sections per volume
                space_group = 401;
            }
        }
        cell_size = grid_size.vec.as<f32>() * m_spacing.flip();

        // Initialize the header.
        std::byte buffer[1024]{};
        auto* buffer_ptr = reinterpret_cast<char*>(buffer);
        std::memcpy(buffer + 0, whd_shape.data(), 12);
        std::memcpy(buffer + 12, &mode, 4);
        // 16-24: sub-volume (nxstart, nystart, nzstart) -> 0.
        std::memcpy(buffer + 28, grid_size.data(), 12);
        std::memcpy(buffer + 40, cell_size.data(), 12);
        f32 angles[3]{90, 90, 90};
        std::memcpy(buffer + 52, angles, 12);
        i32 order[3]{1, 2, 3}; // mapc, mapr, maps
        std::memcpy(buffer + 64, order, 12);
        f32 max{-1}, mean{2};
        // std::memcpy(buffer + 76, &min, 4); // min=0
        std::memcpy(buffer + 80, &max, 4);
        std::memcpy(buffer + 84, &mean, 4);
        {
            i32 tmp{}; // if it is a volume stack, don't overwrite.
            std::memcpy(&tmp, buffer + 88, 4);
            if (tmp <= 401 or space_group != 401)
                std::memcpy(buffer + 88, &space_group, 4);
        }
        std::memcpy(buffer + 92, &m_extended_bytes_nb, 4); // 0.
        // 96-98: creatid -> 0.
        // 98-104: extra data -> 0.
        // 104-108: extType.
        buffer_ptr[104] = 'S';
        buffer_ptr[105] = 'E';
        buffer_ptr[106] = 'R';
        buffer_ptr[107] = 'I';
        // 108-112: nversion -> 0.
        // 112-128: extra data -> 0.
        // 128-132: nint, nreal -> 0.
        // 132-152: extra data -> 0.
        std::memcpy(buffer + 152, &imod_stamp, 4);
        std::memcpy(buffer + 156, &imod_flags, 4);
        // 160-208: idtype, lens, nd1, nd2, vd1, vd2, tiltangles, origin(x,y,z) -> 0.
        // 208-212: cmap -> "MAP ".
        // 212-216: stamp -> [68,65,0,0] or [17,17,0,0].
        buffer_ptr[208] = 'M';
        buffer_ptr[209] = 'A';
        buffer_ptr[210] = 'P';
        buffer_ptr[211] = ' ';
        if constexpr (is_big_endian()) {
            buffer_ptr[212] = 17; // 16 * 1 + 1
            buffer_ptr[213] = 17;
        } else {
            buffer_ptr[212] = 68; // 16 * 4 + 4
            buffer_ptr[213] = 68;
        }
        f32 stddev{-1};
        std::memcpy(buffer + 216, &stddev, 4);
        // std::memcpy(buffer + 220, &n_labels, 4);
        //224-1024: labels -> 0 or unchanged.

        // Write the header to the file.
        check(std::fseek(file, 0, SEEK_SET) == 0, "Failed to seek {}", std::strerror(errno));
        check(std::fwrite(buffer, 1, 1024, file) == 1024, "Failed to write the header (1024 bytes). {}", std::strerror(errno));
    }
}
