#include <cstring>  // memcpy
#include <mutex>
#include <cstdio>

#ifdef NOA_ENABLE_TIFF
#   include <tiffio.h>
#   include <omp.h>
#endif

#include "noa/core/io/IO.hpp"
#include "noa/core/io/ImageFile.hpp"

namespace noa::io {
    auto ImageFileEncoderMrc::read_header(
        std::FILE* file
    ) -> Tuple<Shape<i64, 4>, Vec<f64, 3>, Encoding::Type, Compression> {
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

        return make_tuple(m_shape, m_spacing.as<f64>(), m_dtype, Compression::NONE);
    }

    auto ImageFileEncoderMrc::write_header(
        std::FILE* file,
        const Shape<i64, 4>& shape,
        const Vec<f64, 3>& spacing,
        Encoding::Type dtype,
        Compression compression
    ) -> Tuple<Encoding::Type, Compression> {
        // Sets the member variables.
        m_shape = shape;
        m_spacing = spacing.as<f32>();
        m_dtype = closest_supported_dtype(dtype);
        check(m_dtype != Encoding::UNKNOWN, "The data type is set to unknown");
        check(compression != Compression::UNKNOWN, "The compression scheme is set to unknown");

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
                panic(); // this should be unreachable(), but panic instead of triggering UB
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

        return make_tuple(m_dtype, Compression::NONE);
    }
}

#ifdef NOA_ENABLE_TIFF
namespace {
    using namespace noa;
    using namespace noa::io;

    #ifdef NOA_ENABLE_OPENMP
    constexpr size_t NOA_TIFF_MAX_HANDLES = 16;
    #else
    constexpr size_t NOA_TIFF_MAX_HANDLES = 1;
    #endif

    auto tiff_client_read(thandle_t data, tdata_t buf, tsize_t size) -> tsize_t {
        auto handle = static_cast<ImageFileEncoderTiff::Handle*>(data);
        auto lock = std::scoped_lock(*handle->mutex);

        check(std::fseek(handle->file, handle->offset, SEEK_SET) == 0);
        size_t n_bytes_read = std::fread(buf, 1, safe_cast<size_t>(size), handle->file);
        check(std::ferror(handle->file) == 0);
        handle->offset += static_cast<long>(n_bytes_read);

        return safe_cast<tsize_t>(n_bytes_read);
    }

    auto tiff_client_write(thandle_t data, tdata_t buf, tsize_t size) -> tsize_t {
        auto handle = static_cast<ImageFileEncoderTiff::Handle*>(data);
        // auto lock = std::scoped_lock(*handle->mutex); writing to the file is synchronous, so don't lock

        check(std::fseek(handle->file, handle->offset, SEEK_SET) == 0);
        size_t n_bytes_written = std::fwrite(buf, 1, safe_cast<size_t>(size), handle->file);
        check(std::ferror(handle->file) == 0);
        handle->offset += static_cast<long>(n_bytes_written);

        return safe_cast<tsize_t>(n_bytes_written);
    }

    auto tiff_client_seek(thandle_t data, toff_t offset, int whence) -> toff_t {
        auto handle = static_cast<ImageFileEncoderTiff::Handle*>(data);

        // Convert to offset from the beginning.
        switch (whence) {
            case SEEK_SET: {
                handle->offset = safe_cast<long>(offset);
                break;
            }
            case SEEK_CUR: {
                handle->offset += safe_cast<long>(offset);
                break;
            }
            case SEEK_END: {
                check(std::fseek(handle->file, 0, SEEK_END) == 0);
                long end = std::ftell(handle->file);
                handle->offset = end + safe_cast<long>(offset);;
                break;
            }
            default:
                panic();
        }
        return safe_cast<toff_t>(handle->offset);
    }

    auto tiff_client_close(thandle_t) -> int {
        return 0; // we are not responsible to close the file stream
    }

    auto tiff_client_size(thandle_t data) -> toff_t {
        auto handle = static_cast<ImageFileEncoderTiff::Handle*>(data);
        auto lock = std::scoped_lock(*handle->mutex);
        auto current_pos = std::ftell(handle->file);
        check(current_pos != -1);
        check(std::fseek(handle->file, 0, SEEK_END) == 0);
        auto file_size = std::ftell(handle->file);
        check(file_size != -1);
        check(std::fseek(handle->file, current_pos, SEEK_SET) == 0);
        return safe_cast<toff_t>(file_size);
    }

    auto tiff_client_map(thandle_t, void**, toff_t*) -> int {
        return 0; // not supported
    }

    auto tiff_client_unmap(thandle_t, void*, toff_t) -> void {
        // not supported
    }

    // One string per thread is enough since TIFFFile will immediately throw after the error.
    thread_local std::string s_error_buffer;

    #if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wformat-nonliteral"
    #endif
    void format_message_tiff_(std::string& output, const char* module, const char* fmt, va_list args) {
        if (module) { // module is optional
            output += module;
            output += ": ";
        }
        // Could use va_copy and loop until the entire string is formatted,
        // but that's unlikely to be necessary and may be compiler-specific territory.
        char tmp[400]; // should be enough for all error messages from libtiff
        if (vsnprintf(tmp, 400, fmt, args) > 0)
            output += tmp;
    }
    #if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic pop
    #endif

    // if s_error_buffer is not empty, there's an error waiting to be printed.
    extern "C" void tiff_error(const char* module, const char* fmt, va_list args) {
        s_error_buffer.clear();
        format_message_tiff_(s_error_buffer, module, fmt, args);
    }

    extern "C" void tiff_warning(const char*, const char*, va_list) {
        // TODO
    }

    auto tiff_client_init(ImageFileEncoderTiff::Handle* handle, const char* mode) -> ::TIFF* {
        // Error handling setup.
        thread_local bool thread_is_set{};
        if (not thread_is_set) {
            s_error_buffer.reserve(250);
            ::TIFFSetErrorHandler(tiff_error);
            ::TIFFSetWarningHandler(tiff_warning);
            thread_is_set = true;
        }

        // Create a new TIFF client from the std::FILE.
        ::TIFF* tiff = ::TIFFClientOpen(
            "tifffile", mode, handle,
            tiff_client_read,
            tiff_client_write,
            tiff_client_seek,
            tiff_client_close,
            tiff_client_size,
            tiff_client_map,
            tiff_client_unmap
        );
        check(tiff != nullptr, "Error while opening TIFF client from FILE stream. {}", s_error_buffer);
        return tiff;
    }

    auto tiff_get_dtype(u16 sample_format, u16 bits_per_sample) {
        switch (sample_format) {
            case SAMPLEFORMAT_INT:
                if (bits_per_sample == 8)
                    return Encoding::I8;
                if (bits_per_sample == 16)
                    return Encoding::I16;
                if (bits_per_sample == 32)
                    return Encoding::I32;
            break;
            case SAMPLEFORMAT_UINT:
                if (bits_per_sample == 8)
                    return Encoding::U8;
                if (bits_per_sample == 16)
                    return Encoding::U16;
                if (bits_per_sample == 32)
                    return Encoding::U32;
                if (bits_per_sample == 4)
                    return Encoding::U4;
            break;
            case SAMPLEFORMAT_IEEEFP:
                if (bits_per_sample == 16)
                    return Encoding::F16;
                if (bits_per_sample == 32)
                    return Encoding::F32;
                if (bits_per_sample == 64)
                    return Encoding::F64;
            break;
            case SAMPLEFORMAT_COMPLEXINT:
                if (bits_per_sample == 32)
                    return Encoding::CI16;
            break;
            case SAMPLEFORMAT_COMPLEXIEEEFP:
                if (bits_per_sample == 32)
                    return Encoding::C16;
                if (bits_per_sample == 64)
                    return Encoding::C32;
                if (bits_per_sample == 128)
                    return Encoding::C64;
            break;
            default:
                break;
        }
        return Encoding::UNKNOWN;
    }

    template<typename T, StridesTraits S>
    void flip_rows_(const Span<T, 2, i64, S>& slice) {
        for (i64 row = 0; row < slice.shape()[0] / 2; ++row) {
            auto current_row = slice[row];
            auto opposite_row = slice[slice.shape()[0] - row - 1];
            for (i64 i{}; i < slice.shape()[1]; ++i)
                std::swap(opposite_row[i], current_row[i]);
        }
    }

    void tiff_set_dtype(Encoding::Type dtype, u16* sample_format, u16* bits_per_sample) {
        switch (dtype) {
            case Encoding::I8:
                *bits_per_sample = 8;
                *sample_format = SAMPLEFORMAT_INT;
                break;
            case Encoding::U8:
                *bits_per_sample = 8;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case Encoding::I16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_INT;
                break;
            case Encoding::U16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case Encoding::I32:
                *bits_per_sample = 32;
                *sample_format = SAMPLEFORMAT_INT;
                break;
            case Encoding::U32:
                *bits_per_sample = 32;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case Encoding::I64:
                *bits_per_sample = 64;
                *sample_format = SAMPLEFORMAT_INT;
                break;
            case Encoding::U64:
                *bits_per_sample = 64;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case Encoding::F16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_IEEEFP;
                break;
            case Encoding::F32:
                *bits_per_sample = 32;
                *sample_format = SAMPLEFORMAT_IEEEFP;
                break;
            case Encoding::F64:
                *bits_per_sample = 64;
                *sample_format = SAMPLEFORMAT_IEEEFP;
                break;
            case Encoding::C16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_COMPLEXIEEEFP;
                break;
            case Encoding::C32:
                *bits_per_sample = 32;
                *sample_format = SAMPLEFORMAT_COMPLEXIEEEFP;
                break;
            case Encoding::C64:
                *bits_per_sample = 64;
                *sample_format = SAMPLEFORMAT_COMPLEXIEEEFP;
                break;
            case Encoding::U4:
                *bits_per_sample = 4;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case Encoding::CI16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_COMPLEXINT;
                break;
            default:
                *bits_per_sample = 1;
                *sample_format = SAMPLEFORMAT_VOID;
                break;
        }
    }
}

namespace noa::io {
    auto ImageFileEncoderTiff::read_header(
        std::FILE* file
    ) -> Tuple<Shape<i64, 4>, Vec<f64, 3>, Encoding::Type, Compression> {
        // Previously opened TIFF files should be closed by now, but make sure.
        close();
        if (m_handles == nullptr)
            m_handles = std::make_unique<handle_type[]>(NOA_TIFF_MAX_HANDLES);

        // Open the TIFF file.
        m_handles[0].first = Handle{.mutex = &m_mutex, .file = file};
        auto tiff = tiff_client_init(&m_handles[0].first, "r");
        m_handles[0].second = tiff;

        u16 n_directories{};
        u16 tiff_compression{};
        while (::TIFFSetDirectory(tiff, n_directories)) {
            // Shape.
            Shape2<u32> shape; // hw
            check(::TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, shape.data() + 1) and
                  ::TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, shape.data()),
                  "The input TIFF file does not have the width or height field");

            // Spacing.
            Vec2<f32> pixel_size{}; // hw angpix
            {
                u16 resolution_unit{}; // 1: no units, 2: inch, 3: cm
                const auto has_resolution = TIFFGetField(tiff, TIFFTAG_XRESOLUTION, &pixel_size[1]);
                TIFFGetFieldDefaulted(tiff, TIFFTAG_RESOLUTIONUNIT, &resolution_unit);
                if (resolution_unit > 1 and has_resolution) {
                    if (not TIFFGetField(tiff, TIFFTAG_YRESOLUTION, pixel_size.data()))
                        pixel_size[0] = pixel_size[1];
                    const auto scale = resolution_unit == 2 ? 2.54e8f : 1.00e8f; // to angpix
                    pixel_size = scale / pixel_size;
                }
            }

            // Data type.
            Encoding::Type data_type{};
            {
                u16 photometry{}, sample_per_pixel{}, sample_format{}, bits_per_sample{}, planar_config{};
                check(::TIFFGetField(tiff, TIFFTAG_PHOTOMETRIC, &photometry));
                ::TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLESPERPIXEL, &sample_per_pixel);
                ::TIFFGetFieldDefaulted(tiff, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
                ::TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &sample_format);
                ::TIFFGetFieldDefaulted(tiff, TIFFTAG_PLANARCONFIG, &planar_config);
                check(s_error_buffer.empty(), "Could not read fields of directory {}. {}", n_directories, s_error_buffer);
                check(planar_config == PLANARCONFIG_CONTIG, "Separate planes are not supported. Should be contiguous ");
                check(photometry <= 2, "Photometry are not supported. Should be bi-level or grayscale");
                check(sample_per_pixel == 1, "Samples per pixel should be 1, but got {}", sample_per_pixel);

                data_type = tiff_get_dtype(sample_format, bits_per_sample);
                check(data_type != Encoding::UNKNOWN, "Data type was not recognized in directory {}", n_directories);
                if (data_type == Encoding::U4)
                    shape[1] *= 2;
            }

            // Strips.
            {
                u32 tmp;
                check(TIFFGetField(tiff, TIFFTAG_ROWSPERSTRIP, &tmp) == 1,
                      "Only images divided in strips are currently supported. "
                      "Tiled images are currently not supported");
            }

            // Compression.
            u16 dir_compression{};
            check(TIFFGetField(tiff, TIFFTAG_COMPRESSION, &dir_compression));

            if (n_directories) { // check no mismatch with the other directories
                check(
                    noa::all(allclose(pixel_size, m_spacing)) and
                    shape[0] == m_shape[1] and
                    shape[1] == m_shape[2] and
                    data_type == m_dtype and
                    dir_compression == tiff_compression,
                    "Mismatch detected. Directories with different data type, shape, pixel sizes, or compression, are not supported"
                );
            } else { // save to header
                m_dtype = data_type;
                m_spacing = pixel_size;
                m_shape[1] = shape[0];
                m_shape[2] = shape[1];
                tiff_compression = dir_compression;
            }
            ++n_directories;
        }
        check(s_error_buffer.empty(), "Error occurred while reading directories. {}", s_error_buffer);
        m_shape[0] = n_directories;

        // Extract the compression.
        Compression compression{};
        switch (tiff_compression) {
            case COMPRESSION_NONE:
                compression = Compression::NONE;
                break;
            case COMPRESSION_LZW:
                compression = Compression::LZW;
                break;
            case COMPRESSION_DEFLATE:
                compression = Compression::DEFLATE;
                break;
            default:
                compression = Compression::UNKNOWN;
        }

        // Don't bother resetting the current directory to the first one,
        // read/write operations will reset the directory whenever necessary.

        return make_tuple(
            Shape<i64, 4>{m_shape[0], 1, m_shape[1], m_shape[2]},
            m_spacing.as<f64>().push_front(m_spacing[0] == 0 and m_spacing[1] == 0 ? 0 : 1),
            m_dtype, compression
        );
    }

    auto ImageFileEncoderTiff::write_header(
        std::FILE* file,
        const Shape<i64, 4>& shape,
        const Vec<f64, 3>& spacing,
        Encoding::Type dtype,
        Compression compression
    ) -> Tuple<Encoding::Type, Compression> {
        check(shape[1] == 1, "TIFF files do not support 3d volumes, but got BDHW shape={}", shape);
        check(not shape.is_empty(), "Empty shapes are invalid, but got {}", shape);
        check(all(spacing >= 0), "The pixel size should be positive, got {}", spacing);
        check(dtype != Encoding::UNKNOWN, "The data type is set to unknown");
        check(compression != Compression::UNKNOWN, "The compression scheme is set to unknown");

        m_shape = shape.filter(0, 2, 3);
        m_spacing = spacing.filter(1, 2).as<f32>();
        m_dtype = dtype;
        m_compression = compression;

        // Prepare the handles and open the tiff client.
        close();
        if (m_handles == nullptr)
            m_handles = std::make_unique<handle_type[]>(NOA_TIFF_MAX_HANDLES);

        // Open in read-write-truncate since the TIFF client needs to be able
        // to read from the stream even in writing mode.
        m_handles[0].first = Handle{.mutex = &m_mutex, .file = file};
        auto tiff = tiff_client_init(&m_handles[0].first, "w+");
        m_handles[0].second = tiff;

        // libtiff is so annoying to use, for now disallow reading from new files.
        m_is_write = true;

        // Every dtype and compression are supported.
        return make_tuple(m_dtype, m_compression);
    }

    void ImageFileEncoderTiff::close() const {
        if (m_handles != nullptr) {
            // Close the TIFF clients, but don't deallocate in case we need the encoder again.
            // Importantly, the handles are saved sequentially, so stop at the first nullptr.
            for (size_t i{}; i < NOA_TIFF_MAX_HANDLES; ++i) {
                auto& tiff = m_handles[i].second;
                if (tiff == nullptr)
                    return;
                ::TIFFClose(static_cast<::TIFF*>(tiff));
                tiff = nullptr;
            }
            check(s_error_buffer.empty(), fmt::runtime(s_error_buffer));
        }
    }

    template<typename T>
    void ImageFileEncoderTiff::decode(
        std::FILE* file,
        const Span<T, 4>& output,
        const Vec<i64, 2>& bd_offset,
        bool clamp,
        i32 n_threads
    ) {
        check(m_handles, "Encoder is not initialized");
        check(file == m_handles[0].first.file, "File stream mismatch");

        const auto [b, d, h, w] = output.shape();
        const i64 start = bd_offset[0];
        const i64 end = bd_offset[0] + b;

        check(m_shape[1] == h and m_shape[2] == w,
              "Cannot read 2d slice(s) with shape={} from a file with 2d slice(s) with shape{}",
              output.shape().filter(2, 3), m_shape.pop_front());
        check(d == 1 and bd_offset[1] == 0,
              "Can only read 2d slice(s), but asked to read shape={}, bd_offset={}",
              output.shape(), bd_offset);
        check(m_shape[0] >= end,
              "The file has less slices ({}) that what is about to be read (start:{}, count:{})",
              m_shape[0], start, b);

        check(not m_is_write, "Cannot read slices from a newly created file");

        // One thread per directory, at most.
        // Each thread has its own TIFF file, all pointing to the same file stream.
        n_threads = std::min(n_threads, static_cast<i32>(NOA_TIFF_MAX_HANDLES));
        n_threads = std::min(n_threads, static_cast<i32>(b));
        for (size_t i = 1; i < clamp_cast<size_t>(n_threads); ++i) {
            if (m_handles[i].second == nullptr) {
                m_handles[i].first = Handle{.mutex = &m_mutex, .file = file};
                m_handles[i].second = tiff_client_init(&m_handles[i].first, "r");
            }
        }

        // Per-thread strip buffer.
        std::unique_ptr<std::byte[]> buffer;
        tsize_t current_strip_size{};

        #pragma omp parallel for num_threads(n_threads) default(shared) private(buffer) firstprivate(current_strip_size)
        for (i64 slice = start; slice < end; ++slice) {
            #ifdef NOA_ENABLE_OPENMP
            // Check that OpenMP launched the correct number of threads, otherwise it's a segfault...
            const i32 tid = omp_get_thread_num();
            check(tid < n_threads);
            auto tiff = static_cast<::TIFF*>(m_handles[static_cast<size_t>(tid)].second);
            #else
            auto tiff = static_cast<::TIFF*>(m_handles[0].second);
            #endif
            check(::TIFFSetDirectory(tiff, static_cast<tdir_t>(slice)));

            // Allocate enough memory to store decoded strip.
            const tsize_t strip_size = ::TIFFStripSize(tiff);
            if (strip_size > current_strip_size) {
                buffer = std::make_unique<Byte[]>(safe_cast<size_t>(strip_size));
                current_strip_size = strip_size;
            }

            i64 row_offset = 0;
            for (tstrip_t strip = 0; strip < ::TIFFNumberOfStrips(tiff); ++strip) {
                // TIFF-decode the strip.
                const auto n_bytes = ::TIFFReadEncodedStrip(tiff, strip, buffer.get(), strip_size);
                check(n_bytes != -1,
                      "An error occurred while reading slice={}, strip={}. {}",
                      slice, strip, s_error_buffer);

                // Convert the n_bytes read to number n_rows.
                // Note that the last strip may have fewer rows than the others.
                const auto n_bytes_per_element = Encoding::encoded_size(m_dtype, 1);
                check(is_multiple_of(n_bytes, n_bytes_per_element));
                const auto n_elements = n_bytes / n_bytes_per_element;
                check(is_multiple_of(n_elements, m_shape[2]));
                const auto n_rows = n_elements / m_shape[2];

                // Convert and transfer to the output.
                io::decode(
                    SpanContiguous(buffer.get(), n_bytes),
                    Encoding{.dtype = Encoding::to_dtype<T>(), .clamp = clamp, .endian_swap = false},
                    output.subregion(slice, 0, ni::Slice{row_offset, row_offset + n_rows}),
                    1 // single-threaded
                );
                row_offset += n_rows;
            }

            // The origin must be in the bottom left corner.
            u16 orientation{};
            ::TIFFGetFieldDefaulted(tiff, TIFFTAG_ORIENTATION, &orientation);
            if (orientation == ORIENTATION_TOPLEFT) { // this is the default
                if (output.stride(0) == 1)
                    flip_rows_(output.subregion(slice).filter(2, 3).as_contiguous());
                else
                    flip_rows_(output.subregion(slice).filter(2, 3));
            } else if (orientation != ORIENTATION_BOTLEFT) {
                panic("The orientation of the slice {} is not supported. "
                      "The origin should be at the bottom left (fastest) or top left", slice);
            }
        }
    }

    template<typename T>
    void ImageFileEncoderTiff::encode(
        std::FILE* file,
        const Span<const T, 4>& input,
        const Vec<i64, 2>& bd_offset,
        bool clamp,
        i32 n_threads
    ) {
        (void) n_threads; // no multithreading for you
        check(file == m_handles[0].first.file, "File stream mismatch");

        const auto [b, d, h, w] = input.shape();
        const i64 start = bd_offset[0];
        const i64 end = bd_offset[0] + b;

        check(m_shape[1] == h and m_shape[2] == w,
              "Cannot write 2d slice(s) with shape={} from a file with 2d slice(s) with shape{}",
              input.shape().filter(2, 3), m_shape.pop_front());
        check(d == 1 and bd_offset[1] == 0,
              "Can only read 2d slice(s), but asked to read shape={}, bd_offset={}",
              input.shape(), bd_offset);
        check(m_shape[0] >= end,
              "The file has less slices ({}) that what is about to be writen (start:{}, count:{})",
              m_shape[0], start, b);

        u16 tiff_compression{};
        switch (m_compression) {
            case Compression::NONE:
                tiff_compression = COMPRESSION_NONE;
                break;
            case Compression::LZW:
                tiff_compression = COMPRESSION_LZW;
                break;
            case Compression::DEFLATE:
                tiff_compression = COMPRESSION_DEFLATE;
                break;
            default:
                panic("The compression is not set");
        }

        check(start == current_directory, "Slices should be written in sequential order");

        // TODO This is another reason for ditching libtiff because looking at the TIFF specification
        //      it seems that since we know the shape of the file in advance, we could easily organize
        //      the directory metadata in one block then add each data block at any location in the file.
        //      Plus, all of this could be done very easily in parallel. libtiff just wasn't design for that.

        // Target 64KB per strip. Ensure strip is multiple of a line and if too many strips,
        // increase strip size (double or more).
        const i64 n_bytes_per_row = m_shape[2] * Encoding::encoded_size(m_dtype, 1);
        i64 n_rows_per_strip = divide_up(i64{65'536}, n_bytes_per_row);
        i64 n_strips = divide_up(i64{m_shape[1]}, n_rows_per_strip);
        if (n_strips > 4096) {
            n_rows_per_strip *= (1 + m_shape[1] / 4096);
            n_strips = divide_up(n_rows_per_strip, i64{m_shape[1]});
        }

        const i64 bytes_per_strip = n_rows_per_strip * n_bytes_per_row;
        const auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(bytes_per_strip));

        auto tiff = static_cast<::TIFF*>(m_handles[0].second);
        for (i64 slice = start; slice < end; ++slice) {
            check(::TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, m_shape[2]));
            check(::TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, m_shape[1]));
            check(::TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, n_rows_per_strip));

            // https://libtiff.gitlab.io/libtiff/functions/TIFFGetField.html:
            // TIFFTAG_XRESOLUTION is a float, but GCC warns of an implicit conversion to double...
            #if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
            #   pragma GCC diagnostic push
            #   pragma GCC diagnostic ignored "-Wdouble-promotion"
            #endif
            if (not all(allclose(m_spacing, 0))) {
                check(::TIFFSetField(tiff, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER));
                check(::TIFFSetField(tiff, TIFFTAG_XRESOLUTION, 1.e8f / m_spacing[1]));
                check(::TIFFSetField(tiff, TIFFTAG_YRESOLUTION, 1.e8f / m_spacing[0]));
            }
            #if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
            #   pragma GCC diagnostic pop
            #endif

            check(::TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG));
            check(::TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK));
            check(::TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 1));

            u16 sample_format{}, bits_per_sample{};
            tiff_set_dtype(m_dtype, &sample_format, &bits_per_sample);
            check(::TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, sample_format));
            check(::TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, bits_per_sample));

            check(::TIFFSetField(tiff, TIFFTAG_COMPRESSION, tiff_compression));

            // FIXME Many cryoEM software (RELION, IMOD) are reading TIFFs incorrectly because
            //       they do not check for the orientation and are assuming the default (TOPLEFT).
            //       As such, this will not read this file correctly...
            check(::TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_BOTLEFT));

            i64 irow{};
            for (tstrip_t strip = 0; strip < n_strips; ++strip) {
                io::encode(
                    input.subregion(slice, 0, ni::Slice{irow, irow + n_rows_per_strip}),
                    SpanContiguous(buffer.get(), bytes_per_strip),
                    Encoding{.dtype = Encoding::to_dtype<T>(), .clamp = clamp, .endian_swap = false},
                    1 // single-threaded
                );

                check(::TIFFWriteEncodedStrip(tiff, strip, buffer.get(), static_cast<tmsize_t>(bytes_per_strip)) != -1,
                      "An error occurred while writing slice={}, strip={}. {}",
                      slice, strip, s_error_buffer);

                irow += n_rows_per_strip;
            }
            check(::TIFFWriteDirectory(tiff), "Failed to write slice={}, {}", slice, s_error_buffer);
            current_directory += 1;
        }
    }

    #define NOA_ENCODERTIFF_(T)                                                                                         \
    template void ImageFileEncoderTiff::encode<T>(std::FILE*, const Span<const T, 4>&, const Vec<i64, 2>&, bool, i32);  \
    template void ImageFileEncoderTiff::decode<T>(std::FILE*, const Span<T, 4>&, const Vec<i64, 2>&, bool, i32)

    NOA_ENCODERTIFF_(i8);
    NOA_ENCODERTIFF_(u8);
    NOA_ENCODERTIFF_(i16);
    NOA_ENCODERTIFF_(u16);
    NOA_ENCODERTIFF_(i32);
    NOA_ENCODERTIFF_(u32);
    NOA_ENCODERTIFF_(i64);
    NOA_ENCODERTIFF_(u64);
    NOA_ENCODERTIFF_(f16);
    NOA_ENCODERTIFF_(f32);
    NOA_ENCODERTIFF_(f64);
    NOA_ENCODERTIFF_(c16);
    NOA_ENCODERTIFF_(c32);
    NOA_ENCODERTIFF_(c64);
}
#endif
