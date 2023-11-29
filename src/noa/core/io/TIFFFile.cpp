#if NOA_ENABLE_TIFF

#include <tiffio.h>

#include "noa/core/Types.hpp"
#include "noa/core/io/OS.hpp"
#include "noa/core/io/TIFFFile.hpp"

// TODO Surely there's a more modern library we could use here?

namespace {
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
    extern "C" void errorHandler_(const char* module, const char* fmt, va_list args) {
        s_error_buffer.clear();
        format_message_tiff_(s_error_buffer, module, fmt, args);
    }

    extern "C" void warningHandler_(const char*, const char*, va_list) {
        // For now, ignore warnings...
    }

    // We don't expose the TIFF* in the API, so cast it here.
    // This is of course undefined if ptr is not a TIFF*...
    ::TIFF* get_tiff_(void* ptr) {
        return static_cast<::TIFF*>(ptr);
    }
}

namespace noa::io {
    void TIFFFile::open_(const Path& filename, open_mode_t open_mode) {
        close();

        check(io::is_valid_open_mode(open_mode), "File: {}. Invalid open mode", filename);
        const bool write = open_mode & io::WRITE;
        m_is_read = open_mode & io::READ;
        if (write) {
            check(!m_is_read,
                  "File: {}. Opening a TIFF file in READ|WRITE mode is not supported. "
                  "Should be READ or WRITE", filename);
        } else if (!m_is_read) {
            panic("File: {}. Open mode is not supported. Should be READ or WRITE", filename);
        }

        try {
            if (write && is_file(filename))
                backup(filename, true);
            else
                mkdir(filename.parent_path());
        } catch (...) {
            panic("File: {}. Mode: {}. Could not open the file because of an OS failure",
                  filename, OpenModeStream{open_mode});
        }

        for (u32 it{0}; it < 5; ++it) {
            // h: Read TIFF header only, do not load the first image directory.
            // c: Disable the use of strip chopping when reading images
            m_tiff = TIFFOpen(filename.c_str(), write ? "rhc" : "w");
            if (m_tiff) {
                if (m_is_read) {
                    try {
                        read_header_();
                    } catch (...) {
                        panic("File: {}. Mode:{}. Failed while reading the header",
                              filename, OpenModeStream{open_mode});
                    }
                }
                m_filename = filename;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (m_is_read && !is_file(filename)) {
            panic("File: {}. Mode: {}. Failed to open the file. The file does not exist",
                  filename, OpenModeStream{open_mode});
        }
        panic("File: {}. Mode: {}. Failed to open the file. {}",
              filename, OpenModeStream{open_mode}, s_error_buffer);
    }

    void TIFFFile::close_() {
        ::TIFFClose(get_tiff_(m_tiff));
        if (!s_error_buffer.empty())
            panic("File: {}. An error has occurred while closing the file. {}", m_filename, s_error_buffer);
        m_tiff = nullptr;
        m_filename.clear();
    }

    // The logic comes from IMOD/libiimod/iitif.c::iiTIFFCheck
    void TIFFFile::read_header_() {
        ::TIFF* tiff = get_tiff_(m_tiff);
        uint16_t directories{};
        while (::TIFFSetDirectory(tiff, directories)) {
            // Shape:
            Shape2<u32> shape; // height-width
            if (!::TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, shape.data() + 1) ||
                !::TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, shape.data())) {
                panic("The input TIFF file does not have the width or height field.");
            }

            Vec2<f32> pixel_size{1.f}; // height-width ang/pixel
            { // Pixel sizes:
                u16 resolution_unit{}; // 1: no units, 2: inch, 3: cm
                const auto has_resolution = TIFFGetField(tiff, TIFFTAG_XRESOLUTION, &pixel_size[1]);
                TIFFGetFieldDefaulted(tiff, TIFFTAG_RESOLUTIONUNIT, &resolution_unit);
                if (resolution_unit > 1 && has_resolution) {
                    if (!TIFFGetField(tiff, TIFFTAG_YRESOLUTION, pixel_size.data()))
                        pixel_size[0] = pixel_size[1];
                    const auto scale = resolution_unit == 2 ? 2.54e8f : 1.00e8f;
                    pixel_size = scale / pixel_size;
                }
            }

            DataType data_type{};
            { // Data type:
                u16 photometry{}, sample_per_pixel{}, sample_format{}, bits_per_sample{}, planar_config{};
                ::TIFFGetField(tiff, TIFFTAG_PHOTOMETRIC, &photometry);
                ::TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLESPERPIXEL, &sample_per_pixel);
                ::TIFFGetFieldDefaulted(tiff, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
                ::TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &sample_format);
                ::TIFFGetFieldDefaulted(tiff, TIFFTAG_PLANARCONFIG, &planar_config);
                if (!s_error_buffer.empty())
                    panic("Error occurred while reading sampling fields of directory {}. {}",
                          directories, s_error_buffer);
                else if (planar_config != PLANARCONFIG_CONTIG)
                    panic("Planar configuration separate is not supported. Should be contiguous");
                else if (photometry > 2)
                    panic("Photometry not supported. Should be bi-level or grayscale");
                else if (sample_per_pixel > 1)
                    panic("Samples per pixel should be 1. {} is not supported", sample_per_pixel);
                data_type = get_dtype_(sample_format, bits_per_sample);
                if (data_type == DataType::UNKNOWN)
                    panic("Data type was not recognized in directory {}", directories);
                else if (data_type == DataType::U4)
                    shape[1] *= 2;
            }

            if (directories) { // check no mismatch
                if (any(pixel_size != m_pixel_size) || shape[0] != m_shape[0] ||
                    shape[1] != m_shape[1] || data_type != m_data_type)
                    panic("Mismatch detected. Directories with different data type, shape, or"
                          "pixel sizes are not supported");

            } else { // save to header
                m_data_type = data_type;
                m_pixel_size = pixel_size;
                m_shape[1] = shape[0];
                m_shape[2] = shape[1];
            }
            ++directories;
        }
        if (!s_error_buffer.empty())
            panic("Error occurred while reading directories. {}", s_error_buffer);

        // At this point the current directory is the last one. This is OK since read/write operations
        // will reset the directory based on the desired section/strip.
        m_shape[0] = directories;
    }

    TIFFFile::TIFFFile() {
        static thread_local bool thread_is_set{};
        if (!thread_is_set) {
            s_error_buffer.reserve(250);
            ::TIFFSetErrorHandler(errorHandler_);
            ::TIFFSetWarningHandler(warningHandler_);
            thread_is_set = true;
        }
    }

    void TIFFFile::set_shape(const Shape4<i64>& shape) {
        check(!m_is_read, "Trying to change the shape of the data in read mode is not allowed");
        check(shape[1] == 1,
              "TIFF files do not support 3D volumes, but got shape {}. "
              "To set a stack of 2D images, use the batch dimension "
              "instead of the depth", shape);
        check(noa::all(shape > 0), "The shape should be non-zero positive, but got {}", shape);
        m_shape = shape.filter(0, 2, 3).as<u32>();
    }

    void TIFFFile::set_pixel_size(Vec3<f32> pixel_size) {
        check(!m_is_read, "Trying to change the pixel size of the data in read mode is not allowed");
        check(noa::all(pixel_size >= 0), "The pixel size should be positive, got {}", pixel_size);
        m_pixel_size[0] = pixel_size[1];
        m_pixel_size[1] = pixel_size[2];
    }

    void TIFFFile::set_dtype(DataType data_type) {
        check(!m_is_read, "Trying to change the data type of the data in read mode is not allowed");
        m_data_type = data_type;
    }

    std::string TIFFFile::info_string(bool brief) const noexcept {
        if (brief)
            return fmt::format("Shape: {}; Pixel size: {::.3f}", shape(), pixel_size());

        return fmt::format("Format: TIFF File\n"
                           "Shape (batches, depth, height, width): {}\n"
                           "Pixel size (depth, height, width): {::.3f}\n"
                           "Data type: {}",
                           shape(),
                           pixel_size(),
                           m_data_type);
    }

    void TIFFFile::read_elements(void*, DataType, i64, i64, bool) {
        panic("This function is currently not supported");
    }

    void TIFFFile::read_slice(
            void* output,
            const Strides4<i64>& strides,
            const Shape4<i64>& shape,
            DataType data_type,
            i64 start,
            bool clamp
    ) {
        check(is_open(), "The file should be opened");
        check(m_shape[1] == shape[2] && m_shape[2] == shape[3],
              "File: {}. Cannot read a 2D slice of shape {} from a file with 2D slices of shape {}",
              m_filename, shape.filter(2, 3), m_shape.pop_front());
        check(shape[1] == 1,
              "File {}. Can only read 2D slice(s), but asked to read shape {}",
              m_filename, shape);
        check(m_shape[0] >= start + shape[0],
              "File: {}. The file has less slices ({}) that what is about to be read (start:{}, count:{})",
              m_filename, m_shape[0], start, shape[0]);

        // Output as array of bytes:
        auto* output_ptr = static_cast<Byte*>(output);
        const auto o_bytes_per_elements = serialized_size(data_type, 1);
        const auto i_bytes_per_elements = serialized_size(m_data_type, 1);

        // The strip size should not change between slices since we know they have the same shape and data layout.
        // The compression could be different, but worst case scenario, the strip size is not as optimal
        // as it could have been. Since in most cases we expect the directories to have exactly the same
        // tags, allocate once according to the first directory.
        ::TIFF* tiff = get_tiff_(m_tiff);
        std::unique_ptr<Byte[]> buffer, buffer_flip_row;
        for (i64 slice = start; slice < start + shape[0]; ++slice) {
            [[maybe_unused]] const auto err = ::TIFFSetDirectory(tiff, static_cast<u16>(slice));
            NOA_ASSERT(err); // should be fine since we checked boundary

            // A directory can be divided into multiple strips.
            // For every strip, allocate enough memory to get decoded data.
            // Then send it for conversion.
            const tsize_t strip_size = ::TIFFStripSize(tiff);
            if (!buffer)
                buffer = std::make_unique<Byte[]>(static_cast<size_t>(strip_size));

            i64 row_offset = 0;
            for (tstrip_t strip = 0; strip < ::TIFFNumberOfStrips(tiff); ++strip) {
                const auto bytes_read = ::TIFFReadEncodedStrip(tiff, strip, buffer.get(), strip_size);
                if (bytes_read == -1)
                    panic("File: {}. An error occurred while reading slice:{}, strip:{}. {}",
                          m_filename, slice, strip, s_error_buffer);

                // Convert the bytes read in number of rows read:
                NOA_ASSERT(!(bytes_read % i_bytes_per_elements));
                const auto elements_in_buffer = bytes_read / i_bytes_per_elements;
                NOA_ASSERT(elements_in_buffer % m_shape[2]);
                const auto rows_in_buffer = elements_in_buffer / m_shape[2];
                const auto shape_buffer = Shape4<i64>{1, 1, rows_in_buffer, m_shape[2]};

                // Convert and transfer to output:
                const auto output_offset = ni::offset_at(slice, 0, row_offset, strides) * o_bytes_per_elements;
                try {
                    deserialize(buffer.get(), m_data_type,
                                output_ptr + output_offset, data_type, strides,
                                shape_buffer, clamp);
                } catch (...) {
                    panic("File {}. Failed to read strip with shape {} from the file. "
                          "Deserialize from dtype {} to {}", m_filename, shape_buffer, m_data_type, data_type);
                }
                row_offset += rows_in_buffer;
            }

            // Origin must be in the bottom left corner:
            u16 orientation{};
            ::TIFFGetFieldDefaulted(tiff, TIFFTAG_ORIENTATION, &orientation);
            if (orientation == ORIENTATION_TOPLEFT) { // this is the default
                const i64 bytes_per_row = m_shape[2] * o_bytes_per_elements;
                if (!buffer_flip_row)
                    buffer_flip_row = std::make_unique<Byte[]>(static_cast<size_t>(bytes_per_row));
                flip_y_(output_ptr + slice * strides[0] * o_bytes_per_elements, bytes_per_row,
                        m_shape[1], buffer_flip_row.get());
            } else if (orientation != ORIENTATION_BOTLEFT) {
                panic("File: {}. Orientation of the slice(s) is not supported. "
                      "The origin should be at the bottom left or top left", m_filename);
            }
        }
    }

    void TIFFFile::read_slice(void* output, DataType data_type, i64 start, i64 end, bool clamp) {
        NOA_ASSERT(end >= start);
        const auto slice_shape = Shape4<i64>{end - start, 1, m_shape[2], m_shape[3]};
        return read_slice(output, slice_shape.strides(), slice_shape, data_type, start, clamp);
    }

    void TIFFFile::read_all(void* output, const Strides4<i64>& strides, const Shape4<i64>& shape,
                            DataType data_type, bool clamp) {
        check(shape[0] == m_shape[0],
              "The file shape {} is not compatible with the output shape {}", this->shape(), shape);
        return read_slice(output, strides, shape, data_type, 0, clamp);
    }

    void TIFFFile::read_all(void* output, DataType data_type, bool clamp) {
        return read_slice(output, data_type, 0, m_shape[0], clamp);
    }

    void TIFFFile::write_elements(const void*, DataType, i64, i64, bool) {
        panic("This function is currently not supported");
    }

    void TIFFFile::write_slice(
            const void* input,
            const Strides4<i64>& strides,
            const Shape4<i64>& shape,
            DataType data_type,
            i64 start,
            bool clamp) {
        check(is_open(), "The file should be opened");
        check(all(m_shape > 0),
              "File: {}. The shape of the file is not set or is empty. Set the shape first, "
              "and then write a slice to the file", m_filename);
        check(m_shape[1] == shape[2] && m_shape[2] == shape[3],
              "File: {}. Cannot write a 2D slice of shape {} into a file with 2D slices of shape {}",
              m_filename, shape.filter(2, 3), m_shape.pop_front());
        check(shape[1] == 1,
              "File {}. Can only write 2D slice(s), but asked to write shape {}",
              m_filename, shape);
        check(m_shape[0] >= start + shape[0],
              "File: {}. The file has less slices ({}) that what is about to be written (start:{}, count:{})",
              m_filename, m_shape[0], start, shape[0]);

        if (m_data_type == DataType::UNKNOWN)
            m_data_type = data_type;

        // Output as array of bytes:
        const auto* input_ptr = static_cast<const Byte*>(input);
        const i64 i_bytes_per_elements = io::serialized_size(data_type, 1);
        const i64 o_bytes_per_elements = io::serialized_size(m_data_type, 1);

        // Target 8K per strip. Ensure strip is multiple of a line and if too many strips,
        // increase strip size (double or more).
        const i64 bytes_per_row = m_shape[2] * o_bytes_per_elements;
        i64 rows_per_strip = divide_up(i64{8192}, bytes_per_row);
        i64 strip_count = divide_up(i64{m_shape[1]}, rows_per_strip);
        if (strip_count > 4096) {
            rows_per_strip *= (1 + m_shape[1] / 4096);
            strip_count = divide_up(rows_per_strip, i64{m_shape[1]});
        }
        const auto strip_shape = Shape4<i64>{1, 1, rows_per_strip, m_shape[2]};
        const i64 bytes_per_strip = rows_per_strip * bytes_per_row;
        const auto buffer = std::make_unique<Byte[]>(static_cast<size_t>(bytes_per_strip));

        NOA_ASSERT(shape[0] >= start);
        ::TIFF* tiff = get_tiff_(m_tiff);
        for (i64 slice = start; slice < shape[0] + start; ++slice) {
            ::TIFFSetDirectory(tiff, static_cast<u16>(slice));

            { // Set up relevant tags
                ::TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, m_shape[2]);
                ::TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, m_shape[1]);
                ::TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, rows_per_strip);

                if (any(m_pixel_size != 0.f)) {
                    ::TIFFSetField(tiff, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);
                    ::TIFFSetField(tiff, TIFFTAG_XRESOLUTION, static_cast<double>(1.e8f / m_pixel_size[1]));
                    ::TIFFSetField(tiff, TIFFTAG_YRESOLUTION, static_cast<double>(1.e8f / m_pixel_size[0]));
                }

                ::TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
                ::TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
                ::TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 1);

                u16 sample_format{}, bits_per_sample{};
                set_dtype_(m_data_type, &sample_format, &bits_per_sample);
                ::TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, sample_format);
                ::TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, bits_per_sample);

                // For now, no compression.
                ::TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE);

                // TODO I don't get why IMOD doesn't check the orientation...
                ::TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_BOTLEFT);

                if (!s_error_buffer.empty())
                    panic("File: {}. An error has occurred while setting up the tags. {}",
                          m_filename, s_error_buffer);
            }

            for (tstrip_t strip = 0; strip < strip_count; ++strip) {
                Byte* i_buffer = buffer.get() + strip * bytes_per_strip;
                const i64 input_offset = ni::offset_at(slice, 0, strip * rows_per_strip, strides);
                try {
                    io::serialize(input_ptr + input_offset * i_bytes_per_elements,
                                  data_type, strides, strip_shape,
                                  i_buffer, m_data_type, clamp);
                } catch (...) {
                    panic("File {}. Failed to write strip with shape {} from the file. "
                          "Serialize from dtype {} to {}", m_filename, strip_shape, data_type, m_data_type);
                }
                if (::TIFFWriteEncodedStrip(tiff, strip, i_buffer, static_cast<tmsize_t>(bytes_per_strip)) == -1)
                    panic("File: {}. An error occurred while writing slice:{}, strip:{}. {}",
                          m_filename, slice, strip, s_error_buffer);
            }
            if (!::TIFFWriteDirectory(tiff))
                panic("File: {}. Failed to write slice {} into the file", m_filename, slice);
        }
    }

    void TIFFFile::write_slice(const void* input, DataType data_type, i64 start, i64 end, bool clamp) {
        NOA_ASSERT(end >= start);
        const auto slice_shape = Shape4<i64>{end - start, 1, m_shape[1], m_shape[2]};
        return write_slice(input, slice_shape.strides(), slice_shape, data_type, start, clamp);
    }

    void TIFFFile::write_all(
            const void* input, const Strides4<i64>& strides, const Shape4<i64>& shape,
            DataType data_type, bool clamp
    ) {
        if (all(m_shape == 0)) // first write, set the shape
            this->set_shape(shape);
        return write_slice(input, strides, shape, data_type, 0, clamp);
    }

    void TIFFFile::write_all(const void* input, DataType data_type, bool clamp) {
        return write_slice(input, data_type, 0, m_shape[0], clamp);
    }

    DataType TIFFFile::get_dtype_(u16 sample_format, u16 bits_per_sample) {
        switch (sample_format) {
            case SAMPLEFORMAT_INT:
                if (bits_per_sample == 8) {
                    return DataType::I8;
                } else if (bits_per_sample == 16) {
                    return DataType::I16;
                } else if (bits_per_sample == 32) {
                    return DataType::I32;
                }
                break;
            case SAMPLEFORMAT_UINT:
                if (bits_per_sample == 8) {
                    return DataType::U8;
                } else if (bits_per_sample == 16) {
                    return DataType::U16;
                } else if (bits_per_sample == 32) {
                    return DataType::U32;
                } else if (bits_per_sample == 4) {
                    return DataType::U4;
                }
                break;
            case SAMPLEFORMAT_IEEEFP:
                if (bits_per_sample == 16) {
                    return DataType::F16;
                } else if (bits_per_sample == 32) {
                    return DataType::F32;
                } else if (bits_per_sample == 64) {
                    return DataType::F64;
                }
                break;
            case SAMPLEFORMAT_COMPLEXINT:
                if (bits_per_sample == 32) {
                    return DataType::CI16;
                }
                break;
            case SAMPLEFORMAT_COMPLEXIEEEFP:
                if (bits_per_sample == 32) {
                    return DataType::C16;
                } else if (bits_per_sample == 64) {
                    return DataType::C32;
                } else if (bits_per_sample == 128) {
                    return DataType::C64;
                }
                break;
            default:
                break;
        }
        return DataType::UNKNOWN;
    }

    void TIFFFile::set_dtype_(DataType data_type, u16* sample_format, u16* bits_per_sample) {
        switch (data_type) {
            case DataType::I8:
                *bits_per_sample = 8;
                *sample_format = SAMPLEFORMAT_INT;
                break;
            case DataType::U8:
                *bits_per_sample = 8;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case DataType::I16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_INT;
                break;
            case DataType::U16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case DataType::I32:
                *bits_per_sample = 32;
                *sample_format = SAMPLEFORMAT_INT;
                break;
            case DataType::U32:
                *bits_per_sample = 32;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case DataType::I64:
                *bits_per_sample = 64;
                *sample_format = SAMPLEFORMAT_INT;
                break;
            case DataType::U64:
                *bits_per_sample = 64;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case DataType::F16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_IEEEFP;
                break;
            case DataType::F32:
                *bits_per_sample = 32;
                *sample_format = SAMPLEFORMAT_IEEEFP;
                break;
            case DataType::F64:
                *bits_per_sample = 64;
                *sample_format = SAMPLEFORMAT_IEEEFP;
                break;
            case DataType::C16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_COMPLEXIEEEFP;
                break;
            case DataType::C32:
                *bits_per_sample = 32;
                *sample_format = SAMPLEFORMAT_COMPLEXIEEEFP;
                break;
            case DataType::C64:
                *bits_per_sample = 64;
                *sample_format = SAMPLEFORMAT_COMPLEXIEEEFP;
                break;
            case DataType::U4:
                *bits_per_sample = 4;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case DataType::CI16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_COMPLEXINT;
                break;
            default:
                *bits_per_sample = 1; // default
                *sample_format = SAMPLEFORMAT_VOID;
                break;
        }
    }

    void TIFFFile::flip_y_(Byte* slice, i64 bytes_per_row, i64 rows, Byte* buffer) {
        for (i64 row = 0; row < rows / 2; ++row) {
            Byte* current_row = slice + row * bytes_per_row;
            Byte* opposite_row = slice + (rows - row - 1) * bytes_per_row;
            const auto bytes = static_cast<size_t>(bytes_per_row);
            std::memcpy(buffer, current_row, bytes);
            std::memcpy(current_row, opposite_row, bytes);
            std::memcpy(opposite_row, buffer, bytes);
        }
    }
}

#endif // NOA_ENABLE_TIFF
