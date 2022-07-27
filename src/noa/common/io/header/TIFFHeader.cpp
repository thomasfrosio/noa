#if NOA_ENABLE_TIFF

#include "noa/Session.h"
#include "noa/common/Types.h"
#include "noa/common/OS.h"
#include "noa/common/io/header/TIFFHeader.h"

namespace {
    // One string per thread is enough since TIFFHeader will immediately throw after the error.
    thread_local std::string s_error_buffer;

    void formatMessageTIFF_(std::string& output, const char* module, const char* fmt, va_list args) {
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

    // if s_error_buffer is not empty, there's an error waiting to be printed.
    extern "C" void errorHandler_(const char* module, const char* fmt, va_list args) {
        s_error_buffer.clear();
        formatMessageTIFF_(s_error_buffer, module, fmt, args);
    }

    extern "C" void warningHandler_(const char* module, const char* fmt, va_list args) {
        std::string tmp;
        formatMessageTIFF_(tmp, module, fmt, args);
        noa::Session::logger.warn(tmp);
    }
}

namespace noa::io::details {
    void TIFFHeader::open(const path_t& filename, open_mode_t open_mode) {
        close();

        bool write = open_mode & io::WRITE;
        m_is_read = open_mode & io::READ;
        if (write) {
            if (m_is_read)
                NOA_THROW("Opening a TIFF file in READ|WRITE mode is not supported. Should be READ or WRITE");
        } else if (!m_is_read) {
            NOA_THROW("Open mode not recognized. Should be READ or WRITE");
        }

        try {
            if (write && os::existsFile(filename))
                os::backup(filename, true);
            else
                os::mkdir(filename.parent_path());
        } catch (...) {
            NOA_THROW("OS failure when trying to open the file");
        }

        for (uint32_t it{0}; it < 5; ++it) {
            // h: Read TIFF header only, do not load the first image directory.
            // c: Disable the use of strip chopping when reading images
            m_tiff = TIFFOpen(filename.c_str(), write ? "rhc" : "w");
            if (m_tiff) {
                if (m_is_read)
                    readHeader_();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        NOA_THROW(s_error_buffer);
    }

    void TIFFHeader::close() {
        ::TIFFClose(m_tiff);
        if (!s_error_buffer.empty())
            NOA_THROW("An error has occurred while closing the file. {}", s_error_buffer);
    }

    // The logic comes from IMOD/libiimod/iitif.c::iiTIFFCheck
    void TIFFHeader::readHeader_() {
        uint16_t directories{};
        while (::TIFFSetDirectory(m_tiff, directories)) {
            // Shape:
            uint2_t shape; // height-width
            if (!::TIFFGetField(m_tiff, TIFFTAG_IMAGEWIDTH, shape.get() + 1) ||
                !::TIFFGetField(m_tiff, TIFFTAG_IMAGELENGTH, shape.get())) {
                NOA_THROW("The input TIFF file does not have the width or height field.");
            }

            float2_t pixel_size{1.f}; // height-width ang/pixel
            { // Pixel sizes:
                uint16_t resolution_unit; // 1: no units, 2: inch, 3: cm
                int has_resolution = TIFFGetField(m_tiff, TIFFTAG_XRESOLUTION, &pixel_size[1]);
                TIFFGetFieldDefaulted(m_tiff, TIFFTAG_RESOLUTIONUNIT, &resolution_unit);
                if (resolution_unit > 1 && has_resolution) {
                    if (!TIFFGetField(m_tiff, TIFFTAG_YRESOLUTION, &pixel_size[0]))
                        pixel_size[0] = pixel_size[1];
                    float scale = resolution_unit == 2 ? 2.54e8f : 1.00e8f;
                    pixel_size = scale / pixel_size;
                }
            }

            DataType data_type;
            { // Data type:
                uint16_t photometry, sample_per_pixel, sample_format, bits_per_sample, planar_config;
                ::TIFFGetField(m_tiff, TIFFTAG_PHOTOMETRIC, &photometry);
                ::TIFFGetFieldDefaulted(m_tiff, TIFFTAG_SAMPLESPERPIXEL, &sample_per_pixel);
                ::TIFFGetFieldDefaulted(m_tiff, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
                ::TIFFGetFieldDefaulted(m_tiff, TIFFTAG_SAMPLEFORMAT, &sample_format);
                ::TIFFGetFieldDefaulted(m_tiff, TIFFTAG_PLANARCONFIG, &planar_config);
                if (!s_error_buffer.empty())
                    NOA_THROW("Error occurred while reading sampling fields of directory {}. {}",
                              directories, s_error_buffer);
                else if (planar_config != PLANARCONFIG_CONTIG)
                    NOA_THROW("Planar configuration separate is not supported. Should be contiguous");
                else if (photometry > 2)
                    NOA_THROW("Photometry not supported. Should be bi-level or grayscale");
                else if (sample_per_pixel > 1)
                    NOA_THROW("Samples per pixel should be 1. {} is not supported", sample_per_pixel);
                data_type = getDataType_(sample_format, bits_per_sample);
                if (data_type == DATA_UNKNOWN)
                    NOA_THROW("Data type was not recognized in directory {}", directories);
                else if (data_type == DataType::UINT4)
                    shape[1] *= 2;
            }

            if (directories) { // check no mismatch
                if (any(pixel_size != m_pixel_size) || shape[0] != m_shape[0] ||
                    shape[1] != m_shape[1] || data_type != m_data_type)
                    NOA_THROW("Mismatch detected. Directories with different data type, shape, or"
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
            NOA_THROW("Error occurred while reading directories. {}", s_error_buffer);

        // At this point the current directory is the last one. This is OK since read/write operations
        // will reset the directory based on the desired section/strip.
        m_shape[0] = directories;
    }

    TIFFHeader::TIFFHeader() {
        static thread_local bool thread_is_set{};
        if (!thread_is_set) {
            s_error_buffer.reserve(250);
            ::TIFFSetErrorHandler(errorHandler_);
            ::TIFFSetWarningHandler(warningHandler_);
            thread_is_set = true;
        }
    }

    void TIFFHeader::shape(size4_t shape) {
        if (m_is_read)
            NOA_THROW("Trying to change the shape of the data in read mode is not allowed");
        else if (shape[1] > 1)
            NOA_THROW("TIFF files do not support 3D volumes, but got shape {}. To set a stack of 2D images, "
                      "use the batch dimension instead of the depth", shape);
        m_shape = uint3_t{shape[0], shape[2], shape[3]};
    }

    void TIFFHeader::pixelSize(float3_t pixel_size) {
        if (m_is_read)
            NOA_THROW("Trying to change the pixel size of the data in read mode is not allowed");
        m_pixel_size[0] = pixel_size[1];
        m_pixel_size[1] = pixel_size[2];
    }

    void TIFFHeader::dtype(DataType data_type) {
        if (m_is_read)
            NOA_THROW("Trying to change the data type of the data in read mode is not allowed");
        m_data_type = data_type;
    }

    std::string TIFFHeader::infoString(bool brief) const noexcept {
        if (brief)
            return string::format("Shape: {}; Pixel size: {:.3f}", shape(), pixelSize());

        return string::format("Format: TIFF File\n"
                              "Shape (batches, depth, height, width): {}\n"
                              "Pixel size (depth, height, width): {:.3f}\n"
                              "Data type: {}",
                              shape(),
                              pixelSize(),
                              m_data_type);
    }

    void TIFFHeader::read(void*, DataType, size_t, size_t, bool) {
        NOA_THROW("This function is currently not supported");
    }

    void TIFFHeader::readSlice(void* output, size4_t strides, size4_t shape,
                               DataType data_type, size_t start, bool clamp) {
        if (m_shape[1] != shape[2] || m_shape[2] != shape[3]) {
            NOA_THROW("The file shape {} is not compatible with the output shape {}", this->shape(), shape);
        } else if (shape[1] != 1) {
            NOA_THROW("The file shape {} describes a (stack of) 2D image(s). To read a slice, the provided "
                      "shape should have a depth of 1, but got shape {}", this->shape(), shape);
        } else if (m_shape[0] <= start + shape[0]) {
            NOA_THROW("The file has less slices ({}) that what is about to be read (start:{}, count:{})",
                      m_shape[0], start, shape[0]);
        }

        // Output as array of bytes:
        auto* output_ptr = static_cast<byte_t*>(output);
        const size_t o_bytes_per_elements = serializedSize(data_type, 1);
        const size_t i_bytes_per_elements = serializedSize(m_data_type, 1);

        // The strip size should not change between slices since we know they have the same shape and data layout.
        // The compression could be different, but worst case scenario, the strip size is not as optimal
        // as it could have been. Since in most cases we expect the directories to have exactly the same
        // tags, allocate once according to the first directory.
        std::unique_ptr<byte_t[]> buffer, buffer_flip_row;
        for (size_t slice = start; slice < start + shape[0]; ++slice) {
            [[maybe_unused]] const int err = ::TIFFSetDirectory(m_tiff, static_cast<uint16_t>(slice));
            NOA_ASSERT(err); // should be fine since we checked boundary

            // A directory can be divided into multiple strips.
            // For every strip, allocate enough memory to get decoded data.
            // Then send it for conversion.
            const tsize_t strip_size = ::TIFFStripSize(m_tiff);
            if (!buffer)
                buffer = std::make_unique<byte_t[]>(static_cast<size_t>(strip_size));

            size_t row_offset = 0;
            for (tstrip_t strip = 0; strip < ::TIFFNumberOfStrips(m_tiff); ++strip) {
                const tsize_t bytes_read = ::TIFFReadEncodedStrip(m_tiff, strip, buffer.get(), strip_size);
                if (bytes_read == -1)
                    NOA_THROW("An error occurred while reading slice:{}, strip:{}. {}", slice, strip, s_error_buffer);

                // Convert the bytes read in number of rows read:
                NOA_ASSERT(!(static_cast<size_t>(bytes_read) % i_bytes_per_elements));
                const size_t elements_in_buffer = static_cast<size_t>(bytes_read) / i_bytes_per_elements;
                NOA_ASSERT(elements_in_buffer % m_shape[2]);
                const size_t rows_in_buffer = elements_in_buffer / m_shape[2];
                const size4_t shape_buffer{1, 1, rows_in_buffer, m_shape[2]};

                // Convert and transfer to output:
                const size_t output_offset = indexing::at(slice, 0, row_offset, strides) * o_bytes_per_elements;
                deserialize(buffer.get(), m_data_type,
                            output_ptr + output_offset, data_type, strides,
                            shape_buffer, clamp);
                row_offset += rows_in_buffer;
            }

            // Origin must be in the bottom left corner:
            uint16_t orientation;
            ::TIFFGetFieldDefaulted(m_tiff, TIFFTAG_ORIENTATION, &orientation);
            if (orientation == ORIENTATION_TOPLEFT) { // this is the default
                const size_t bytes_per_row = m_shape[2] * o_bytes_per_elements;
                if (!buffer_flip_row)
                    buffer_flip_row = std::make_unique<byte_t[]>(bytes_per_row);
                flipY_(output_ptr + slice * strides[0] * o_bytes_per_elements, bytes_per_row,
                       m_shape[1], buffer_flip_row.get());
            } else if (orientation != ORIENTATION_BOTLEFT) {
                NOA_THROW("Orientation of the slice(s) is not supported. "
                          "The origin should be at the bottom left or top left.");
            }
        }
    }

    void TIFFHeader::readSlice(void* output, DataType data_type, size_t start, size_t end, bool clamp) {
        NOA_ASSERT(end >= start);
        const size4_t slice_shape{end - start, 1, m_shape[2], m_shape[3]};
        return readSlice(output, slice_shape.strides(), slice_shape, data_type, start, clamp);
    }

    void TIFFHeader::readAll(void* output, size4_t strides, size4_t shape, DataType data_type, bool clamp) {
        if (shape[0] != m_shape[0] || shape[2] != m_shape[1] || shape[3] != m_shape[2])
            NOA_THROW("The file shape {} is not compatible with the output shape {}", this->shape(), shape);
        return readSlice(output, strides, shape, data_type, 0, clamp);
    }

    void TIFFHeader::readAll(void* output, DataType data_type, bool clamp) {
        return readSlice(output, data_type, 0, m_shape[0], clamp);
    }

    void TIFFHeader::write(const void*, DataType, size_t, size_t, bool) {
        NOA_THROW("This function is currently not supported");
    }

    void TIFFHeader::writeSlice(const void* input, size4_t strides, size4_t shape,
                                DataType data_type, size_t start, bool clamp) {
        if (m_data_type == DataType::DATA_UNKNOWN)
            m_data_type = data_type;

        if (any(m_shape == 0)) {
            NOA_THROW("The shape of the file is not set or is empty. "
                      "Set the shape first, and then write a slice to the file");
        } else if (m_shape[1] != shape[2] || m_shape[2] != shape[3]) {
            NOA_THROW("The file shape {} is not compatible with the input shape {}", this->shape(), shape);
        } else if (shape[1] != 1) {
            NOA_THROW("The file shape {} describes a (stack of) 2D image(s). To write a slice, the provided "
                      "shape should have a depth of 1, but got shape {}", this->shape(), shape);
        } else if (m_shape[0] <= start + shape[0]) {
            NOA_THROW("The file has less slices ({}) that what is about to be written (start:{}, count:{})",
                      m_shape[0], start, shape[0]);
        }

        // Output as array of bytes:
        const auto* input_ptr = static_cast<const byte_t*>(input);
        const size_t i_bytes_per_elements = serializedSize(data_type, 1);
        const size_t o_bytes_per_elements = serializedSize(m_data_type, 1);

        // Target 8K per strip. Ensure strip is multiple of a line and if too many strips,
        // increase strip size (double or more).
        const size_t bytes_per_row = m_shape[2] * o_bytes_per_elements;
        size_t rows_per_strip = math::divideUp(size_t{8192}, bytes_per_row);
        size_t strip_count = math::divideUp(size_t{m_shape[1]}, rows_per_strip);
        if (strip_count > 4096) {
            rows_per_strip *= (1 + m_shape[1] / 4096);
            strip_count = math::divideUp(rows_per_strip, size_t{m_shape[1]});
        }
        const size4_t strip_shape{1, 1, rows_per_strip, m_shape[2]};
        const size_t bytes_per_strip = rows_per_strip * bytes_per_row;
        std::unique_ptr<byte_t[]> buffer = std::make_unique<byte_t[]>(bytes_per_strip);

        NOA_ASSERT(shape[0] >= start);
        for (size_t slice = start; slice < shape[0] + start; ++slice) {
            ::TIFFSetDirectory(m_tiff, static_cast<uint16_t>(slice));

            { // Set up relevant tags
                ::TIFFSetField(m_tiff, TIFFTAG_IMAGEWIDTH, m_shape[2]);
                ::TIFFSetField(m_tiff, TIFFTAG_IMAGELENGTH, m_shape[1]);
                ::TIFFSetField(m_tiff, TIFFTAG_ROWSPERSTRIP, rows_per_strip);

                if (any(m_pixel_size != 0.f)) {
                    ::TIFFSetField(m_tiff, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);
                    ::TIFFSetField(m_tiff, TIFFTAG_XRESOLUTION, static_cast<double>(1.e8f / m_pixel_size[1]));
                    ::TIFFSetField(m_tiff, TIFFTAG_YRESOLUTION, static_cast<double>(1.e8f / m_pixel_size[0]));
                }

                ::TIFFSetField(m_tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
                ::TIFFSetField(m_tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
                ::TIFFSetField(m_tiff, TIFFTAG_SAMPLESPERPIXEL, 1);

                uint16_t sample_format, bits_per_sample;
                setDataType_(m_data_type, &sample_format, &bits_per_sample);
                ::TIFFSetField(m_tiff, TIFFTAG_SAMPLEFORMAT, sample_format);
                ::TIFFSetField(m_tiff, TIFFTAG_BITSPERSAMPLE, bits_per_sample);

                // For now, no compression.
                ::TIFFSetField(m_tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE);

                // TODO I don't get why IMOD doesn't check the orientation...
                ::TIFFSetField(m_tiff, TIFFTAG_ORIENTATION, ORIENTATION_BOTLEFT);

                if (!s_error_buffer.empty())
                    NOA_THROW("An error has occurred while setting up the tags. {}", s_error_buffer);
            }

            for (tstrip_t strip = 0; strip < strip_count; ++strip) {
                byte_t* i_buffer = buffer.get() + strip * bytes_per_strip;
                const size_t input_offset = indexing::at(slice, 0, strip * rows_per_strip, strides);
                serialize(input_ptr + input_offset * i_bytes_per_elements, data_type, strides, strip_shape,
                          i_buffer, m_data_type, clamp);

                if (::TIFFWriteEncodedStrip(m_tiff, strip, i_buffer, static_cast<tmsize_t>(bytes_per_strip)) == -1)
                    NOA_THROW("An error occurred while writing slice:{}, strip:{}. {}", slice, strip, s_error_buffer);
            }
            if (!::TIFFWriteDirectory(m_tiff))
                NOA_THROW("Failed to write slice {} into the file", slice);
        }
    }

    void TIFFHeader::writeSlice(const void* input, DataType data_type, size_t start, size_t end, bool clamp) {
        NOA_ASSERT(end >= start);
        const size4_t slice_shape{end - start, 1, m_shape[1], m_shape[2]};
        return writeSlice(input, slice_shape.strides(), slice_shape, data_type, start, clamp);
    }

    void TIFFHeader::writeAll(const void* input, size4_t strides, size4_t shape, DataType data_type, bool clamp) {
        if (all(m_shape == 0)) // first write, set the shape
            this->shape(shape);
        return writeSlice(input, strides, shape, data_type, 0, clamp);
    }

    void TIFFHeader::writeAll(const void* input, DataType data_type, bool clamp) {
        if (any(m_shape == 0)) {
            NOA_THROW("The shape of the file is not set or is empty. "
                      "Set the shape first, and then write something to the file");
        }
        return writeSlice(input, data_type, 0, m_shape[0], clamp);
    }

    DataType TIFFHeader::getDataType_(uint16_t sample_format, uint16_t bits_per_sample) {
        switch (sample_format) {
            case SAMPLEFORMAT_INT:
                if (bits_per_sample == 8) {
                    return DataType::INT8;
                } else if (bits_per_sample == 16) {
                    return DataType::INT16;
                } else if (bits_per_sample == 32) {
                    return DataType::INT32;
                }
                break;
            case SAMPLEFORMAT_UINT:
                if (bits_per_sample == 8) {
                    return DataType::UINT8;
                } else if (bits_per_sample == 16) {
                    return DataType::UINT16;
                } else if (bits_per_sample == 32) {
                    return DataType::UINT32;
                } else if (bits_per_sample == 4) {
                    return DataType::UINT4;
                }
                break;
            case SAMPLEFORMAT_IEEEFP:
                if (bits_per_sample == 16) {
                    return DataType::FLOAT16;
                } else if (bits_per_sample == 32) {
                    return DataType::FLOAT32;
                } else if (bits_per_sample == 64) {
                    return DataType::FLOAT64;
                }
                break;
            case SAMPLEFORMAT_COMPLEXINT:
                if (bits_per_sample == 32) {
                    return DataType::CINT16;
                }
                break;
            case SAMPLEFORMAT_COMPLEXIEEEFP:
                if (bits_per_sample == 32) {
                    return DataType::CFLOAT16;
                } else if (bits_per_sample == 64) {
                    return DataType::CFLOAT32;
                } else if (bits_per_sample == 128) {
                    return DataType::CFLOAT64;
                }
            default:
                break;
        }
        return DataType::DATA_UNKNOWN;
    }

    void TIFFHeader::setDataType_(DataType data_type, uint16_t* sample_format, uint16_t* bits_per_sample) {
        switch (data_type) {
            case DataType::INT8:
                *bits_per_sample = 8;
                *sample_format = SAMPLEFORMAT_INT;
                break;
            case DataType::UINT8:
                *bits_per_sample = 8;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case DataType::INT16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_INT;
                break;
            case DataType::UINT16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case DataType::INT32:
                *bits_per_sample = 32;
                *sample_format = SAMPLEFORMAT_INT;
                break;
            case DataType::UINT32:
                *bits_per_sample = 32;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case DataType::INT64:
                *bits_per_sample = 64;
                *sample_format = SAMPLEFORMAT_INT;
                break;
            case DataType::UINT64:
                *bits_per_sample = 64;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case DataType::FLOAT16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_IEEEFP;
                break;
            case DataType::FLOAT32:
                *bits_per_sample = 32;
                *sample_format = SAMPLEFORMAT_IEEEFP;
                break;
            case DataType::FLOAT64:
                *bits_per_sample = 64;
                *sample_format = SAMPLEFORMAT_IEEEFP;
                break;
            case DataType::CFLOAT16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_COMPLEXIEEEFP;
                break;
            case DataType::CFLOAT32:
                *bits_per_sample = 32;
                *sample_format = SAMPLEFORMAT_COMPLEXIEEEFP;
                break;
            case DataType::CFLOAT64:
                *bits_per_sample = 64;
                *sample_format = SAMPLEFORMAT_COMPLEXIEEEFP;
                break;
            case DataType::UINT4:
                *bits_per_sample = 4;
                *sample_format = SAMPLEFORMAT_UINT;
                break;
            case DataType::CINT16:
                *bits_per_sample = 16;
                *sample_format = SAMPLEFORMAT_COMPLEXINT;
                break;
            default:
                *bits_per_sample = 1; // default
                *sample_format = SAMPLEFORMAT_VOID;
                break;
        }
    }

    void TIFFHeader::flipY_(byte_t* slice, size_t bytes_per_row, size_t rows, byte_t* buffer) {
        for (size_t row = 0; row < rows / 2; ++row) {
            byte_t* current_row = slice + row * bytes_per_row;
            byte_t* opposite_row = slice + (rows - row - 1) * bytes_per_row;
            std::memcpy(buffer, current_row, bytes_per_row);
            std::memcpy(current_row, opposite_row, bytes_per_row);
            std::memcpy(opposite_row, buffer, bytes_per_row);
        }
    }
}

#endif // NOA_ENABLE_TIFF
