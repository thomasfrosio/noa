#if NOA_ENABLE_TIFF

#include <tiffio.h>

#include "noa/core/Types.hpp"
#include "noa/core/io/OS.hpp"
#include "noa/core/io/TIFFFile.hpp"

// TODO Surely there's a more modern library we could use here?

namespace {
    // One string per thread is enough since TiffFile will immediately throw after the error.
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
    void TiffFile::open_(const Path& filename, Open open_mode) {
        close();

        const bool write = open_mode.write;
        m_is_read = open_mode.read;
        if (write) {
            check(not m_is_read,
                  "File: {}. Opening a TIFF file in read-write mode is not supported. Should be read or write",
                  filename);
        } else if (not m_is_read) {
            panic("File: {}. Open mode is not supported. Should be read or write", filename);
        }

        try {
            if (write and is_file(filename))
                backup(filename, true);
            else
                mkdir(filename.parent_path());
        } catch (...) {
            panic("File: {}. {}. Could not open the file because of an OS failure", filename, open_mode);
        }

        for (u32 it{0}; it < 5; ++it) {
            // h: Read TIFF header only, do not load the first image directory.
            // c: Disable the use of strip chopping when reading images
            m_tiff = TIFFOpen(filename.c_str(), write ? "rhc" : "w");
            if (m_tiff) {
                if (m_is_read) {
                    try {
                        tiff_read_header_();
                    } catch (...) {
                        panic("File: {}. {}. Failed while reading the header", filename, open_mode);
                    }
                }
                m_filename = filename;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (m_is_read and not is_file(filename))
            panic("File: {}. {}. Failed to open the file. The file does not exist", filename, open_mode);
        panic("File: {}. {}. Failed to open the file. {}", filename, open_mode, s_error_buffer);
    }

    void TiffFile::close_() {
        ::TIFFClose(get_tiff_(m_tiff));
        if (not s_error_buffer.empty())
            panic("File: {}. An error has occurred while closing the file. {}", m_filename, s_error_buffer);
        m_tiff = nullptr;
        m_filename.clear();
    }

    // The logic comes from IMOD/libiimod/iitif.c::iiTIFFCheck
    void TiffFile::tiff_read_header_() {
        ::TIFF* tiff = get_tiff_(m_tiff);
        uint16_t directories{};
        while (::TIFFSetDirectory(tiff, directories)) {
            Shape2<u32> shape; // height-width
            if (not ::TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, shape.data() + 1) or
                not ::TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, shape.data())) {
                panic("The input TIFF file does not have the width or height field.");
            }

            Vec2<f32> pixel_size{1.f}; // height-width ang/pixel
            {
                u16 resolution_unit{}; // 1: no units, 2: inch, 3: cm
                const auto has_resolution = TIFFGetField(tiff, TIFFTAG_XRESOLUTION, &pixel_size[1]);
                TIFFGetFieldDefaulted(tiff, TIFFTAG_RESOLUTIONUNIT, &resolution_unit);
                if (resolution_unit > 1 and has_resolution) {
                    if (not TIFFGetField(tiff, TIFFTAG_YRESOLUTION, pixel_size.data()))
                        pixel_size[0] = pixel_size[1];
                    const auto scale = resolution_unit == 2 ? 2.54e8f : 1.00e8f;
                    pixel_size = scale / pixel_size;
                }
            }

            Encoding::Format encoding_format;
            {
                u16 photometry{}, sample_per_pixel{}, sample_format{}, bits_per_sample{}, planar_config{};
                ::TIFFGetField(tiff, TIFFTAG_PHOTOMETRIC, &photometry);
                ::TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLESPERPIXEL, &sample_per_pixel);
                ::TIFFGetFieldDefaulted(tiff, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
                ::TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &sample_format);
                ::TIFFGetFieldDefaulted(tiff, TIFFTAG_PLANARCONFIG, &planar_config);
                if (not s_error_buffer.empty())
                    panic("Error occurred while reading sampling fields of directory {}. {}",
                          directories, s_error_buffer);
                else if (planar_config != PLANARCONFIG_CONTIG)
                    panic("Planar configuration separate is not supported. Should be contiguous");
                else if (photometry > 2)
                    panic("Photometry not supported. Should be bi-level or grayscale");
                else if (sample_per_pixel > 1)
                    panic("Samples per pixel should be 1. {} is not supported", sample_per_pixel);
                encoding_format = get_encoding_format_(sample_format, bits_per_sample);
                if (encoding_format == Encoding::UNKNOWN)
                    panic("Data type was not recognized in directory {}", directories);
                else if (encoding_format == Encoding::U4)
                    shape[1] *= 2;
            }

            if (directories) { // check no mismatch
                if (not all(allclose(pixel_size, m_pixel_size.as<f32>())) or
                    shape[0] != m_shape[0] or
                    shape[1] != m_shape[1] or
                    encoding_format != m_encoding_format)
                    panic("Mismatch detected. Directories with different data type, shape, or"
                          "pixel sizes are not supported");

            } else { // save to header
                m_encoding_format = encoding_format;
                m_pixel_size = pixel_size.as<f64>();
                m_shape[1] = shape[0];
                m_shape[2] = shape[1];
            }
            ++directories;
        }
        if (not s_error_buffer.empty())
            panic("Error occurred while reading directories. {}", s_error_buffer);

        // At this point the current directory is the last one. This is OK since read/write operations
        // will reset the directory based on the desired section/strip.
        m_shape[0] = directories;
    }

    TiffFile::TiffFile() {
        static thread_local bool thread_is_set{};
        if (not thread_is_set) {
            s_error_buffer.reserve(250);
            ::TIFFSetErrorHandler(errorHandler_);
            ::TIFFSetWarningHandler(warningHandler_);
            thread_is_set = true;
        }
    }

    void TiffFile::set_shape(const Shape4<i64>& shape) {
        check(not m_is_read, "Trying to change the shape of the data in read mode is not allowed");
        check(shape[1] == 1,
              "TIFF files do not support 3D volumes, but got shape {}. "
              "To set a stack of 2D images, use the batch dimension "
              "instead of the depth", shape);
        check(all(shape > 0), "The shape should be non-zero positive, but got {}", shape);
        m_shape = shape.filter(0, 2, 3);
    }

    void TiffFile::set_pixel_size(const Vec3<f64>& pixel_size) {
        check(not m_is_read, "Trying to change the pixel size of the data in read mode is not allowed");
        check(all(pixel_size >= 0), "The pixel size should be positive, got {}", pixel_size);
        m_pixel_size = pixel_size.filter(1, 2);
    }

    void TiffFile::set_encoding_format(Encoding::Format encoding_format) {
        check(not m_is_read, "Trying to change the data type of the data in read mode is not allowed");
        m_encoding_format = encoding_format;
    }

    std::string TiffFile::info_string(bool brief) const noexcept {
        if (brief)
            return fmt::format("Shape: {}; Pixel size: {::.3f}", shape(), pixel_size());

        return fmt::format("Format: TIFF File\n"
                           "Shape (batch, depth, height, width): {}\n"
                           "Pixel size (depth, height, width): {::.3f}\n"
                           "Data type: {}",
                           shape(),
                           pixel_size(),
                           m_encoding_format);
    }

    void TiffFile::tiff_set_directory_(i64 slice) {
        check(::TIFFSetDirectory(get_tiff_(m_tiff), safe_cast<u16>(slice)) == 0);
    }

    auto TiffFile::tiff_strip_properties_() -> Pair<i64, i64> {
        auto* tiff = get_tiff_(m_tiff);
        return Pair{safe_cast<i64>(::TIFFStripSize(tiff)),
                    safe_cast<i64>(::TIFFNumberOfStrips(tiff))};
    }

    auto TiffFile::tiff_read_encoded_strip_(i64 slice, i64 strip, Byte* buffer, i64 strip_size) -> i64 {
        const i64 bytes_read = ::TIFFReadEncodedStrip(get_tiff_(m_tiff), safe_cast<u32>(strip), buffer, strip_size);
        if (bytes_read == -1)
            panic("File: {}. An error occurred while reading slice={}, strip={}. {}",
                  m_filename, slice, strip, s_error_buffer);
        return bytes_read;
    }

    void TiffFile::tiff_write_encoded_strip(i64 slice, i64 strip, Byte* buffer, i64 strip_size) {
        if (::TIFFWriteEncodedStrip(get_tiff_(m_tiff), safe_cast<u32>(strip), buffer, strip_size) == -1)
            panic("File: {}. An error occurred while writing slice={}, strip={}. {}",
                  m_filename, slice, strip, s_error_buffer);
    }

    void TiffFile::tiff_set_header_(i64 n_rows_per_strip) {
        auto* tiff = get_tiff_(m_tiff);

            ::TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, m_shape[2]);
            ::TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, m_shape[1]);
            ::TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, n_rows_per_strip);

            if (any(m_pixel_size != 0.)) {
                ::TIFFSetField(tiff, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);
                ::TIFFSetField(tiff, TIFFTAG_XRESOLUTION, static_cast<f32>(1.e8 / m_pixel_size[1]));
                ::TIFFSetField(tiff, TIFFTAG_YRESOLUTION, static_cast<f32>(1.e8 / m_pixel_size[0]));
            }

            ::TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            ::TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
            ::TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 1);

            u16 sample_format{}, bits_per_sample{};
            set_dtype_(m_encoding_format, &sample_format, &bits_per_sample);
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

    void TiffFile::tiff_write_directory() {
        if (not ::TIFFWriteDirectory(get_tiff_(m_tiff)))
            panic("File: {}. Failed to write a slice into the file", m_filename);
    }

    Encoding::Format TiffFile::get_encoding_format_(u16 sample_format, u16 bits_per_sample) {
        switch (sample_format) {
            case SAMPLEFORMAT_INT:
                if (bits_per_sample == 8) {
                    return Encoding::I8;
                } else if (bits_per_sample == 16) {
                    return Encoding::I16;
                } else if (bits_per_sample == 32) {
                    return Encoding::I32;
                }
                break;
            case SAMPLEFORMAT_UINT:
                if (bits_per_sample == 8) {
                    return Encoding::U8;
                } else if (bits_per_sample == 16) {
                    return Encoding::U16;
                } else if (bits_per_sample == 32) {
                    return Encoding::U32;
                } else if (bits_per_sample == 4) {
                    return Encoding::U4;
                }
                break;
            case SAMPLEFORMAT_IEEEFP:
                if (bits_per_sample == 16) {
                    return Encoding::F16;
                } else if (bits_per_sample == 32) {
                    return Encoding::F32;
                } else if (bits_per_sample == 64) {
                    return Encoding::F64;
                }
                break;
            case SAMPLEFORMAT_COMPLEXINT:
                if (bits_per_sample == 32) {
                    return Encoding::CI16;
                }
                break;
            case SAMPLEFORMAT_COMPLEXIEEEFP:
                if (bits_per_sample == 32) {
                    return Encoding::C16;
                } else if (bits_per_sample == 64) {
                    return Encoding::C32;
                } else if (bits_per_sample == 128) {
                    return Encoding::C64;
                }
                break;
            default:
                break;
        }
        return Encoding::UNKNOWN;
    }

    void TiffFile::set_dtype_(Encoding::Format data_type, u16* sample_format, u16* bits_per_sample) {
        switch (data_type) {
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
                *bits_per_sample = 1; // default
                *sample_format = SAMPLEFORMAT_VOID;
                break;
        }
    }

    void TiffFile::flip_y_if_necessary_(const Span<Byte, 2>& data, std::unique_ptr<Byte[]>& buffer) {
        u16 orientation{};
        ::TIFFGetFieldDefaulted(get_tiff_(m_tiff), TIFFTAG_ORIENTATION, &orientation);
        if (orientation == ORIENTATION_TOPLEFT) { // this is the default
            if (not buffer)
                buffer = std::make_unique<Byte[]>(static_cast<size_t>(data.shape()[1]));

            const i64 bytes_per_row = data.shape()[1];
            const i64 n_rows = data.shape()[0];
            for (i64 row{}; row < n_rows / 2; ++row) {
                Byte* current_row = data.get() + row * bytes_per_row;
                Byte* opposite_row = data.get() + (n_rows - row - 1) * bytes_per_row;
                const auto bytes = static_cast<size_t>(bytes_per_row);
                std::memcpy(buffer.get(), current_row, bytes);
                std::memcpy(current_row, opposite_row, bytes);
                std::memcpy(opposite_row, buffer.get(), bytes);
            }
        } else if (orientation != ORIENTATION_BOTLEFT) {
            panic("File: {}. Orientation of the slice(s) is not supported. "
                  "The origin should be at the bottom left or top left", m_filename);
        }
    }
}

#endif // NOA_ENABLE_TIFF
