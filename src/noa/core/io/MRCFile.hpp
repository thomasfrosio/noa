#pragma once

#include <fstream>

#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/io/IO.hpp"
#include "noa/core/io/Stats.hpp"

namespace noa::io {
    /// Image file with the MRC format.
    /// \note Pixel size: If the cell size is equal to zero, the pixel size will be 0. This is allowed since in some
    ///       cases the pixel size is ignored. Thus, one should check the returned value of pixel_size() before using
    ///       it. Similarly, set_pixel_size() can save a pixel size equal to 0, which is the default if a new file is
    ///       closed without calling set_pixel_size().
    ///
    /// \note Endianness: It is not possible to change the endianness of existing data. As such, in the rare case of
    ///       writing to an existing file (i.e. read|write mode), the written data may have to be swapped to match the
    ///       file's endianness. This will be done automatically when writing to the file.
    ///
    /// \note Encoding format: In the rare case of writing to an existing file (i.e. read|write mode), the format
    ///       cannot be changed. In other cases, the user should only set the format once before writing anything
    ///       to the file.
    ///
    /// \note Dimension order: The map ordering should be mapc=1, mapr=2 and maps=3. Anything else is not supported
    ///       and an exception is thrown when opening a file with a different ordering.
    ///
    /// \note Unused flags: The extended header, the origin (xorg, yorg, zorg), nversion and other parts of the header
    ///       are ignored (see full detail on what is ignored in write_header_()). These are set to 0 or to the
    ///       expected default value, or are left unchanged in non-overwriting mode.
    ///
    /// \see https://bio3d.colorado.edu/imod/doc/mrc_format.txt or
    ///      https://www.ccpem.ac.uk/mrc_format/mrc2014.php
    class MrcFile {
    public:
        MrcFile() = default;
        MrcFile(const Path& path, Open mode) { open_(path, mode); }

        MrcFile(const MrcFile&) noexcept = delete;
        MrcFile& operator=(const MrcFile&) noexcept = delete;

        MrcFile(MrcFile&&) noexcept = default;
        MrcFile& operator=(MrcFile&&) noexcept = default;

        ~MrcFile() { close_(); }

    public:
        void open(const Path& path, Open open_mode) { open_(path, open_mode); }
        void close() { close_(); }

    public:
        [[nodiscard]] auto is_open() const noexcept -> bool { return m_fstream.is_open(); }
        [[nodiscard]] auto filename() const noexcept -> const Path& { return m_filename; }
        [[nodiscard]] static auto is_supported_extension(std::string_view extension) noexcept -> bool {
            using namespace std::string_view_literals;
            return extension == ".mrc"sv or extension == ".mrcs"sv;
        }

        [[nodiscard]] auto info_string(bool brief) const noexcept -> std::string {
            if (brief)
                return fmt::format("Shape: {}; Pixel size: {}", m_header.shape, m_header.pixel_size);

            return fmt::format("Format: MRC File\n"
                               "Shape (batch, depth, height, width): {}\n"
                               "Pixel size (depth, height, width): {::.3f}\n"
                               "Data type: {}\n"
                               "Labels: {}\n"
                               "Extended header: {} bytes",
                               m_header.shape,
                               m_header.pixel_size,
                               m_header.encoding_format,
                               m_header.nb_labels,
                               m_header.extended_bytes_nb);
        }

        [[nodiscard]] auto shape() const noexcept -> const Shape4<i64>& { return m_header.shape; }

        void set_shape(const Shape4<i64>& new_shape) {
            check(m_open_mode.write,
                  "Trying to change the shape of the data in read mode is not allowed. "
                  "Hint: to fix the header of a file, open it in read-write mode");
            check(not new_shape.is_empty(), "The shape should be non-zero positive, but got {}", new_shape);
            m_header.shape = new_shape;
        }

        [[nodiscard]] auto stats() const noexcept -> Stats<f64> {
            Stats<f64> out;
            if (m_header.min != 0 or m_header.max != -1 or m_header.mean != -2) {
                out.set_min(m_header.min);
                out.set_max(m_header.max);
                out.set_mean(m_header.mean);
            }
            if (m_header.std >= 0)
                out.set_std(m_header.std);
            return out;
        }

        void set_stats(const Stats<f64>& stats) {
            // In reading mode, this will have no effect.
            if (stats.has_min())
                m_header.min = static_cast<f32>(stats.min());
            if (stats.has_max())
                m_header.max = static_cast<f32>(stats.max());
            if (stats.has_mean())
                m_header.mean = static_cast<f32>(stats.mean());
            if (stats.has_std())
                m_header.std = static_cast<f32>(stats.std());
        }

        [[nodiscard]] auto pixel_size() const noexcept -> Vec3<f64> {
            return m_header.pixel_size.as<f64>();
        }

        void set_pixel_size(const Vec<f64, 3>& new_pixel_size) {
            check(m_open_mode.write,
                  "Trying to change the pixel size of the file in read mode is not allowed. "
                  "Hint: to fix the header of a file, open it in read-write mode");
            check(all(new_pixel_size >= 0), "The pixel size should be positive, got {}", new_pixel_size);
            m_header.pixel_size = new_pixel_size.as<f32>();
        }

        [[nodiscard]] auto encoding_format() const noexcept -> Encoding::Format {
            return m_header.encoding_format;
        }

        void set_encoding_format(Encoding::Format encoding_format) {
            check(m_open_mode.write,
                  "Trying to change the encoding of the file in read mode is not allowed. "
                  "Hint: to fix the header of a file, open it in read-write mode");
            switch (encoding_format) {
                case Encoding::Format::U4:
                case Encoding::Format::I8:
                case Encoding::Format::U8:
                case Encoding::Format::I16:
                case Encoding::Format::U16:
                case Encoding::Format::F16:
                case Encoding::Format::F32:
                case Encoding::Format::C32:
                case Encoding::Format::CI16:
                    m_header.encoding_format = encoding_format;
                    break;
                default:
                    panic("{} is not supported", encoding_format);
            }
        }

    public:
        template<typename T, StridesTraits S>
        void read_elements(const Span<T, 1, i64, S>& output, i64 start_at, bool clamp) {
            check(is_open(), "The file should be opened");
            check(m_header.encoding_format != Encoding::U4,
                  "File: {}. The 4bits format (mode 101) is not supported. Use read_slice or read_all instead",
                  m_filename);

            const auto encoding = Encoding{m_header.encoding_format, clamp, m_header.is_endian_swapped};
            const auto offset = header_offset_() + encoding.encoded_size(start_at);
            m_fstream.seekg(offset);
            if (m_fstream.fail()) {
                m_fstream.clear();
                panic("File: {}. Could not seek to the desired offset ({})", m_filename, offset);
            }

            try {
                deserialize(m_fstream, encoding, output.as_strided_4d());
            } catch (...) {
                panic("File {}. Failed to read from the file stream, n_elements={}, encoding_format={}, start_at=",
                      m_filename, output.n_elements(), encoding.format, start_at);
            }
        }

        template<typename T, StridesTraits S>
        void read_slice(const Span<T, 4, i64, S>& output, i64 start_at, bool clamp) {
            check(is_open(), "The file should be opened");

            // Read either a 2d slice from a stack of 2d images or from a 3d volume.
            check(m_header.shape[1] == 1 or m_header.shape[0] == 1,
                  "File {}. This function only supports stack of 2d image(s) "
                  "or a single 3d volume, but got file shape {}",
                  m_filename, m_header.shape);
            check(output.shape()[1] == 1,
                  "File {}. Can only read 2d slice(s), but asked to read shape {}",
                  m_filename, output.shape());
            check(m_header.shape[2] == output.shape()[2] and m_header.shape[3] == output.shape()[3],
                  "File: {}. Cannot read a 2d slice of shape {} from a file with 2d slices of shape {}",
                  m_filename, output.shape().filter(2, 3), m_header.shape.filter(2, 3));

            // Make sure it doesn't go out of bound.
            const bool file_is_volume = m_header.shape[0] == 1 and m_header.shape[1] > 1;
            check(m_header.shape[file_is_volume] >= start_at + output.shape()[0],
                  "File: {}. The file has less slices ({}) that what is about to be read (start:{}, count:{})",
                  m_filename, m_header.shape[file_is_volume], start_at, output.shape()[0]);

            const auto encoding = Encoding{m_header.encoding_format, clamp, m_header.is_endian_swapped};
            const auto n_elements_per_slice = m_header.shape[2] * m_header.shape[3];
            const i64 offset = header_offset_() + start_at * encoding.encoded_size(n_elements_per_slice);
            m_fstream.seekg(offset);
            if (m_fstream.fail()) {
                m_fstream.clear();
                panic("File: {}. Could not seek to the desired offset ({})",
                      m_filename, offset);
            }
            try {
                deserialize(m_fstream, encoding, output.as_strided());
            } catch (...) {
                panic("File {}. Failed to read from the file stream, shape={}, encoding_format={}, start_at={}",
                      m_filename, output.shape(), encoding.format, start_at);
            }
        }

        template<typename T, StridesTraits S>
        void read_all(const Span<T, 4, i64, S>& output, bool clamp) {
            check(is_open(), "The file should be opened");
            check(all(output.shape() == m_header.shape),
                  "File: {}. The file shape {} is not compatible with the output shape {}",
                  m_filename, m_header.shape, output.shape());

            m_fstream.seekg(header_offset_());
            if (m_fstream.fail()) {
                m_fstream.clear();
                panic("File: {}. Could not seek to the desired offset ({})",
                      m_filename, header_offset_());
            }

            const auto encoding = Encoding{m_header.encoding_format, clamp, m_header.is_endian_swapped};
            try {
                deserialize(m_fstream, encoding, output.as_strided());
            } catch (...) {
                panic("File {}. Failed to read from the file stream, shape={}, encoding_format={}",
                      m_filename, output.shape(), encoding.format);
            }
        }

        template<typename T, StridesTraits S>
        void write_elements(const Span<const T, 1, i64, S>& input, i64 start_at, bool clamp) {
            check(is_open(), "The file should be opened");

            if (m_header.encoding_format == Encoding::UNKNOWN) // first write
                m_header.encoding_format = closest_supported_encoding_format_(Encoding::to_format<T>());

            check(m_header.encoding_format != Encoding::U4,
                  "File: {}. The 4bits format (mode 101) is not supported. "
                  "Use write_slice or write_all instead", m_filename);

            const auto encoding = Encoding{m_header.encoding_format, clamp, m_header.is_endian_swapped};
            const auto offset = header_offset_() + encoding.encoded_size(start_at);
            m_fstream.seekp(offset);
            if (m_fstream.fail()) {
                m_fstream.clear();
                panic("File: {}. Could not seek to the desired offset ({})", m_filename, offset);
            }

            try {
                serialize(input.as_strided_4d(), m_fstream, encoding);
            } catch (...) {
                panic("File {}. Failed to write to the file stream, n_elements={}, encoding_format={}, start_at=",
                      m_filename, input.n_elements(), encoding.format, start_at);
            }
        }

        template<typename T, StridesTraits S>
        void write_slice(const Span<const T, 4, i64, S>& input, i64 start_at, bool clamp) {
            check(is_open(), "The file should be opened");

            // For writing a slice, it's best if we require the shape to be already set.
            check(all(m_header.shape > 0),
                  "File: {}. The shape of the file is not set or is empty. "
                  "Set the shape first, and then write a slice to the file",
                  m_filename);

            // Write a 2d slice into either a stack of 2d images or into a 3d volume.
            check(m_header.shape[1] == 1 or m_header.shape[0] == 1,
                  "File {}. This function only supports stack of 2d image(s) "
                  "or a single 3d volume, but got file shape {}",
                  m_filename, m_header.shape);
            check(input.shape()[1] == 1,
                  "File {}. Can only write 2d slice(s), but asked to write a shape of {}",
                  m_filename, input.shape());
            check(m_header.shape[2] == input.shape()[2] and m_header.shape[3] == input.shape()[3],
                  "File: {}. Cannot write a 2d slice of shape {} into a file with 2d slices of shape {}",
                  m_filename, input.shape().filter(2, 3), m_header.shape.filter(2, 3));

            // Make sure it doesn't go out of bound.
            const bool file_is_volume = m_header.shape[0] == 1 and m_header.shape[1] > 1;
            check(m_header.shape[file_is_volume] >= start_at + input.shape()[0],
                  "File: {}. The file has less slices ({}) that what is about to be written (start:{}, count:{})",
                  m_filename, m_header.shape[file_is_volume], start_at, input.shape()[0]);

            if (m_header.encoding_format == Encoding::UNKNOWN) // first write
                m_header.encoding_format = closest_supported_encoding_format_(Encoding::to_format<T>());

            const auto encoding = Encoding{m_header.encoding_format, clamp, m_header.is_endian_swapped};
            const auto n_elements_per_slice = m_header.shape[2] * m_header.shape[3];
            const auto offset = header_offset_() + start_at * encoding.encoded_size(n_elements_per_slice);
            m_fstream.seekp(offset);
            if (m_fstream.fail()) {
                m_fstream.clear();
                panic("File {}. Could not seek to the desired offset ({})", m_filename, offset);
            }
            try {
                serialize(input.as_strided(), m_fstream, encoding);
            } catch (...) {
                panic("File {}. Failed to write to the file stream, shape={}, encoding_format={}, start_at=",
                      m_filename, input.shape(), encoding.format, start_at);
            }
        }

        template<typename T, StridesTraits S>
        void write_all(const Span<const T, 4, i64, S>& input, bool clamp) {
            check(is_open(), "The file should be opened");

            if (m_header.encoding_format == Encoding::UNKNOWN) // first write
                m_header.encoding_format = closest_supported_encoding_format_(Encoding::to_format<T>());

            if (m_header.shape.is_empty()) { // first write, set the shape
                m_header.shape = input.shape();
            } else {
                check(vall(Equal{}, input.shape(), m_header.shape),
                      "File: {}. The file shape {} is not compatible with the input shape {}",
                      m_filename, m_header.shape, input.shape());
            }

            m_fstream.seekp(header_offset_());
            if (m_fstream.fail()) {
                m_fstream.clear();
                panic("File: {}. Could not seek to the desired offset ({})",
                      m_filename, header_offset_());
            }

            const auto encoding = Encoding{m_header.encoding_format, clamp, m_header.is_endian_swapped};
            try {
                serialize(input.as_strided(), m_fstream, encoding);
            } catch (...) {
                panic("File {}. Failed to write to the file stream, shape={}, encoding_format={}",
                      m_filename, input.shape(), encoding.format);
            }
        }

    private:
        void open_(const Path& filename, Open mode,
                   const std::source_location& location = std::source_location::current());

        // Reads and checks the header of an existing file.
        // Throws if the header doesn't look like an MRC header or if the MRC file is not supported.
        // If read|write mode, the header is saved in m_header.buffer. This will be used before closing the file.
        void read_header_(const Path& filename);

        // Swap the endianness of the header.
        // The buffer is at least the first 224 bytes of the MRC header.
        //
        // In read or read|write mode, the data of the header should be swapped if the endianness of the file is
        // swapped. This function should be called just after reading the header AND just before writing it.
        // All used flags are swapped. Some unused flags are left unchanged.
        static void swap_header_(Byte* buffer) {
            swap_endian(buffer, 24, 4); // from 0 (nx) to 96 (next, included).
            swap_endian(buffer + 152, 2, 4); // imodStamp, imodFlags
            swap_endian(buffer + 216, 2, 4); // rms, nlabl
        }

        // Closes the stream. Separate function so that the destructor can call close().
        // In writing mode, the header flags will be written to the file's header.
        void close_();

        // Sets the header to default values.
        // This function should only be called to initialize the header before closing a (overwritten or new) file.
        static void default_header_(Byte* buffer);

        // Writes the header to the file's header. Only called before closing a file.
        // Buffer containing the header. Only the used values (see m_header) are going to be modified before write
        // buffer to the file. As such, unused values should either be set to the defaults (overwrite mode, see
        // default_header_()) OR taken from an existing file (read|write mode, see read_header_()).
        void write_header_(Byte* buffer);

        // Gets the offset to the data.
        [[nodiscard]] constexpr auto header_offset_() const noexcept -> i64 {
            return 1024 + m_header.extended_bytes_nb;
        }

        // This is to set the default data type of the file in the first write operation of a new file in writing mode.
        // data_type is the type of the real data that is passed in, and we return the "closest" supported type.
        static auto closest_supported_encoding_format_(Encoding::Format encoding_format) -> Encoding::Format {
            switch (encoding_format) {
                case Encoding::I8:
                case Encoding::U8:
                case Encoding::I16:
                case Encoding::U16:
                case Encoding::F16:
                    return encoding_format;
                case Encoding::I32:
                case Encoding::U32:
                case Encoding::I64:
                case Encoding::U64:
                case Encoding::F32:
                case Encoding::F64:
                    return Encoding::F32;
                case Encoding::C16:
                case Encoding::C32:
                case Encoding::C64:
                    return Encoding::C32;
                default:
                    return Encoding::UNKNOWN;
            }
        }

    private:
        std::fstream m_fstream{};
        Path m_filename{};
        Open m_open_mode{};

        struct HeaderData {
            // Buffer containing the 1024 bytes of the header.
            // Only used if the header needs to be saved, that is, in "read|write" mode.
            std::unique_ptr<Byte[]> buffer{nullptr};
            Encoding::Format encoding_format{Encoding::UNKNOWN};

            Shape4<i64> shape{};            // BDHW order.
            Vec3<f32> pixel_size{};         // Pixel spacing (DHW order) = cell_size / shape.

            f32 min{0};                     // Minimum pixel value.
            f32 max{-1};                    // Maximum pixel value.
            f32 mean{-2};                   // Mean pixel value.
            f32 std{-1};                    // Stdev. Negative if not computed.

            i32 extended_bytes_nb{};        // Number of bytes in extended header.

            bool is_endian_swapped{false};  // Whether the endianness of the data is swapped.
            i32 nb_labels{};                // Number of labels with useful data.
        } m_header{};
    };
}
