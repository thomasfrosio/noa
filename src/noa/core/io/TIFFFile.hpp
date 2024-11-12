#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/io/IO.hpp"
#include "noa/core/io/Stats.hpp"

#ifdef NOA_ENABLE_TIFF
#   ifdef NOA_IS_OFFLINE

namespace noa::io {
    class TiffFile {
    public:
        TiffFile();
        TiffFile(const Path& filename, Open open_mode) : TiffFile() { open_(filename, open_mode); }

        TiffFile(const TiffFile&) noexcept = delete;
        TiffFile& operator=(const TiffFile&) noexcept = delete;

        TiffFile(TiffFile&&) noexcept = default;
        TiffFile& operator=(TiffFile&&) noexcept = default;

        ~TiffFile() { close_(); }

    public:
        void open(const Path& filename, Open open_mode) { open_(filename, open_mode); }
        void close() { close_(); }

    public:
        [[nodiscard]] bool is_open() const noexcept { return m_tiff; }
        [[nodiscard]] const Path& filename() const noexcept { return m_filename; }
        [[nodiscard]] std::string info_string(bool brief) const noexcept;

        [[nodiscard]] static auto is_supported_extension(std::string_view extension) noexcept -> bool {
            using namespace std::string_view_literals;
            return extension == ".tiff"sv or extension == ".tif"sv;
        }

    public:
        [[nodiscard]] Shape4<i64> shape() const noexcept { return {m_shape[0], 1, m_shape[1], m_shape[2]}; }
        [[nodiscard]] Stats<f64> stats() const noexcept { return {}; }
        [[nodiscard]] Vec3<f64> pixel_size() const noexcept { return {0., m_pixel_size[0], m_pixel_size[1]}; }
        [[nodiscard]] Encoding::Format encoding_format() const noexcept { return m_encoding_format; }

        void set_shape(const Shape4<i64>& shape);
        void set_stats(Stats<f64>) { /* Ignore for now */ }
        void set_pixel_size(const Vec3<f64>&);
        void set_encoding_format(Encoding::Format);

    public:
        template<typename T, StridesTraits S>
        void read_elements(const Span<T, 1, i64, S>& output, i64 start_at, bool clamp) {
            panic("This function is currently not supported");
        }

        template<typename T, StridesTraits S>
        void read_slice(const Span<T, 4, i64, S>& output, i64 start_at, bool clamp) {
            check(is_open(), "The file should be opened");
            check(m_shape[1] == output.shape()[2] and m_shape[2] == output.shape()[3],
                  "File: {}. Cannot read a 2D slice of shape {} from a file with 2D slices of shape {}",
                  m_filename, output.shape().filter(2, 3), m_shape.pop_front());
            check(output.shape()[1] == 1,
                  "File {}. Can only read 2D slice(s), but asked to read shape {}",
                  m_filename, output.shape());
            check(m_shape[0] >= start_at + output.shape()[0],
                  "File: {}. The file has less slices ({}) that what is about to be read (start:{}, count:{})",
                  m_filename, m_shape[0], start_at, output.shape()[0]);

            auto encoding = Encoding{m_encoding_format, clamp}; // endianness is handled by the tiff library
            const auto i_bytes_per_elements = encoding.encoded_size(1);

            // The strip size should not change between slices since we know they have the same shape and data layout.
            // The compression could be different, but worst case scenario, the strip size is not as optimal
            // as it could have been. Since in most cases we expect the directories to have exactly the same
            // tags, allocate once according to the first directory.
            std::unique_ptr<Byte[]> buffer, buffer_flip_row;
            for (i64 slice = start_at; slice < start_at + output.shape()[0]; ++slice) {
                tiff_set_directory_(slice);

                // A directory can be divided into multiple strips.
                // For every strip, allocate enough memory to get decoded data.
                // Then send it for conversion.
                const auto [strip_size, n_strips] = tiff_strip_properties_();
                if (not buffer)
                    buffer = std::make_unique<Byte[]>(static_cast<size_t>(strip_size));

                i64 row_offset{};
                for (i64 strip{}; strip < n_strips; ++strip) {
                    const i64 bytes_read = tiff_read_encoded_strip_(slice, strip, buffer.get(), strip_size);

                    // Convert the bytes read in number of rows read:
                    check(is_multiple_of(bytes_read, i_bytes_per_elements));
                    const auto n_elements_in_buffer = bytes_read / i_bytes_per_elements;
                    const auto n_elements_per_row = output.shape()[3];
                    check(is_multiple_of(n_elements_in_buffer, n_elements_per_row));
                    const auto n_rows_in_buffer = n_elements_in_buffer / n_elements_per_row;

                    // Convert and transfer to output:
                    auto serialized_data = SpanContiguous<const Byte, 1>(buffer.get(), n_elements_in_buffer);
                    auto deserialized_data = output.subregion(
                            slice, 0, ni::Slice{row_offset, row_offset + n_rows_in_buffer});
                    try {
                        deserialize(serialized_data, encoding, deserialized_data);
                    } catch (...) {
                        panic("File {}. Failed to read strip={} with shape={} from the file. Deserialize from {} to {}",
                              m_filename, strip, deserialized_data.shape(), encoding.format, ns::stringify<T>());
                    }
                    row_offset += n_rows_in_buffer;
                }

                flip_y_if_necessary_(
                        output.subregion(slice).template span<Byte, 2, i64, StridesTraits::STRIDED>(),
                        buffer_flip_row);
            }
        }

        template<typename T, StridesTraits S>
        void read_all(const Span<T, 4, i64, S>& output, bool clamp) {
            check(output.shape()[0] == m_shape[0],
                  "The file shape {} is not compatible with the output shape {}", shape(), output.shape());
            return read_slice(output, 0, clamp);
        }

        template<typename T, StridesTraits S>
        void write_elements(const Span<const T, 1, i64, S>& input, i64 start_at, bool clamp) {
            panic("This function is currently not supported");
        }

        template<typename T, StridesTraits S>
        void write_slice(const Span<const T, 4, i64, S>& input, i64 start_at, bool clamp) {
            check(is_open(), "The file should be opened");
            check(not m_shape.is_empty(),
                  "File: {}. The shape of the file is not set or is empty. Set the shape first, "
                  "and then write a slice to the file", m_filename);
            check(m_shape[1] == input.shape()[2] and m_shape[2] == input.shape()[3],
                  "File: {}. Cannot write a 2D slice of shape {} into a file with 2D slices of shape {}",
                  m_filename, input.shape().filter(2, 3), m_shape.pop_front());
            check(input.shape()[1] == 1,
                  "File {}. Can only write 2D slice(s), but asked to write shape {}",
                  m_filename, input.shape());
            check(m_shape[0] >= start_at + input.shape()[0],
                  "File: {}. The file has less slices ({}) that what is about to be written (start:{}, count:{})",
                  m_filename, m_shape[0], start_at, input.shape()[0]);

            if (m_encoding_format == Encoding::UNKNOWN) // first write
                m_encoding_format = Encoding::to_format<T>();

            // Output as array of bytes:
            const auto encoding = Encoding{m_encoding_format, clamp};
            const i64 o_bytes_per_elements = encoding.encoded_size(1);

            // Target 8K per strip.
            // Ensure strip is multiple of a line and if too many strips, increase strip size (double or more).
            const i64 bytes_per_row = m_shape[2] * o_bytes_per_elements;
            i64 n_rows_per_strip = divide_up(i64{8192}, bytes_per_row);
            i64 n_strips = divide_up(i64{m_shape[1]}, n_rows_per_strip);
            if (n_strips > 4096) {
                n_rows_per_strip *= (1 + m_shape[1] / 4096);
                n_strips = divide_up(n_rows_per_strip, i64{m_shape[1]});
            }
            const i64 bytes_per_strip = n_rows_per_strip * bytes_per_row;
            const auto buffer = std::make_unique<Byte[]>(static_cast<size_t>(bytes_per_strip));
            const auto buffer_span = SpanContiguous<Byte, 1>(buffer.get(), bytes_per_strip);

            NOA_ASSERT(shape()[0] >= start_at);
            for (i64 slice = start_at; slice < shape()[0] + start_at; ++slice) {
                tiff_set_directory_(slice);
                tiff_set_header_(n_rows_per_strip);

                for (i64 strip{}; strip < n_strips; ++strip) {
                    const i64 current_row = strip * n_rows_per_strip;
                    auto serialized_data = input.subregion(slice, 0, ni::Slice{current_row, current_row + n_rows_per_strip});
                    try {
                        serialize(serialized_data, buffer_span, encoding);
                    } catch (...) {
                        panic("File {}. Failed to write strip={} with shape={} from the file. Serialize from {} to {}",
                              m_filename, strip, serialized_data.shape(), encoding.format, ns::stringify<T>());
                    }
                    tiff_write_encoded_strip(slice, strip, buffer_span.get(), bytes_per_strip);
                }
                tiff_write_directory();
            }
        }

        template<typename T>
        void write_all(const Span<const T, 4>& input, bool clamp) {
            if (m_shape.is_empty()) // first write, set the shape
                set_shape(input.shape());
            return write_slice(input, 0, clamp);
        }

    private:
        void open_(const Path& filename, Open mode);
        void close_();

        static Encoding::Format get_encoding_format_(u16 sample_format, u16 bits_per_sample);
        static void set_dtype_(Encoding::Format data_type, u16* sample_format, u16* bits_per_sample);

        void tiff_read_header_();
        void tiff_set_header_(i64 n_rows_per_strip);
        void tiff_set_directory_(i64 slice);
        void tiff_write_directory();
        auto tiff_strip_properties_() -> Pair<i64, i64>;
        auto tiff_read_encoded_strip_(i64 slice, i64 strip, Byte* buffer, i64 strip_size) -> i64;
        void tiff_write_encoded_strip(i64 slice, i64 strip, Byte* buffer, i64 strip_size);

        void flip_y_if_necessary_(const Span<Byte, 2>& slice, std::unique_ptr<Byte[]>& buffer);

    private:
        void* m_tiff{};
        Path m_filename{};
        Shape3<i64> m_shape{}; // BHW
        Vec2<f64> m_pixel_size{}; // HW
        Encoding::Format m_encoding_format{Encoding::UNKNOWN};
        bool m_is_read{};
    };
}

#   endif
#endif
