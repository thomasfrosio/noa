#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/io/Encoding.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Span.hpp"

namespace noa::traits {
    template<typename T, typename U = f32>
    concept image_encorder = requires(
        T t,
        std::FILE* file,
        const Span<U, 4>& span,
        const Shape<i64, 4>& shape,
        const Vec<f64, 3>& spacing,
        noa::io::Encoding::Type dtype,
        Vec<i64, 2>& offset,
        bool clamp,
        i32 n_threads,
        std::string_view extension
    ) {
        { t.read_header(file) } -> std::same_as<Tuple<Shape<i64, 4>, Vec<f64, 3>, noa::io::Encoding::Type>>;
        { t.write_header(file, shape, spacing, dtype) } -> std::same_as<void>;
        { t.close() } -> std::same_as<void>;
        { t.decode(file, span, offset, clamp, n_threads) } -> std::same_as<void>;
        { t.encode(file, span.as_const(), offset, clamp, n_threads) } -> std::same_as<void>;
        { T::is_supported_extension(extension) } noexcept -> std::same_as<bool>;
        { T::is_supported_stream(file) } noexcept -> std::same_as<bool>;
        { T::required_file_size(shape, dtype) } noexcept -> std::same_as<i64>;
        { T::closest_supported_dtype(dtype) } noexcept -> std::same_as<noa::io::Encoding::Type>;
    };

    template<typename T>
    concept image_encorder_supported_value_type =
        nt::any_of<T, i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, c16, c32, c64>;
}

// TODO Add TIFF, JPEG, PNG and EER.

namespace noa::io {
    /// MRC file encoder and decoder.
    /// \details Limitations/notes:
    ///     - Modifying an existing file is not supported. It is either reading an existing file or writing a new one.
    ///     - The spacing can be set to 0, indicating the spacing is unset.
    ///     - The header, and thus the shape, is set when opening the file.
    ///     - The extended header, the origin (xorg, yorg, zorg), nversion, min/max/mean/std and other parts of the
    ///       header are ignored. When writing a new file, these are set to 0 or to the expected default value.
    ///     - The map ordering should be mapc=1, mapr=2 and maps=3. Anything else is not supported, and an exception
    ///       is thrown when opening a file with a different ordering.
    /// \see https://bio3d.colorado.edu/imod/doc/mrc_format.txt or
    ///      https://www.ccpem.ac.uk/mrc_format/mrc2014.php
    struct EncoderMrc {
        auto read_header(
            std::FILE* file
        ) -> Tuple<Shape<i64, 4>, Vec<f64, 3>, Encoding::Type>;

        void write_header(
            std::FILE* file,
            const Shape<i64, 4>& shape,
            const Vec<f64, 3>& spacing,
            Encoding::Type dtype
        );

        void close() const {
            // We write the header directly when opening the file, so we have nothing to do here.
        }

        template<typename T>
        void decode(
            std::FILE* file,
            const Span<T, 4>& output,
            const Vec<i64, 2>& bd_offset,
            bool clamp,
            i32 n_threads
        ) {
            const auto encoding = Encoding{
                .dtype = m_dtype,
                .clamp = clamp,
                .endian_swap = m_is_endian_swapped
            };
            const i64 byte_offset =
                HEADER_SIZE + m_extended_bytes_nb +
                encoding.encoded_size(ni::offset_at(m_shape.strides(), bd_offset));

            check(std::fseek(file, byte_offset, SEEK_SET) == 0,
                  "Failed to seek at bd_offset={} (bytes={}). {}",
                  bd_offset, byte_offset, std::strerror(errno));
            noa::io::decode(file, encoding, output, n_threads);
        }

        template<typename T>
        void encode(
            std::FILE* file,
            const Span<const T, 4>& input,
            const Vec<i64, 2>& bd_offset,
            bool clamp,
            i32 n_threads
        ) {
            const auto encoding = Encoding{
                .dtype = m_dtype,
                .clamp = clamp,
                .endian_swap = m_is_endian_swapped
            };
            const i64 byte_offset =
                HEADER_SIZE + m_extended_bytes_nb +
                encoding.encoded_size(ni::offset_at(m_shape.strides(), bd_offset));

            check(std::fseek(file, byte_offset, SEEK_SET) == 0,
                  "Failed to seek at bd_offset={} (bytes={}). {}",
                  bd_offset, byte_offset, std::strerror(errno));
            noa::io::encode(input, file, encoding, n_threads);
        }

        static auto is_supported_extension(std::string_view extension) noexcept -> bool {
            using namespace std::string_view_literals;
            return extension == ".mrc"sv or extension == ".mrcs"sv or extension == ".st"sv;
        }

        static auto is_supported_stream(std::FILE* file) noexcept -> bool {
            auto current_pos = std::ftell(file); // get current position
            check(current_pos != -1);

            // Check file size is larger than the MRC header.
            check(std::fseek(file, 0, SEEK_END) == 0);
            auto size = std::ftell(file);
            check(size != -1);
            if (size <= 1024)
                return false;

            // Read the stamp.
            char stamp[4];
            check(std::fseek(file, 212, SEEK_SET) == 0);
            check(std::fread(stamp, 1, 4, file) == 4);

            check(std::fseek(file, current_pos, SEEK_SET) == 0); // go back to the original position

            if (not (stamp[0] == 68 and stamp[1] == 65) and
                not (stamp[0] == 68 and stamp[1] == 68) and
                not (stamp[0] == 17 and stamp[1] == 17))
                return false;
            return stamp[2] == 0 and stamp[3] == 0;
        }

        static auto required_file_size(const Shape<i64, 4>& shape, Encoding::Type dtype) noexcept -> i64 {
            // The MRC encoder doesn't resize the stream, so let the BinaryFile do the resizing when opening the stream.
            return HEADER_SIZE + Encoding::encoded_size(dtype, shape.n_elements());
        }

        static auto closest_supported_dtype(Encoding::Type dtype) noexcept -> Encoding::Type {
            switch (dtype) {
                case Encoding::I8:
                case Encoding::U8:
                case Encoding::I16:
                case Encoding::U16:
                case Encoding::F16:
                    return dtype;
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
        static constexpr i64 HEADER_SIZE = 1024;
        Shape<i64, 4> m_shape{}; // BDHW order
        Vec<f32, 3> m_spacing{}; // DHW order
        Encoding::Type m_dtype{};
        i32 m_extended_bytes_nb{};
        bool m_is_endian_swapped{false};
    };

    /// Simple tiff file encoder and decoder.
    /// \details Limitations/notes:
    ///     - Only bi-level or grayscale images are supported.
    ///     - Only strips (no tiles) and contiguous samples.
    ///     - Only 2d images are supported.
    ///     - Files are written (see the encode function) uncompressed and using a single thread.
    ///       On the other hand, reading compressed files is allowed, and TIFF directories can be
    ///       distributed amongst multiple threads.
    ///     - Slices should be written sequentially. While our API allows writing slices in any order,
    ///       libtiff doesn't support that, so we check for this and throw an error if slices are written
    ///       non-sequentially.
    /// \see https://download.osgeo.org/libtiff/doc/TIFF6.pdf
    ///      https://libtiff.gitlab.io/libtiff/index.html
    /// \usage
    ///     - In read mode, call read_header(), then decode(), then close().
    ///     See ImageFile for more details.
    struct EncoderTiff {
        static auto is_supported_extension(std::string_view extension) noexcept -> bool {
            using namespace std::string_view_literals;
            return extension == ".tif"sv or extension == ".tiff"sv;
        }

        static auto is_supported_stream(std::FILE* file) noexcept -> bool {
            u16 stamp[2];
            auto current_pos = std::ftell(file); // get current position
            check(current_pos != -1);
            check(std::fseek(file, 0, SEEK_SET) == 0); // go to the beginning
            check(std::fread(stamp, 1, 4, file) == 4); // read stamp
            check(std::fseek(file, current_pos, SEEK_SET) == 0); // go back to the original position
            return (stamp[0] == 0x4949 or stamp[0] == 0x4d4d) and
                   (stamp[1] == 0x002a or stamp[1] == 0x2a00);
        }

        static auto required_file_size(const Shape<i64, 4>&, Encoding::Type) noexcept -> i64 {
            return -1; // the TIFF encoder will handle the stream resizing during writing operations
        }

        static auto closest_supported_dtype(Encoding::Type dtype) noexcept -> Encoding::Type {
            return dtype; // TIFF encoder supports all encoding types
        }

    public:
        auto read_header(
            std::FILE* file
        ) -> Tuple<Shape<i64, 4>, Vec<f64, 3>, Encoding::Type>;

        void write_header(
            std::FILE* file,
            const Shape<i64, 4>& shape,
            const Vec<f64, 3>& spacing,
            Encoding::Type dtype
        );

        void close() const;

        template<typename T>
        void decode(
            std::FILE* file,
            const Span<T, 4>& output,
            const Vec<i64, 2>& bd_offset,
            bool clamp,
            i32 n_threads
        );

        template<typename T>
        void encode(
            std::FILE* file,
            const Span<const T, 4>& input,
            const Vec<i64, 2>& bd_offset,
            bool clamp,
            i32 n_threads
        );

    public: // move-only - not thread-safe!
        EncoderTiff() = default;
        EncoderTiff(const EncoderTiff& rhs) = delete;
        EncoderTiff& operator=(const EncoderTiff& rhs) = delete;
        EncoderTiff(EncoderTiff&& rhs) noexcept {
            m_shape = rhs.m_shape;
            m_spacing = rhs.m_spacing;
            m_dtype = rhs.m_dtype;
            m_handles = std::move(rhs.m_handles);
            current_directory = rhs.current_directory;
            m_is_write = rhs.m_is_write;
        }
        EncoderTiff& operator=(EncoderTiff&& rhs) noexcept {
            if (this != &rhs) {
                m_shape = rhs.m_shape;
                m_spacing = rhs.m_spacing;
                m_dtype = rhs.m_dtype;
                m_handles = std::move(rhs.m_handles);
                current_directory = rhs.current_directory;
                m_is_write = rhs.m_is_write;
            }
            return *this;
        }

    public:
        // For multithreading support, each thread is assigned its own TIFF handle.
        // Note: we cannot easily use a vector because since we pass a pointer to the tiff library,
        // we need to make sure these handles are not relocated.
        struct Handle {
            std::mutex* mutex{};
            std::FILE* file{};
            long offset{};
        };
        using handle_type = Pair<Handle, void*>;

    private:
        Shape<i64, 3> m_shape{}; // BHW order
        Vec<f32, 2> m_spacing{}; // HW order
        Encoding::Type m_dtype{};

        // Create multiple TIFF handles from the same file stream to enable parallel decoding.
        std::mutex m_mutex;
        std::unique_ptr<handle_type[]> m_handles{};
        i32 current_directory{};
        bool m_is_write{};
    };
}
